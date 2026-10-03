"""Helpers to compare the adjoint gradient with finite differences."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
import pyrtid.forward as dmfwd
import pyrtid.inverse as dminv
from pyrtid.inverse.asm.gradients import (
    compute_adjoint_gradient,
    compute_fd_gradient,
)
from pyrtid.inverse.loss_function import get_predictions_matching_observations
from pyrtid.inverse.params import update_parameters_from_model
from quickpaver import RectilinearGrid

DEFAULT_NODES = (1, 2, 5, 6)


def build_gravity_model(
    nx: int = 4,
    ny: int = 1,
    nz: int = 3,
    vertical_axis: dmfwd.VerticalAxis = dmfwd.VerticalAxis.Z,
    regime: str = "transient",
    cst_conc: bool = False,
    seed: int = 0,
    explicit: bool = True,
) -> dmfwd.ForwardModel:
    """Build a small density driven (gravity) reactive transport model."""
    rng = np.random.default_rng(seed)
    grid = RectilinearGrid(nx=nx, ny=ny, nz=nz, dx=5.0, dy=4.0, dz=1.0)
    time_params = dmfwd.TimeParameters(
        duration=3600 * 24 * 1.0, dt_init=3600 * 6, dt_max=3600 * 6, dt_min=3600 * 6
    )
    flow_params = dmfwd.FlowParameters(
        permeability=1e-4,
        storage_coefficient=1e-3,
        regime=(
            dmfwd.FlowRegime.TRANSIENT
            if regime == "transient"
            else dmfwd.FlowRegime.STATIONARY
        ),
        crank_nicolson=0.8,
        is_gravity=True,
        vertical_axis=vertical_axis,
    )
    tr_params = dmfwd.TransportParameters(
        porosity=0.25, diffusion=1e-5, dispersivity=0.5
    )
    gch_params = dmfwd.GeochemicalParameters(
        conc=1e-8,
        conc2=1e-3,
        grade=1e-1,
        grade2=1e-10,
        kv=-6.9e-6,
        As=13.5,
        Ks=6.3e-4,
        stocoef=1.0,
        use_explicit_formulation=explicit,
    )
    model = dmfwd.ForwardModel(
        grid, time_params, flow_params, tr_params=tr_params, gch_params=gch_params
    )
    nn = grid.n_grid_cells
    first = (slice(0, 1), slice(None), slice(None))
    last = (slice(nx - 1, nx), slice(None), slice(None))
    model.add_boundary_conditions(dmfwd.ConstantHead(span=first, values=2.0))
    model.add_boundary_conditions(dmfwd.ConstantHead(span=last, values=-1.0))
    if cst_conc:
        model.add_boundary_conditions(
            dmfwd.ConstantConcentration(span=first, values=1e-3)
        )
    model.add_src_term(
        dmfwd.SourceTerm(
            "inj",
            node_ids=np.array([nn // 2]),
            times=np.array([0.0, 3600 * 12.0]),
            flowrates=np.array([2e-3, 0.0]),
            concentrations=np.array([[1e-3, 5e-4], [0.0, 0.0]]),
        )
    )
    model.fl_model.permeability = 1e-4 * (1 + 0.5 * rng.random(grid.shape))
    model.tr_model.porosity = 0.25 * (1 + 0.2 * rng.random(grid.shape))
    model.tr_model.set_initial_conc(1e-4 * (1 + rng.random(grid.shape)), sp=1)
    model.tr_model.set_initial_conc(1e-5 * (1 + rng.random(grid.shape)), sp=0)
    return model


def make_observables(
    model: dmfwd.ForwardModel,
    variables: Sequence[tuple[dminv.StateVariable, int | None]],
    nodes: tuple[int, ...] = DEFAULT_NODES,
    seed: int = 1,
    noise: float = 0.1,
    relative_uncertainty: bool = False,
) -> list[dminv.Observable]:
    """Build observables with values equal to a perturbed forward result."""
    rng = np.random.default_rng(seed)
    n = model.grid.n_grid_cells
    nodes = tuple(i for i in nodes if i < n)
    duration = model.time_params.duration
    times = np.array([0.0, duration * 0.23, duration * 0.55, duration])
    work = copy.deepcopy(model)
    work.reinit()
    dmfwd.ForwardSolver(work).solve()
    obs = []
    for state_variable, sp in variables:
        for node in nodes:
            o = dminv.Observable(
                state_variable=state_variable,
                node_indices=node,
                times=times,
                values=np.zeros(times.size),
                uncertainties=1.0,
                sp=sp,
            )
            pred = get_predictions_matching_observations(work, [o], duration)
            scale = max(float(np.max(np.abs(pred))), 1e-12)
            values = pred + noise * scale * (rng.random(pred.shape) + 0.5)
            obs.append(
                dminv.Observable(
                    state_variable=state_variable,
                    node_indices=node,
                    times=times,
                    values=values,
                    uncertainties=0.1 * scale if relative_uncertainty else 1.0,
                    sp=sp,
                )
            )
    return obs


def solve_adjoint(
    model: dmfwd.ForwardModel,
    parameters: list[dminv.AdjustableParameter],
    observables: list[dminv.Observable],
    hm_end_time: float | None = None,
    crank_flow: float | None = None,
    is_verbose: bool = False,
    max_nafpi: int = 30,
    **adj_kwargs,
) -> tuple[dmfwd.ForwardModel, dminv.AdjointModel]:
    """Solve the forward and the adjoint problems (the input model is copied)."""
    model = copy.deepcopy(model)
    update_parameters_from_model(model, parameters)
    dmfwd.ForwardSolver(model).solve()
    adj_model = dminv.AdjointModel(
        model.grid,
        model.time_params,
        model.fl_model.is_gravity,
        model.tr_model.n_sp,
        **adj_kwargs,
    )
    if crank_flow is not None:
        adj_model.a_fl_model.set_crank_nicolson(crank_flow)
    asolver = dminv.AdjointSolver(model, adj_model)
    asolver.solve(
        observables,
        hm_end_time=hm_end_time,
        is_verbose=is_verbose,
        max_nafpi=max_nafpi,
    )
    return model, asolver.adj_model


def check_gradient(
    model: dmfwd.ForwardModel,
    parameters: list[dminv.AdjustableParameter],
    observables: list[dminv.Observable],
    eps: float | None = None,
    hm_end_time: float | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the adjoint and finite differences gradients."""
    fwd, adj_model = solve_adjoint(
        model, parameters, observables, hm_end_time=hm_end_time, **kwargs
    )
    adj = compute_adjoint_gradient(fwd, adj_model, parameters)
    fd = compute_fd_gradient(
        fwd, observables, parameters, eps=eps, max_obs_time=hm_end_time
    )
    return adj, fd
