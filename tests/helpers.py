"""Shared builders for the tests: a small 2D reactive transport model."""

from __future__ import annotations

import inv_toolbox
import numpy as np
import pyrtid.forward as dmfwd
import pyrtid.inverse as dminv
from quickpaver import RectilinearGrid

PARAMETER_FACTORIES = {
    "K": lambda: dminv.AdjustableParameter(
        dminv.ParameterName.PERMEABILITY,
        lbounds=1e-9,
        ubounds=1.0,
        preconditioner=inv_toolbox.utils.LogTransform(),
    ),
    "SS": lambda: dminv.AdjustableParameter(
        dminv.ParameterName.STORAGE_COEFFICIENT,
        lbounds=1e-9,
        ubounds=1.0,
        preconditioner=inv_toolbox.utils.LogTransform(),
    ),
    "H0": lambda: dminv.AdjustableParameter(
        dminv.ParameterName.INITIAL_HEAD, lbounds=-100, ubounds=100
    ),
    "W": lambda: dminv.AdjustableParameter(
        dminv.ParameterName.POROSITY, lbounds=1e-3, ubounds=0.9
    ),
    "D": lambda: dminv.AdjustableParameter(
        dminv.ParameterName.DIFFUSION,
        lbounds=1e-12,
        ubounds=1.0,
        preconditioner=inv_toolbox.utils.LogTransform(),
    ),
    "A": lambda: dminv.AdjustableParameter(
        dminv.ParameterName.DISPERSIVITY, lbounds=1e-9, ubounds=100.0
    ),
    "C0": lambda: dminv.AdjustableParameter(
        dminv.ParameterName.INITIAL_CONCENTRATION, sp=0, lbounds=0.0, ubounds=1.0
    ),
    "C1": lambda: dminv.AdjustableParameter(
        dminv.ParameterName.INITIAL_CONCENTRATION, sp=1, lbounds=0.0, ubounds=1.0
    ),
    "G0": lambda: dminv.AdjustableParameter(
        dminv.ParameterName.INITIAL_GRADE, sp=0, lbounds=0.0, ubounds=10.0
    ),
    "G1": lambda: dminv.AdjustableParameter(
        dminv.ParameterName.INITIAL_GRADE, sp=1, lbounds=0.0, ubounds=10.0
    ),
}


def build_model(
    regime: str = "transient",
    nx: int = 8,
    ny: int = 3,
    nz: int = 1,
    skip_rt: bool = False,
    explicit: bool = True,
    cst_conc: bool = False,
    seed: int = 0,
    kv: float = -6.9e-6,
    grade2: float = 1e-10,
) -> dmfwd.ForwardModel:
    """Build a small reactive transport model (injection and pumping wells)."""
    rng = np.random.default_rng(seed)
    grid = RectilinearGrid(nx=nx, ny=ny, nz=nz, dx=5.0, dy=4.0, dz=1.0)
    time_params = dmfwd.TimeParameters(
        duration=3600 * 24 * 1.0, dt_init=3600 * 4, dt_max=3600 * 4, dt_min=3600 * 4
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
    )
    tr_params = dmfwd.TransportParameters(
        porosity=0.25, diffusion=1e-5, dispersivity=0.5, is_skip_rt=skip_rt
    )
    gch_params = dmfwd.GeochemicalParameters(
        conc=1e-8,
        conc2=1e-3,
        grade=1e-1,
        grade2=grade2,
        kv=kv,
        As=13.5,
        Ks=6.3e-4,
        stocoef=1.0,
        use_explicit_formulation=explicit,
    )
    model = dmfwd.ForwardModel(
        grid, time_params, flow_params, tr_params=tr_params, gch_params=gch_params
    )
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
            node_ids=np.array([nx // 2 + nx * (ny // 2)]),
            times=np.array([0.0, 3600 * 12.0]),
            flowrates=np.array([2e-3, 0.0]),
            concentrations=np.array([[1e-3, 5e-4], [0.0, 0.0]]),
        )
    )
    model.add_src_term(
        dmfwd.SourceTerm(
            "pump",
            node_ids=np.array([2]),
            times=np.array([0.0, 3600 * 20.0]),
            flowrates=np.array([-1e-3, 0.0]),
            concentrations=np.array([[0.0, 0.0], [0.0, 0.0]]),
        )
    )
    model.fl_model.permeability = 1e-4 * (1 + 0.5 * rng.random(grid.shape))
    model.tr_model.porosity = 0.25 * (1 + 0.2 * rng.random(grid.shape))
    model.tr_model.set_initial_conc(1e-4 * (1 + rng.random(grid.shape)), sp=1)
    model.tr_model.set_initial_conc(1e-5 * (1 + rng.random(grid.shape)), sp=0)
    return model


def build_observables(
    model: dmfwd.ForwardModel, nodes: tuple[int, ...] = (3, 9, 11)
) -> list[dminv.Observable]:
    """Observe the head, the concentrations and the grades at a few nodes."""
    times = np.array([0.0, 3600 * 5.5, 3600 * 13.0, model.time_params.duration])
    obs = []
    for state_variable, sp in (
        (dminv.StateVariable.HEAD, None),
        (dminv.StateVariable.CONCENTRATION, 0),
        (dminv.StateVariable.CONCENTRATION, 1),
        (dminv.StateVariable.GRADE, 0),
        (dminv.StateVariable.GRADE, 1),
    ):
        for node in nodes:
            obs.append(
                dminv.Observable(
                    state_variable=state_variable,
                    node_indices=node,
                    times=times,
                    values=np.zeros(times.size),
                    uncertainties=1.0,
                    sp=sp,
                )
            )
    return obs


def build_parameters(names: list[str]) -> list[dminv.AdjustableParameter]:
    """Build adjustable parameters from keys of ``PARAMETER_FACTORIES``."""
    return [PARAMETER_FACTORIES[name]() for name in names]
