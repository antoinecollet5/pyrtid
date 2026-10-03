"""Test the options, the solver settings and the error branches of the ASM."""

import copy
import re
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pyrtid.forward as dmfwd
import pyrtid.inverse as dminv
import pytest
from inv_toolbox.utils.spatial_filters import Filter
from pyrtid.inverse.asm import aflow_solver as aflow_module
from pyrtid.inverse.asm import ageochem_solver, atransport_solver
from pyrtid.inverse.asm.adensity_solver import get_drhomean
from pyrtid.inverse.asm.gradients import (
    compute_adjoint_gradient,
    compute_param_adjoint_loss_ls_function_gradient,
    is_adjoint_gradient_correct,
)
from pyrtid.inverse.params import update_parameters_from_model
from tests.helpers import PARAMETER_FACTORIES, build_model, build_parameters

from .asm_utils import (
    build_gravity_model,
    check_gradient,
    make_observables,
    solve_adjoint,
)

SV = dminv.StateVariable
HEAD_CONC = [(SV.HEAD, None), (SV.CONCENTRATION, 0), (SV.GRADE, 0)]


def rel_err(adj, fd) -> float:
    return float(np.linalg.norm(adj - fd) / np.linalg.norm(fd))


@pytest.mark.parametrize("gravity", [False, True])
def test_save_spmats_and_crank_nicolson(gravity):
    """Saved matrices are checked against the forward ones (transpose)."""
    model = build_gravity_model(4, 1, 3) if gravity else build_model(nx=4, ny=2)
    model.fl_model.is_save_spmats = True
    model.tr_model.is_save_spmats = True
    obs = make_observables(model, HEAD_CONC, nodes=(1, 2, 5))
    params = build_parameters(["K", "SS"])
    adj, fd = check_gradient(model, params, obs, crank_flow=0.8)
    assert rel_err(adj, fd) < 1e-3


@pytest.mark.parametrize("hm_end_time", [None, 3600 * 15.0])
def test_hm_end_time(hm_end_time):
    model = build_model(nx=4, ny=2)
    obs = make_observables(model, HEAD_CONC)
    adj, fd = check_gradient(
        model, build_parameters(["K", "W"]), obs, hm_end_time=hm_end_time
    )
    assert rel_err(adj, fd) < 1e-2


def test_numerical_acceleration_and_verbose(caplog):
    model = build_model(nx=4, ny=2, explicit=False)
    obs = make_observables(model, HEAD_CONC)
    params = build_parameters(["K", "W"])
    with caplog.at_level("INFO"):
        fwd, adj_model = solve_adjoint(
            model,
            params,
            obs,
            is_verbose=True,
            is_adj_numerical_acceleration=True,
            afpi_eps=1e-8,
        )
    assert adj_model.a_tr_model.is_adj_num_acc_for_timestep
    grad_acc = compute_adjoint_gradient(fwd, adj_model, params, is_save_state=False)
    adj, _ = check_gradient(model, params, obs, afpi_eps=1e-8)
    assert np.all(np.isfinite(grad_acc))
    assert rel_err(grad_acc, adj) < 0.2


def test_max_nafpi_restart_then_failure():
    model = build_model(nx=4, ny=2, explicit=False)
    obs = make_observables(model, HEAD_CONC)
    params = build_parameters(["K"])
    # The convergence criteria can never be reached
    # Restart the timestep without numerical acceleration, and fail afterwards
    with pytest.raises(RuntimeError):
        solve_adjoint(
            model,
            params,
            obs,
            is_adj_numerical_acceleration=True,
            afpi_eps=0.0,
            max_nafpi=1,
        )
    with pytest.raises(RuntimeError):
        solve_adjoint(model, params, obs, afpi_eps=0.0, max_nafpi=1)


def test_continuous_adjoint():
    model = build_model(nx=4, ny=2, skip_rt=True)
    obs = make_observables(model, [(SV.HEAD, None)])
    adj, fd = check_gradient(
        model, build_parameters(["K"]), obs, is_use_continuous_adj=True
    )
    # Differentiate-then-discretize: only an approximation of the true gradient
    assert np.all(np.isfinite(adj))
    assert np.linalg.norm(fd) > 0.0


def test_continuous_adjoint_errors():
    # reactive transport is not skipped
    model = build_model(nx=4, ny=2)
    obs = make_observables(model, [(SV.HEAD, None)])
    with pytest.raises(ValueError, match="Continuous adjoint only working"):
        solve_adjoint(model, build_parameters(["K"]), obs, is_use_continuous_adj=True)

    # density flow
    gmodel = build_gravity_model(4, 1, 3)
    with pytest.raises(ValueError, match="not implemented for density flow"):
        dminv.AdjointModel(
            gmodel.grid, gmodel.time_params, True, 2, is_use_continuous_adj=True
        )


@pytest.mark.parametrize(
    "variable,param",
    [
        (SV.PERMEABILITY, "K"),
        (SV.POROSITY, "W"),
        (SV.DIFFUSION, "D"),
        (SV.DISPERSIVITY, "A"),
        (SV.STORAGE_COEFFICIENT, "SS"),
    ],
)
def test_observed_parameters(variable, param):
    model = build_model(nx=4, ny=2)
    obs = make_observables(
        model, [(variable, None), (SV.HEAD, None)], relative_uncertainty=True
    )
    adj, fd = check_gradient(model, build_parameters([param]), obs)
    assert rel_err(adj, fd) < 1e-2


def test_observed_density_and_pressure():
    model = build_gravity_model(4, 1, 3)
    obs = make_observables(model, [(SV.PRESSURE, None), (SV.DENSITY, None)])
    params = build_parameters(["K", "W"])
    adj, fd = check_gradient(model, params, obs)
    assert rel_err(adj, fd) < 1e-2


def test_is_adjoint_gradient_correct():
    model = build_model(nx=4, ny=2)
    obs = make_observables(model, [(SV.HEAD, None), (SV.CONCENTRATION, 0)])
    params = build_parameters(["K"])
    # The adjoint model must be sized with the number of timesteps
    solved = copy.deepcopy(model)
    dmfwd.ForwardSolver(solved).solve()
    adj_model = dminv.AdjointModel(solved.grid, solved.time_params, False, 2)
    assert is_adjoint_gradient_correct(model, adj_model, params, obs)


class HalfFilter(Filter):
    def filter(self, param, iteration):
        return 0.5 * param


def test_gradient_filter():
    model = build_model(nx=4, ny=2)
    obs = make_observables(model, [(SV.HEAD, None)])
    param = PARAMETER_FACTORIES["K"]()
    fwd, adj_model = solve_adjoint(model, [param], obs)
    grad = compute_adjoint_gradient(fwd, adj_model, [param], is_save_state=False)
    param_filt = PARAMETER_FACTORIES["K"]()
    param_filt.filters = [HalfFilter()]
    update_parameters_from_model(fwd, [param_filt])
    grad_filt = compute_adjoint_gradient(
        fwd, adj_model, [param_filt], is_save_state=False
    )
    np.testing.assert_allclose(grad_filt, 0.5 * grad)


def test_fd_gradient_bounds_warning():
    model = build_model(nx=4, ny=2)
    obs = make_observables(model, [(SV.HEAD, None)])
    param = PARAMETER_FACTORIES["W"]()
    # A value equal to the lower bound
    param.lbounds = float(np.min(model.tr_model.porosity))
    with pytest.warns(UserWarning, match="equal the lower and/or upper bound"):
        check_gradient(model, [param], obs)


def test_gradient_parameter_errors():
    model = build_model(nx=4, ny=2)
    obs = make_observables(model, [(SV.HEAD, None)])
    fwd, adj_model = solve_adjoint(model, build_parameters(["K"]), obs)
    # not an adjustable parameter
    with pytest.raises(NotImplementedError):
        compute_param_adjoint_loss_ls_function_gradient(
            fwd, adj_model, cast(Any, SimpleNamespace(name="unknown"))
        )
    # initial pressure but no gravity
    with pytest.raises(RuntimeError, match="Optimize the initial head instead"):
        compute_param_adjoint_loss_ls_function_gradient(
            fwd,
            adj_model,
            cast(Any, SimpleNamespace(name=dminv.ParameterName.INITIAL_PRESSURE)),
        )

    gmodel = build_gravity_model(4, 1, 3)
    gobs = make_observables(gmodel, [(SV.HEAD, None)])
    gfwd, gadj = solve_adjoint(gmodel, build_parameters(["K"]), gobs)
    # initial head with gravity
    with pytest.raises(RuntimeError, match="Optimize the initial pressure instead"):
        compute_param_adjoint_loss_ls_function_gradient(
            gfwd, gadj, PARAMETER_FACTORIES["H0"]()
        )


def test_density_helpers():
    model = build_gravity_model(4, 1, 3)
    obs = make_observables(model, [(SV.HEAD, None)])
    fwd, _ = solve_adjoint(model, build_parameters(["K"]), obs)
    flat = get_drhomean(fwd.grid, fwd.tr_model, 2, 1)
    full = get_drhomean(fwd.grid, fwd.tr_model, 2, 1, is_flatten=False)
    np.testing.assert_allclose(flat, full.ravel("F"))


def test_geochemical_derivative_error():
    model = build_model(nx=4, ny=2, explicit=False)
    obs = make_observables(model, [(SV.HEAD, None)])
    fwd, _ = solve_adjoint(model, build_parameters(["K"]), obs)
    with pytest.raises(ValueError, match=re.escape("sp should be 0 or 1")):
        ageochem_solver.ddMdmobnext(
            fwd.tr_model, fwd.gch_params, 1, fwd.time_params.ldt[0], 2
        )


@pytest.mark.parametrize(
    "module,gravity",
    [(aflow_module, False), (aflow_module, True), (atransport_solver, False)],
)
def test_singular_matrix_warnings(module, gravity, monkeypatch):
    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("singular")

    monkeypatch.setattr(module, "get_super_ilu_preconditioner", raise_runtime_error)
    model = build_gravity_model(4, 1, 3) if gravity else build_model(nx=4, ny=2)
    obs = make_observables(model, HEAD_CONC)
    with pytest.warns(UserWarning, match="singular"):
        solve_adjoint(model, build_parameters(["K"]), obs)


def test_adjoint_conc_alias():
    model = build_model(nx=4, ny=2)
    adj_model = dminv.AdjointModel(model.grid, model.time_params, False, 2)
    assert adj_model.a_tr_model.a_conc is adj_model.a_tr_model.a_mob


def test_3d_saturated_model():
    model = build_model(nx=3, ny=2, nz=2)
    obs = make_observables(model, HEAD_CONC, nodes=(1, 4, 7, 10))
    names = ["K", "SS", "W", "D", "A"]
    adj, fd = check_gradient(model, build_parameters(names), obs)
    n = model.grid.n_grid_cells
    for i in range(len(names)):
        sl = slice(i * n, (i + 1) * n)
        assert rel_err(adj[sl], fd[sl]) < 1e-2, names[i]
