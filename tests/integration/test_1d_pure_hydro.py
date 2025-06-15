import copy
import logging
from typing import Iterable, Tuple

import numpy as np
import pyrtid
import pyrtid.forward as dmfwd
import pyrtid.inverse as dminv
import pytest
import scipy as sp
from pyrtid.utils import NDArrayFloat, RectilinearGrid, indices_to_node_number
from pyrtid.utils.operators import get_angle_btw_vectors_deg

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("this is a test")

MIN_VAL_K = 2e-4
MAX_VAL_K = 1e-2


def get_forward_and_obs(
    flow_regime: dmfwd.FlowRegime,
) -> Tuple[dmfwd.ForwardModel, dminv.Observables]:
    nx = 40  # number of voxels along the x axis
    ny = 1  # number of voxels along the y axis
    nz = 1
    dx = 9.3  # voxel dimension along the x axis
    dy = 8.45  # voxel dimension along the y axis
    dz = 1.0  # voxel dimension along the z axis

    # Time parameters
    duration_in_d = 7.0  # duration in days
    duration_in_s = duration_in_d * 3600 * 24
    dt_init = 3600 * 6  # timestep in seconds
    dt_max = dt_init
    dt_min = dt_init

    # Hydro parameters
    k0 = 1e-4  # general permeability
    storage_coefficient = 1e-3

    courant_factor = 20
    crank_nicolson = 0.8  # enough to ensure stability and test "semi-implcit"
    cst_head_left = 2  # m
    cst_head_right = -3  # m

    production_locations = [12, 28]
    injection_locations = [4, 20, 36]

    perm_reference = np.ones((nx, ny, nz)) * MIN_VAL_K * 1.5  # m2/s
    perm_reference[10:20, 0] = 2e-3

    # add a bit of noise
    # perm_reference *=
    # (1 + np.random.default_rng(2023).random(size=perm_reference.shape))

    # Initial estimate = an homogeneous value
    perm_estimate = np.ones((nx, ny, nz)) * MIN_VAL_K  # m2/s
    # perm_reference = np.ones((nx, ny)) * MIN_VAL_K * 4  # m2/s
    # add a bit of noise
    # perm_estimate *=
    # (1 + np.random.default_rng(2025).random(size=perm_reference.shape))

    # Simulation on 31 days. We use a 4h timestep.
    time_params = dmfwd.TimeParameters(
        duration=duration_in_s,
        dt_init=dt_init,
        dt_max=dt_max,
        dt_min=dt_min,
        courant_factor=courant_factor,
    )
    grid = RectilinearGrid(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)
    fl_params = dmfwd.FlowParameters(
        permeability=k0,
        storage_coefficient=storage_coefficient,
        regime={
            "stationary": dmfwd.FlowRegime.STATIONARY,
            "transient": dmfwd.FlowRegime.TRANSIENT,
        }[flow_regime],
        crank_nicolson=crank_nicolson,
    )

    base_model = dmfwd.ForwardModel(
        grid,
        time_params,
        fl_params,
        tr_params=dmfwd.TransportParameters(is_skip_rt=True),
    )

    # Boundary conditions
    base_model.add_boundary_conditions(
        dmfwd.ConstantHead(span=(slice(0, 1), slice(None), slice(None)))
    )
    base_model.fl_model.lhead[0][0, :, :] = cst_head_left
    base_model.add_boundary_conditions(
        dmfwd.ConstantHead(span=(slice(nx - 1, nx), slice(None), slice(None)))
    )
    base_model.fl_model.lhead[0][-1, :, :] = cst_head_right

    day = 0
    prod_flw = -8.0 / 3600  # 8 m3/h

    # two successive pumping on days 1 and 2
    for loc in production_locations:
        day += 1
        sink_term = dmfwd.SourceTerm(
            f"producer loc # {loc}",
            node_ids=np.array(indices_to_node_number(ix=loc, nx=nx)),
            # pumping on 1 day
            times=np.array([day, day + 1], dtype=np.float64) * 3600 * 24,
            flowrates=np.array([prod_flw, 0.0]),
            concentrations=np.array([[0.0, 0.0], [0.0, 0.0]]),
        )
        base_model.add_src_term(sink_term)

    # three successive injections on days 3, 4 and 5
    for i, loc in enumerate(injection_locations):
        day += 1
        source_term = dmfwd.SourceTerm(
            f"injector loc # {loc}",
            node_ids=np.array(indices_to_node_number(ix=loc, nx=nx)),
            times=np.array([day, day + 1], dtype=np.float64) * 3600 * 24,
            flowrates=np.array([-prod_flw, 0.0]),
            concentrations=np.array([[0.0, 0.0], [0.0, 0.0]]),
        )
        base_model.add_src_term(source_term)

    # - Create two models with different permeabilities

    # copy the base model
    model_reference = copy.deepcopy(base_model)
    # update permeability
    model_reference.fl_model.permeability = perm_reference
    # solve the flow problem
    dmfwd.ForwardSolver(model_reference).solve()

    # copy the base model
    model_initial_guess = copy.deepcopy(base_model)
    # update permeability
    model_initial_guess.fl_model.permeability = perm_estimate
    # solve the flow problem
    dmfwd.ForwardSolver(model_initial_guess).solve()

    freq_measure_in_d = 0.5  # one measure every two days
    n_sampled_days = duration_in_d / freq_measure_in_d

    obs_times_in_s = np.arange(n_sampled_days) * 3600 * 24 * freq_measure_in_d

    obs_times_in_s_head = obs_times_in_s

    logging.info(f"obs_times_in_s_head = {obs_times_in_s_head}")

    noise_std_head: float = 3.0  # This is an absolute value in m

    def get_white_noise(shape: Iterable[int], noise_std: float) -> NDArrayFloat:
        """
        Return the input with some added white noise.

        Note
        ----
        The parameters are hardcoded to be consistent in the notebook.
        Change the function directly.
        """
        mean_noise = 0.0  # mean
        return np.random.default_rng(2021).normal(
            mean_noise, noise_std, size=np.array(shape)
        )

    # Generate a white noise
    obs_noise_head: NDArrayFloat = get_white_noise(
        (len(production_locations), obs_times_in_s_head.size), noise_std_head
    )

    # - Check the intensity of the noise for the head

    all_times = model_initial_guess.time_params.times

    obs_values_head = np.zeros((len(production_locations), obs_times_in_s_head.size))

    for count, ix in enumerate(production_locations):
        # interpolate the values
        obs_values_head[count] = (
            sp.interpolate.interp1d(
                all_times, model_reference.fl_model.head[ix, 0, 0, :], kind="cubic"
            )(obs_times_in_s_head)
            + obs_noise_head[count, :]
        )

    count = 0

    ix = production_locations[count]

    # ## Gradient with respect to the permeability

    observables = []
    for i, node_id in enumerate(production_locations):
        observables.append(
            pyrtid.inverse.Observable(
                state_variable=pyrtid.inverse.StateVariable.HEAD,
                node_indices=node_id,
                times=obs_times_in_s_head,
                values=obs_values_head[i],
                uncertainties=noise_std_head,
            )
        )

    return model_initial_guess, observables


def get_executor_k(
    flow_regime: dmfwd.FlowRegime,
) -> Tuple[dminv.AdjustableParameter, dminv.LBFGSBInversionExecutor]:
    model_initial_guess, observables = get_forward_and_obs(flow_regime)

    gsc = dminv.GradientScalerConfig(
        max_change_target=np.log(5),  # max an order of magnitude for the first update
        pcd_change_eval=dminv.LogTransform(),
        lb=1e-10,
        ub=1e10,
    )

    param_k = dminv.AdjustableParameter(
        name=dminv.ParameterName.PERMEABILITY,
        # /10 to avoid finite differences fail because of bounds clipping
        lbounds=MIN_VAL_K / 10,
        # * 10 to avoid finite differences fail because of bounds clipping
        ubounds=MAX_VAL_K * 10,
        preconditioner=dminv.LogTransform(),
        regularizators=dminv.regularization.TikhonovRegularizator(
            grid=model_initial_guess.grid, preconditioner=dminv.LogTransform()
        ),
        gradient_scaler_config=gsc,
        reg_weight_update_strategy=dminv.regularization.ConstantRegWeight(200.0),
    )

    # Create an executor to keep track of the adjoint model
    solver_config = dminv.LBFGSBSolverConfig(
        maxfun=20,
        maxiter=20,
        ftol=1e-3,
        gtol=1e-3,
        is_check_gradient=False,
        is_adj_numerical_acceleration=True,
        afpi_eps=1e-15,
        is_use_continuous_adj=False,
    )
    model_adjusted_k = copy.deepcopy(model_initial_guess)
    inverse_model_k = dminv.InverseModel(param_k, observables)
    executor_k = dminv.LBFGSBInversionExecutor(
        model_adjusted_k, inverse_model_k, solver_config
    )

    return param_k, executor_k


@pytest.mark.parametrize("flow_regime", ("transient", "stationary"))
def test_grad_k(flow_regime: dmfwd.FlowRegime) -> None:
    param_k, executor_k = get_executor_k(flow_regime)

    # executor_k.eval_loss(
    #     param_k.preconditioner(param_k.values.ravel("F")), is_save_state=False
    # )

    is_grad_ok = executor_k.is_adjoint_gradient_correct(
        max_workers=4, is_verbose=False, eps=1e-6
    )

    assert is_grad_ok
    assert (
        get_angle_btw_vectors_deg(
            param_k.grad_adj_history[0], param_k.grad_fd_history[0]
        )
        < 1e-5
    )


@pytest.mark.parametrize("flow_regime", ("transient", "stationary"))
def test_optim_k(flow_regime: dmfwd.FlowRegime) -> None:
    _, executor_k = get_executor_k(flow_regime)
    executor_k.run()


def get_executor_h0(
    flow_regime: dmfwd.FlowRegime,
) -> Tuple[dminv.AdjustableParameter, dminv.LBFGSBInversionExecutor]:
    model_initial_guess, observables = get_forward_and_obs(flow_regime)

    param_h0 = dminv.AdjustableParameter(
        name=dminv.ParameterName.INITIAL_HEAD,
        # /10 to avoid finite differences fail because of bounds clipping
        lbounds=-100,
        # * 10 to avoid finite differences fail because of bounds clipping
        ubounds=100,
    )

    # Create an executor to keep track of the adjoint model
    solver_config = dminv.LBFGSBSolverConfig(
        maxfun=20,
        maxiter=20,
        ftol=1e-4,
        gtol=1e-4,
        is_check_gradient=False,
        is_adj_numerical_acceleration=True,
        afpi_eps=1e-15,
    )
    model_adjusted_h0 = copy.deepcopy(model_initial_guess)
    inverse_model_h0 = dminv.InverseModel(param_h0, observables)
    executor_h0 = dminv.LBFGSBInversionExecutor(
        model_adjusted_h0, inverse_model_h0, solver_config
    )

    return param_h0, executor_h0


@pytest.mark.parametrize("flow_regime", ("transient", "stationary"))
def test_grad_h0(flow_regime: dmfwd.FlowRegime) -> None:
    param_h0, executor_h0 = get_executor_k(flow_regime)
    # executor_h0.eval_loss(param_h0.preconditioner(param_h0.values.ravel("F")))

    is_grad_h0_ok = executor_h0.is_adjoint_gradient_correct(
        max_workers=4, is_verbose=False, eps=1e-4
    )
    assert is_grad_h0_ok
    assert (
        get_angle_btw_vectors_deg(
            param_h0.grad_adj_history[0], param_h0.grad_fd_history[0]
        )
        < 1e-5
    )


def get_executor_sc(
    flow_regime: dmfwd.FlowRegime,
) -> Tuple[dminv.AdjustableParameter, dminv.LBFGSBInversionExecutor]:
    model_initial_guess, observables = get_forward_and_obs(flow_regime)

    param_sc = dminv.AdjustableParameter(
        name=dminv.ParameterName.STORAGE_COEFFICIENT,
        # /10 to avoid finite differences fail because of bounds clipping
        lbounds=1e-6,
        # * 10 to avoid finite differences fail because of bounds clipping
        ubounds=1e-1,
        preconditioner=dminv.LogTransform(),
    )

    # Create an executor to keep track of the adjoint model
    solver_config = dminv.LBFGSBSolverConfig(
        maxfun=20,
        maxiter=20,
        ftol=1e-4,
        gtol=1e-4,
        is_check_gradient=False,
        is_adj_numerical_acceleration=True,
        afpi_eps=1e-15,
    )
    model_adjusted_sc = copy.deepcopy(model_initial_guess)
    inverse_model_sc = dminv.InverseModel(param_sc, observables)
    executor_sc = dminv.LBFGSBInversionExecutor(
        model_adjusted_sc, inverse_model_sc, solver_config
    )

    return param_sc, executor_sc


@pytest.mark.parametrize("flow_regime", ("transient", "stationary"))
def test_grad_sc(flow_regime: dmfwd.FlowRegime) -> None:
    param_sc, executor_sc = get_executor_sc(flow_regime)

    # executor_sc.eval_loss(param_sc.preconditioner(param_sc.values.ravel("F")))

    is_grad_sc_ok = executor_sc.is_adjoint_gradient_correct(
        max_workers=4, is_verbose=False, eps=1e-6
    )

    assert is_grad_sc_ok
    assert (
        get_angle_btw_vectors_deg(
            param_sc.grad_adj_history[0], param_sc.grad_fd_history[0]
        )
        < 1e-5
    )
