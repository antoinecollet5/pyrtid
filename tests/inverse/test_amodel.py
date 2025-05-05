import copy

import numpy as np
import pyrtid.forward as dmfwd
import pyrtid.inverse as dminv
import pytest
from pyrtid.utils import MeanType, NDArrayFloat, finite_gradient


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((), {}),
        ((1e-10,), {}),
        ((1e-10, True), {}),
        ((1e-10,), {"is_adj_numerical_acceleration": True}),
        ((), {"afpi_eps": 1e-10, "is_adj_numerical_acceleration": True}),
        ((), {"afpi_eps": 1e-10}),
    ],
)
def test_adjoint_model_init(args, kwargs) -> None:
    time_params = dmfwd.TimeParameters(duration=1.0, dt_init=1.0)
    grid = dmfwd.Geometry(nx=5, ny=5, dx=4.5, dy=7.5)
    dminv.AdjointModel(grid, time_params, False, 2, *args, **kwargs)


@pytest.mark.parametrize(
    "max_obs_time, mean_type",
    [
        (None, MeanType.ARITHMETIC),
        (None, MeanType.HARMONIC),
        (0.5, MeanType.ARITHMETIC),
        (1.0, MeanType.GEOMETRIC),
        (10.0, MeanType.ARITHMETIC),
    ],
)
def test_init_adjoint_sources(max_obs_time, mean_type) -> None:
    time_params = dmfwd.TimeParameters(duration=1.0, dt_init=1.0)
    grid = dmfwd.Geometry(nx=5, ny=5, dx=4.5, dy=7.5)
    fl_params = dmfwd.FlowParameters(1e-5)
    tr_params = dmfwd.TransportParameters(diffusion=1.0, porosity=0.23)
    gch_params = dmfwd.GeochemicalParameters(1.0, 0.0)

    model = dmfwd.ForwardModel(
        grid,
        time_params,
        fl_params,
        tr_params,
        gch_params,
    )

    # generate synthetic data
    # I don't understand why is does not work if we leave it to 997... ???
    model.tr_model.ldensity.append(
        np.abs(np.random.default_rng(2023).random((grid.nx, grid.ny)) + 2.0)
    )
    model.tr_model.ldensity.append(
        np.abs(np.random.default_rng(2023).random((grid.nx, grid.ny)) + 2.0)
    )
    model.tr_model.ldensity.append(
        np.abs(np.random.default_rng(2023).random((grid.nx, grid.ny)) + 3.0)
    )
    model.tr_model.lmob.append(
        np.abs(
            np.random.default_rng(2023).random((model.tr_model.n_sp, grid.nx, grid.ny))
            + 2.0
        )
    )
    model.tr_model.lmob.append(
        np.abs(
            np.random.default_rng(2023).random((model.tr_model.n_sp, grid.nx, grid.ny))
            + 3.0
        )
    )
    model.tr_model.limmob.append(
        np.abs(
            np.random.default_rng(2023).random((model.tr_model.n_sp, grid.nx, grid.ny))
            + 2.0
        )
    )
    model.tr_model.limmob.append(
        np.abs(
            np.random.default_rng(2023).random((model.tr_model.n_sp, grid.nx, grid.ny))
            + 3.0
        )
    )
    model.fl_model.lhead.append(
        np.abs(np.random.default_rng(2023).random((grid.nx, grid.ny)) + 2.0)
    )
    model.fl_model.lhead.append(
        np.abs(np.random.default_rng(2023).random((grid.nx, grid.ny)) + 3.0)
    )

    for head in model.fl_model.lhead:
        model.fl_model.lpressure.append(model.fl_model.head_to_pressure(head))

    model.time_params.save_dt()
    model.time_params.save_dt()

    model.fl_model.permeability = np.abs(
        np.random.default_rng(2023).random((grid.nx, grid.ny))
    )

    observables = [
        dminv.Observable(
            dminv.StateVariable.CONCENTRATION,
            node_indices=[2, 4, 6],
            times=np.array([0.0, 0.5, 1.0, 1.56, 2.6]),
            values=np.array([1, 1, 1, 1, 1]),
            uncertainties=np.array([0.289, 0.25, 0.27, 0.256, 0.25]),
            mean_type=mean_type,
            sp=0,
        ),
        dminv.Observable(
            dminv.StateVariable.CONCENTRATION,
            node_indices=[5, 10, 11],
            times=np.array([0.0, 0.2, 0.3, 1.3, 5.6]),
            values=np.array([0.289, 0.25, 0.27, 0.256, 0.25]),
            uncertainties=np.array([0.289, 0.25, 0.27, 0.256, 0.25]),
            mean_type=mean_type,
            sp=0,
        ),
        dminv.Observable(
            dminv.StateVariable.PERMEABILITY,
            node_indices=[5, 10, 11],
            times=np.array([0.0, 0.2, 0.3, 1.3, 5.6]),
            values=np.array([0.289, 0.25, 0.27, 0.256, 0.25]),
            uncertainties=np.array([0.289, 0.25, 0.27, 0.256, 0.25]),
            mean_type=mean_type,
        ),
    ]

    # Add all possible obervable instance
    for state_var in dminv.StateVariable:
        if mean_type != MeanType.ARITHMETIC:
            # because pressure and head can be negative -> not compatible
            # with geometric and hamonic means
            if state_var in [dminv.StateVariable.HEAD, dminv.StateVariable.PRESSURE]:
                continue
        if state_var in [dminv.StateVariable.PRESSURE]:
            continue
        observables.append(
            dminv.Observable(
                state_var,
                node_indices=[5, 10, 11],
                times=np.array([0.0, 0.2, 0.3, 1.3, 5.6]),
                values=np.array([0.289, 0.25, 0.27, 0.256, 0.25]),
                uncertainties=np.array([0.289, 0.25, 0.27, 0.256, 0.25]),
                mean_type=mean_type,
                sp=0,
            )
        )

    adj_model = dminv.AdjointModel(grid, time_params, False, model.tr_model.n_sp)
    adj_model.init_adjoint_sources(
        copy.copy(model), observables, hm_end_time=max_obs_time
    )

    def wrapper_conc(arr: NDArrayFloat) -> float:
        model1 = copy.copy(model)
        for i in range(arr.shape[-1]):
            model1.tr_model.lmob[i][0] = arr[:, :, i]
        return dminv.eval_model_loss_ls(model1, observables, max_obs_time)

    np.testing.assert_allclose(
        adj_model.a_tr_model.a_conc_sources[0]
        .toarray()
        .reshape(grid.nx, grid.ny, time_params.nt, order="F"),
        finite_gradient(model.tr_model.mob[0], wrapper_conc),
    )

    def wrapper_perm(arr: NDArrayFloat) -> float:
        model2 = copy.copy(model)
        model2.fl_model.permeability = arr
        return dminv.eval_model_loss_ls(model2, observables, max_obs_time)

    np.testing.assert_allclose(
        adj_model.a_fl_model.a_permeability_sources.toarray().reshape(
            grid.nx, grid.ny, order="F"
        ),
        finite_gradient(model.fl_model.permeability, wrapper_perm),
    )
