import numpy as np
import pyrtid.forward as dmfwd
import pyrtid.inverse as dminv
import pytest
from pyrtid.inverse.loss_function import (
    get_model_loss_function,
    get_model_ls_loss_function,
    get_model_reg_loss_function,
    ls_loss_function,
)
from pyrtid.inverse.obs import Observable, StateVariable
from pyrtid.inverse.regularization import TikhonovRegularizatorIsotropic


@pytest.mark.parametrize(
    "x_pred, x_obs, std, expected_loss",
    [
        (np.zeros(100), np.zeros(100), np.ones(100), 0),
        (np.zeros(100), np.ones(100), np.ones(100), 0.5),
        (np.zeros(1000), np.ones(1000), np.ones(1000), 0.5),
        (np.zeros(100), np.ones(100) * 2.0, np.ones(100), 2.0),
        (np.zeros(100), np.ones(100) * 2.0, np.ones(100) * 2.0, 0.5),
    ],
)
def test_ls_loss_function(x_pred, x_obs, std, expected_loss) -> None:
    assert ls_loss_function(x_obs, x_pred, std) == expected_loss


@pytest.mark.parametrize(
    "max_obs_time, expected_ls_loss_function, expected_reg_loss_function,"
    " jreg_weight, expected_total_loss_function",
    [
        (None, 14.3234, 3.01311, 2.0, 20.3496),
        (
            0.5,
            12.2510,
            3.01311,
            0.0,
            12.2510,
        ),
    ],
)
def test_get_model_loss_function(
    max_obs_time,
    expected_ls_loss_function,
    expected_reg_loss_function,
    jreg_weight,
    expected_total_loss_function,
) -> None:
    time_params = dmfwd.TimeParameters(duration=1.0, dt_init=1.0)
    geometry = dmfwd.Geometry(nx=20, ny=20, dx=4.5, dy=7.5)
    fl_params = dmfwd.FlowParameters(1e-5)
    tr_params = dmfwd.TransportParameters(1.0, 0.23)
    gch_params = dmfwd.GeochemicalParameters(1.0, 0.0)

    model = dmfwd.ForwardModel(
        geometry,
        time_params,
        fl_params,
        tr_params,
        gch_params,
    )

    # generate synthetic data
    model.tr_model.lmob.append(np.ones((2, 20, 20)) * 2.0)
    model.tr_model.lmob.append(np.ones((2, 20, 20)) * 3.0)
    model.time_params.save_dt()
    model.time_params.save_dt()

    model.tr_model.porosity[:, :] = np.random.default_rng(2023).random(size=(20, 20))

    obs1 = Observable(
        StateVariable.CONCENTRATION,
        node_indices=[2, 4],
        times=np.array([0.0, 0.5, 1.0, 1.56, 2.6]),
        values=np.array([2.9, 5.9, 10.8, 1.8, 8.0]),
        uncertainties=1.0,
    )

    obs2 = Observable(
        StateVariable.CONCENTRATION,
        node_indices=[9, 10, 11],
        times=np.array([0.0, 0.2, 0.3, 4.3, 5.6]),
        values=np.array([3.9, 8.0, 8.0, 9.56, 0.0]),
        uncertainties=1.0,
    )

    assert np.isclose(
        get_model_ls_loss_function(model, [obs1, obs2], max_obs_time),
        expected_ls_loss_function,
        rtol=1e-2,
    )

    param = dminv.AdjustableParameter(
        dminv.ParameterName.POROSITY,
        regularizators=TikhonovRegularizatorIsotropic(geometry.dx, geometry.dy),
    )

    assert np.isclose(
        get_model_reg_loss_function(model, param), expected_reg_loss_function, rtol=1e-2
    )

    assert np.isclose(
        get_model_loss_function(model, [obs1, obs2], param, max_obs_time, jreg_weight),
        expected_total_loss_function,
        rtol=1e-2,
    )
