import numpy as np
import pyrtid.forward as dmfwd
import pyrtid.inverse as dminv
import pytest
from pyrtid.inverse.loss_function import (
    eval_loss_ls,
    eval_model_loss_function,
    eval_model_loss_ls,
)
from pyrtid.inverse.obs import Observable, StateVariable
from pyrtid.inverse.params import eval_weighted_loss_reg
from pyrtid.inverse.regularization import TikhonovRegularizator
from pyrtid.utils import RectilinearGrid


@pytest.mark.parametrize(
    "d_pred, d_obs, std, expected_loss",
    [
        (np.zeros(100), np.zeros(100), np.ones(100), 0),
        (np.zeros(100), np.ones(100), np.ones(100), 50.0),
        (np.zeros(1000), np.ones(1000), np.ones(1000), 500.0),
        (np.zeros(100), np.ones(100) * 2.0, np.ones(100), 200.0),
        (np.zeros(100), np.ones(100) * 2.0, np.ones(100) * 2.0, 50.0),
    ],
)
def test_loss_ls_function(d_pred, d_obs, std, expected_loss) -> None:
    assert eval_loss_ls(d_obs, d_pred, std) == expected_loss


@pytest.mark.parametrize(
    "max_obs_time, expected_loss_ls_function, expected_loss_reg_function,"
    " reg_weight, expected_total_loss_function",
    [
        (None, 100.2838, 2.108667 * 2.0, 2.0, 104.481134),
        (
            0.5,
            61.2510,
            0.0,
            0.0,
            61.2510,
        ),
    ],
)
def test_eval_model_loss_function(
    max_obs_time,
    expected_loss_ls_function,
    expected_loss_reg_function,
    reg_weight,
    expected_total_loss_function,
) -> None:
    time_params = dmfwd.TimeParameters(duration=1.0, dt_init=1.0)
    grid = RectilinearGrid(nx=20, ny=20, dx=4.5, dy=7.5)
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
    model.tr_model.lmob.append(np.ones((2, 20, 20, 1)) * 2.0)
    model.tr_model.lmob.append(np.ones((2, 20, 20, 1)) * 3.0)
    model.time_params.save_dt()
    model.time_params.save_dt()

    model.tr_model.porosity[:, :] = np.random.default_rng(2023).random(size=(20, 20, 1))

    obs1 = Observable(
        StateVariable.CONCENTRATION,
        node_indices=[2, 4],
        times=np.array([0.0, 0.5, 1.0, 1.56, 2.6]),
        values=np.array([2.9, 5.9, 10.8, 1.8, 8.0]),
        uncertainties=1.0,
        sp=0,
    )

    obs2 = Observable(
        StateVariable.CONCENTRATION,
        node_indices=[9, 10, 11],
        times=np.array([0.0, 0.2, 0.3, 4.3, 5.6]),
        values=np.array([3.9, 8.0, 8.0, 9.56, 0.0]),
        uncertainties=1.0,
        sp=0,
    )

    np.testing.assert_allclose(
        eval_model_loss_ls(model, [obs1, obs2], max_obs_time),
        expected_loss_ls_function,
        rtol=1e-2,
    )

    param = dminv.AdjustableParameter(
        dminv.ParameterName.POROSITY,
        regularizators=TikhonovRegularizator(grid),
        reg_weight_update_strategy=dminv.regularization.ConstantRegWeight(reg_weight),
    )

    np.testing.assert_allclose(
        eval_weighted_loss_reg(param, model),
        expected_loss_reg_function,
        rtol=1e-2,
    )

    np.testing.assert_allclose(
        eval_model_loss_function(model, [obs1, obs2], param, max_obs_time),
        expected_total_loss_function,
        rtol=1e-2,
    )
