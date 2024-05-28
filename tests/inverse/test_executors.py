import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from pyrtid.inverse.executors.base import DataModel

MATCH_COV_OBS_ERR = re.escape(
    r"cov_obs must be either a 1D array of diagonal covariances, "
    r"or a 2D covariance matrix. The shape is either (Nobs) or (Nobs, Nobs)."
)


@pytest.mark.parametrize(
    "cov_obs_shape,s_init_shape,expected_n_ensemble,expected_error",
    [
        ((10,), (100,), 1, does_not_raise()),
        ((10, 10), (100, 20), 20, does_not_raise()),
        ((11,), (100,), 1, pytest.raises(ValueError, match=MATCH_COV_OBS_ERR)),
        ((10, 20), (100,), 1, pytest.raises(ValueError, match=MATCH_COV_OBS_ERR)),
        ((10, 10, 10), (100,), 1, pytest.raises(ValueError, match=MATCH_COV_OBS_ERR)),
    ],
)
def test_data_model(
    cov_obs_shape, s_init_shape, expected_n_ensemble, expected_error
) -> None:
    obs = np.zeros((10))
    s_init = np.zeros((s_init_shape))
    cov_obs = np.zeros((cov_obs_shape))

    with expected_error:
        data_model = DataModel(obs, s_init, cov_obs)
        data_model.cov_obs
        assert data_model.d_dim == 10
        assert data_model.s_dim == 100
        assert data_model.n_ensemble == expected_n_ensemble
