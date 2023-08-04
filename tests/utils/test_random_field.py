import gstools as gs
import numpy as np

from pyrtid.utils import (
    gen_random_ensemble,
    get_log_normalized_mean_from_normal_params,
    get_log_normalized_std_from_normal_params,
    get_normalized_mean_from_lognormal_params,
    get_normalized_std_from_lognormal_params,
)


def test_gen_random_ensemble() -> None:
    nx = 5
    ny = 5

    min = 10
    max = 20
    n_ensemble = 2
    # Compute the mean and the standard deviation that the distribution should have so
    # that <99% of the values are between min and max
    mean = (max + min) / 2
    stdev = (max - min) / 2 / 3  # std ~ 1/6 of the distribution interval

    # Note the dimension is always 3
    model = gs.Gaussian
    # Classic gaussian covariance model
    m = gen_random_ensemble(
        model=model,
        n_ensemble=n_ensemble,
        var=stdev**2,
        len_scale=10,
        mean=mean,
        nx=nx,
        ny=ny,
        seed=2222,
    )

    assert m.shape == (n_ensemble, nx, ny, 1)

    # Log-noramlized cov model
    normalizer = gs.normalizer.LogNormal
    min = 1e-5
    max = 1e-1
    n_ensemble = 2
    # Compute the mean and the standard deviation of the desired
    # log-normal distribution.
    meanlog = (max + min) / 2
    stdevlog = (max - min) / 2  # std
    mean = get_normalized_mean_from_lognormal_params(meanlog, stdevlog)
    stdev = get_normalized_std_from_lognormal_params(meanlog, stdevlog)

    m_log = gen_random_ensemble(
        model=model,
        n_ensemble=n_ensemble,
        var=stdev**2,
        len_scale=10,
        mean=mean,
        nx=nx,
        ny=ny,
        seed=2222,
        normalizer=normalizer,
    )
    assert m_log.shape == (n_ensemble, nx, ny, 1)


def test_param_log_normalization() -> None:
    meanlog = 6.0515
    stdlog = 0.3703

    mean = get_normalized_mean_from_lognormal_params(meanlog, stdlog)
    std = get_normalized_std_from_lognormal_params(meanlog, stdlog)

    assert round(mean, 3) == 1.798
    assert round(std, 3) == 0.061

    np.testing.assert_allclose(
        get_log_normalized_mean_from_normal_params(mean, std), meanlog
    )
    np.testing.assert_allclose(
        get_log_normalized_std_from_normal_params(mean, std), stdlog
    )
