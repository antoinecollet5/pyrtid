import logging
from contextlib import nullcontext as does_not_raise
from typing import Optional

import numdifftools as nd
import numpy as np
import pyrtid.forward as dmfwd
import pyrtid.inverse as dminv
import pyrtid.utils.spde as spde
import pytest
from pyrtid.inverse.preconditioner import (
    GradientScalerConfig,
    arctanh_wrapper,
    darctanh_wrapper,
    dtanh_wrapper,
    gd_parametrize,
    get_factor_enforcing_grad_inf_norm,
    get_gd_weights,
    get_max_update,
    get_theta_init_normal,
    get_theta_init_uniform,
    logistic,
    logit,
    scale_pcd,
    tanh_wrapper,
    to_new_range,
    to_new_range_derivative,
)
from pyrtid.utils import (
    NDArrayFloat,
    NDArrayInt,
    indices_to_node_number,
    sparse_cholesky,
)
from scipy._lib._util import check_random_state  # To handle random_state

logger = logging.getLogger("ROOT")
scaler_log = logging.getLogger("SCALER")
logger.setLevel(logging.INFO)
scaler_log.setLevel(logging.INFO)


@pytest.mark.parametrize(
    "ne, expected_exception",
    (
        [100, does_not_raise()],
        [0, pytest.raises(ValueError, match=r"ne must be an integer, >=2.")],
        [
            "smth not supported",
            pytest.raises(ValueError, match=r"ne must be an integer, >=2."),
        ],
    ),
)
def test_get_theta_init_uniform(ne: int, expected_exception) -> None:
    with expected_exception:
        np.testing.assert_allclose(
            get_gd_weights(get_theta_init_uniform(ne)), np.ones(ne) / np.sqrt(ne)
        )


@pytest.mark.parametrize(
    "ne, mu, sigma, random_state",
    ([100, 0.5, 0.15, 2015],),
)
def test_get_theta_init_normal(
    ne: int, sigma: float, mu: float, random_state: int
) -> None:
    theta_init = get_theta_init_normal(
        ne, mu=mu, sigma=sigma, random_state=random_state
    )
    a = check_random_state(random_state).normal(loc=mu, scale=sigma, size=ne)
    expected_a = a / np.linalg.norm(a)
    np.testing.assert_allclose(get_gd_weights(theta_init), expected_a)


@pytest.mark.parametrize(
    "theta, expected_exception",
    (
        [get_theta_init_uniform(100), does_not_raise()],
        [np.random.default_rng(2024).normal(0, 5, 10), does_not_raise()],
        [np.array([]), pytest.raises(ValueError, match="The theta vector is empty!")],
    ),
)
def test_get_gd_weights(theta, expected_exception) -> None:
    with expected_exception:
        weights = get_gd_weights(theta)
        np.testing.assert_almost_equal(np.sum(weights**2), 1.0)


@pytest.mark.parametrize(
    "ne, expected_exception",
    (
        [100, does_not_raise()],
        [10, does_not_raise()],
    ),
)
def test_gd_parametrize(ne, expected_exception) -> None:
    with expected_exception:
        weights = get_gd_weights(get_theta_init_uniform(ne))
        W = np.random.default_rng(2024).normal(0, 1.0, size=(100000, ne))
        w = gd_parametrize(W, weights)
        np.testing.assert_almost_equal(np.std(w), 1.0, decimal=2)
        np.testing.assert_almost_equal(np.mean(w), 0.0, decimal=2)


@pytest.mark.parametrize(
    "precond,args,kwargs,lbounds, ubounds, eps,expected_exception",
    (
        [
            dminv.NoTransform,
            (),
            {},
            np.ones(10) * 0.1,
            np.ones(10) * 10,
            1e-5,
            does_not_raise(),
        ],
        [
            dminv.LogTransform,
            (),
            {},
            np.ones(10) * 0.1,
            np.ones(10) * 10,
            1e-5,
            does_not_raise(),
        ],
        [
            dminv.SqrtTransform,
            (),
            {},
            np.ones(10) * 0.1,
            np.ones(10) * 10,
            1e-5,
            does_not_raise(),
        ],
        [
            dminv.SqrtTransform,
            (),
            {},
            np.ones(10) * 0.1,
            np.ones(10) * 10,
            1e-5,
            does_not_raise(),
        ],
        [
            dminv.Normalizer,
            (np.random.default_rng(2024).normal(50.0, 100.0, 10),),
            {},
            np.ones(10) * 0.1,
            np.ones(10) * 10,
            1e-5,
            does_not_raise(),
        ],
        [
            dminv.BoundsRescaler,
            (np.ones(10) * 0.1, np.ones(10) * 10),
            {},
            np.ones(10) * 0.1,
            np.ones(10) * 10,
            1e-5,
            does_not_raise(),
        ],
        [
            dminv.LinearTransform,
            (-16.8, 4.98),
            {},
            np.ones(10) * -23.9,
            np.ones(10) * 89.0,
            1e-5,
            does_not_raise(),
        ],
        [
            dminv.InvAbsTransform,
            (),
            {},
            np.ones(10) * 23.9,
            np.ones(10) * 89.0,
            1e-5,
            does_not_raise(),
        ],
        [
            dminv.ChainedTransforms,
            ((dminv.LinearTransform(slope=-16.8, y_intercept=4.98),),),
            {},
            np.ones(10) * -23.9,
            np.ones(10) * 89.0,
            1e-5,
            does_not_raise(),
        ],
        [
            dminv.ChainedTransforms,
            (
                (
                    dminv.InvAbsTransform(),
                    dminv.LinearTransform(slope=-16.8, y_intercept=4.98),
                ),
            ),
            {},
            np.ones(10) * 23.9,
            np.ones(10) * 89.0,
            1e-5,
            does_not_raise(),
        ],
        [
            dminv.ChainedTransforms,
            (
                (
                    dminv.Normalizer(
                        np.random.default_rng(2024).normal(50.0, 100.0, 10)
                    ),
                    dminv.LinearTransform(slope=-16.8, y_intercept=4.98),
                    dminv.InvAbsTransform(),
                ),
            ),
            {},
            np.ones(10) * -23.9,
            np.ones(10) * 89.0,
            1e-5,
            does_not_raise(),
        ],
        [
            dminv.SigmoidRescaler,
            (),
            {},
            np.ones(10) * 0.0,
            np.ones(10) * 1.0,
            1e-5,
            does_not_raise(),
        ],
        [
            dminv.RangeRescaler,
            (-1.0, 22.0, 0.0, 1.0),
            {},
            np.ones(10) * 0.0,
            np.ones(10) * 1.0,
            1e-5,
            does_not_raise(),
        ],
        [
            dminv.SubSelector,
            ([1, 2, 6], dmfwd.Geometry(nx=5, ny=2, dx=1.0, dy=1.0)),
            {},
            np.ones(10) * 0.1,
            np.ones(10) * 1.0,
            1e-5,
            does_not_raise(),
        ],
    ),
)
def test_preconditioners(
    precond: dminv.Preconditioner,
    args,
    kwargs,
    lbounds,
    ubounds,
    eps,
    expected_exception,
) -> None:
    with expected_exception:
        p: dminv.Preconditioner = precond(*args, **kwargs)
        p.test_preconditioner(lbounds=lbounds, ubounds=ubounds, eps=eps)
        p.transform_bounds(np.vstack([lbounds, ubounds]).T)


def test_bad_preconditioner() -> None:
    class WrongPcd(dminv.Preconditioner):
        def _transform(self, s_cur: NDArrayFloat) -> NDArrayFloat:
            """
            Apply the preconditioning/parametrization.

            Parameters
            ----------
            s_raw : NDArrayFloat
                Non-conditioned parameter values.

            Returns
            -------
            NDArrayFloat
                Conditioned parameter values.
            """
            return s_cur

        def _backtransform(self, s_cond: NDArrayFloat) -> NDArrayFloat:
            """
            Apply the back-preconditioning/parametrization.

            Parameters
            ----------
            s_cond : NDArrayFloat
                Conditioned parameter values.

            Returns
            -------
            NDArrayFloat
                Non-conditioned parameter values.
            """
            return s_cond - 1  # this does not match the transform

        def _dtransform_vec(
            self, s_raw: NDArrayFloat, gradient: NDArrayFloat
        ) -> NDArrayFloat:
            """
            Return the transform gradient of a function to match the new parameter.
            """
            return s_raw * gradient  # this does not match the transform

        def _dbacktransform_vec(
            self, s_raw: NDArrayFloat, gradient: NDArrayFloat
        ) -> NDArrayFloat:
            """
            Return the transform gradient of a function to match the new parameter.
            """
            return s_raw * gradient  # this does not match the transform

        def _dbacktransform_inv_vec(
            self, s_cond: NDArrayFloat, gradient: NDArrayFloat
        ) -> NDArrayFloat:
            """
            Return the inverse of the backtransform 1st derivative times a vector.
            """
            return s_cond * gradient  # this does not match the transform

    pcd = WrongPcd()
    with pytest.raises(
        ValueError,
        match=(
            "The given backconditioner does not match the preconditioner! or"
            " the provided bounds are not correct."
        ),
    ):
        pcd.test_preconditioner(1.0, 2.0)


def test_std_rescaler() -> None:
    s_prior = np.random.default_rng(2024).normal(50.0, 100.0, 250)

    for pcd in [dminv.StdRescaler(s_prior), dminv.StdRescaler(s_prior, 100.0)]:
        s_cond = pcd.transform(s_prior)
        std = np.std(s_cond)
        mean = np.mean(s_cond)

        # test the correctness
        pcd.test_preconditioner(-1e-6, 1e6, shape=(250), eps=1e-3)

        # test that the scaling is correct -> should be zero because we remove the prior
        np.testing.assert_allclose(
            np.array([0.0, 0.0]), np.array([mean, std]), rtol=1e-5, atol=1e-5
        )

        # we divide by 2
        s_cur = np.random.default_rng(2024).normal(25.0, 50.0, 250)
        s_cond = pcd.transform(s_cur)
        std = np.std(s_cond)
        mean = np.mean(s_cond)
        # test that the scaling is correct
        np.testing.assert_allclose(
            np.array([-0.25, 0.5]), np.array([mean, std]), rtol=1e-2, atol=1e-2
        )


def test_normalizer() -> None:
    s_prior = np.random.default_rng(2024).normal(50.0, 100.0, 50000)
    pcd = dminv.Normalizer(s_prior)

    s_cond = pcd.transform(s_prior)
    std = np.std(s_cond)
    mean = np.mean(s_cond)

    # test that the scaling is correct
    np.testing.assert_allclose(
        np.array([0.0, 1.0]), np.array([mean, std]), rtol=1e-5, atol=1e-5
    )

    # we divide by 2
    s_cur = np.random.default_rng(2024).normal(25.0, 50.0, 50000)
    s_cond = pcd.transform(s_cur)
    std = np.std(s_cond)
    mean = np.mean(s_cond)
    # test that the scaling is correct
    np.testing.assert_allclose(
        np.array([-0.25, 0.5]), np.array([mean, std]), rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize("is_update_mean", [False, True])
def test_GDP_SPDE(is_update_mean: bool) -> None:
    ne = 100
    # Grid
    nx = 20  # number of voxels along the x axis
    ny = 20  # number of voxels along the y axis
    nz = 1
    dx = 5.0  # voxel dimension along the x axis
    dy = 5.0  # voxel dimension along the y axis
    dz = 5.0

    len_scale = 20.0  # m
    kappa = 1 / len_scale
    scaling_factor = 1.0

    mean = 100.0  # trend of the field
    std = 150.0  # standard deviation of the field

    # Create a precision matrix
    Q_ref = spde.get_precision_matrix(
        nx, ny, nz, dx, dy, dz, kappa, scaling_factor, spatial_dim=2, sigma=std
    )
    cholQ_ref = sparse_cholesky(Q_ref)
    # Non conditional simulation -> change the random state to obtain a different field
    simu_ = spde.simu_nc(cholQ_ref, random_state=2026).reshape(ny, nx).T
    reference_grade_ppm = np.abs(simu_ + mean)

    # Conditioning data
    _ix = np.array([int(nx / 4), 2 * int(nx / 4), 3 * int(nx / 4)])
    _iy = np.array([int(ny / 5), 2 * int(ny / 5), 3 * int(ny / 5), 4 * int(ny / 5)])
    dat_coords = np.array(np.meshgrid(_ix, _iy)).reshape(2, -1)
    # Get the node numbers
    dat_nn: NDArrayInt = indices_to_node_number(dat_coords[0, :], nx, dat_coords[1, :])
    dat_val = reference_grade_ppm.ravel("F")[dat_nn]

    # Condition with the exact data -> we assume a large noise over the data
    dat_var = np.ones(dat_val.size) * (100**2)

    # Generate new points with error -> some variance on the measures
    dat_val_noisy = dat_val + np.sqrt(dat_var) * (
        np.random.default_rng(2048).normal(scale=0.1, size=dat_val.size)
    )

    # Compute the average on the data points (trend)
    estimated_mean = float(np.average(dat_val_noisy))
    estimated_std = float(np.std(dat_val_noisy))

    scaling_factor = 1

    # Create a precision matrix
    Q_nc = spde.get_precision_matrix(
        nx,
        ny,
        1,
        dx,
        dy,
        1.0,
        kappa,
        scaling_factor,
        spatial_dim=2,
        sigma=estimated_std,
    )
    Q_c = spde.condition_precision_matrix(Q_nc, dat_nn, dat_var)

    # Decompose with cholesky
    cholQ_nc = sparse_cholesky(Q_nc)
    cholQ_c = sparse_cholesky(Q_c)

    lbounds = np.ones((nx * ny)) * -1000
    ubounds = np.ones((nx * ny)) * 1500
    theta_test = get_theta_init_uniform(ne) * (
        1 + 0.1 * np.random.default_rng(2024).normal(size=ne - 1)
    )

    # Non conditional simulations
    dminv.GDPNCS(
        ne, Q_nc, estimated_mean, is_update_mean=is_update_mean
    ).test_preconditioner(lbounds, ubounds)
    # with extra parameters
    pcd_gdpncs = dminv.GDPNCS(
        ne,
        Q_nc,
        estimated_mean,
        theta=theta_test,
        cholQ_nc=cholQ_nc,
        random_state=2024,
        is_update_mean=is_update_mean,
    )
    s_nc = pcd_gdpncs.backtransform(pcd_gdpncs(np.zeros(cholQ_nc.P().size)))
    np.testing.assert_allclose(pcd_gdpncs.theta, theta_test, rtol=1e-5)
    pcd_gdpncs.test_preconditioner(lbounds, ubounds, eps=1e-8)
    pcd_gdpncs.transform_bounds(np.vstack([lbounds, ubounds]).T)

    # Test gradient scaling
    grad_nc = (
        np.random.default_rng(2024).normal(scale=1.0, size=(nx * ny * nz)) * 1e-5 + 2e-5
    )  # 1.0

    gsc_nc = GradientScalerConfig(
        max_workers=10,
        max_change_target=1e-1,
        n_samples_in_first_round=10,
        rtol=1e-2,  # 1 percent precision
    )

    initial_max_update = get_max_update(1.0, pcd_gdpncs, s_nc, grad_nc)
    logger.info(f"initial_max_update = {initial_max_update}")

    sf_gdpncs = get_factor_enforcing_grad_inf_norm(
        s_nc,
        grad_nc,
        pcd_gdpncs,
        gsc_nc,
        logger=logging.getLogger("SCALER"),
    )

    new_max_update = get_max_update(sf_gdpncs, pcd_gdpncs, s_nc, grad_nc)
    logger.info(f"new_max_update = {new_max_update}")

    np.testing.assert_allclose(
        new_max_update, gsc_nc.max_change_target, rtol=gsc_nc.rtol
    )

    # Conditional simulations
    pcd_gdpcs = dminv.GDPCS(
        ne,
        Q_nc,
        Q_c,
        estimated_mean,
        dat_nn,
        dat_val,
        dat_var,
        is_update_mean=False,
    )
    pcd_gdpcs.test_preconditioner(lbounds, ubounds)
    # with extra parameters
    pcd_gdpcs = dminv.GDPCS(
        ne,
        Q_nc,
        Q_c,
        estimated_mean,
        dat_nn,
        dat_val,
        dat_var,
        theta=theta_test,
        cholQ_nc=cholQ_nc,
        cholQ_c=cholQ_c,
        random_state=2024,
        is_update_mean=is_update_mean,
    )
    pcd_gdpcs.smart_copy().test_preconditioner(lbounds, ubounds, rtol=1e-4, eps=1e-6)
    # pcd_gdpcs.test_preconditioner(lbounds, ubounds, rtol=1e-4, eps=1e-6)
    pcd_gdpcs.transform_bounds(np.vstack([lbounds, ubounds]).T)
    s_nc = pcd_gdpcs.backtransform(pcd_gdpcs(np.zeros(cholQ_nc.P().size)))
    np.testing.assert_allclose(pcd_gdpcs.theta, theta_test)

    grad_nc = (
        np.random.default_rng(2024).normal(scale=1.0, size=(nx * ny * nz)) * 2e1 + 3e1
    )  # 1.0

    gsc_cond = GradientScalerConfig(
        max_change_target=100.0,
        n_samples_in_first_round=60,
        rtol=1e-2,  # 1 percent precision
    )

    initial_max_update = get_max_update(1.0, pcd_gdpcs, s_nc, grad_nc)
    logger.info(f"initial_max_update = {initial_max_update}")

    sf_gdpcs = get_factor_enforcing_grad_inf_norm(
        s_nc,
        grad_nc,
        pcd_gdpcs,
        gsc_cond,
        logger=scaler_log,
    )
    new_max_update = get_max_update(sf_gdpcs, pcd_gdpcs, s_nc, grad_nc)
    logger.info(f"new_max_update = {new_max_update}")

    np.testing.assert_allclose(
        new_max_update, gsc_cond.max_change_target, rtol=gsc_cond.rtol
    )

    if not is_update_mean:
        # now make a tests that fails -> does not find a satisfying scaling scalar
        sf_gdpcs = get_factor_enforcing_grad_inf_norm(
            s_nc,
            grad_nc,
            pcd_gdpcs,
            GradientScalerConfig(
                max_change_target=10000.0,
                n_samples_in_first_round=10,
                rtol=1e-2,  # 1 percent precision
            ),
            logger=scaler_log,
        )
        # so the preconditioner is not modified

        assert sf_gdpcs == 1


@pytest.mark.parametrize(
    "s0,rate, supremum", [(0.1, 1.0, 1.0), (-4, 0.5, 5.3), (2.1, 2.0, 2.09)]
)
def test_logistic(s0: float, rate: float, supremum: float) -> None:
    x = np.linspace(-5, 5, 100)
    y = logistic(x, s0=s0, rate=rate, supremum=supremum)
    np.testing.assert_allclose(x, logit(y, s0=s0, rate=rate, supremum=supremum))

    np.testing.assert_allclose(y.max(), supremum, rtol=1e-1)
    np.testing.assert_allclose(
        logistic(s0, s0=s0, rate=rate, supremum=supremum), 0.5 * supremum, rtol=1e-2
    )


def test_rescale_to_bounds() -> None:
    x = np.linspace(-5, 5, 100)
    y = to_new_range(x, -5, 5, -1.0, 1.0)
    assert np.max(y) == 1.0
    assert np.min(y) == -1.0

    g = np.linspace(-10, 10, 100)

    def to_new_range_wrapper(s) -> NDArrayFloat:
        return to_new_range(s, -5, 5, -1.0, 1.0)

    np.testing.assert_allclose(
        to_new_range_derivative(x, -5, 5, -1.0, 1.0) * g,
        nd.Jacobian(to_new_range_wrapper)(x) @ g,
    )

    y = to_new_range(x, -5, 5, 1e-10, 1e-1, is_log10=True)
    assert np.max(y) == 1e-1
    assert np.min(y) == 1e-10

    def to_new_range_wrapper_log(s) -> NDArrayFloat:
        return to_new_range(s, -1, 1.0, 1.0, 10, is_log10=True)

    x = np.logspace(-1, 1, 100)
    to_new_range(x, -1.0, 1.0, 1, 10, is_log10=True)

    np.testing.assert_allclose(
        to_new_range_derivative(x, -1, 1, 1.0, 10.0, is_log10=True) * g,
        nd.Jacobian(to_new_range_wrapper_log)(x) @ g,
    )


@pytest.mark.parametrize(
    "s0,rate, supremum", [(0.1, 1.0, 1.0), (-4, 0.5, 5.3), (2.1, 2.0, 2.09)]
)
def test_tanh_wrapper(s0: float, rate: float, supremum: float) -> None:
    x = np.linspace(-5, 5, 100)
    y = tanh_wrapper(x, s0, rate, supremum)

    np.testing.assert_allclose(x, arctanh_wrapper(y, s0, rate, supremum), rtol=1e-5)

    g = np.linspace(-10, 10, 100)

    def tanh_wrapper2(s) -> NDArrayFloat:
        return tanh_wrapper(s, s0, rate, supremum)

    np.testing.assert_allclose(
        dtanh_wrapper(x, s0, rate, supremum) * g,
        nd.Jacobian(tanh_wrapper2)(x) @ g,
        rtol=1e-5,
        atol=1e-5,
    )

    def arctanh_wrapper2(s) -> NDArrayFloat:
        return arctanh_wrapper(s, s0, rate, supremum)

    if rate < 1.5:
        np.testing.assert_allclose(
            darctanh_wrapper(y, s0, rate, supremum) * g,
            nd.Jacobian(arctanh_wrapper2, step=1e-10)(y) @ g,
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.parametrize(
    "is_log10, rate",
    [(True, 1.0), (True, 2.0), (True, 3.0), (False, 1.0), (False, 2.0), (False, 3.0)],
)
def test_sigmoid_rescaler_bounded(is_log10, rate):
    pcd = dminv.SigmoidRescalerBounded(1e-9, 1e-4, rate=rate, is_log10=is_log10)
    x = np.linspace(-5, 5, 100)
    y = pcd.backtransform(x)
    x2 = pcd.transform(y)
    np.testing.assert_allclose(x, x2, rtol=1e-4)

    pcd.test_preconditioner(1e-9, 1e-4, shape=(100,), eps=1e-9, rtol=1e-4)

    np.testing.assert_allclose(
        pcd.transform_bounds(np.array([[1, 2, 3], [2, 3, 5]]).T),
        np.array([[-10, -10, -10], [10, 10, 10]]).T,
    )

    # pcd.transform_bounds(np.array([[1, 2, 3], [2,3,5]]).T)


def test_uniform2gaussian() -> None:
    pcd = dminv.Uniform2Gaussian(ud_lbound=-3, ud_ubound=5.0, gd_mu=2.0, gd_std=12.78)
    pcd.test_preconditioner(-2, 2)


def test_boundsclipper() -> None:
    pcd = dminv.BoundsClipper(np.ones(15) * -1.0, np.ones(15) * 5.0)

    test_data = np.arange(-5, 10, 1, dtype=np.float64)

    np.testing.assert_array_equal(
        np.array(
            [
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                5.0,
                5.0,
                5.0,
                5.0,
            ]
        ),
        pcd.backtransform(test_data),
    )

    np.testing.assert_array_equal(
        np.array(
            [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0]
        ),
        pcd.dbacktransform_vec(test_data, np.ones(15)),
    )

    gradient = np.random.normal(0, 25, size=15)
    np.testing.assert_allclose(
        pcd.dbacktransform_vec(test_data, gradient),
        # Finite difference differentiation
        nd.Jacobian(pcd.backtransform, step=None)(test_data).T @ gradient,  # type: ignore
        rtol=1e-5,
    )

    pcd.test_preconditioner(lbounds=np.zeros(15), ubounds=np.ones(15) * 10.0)


@pytest.mark.parametrize(
    "pcd,pcd_change_eval, max_change_target,lb_nc, ub_nc",
    (
        [
            dminv.LinearTransform(slope=50.0, y_intercept=0.0),
            dminv.NoTransform(),
            0.8,
            None,
            None,
        ],
        [
            dminv.LinearTransform(slope=50.0, y_intercept=0.0),
            dminv.NoTransform(),
            np.log(10),
            None,
            None,
        ],
        [
            dminv.LinearTransform(slope=50.0, y_intercept=0.0),
            dminv.LogTransform(),
            0.8,
            None,
            None,
        ],
        [
            dminv.LinearTransform(slope=50.0, y_intercept=0.0),
            dminv.LogTransform(),
            np.log(10),
            None,
            None,
        ],
        [dminv.LogTransform(), dminv.LogTransform(), 0.8, 1e-7, 1e-2],
        [dminv.LogTransform(), dminv.LogTransform(), np.log(10), 1e-7, 1e-2],
    ),
)
def test_gradient_scaling(
    pcd: dminv.Preconditioner,
    pcd_change_eval: dminv.Preconditioner,
    max_change_target: bool,
    lb_nc: Optional[float],
    ub_nc: Optional[float],
) -> None:
    s_nc = np.ones(10) * 1e-4  # 0.1
    grad_nc = -np.ones_like(s_nc) * 600.0  # 1.0

    gsc = GradientScalerConfig(
        max_workers=10,
        max_change_target=max_change_target,
        pcd_change_eval=pcd_change_eval,
        n_samples_in_first_round=10,
        rtol=1e-2,  # 1 percent precision
        lb=1e-10,
        ub=1e10,
    )

    initial_max_update = get_max_update(
        1.0, pcd, s_nc, grad_nc, gsc, lb_nc=lb_nc, ub_nc=ub_nc
    )
    logger.info(f"Initial_max_update = {initial_max_update}\n")

    scaling_factor = get_factor_enforcing_grad_inf_norm(
        s_nc, grad_nc, pcd, gsc, logger=scaler_log, lb_nc=lb_nc, ub_nc=ub_nc
    )
    new_max_update = get_max_update(
        scaling_factor, pcd, s_nc, grad_nc, gsc, lb_nc=lb_nc, ub_nc=ub_nc
    )
    logging.info(f"New_max_update = {new_max_update}")

    np.testing.assert_allclose(new_max_update, gsc.max_change_target, rtol=gsc.rtol)

    # Call again -> the preconditioner should not be modified
    new_scaling_factor = get_factor_enforcing_grad_inf_norm(
        s_nc,
        grad_nc,
        scale_pcd(scaling_factor, pcd),
        gsc,
        logger=scaler_log,
        lb_nc=lb_nc,
        ub_nc=ub_nc,
    )
    assert new_scaling_factor == 1.0
