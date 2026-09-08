import logging
from contextlib import nullcontext as does_not_raise
from typing import Optional, no_type_check

import covmats
import numdifftools as nd
import numpy as np
import pyrtid.inverse as dminv
import pytest
import scipy as sp
from covmats._sparse_helpers import get_SPD_sparse_n11_example
from pyrtid.utils import (
    NDArrayFloat,
    RectilinearGrid,
    check_random_state,
)
from pyrtid.utils.preconditioner import (
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


@no_type_check
def _get_L_D_P(A: sp.sparse.sparray):
    """
    Return L, D and P from the factorization L @ D @ L' = P @ A @ P' using sksparse.

    Note that sksparse uses SuiteSparse which is LGPL licence.
    """
    import sksparse.cholmod as cholmod

    # Need to take the API change into account
    try:
        # sksparse 4.x
        L, D, P = cholmod.ldl(A, order="amd")
    except AttributeError:
        # sksparse 5.x
        f = cholmod.cholesky(A)
        (L, D), P = f.L_D(), f.P()
    return L, D, P


logger = logging.getLogger("ROOT")
scaler_log = logging.getLogger("SCALER")
logger.setLevel(logging.INFO)
scaler_log.setLevel(logging.INFO)


def _get_scf(A: sp.sparse.sparray) -> covmats.SparseCholeskyFactor:
    """Return a cholesky factorization of the precision matrix."""
    return covmats.SparseCholeskyFactor(*_get_L_D_P(A))


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
            ([1, 2, 6], RectilinearGrid(nx=5, ny=2, dx=1.0, dy=1.0)),
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
            self, s_cond: NDArrayFloat, gradient: NDArrayFloat
        ) -> NDArrayFloat:
            """
            Return the transform gradient of a function to match the new parameter.
            """
            return s_cond * gradient  # this does not match the transform

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


# ---------------------------------------------------------------------------
# GDPNCS / GDPCS: Gradual Deformation parametrization, now built on covmats
# (dense/sparse covariance representations) instead of the old ad hoc
# SPDE/precision-matrix code.
# ---------------------------------------------------------------------------


def _get_dense_cov(n: int = 9, seed: int = 5) -> covmats.CovViaCholesky:
    """A small, well-conditioned dense covariance for GDPNCS/GDPCS tests."""
    rng = np.random.default_rng(seed)
    A = rng.random((n, n))
    A = A @ A.T + n * np.eye(n)  # strongly positive definite
    return covmats.CovViaCholesky(np.linalg.cholesky(A))


def _get_sparse_cov(seed: int = 2026) -> covmats.CovViaSparseCholesky:
    """A sparse covariance (via SparseCholeskyFactor) for GDPNCS/GDPCS tests."""
    A = get_SPD_sparse_n11_example(seed=seed)
    return covmats.CovViaSparseCholesky(_get_scf(A))


def _dense_point_obs(idx, n: int) -> NDArrayFloat:
    """Dense (non-LinearOperator) point-observation matrix, for coverage of
    the `aslinearoperator` fallback in `GDPCS._dbacktransform_vec`."""
    H = np.zeros((len(idx), n))
    H[np.arange(len(idx)), idx] = 1.0
    return H


def test_preconditioner_1d_vector_validation() -> None:
    pcd = dminv.NoTransform()
    bad = np.ones((2, 2))
    grad_bad = np.ones((2, 2))
    with pytest.raises(ValueError, match="'transform' method expects a 1D vector!"):
        pcd.transform(bad)
    with pytest.raises(ValueError, match="'backtransform' method expects a 1D vector!"):
        pcd.backtransform(bad)
    with pytest.raises(ValueError, match="'dtransform_vec' method expects 1D vectors!"):
        pcd.dtransform_vec(bad, grad_bad)
    with pytest.raises(
        ValueError, match="'dbacktransform_vec' method expects 1D vectors!"
    ):
        pcd.dbacktransform_vec(bad, grad_bad)
    with pytest.raises(
        ValueError, match="'dbacktransform_inv_vec' method expects 1D vectors!"
    ):
        pcd.dbacktransform_inv_vec(bad, grad_bad)


@pytest.mark.parametrize("is_update_mean", [False, True])
def test_GDPNCS(is_update_mean: bool) -> None:
    ne = 8
    cov = _get_dense_cov()
    n = cov.shape[0]

    theta_test = get_theta_init_uniform(ne) * (
        1 + 0.05 * np.random.default_rng(2024).normal(size=ne - 1)
    )
    pcd = dminv.GDPNCS(
        ne,
        cov,
        estimated_mean=12.5,
        theta=theta_test,
        random_state=2024,
        is_update_mean=is_update_mean,
    )
    np.testing.assert_allclose(pcd.theta, theta_test)

    lbounds = np.ones(n) * -1e10
    ubounds = np.ones(n) * 1e10
    # exercises transform/backtransform round-trip, dtransform_vec,
    # dbacktransform_vec (finite-difference-checked), and
    # dbacktransform_inv_vec (caught NotImplementedError)
    pcd.test_preconditioner(lbounds, ubounds, eps=1e-6, rtol=1e-3)

    n_cond = (ne - 1) + (1 if is_update_mean else 0)
    bounds = pcd.transform_bounds(np.vstack([lbounds, ubounds]).T)
    assert bounds.shape == (n_cond, 2)
    np.testing.assert_array_equal(bounds[:, 0], -1e100)
    np.testing.assert_array_equal(bounds[:, 1], 1e100)

    # smart_copy must deep-copy theta/estimated_mean
    cp = pcd.smart_copy()
    cp.theta[0] += 1.0
    cp.estimated_mean += 1.0
    assert not np.allclose(cp.theta, pcd.theta)
    assert cp.estimated_mean != pcd.estimated_mean


def test_GDPNCS_default_theta() -> None:
    ne = 6
    cov = _get_dense_cov()
    pcd = dminv.GDPNCS(ne, cov, estimated_mean=1.0, random_state=7)
    # default theta -> all Ne realizations equally weighted
    np.testing.assert_allclose(get_gd_weights(pcd.theta), np.ones(ne) / np.sqrt(ne))


def test_GDPNCS_sparse_cov() -> None:
    cov = _get_sparse_cov()
    pcd = dminv.GDPNCS(6, cov, estimated_mean=0.0, random_state=11)
    lbounds = np.ones(cov.shape[0]) * -1e10
    ubounds = np.ones(cov.shape[0]) * 1e10
    pcd.test_preconditioner(lbounds, ubounds, eps=1e-6, rtol=1e-3)


def test_GDPNCS_colorize_adjoint_not_implemented(monkeypatch) -> None:
    cov = _get_dense_cov()
    pcd = dminv.GDPNCS(6, cov, estimated_mean=1.0, random_state=1)
    s_cond = pcd.transform(np.zeros(cov.shape[0]))
    monkeypatch.delattr(covmats.CovarianceMatrix, "colorize_adjoint")
    with pytest.raises(
        NotImplementedError,
        match=r"GDPNCS\._dbacktransform_vec requires `cov\.colorize_adjoint`",
    ):
        pcd.dbacktransform_vec(s_cond, np.ones(cov.shape[0]))


@pytest.mark.parametrize("is_update_mean", [False, True])
@pytest.mark.parametrize("obs_op_as_dense_array", [False, True])
@pytest.mark.parametrize("obs_cov_as_covmat", [False, True])
def test_GDPCS(
    is_update_mean: bool, obs_op_as_dense_array: bool, obs_cov_as_covmat: bool
) -> None:
    ne = 8
    cov = _get_dense_cov()
    n = cov.shape[0]
    obs_idx = [1, 4, 7]

    H = (
        _dense_point_obs(obs_idx, n)
        if obs_op_as_dense_array
        else covmats.make_point_observation_operator(obs_idx, n=n)
    )
    obs_values = np.array([0.2, -0.1, 0.4])
    obs_cov = (
        covmats.CovViaDiagonal(np.array([0.1, 0.1, 0.1])) if obs_cov_as_covmat else 0.1
    )

    theta_test = get_theta_init_uniform(ne)
    pcd = dminv.GDPCS(
        ne,
        cov,
        H,
        obs_values,
        obs_cov,
        estimated_mean=2.0,
        theta=theta_test,
        random_state=2024,
        is_update_mean=is_update_mean,
    )
    np.testing.assert_allclose(pcd.theta, theta_test)

    lbounds = np.ones(n) * -1e10
    ubounds = np.ones(n) * 1e10
    # exercises _backtransform (Matheron's rule implemented directly via
    # `_solve_data_space_system`, a conjugate-gradient solve shared with the
    # gradient), dbacktransform_vec (finite-difference-checked, both the
    # `obs_op` LinearOperator-vs-array and `obs_cov` CovarianceMatrix-vs-scalar
    # branches), and dbacktransform_inv_vec (caught NotImplementedError).
    pcd.test_preconditioner(lbounds, ubounds, eps=1e-6, rtol=1e-3)

    # the conditioned field should be deterministic across repeated calls at
    # the same theta (fixed, once-drawn observation-noise realization eps_u)
    s_cond = pcd.transform(np.zeros(n))
    field1 = pcd.backtransform(s_cond)
    field2 = pcd.backtransform(s_cond)
    np.testing.assert_allclose(field1, field2)


def test_GDPCS_random_state_as_generator() -> None:
    """`random_state` can be an already-built `np.random.Generator`, not
    just an int/None (which `check_random_state` turns into a legacy
    `np.random.RandomState`). `GDPCS` only ever calls `.normal()` /
    `.standard_normal()` on it, which both classes support identically, so
    both must work."""
    ne = 6
    cov = _get_dense_cov()
    n = cov.shape[0]
    H = covmats.make_point_observation_operator([0, 3], n=n)
    pcd = dminv.GDPCS(
        ne,
        cov,
        H,
        np.array([0.1, -0.2]),
        0.05,
        estimated_mean=0.0,
        random_state=np.random.default_rng(42),
    )
    field = pcd.backtransform(pcd.transform(np.zeros(n)))
    assert field.shape == (n,)


def test_GDPCS_colorize_adjoint_not_implemented(monkeypatch) -> None:
    ne = 6
    cov = _get_dense_cov()
    n = cov.shape[0]
    H = covmats.make_point_observation_operator([0, 2], n=n)
    pcd = dminv.GDPCS(
        ne, cov, H, np.array([0.1, 0.2]), 0.1, estimated_mean=0.0, random_state=4
    )
    s_cond = pcd.transform(np.zeros(n))
    monkeypatch.delattr(covmats.CovarianceMatrix, "colorize_adjoint")
    with pytest.raises(
        NotImplementedError,
        match=r"GDPCS\._dbacktransform_vec requires `cov\.colorize_adjoint`",
    ):
        pcd.dbacktransform_vec(s_cond, np.ones(n))


@pytest.mark.parametrize("via_gradient", [False, True])
def test_GDPCS_cg_not_converged(monkeypatch, via_gradient: bool) -> None:
    """`_solve_data_space_system` is shared by `_backtransform` and
    `_dbacktransform_vec`; a non-converging CG solve must surface as a
    `RuntimeError` from either call site."""
    ne = 6
    cov = _get_dense_cov()
    n = cov.shape[0]
    H = covmats.make_point_observation_operator([0, 2], n=n)
    pcd = dminv.GDPCS(
        ne, cov, H, np.array([0.1, 0.2]), 0.1, estimated_mean=0.0, random_state=5
    )
    s_cond = pcd.transform(np.zeros(n))

    def _fake_cg(A, b, **kwargs):
        return np.zeros_like(b), 1  # info != 0 -> did not converge

    monkeypatch.setattr(sp.sparse.linalg, "cg", _fake_cg)
    with pytest.raises(
        RuntimeError,
        match="the conjugate-gradient solve for the data-space system did not converge",
    ):
        if via_gradient:
            pcd.dbacktransform_vec(s_cond, np.ones(n))
        else:
            pcd.backtransform(s_cond)


def test_gradient_scaler_config_default_pcd_change_eval() -> None:
    gsc = GradientScalerConfig(max_change_target=1.0)
    assert isinstance(gsc.pcd_change_eval, dminv.NoTransform)


def test_is_picklable() -> None:
    from pyrtid.utils.preconditioner import is_picklable

    assert is_picklable(dminv.NoTransform()) is True
    # a generator cannot be pickled (raises TypeError, caught by is_picklable)
    assert is_picklable((x for x in range(3))) is False


def test_get_max_update_without_gsc() -> None:
    pcd = dminv.LinearTransform(slope=50.0, y_intercept=0.0)
    s_nc = np.ones(10) * 1e-4
    grad_nc = -np.ones_like(s_nc) * 600.0
    # gsc=None -> uses the raw (unscaled) difference instead of pcd_change_eval
    update = get_max_update(1.0, pcd, s_nc, grad_nc)
    assert update > 0


def test_get_factor_enforcing_grad_inf_norm_multi_round_and_sequential() -> None:
    pcd = dminv.LinearTransform(slope=50.0, y_intercept=0.0)
    s_nc = np.ones(10) * 1e-4
    grad_nc = -np.ones_like(s_nc) * 600.0

    # very tight rtol forces several refinement rounds, and max_workers=1
    # forces the sequential (non-multiprocessing) code path
    gsc = GradientScalerConfig(
        max_workers=1,
        max_change_target=0.8,
        pcd_change_eval=dminv.NoTransform(),
        n_samples_in_first_round=10,
        rtol=1e-5,
        lb=1e-10,
        ub=1e10,
    )
    scaling_factor = get_factor_enforcing_grad_inf_norm(
        s_nc, grad_nc, pcd, gsc, logger=scaler_log
    )
    new_max_update = get_max_update(scaling_factor, pcd, s_nc, grad_nc, gsc)
    np.testing.assert_allclose(new_max_update, gsc.max_change_target, rtol=1e-4)


def test_get_factor_enforcing_grad_inf_norm_does_not_converge() -> None:
    pcd = dminv.LinearTransform(slope=50.0, y_intercept=0.0)
    s_nc = np.ones(10) * 1e-4
    grad_nc = -np.ones_like(s_nc) * 600.0

    # a negative target is unreachable (updates are non-negative norms) ->
    # exhausts the 5 rounds and falls back to a scaling factor of 1.0
    gsc = GradientScalerConfig(
        max_workers=10,
        max_change_target=-1.0,
        n_samples_in_first_round=10,
        rtol=1e-2,
        lb=1e-10,
        ub=1e10,
    )
    scaling_factor = get_factor_enforcing_grad_inf_norm(
        s_nc, grad_nc, pcd, gsc, logger=scaler_log
    )
    assert scaling_factor == 1.0


def test_get_factor_enforcing_grad_inf_norm_n_samples_already_high() -> None:
    pcd = dminv.LinearTransform(slope=50.0, y_intercept=0.0)
    s_nc = np.ones(10) * 1e-4
    grad_nc = -np.ones_like(s_nc) * 600.0

    # n_samples_in_first_round >= 50 -> used as-is (no bump-to-50 branch)
    gsc = GradientScalerConfig(
        max_workers=10,
        max_change_target=0.8,
        pcd_change_eval=dminv.NoTransform(),
        n_samples_in_first_round=60,
        rtol=1e-2,
        lb=1e-10,
        ub=1e10,
    )
    scaling_factor = get_factor_enforcing_grad_inf_norm(
        s_nc, grad_nc, pcd, gsc, logger=scaler_log
    )
    new_max_update = get_max_update(scaling_factor, pcd, s_nc, grad_nc, gsc)
    np.testing.assert_allclose(new_max_update, gsc.max_change_target, rtol=gsc.rtol)


def test_preconditioner_out_of_bounds() -> None:
    pcd = dminv.SqrtTransform()  # LBOUND_RAW = 0.0, UBOUND_RAW = +inf
    with pytest.raises(ValueError, match="do not match with the"):
        pcd.transform(np.array([-1.0, 2.0]))

    pcd2 = dminv.SigmoidRescalerBounded(1e-9, 1e-4, rate=1.0, is_log10=True)
    with pytest.raises(ValueError, match="do not match with the"):
        pcd2.backtransform(np.array([-50.0, 50.0]))


def test_sub_selector_dbacktransform_inv_vec() -> None:
    grid = RectilinearGrid(nx=5, ny=2, dx=1.0, dy=1.0)
    pcd = dminv.SubSelector([1, 2, 6], grid)
    s_cond = pcd.transform(np.arange(grid.n_grid_cells, dtype=np.float64))
    out = pcd.dbacktransform_inv_vec(s_cond, np.array([10.0, 20.0, 30.0]))
    expected = np.zeros(grid.n_grid_cells)
    expected[[1, 2, 6]] = [10.0, 20.0, 30.0]
    np.testing.assert_array_equal(out, expected)


def test_slicer() -> None:
    grid = RectilinearGrid(nx=4, ny=3, dx=1.0, dy=1.0)
    pcd = dminv.Slicer(grid, span=(slice(0, 2), slice(None)))
    field = np.arange(grid.n_grid_cells, dtype=np.float64)
    s_cond = pcd.transform(field)
    assert s_cond.size == 2 * 3


def test_boundsclipper_transform_validation() -> None:
    pcd = dminv.BoundsClipper(np.ones(5) * -1.0, np.ones(5) * 5.0)
    with pytest.raises(ValueError, match="values for which s_raw < lbound!"):
        pcd.transform(np.array([-2.0, 0.0, 1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="values for which s_raw > ubound!"):
        pcd.transform(np.array([-1.0, 0.0, 1.0, 2.0, 6.0]))

    # in-bounds values pass through untouched
    in_bounds = np.array([-1.0, 0.0, 1.0, 2.0, 5.0])
    np.testing.assert_array_equal(pcd.transform(in_bounds), in_bounds)

    # trivial passthrough methods
    np.testing.assert_array_equal(pcd.dtransform_vec(in_bounds, np.ones(5)), np.ones(5))
    np.testing.assert_array_equal(
        pcd.dbacktransform_inv_vec(in_bounds, np.ones(5)), np.ones(5)
    )
    bounds = np.array([[-1.0, 5.0]] * 5)
    np.testing.assert_array_equal(pcd.transform_bounds(bounds), bounds)


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
