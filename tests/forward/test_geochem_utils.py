"""Tests of the numerical helpers of the local geochemical solvers."""

import logging

import numpy as np
import pytest
from pyrtid.forward.geochem_utils import (
    backtracking_linesearch,
    get_polish,
    newton,
    solve_with_svd,
    standalone_linesearch,
)


def _quad(x: np.ndarray) -> float:
    return float(0.5 * x @ x)


def test_backtracking_full_step_accepted() -> None:
    x = np.array([3.0, -2.0])
    alpha = backtracking_linesearch(_quad, x, -x, _quad(x), -float(x @ x))
    assert alpha == 1.0


def test_backtracking_shrinks_overshooting_step() -> None:
    x = np.array([3.0, -2.0])
    d = -10.0 * x
    f0, slope = _quad(x), float(x @ d)
    alpha = backtracking_linesearch(_quad, x, d, f0, slope)
    assert alpha is not None
    assert 0.0 < alpha < 1.0
    # Armijo condition
    assert _quad(x + alpha * d) <= f0 + 1e-4 * alpha * slope


def test_backtracking_non_finite_values_are_backtracked() -> None:
    def fun(x: np.ndarray) -> float:
        return _quad(x) if np.linalg.norm(x) < 3.0 else np.inf

    x = np.array([1.0, 0.0])
    d = np.array([-10.0, 0.0])
    alpha = backtracking_linesearch(fun, x, d, fun(x), float(x @ d))
    assert alpha is not None
    assert np.isfinite(fun(x + alpha * d))


def test_backtracking_not_a_descent_direction() -> None:
    x = np.array([1.0, 1.0])
    assert backtracking_linesearch(_quad, x, x, _quad(x), float(x @ x)) is None


def test_backtracking_fails_when_no_decrease_is_found() -> None:
    x = np.array([1.0])
    # the objective never decreases along d, although the slope claims it does
    alpha = backtracking_linesearch(
        lambda y: 10.0 + 0.0 * y[0], x, np.array([-1.0]), 0.5, -1.0, max_iter=5
    )
    assert alpha is None


def test_standalone_linesearch_wolfe_step() -> None:
    pytest.importorskip("lbfgsb")
    x0 = np.array([3.0, -2.0])
    alpha, nfev, ngev, fnew, fold, gnew = standalone_linesearch(
        x0, _quad, lambda x: x.copy(), -x0, iprint=-1
    )
    assert alpha is not None
    assert fold == pytest.approx(_quad(x0))
    assert fnew < fold
    assert fnew == pytest.approx(_quad(x0 - alpha * x0))
    np.testing.assert_allclose(gnew, x0 - alpha * x0)
    assert nfev >= 1
    assert ngev >= 1


def test_standalone_linesearch_with_bounds_and_logger() -> None:
    pytest.importorskip("lbfgsb")
    x0 = np.array([3.0, -2.0])
    alpha, *_ = standalone_linesearch(
        x0,
        _quad,
        lambda x: x.copy(),
        -x0,
        bounds=np.array([[0.0, 5.0], [-5.0, 5.0]]),
        iprint=99,
        logger=logging.getLogger("test"),
    )
    assert alpha == pytest.approx(1.0)


def test_standalone_linesearch_failure_on_ascent_direction() -> None:
    pytest.importorskip("lbfgsb")
    x0 = np.array([3.0, -2.0])
    d = x0.copy()  # ascent direction
    alpha, _, _, fnew, fold, gnew = standalone_linesearch(
        x0, _quad, lambda x: x.copy(), d, iprint=-1
    )
    assert alpha is None
    assert fnew == fold
    np.testing.assert_array_equal(gnew, d)


def test_get_polish() -> None:
    c = np.array([1.0, 2.0, 3.0])
    # small relative steps -> no damping
    np.testing.assert_array_equal(get_polish(0.1 * c, c), np.ones(3))
    # a null component -> relative changes are undefined -> no damping
    np.testing.assert_array_equal(
        get_polish(np.array([1.0, 1.0]), np.array([0.0, 1.0])), np.ones(2)
    )
    # large steps are damped, whatever the sign
    dc = np.array([5.0, -5.0, 0.1])
    pf = get_polish(dc, c)
    assert pf[2] == 1.0
    assert 0.0 < pf[0] < 1.0
    assert 0.0 < pf[1] < 1.0
    # the relative change is bounded by b=3 whatever the size of the step
    big = get_polish(1e6 * c, c)
    assert np.all(np.abs(big * 1e6) < 3.0)


def test_solve_with_svd_matches_dense_solver() -> None:
    rng = np.random.default_rng(0)
    a = rng.random((4, 4)) + np.eye(4)
    b = rng.random(4)
    np.testing.assert_allclose(solve_with_svd(a, b), np.linalg.solve(a, b))
    b2 = rng.random((4, 2))
    np.testing.assert_allclose(solve_with_svd(a, b2), np.linalg.solve(a, b2))


def test_solve_with_svd_singular_and_null_matrices() -> None:
    a = np.array([[1.0, 1.0], [1.0, 1.0]])
    b = np.array([2.0, 2.0])
    x = solve_with_svd(a, b)
    # minimum norm solution
    np.testing.assert_allclose(x, [1.0, 1.0])
    # null matrix -> null solution
    np.testing.assert_array_equal(solve_with_svd(np.zeros((2, 2)), b), np.zeros(2))
    np.testing.assert_array_equal(
        solve_with_svd(np.zeros((2, 2)), np.ones((2, 3))), np.zeros((2, 3))
    )
    with pytest.raises(ValueError, match="positive"):
        solve_with_svd(a, b, atol=-1.0)
    with pytest.raises(ValueError, match="positive"):
        solve_with_svd(a, b, rtol=-1.0)


def test_newton_converges_quadratically_on_scalar_root() -> None:
    def res(x: np.ndarray) -> np.ndarray:
        return x**2 - 2.0

    def inv_jac_res(x: np.ndarray) -> np.ndarray:
        return res(x) / (2.0 * x)

    out = newton(np.array([1.0]), res, inv_jac_res, atol=1e-12)
    assert out.success
    assert out.status == "convergence"
    assert out.x[0] == pytest.approx(np.sqrt(2.0))
    assert out.nit < 10
    assert out.fun <= 1e-12


def test_newton_already_converged_and_damped() -> None:
    def res(x: np.ndarray) -> np.ndarray:
        return x - 1.0

    def inv_jac_res(x: np.ndarray) -> np.ndarray:
        return x - 1.0

    out = newton(np.array([1.0]), res, inv_jac_res, atol=1e-10)
    assert out.nit == 0
    assert out.success
    # a half step line search converges geometrically
    calls = []

    def half(x: np.ndarray, dx: np.ndarray, it: int) -> float:
        calls.append(it)
        return 0.5

    out = newton(np.array([5.0]), res, inv_jac_res, atol=1e-6, linesearch=half)
    assert out.success
    assert calls == list(range(1, out.nit + 1))
    assert out.x[0] == pytest.approx(1.0, abs=1e-5)


def test_newton_max_iter() -> None:
    def res(x: np.ndarray) -> np.ndarray:
        return x - 1.0

    out = newton(
        np.array([100.0]),
        res,
        lambda x: res(x),
        atol=1e-30,
        rtol=0.0,
        linesearch=lambda x, dx, it: 0.1,
        max_iter=3,
    )
    assert not out.success
    assert out.status == "max_iter"
    assert out.nit == 3
    assert "3 iterations" in out.message


def test_newton_per_component_damping() -> None:
    """The line search may return one damping factor per component."""
    target = np.array([1.0, 2.0])

    def res(x: np.ndarray) -> np.ndarray:
        return x - target

    out = newton(
        np.array([5.0, 5.0]),
        res,
        res,
        atol=1e-10,
        linesearch=lambda x, dx, it: np.array([1.0, 0.5]),
    )
    assert out.success
    np.testing.assert_allclose(out.x, target, atol=1e-8)
    # the first component converges in one step, the second needs several
    assert out.nit > 1
