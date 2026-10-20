"""
Check that the solver configurations expose all the arguments of the wrapped libraries.

Two kinds of tests:

- generic ones, comparing the signatures of the library entry points with the fields
  of the configuration dataclasses (a new argument in a library makes them fail until
  it is exposed or explicitly listed as internal),
- forwarding ones, checking that non default values reach the library.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import warnings
from typing import Any

import covmats
import lbfgsb
import numpy as np
import pyesmda
import pypcga
import pytest
import scipy.optimize
import stochopy.optimize as stochopy_optimize
from pyrtid.inverse.executors import (
    ESMDADMCInversionExecutor,
    ESMDADMCSolverConfig,
    ESMDAInversionExecutor,
    ESMDARSInversionExecutor,
    ESMDARSSolverConfig,
    ESMDASolverConfig,
    LBFGSBInversionExecutor,
    LBFGSBSolverConfig,
    PCGAInversionExecutor,
    PCGASolverConfig,
    ScipyInversionExecutor,
    ScipySolverConfig,
    StochopyInversionExecutor,
    StochopySolverConfig,
)
from pyrtid.inverse.executors import base as executors_base
from pyrtid.inverse.executors.stochopy import (
    STOCHOPY_METHOD_OPTIONS,
    STOCHOPY_OPTION_FIELDS,
)
from tests.executor_helpers import tiny_inverse_model

pytestmark = pytest.mark.filterwarnings(
    "ignore:Adjusted parameter:UserWarning", "ignore:divide by zero:RuntimeWarning"
)


def _params(func: Any) -> list[str]:
    return [p for p in inspect.signature(func).parameters if p != "self"]


def _fields(cls: Any) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def _missing(
    func: Any,
    cls: type,
    internal: set[str],
    renamed: dict[str, str] | None = None,
) -> set[str]:
    """Library arguments neither internal nor exposed (possibly renamed)."""
    renamed = renamed or {}
    fields = _fields(cls)
    return {
        p
        for p in _params(func)
        if p not in internal and renamed.get(p, p) not in fields
    }


# --------------------------------------------------------------------------- #
# Generic: library signature vs configuration fields
# --------------------------------------------------------------------------- #

ESMDA_INTERNAL = {
    "obs",
    "m_init",
    "cov_obs",
    "forward_model",
    "forward_model_args",
    "forward_model_kwargs",
    "m_bounds",  # taken from the adjusted parameters
    "seed",  # deprecated in pyesmda in favour of ``random_state``
}
ESMDA_RENAMED = {
    "cov_mm_inflation_factor": "cov_ss_inflation_factor",
    "C_MD_localization": "C_SD_localization",
    "std_m_prior": "std_s_prior",
}


def test_lbfgsb_exposes_all_arguments() -> None:
    internal = {
        "x0",
        "fun",
        "jac",  # always provided by the executor
        "update_fun_def",  # adaptive regularization, handled by the executor
        "bounds",  # taken from the adjusted parameters
        "checkpoint",  # gradient scaling, handled by the executor
        "callback",  # PyRTID stop criterion (see ``stol``)
    }
    assert _missing(lbfgsb.minimize_lbfgsb, LBFGSBSolverConfig, internal) == set()


def test_scipy_exposes_all_arguments() -> None:
    internal = {"fun", "x0", "args", "jac", "bounds"}
    missing = _missing(
        scipy.optimize.minimize,
        ScipySolverConfig,
        internal,
        renamed={"method": "solver_name", "options": "solver_options"},
    )
    assert missing == set()


@pytest.mark.parametrize(
    ("lib", "cfg"),
    [
        (pyesmda.ESMDA, ESMDASolverConfig),
        (pyesmda.ESMDA_RS, ESMDARSSolverConfig),
        (pyesmda.ESMDA_DMC, ESMDADMCSolverConfig),
    ],
)
def test_esmda_exposes_all_arguments(lib, cfg) -> None:
    assert _missing(lib.__init__, cfg, ESMDA_INTERNAL, ESMDA_RENAMED) == set()


def test_pcga_exposes_all_arguments() -> None:
    internal = {"s_init", "obs", "cov_obs", "forward_model"}
    missing = _missing(
        pypcga.PCGA.__init__, PCGASolverConfig, internal, {"Q": "eig_cov"}
    )
    assert missing == set()


ALL_CONFIGS = [
    executors_base.BaseSolverConfig,
    executors_base.AdjointSolverConfig,
    executors_base.FSMSolverConfig,
    LBFGSBSolverConfig,
    ScipySolverConfig,
    StochopySolverConfig,
    ESMDASolverConfig,
    ESMDARSSolverConfig,
    ESMDADMCSolverConfig,
    PCGASolverConfig,
]


@pytest.mark.parametrize("cls", ALL_CONFIGS)
def test_every_field_is_documented(cls) -> None:
    doc = cls.__doc__ or ""
    undocumented = {
        name
        for name in _fields(cls)
        if f"\n    {name}:" not in doc
        and f"\n    {name} :" not in doc
        and f"{name}: " not in doc
    }
    assert undocumented == set()


def test_default_behavior_is_unchanged() -> None:
    """Defaults must keep the historical values of the configurations."""
    cfg = LBFGSBSolverConfig()
    assert (cfg.maxfun, cfg.ftol_linesearch, cfg.xtol_linesearch) == (100, 1e-4, 1e-5)
    assert (cfg.eps, cfg.finite_diff_rel_step) == (1e-8, None)
    assert cfg.is_use_numba_jit and not cfg.is_check_factorizations
    assert cfg.logger is logging.getLogger("L-BFGS-B")
    assert PCGASolverConfig().eps == 1e-8
    assert ESMDASolverConfig().max_failure_fraction == 0.0
    scipy_cfg = ScipySolverConfig()
    assert scipy_cfg.constraints == ()
    assert all(getattr(scipy_cfg, f) is None for f in ("tol", "maxiter", "maxfun"))
    stoch = StochopySolverConfig()
    assert all(getattr(stoch, f) is None for f in STOCHOPY_OPTION_FIELDS)


# --------------------------------------------------------------------------- #
# Forwarding
# --------------------------------------------------------------------------- #


def test_lbfgsb_forwards_all_options(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return scipy.optimize.OptimizeResult(x=kwargs["x0"])

    monkeypatch.setattr("pyrtid.inverse.executors.lbfgsb.minimize_lbfgsb", fake)
    logger = logging.getLogger("custom")
    options: dict[str, Any] = {
        "maxcor": 7,
        "ftarget": 0.5,
        "ftol": 1e-3,
        "gtol": 1e-4,
        "maxiter": 3,
        "maxfun": 11,
        "iprint": 1,
        "maxls": 9,
        "max_steplength": 12.0,
        "ftol_linesearch": 1e-2,
        "gtol_linesearch": 0.5,
        "xtol_linesearch": 0.2,
        "eps": 1e-6,
        "finite_diff_rel_step": 1e-5,
        "eps_SY": 1e-10,
        "is_check_factorizations": True,
        "is_use_numba_jit": False,
        "logger": logger,
    }
    model, inv = tiny_inverse_model(("K",))
    LBFGSBInversionExecutor(model, inv, LBFGSBSolverConfig(**options)).run()
    for key, value in options.items():
        assert captured[key] == value, key
    # the library defaults of all the other arguments are not overridden
    assert captured["callback"] is not None


def test_lbfgsb_gtol_ftarget_accept_callables(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return scipy.optimize.OptimizeResult(x=kwargs["x0"])

    monkeypatch.setattr("pyrtid.inverse.executors.lbfgsb.minimize_lbfgsb", fake)
    model, inv = tiny_inverse_model(("K",))
    cfg = LBFGSBSolverConfig(gtol=lambda: 1e-3, ftarget=lambda: 0.1)
    LBFGSBInversionExecutor(model, inv, cfg).run()
    assert captured["gtol"]() == 1e-3
    assert captured["ftarget"]() == 0.1


def _run_scipy(monkeypatch, **cfg) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake(fun, x0, **kwargs):
        captured.update(kwargs)
        return scipy.optimize.OptimizeResult(x=x0)

    monkeypatch.setattr("pyrtid.inverse.executors.scipy.scipy_minimize", fake)
    model, inv = tiny_inverse_model(("K",))
    ScipyInversionExecutor(model, inv, ScipySolverConfig(**cfg)).run()
    return captured


def test_scipy_forwards_arguments_and_options(monkeypatch) -> None:
    def callback(xk) -> None:
        return None

    def hessp(x, p):
        return p

    constraints = {"type": "ineq", "fun": lambda x: 1.0}
    captured = _run_scipy(
        monkeypatch,
        solver_name="trust-constr",
        hess="2-point",
        hessp=hessp,
        constraints=constraints,
        tol=1e-3,
        callback=callback,
        maxiter=5,
        maxfun=77,
        ftol=1e-4,
        gtol=1e-5,
        xtol=1e-6,
        maxcor=3,
        maxls=4,
        eps=1e-7,
        finite_diff_rel_step=1e-3,
        disp=True,
    )
    assert captured["method"] == "trust-constr"
    assert captured["hess"] == "2-point"
    assert captured["hessp"] is hessp
    assert captured["constraints"] is constraints
    assert captured["tol"] == 1e-3
    assert captured["callback"] is callback
    assert captured["options"] == {
        "maxiter": 5,
        "maxfun": 77,
        "ftol": 1e-4,
        "gtol": 1e-5,
        "xtol": 1e-6,
        "maxcor": 3,
        "maxls": 4,
        "eps": 1e-7,
        "finite_diff_rel_step": 1e-3,
        "disp": True,
    }


def test_scipy_default_forwarding_and_precedence(monkeypatch) -> None:
    captured = _run_scipy(monkeypatch)
    assert captured["options"] == {"maxfun": 15000}
    assert captured["hess"] is None
    assert captured["constraints"] == ()
    assert captured["tol"] is None
    # typed fields win over solver_options
    captured = _run_scipy(
        monkeypatch, solver_options={"maxiter": 1, "maxcor": 2}, maxiter=9
    )
    assert captured["options"] == {"maxiter": 9, "maxcor": 2, "maxfun": 15000}


def test_scipy_typed_maxfun_is_the_rounds_budget(monkeypatch) -> None:
    model, inv = tiny_inverse_model(("K",))
    cfg = ScipySolverConfig(
        maxfun=20,
        solver_options={"maxfun": 1000},
        max_optimization_round_nb=2,
        max_fun_first_round=8,
        max_fun_per_round=6,
    )
    ex = ScipyInversionExecutor(model, inv, cfg)
    assert ex._get_options_dict(cfg, 0, 1)["maxfun"] == 6
    assert ex._get_options_dict(cfg, 18, 2)["maxfun"] == 2


def _run_stochopy(monkeypatch, **cfg) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake(fun, bounds, **kwargs):
        captured.update(kwargs)
        return stochopy_optimize.OptimizeResult(x=kwargs["x0"])

    monkeypatch.setattr("pyrtid.inverse.executors.stochopy.stochopy_minimize", fake)
    model, inv = tiny_inverse_model(("K",))
    StochopyInversionExecutor(model, inv, StochopySolverConfig(**cfg)).run()
    return captured


@pytest.mark.parametrize(
    ("method", "options"),
    [
        (
            "cmaes",
            {"sigma": 0.3, "muperc": 0.4, "constraints": "Penalty"},
        ),
        (
            "cpso",
            {
                "inertia": 0.5,
                "cognitivity": 1.0,
                "sociability": 1.1,
                "competitivity": 0.9,
                "updating": "deferred",
                "constraints": "Penalty",
            },
        ),
        (
            "de",
            {
                "mutation": 0.6,
                "recombination": 0.8,
                "strategy": "rand1bin",
                "updating": "deferred",
            },
        ),
        ("na", {"nrperc": 0.3}),
        ("pso", {"inertia": 0.5, "cognitivity": 1.0, "sociability": 1.1}),
        ("vdcma", {"sigma": 0.2, "muperc": 0.3}),
    ],
)
def test_stochopy_forwards_method_options(monkeypatch, method, options) -> None:
    generic: dict[str, Any] = {
        "maxiter": 4,
        "popsize": 6,
        "xtol": 1e-6,
        "ftol": 1e-7,
        "workers": 1,
        "backend": "thread",
        "return_all": True,
        "verbosity": 0.5,
    }
    captured = _run_stochopy(
        monkeypatch, solver_name=method, seed=3, **generic, **options
    )
    assert captured["method"] == method
    forwarded = captured["options"]
    for key, value in {**generic, **options, "seed": 3}.items():
        assert forwarded[key] == value, key
    assert set(forwarded) <= set(STOCHOPY_METHOD_OPTIONS[method]) | {"seed"}


def test_stochopy_callback_and_defaults(monkeypatch) -> None:
    captured = _run_stochopy(monkeypatch)
    assert captured["options"] == {}
    assert captured["callback"] is None

    def callback(x, state) -> None:
        return None

    captured = _run_stochopy(
        monkeypatch, callback=callback, solver_options={"popsize": 3}, popsize=5
    )
    assert captured["callback"] is callback
    assert captured["options"] == {"popsize": 5}


def test_stochopy_rejects_unsupported_option() -> None:
    model, inv = tiny_inverse_model(("K",))
    cfg = StochopySolverConfig(solver_name="na", sigma=0.1)
    ex = StochopyInversionExecutor(model, inv, cfg)
    with pytest.raises(ValueError, match="'sigma' is not supported"):
        ex._get_options_dict(cfg, 0)


@pytest.mark.parametrize("method", sorted(STOCHOPY_METHOD_OPTIONS))
def test_stochopy_real_run_with_typed_options(method) -> None:
    """All the solvers (also the population based ones) run with typed options."""
    model, inv = tiny_inverse_model(("K",))
    cfg = StochopySolverConfig(
        solver_name=method, maxiter=2, popsize=4, seed=0, xtol=1e-9, ftol=1e-9
    )
    res = StochopyInversionExecutor(model, inv, cfg).run()
    assert res.x.shape == (model.grid.n_grid_cells,)


def test_stochopy_initial_guess() -> None:
    model, inv = tiny_inverse_model(("K",))
    n = model.grid.n_grid_cells
    bounds = np.column_stack([np.full(n, -1.0), np.full(n, 1.0)])
    x0 = np.full(n, 0.25)
    vector = StochopyInversionExecutor(model, inv, StochopySolverConfig())
    assert vector._get_initial_guess(x0, bounds, {}) is x0
    pop_cfg = StochopySolverConfig(solver_name="pso", seed=1)
    ex = StochopyInversionExecutor(model, inv, pop_cfg)
    pop = ex._get_initial_guess(x0, bounds, {"popsize": 5})
    assert pop.shape == (5, n)
    assert np.array_equal(pop[0], x0)
    assert np.min(pop[1:]) >= -1.0 and np.max(pop[1:]) <= 1.0
    assert np.array_equal(pop, ex._get_initial_guess(x0, bounds, {"popsize": 5}))
    assert ex._get_initial_guess(x0, bounds, {}).shape == (10, n)


def _make_ensemble(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.log(1e-4) + 0.3 * rng.standard_normal((n, 4))


class _FakeESMDA:
    """Record the constructor arguments."""

    last: dict[str, Any] = {}

    def __init__(self, *args, **kwargs) -> None:
        type(self).last = {"args": args, **kwargs}


@pytest.mark.parametrize(
    ("ex_cls", "cfg_cls", "lib_name", "extra"),
    [
        (
            ESMDAInversionExecutor,
            ESMDASolverConfig,
            "ESMDA",
            {
                "n_assimilations": 3,
                "cov_obs_inflation_factors": [3.0, 2.0, 1.0],
            },
        ),
        (
            ESMDARSInversionExecutor,
            ESMDARSSolverConfig,
            "ESMDA_RS",
            {"std_s_prior": np.full(10, 0.2)},
        ),
        (ESMDADMCInversionExecutor, ESMDADMCSolverConfig, "ESMDA_DMC", {}),
    ],
)
def test_esmda_forwards_all_arguments(
    monkeypatch, ex_cls, cfg_cls, lib_name, extra
) -> None:
    monkeypatch.setattr(f"pyrtid.inverse.executors.esmda.{lib_name}", _FakeESMDA)
    model, inv = tiny_inverse_model(("K",))
    n = model.grid.n_grid_cells
    logger = logging.getLogger("custom-esmda")
    loc = pyesmda.NoLocalization()
    common: dict[str, Any] = {
        "inversion_type": pyesmda.ESMDAInversionType.CHOLESKY,
        "cov_ss_inflation_factor": 1.1,
        "C_DD_localization": loc,
        "C_SD_localization": loc,
        "save_ensembles_history": True,
        "is_forecast_for_last_assimilation": False,
        "batch_size": 7,
        "is_parallel_analyse_step": False,
        "truncation": 0.8,
        "max_failure_fraction": 0.25,
        "logger": logger,
    }
    ex_cls(model, inv, cfg_cls(**common, **extra), s_init=_make_ensemble(n))
    got = _FakeESMDA.last
    renamed = {"cov_ss_inflation_factor": "cov_mm_inflation_factor"}
    renamed |= {"C_SD_localization": "C_MD_localization"}
    renamed |= {"std_s_prior": "std_m_prior"}
    for key, value in {**common, **extra}.items():
        forwarded = got[renamed.get(key, key)]
        if isinstance(value, np.ndarray):
            assert np.array_equal(forwarded, value), key
        else:
            assert forwarded == value, key
    assert got["m_bounds"].shape == (n, 2)
    assert got["random_state"] is not None


def test_esmda_real_library_accepts_the_new_arguments() -> None:
    model, inv = tiny_inverse_model(("K",))
    n = model.grid.n_grid_cells
    for ex_cls, cfg_cls in (
        (ESMDAInversionExecutor, ESMDASolverConfig),
        (ESMDARSInversionExecutor, ESMDARSSolverConfig),
        (ESMDADMCInversionExecutor, ESMDADMCSolverConfig),
    ):
        cfg = cfg_cls(
            truncation=0.9,
            max_failure_fraction=0.5,
            inversion_type=pyesmda.ESMDAInversionType.NAIVE,
        )
        ex = ex_cls(model, inv, cfg, s_init=_make_ensemble(n))
        assert ex.solver.truncation == 0.9
        assert ex.solver.max_failure_fraction == 0.5
        assert ex.solver.inversion_type == pyesmda.ESMDAInversionType.NAIVE


@pytest.mark.parametrize(
    "ex_cls",
    [ESMDAInversionExecutor, ESMDARSInversionExecutor, ESMDADMCInversionExecutor],
)
def test_esmda_updates_model_with_posterior_mean(ex_cls) -> None:
    """The model receives the mean of the posterior ensemble (axis=1)."""
    cfg_cls = {
        ESMDAInversionExecutor: ESMDASolverConfig,
        ESMDARSInversionExecutor: ESMDARSSolverConfig,
        ESMDADMCInversionExecutor: ESMDADMCSolverConfig,
    }[ex_cls]
    model, inv = tiny_inverse_model(("K",))
    n = model.grid.n_grid_cells
    kwargs = {"n_assimilations": 2} if ex_cls is ESMDAInversionExecutor else {}
    ex = ex_cls(model, inv, cfg_cls(**kwargs), s_init=_make_ensemble(n))
    ex.run()
    posterior = ex.solver.m_posterior
    assert posterior.shape == (n, 4)
    expected = np.exp(posterior.mean(axis=1))
    assert np.allclose(model.fl_model.permeability.ravel("F"), expected)


def test_pcga_forwards_all_arguments(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakePCGA:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("pyrtid.inverse.executors.pcga.PCGA", FakePCGA)
    model, inv = tiny_inverse_model(("K",))
    n = model.grid.n_grid_cells
    q = covmats.eigen_factorize_cov_mat(
        covmats.CovViaDiagonal(np.full(n, 0.1)), n_pc=3, random_state=0
    )
    drift = covmats.ConstantDriftMatrix(n)
    logger = logging.getLogger("custom-pcga")

    def callback(*args, **kwargs) -> None:
        return None

    options: dict[str, Any] = {
        "drift": drift,
        "prior_s_var": 0.3,
        "callback": callback,
        "is_line_search": True,
        "is_lm": True,
        "is_direct_solve": True,
        "is_objfun_exact": True,
        "max_it_lm": 3,
        "max_workers_lm": 5,
        "alphamax_lm": 50.0,
        "lm_smin": -1.0,
        "lm_smax": 1.0,
        "max_it_ls": 4,
        "maxiter": 6,
        "ftarget": 0.1,
        "ftol": 1e-3,
        "restol": 1e-1,
        "logger": logger,
        "is_save_jac": True,
        "eps": 1e-6,
    }
    PCGAInversionExecutor(model, inv, PCGASolverConfig(eig_cov=q, **options))
    assert captured["Q"] is q
    for key, value in options.items():
        assert captured[key] is value or captured[key] == value, key
    assert captured["random_state"] is not None


def test_pcga_eps_is_a_dataclass_field() -> None:
    """``eps`` used to be a bare class attribute (not a field, not iterated)."""
    assert "eps" in _fields(PCGASolverConfig)
    cfg = PCGASolverConfig(eps=1e-3)
    assert cfg.eps == 1e-3
    assert 1e-3 in list(cfg)


def test_rerun_resets_the_history() -> None:
    """``run`` starts from a clean inverse model (the former reset TODO)."""
    model, inv = tiny_inverse_model(("K",))
    ex = ScipyInversionExecutor(
        model, inv, ScipySolverConfig(solver_options={"maxiter": 1})
    )
    ex.run()
    first = inv.nb_f_calls
    ex.inv_model.optimization_round_nb = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ex.run()
    assert inv.nb_f_calls <= first + 1
    assert len(inv.loss_history) == inv.nb_f_calls
