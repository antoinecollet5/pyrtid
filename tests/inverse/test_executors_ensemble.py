"""Tests of the FSM based executors: ES-MDA (3 flavours), PCGA and the FSM API."""

from typing import Any

import covmats
import numpy as np
import pytest
from pyrtid.inverse.executors import (
    ESMDADMCInversionExecutor,
    ESMDADMCSolverConfig,
    ESMDAInversionExecutor,
    ESMDARSInversionExecutor,
    ESMDARSSolverConfig,
    ESMDASolverConfig,
    PCGAInversionExecutor,
    PCGASolverConfig,
)
from pyrtid.inverse.executors.base import DataModel
from tests.executor_helpers import tiny_inverse_model

N_ENS = 4


def make_ensemble(n: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.log(1e-4) + 0.3 * rng.standard_normal((n, N_ENS))


def make_esmda(cls, cfg_cls, names=("K",), **cfg):
    model, inv = tiny_inverse_model(names)
    n = model.grid.n_grid_cells * len(names)
    ex = cls(model, inv, cfg_cls(**cfg), s_init=make_ensemble(n))
    return ex, model, inv


def test_cov_obs_as_covmats() -> None:
    diag = DataModel(np.zeros(3), np.zeros(5), np.array([1.0, 2.0, 3.0]))
    cov = diag.get_cov_obs_as_covmats()
    assert isinstance(cov, covmats.CovViaDiagonal)
    full = np.array([[2.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 1.0]])
    cov = DataModel(np.zeros(3), np.zeros(5), full).get_cov_obs_as_covmats()
    assert isinstance(cov, covmats.CovViaCholesky)
    assert np.allclose(cov.to_dense(), full) if hasattr(cov, "to_dense") else True


@pytest.mark.parametrize("is_parallel", [False, True])
def test_esmda_run(is_parallel) -> None:
    ex, model, inv = make_esmda(
        ESMDAInversionExecutor,
        ESMDASolverConfig,
        n_assimilations=2,
        save_ensembles_history=True,
        is_parallel=is_parallel,
        max_workers=2,
    )
    assert ex._get_solver_name() == "ESMDA"
    assert ex.get_display_dict() == {"Number of realizations": N_ENS}
    ex.run()
    # initial ensemble + one posterior per assimilation
    assert len(ex.s_history) == 3
    assert all(h.shape == (10, N_ENS) for h in ex.s_history)
    assert not np.allclose(ex.s_history[0], ex.s_history[-1])
    # one forward run per member and per assimilation (+ the forecast)
    assert len(inv.loss_history) >= 2 * N_ENS
    # the model holds the ensemble mean (back-transformed)
    mean = np.mean(ex.solver.m_posterior, axis=1)
    assert np.allclose(model.fl_model.permeability.ravel("F"), np.exp(mean))
    assert len(inv.parameters_to_adjust[0].archived_values) >= 1


def test_esmda_without_history() -> None:
    ex, _, _ = make_esmda(ESMDAInversionExecutor, ESMDASolverConfig, n_assimilations=2)
    ex.run()
    assert ex.s_history == []
    assert ex.solver.m_posterior.shape == (10, N_ENS)


def test_esmda_two_parameters() -> None:
    ex, model, _ = make_esmda(
        ESMDAInversionExecutor,
        ESMDASolverConfig,
        names=("K", "W"),
        n_assimilations=2,
    )
    assert ex.data_model.s_dim == 20
    ex.run()
    assert ex.solver.m_posterior.shape == (20, N_ENS)
    assert np.all(model.tr_model.porosity <= 0.9)


def test_esmda_rs_run() -> None:
    ex, _, inv = make_esmda(
        ESMDARSInversionExecutor,
        ESMDARSSolverConfig,
        std_s_prior=np.full(10, 0.3),
        save_ensembles_history=True,
    )
    assert ex._get_solver_name() == "ESMDA-RS"
    ex.run()
    assert len(ex.s_history) >= 2
    assert ex.solver.m_posterior.shape == (10, N_ENS)
    assert len(inv.loss_history) >= N_ENS


def test_esmda_dmc_run() -> None:
    ex, _, inv = make_esmda(
        ESMDADMCInversionExecutor, ESMDADMCSolverConfig, save_ensembles_history=True
    )
    assert ex._get_solver_name() == "ESMDA-DMC"
    ex.run()
    assert len(ex.s_history) >= 2
    assert len(inv.loss_history) >= N_ENS


# --------------------------------------------------------------------------- #
# FSM API
# --------------------------------------------------------------------------- #


@pytest.fixture
def fsm_executor():
    ex, _, inv = make_esmda(
        ESMDAInversionExecutor, ESMDASolverConfig, n_assimilations=2
    )
    return ex, inv


def test_run_fsm_matches_finite_differences(fsm_executor) -> None:
    ex, inv = fsm_executor
    s = ex.data_model.s_init[:, 0].copy()
    vecs = np.zeros((10, 2))
    vecs[3, 0] = 1.0
    vecs[:, 1] = 1.0
    d_pred, jacvecs = ex.run_fsm(s, vecs, 1, is_verbose=True)
    assert d_pred.shape == (ex.data_model.d_dim,)
    assert jacvecs.shape == (ex.data_model.d_dim, 2)
    assert len(inv.list_d_pred) == 1
    eps = 1e-5
    d_plus = ex.run_fsm(s + eps * vecs[:, 1], vecs, 2, is_save_state=False)[0]
    d_minus = ex.run_fsm(s - eps * vecs[:, 1], vecs, 3, is_save_state=False)[0]
    fd = (d_plus - d_minus) / (2 * eps)
    scale = np.abs(jacvecs[:, 1]).max()
    assert np.allclose(jacvecs[:, 1], fd, atol=1e-3 * scale)
    assert len(inv.list_d_pred) == 1  # not saved
    with pytest.raises(AssertionError):
        ex.run_fsm(s, np.zeros((3, 1)), 4)


def test_run_fsm_pre_run_transformation() -> None:
    called = []
    model, inv = tiny_inverse_model(("K",))
    ex = ESMDAInversionExecutor(
        model,
        inv,
        ESMDASolverConfig(n_assimilations=2),
        pre_run_transformation=lambda m: called.append(m),
        s_init=make_ensemble(),
    )
    ex.run_fsm(ex.data_model.s_init[:, 0], np.ones((10, 1)), 1)
    assert called == [model]


def test_fsm_checks(fsm_executor) -> None:
    ex, _ = fsm_executor
    ex.run_fsm(ex.data_model.s_init[:, 0], np.ones((10, 1)), 1)
    assert ex.is_fsm_jacvec_correct(np.ones((10, 2)), max_workers=1, is_verbose=True)
    assert ex.is_fsm_jacobian_correct(max_workers=1)
    with pytest.raises(AssertionError):
        ex.is_fsm_jacvec_correct(np.ones((3, 1)))


# --------------------------------------------------------------------------- #
# PCGA
# --------------------------------------------------------------------------- #


def make_pcga(**cfg):
    model, inv = tiny_inverse_model(("K",))
    n = model.grid.n_grid_cells
    q = covmats.eigen_factorize_cov_mat(
        covmats.CovViaDiagonal(np.full(n, 0.1)), n_pc=3, random_state=0
    )
    cfg: dict[str, Any] = {
        "eig_cov": q,
        "drift": covmats.ConstantDriftMatrix(n),
        "prior_s_var": 0.1,
        "is_save_jac": True,
        "maxiter": 2,
        **cfg,
    }
    return PCGAInversionExecutor(model, inv, PCGASolverConfig(**cfg)), model, inv


def test_pcga_requires_eig_cov() -> None:
    model, inv = tiny_inverse_model(("K",))
    with pytest.raises(ValueError, match="eig_cov"):
        PCGAInversionExecutor(model, inv, PCGASolverConfig())


@pytest.mark.parametrize("cfg", [{}, {"is_line_search": True}])
def test_pcga_run(cfg) -> None:
    ex, model, inv = make_pcga(**cfg)
    assert ex._get_solver_name() == "PCGA"
    s_hat, simul_obs, post_diagv, iter_best = ex.run()
    n = model.grid.n_grid_cells
    assert np.shape(s_hat) == (n, 1) or np.shape(s_hat) == (n,)
    assert np.size(simul_obs) == ex.data_model.d_dim
    assert np.size(post_diagv) == n
    assert iter_best >= 0
    assert len(inv.loss_history) >= 2
    assert len(inv.parameters_to_adjust[0].archived_values) >= 1
    assert np.allclose(
        np.log(model.fl_model.permeability.ravel("F")),
        np.clip(np.ravel(s_hat), np.log(1e-9), 0.0),
    )


def test_pcga_config_iter() -> None:
    cfg = PCGASolverConfig(maxiter=3)
    assert 3 in list(cfg)
