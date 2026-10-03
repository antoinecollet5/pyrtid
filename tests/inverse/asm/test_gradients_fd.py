"""Compare the adjoint gradient with finite differences (saturated flow)."""

import numpy as np
import pyrtid.inverse as dminv
import pytest
from tests.helpers import build_model, build_parameters

from .asm_utils import check_gradient, make_observables

SV = dminv.StateVariable
ALL_VARS = [
    (SV.HEAD, None),
    (SV.CONCENTRATION, 0),
    (SV.CONCENTRATION, 1),
    (SV.GRADE, 0),
    (SV.GRADE, 1),
]
ALL_PARAMS = ["K", "SS", "H0", "W", "D", "A", "C0", "C1", "G0", "G1"]
# The first column of the grid has a constant concentration (nodes 0 and 4 for a
# 4x2 grid): the adjoint gradient is not defined there.
CST_CONC_NODES = (0, 4)

# Relative L2 error tolerances for the explicit chemistry (default) and for the
# implicit chemistry (the adjoint is only approximate for some parameters).
TOL_EXPLICIT = {"K": 1e-4, "default": 1e-2}
TOL_IMPLICIT = {
    "K": 1e-3,
    "SS": 1e-3,
    "H0": 1e-3,
    "G0": 1e-2,
    "default": 5.0,  # sanity check only
}


def relative_error(adj: np.ndarray, fd: np.ndarray) -> float:
    return float(np.linalg.norm(adj - fd) / max(np.linalg.norm(fd), 1e-300))


def assert_gradients_close(model, adj, fd, names, tols, cst_conc=False) -> None:
    n = model.grid.n_grid_cells
    keep = np.ones(n, dtype=bool)
    if cst_conc:
        keep[list(CST_CONC_NODES)] = False
    assert np.linalg.norm(fd) > 0.0
    for i, name in enumerate(names):
        sl = slice(i * n, (i + 1) * n)
        a, f = adj[sl][keep], fd[sl][keep]
        if np.linalg.norm(f) == 0.0:  # not sensitive
            assert np.linalg.norm(a) < 1e-8
            continue
        err = relative_error(a, f)
        assert err < tols.get(name, tols["default"]), (name, err)


@pytest.mark.parametrize(
    "regime,explicit,cst_conc",
    [
        ("transient", True, False),
        ("stationary", True, False),
        ("transient", True, True),
        ("transient", False, False),
        ("stationary", False, True),
    ],
)
def test_all_parameters(regime, explicit, cst_conc):
    model = build_model(
        regime=regime,
        nx=4,
        ny=2,
        explicit=explicit,
        cst_conc=cst_conc,
        grade2=0.05,
    )
    obs = make_observables(model, ALL_VARS)
    adj, fd = check_gradient(model, build_parameters(ALL_PARAMS), obs)
    tols = TOL_EXPLICIT if explicit else TOL_IMPLICIT
    assert_gradients_close(model, adj, fd, ALL_PARAMS, tols, cst_conc)
