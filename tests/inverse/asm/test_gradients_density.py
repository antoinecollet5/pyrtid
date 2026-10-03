"""Compare the adjoint gradient with finite differences (density driven flow)."""

import numpy as np
import pyrtid.forward as dmfwd
import pyrtid.inverse as dminv
import pytest
from tests.helpers import PARAMETER_FACTORIES

from .asm_utils import build_gravity_model, check_gradient, make_observables

SV = dminv.StateVariable
VARS = [
    (SV.HEAD, None),
    (SV.PRESSURE, None),
    (SV.DENSITY, None),
    (SV.CONCENTRATION, 0),
    (SV.GRADE, 0),
]
FACTORIES = {
    **PARAMETER_FACTORIES,
    "P0": lambda: dminv.AdjustableParameter(
        dminv.ParameterName.INITIAL_PRESSURE, lbounds=-1e7, ubounds=1e7
    ),
}
FLOW_NAMES = ["K", "SS", "P0", "W", "D", "A"]
CASES = [
    (dmfwd.VerticalAxis.Z, (4, 1, 3), "transient", (1, 2, 3, 4)),
    (dmfwd.VerticalAxis.X, (5, 2, 1), "transient", (1, 2, 3, 6, 7)),
    (dmfwd.VerticalAxis.Y, (3, 3, 1), "stationary", (1, 2, 3, 4)),
]


def assert_close(adj, fd, names, n, tol) -> None:
    """Check each parameter (insensitive ones are only checked for being tiny)."""
    scale = np.linalg.norm(fd)
    assert scale > 0.0
    for i, name in enumerate(names):
        a, f = adj[i * n : (i + 1) * n], fd[i * n : (i + 1) * n]
        err = np.linalg.norm(a - f)
        assert err <= tol * max(np.linalg.norm(f), 1e-4 * scale), (name, err)


@pytest.mark.parametrize("axis,shape,regime,nodes", CASES)
def test_flow_and_transport_parameters(axis, shape, regime, nodes):
    model = build_gravity_model(*shape, vertical_axis=axis, regime=regime)
    obs = make_observables(model, VARS, nodes=nodes)
    adj, fd = check_gradient(model, [FACTORIES[n]() for n in FLOW_NAMES], obs)
    assert_close(adj, fd, FLOW_NAMES, model.grid.n_grid_cells, 1e-2)


@pytest.mark.parametrize("axis,shape,regime,nodes", CASES)
def test_initial_species(axis, shape, regime, nodes):
    # The dependency of the flow on the initial concentration (density) is not
    # accounted for, so only observe the species.
    model = build_gravity_model(*shape, vertical_axis=axis, regime=regime)
    obs = make_observables(model, [(SV.CONCENTRATION, 0), (SV.GRADE, 0)], nodes=nodes)
    params = [FACTORIES[n]() for n in ("C0", "G0")]
    adj, fd = check_gradient(model, params, obs)
    assert_close(adj, fd, ("C0", "G0"), model.grid.n_grid_cells, 1e-2)
