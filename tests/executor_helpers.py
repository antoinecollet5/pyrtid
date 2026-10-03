"""Builders to run the inversion executors end to end on a tiny model."""

from __future__ import annotations

import numpy as np
import pyrtid.forward as dmfwd
import pyrtid.inverse as dminv
from pyrtid.inverse.model import InverseModel

from tests.helpers import build_model, build_observables, build_parameters

NX, NY = 5, 2


def tiny_model(**kwargs) -> dmfwd.ForwardModel:
    """Build a 5x2 model."""
    return build_model(nx=NX, ny=NY, **kwargs)


def tiny_observables(
    model: dmfwd.ForwardModel, n_obs_kinds: int = 2
) -> list[dminv.Observable]:
    """Synthetic head (and concentration) observations from a perturbed model."""
    all_obs = build_observables(model, nodes=(1, 4, 7))
    # obs are ordered by state variable: head (3), conc0 (3), conc1 (3), ...
    obs = [o for i, o in enumerate(all_obs) if i // 3 < n_obs_kinds and i % 3 != 1]
    truth = tiny_model()
    truth.fl_model.permeability = truth.fl_model.permeability * 2.0
    truth.tr_model.porosity = truth.tr_model.porosity * 0.8
    dmfwd.ForwardSolver(truth).solve()
    out = []
    for o in obs:
        values = dminv.obs.get_predictions_matching_observations(truth, [o])
        out.append(
            dminv.Observable(
                state_variable=o.state_variable,
                node_indices=o.node_indices,
                times=o.times,
                values=values,
                uncertainties=np.abs(values).max() * 0.05 + 1e-12,
                sp=o.sp,
            )
        )
    obs = out
    return obs


def tiny_inverse_model(
    names: tuple[str, ...] = ("K",), n_obs_kinds: int = 1
) -> tuple[dmfwd.ForwardModel, InverseModel]:
    """Return a forward model and an inverse model adjusting ``names``."""
    model = tiny_model()
    return model, InverseModel(
        build_parameters(list(names)), tiny_observables(model, n_obs_kinds)
    )
