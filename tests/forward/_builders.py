"""Extra builders of forward models for the tests of the forward package."""

from __future__ import annotations

import numpy as np
import pyrtid.forward as dmfwd
from quickpaver import RectilinearGrid

DAY = 3600.0 * 24


def build_3d_model(
    nx: int = 4,
    ny: int = 3,
    nz: int = 3,
    gravity: bool = False,
    regime: str = "transient",
    duration: float = DAY,
    dt: float = 3600.0 * 6,
    all_faces: bool = True,
    faces: list[str] | None = None,
    skip_rt: bool = True,
    explicit: bool = True,
    vertical_axis: dmfwd.VerticalAxis = dmfwd.VerticalAxis.Z,
    **gch_kwargs,
) -> dmfwd.ForwardModel:
    """Build a small 3D model, optionally with constant heads on all the faces."""
    grid = RectilinearGrid(nx=nx, ny=ny, nz=nz, dx=5.0, dy=4.0, dz=2.0)
    time_params = dmfwd.TimeParameters(
        duration=duration, dt_init=dt, dt_max=dt, dt_min=dt
    )
    flow_params = dmfwd.FlowParameters(
        permeability=1e-4,
        storage_coefficient=1e-3,
        regime=(
            dmfwd.FlowRegime.TRANSIENT
            if regime == "transient"
            else dmfwd.FlowRegime.STATIONARY
        ),
        crank_nicolson=0.8,
        is_gravity=gravity,
        vertical_axis=vertical_axis,
    )
    tr_params = dmfwd.TransportParameters(
        porosity=0.25, diffusion=1e-5, dispersivity=0.5, is_skip_rt=skip_rt
    )
    gch_params = dmfwd.GeochemicalParameters(
        conc=1e-8,
        conc2=1e-3,
        grade=1e-1,
        grade2=1e-10,
        kv=-6.9e-6,
        As=13.5,
        Ks=6.3e-4,
        stocoef=1.0,
        use_explicit_formulation=explicit,
        **gch_kwargs,
    )
    model = dmfwd.ForwardModel(
        grid, time_params, flow_params, tr_params=tr_params, gch_params=gch_params
    )
    full = slice(None)
    spans = {
        "west": (slice(0, 1), full, full),
        "east": (slice(nx - 1, nx), full, full),
        "south": (full, slice(0, 1), full),
        "north": (full, slice(ny - 1, ny), full),
        "bottom": (full, full, slice(0, 1)),
        "top": (full, full, slice(nz - 1, nz)),
    }
    heads = {"west": 2.0, "east": 1.0, "south": 1.5, "north": 1.2}
    heads.update({"bottom": 1.6, "top": 1.4})
    keys = list(spans) if all_faces else ["west", "east"]
    if faces is not None:
        keys = faces
    for key in keys:
        model.add_boundary_conditions(
            dmfwd.ConstantHead(span=spans[key], values=heads[key])
        )
    return model


def add_wells(model: dmfwd.ForwardModel) -> None:
    """Add an injection and a pumping well."""
    nx, ny = model.grid.nx, model.grid.ny
    model.add_src_term(
        dmfwd.SourceTerm(
            "inj",
            node_ids=np.array([nx // 2 + nx * (ny // 2)]),
            times=np.array([0.0, 3600 * 12.0]),
            flowrates=np.array([2e-3, 0.0]),
            concentrations=np.array([[1e-3, 5e-4], [0.0, 0.0]]),
        )
    )
