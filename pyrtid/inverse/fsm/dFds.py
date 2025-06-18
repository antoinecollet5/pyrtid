"""
Provides the derivatives of the equations F with respect to the parameters s.

This is required to build the right hand side in the forward sensitivity method.
"""

from __future__ import annotations

import numpy as np

from pyrtid.forward.flow_solver import GRAVITY, WATER_DENSITY, get_rhomean
from pyrtid.forward.models import FlowRegime, ForwardModel, VerticalAxis
from pyrtid.utils import (
    NDArrayFloat,
    dxi_harmonic_mean,
    get_extended_grid_shape,
    harmonic_mean,
    np_cache,
)


def dFhdKv(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat:
    """
    Compute the gradient with respect to the permeability using head observations.

    Parameters
    ----------
    fwd_model : ForwardModel
        The forward model which contains all forward variables and parameters.
    time_index: int

    vecs: NDArrayFloat

    Returns
    -------
    NDArrayFloat
        Gradient with respect to the permeability using head observations.
    """
    # Case one = density flow = no impact of the permeability
    if fwd_model.fl_model.is_gravity:
        return np.zeros_like(fwd_model.fl_model.lhead[0])

    # Case two = density flow = impact of the permeability
    shape = vecs.shape
    permeability = fwd_model.fl_model.permeability
    crank_flow: float = fwd_model.fl_model.crank_nicolson

    head = fwd_model.fl_model.lhead[time_index]
    if time_index != 0:
        head_prev = fwd_model.fl_model.lhead[time_index - 1]
    else:
        # make a reference just for linting
        # it won't be used anyway
        head_prev = head

    out = np.zeros(shape)

    for n, axis in zip(fwd_model.grid.shape, (0, 1, 2)):
        if n < 2:
            continue

        fwd_slicer = fwd_model.grid.get_slicer_forward(axis)
        bwd_slicer = fwd_model.grid.get_slicer_backward(axis)

        dKijdKxv = (
            dxi_harmonic_mean(permeability[bwd_slicer], permeability[fwd_slicer])[
                :, :, np.newaxis
            ]
            * vecs[*bwd_slicer, :]
            + dxi_harmonic_mean(permeability[fwd_slicer], permeability[bwd_slicer])[
                :, :, np.newaxis
            ]
            * vecs[*fwd_slicer, :]
        )
        tmp = fwd_model.grid.gamma_ij(axis) / fwd_model.grid.pipj(axis)

        # For all n != 0
        if time_index != 0:
            lhs = (
                (
                    crank_flow * (head[bwd_slicer] - head[fwd_slicer])
                    + (1.0 - crank_flow)
                    * (head_prev[bwd_slicer] - head_prev[fwd_slicer])
                )
                * tmp
                / fwd_model.grid.grid_cell_volume
            )
            # Forward
            out[*fwd_slicer, :] -= (
                lhs / fwd_model.fl_model.storage_coefficient[fwd_slicer]
            )[:, :, :, np.newaxis] * dKijdKxv
            # Backward scheme
            out[*bwd_slicer, :] += (
                lhs / fwd_model.fl_model.storage_coefficient[bwd_slicer]
            )[:, :, :, :, np.newaxis] * dKijdKxv

        # Handle the stationary case for n == 0
        elif fwd_model.fl_model.regime == FlowRegime.STATIONARY:
            # Forward
            lhs = (
                (head[bwd_slicer] - head[fwd_slicer])[:, :, :, np.newaxis]
                * tmp
                / fwd_model.grid.grid_cell_volume
                * dKijdKxv
            )
            out[*fwd_slicer, :] -= lhs
            out[*bwd_slicer, :] += lhs

    out[
        fwd_model.fl_model.cst_head_indices[0],
        fwd_model.fl_model.cst_head_indices[1],
        fwd_model.fl_model.cst_head_indices[2],
        :,
    ] = 0

    return out


def dFhdSsv(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat:
    # Case one = density flow = no impact of the storage coefficient
    if fwd_model.fl_model.is_gravity:
        return np.zeros(
            (fwd_model.fl_model.lhead[0].size, vecs.shape[1]), dtype=np.float64
        )

    # Case two = stationary flow = impact of the storage coefficient
    shape = vecs.shape
    permeability = fwd_model.fl_model.permeability
    crank_flow: float = fwd_model.fl_model.crank_nicolson

    head = fwd_model.fl_model.lhead[time_index]
    if time_index != 0:
        head_prev = fwd_model.fl_model.lhead[time_index - 1]
    else:
        # make a reference just for linting
        # it won't be used anyway
        head_prev = head

    out = np.zeros(shape)

    dinvSsV = (
        -1.0 / (fwd_model.fl_model.storage_coefficient**2)[:, :, :, np.newaxis] * vecs
    )

    for n, axis in zip(fwd_model.grid.shape, (0, 1, 2)):
        if n < 2:
            continue

        fwd_slicer = fwd_model.grid.get_slicer_forward(axis)
        bwd_slicer = fwd_model.grid.get_slicer_backward(axis)

        Kij = harmonic_mean(permeability[bwd_slicer], permeability[fwd_slicer])
        tmp = fwd_model.grid.gamma_ij(axis) / fwd_model.grid.pipj(axis)

        # For all n != 0
        if time_index != 0:
            lhs = (
                (
                    crank_flow * (head[bwd_slicer] - head[fwd_slicer])
                    + (1.0 - crank_flow)
                    * (head_prev[bwd_slicer] - head_prev[fwd_slicer])
                )
                * tmp
                * Kij
                / fwd_model.grid.grid_cell_volume
            )
            # Forward
            out[*fwd_slicer, :] -= lhs[:, :, :, np.newaxis] * dinvSsV[*fwd_slicer, :]
            # Backward scheme
            out[*bwd_slicer, :] += lhs[:, :, :, np.newaxis] * dinvSsV[*bwd_slicer, :]

    out -= (
        fwd_model.fl_model.crank_nicolson * fwd_model.fl_model.lunitflow[time_index]
        + (1.0 - fwd_model.fl_model.crank_nicolson)
        * fwd_model.fl_model.lunitflow[time_index - 1]
    )[:, :, :, np.newaxis] * dinvSsV

    out[
        fwd_model.fl_model.cst_head_indices[0],
        fwd_model.fl_model.cst_head_indices[1],
        fwd_model.fl_model.cst_head_indices[2],
    ] = 0

    return out


def dFpdKv(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat:
    # Case one = saturated flow = no impact of K (permeability) on Fp
    if not fwd_model.fl_model.is_gravity:
        return np.zeros(
            (fwd_model.fl_model.lpressure[0].size, vecs.shape[1]), dtype=np.float64
        )
    # Case two = density flow = impact of the permeability on Fp
    shape = vecs.shape
    permeability = fwd_model.fl_model.permeability
    crank_flow: float = fwd_model.fl_model.crank_nicolson

    pressure = fwd_model.fl_model.lpressure[time_index]
    if time_index != 0:
        pressure_prev = fwd_model.fl_model.lpressure[time_index - 1]
    else:
        # make a reference just for linting
        # it won't be used anyway
        pressure_prev = pressure

    out = np.zeros(shape)

    for n, axis in zip(fwd_model.grid.shape, (0, 1, 2)):
        if n < 2:
            continue

        fwd_slicer = fwd_model.grid.get_slicer_forward(axis)
        bwd_slicer = fwd_model.grid.get_slicer_backward(axis)

        dKijdKxv = (
            dxi_harmonic_mean(permeability[bwd_slicer], permeability[fwd_slicer])[
                :, :, :, np.newaxis
            ]
            * vecs[*bwd_slicer, :]
            + dxi_harmonic_mean(permeability[fwd_slicer], permeability[bwd_slicer])[
                :, :, :, np.newaxis
            ]
            * vecs[*fwd_slicer, :]
        )

        # For all n != 0
        if time_index != 0:
            rhoij = get_rhomean(
                fwd_model.grid,
                fwd_model.tr_model,
                axis=axis,
                time_index=time_index - 1,
                is_flatten=False,
            )[fwd_slicer]
            if (
                (fwd_model.fl_model.vertical_axis == VerticalAxis.X and axis == 0)
                or (fwd_model.fl_model.vertical_axis == VerticalAxis.Y and axis == 1)
                or (fwd_model.fl_model.vertical_axis == VerticalAxis.Z and axis == 2)
            ):
                rhoijg = rhoij * GRAVITY
            else:
                rhoijg = 0.0
            lhs = (
                (
                    (
                        crank_flow * (pressure[bwd_slicer] - pressure[fwd_slicer])
                        + (1.0 - crank_flow)
                        * (pressure_prev[bwd_slicer] - pressure_prev[fwd_slicer])
                    )
                    / fwd_model.grid.pipj(axis)
                    + rhoijg
                )
                * fwd_model.grid.gamma_ij(axis)
                * rhoij
                / WATER_DENSITY
                / fwd_model.grid.grid_cell_volume
            )
            # Forward
            out[*fwd_slicer, :] -= (
                lhs / fwd_model.fl_model.storage_coefficient[fwd_slicer]
            )[:, :, :, np.newaxis] * dKijdKxv
            # Backward scheme
            out[*bwd_slicer, :] += (
                lhs / fwd_model.fl_model.storage_coefficient[bwd_slicer]
            )[:, :, :, np.newaxis] * dKijdKxv

        # Handle the stationary case for n == 0
        elif fwd_model.fl_model.regime == FlowRegime.STATIONARY:
            z = fwd_model.fl_model._get_mesh_center_vertical_pos()[:, :, :, np.newaxis]
            lhs = (
                (
                    (pressure[bwd_slicer] - pressure[fwd_slicer])[:, :, :, np.newaxis]
                    / GRAVITY
                    / WATER_DENSITY
                    - z[fwd_slicer]
                    + z[bwd_slicer]
                )
                * fwd_model.grid.gamma_ij(axis)
                / fwd_model.grid.pipj(axis)
                / fwd_model.grid.grid_cell_volume
            ) * dKijdKxv

            out[*fwd_slicer, :] -= lhs
            out[*bwd_slicer, :] += lhs

    out[
        fwd_model.fl_model.cst_head_indices[0],
        fwd_model.fl_model.cst_head_indices[1],
        fwd_model.fl_model.cst_head_indices[2],
    ] = 0

    return out


def dFpdSsv(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat:
    # Case one = saturated flow = no impact of Ss on Fp
    if not fwd_model.fl_model.is_gravity:
        return np.zeros(
            (fwd_model.fl_model.lpressure[0].size, vecs.shape[1]), dtype=np.float64
        )
    # Case two = density flow = impact of the storage coefficient on Fp
    shape = vecs.shape
    permeability = fwd_model.fl_model.permeability
    crank_flow: float = fwd_model.fl_model.crank_nicolson

    pressure = fwd_model.fl_model.lpressure[time_index]
    if time_index != 0:
        pressure_prev = fwd_model.fl_model.lpressure[time_index - 1]
    else:
        # make a reference just for linting
        # it won't be used anyway
        pressure_prev = pressure

    out = np.zeros(shape)

    dinvSsV = (
        -1.0 / (fwd_model.fl_model.storage_coefficient**2)[:, :, :, np.newaxis] * vecs
    )

    for n, axis in zip(fwd_model.grid.shape, (0, 1, 2)):
        if n < 2:
            continue

        fwd_slicer = fwd_model.grid.get_slicer_forward(axis)
        bwd_slicer = fwd_model.grid.get_slicer_backward(axis)

        Kij = harmonic_mean(permeability[bwd_slicer], permeability[fwd_slicer])
        rhoij = get_rhomean(
            fwd_model.grid,
            fwd_model.tr_model,
            axis=axis,
            time_index=time_index - 1,
            is_flatten=False,
        )[fwd_slicer]
        # For all n != 0
        if time_index != 0:
            if (
                (fwd_model.fl_model.vertical_axis == VerticalAxis.X and axis == 0)
                or (fwd_model.fl_model.vertical_axis == VerticalAxis.Y and axis == 1)
                or (fwd_model.fl_model.vertical_axis == VerticalAxis.Z and axis == 2)
            ):
                rhoijg = rhoij * GRAVITY
            else:
                rhoijg = 0.0
            lhs = (
                (
                    (
                        crank_flow * (pressure[bwd_slicer] - pressure[fwd_slicer])
                        + (1.0 - crank_flow)
                        * (pressure_prev[bwd_slicer] - pressure_prev[fwd_slicer])
                    )
                    / fwd_model.grid.pipj(axis)
                    + rhoijg
                )
                * fwd_model.grid.gamma_ij(axis)
                * rhoij
                * Kij
                / WATER_DENSITY
                / fwd_model.grid.grid_cell_volume
            )
            # Forward
            out[*fwd_slicer, :] -= lhs[:, :, :, np.newaxis] * dinvSsV[*fwd_slicer, :]
            # Backward scheme
            out[*bwd_slicer, :] += lhs[:, :, :, np.newaxis] * dinvSsV[*bwd_slicer, :]

    out -= (
        (
            fwd_model.fl_model.crank_nicolson * fwd_model.fl_model.lunitflow[time_index]
            + (1.0 - fwd_model.fl_model.crank_nicolson)
            * fwd_model.fl_model.lunitflow[time_index - 1]
        )
        * fwd_model.tr_model.ldensity[time_index - 1]
        * GRAVITY
    )[:, :, :, np.newaxis] * dinvSsV

    out[
        fwd_model.fl_model.cst_head_indices[0],
        fwd_model.fl_model.cst_head_indices[1],
        fwd_model.fl_model.cst_head_indices[2],
    ] = 0

    return out


def dFUdKv(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat, axis: int
) -> NDArrayFloat:
    """_summary_

    Parameters
    ----------
    fwd_model : ForwardModel
        _description_
    time_index : int
        _description_
    vecs : NDArrayFloat
        _description_
    axis : int
        _description_

    Returns
    -------
    NDArrayFloat
        _description_
    """
    permeability = fwd_model.fl_model.permeability

    # TODO: update this
    out = np.zeros(
        (*get_extended_grid_shape(fwd_model.grid, axis=axis, extend=1), vecs.shape[-1]),
        dtype=np.float64,
    )

    fwd_slicer = fwd_model.grid.get_slicer_forward(axis)
    bwd_slicer = fwd_model.grid.get_slicer_backward(axis)

    dKijdKxv = (
        dxi_harmonic_mean(permeability[bwd_slicer], permeability[fwd_slicer])[
            :, :, np.newaxis
        ]
        * vecs[*bwd_slicer, :]
        + dxi_harmonic_mean(permeability[fwd_slicer], permeability[bwd_slicer])[
            :, :, :, np.newaxis
        ]
        * vecs[*fwd_slicer, :]
    )

    if fwd_model.fl_model.is_gravity:
        rhomean = get_rhomean(
            fwd_model.grid,
            fwd_model.tr_model,
            axis=axis,
            time_index=time_index - 1,
            is_flatten=False,
        )[:, :-1]
        # TODO review this [:, :-1]

        if (
            (fwd_model.fl_model.vertical_axis == VerticalAxis.X and axis == 0)
            or (fwd_model.fl_model.vertical_axis == VerticalAxis.Y and axis == 1)
            or (fwd_model.fl_model.vertical_axis == VerticalAxis.Z and axis == 2)
        ):
            rho_ij_g = rhomean * GRAVITY
            if time_index == 0:
                rho_ij_g[:, :] = WATER_DENSITY * GRAVITY
        else:
            rho_ij_g = np.zeros_like(rhomean)

        pressure = fwd_model.fl_model.lpressure[time_index]
        out[*bwd_slicer, :] += (
            dKijdKxv
            * (
                (
                    (pressure[bwd_slicer] - pressure[fwd_slicer])
                    / fwd_model.grid.pipj(axis)
                    + rho_ij_g
                )
                / WATER_DENSITY
                / GRAVITY
            )[:, :, :, np.newaxis]
        )
    else:
        head = fwd_model.fl_model.lhead[time_index]
        out[*bwd_slicer, :] += (
            dKijdKxv
            * ((head[bwd_slicer] - head[fwd_slicer]) / fwd_model.grid.pipj(axis))[
                :, :, :, np.newaxis
            ]
        )
    return out


def dFhdhimp(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat:
    """_summary_

    Parameters
    ----------
    fwd_model : ForwardModel
        _description_
    time_index : int
        _description_
    vecs : NDArrayFloat
        _description_

    Returns
    -------
    NDArrayFloat
        _description_
    """
    # no impact in the density case
    if fwd_model.fl_model.is_gravity:
        return np.zeros(
            (fwd_model.fl_model.lhead[0].size, vecs.shape[-1]), dtype=np.float64
        )
    # otherwise, we must take the stationary case into account
    masked_vecs = vecs.copy().reshape(-1, vecs.shape[-1], order="F")
    if not (fwd_model.fl_model.regime == FlowRegime.TRANSIENT and time_index == 0):
        masked_vecs[fwd_model.fl_model.free_head_nn, :] = 0.0
    return fwd_model.fl_model.q_next @ masked_vecs


def dFpdpimp(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat:
    """_summary_

    Parameters
    ----------
    fwd_model : ForwardModel
        _description_
    time_index : int
        _description_
    vecs : NDArrayFloat
        _description_

    Returns
    -------
    NDArrayFloat
        _description_
    """
    # no impact in the density case
    if not fwd_model.fl_model.is_gravity:
        return np.zeros(
            (fwd_model.fl_model.lpressure[0].size, vecs.shape[-1]), dtype=np.float64
        )
    # otherwise, we must take the stationary case into account
    masked_vecs = vecs.copy().reshape(-1, vecs.shape[-1], order="F")
    if not (fwd_model.fl_model.regime == FlowRegime.TRANSIENT and time_index == 0):
        masked_vecs[fwd_model.fl_model.free_head_nn, :] = 0.0
    return fwd_model.fl_model.q_next @ masked_vecs


@np_cache(pos=1)
def dFDdwv(fwd_model: ForwardModel, vecs: NDArrayFloat) -> NDArrayFloat:
    return fwd_model.tr_model.diffusion[:, :, :, np.newaxis] * vecs


@np_cache(pos=1)
def dFDdDv(fwd_model: ForwardModel, vecs: NDArrayFloat) -> NDArrayFloat:
    return fwd_model.tr_model.porosity[:, :, :, np.newaxis] * vecs


def dFDddispv(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat:
    return (
        fwd_model.fl_model.get_u_darcy_norm_sample(time_index)[:, :, :, np.newaxis]
        * vecs
    )


def dFcdwv(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat:
    # no need to take constant conc into account because conc(n+1) - conc(n) is null in
    # such case
    return (
        (fwd_model.tr_model.lmob[time_index] - fwd_model.tr_model.lmob[time_index - 1])
        / fwd_model.time_params.dt
    )[:, :, :, :, np.newaxis] * vecs[np.newaxis, :, :, :, :]


def dFcdcimp(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat:
    if time_index == 0:
        return -vecs
    masked_vecs = vecs.copy().reshape(-1, vecs.shape[-1], order="F")
    masked_vecs[fwd_model.tr_model.free_conc_nn, :] = 0.0
    return fwd_model.tr_model.q_next @ masked_vecs


def dFmdmimp(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat:
    if time_index == 0:
        return -vecs
    return np.zeros_like(vecs)
