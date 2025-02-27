"""
Provides the derivatives of the equations F with respect to the parameters s.

This is required to build the right hand side in the forward sensitivity method.
"""

from __future__ import annotations

import numpy as np

from pyrtid.forward.flow_solver import GRAVITY, WATER_DENSITY, get_rhomean
from pyrtid.forward.models import FlowRegime, ForwardModel, VerticalAxis
from pyrtid.utils import NDArrayFloat, dxi_harmonic_mean, harmonic_mean, np_cache


def dFhdKv(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat:
    """
    Compute the gradient with respect to the permeability using head observations.

    Parameters
    ----------
    fwd_model : ForwardModel
        The forward model which contains all forward variables and parameters.
    adj_model : AdjointModel
        The adjoint model which contains all adjoint variables and parameters.

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

    # Consider the x axis
    if shape[0] > 1:
        dKijdKxv = (
            dxi_harmonic_mean(permeability[1:, :], permeability[:-1, :])[
                :, :, np.newaxis
            ]
            * vecs[1:, :, :]
            + dxi_harmonic_mean(permeability[:-1, :], permeability[1:, :])[
                :, :, np.newaxis
            ]
            * vecs[:-1, :, :]
        )
        tmp = fwd_model.geometry.gamma_ij_x / fwd_model.geometry.dx

        # For all n != 0
        if time_index != 0:
            lhs = (
                (
                    crank_flow * (head[1:, :] - head[:-1, :])
                    + (1.0 - crank_flow) * (head_prev[1:, :] - head_prev[:-1, :])
                )
                * tmp
                / fwd_model.geometry.grid_cell_volume
            )
            # Forward
            out[:-1, :, :] -= (lhs / fwd_model.fl_model.storage_coefficient[:-1, :])[
                :, :, np.newaxis
            ] * dKijdKxv
            # Backward scheme
            out[1:, :, :] += (lhs / fwd_model.fl_model.storage_coefficient[1:, :])[
                :, :, np.newaxis
            ] * dKijdKxv

        # Handle the stationary case for n == 0
        elif fwd_model.fl_model.regime == FlowRegime.STATIONARY:
            # Forward
            lhs = (
                (head[1:, :] - head[:-1, :])[:, :, np.newaxis]
                * tmp
                / fwd_model.geometry.grid_cell_volume
                * dKijdKxv
            )
            out[:-1, :, :] -= lhs
            out[1:, :, :] += lhs

    # Consider the y axis for 2D cases
    # Consider the x axis
    if shape[1] > 1:
        dKijdKyv = (
            dxi_harmonic_mean(permeability[:, 1:], permeability[:, :-1])[
                :, :, np.newaxis
            ]
            * vecs[:, 1:, :]
            + dxi_harmonic_mean(permeability[:, :-1], permeability[:, 1:])[
                :, :, np.newaxis
            ]
            * vecs[:, :-1, :]
        )
        tmp = fwd_model.geometry.gamma_ij_y / fwd_model.geometry.dy

        # For all n != 0
        if time_index != 0:
            lhs = (
                (
                    crank_flow * (head[:, 1:] - head[:, :-1])
                    + (1.0 - crank_flow) * (head_prev[:, 1:] - head_prev[:, :-1])
                )
                * tmp
                / fwd_model.geometry.grid_cell_volume
            )
            # Forward
            out[:, :-1, :] -= (lhs / fwd_model.fl_model.storage_coefficient[:, :-1])[
                :, :, np.newaxis
            ] * dKijdKyv
            # Backward scheme
            out[:, 1:, :] += (lhs / fwd_model.fl_model.storage_coefficient[:, 1:])[
                :, :, np.newaxis
            ] * dKijdKyv

        # Handle the stationary case for n == 0
        elif fwd_model.fl_model.regime == FlowRegime.STATIONARY:
            # Forward
            lhs = (
                (head[:, 1:] - head[:, :-1])[:, :, np.newaxis]
                * tmp
                / fwd_model.geometry.grid_cell_volume
                * dKijdKyv
            )
            out[:, :-1, :] -= lhs
            out[:, 1:, :] += lhs

    out[
        fwd_model.fl_model.cst_head_indices[0], fwd_model.fl_model.cst_head_indices[1]
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
        -1.0 / (fwd_model.fl_model.storage_coefficient**2)[:, :, np.newaxis] * vecs
    )

    # Consider the x axis
    if shape[0] > 1:
        Kij = harmonic_mean(permeability[1:, :], permeability[:-1, :])
        tmp = fwd_model.geometry.gamma_ij_x / fwd_model.geometry.dx

        # For all n != 0
        if time_index != 0:
            lhs = (
                (
                    crank_flow * (head[1:, :] - head[:-1, :])
                    + (1.0 - crank_flow) * (head_prev[1:, :] - head_prev[:-1, :])
                )
                * tmp
                * Kij
                / fwd_model.geometry.grid_cell_volume
            )
            # Forward
            out[:-1, :, :] -= lhs[:, :, np.newaxis] * dinvSsV[:-1, :, :]
            # Backward scheme
            out[1:, :, :] += lhs[:, :, np.newaxis] * dinvSsV[1:, :, :]

    # Consider the y axis for 2D cases
    # Consider the x axis
    if shape[1] > 1:
        Kij = harmonic_mean(permeability[:, 1:], permeability[:, :-1])
        tmp = fwd_model.geometry.gamma_ij_y / fwd_model.geometry.dy

        # For all n != 0
        if time_index != 0:
            lhs = (
                (
                    crank_flow * (head[:, 1:] - head[:, :-1])
                    + (1.0 - crank_flow) * (head_prev[:, 1:] - head_prev[:, :-1])
                )
                * tmp
                * Kij
                / fwd_model.geometry.grid_cell_volume
            )
            # Forward
            out[:, :-1, :] -= lhs[:, :, np.newaxis] * dinvSsV[:, :-1, :]
            # Backward scheme
            out[:, 1:, :] += lhs[:, :, np.newaxis] * dinvSsV[:, 1:, :]

    out -= (
        fwd_model.fl_model.crank_nicolson * fwd_model.fl_model.lunitflow[time_index]
        + (1.0 - fwd_model.fl_model.crank_nicolson)
        * fwd_model.fl_model.lunitflow[time_index - 1]
    )[:, :, np.newaxis] * dinvSsV

    out[
        fwd_model.fl_model.cst_head_indices[0], fwd_model.fl_model.cst_head_indices[1]
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

    # Consider the x axis
    if shape[0] > 1:
        dKijdKxv = (
            dxi_harmonic_mean(permeability[1:, :], permeability[:-1, :])[
                :, :, np.newaxis
            ]
            * vecs[1:, :, :]
            + dxi_harmonic_mean(permeability[:-1, :], permeability[1:, :])[
                :, :, np.newaxis
            ]
            * vecs[:-1, :, :]
        )

        # For all n != 0
        if time_index != 0:
            rhoij = get_rhomean(
                fwd_model.geometry,
                fwd_model.tr_model,
                axis=0,
                time_index=time_index - 1,
                is_flatten=False,
            )[:-1, :]
            if fwd_model.fl_model.vertical_axis == VerticalAxis.X:
                rhoijg = rhoij * GRAVITY
            else:
                rhoijg = 0.0
            lhs = (
                (
                    (
                        crank_flow * (pressure[1:, :] - pressure[:-1, :])
                        + (1.0 - crank_flow)
                        * (pressure_prev[1:, :] - pressure_prev[:-1, :])
                    )
                    / fwd_model.geometry.dx
                    + rhoijg
                )
                * fwd_model.geometry.gamma_ij_x
                * rhoij
                / WATER_DENSITY
                / fwd_model.geometry.grid_cell_volume
            )
            # Forward
            out[:-1, :, :] -= (lhs / fwd_model.fl_model.storage_coefficient[:-1, :])[
                :, :, np.newaxis
            ] * dKijdKxv
            # Backward scheme
            out[1:, :, :] += (lhs / fwd_model.fl_model.storage_coefficient[1:, :])[
                :, :, np.newaxis
            ] * dKijdKxv

        # Handle the stationary case for n == 0
        elif fwd_model.fl_model.regime == FlowRegime.STATIONARY:
            z = fwd_model.fl_model._get_mesh_center_vertical_pos().T[:, :, np.newaxis]
            lhs = (
                (
                    (pressure[1:, :] - pressure[:-1, :])[:, :, np.newaxis]
                    / GRAVITY
                    / WATER_DENSITY
                    - z[:-1, :]
                    + z[1:, :]
                )
                * fwd_model.geometry.gamma_ij_x
                / fwd_model.geometry.dx
                / fwd_model.geometry.grid_cell_volume
            ) * dKijdKxv

            out[:-1, :, :] -= lhs
            out[1:, :, :] += lhs

    # Consider the y axis for 2D cases
    # Consider the x axis
    if shape[1] > 1:
        dKijdKyv = (
            dxi_harmonic_mean(permeability[:, 1:], permeability[:, :-1])[
                :, :, np.newaxis
            ]
            * vecs[:, 1:, :]
            + dxi_harmonic_mean(permeability[:, :-1], permeability[:, 1:])[
                :, :, np.newaxis
            ]
            * vecs[:, :-1, :]
        )

        # For all n != 0
        if time_index != 0:
            rhoij = get_rhomean(
                fwd_model.geometry,
                fwd_model.tr_model,
                axis=1,
                time_index=time_index - 1,
                is_flatten=False,
            )[:, :-1]

            if fwd_model.fl_model.vertical_axis == VerticalAxis.Y:
                rhoijg = rhoij * GRAVITY
            else:
                rhoijg = 0.0

            lhs = (
                (
                    (
                        crank_flow * (pressure[:, 1:] - pressure[:, :-1])
                        + (1.0 - crank_flow)
                        * (pressure_prev[:, 1:] - pressure_prev[:, :-1])
                    )
                    / fwd_model.geometry.dy
                    + rhoijg
                )
                * fwd_model.geometry.gamma_ij_y
                * rhoij
                / WATER_DENSITY
                / fwd_model.geometry.grid_cell_volume
            )
            # Forward
            out[:, :-1, :] -= (lhs / fwd_model.fl_model.storage_coefficient[:, :-1])[
                :, :, np.newaxis
            ] * dKijdKyv
            # Backward scheme
            out[:, 1:, :] += (lhs / fwd_model.fl_model.storage_coefficient[:, 1:])[
                :, :, np.newaxis
            ] * dKijdKyv

        # Handle the stationary case for n == 0
        elif fwd_model.fl_model.regime == FlowRegime.STATIONARY:
            z = fwd_model.fl_model._get_mesh_center_vertical_pos().T[:, :, np.newaxis]
            lhs = (
                (
                    (pressure[:, 1:] - pressure[:, :-1])[:, :, np.newaxis]
                    / GRAVITY
                    / WATER_DENSITY
                    - z[:, :-1]
                    + z[:, 1:]
                )
                * fwd_model.geometry.gamma_ij_y
                / fwd_model.geometry.dy
                / fwd_model.geometry.grid_cell_volume
            ) * dKijdKyv

            out[:, :-1, :] -= lhs
            out[:, 1:, :] += lhs

    out[
        fwd_model.fl_model.cst_head_indices[0], fwd_model.fl_model.cst_head_indices[1]
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
        -1.0 / (fwd_model.fl_model.storage_coefficient**2)[:, :, np.newaxis] * vecs
    )

    # Consider the x axis
    if shape[0] > 1:
        Kij = harmonic_mean(permeability[1:, :], permeability[:-1, :])
        rhoij = get_rhomean(
            fwd_model.geometry,
            fwd_model.tr_model,
            axis=0,
            time_index=time_index - 1,
            is_flatten=False,
        )[:-1, :]
        # For all n != 0
        if time_index != 0:
            if fwd_model.fl_model.vertical_axis == VerticalAxis.X:
                rhoijg = rhoij * GRAVITY
            else:
                rhoijg = 0.0
            lhs = (
                (
                    (
                        crank_flow * (pressure[1:, :] - pressure[:-1, :])
                        + (1.0 - crank_flow)
                        * (pressure_prev[1:, :] - pressure_prev[:-1, :])
                    )
                    / fwd_model.geometry.dx
                    + rhoijg
                )
                * fwd_model.geometry.gamma_ij_x
                * rhoij
                * Kij
                / WATER_DENSITY
                / fwd_model.geometry.grid_cell_volume
            )
            # Forward
            out[:-1, :, :] -= lhs[:, :, np.newaxis] * dinvSsV[:-1, :, :]
            # Backward scheme
            out[1:, :, :] += lhs[:, :, np.newaxis] * dinvSsV[1:, :, :]

    # Consider the y axis for 2D cases
    if shape[1] > 1:
        Kij = harmonic_mean(permeability[:, 1:], permeability[:, :-1])
        rhoij = get_rhomean(
            fwd_model.geometry,
            fwd_model.tr_model,
            axis=1,
            time_index=time_index - 1,
            is_flatten=False,
        )[:, :-1]

        # For all n != 0
        if time_index != 0:
            if fwd_model.fl_model.vertical_axis == VerticalAxis.Y:
                rhoijg = rhoij * GRAVITY
            else:
                rhoijg = 0.0

            lhs = (
                (
                    (
                        crank_flow * (pressure[:, 1:] - pressure[:, :-1])
                        + (1.0 - crank_flow)
                        * (pressure_prev[:, 1:] - pressure_prev[:, :-1])
                    )
                    / fwd_model.geometry.dy
                    + rhoijg
                )
                * fwd_model.geometry.gamma_ij_y
                * Kij
                * rhoij
                / WATER_DENSITY
                / fwd_model.geometry.grid_cell_volume
            )
            # Forward
            out[:, :-1, :] -= lhs[:, :, np.newaxis] * dinvSsV[:, :-1, :]
            # Backward scheme
            out[:, 1:, :] += lhs[:, :, np.newaxis] * dinvSsV[:, 1:, :]

    out -= (
        (
            fwd_model.fl_model.crank_nicolson * fwd_model.fl_model.lunitflow[time_index]
            + (1.0 - fwd_model.fl_model.crank_nicolson)
            * fwd_model.fl_model.lunitflow[time_index - 1]
        )
        * fwd_model.tr_model.ldensity[time_index - 1]
        * GRAVITY
    )[:, :, np.newaxis] * dinvSsV

    out[
        fwd_model.fl_model.cst_head_indices[0], fwd_model.fl_model.cst_head_indices[1]
    ] = 0

    return out


def dFUxdKv(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat:
    permeability = fwd_model.fl_model.permeability
    out = np.zeros(
        (fwd_model.geometry.nx + 1, fwd_model.geometry.ny, vecs.shape[-1]),
        dtype=np.float64,
    )
    dKijdKxv = (
        dxi_harmonic_mean(permeability[1:, :], permeability[:-1, :])[:, :, np.newaxis]
        * vecs[1:, :, :]
        + dxi_harmonic_mean(permeability[:-1, :], permeability[1:, :])[:, :, np.newaxis]
        * vecs[:-1, :, :]
    )

    if fwd_model.fl_model.is_gravity:
        rhomean = get_rhomean(
            fwd_model.geometry,
            fwd_model.tr_model,
            axis=0,
            time_index=time_index - 1,
            is_flatten=False,
        )[:, :-1]
        if fwd_model.fl_model.vertical_axis == VerticalAxis.X:
            rho_ij_g = rhomean * GRAVITY
            if time_index == 0:
                rho_ij_g[:, :] = WATER_DENSITY * GRAVITY
        else:
            rho_ij_g = np.zeros_like(rhomean)

        pressure = fwd_model.fl_model.lpressure[time_index]
        out[1:-1, :, :] += (
            dKijdKxv
            * (
                (
                    (pressure[1:, :] - pressure[:-1, :]) / fwd_model.geometry.dx
                    + rho_ij_g
                )
                / WATER_DENSITY
                / GRAVITY
            )[:, :, np.newaxis]
        )
    else:
        head = fwd_model.fl_model.lhead[time_index]
        out[1:-1, :, :] += (
            dKijdKxv
            * ((head[1:, :] - head[:-1, :]) / fwd_model.geometry.dx)[:, :, np.newaxis]
        )
    return out


def dFUydKv(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat:
    permeability = fwd_model.fl_model.permeability
    out = np.zeros(
        (fwd_model.geometry.nx, fwd_model.geometry.ny + 1, vecs.shape[-1]),
        dtype=np.float64,
    )
    dKijdKyv = (
        dxi_harmonic_mean(permeability[:, 1:], permeability[:, :-1])[:, :, np.newaxis]
        * vecs[:, 1:, :]
        + dxi_harmonic_mean(permeability[:, :-1], permeability[:, 1:])[:, :, np.newaxis]
        * vecs[:, :-1, :]
    )

    if fwd_model.fl_model.is_gravity:
        rhomean = get_rhomean(
            fwd_model.geometry,
            fwd_model.tr_model,
            axis=1,
            time_index=time_index - 1,
            is_flatten=False,
        )[:-1, :]
        if fwd_model.fl_model.vertical_axis == VerticalAxis.X:
            rho_ij_g = rhomean * GRAVITY
            if time_index == 0:
                rho_ij_g[:, :] = WATER_DENSITY * GRAVITY
        else:
            rho_ij_g = np.zeros_like(rhomean)

        pressure = fwd_model.fl_model.lpressure[time_index]
        out[:, 1:-1, :] += (
            dKijdKyv
            * (
                (
                    (pressure[:, 1:] - pressure[:, :-1]) / fwd_model.geometry.dy
                    + rho_ij_g
                )
                / WATER_DENSITY
                / GRAVITY
            )[:, :, np.newaxis]
        )
    else:
        head = fwd_model.fl_model.lhead[time_index]
        out[:, 1:-1, :] += (
            dKijdKyv
            * ((head[:, 1:] - head[:, :-1]) / fwd_model.geometry.dy)[:, :, np.newaxis]
        )
    return out


def dFhdhimp(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat:
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
    return fwd_model.tr_model.diffusion[:, :, np.newaxis] * vecs


@np_cache(pos=1)
def dFDdDv(fwd_model: ForwardModel, vecs: NDArrayFloat) -> NDArrayFloat:
    return fwd_model.tr_model.porosity[:, :, np.newaxis] * vecs


def dFDddispv(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat:
    return (
        fwd_model.fl_model.get_u_darcy_norm_sample(time_index)[:, :, np.newaxis] * vecs
    )


def dFcdwv(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat:
    # no need to take constant conc into account because conc(n+1) - conc(n) is null in
    # such case
    return (
        (fwd_model.tr_model.lmob[time_index] - fwd_model.tr_model.lmob[time_index - 1])
        / fwd_model.time_params.dt
    )[:, :, :, np.newaxis] * vecs[np.newaxis, :, :, :]


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
