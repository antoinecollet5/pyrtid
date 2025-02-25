"""
Provides the derivatives of the equations F with respect to the parameters s.

This is required to build the right hand side in the forward sensitivity method.
"""

from __future__ import annotations

import numpy as np

from pyrtid.forward.flow_solver import GRAVITY, WATER_DENSITY, get_rhomean
from pyrtid.forward.models import FlowRegime, ForwardModel, VerticalAxis
from pyrtid.utils import NDArrayFloat, dxi_harmonic_mean, harmonic_mean


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

    grad = np.zeros(shape)

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
            grad[:-1, :, :] -= (lhs / fwd_model.fl_model.storage_coefficient[:-1, :])[
                :, :, np.newaxis
            ] * dKijdKxv
            # Backward scheme
            grad[1:, :, :] += (lhs / fwd_model.fl_model.storage_coefficient[1:, :])[
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
            grad[:-1, :, :] -= lhs
            grad[1:, :, :] += lhs

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
            grad[:, :-1, :] -= (lhs / fwd_model.fl_model.storage_coefficient[:, :-1])[
                :, :, np.newaxis
            ] * dKijdKyv
            # Backward scheme
            grad[:, 1:, :] += (lhs / fwd_model.fl_model.storage_coefficient[:, 1:])[
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
            grad[:, :-1, :] -= lhs
            grad[:, 1:, :] += lhs

    grad[
        fwd_model.fl_model.cst_head_indices[0], fwd_model.fl_model.cst_head_indices[1]
    ] = 0

    return grad


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

    grad = np.zeros(shape)

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
            grad[:-1, :, :] -= lhs[:, :, np.newaxis] * dinvSsV[:-1, :, :]
            # Backward scheme
            grad[1:, :, :] += lhs[:, :, np.newaxis] * dinvSsV[1:, :, :]

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
            grad[:, :-1, :] -= lhs[:, :, np.newaxis] * dinvSsV[:, :-1, :]
            # Backward scheme
            grad[:, 1:, :] += lhs[:, :, np.newaxis] * dinvSsV[:, 1:, :]

    grad[
        fwd_model.fl_model.cst_head_indices[0], fwd_model.fl_model.cst_head_indices[1]
    ] = 0

    return grad


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

    grad = np.zeros(shape)

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
            grad[:-1, :, :] -= (lhs / fwd_model.fl_model.storage_coefficient[:-1, :])[
                :, :, np.newaxis
            ] * dKijdKxv
            # Backward scheme
            grad[1:, :, :] += (lhs / fwd_model.fl_model.storage_coefficient[1:, :])[
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

            grad[:-1, :, :] -= lhs
            grad[1:, :, :] += lhs

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
            grad[:, :-1, :] -= (lhs / fwd_model.fl_model.storage_coefficient[:, :-1])[
                :, :, np.newaxis
            ] * dKijdKyv
            # Backward scheme
            grad[:, 1:, :] += (lhs / fwd_model.fl_model.storage_coefficient[:, 1:])[
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

            grad[:, :-1, :] -= lhs
            grad[:, 1:, :] += lhs

    grad[
        fwd_model.fl_model.cst_head_indices[0], fwd_model.fl_model.cst_head_indices[1]
    ] = 0

    return grad


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

    grad = np.zeros(shape)

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
                * fwd_model.geometry.gamma_ij_y
                * rhoij
                * Kij
                / WATER_DENSITY
                / fwd_model.geometry.grid_cell_volume
            )
            # Forward
            grad[:-1, :, :] -= lhs[:, :, np.newaxis] * dinvSsV[:-1, :, :]
            # Backward scheme
            grad[1:, :, :] += lhs[:, :, np.newaxis] * dinvSsV[1:, :, :]

    # Consider the y axis for 2D cases
    # Consider the x axis
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
            grad[:, :-1, :] -= lhs[:, :, np.newaxis] * dinvSsV[:, :-1, :]
            # Backward scheme
            grad[:, 1:, :] += lhs[:, :, np.newaxis] * dinvSsV[:, 1:, :]

    grad[
        fwd_model.fl_model.cst_head_indices[0], fwd_model.fl_model.cst_head_indices[1]
    ] = 0

    return grad


def dFUxdKv(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat:
    # Here: impact of the gravity in both cases.

    if not fwd_model.fl_model.is_gravity:
        return np.zeros_like(fwd_model.fl_model.lpressure[0])
    # Case two = density flow = impact of density

    return vecs


def dFUydKv(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat:
    # Here: impact of the gravity in both cases.

    if not fwd_model.fl_model.is_gravity:
        return np.zeros_like(fwd_model.fl_model.lpressure[0])
    # Case two = density flow = impact of density

    return vecs


def dFhdhimp(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat: ...


def dFhdpimp(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat: ...


def dFDdwv(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat: ...


def dFcdwv(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat: ...


def dFDdDv(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat: ...


def dFDdav(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat: ...


def dFcdcimp(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat: ...


def dFmdcimp(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat: ...


def dFcdmimp(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat: ...


def dFmdmimp(
    fwd_model: ForwardModel, time_index: int, vecs: NDArrayFloat
) -> NDArrayFloat: ...
