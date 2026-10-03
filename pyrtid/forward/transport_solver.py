# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

r"""
Provide the transport solver.

The transport of the mobile species is described by the advection-dispersion equation
with a chemical source term (the variation of the immobile species):

.. math::
    \omega \dfrac{\partial c}{\partial t} + \nabla \cdot (u c)
    - \nabla \cdot (\omega D \nabla c) = q - \omega \dfrac{\partial \overline{c}}
    {\partial t}

where :math:`\omega` is the porosity, :math:`u` the darcy velocity, :math:`D` the
effective diffusion/dispersion and :math:`q` the sources. The equation is discretized
with finite volumes (upwind scheme for the advection) and a Crank-Nicolson time
scheme, with independent weights for the advection and the diffusion.
"""

from __future__ import annotations

import warnings
from typing import List, Tuple, Union

import numpy as np
from inv_toolbox.utils import get_super_ilu_preconditioner, harmonic_mean
from quickpaver import RectilinearGrid
from scipy.sparse import lil_array
from scipy.sparse.linalg import gmres

from pyrtid.forward.models import (
    FlowModel,
    SparseMatrixBuilder,
    TimeParameters,
    TransportModel,
    add_entries,
    add_to_diagonal,
    get_owner_neigh_indices,
)
from pyrtid.utils import NDArrayFloat

# The matrices can be filled in a lil_array or in a SparseMatrixBuilder
MatrixLike = Union[lil_array, SparseMatrixBuilder]


def _get_u_darcy_list(fl_model: FlowModel, axis: int) -> List[NDArrayFloat]:
    """
    Return the list of darcy velocities (at the faces) along ``axis``, for all times.

    The lists are used rather than the ``u_darcy_*`` properties of the flow model,
    which copy the velocities of all the times.
    """
    if axis == 0:
        return fl_model.lu_darcy_x
    elif axis == 1:
        return fl_model.lu_darcy_y
    elif axis == 2:
        return fl_model.lu_darcy_z
    raise ValueError(f"`axis` should be among [0, 1, 2], got {axis}.")


def fill_trmat_for_axis(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: MatrixLike,
    q_prev: MatrixLike,
    disp: NDArrayFloat,
    time_index: int,
    axis: int,
) -> None:
    """
    Add the contribution of the faces along ``axis`` to the transport matrices.

    The diffusion/dispersion is centered (harmonic mean of the coefficients at the
    faces) and the advection uses an upwind scheme.

    Parameters
    ----------
    grid : RectilinearGrid
        The grid.
    fl_model : FlowModel
        The flow model, which holds the darcy velocities.
    tr_model : TransportModel
        The transport model.
    q_next : Union[lil_array, SparseMatrixBuilder]
        Matrix of the terms at the next time (modified in place).
    q_prev : Union[lil_array, SparseMatrixBuilder]
        Matrix of the terms at the previous time (modified in place).
    disp : NDArrayFloat
        Diffusion/dispersion coefficient with shape (nx, ny, nz).
    time_index : int
        The time index (>= 1).
    axis : int
        Axis (0 for x, 1 for y, 2 for z).
    """
    u_darcy = _get_u_darcy_list(fl_model, axis)

    crank_adv: float = tr_model.crank_nicolson_advection
    crank_diff: float = tr_model.crank_nicolson_diffusion
    fwd_slicer = grid.get_slicer_forward(axis)
    bwd_slicer = grid.get_slicer_backward(axis)

    dmean: NDArrayFloat = np.zeros(grid.shape, dtype=np.float64)
    dmean[fwd_slicer] = harmonic_mean(disp[fwd_slicer], disp[bwd_slicer])
    dmean = dmean.flatten(order="F")

    tmp_diff: float = (
        grid.gc_face_area_m2(axis) / grid.pipj_m(axis) / grid.grid_cell_volume_m3
    )

    tmp_un = np.zeros(grid.shape)
    tmp_un[fwd_slicer] = u_darcy[time_index][bwd_slicer]
    tmp_un_old = np.zeros(grid.shape)
    tmp_un_old[fwd_slicer] = u_darcy[time_index - 1][bwd_slicer]

    un = tmp_un.flatten(order="F")
    un_old = tmp_un_old.flatten(order="F")

    tmp_adv = grid.gc_face_area_m2(axis) / grid.grid_cell_volume_m3

    # Forward scheme:
    normal = 1.0
    idc_owner, idc_neigh = get_owner_neigh_indices(
        grid,
        fwd_slicer,
        bwd_slicer,
        owner_indices_to_keep=tr_model.free_conc_nn,
    )

    add_entries(
        q_next,
        idc_owner,
        idc_owner,
        crank_diff * dmean[idc_owner] * tmp_diff
        + (
            crank_adv
            * np.where(normal * un > 0.0, normal * un, 0.0)[idc_owner]
            * tmp_adv
        ),  # noqa: E501
    )
    add_entries(
        q_next,
        idc_owner,
        idc_neigh,
        -(crank_diff * dmean[idc_owner] * tmp_diff)
        + (
            crank_adv
            * np.where(normal * un <= 0.0, normal * un, 0.0)[idc_owner]
            * tmp_adv
        ),
    )
    add_entries(
        q_prev,
        idc_owner,
        idc_owner,
        -(
            (1.0 - crank_diff) * dmean[idc_owner] * tmp_diff
            + (
                (1 - crank_adv)
                * np.where(normal * un_old > 0.0, normal * un_old, 0.0)[idc_owner]
                * tmp_adv
            )
        ),
    )
    add_entries(
        q_prev,
        idc_owner,
        idc_neigh,
        -(
            -((1.0 - crank_diff) * dmean[idc_owner] * tmp_diff)
            + (
                (1 - crank_adv)
                * np.where(normal * un_old <= 0.0, normal * un_old, 0.0)[idc_owner]
                * tmp_adv
            )
        ),
    )

    # Backward scheme
    normal = -1.0
    idc_owner, idc_neigh = get_owner_neigh_indices(
        grid,
        bwd_slicer,
        fwd_slicer,
        owner_indices_to_keep=tr_model.free_conc_nn,
    )

    add_entries(
        q_next,
        idc_owner,
        idc_owner,
        crank_diff * dmean[idc_neigh] * tmp_diff
        + (
            crank_adv
            * np.where(normal * un > 0.0, normal * un, 0.0)[idc_neigh]
            * tmp_adv
        ),  # noqa: E501
    )
    add_entries(
        q_next,
        idc_owner,
        idc_neigh,
        -(crank_diff * dmean[idc_neigh] * tmp_diff)
        + (
            crank_adv
            * np.where(normal * un <= 0.0, normal * un, 0.0)[idc_neigh]
            * tmp_adv
        ),
    )
    add_entries(
        q_prev,
        idc_owner,
        idc_owner,
        -(
            (1.0 - crank_diff) * dmean[idc_neigh] * tmp_diff
            + (
                (1.0 - crank_adv)
                * np.where(normal * un_old > 0.0, normal * un_old, 0.0)[idc_neigh]
                * tmp_adv
            )
        ),
    )
    add_entries(
        q_prev,
        idc_owner,
        idc_neigh,
        -(
            -(1.0 - crank_diff) * dmean[idc_neigh] * tmp_diff
            + (
                (1.0 - crank_adv)
                * np.where(normal * un_old <= 0.0, normal * un_old, 0.0)[idc_neigh]
                * tmp_adv
            )
        ),
    )


def _assemble_transport_matrices(
    grid: RectilinearGrid,
    tr_model: TransportModel,
    fl_model: FlowModel,
    time_index: int,
    q_next: MatrixLike,
    q_prev: MatrixLike,
) -> None:
    """Fill ``q_next`` and ``q_prev`` with all the terms of the transport (no 1/dt)."""
    # diffusion + dispersivity
    disp = (
        tr_model.effective_diffusion
        + tr_model.dispersivity * fl_model.get_u_darcy_norm_sample(time_index)
    )

    for n, axis in zip(grid.shape, (0, 1, 2)):
        if n >= 2:
            fill_trmat_for_axis(
                grid, fl_model, tr_model, q_next, q_prev, disp, time_index, axis
            )

    _apply_transport_sink_term(fl_model, tr_model, q_next, q_prev, time_index)

    _apply_divergence_effect(fl_model, tr_model, q_next, q_prev, time_index)

    # Handle boundary conditions
    _add_transport_boundary_conditions(
        grid, fl_model, tr_model, q_next, q_prev, time_index
    )


def make_transport_matrices(
    grid: RectilinearGrid,
    tr_model: TransportModel,
    fl_model: FlowModel,
    time_index: int,
) -> Tuple[lil_array, lil_array]:
    """
    Make matrices for the transport, without the time derivative term.

    Parameters
    ----------
    grid : RectilinearGrid
        The grid.
    tr_model : TransportModel
        The transport model.
    fl_model : FlowModel
        The flow model.
    time_index: int
        The iteration, or timestep id.

    Returns
    -------
    Tuple[lil_array, lil_array]
        The matrices of the terms at the next and at the previous time.
    """
    dim = grid.n_grid_cells
    q_next = SparseMatrixBuilder((dim, dim))
    q_prev = SparseMatrixBuilder((dim, dim))
    _assemble_transport_matrices(grid, tr_model, fl_model, time_index, q_next, q_prev)
    return q_next.tolil(), q_prev.tolil()


def _apply_transport_sink_term(
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: MatrixLike,
    q_prev: MatrixLike,
    time_index: int,
) -> None:
    """Add the sink terms (pumping) to the diagonal of the transport matrices."""
    flw = fl_model.lunitflow[time_index].flatten(order="F")
    _flw = np.where(flw < 0, flw, 0.0)  # keep only negative flowrates
    flw_old = fl_model.lunitflow[time_index - 1].flatten(order="F")
    _flw_old = np.where(flw_old < 0, flw_old, 0.0)  # keep only negative flowrates
    add_to_diagonal(q_next, -tr_model.crank_nicolson_advection * _flw)
    add_to_diagonal(q_prev, (1 - tr_model.crank_nicolson_advection) * _flw_old)


def _apply_divergence_effect(
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: MatrixLike,
    q_prev: MatrixLike,
    time_index: int,
) -> None:
    """
    Take into account the divergence of the velocity: dcdt+U.grad(c)=L(u).

    The divergence due to the flow sources is excluded, since the sources are
    handled by the sink term and by the concentration sources.
    """

    div = (fl_model.lu_darcy_div[time_index] - fl_model.lunitflow[time_index]).flatten(
        order="F"
    )
    div_old = (
        fl_model.lu_darcy_div[time_index - 1] - fl_model.lunitflow[time_index - 1]
    ).flatten(order="F")

    add_to_diagonal(q_next, -tr_model.crank_nicolson_advection * div)
    add_to_diagonal(q_prev, (1 - tr_model.crank_nicolson_advection) * div_old)


def _add_transport_boundary_conditions_for_axis(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: MatrixLike,
    q_prev: MatrixLike,
    time_index: int,
    axis: int,
) -> None:
    """
    Add the zero concentration gradient condition on the domain borders ``axis``.

    The concentration at the border faces is the one of the adjacent grid cell. The
    advective flux through the border faces is added to the diagonal.
    """
    u_darcy = _get_u_darcy_list(fl_model, axis)
    if axis == 0:
        bd1_slicer = (slice(0, 1), slice(None), slice(None))
        bd2_slicer = (slice(grid.nx - 1, grid.nx), slice(None), slice(None))
    elif axis == 1:
        bd1_slicer = (slice(None), slice(0, 1), slice(None))
        bd2_slicer = (slice(None), slice(grid.ny - 1, grid.ny), slice(None))
    else:
        bd1_slicer = (slice(None), slice(None), slice(0, 1))
        bd2_slicer = (slice(None), slice(None), slice(grid.nz - 1, grid.nz))

    fwd_slicer = grid.get_slicer_forward(axis, shift=1)
    bwd_slicer = grid.get_slicer_backward(axis, shift=1)

    idc_left_border, idc_right_border = get_owner_neigh_indices(
        grid,
        bd1_slicer,
        bd2_slicer,
    )
    tmp = grid.gc_face_area_m2(axis) / grid.grid_cell_volume_m3

    # left border
    _un = u_darcy[time_index][fwd_slicer].ravel("F")[idc_left_border]
    _un_old = u_darcy[time_index - 1][fwd_slicer].ravel("F")[idc_left_border]
    normal = -1.0
    add_entries(
        q_next,
        idc_left_border,
        idc_left_border,
        tr_model.crank_nicolson_advection * _un * tmp * normal,
    )
    add_entries(
        q_prev,
        idc_left_border,
        idc_left_border,
        -((1 - tr_model.crank_nicolson_advection) * _un_old * tmp * normal),
    )

    # right border
    _un = u_darcy[time_index][bwd_slicer].ravel("F")[idc_right_border]
    _un_old = u_darcy[time_index - 1][bwd_slicer].ravel("F")[idc_right_border]
    normal = 1.0
    add_entries(
        q_next,
        idc_right_border,
        idc_right_border,
        tr_model.crank_nicolson_advection * _un * tmp * normal,
    )
    add_entries(
        q_prev,
        idc_right_border,
        idc_right_border,
        -((1 - tr_model.crank_nicolson_advection) * _un_old * tmp * normal),
    )


def _add_transport_boundary_conditions(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: MatrixLike,
    q_prev: MatrixLike,
    time_index: int,
) -> None:
    """Add the boundary conditions to the matrix."""
    # We get the indices of the borders and we apply a zero gradient.

    for n, axis in zip(grid.shape, (0, 1, 2)):
        if n >= 2:
            _add_transport_boundary_conditions_for_axis(
                grid, fl_model, tr_model, q_next, q_prev, time_index, axis
            )


def solve_transport_semi_implicit(
    grid: RectilinearGrid,
    fl_model: FlowModel,
    tr_model: TransportModel,
    conc_sources: NDArrayFloat,
    conc_sources_old: NDArrayFloat,
    time_params: TimeParameters,
    time_index: int,
    nfpi: int,
) -> int:
    """
    Compute the transport of the mobile concentrations.

    The mobile concentrations are stored in ``tr_model.lmob[time_index]``. The matrices
    are built at the first fixed point iteration only, and reused afterwards: only
    the chemical source term changes from one iteration to the other.

    Parameters
    ----------
    grid: RectilinearGrid
        RectilinearGrid of the system.
    fl_model: FlowModel
        The flow model.
    tr_model: TransportModel
        The transport model.
    conc_sources: NDArrayFloat
        Concentration sources at the current time, with shape (n_sp, nx, ny, nz).
    conc_sources_old: NDArrayFloat
        Concentration sources at the previous time.
    time_params: TimeParameters
        Time parameters of the system.
    time_index: int
        The iteration, or timestep id.
    nfpi:
        Number of fixed point iterations (1 for the first iteration of the timestep).

    Returns
    -------
    int
        The exit code of the linear solver (0 for a successful exit). If several
        species fail, the code of the first one is returned.
    """
    n_sp = tr_model.n_sp

    # The matrix with respect to the diffusion never changes.
    # The matrix with respect to the advection only needs to be updated if the head
    # have changed.
    if nfpi == 1:
        dim = grid.n_grid_cells
        builder_next = SparseMatrixBuilder((dim, dim))
        builder_prev = SparseMatrixBuilder((dim, dim))
        _assemble_transport_matrices(
            grid, tr_model, fl_model, time_index, builder_next, builder_prev
        )

        # Add 1/dt for the left term contribution
        porosity_over_dt = tr_model.porosity.flatten("F") / time_params.dt
        add_to_diagonal(builder_next, porosity_over_dt)
        add_to_diagonal(builder_prev, porosity_over_dt)

        # csc format for efficiency
        q_next = builder_next.tocsc()
        q_prev = builder_prev.tocsc()
        tr_model.q_next = q_next
        tr_model.q_prev = q_prev

        if tr_model.is_save_spmats:
            tr_model.l_q_next.append(q_next)
            tr_model.l_q_prev.append(q_prev)

        # Build the LU preconditioning -> to do only once.
        try:
            super_ilu, preconditioner = get_super_ilu_preconditioner(
                q_next, drop_tol=1e-10, fill_factor=100
            )
        except RuntimeError:
            super_ilu, preconditioner = None, None
            warnings.warn(
                f"SuperILU: q_next is singular in transport at it={time_index}!"
            )

        tr_model.super_ilu = super_ilu
        tr_model.preconditioner = preconditioner
    else:
        q_next = tr_model.q_next
        q_prev = tr_model.q_prev
        super_ilu = tr_model.super_ilu
        preconditioner = tr_model.preconditioner

    # Multiply prev matrix by prev vector
    tmp = q_prev.dot(tr_model.lmob[time_index - 1].reshape(n_sp, -1, order="F").T).T

    # Chemical source term
    if tr_model.is_num_acc_for_timestep and nfpi == 1 and time_index != 1:
        dmdt = tr_model.limmob[time_index - 1] - tr_model.limmob[time_index - 2]
        # avoid negative values
        if np.any(tr_model.lmob[time_index - 1] - dmdt < 0):
            dmdt = tr_model.limmob[time_index] - tr_model.limmob[time_index - 1]
    else:
        dmdt = tr_model.limmob[time_index] - tr_model.limmob[time_index - 1]

    # The volume is included in the diffusion term
    tmp -= (
        dmdt.reshape(n_sp, -1, order="F")
        * tr_model.porosity.ravel("F")
        / time_params.dt
    )

    # Add the source terms -> depends on the advection (positive flowrates = injection)
    tmp[:, :] += tr_model.crank_nicolson_advection * conc_sources.reshape(
        n_sp, -1, order="F"
    ) + (1.0 - tr_model.crank_nicolson_advection) * conc_sources_old.reshape(
        n_sp, -1, order="F"
    )

    # Solve Ax = b with A sparse using LU preconditioner
    exit_code = 0
    for isp in range(n_sp):
        tmp[isp, :], _exit_code = gmres(
            q_next,
            tmp[isp, :],
            x0=super_ilu.solve(tmp[isp, :]) if super_ilu is not None else None,
            M=preconditioner,
            rtol=tr_model.rtol,
        )
        if _exit_code != 0:
            warnings.warn(
                f"The GMRES solver of the transport did not converge for species "
                f"{isp} at it={time_index} (exit code {_exit_code})."
            )
            exit_code = exit_code or _exit_code

    tr_model.lmob[time_index] = tmp.reshape(n_sp, *grid.shape, order="F")

    return exit_code
