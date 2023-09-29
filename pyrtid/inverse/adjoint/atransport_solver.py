"""Provide an adjoint solver for the transport operator."""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy.sparse import lil_array, lil_matrix
from scipy.sparse.linalg import gmres

from pyrtid.forward.models import (
    FlowModel,
    GeochemicalParameters,
    Geometry,
    TimeParameters,
    TransportModel,
    get_owner_neigh_indices,
)
from pyrtid.forward.solver import VERY_SMALL_NUMBER, get_max_coupling_error
from pyrtid.inverse.adjoint.amodels import AdjointTransportModel
from pyrtid.utils import harmonic_mean
from pyrtid.utils.operators import get_super_ilu_preconditioner
from pyrtid.utils.types import NDArrayFloat


def get_adjoint_max_coupling_error(
    atr_model: AdjointTransportModel, time_index: int
) -> float:
    r"""
    Return the maximum adjoint chemistry-transport coupling error.

    The fixed point iteration convergence criteria reads:

    .. math::
        \text{max} \left\lVert 1 - \dfrac{\lambda_{c}^{n, k+1}}
        {\lambda_{c}^{n, k}} \right\rVert  < \epsilon

    with $k$ the number of fixed point iterations.

    This error is evaluated from the mobile adjoint concentrations.
    """
    return get_max_coupling_error(
        atr_model.a_conc[:, :, time_index], atr_model.a_conc_prev
    )


def solve_adj_initial_transport(
    geometry: Geometry,
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_tr_model: AdjointTransportModel,
    time_params: TimeParameters,
    time_index: int,
    nafpi: int,
) -> int:
    """
    Solving the adjoint diffusivity equation:

    dc/dt = div D grad c + ...
    """

    tr_model.crank_nicolson_advection

    if nafpi == 1:
        q_next = a_tr_model.q_next_diffusion.copy().tolil()
        q_prev = a_tr_model.q_prev_diffusion.copy().tolil()

        # Update q_next and q_prev with the advection term (must be copied)
        # Note that this is required at the first fixed point iteration only,
        # afterwards, only the chemical source term varies.
        _add_advection_to_adj_transport_matrices(
            geometry, fl_model, tr_model, a_tr_model, q_next, q_prev, time_index
        )

        # Add 1/dt * \omgea for the left term contribution
        # Note; if the porosity and the timesteps are added here, it is to get the
        # highest values as possible on the diagonal of the matrices
        # -> better conditionning and easier LU preconditioning.
        q_next.setdiag(
            q_next.diagonal() + tr_model.porosity.flatten("F") / time_params.ldt[-1]
        )

        a_tr_model.q_next = q_next
    else:
        q_next = a_tr_model.q_next

    # Multiply prev matrix by prev vector
    tmp = np.zeros(geometry.nx * geometry.ny)

    # Add the adjoint source terms
    # Add the source terms
    tmp += (
        a_tr_model.a_conc_sources.getcol(time_index).todense().ravel()
        / geometry.mesh_volume
    )

    # Add the adjoint geochem source term
    tmp -= (a_tr_model.a_gch_src_term / geometry.mesh_volume).ravel("F")

    # Build the LU preconditioning
    preconditioner = get_super_ilu_preconditioner(q_next.tocsc())

    # Solve Ax = b with A sparse using LU preconditioner
    res, exit_code = gmres(q_next.tocsc(), tmp, M=preconditioner, atol=1e-15)
    # Note: we go backward in time, so time_index -1...
    a_tr_model.a_conc[:, :, time_index] = res.reshape(geometry.ny, geometry.nx).T

    return exit_code


def init_adjoint_tr_variables_fpi(
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_tr_model: AdjointTransportModel,
    gch_params: GeochemicalParameters,
    geometry: Geometry,
    time_params: TimeParameters,
    is_verbose: bool = False,
) -> None:
    r"""
    Initiate the initial (at tmax) adjoint transport variable using FPI.

    FPI stands for fixed point iterations.

    .. math::
        \begin{cases}
            \lambda_{c_{i}}^{N+1} = \dfrac{\Delta t^{N} }{\mathcal{A}_{i} \omega_{i}}
            \left( - \dfrac{c_{i}^{N+1, \mathrm{obs}} - c_{i}^{N+1, \mathrm{calc}}}{
                \left(\sigma_{c_{i}}^{N+1, \mathrm{obs}}\right)^{2}} - \lambda_{
                    \overline{c}_{i}}^{N+1} \Delta t^{N}  k_{v} A_{s} \dfrac{
                        \overline{c}_{i}^{N}}{Ks} \right), & \text{Initial condition}
            \\\\
            \lambda_{\overline{c}_{i}}^{N+1} =  - \dfrac{\mathcal{A}_{i}
            \omega_{i}}{\Delta t^{N}} \lambda_{c_{i}}^{N+1} - \dfrac{
                \overline{c}_{i}^{N+1, \mathrm{obs}} - \overline{c}_{i}^{N+1,
                \mathrm{calc}}}{\left(\sigma_{\overline{c}_{i}}^{N+1,
                \mathrm{obs}}\right)^{2}}, & \text{Initial condition}
            \\\\
        \end{cases}


    The convergence criteria is given by:

    .. math::
        \text{max} \left\lVert 1 - \dfrac{\lambda_{c}^{N+1, k+1}}
        {\lambda_{c}^{N+1, k}} \right\rVert  < \epsilon

    with $k$ the number of fixed point iterations.

    Parameters
    ----------
    tr_model : TransportModel
        _description_
    a_tr_model : AdjointTransportModel
        _description_
    gch_params : GeochemicalParameters
        _description_
    geometry : Geometry
        _description_
    time_params : TimeParameters
        _description_
    is_verbose : bool, optional
        _description_, by default False
    """
    if is_verbose:
        logging.info(" - Adjoint transport FPI initialization!")
    has_converged = False
    nafpi = 0

    # Initiate the concentrations to a very small number so that
    # a_conc_prev is not zero when calling get_adjoint_max_coupling_error
    # for the first time (end of the first loop)
    a_tr_model.a_conc[:, :, -1] = VERY_SMALL_NUMBER

    tmp = (
        time_params.ldt[-1]
        * gch_params.kv
        * gch_params.As
        * tr_model.lgrade[-2]
        / gch_params.Ks
    )

    while not has_converged:
        nafpi += 1
        # Copy for the convergence check
        a_tr_model.a_conc_prev = a_tr_model.a_conc[:, :, -1].copy()

        # Compute adjoint grades
        a_tr_model.a_grade[:, :, -1] = (
            -a_tr_model.a_conc[:, :, -1]
            * (tr_model.porosity * geometry.mesh_volume)
            / time_params.ldt[-1]
        ) + a_tr_model.a_grade_sources.getcol(-1).reshape(geometry.shape, order="F")

        # Source term for the transport
        a_tr_model.a_gch_src_term = a_tr_model.a_grade[:, :, -1] * tmp

        # Solve adjoint transport
        solve_adj_initial_transport(
            geometry, fl_model, tr_model, a_tr_model, time_params, -1, nafpi
        )

        has_converged = (
            get_adjoint_max_coupling_error(a_tr_model, -1) < tr_model.fpi_eps
        )

        if is_verbose:
            logging.info(
                f"max-coupling error at it = {time_params.nt}-{nafpi}: "
                f"{get_adjoint_max_coupling_error(a_tr_model, -1)}"
            )
        has_converged = (
            get_adjoint_max_coupling_error(a_tr_model, -1) < tr_model.fpi_eps
        )
        if is_verbose:
            logging.info(f"has-converged ?: {has_converged}")


def make_transient_adj_transport_matrices(
    geometry: Geometry, tr_model: TransportModel, time_params: TimeParameters
) -> Tuple[lil_matrix, lil_matrix]:
    """
    Make matrices for the transient transport.

    Note
    ----
    Since the diffusion coefficient and porosity does not vary with time,
    matrices q_prev and q_next are the same.
    """

    dim = geometry.nx * geometry.ny
    q_prev = lil_array((dim, dim), dtype=np.float64)
    q_next = lil_array((dim, dim), dtype=np.float64)

    # X contribution
    if geometry.nx >= 2:
        dmean: NDArrayFloat = np.zeros((geometry.nx, geometry.ny), dtype=np.float64)
        dmean[:-1, :] = harmonic_mean(
            tr_model.effective_diffusion[:-1, :], tr_model.effective_diffusion[1:, :]
        )
        dmean = dmean.flatten(order="F")

        # Forward scheme:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(0, geometry.nx - 1), slice(None)),
            (slice(1, geometry.nx), slice(None)),
            tr_model.cst_conc_indices,
        )

        tmp = geometry.dy / geometry.dx / geometry.mesh_volume

        q_next[idc_owner, idc_neigh] -= (
            tr_model.crank_nicolson_diffusion * dmean[idc_owner] * tmp
        )  # type: ignore
        q_next[idc_owner, idc_owner] += (
            tr_model.crank_nicolson_diffusion * dmean[idc_owner] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_owner] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_owner] * tmp
        )  # type: ignore

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(1, geometry.nx), slice(None)),
            (slice(0, geometry.nx - 1), slice(None)),
            tr_model.cst_conc_indices,
        )

        q_next[idc_owner, idc_neigh] -= (
            tr_model.crank_nicolson_diffusion * dmean[idc_neigh] * tmp
        )  # type: ignore
        q_next[idc_owner, idc_owner] += (
            tr_model.crank_nicolson_diffusion * dmean[idc_neigh] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_neigh] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_neigh] * tmp
        )  # type: ignore

    # Y contribution
    if geometry.ny >= 2:
        dmean: NDArrayFloat = np.zeros((geometry.nx, geometry.ny), dtype=np.float64)
        dmean[:, :-1] = harmonic_mean(
            tr_model.effective_diffusion[:, :-1], tr_model.effective_diffusion[:, 1:]
        )
        dmean = dmean.flatten(order="F")

        # Forward scheme:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, geometry.ny - 1)),
            (slice(None), slice(1, geometry.ny)),
            tr_model.cst_conc_indices,
        )

        tmp = geometry.dx / geometry.dy / geometry.mesh_volume

        q_next[idc_owner, idc_neigh] -= (
            tr_model.crank_nicolson_diffusion * dmean[idc_owner] * tmp
        )  # type: ignore
        q_next[idc_owner, idc_owner] += (
            tr_model.crank_nicolson_diffusion * dmean[idc_owner] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_owner] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_owner] * tmp
        )  # type: ignore

        # Backward scheme
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(1, geometry.ny)),
            (slice(None), slice(0, geometry.ny - 1)),
            tr_model.cst_conc_indices,
        )

        q_next[idc_owner, idc_neigh] -= (
            tr_model.crank_nicolson_diffusion * dmean[idc_neigh] * tmp
        )  # type: ignore
        q_next[idc_owner, idc_owner] += (
            tr_model.crank_nicolson_diffusion * dmean[idc_neigh] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] += (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_neigh] * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1.0 - tr_model.crank_nicolson_diffusion) * dmean[idc_neigh] * tmp
        )  # type: ignore

    return q_next, q_prev


def _add_advection_to_adj_transport_matrices(
    geometry: Geometry,
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_tr_model: AdjointTransportModel,
    q_next: lil_matrix,
    q_prev: lil_matrix,
    time_index: int,
) -> None:
    crank_adv = tr_model.crank_nicolson_advection

    # X contribution
    if geometry.nx >= 2:
        tmp = np.zeros((geometry.nx, geometry.ny))
        tmp[:-1, :] = fl_model.u_darcy_x[1:-1, :, time_index]
        un_x = tmp.flatten(order="F")

        # Forward scheme:
        normal = 1.0
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(0, geometry.nx - 1), slice(None)),
            (slice(1, geometry.nx), slice(None)),
            tr_model.cst_conc_indices,
        )
        tmp = geometry.dy / geometry.mesh_volume

        tmp_un_pos = np.where(normal * un_x > 0.0, normal * un_x, 0.0)[idc_owner]

        q_next[idc_owner, idc_owner] += crank_adv * tmp_un_pos * tmp  # type: ignore
        q_next[idc_owner, idc_neigh] -= crank_adv * tmp_un_pos * tmp  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1 - crank_adv) * tmp_un_pos * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] += (
            (1 - crank_adv) * tmp_un_pos * tmp
        )  # type: ignore

        # Backward scheme
        normal = -1.0
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(1, geometry.nx), slice(None)),
            (slice(0, geometry.nx - 1), slice(None)),
            tr_model.cst_conc_indices,
        )

        tmp_un_pos = np.where(normal * un_x > 0.0, normal * un_x, 0.0)[idc_neigh]

        q_next[idc_owner, idc_owner] += crank_adv * tmp_un_pos * tmp  # type: ignore
        q_next[idc_owner, idc_neigh] -= crank_adv * tmp_un_pos * tmp  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1 - crank_adv) * tmp_un_pos * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] += (
            (1 - crank_adv) * tmp_un_pos * tmp
        )  # type: ignore

    # Y contribution
    if geometry.ny >= 2:
        tmp = np.zeros((geometry.nx, geometry.ny))
        tmp[:, :-1] = fl_model.u_darcy_y[:, 1:-1, time_index]
        un_y = tmp.flatten(order="F")

        # Forward scheme:
        normal = 1.0
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, geometry.ny - 1)),
            (slice(None), slice(1, geometry.ny)),
            tr_model.cst_conc_indices,
        )
        tmp = geometry.dx / geometry.mesh_volume

        tmp_un_pos = np.where(normal * un_y > 0.0, normal * un_y, 0.0)[idc_owner]

        q_next[idc_owner, idc_owner] += crank_adv * tmp_un_pos * tmp  # type: ignore
        q_next[idc_owner, idc_neigh] -= crank_adv * tmp_un_pos * tmp  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1 - crank_adv) * tmp_un_pos * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] += (
            (1 - crank_adv) * tmp_un_pos * tmp
        )  # type: ignore

        # Backward scheme
        normal = -1.0
        idc_owner, idc_neigh = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(1, geometry.ny)),
            (slice(None), slice(0, geometry.ny - 1)),
            tr_model.cst_conc_indices,
        )

        tmp_un_pos = np.where(normal * un_y > 0.0, normal * un_y, 0.0)[idc_neigh]

        q_next[idc_owner, idc_owner] += crank_adv * tmp_un_pos * tmp  # type: ignore
        q_next[idc_owner, idc_neigh] -= crank_adv * tmp_un_pos * tmp  # type: ignore
        q_prev[idc_owner, idc_owner] -= (
            (1 - crank_adv) * tmp_un_pos * tmp
        )  # type: ignore
        q_prev[idc_owner, idc_neigh] += (
            (1 - crank_adv) * tmp_un_pos * tmp
        )  # type: ignore

    _apply_adj_transport_sink_term(fl_model, tr_model, q_next, q_prev, time_index)

    _apply_adj_divergence_effect(fl_model, tr_model, q_next, q_prev, time_index)

    # Handle boundary conditions
    _add_adj_transport_boundary_conditions(
        geometry, fl_model, tr_model, q_next, q_prev, time_index
    )


def _apply_adj_transport_sink_term(
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: lil_matrix,
    q_prev: lil_matrix,
    time_index: int,
) -> None:
    flw = fl_model.lunitflow[time_index].flatten(order="F")
    _flw = np.where(flw < 0, flw, 0.0)  # keep only negative flowrates
    q_next.setdiag(q_next.diagonal() - tr_model.crank_nicolson_advection * _flw)
    q_prev.setdiag(q_prev.diagonal() + (1 - tr_model.crank_nicolson_advection) * _flw)


def _apply_adj_divergence_effect(
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: lil_matrix,
    q_prev: lil_matrix,
    time_index: int,
) -> None:
    """Take into account the divergence: dcdt+U.grad(c)=L(u)."""
    src = fl_model.lunitflow[time_index]
    src_old = fl_model.lunitflow[time_index]

    div = (fl_model.u_darcy_div[:, :, time_index] - src).flatten(order="F")
    div_old = (fl_model.u_darcy_div[:, :, time_index] - src_old).flatten(order="F")

    q_next.setdiag(q_next.diagonal() - tr_model.crank_nicolson_advection * div)
    q_prev.setdiag(
        q_prev.diagonal() + (1 - tr_model.crank_nicolson_advection) * div_old
    )


def _add_adj_transport_boundary_conditions(
    geometry: Geometry,
    fl_model: FlowModel,
    tr_model: TransportModel,
    q_next: lil_matrix,
    q_prev: lil_matrix,
    time_index: int,
) -> None:
    """Add the boundary conditions to the matrix."""
    # X contribution
    if geometry.nx >= 2:
        # We get the indices of the four borders and we apply a zero-conc gradient.
        idc_left, idc_right = get_owner_neigh_indices(
            geometry,
            (slice(0, 1), slice(None)),
            (slice(geometry.nx - 1, geometry.nx), slice(None)),
            np.array([]),
        )
        tmp = geometry.dy / geometry.mesh_volume

        _un = fl_model.u_darcy_x[:-1, :, time_index].ravel("F")[idc_left]
        # _un_old = fl_model.u_darcy_x[:-1, :, time_index + 1].ravel("F")[idc_left]
        normal = -1.0
        q_next[idc_left, idc_left] += (
            tr_model.crank_nicolson_advection * _un * tmp * normal
        )  # type: ignore
        q_prev[idc_left, idc_left] -= (
            (1 - tr_model.crank_nicolson_advection) * _un * tmp * normal
        )  # type: ignore

        _un = fl_model.u_darcy_x[1:, :, time_index].ravel("F")[idc_right]
        # _un_old = fl_model.u_darcy_x[1:, :, time_index + 1].ravel("F")[idc_right]
        normal = 1.0
        q_next[idc_right, idc_right] += (
            tr_model.crank_nicolson_advection * _un * tmp * normal
        )  # type: ignore
        q_prev[idc_right, idc_right] -= (
            (1 - tr_model.crank_nicolson_advection) * _un * tmp * normal
        )  # type: ignore

    # Y contribution
    if geometry.ny >= 2:
        # We get the indices of the four borders and we apply a zero-conc gradient.
        idc_left, idc_right = get_owner_neigh_indices(
            geometry,
            (slice(None), slice(0, 1)),
            (slice(None), slice(geometry.ny - 1, geometry.ny)),
            np.array([]),
        )
        tmp = geometry.dy / geometry.mesh_volume

        _un = fl_model.u_darcy_y[:, :-1, time_index].ravel("F")[idc_left]
        normal = -1.0
        q_next[idc_left, idc_left] += (
            tr_model.crank_nicolson_advection * _un * tmp * normal
        )  # type: ignore
        q_prev[idc_left, idc_left] -= (
            (1 - tr_model.crank_nicolson_advection) * _un * tmp * normal
        )  # type: ignore

        _un = fl_model.u_darcy_y[:, 1:, time_index].ravel("F")[idc_right]
        normal = 1.0
        q_next[idc_right, idc_right] += (
            tr_model.crank_nicolson_advection * _un * tmp * normal
        )  # type: ignore
        q_prev[idc_right, idc_right] -= (
            (1 - tr_model.crank_nicolson_advection) * _un * tmp * normal
        )  # type: ignore


def solve_adj_transport_transient_semi_implicit(
    geometry: Geometry,
    fl_model: FlowModel,
    tr_model: TransportModel,
    a_tr_model: AdjointTransportModel,
    time_params: TimeParameters,
    time_index: int,
    nafpi: int,
) -> int:
    """Solving the adjoint transport equation."""

    # The matrix with respect to the diffusion never changes.
    # The matrix with respect to the advection only needs to be updated at the first
    # fix point iteration
    if nafpi == 1:
        q_next = tr_model.q_next_diffusion.copy()
        q_prev = tr_model.q_prev_diffusion.copy()

        # Update q_next and q_prev with the advection term (must be copied)
        # Note that this is required at the first fixed point iteration only,
        # afterwards, only the chemical source term varies.
        _add_advection_to_adj_transport_matrices(
            geometry, fl_model, tr_model, a_tr_model, q_next, q_prev, time_index
        )

        # Add 1/dt * \omgea for the left term contribution
        # Note; if the porosity and the timesteps are added here, it is to get the
        # highest values as possible on the diagonal of the matrices
        # -> better conditionning and easier LU preconditioning.
        q_next.setdiag(
            q_next.diagonal()
            + tr_model.porosity.flatten("F") / time_params.ldt[time_index - 1]
        )
        q_prev.setdiag(
            q_prev.diagonal()
            + tr_model.porosity.flatten("F") / time_params.ldt[time_index]
        )

        a_tr_model.q_next = q_next
        a_tr_model.q_prev = q_prev
    else:
        q_next = a_tr_model.q_next
        q_prev = a_tr_model.q_prev

    # Get the previous vector
    prev_vector = a_tr_model.a_conc[:, :, time_index + 1].ravel("F")

    # Multiply prev matrix by prev vector
    tmp: NDArrayFloat = q_prev.dot(prev_vector)

    # Add the source terms
    tmp += (
        a_tr_model.a_conc_sources.getcol(time_index).todense().ravel()
        / geometry.mesh_volume
    )

    # Add the adjoint geochem source term
    tmp -= a_tr_model.a_gch_src_term.ravel(order="F") / geometry.mesh_volume

    # Build the LU preconditioning
    preconditioner = get_super_ilu_preconditioner(q_next.tocsc())

    # Solve Ax = b with A sparse using LU preconditioner
    res, exit_code = gmres(q_next.tocsc(), tmp, M=preconditioner, atol=1e-15)
    # Note: we go backward in time, so time_index -1...
    a_tr_model.a_conc[:, :, time_index] = res.reshape(geometry.ny, geometry.nx).T

    return exit_code
