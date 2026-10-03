# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

r"""
Provide the derivatives of the discretized equations with respect to the parameters.

In the forward sensitivity method (FSM), the sensitivities :math:`z^n` of the state
variables to a perturbation :math:`v` of the parameters :math:`s` follow, at each
timestep :math:`n`, the linear system:

.. math::
    \frac{\partial F^n}{\partial u^n} z^n = - \frac{\partial F^n}{\partial u^{n-1}}
    z^{n-1} - \frac{\partial F^n}{\partial s} v

This module provides the "forcing" terms :math:`\frac{\partial F^n}{\partial s} v` of
the equations. As the discretized operators (flow and transport matrices, darcy
velocities, ...) are assembled by the forward code, their directional derivatives
are obtained by differencing the assembly itself, which is cheap (no linear system
is solved) and guarantees the consistency with the forward discretization:

- the operators are affine with respect to the heads and to the velocities (for a
  given sign of the velocity, i.e., a given upwind direction), so the
  derivatives with respect to these variables are exact;
- the others (harmonic means of the permeability and of the dispersion) are smooth,
  and are differentiated with a centered scheme with a small relative step. The
  error is of the order of :math:`10^{-10}`.

The functions work on one direction (i.e., one column of the matrix of vectors to
multiply with the Jacobian matrix) at a time.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from quickpaver import RectilinearGrid
from scipy import sparse

from pyrtid.forward.flow_solver import (
    _add_time_derivative_and_cst_head,
    _assemble_transient_flow_matrices,
    compute_u_darcy,
    compute_u_darcy_div,
    make_stationary_flow_matrices,
)
from pyrtid.forward.models import (
    FlowModel,
    ForwardModel,
    SparseMatrixBuilder,
)
from pyrtid.forward.transport_solver import _assemble_transport_matrices
from pyrtid.utils import NDArrayFloat

# Relative size of the perturbation used to differentiate the smooth operators.
REL_STEP = 1e-6


@dataclass
class FlowFields:
    """
    Fields of the flow which are used by the transport at one time.

    Attributes
    ----------
    u_x : NDArrayFloat
        Darcy velocity at the faces along x with shape (nx + 1, ny, nz, [ne]).
    u_y : NDArrayFloat
        Darcy velocity at the faces along y with shape (nx, ny + 1, nz, [ne]).
    u_z : NDArrayFloat
        Darcy velocity at the faces along z with shape (nx, ny, nz + 1, [ne]).
    div : NDArrayFloat
        Divergence of the darcy velocity with shape (nx, ny, nz, [ne]).
    unitflow : NDArrayFloat
        Unit flow (sources and sinks) with shape (nx, ny, nz, [ne]).
    """

    u_x: NDArrayFloat
    u_y: NDArrayFloat
    u_z: NDArrayFloat
    div: NDArrayFloat
    unitflow: NDArrayFloat

    @classmethod
    def zeros(cls, grid: RectilinearGrid, ne: int) -> FlowFields:
        """Return null fields (``ne`` is the number of directions)."""
        nx, ny, nz = grid.shape
        return cls(
            np.zeros((nx + 1, ny, nz, ne)),
            np.zeros((nx, ny + 1, nz, ne)),
            np.zeros((nx, ny, nz + 1, ne)),
            np.zeros((nx, ny, nz, ne)),
            np.zeros((nx, ny, nz, ne)),
        )

    @classmethod
    def from_model(cls, fl_model: FlowModel, time_index: int) -> FlowFields:
        """Return the fields stored in the flow model at the given time."""
        return cls(
            fl_model.lu_darcy_x[time_index],
            fl_model.lu_darcy_y[time_index],
            fl_model.lu_darcy_z[time_index],
            fl_model.lu_darcy_div[time_index],
            fl_model.lunitflow[time_index],
        )

    def column(self, idx: int) -> FlowFields:
        """Return the fields of the direction ``idx`` (for fields with directions)."""
        return FlowFields(
            self.u_x[..., idx],
            self.u_y[..., idx],
            self.u_z[..., idx],
            self.div[..., idx],
            self.unitflow[..., idx],
        )

    def set_column(self, idx: int, other: FlowFields) -> None:
        """Set the fields of the direction ``idx``."""
        self.u_x[..., idx] = other.u_x
        self.u_y[..., idx] = other.u_y
        self.u_z[..., idx] = other.u_z
        self.div[..., idx] = other.div
        self.unitflow[..., idx] = other.unitflow

    def shifted(self, other: FlowFields, eps: float) -> FlowFields:
        """Return ``self + eps * other``."""
        return FlowFields(
            self.u_x + eps * other.u_x,
            self.u_y + eps * other.u_y,
            self.u_z + eps * other.u_z,
            self.div + eps * other.div,
            self.unitflow + eps * other.unitflow,
        )

    def max_abs_velocity(self) -> float:
        """Return the maximum absolute value of the velocities."""
        return max(
            float(np.max(np.abs(arr))) if arr.size else 0.0
            for arr in (self.u_x, self.u_y, self.u_z)
        )


def _rel_step(direction: NDArrayFloat, values: NDArrayFloat) -> float:
    """
    Return a step eps such that ``eps * direction`` is small compared to ``values``.

    The relative variation is ``REL_STEP``. If the direction is null, 0 is returned.
    """
    dmax = float(np.max(np.abs(direction)))
    if dmax == 0.0:
        return 0.0
    vmax = float(np.max(np.abs(values)))
    return REL_STEP * (vmax if vmax > 0.0 else 1.0) / dmax


def _rel_step_ratio(direction: NDArrayFloat, values: NDArrayFloat) -> float:
    """Same as :func:`_rel_step` but the variation is relative cell by cell."""
    dmax = float(np.max(np.abs(direction / values)))
    return 0.0 if dmax == 0.0 else REL_STEP / dmax


def _scratch_flow_model(
    fl_model: FlowModel,
    permeability: NDArrayFloat,
    storage_coefficient: NDArrayFloat | None = None,
) -> FlowModel:
    """Return a shallow copy of the flow model, with other parameters if given."""
    scratch = copy.copy(fl_model)
    scratch.permeability = permeability
    if storage_coefficient is not None:
        scratch.storage_coefficient = storage_coefficient
    return scratch


# ---------------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------------


def get_flow_sources(model: ForwardModel, time_index: int) -> NDArrayFloat:
    """
    Return the flow sources (1/s) of the given time index, before any correction.

    The sources stored in the flow model are modified in place for the constant
    head grid cells. This function recomputes the original ones, as done by the
    forward solver.

    Parameters
    ----------
    model : ForwardModel
        The forward model, solved up to ``time_index`` at least.
    time_index : int
        The time index.

    Returns
    -------
    NDArrayFloat
        Sources with shape (nx, ny, nz).
    """
    time = float(np.sum(model.time_params.ldt[: max(time_index - 1, 0)]))
    return model.get_sources(time, model.grid)[0]


def _transient_flow_residual(
    model: ForwardModel,
    time_index: int,
    dt: float,
    permeability: NDArrayFloat,
    storage_coefficient: NDArrayFloat,
    sources: NDArrayFloat,
    sources_old: NDArrayFloat,
) -> NDArrayFloat:
    """
    Return the residual of the (saturated) transient flow equation of one timestep.

    The heads at the new and at the previous times are the ones of the model. The
    constant head rows are removed.
    """
    fl_model = _scratch_flow_model(model.fl_model, permeability, storage_coefficient)
    builder_next, builder_prev = _assemble_transient_flow_matrices(
        model.grid, fl_model, model.tr_model, time_index
    )
    q_next, q_prev = _add_time_derivative_and_cst_head(
        fl_model, builder_next.tocsc(), builder_prev.tocsc(), dt
    )
    rhs = (
        fl_model.crank_nicolson * sources.ravel(order="F")
        + (1.0 - fl_model.crank_nicolson) * sources_old.ravel(order="F")
    ) / storage_coefficient.ravel(order="F")
    rhs += q_prev @ fl_model.lhead[time_index - 1].ravel(order="F")
    res = rhs - q_next @ fl_model.lhead[time_index].ravel(order="F")
    res[fl_model.cst_head_nn] = 0.0
    return res


def dFhds_transient(
    model: ForwardModel,
    time_index: int,
    dt: float,
    d_permeability: NDArrayFloat | None,
    d_storage_coefficient: NDArrayFloat | None,
) -> NDArrayFloat:
    r"""
    Return the product of the derivative of the flow equation with a direction.

    It is the derivative of the residual (right hand side minus left hand side) of
    the head equation of the timestep ``time_index``, with respect to the
    permeability and to the storage coefficient. The sign is such that the
    sensitivities solve :math:`A z^n = B z^{n-1} + \mathrm{forcing}`.

    Parameters
    ----------
    model : ForwardModel
        The forward model, solved up to ``time_index``.
    time_index : int
        The time index (>= 1).
    dt : float
        The duration of the timestep.
    d_permeability : NDArrayFloat | None
        Direction for the permeability, with shape (nx, ny, nz). None if null.
    d_storage_coefficient : NDArrayFloat | None
        Direction for the storage coefficient, with shape (nx, ny, nz).
        None if null.

    Returns
    -------
    NDArrayFloat
        The forcing, with shape (nx * ny * nz,).
    """
    fl_model = model.fl_model
    out = np.zeros(model.grid.n_grid_cells)
    sources = get_flow_sources(model, time_index)
    sources_old = get_flow_sources(model, time_index - 1)
    kperm = fl_model.permeability
    ss = fl_model.storage_coefficient

    def _res(k: NDArrayFloat, s: NDArrayFloat) -> NDArrayFloat:
        return _transient_flow_residual(
            model, time_index, dt, k, s, sources, sources_old
        )

    if d_permeability is not None:
        eps = _rel_step_ratio(d_permeability, kperm)
        if eps > 0.0:
            out += (
                _res(kperm + eps * d_permeability, ss)
                - _res(kperm - eps * d_permeability, ss)
            ) / (2.0 * eps)
    if d_storage_coefficient is not None:
        eps = _rel_step_ratio(d_storage_coefficient, ss)
        if eps > 0.0:
            out += (
                _res(kperm, ss + eps * d_storage_coefficient)
                - _res(kperm, ss - eps * d_storage_coefficient)
            ) / (2.0 * eps)
    return out


def dFhds_stationary(
    model: ForwardModel, d_permeability: NDArrayFloat | None
) -> NDArrayFloat:
    r"""
    Return the product of the derivative of the stationary flow with a direction.

    The stationary flow equation is :math:`A(K) h = b`. The returned forcing is
    :math:`- \frac{\partial A}{\partial K} v \, h` (without the constant head rows).

    Parameters
    ----------
    model : ForwardModel
        The forward model, solved up to the time 0 at least.
    d_permeability : NDArrayFloat | None
        Direction for the permeability, with shape (nx, ny, nz). None if null.

    Returns
    -------
    NDArrayFloat
        The forcing, with shape (nx * ny * nz,).
    """
    fl_model = model.fl_model
    out = np.zeros(model.grid.n_grid_cells)
    if d_permeability is None:
        return out
    kperm = fl_model.permeability
    eps = _rel_step_ratio(d_permeability, kperm)
    if eps == 0.0:
        return out
    head = fl_model.lhead[0].ravel(order="F")

    def _res(k: NDArrayFloat) -> NDArrayFloat:
        matrix = make_stationary_flow_matrices(
            model.grid, _scratch_flow_model(fl_model, k)
        )
        return -(matrix.tocsc() @ head)

    out += (_res(kperm + eps * d_permeability) - _res(kperm - eps * d_permeability)) / (
        2.0 * eps
    )
    out[fl_model.cst_head_nn] = 0.0
    return out


def get_flow_fields(
    model: ForwardModel,
    permeability: NDArrayFloat,
    head: NDArrayFloat,
    unitflow: NDArrayFloat,
) -> FlowFields:
    """
    Return the darcy velocities, their divergence and the unit flow.

    This is a side effect free version of what the forward solver does after
    the head has been computed (saturated flow).

    Parameters
    ----------
    model : ForwardModel
        The forward model.
    permeability : NDArrayFloat
        Permeability, with shape (nx, ny, nz).
    head : NDArrayFloat
        Head, with shape (nx, ny, nz).
    unitflow : NDArrayFloat
        Flow sources before the correction on the constant head grid cells.

    Returns
    -------
    FlowFields
        The fields.
    """
    scratch = _scratch_flow_model(model.fl_model, permeability)
    scratch.lhead = [head]
    scratch.lu_darcy_x = []
    scratch.lu_darcy_y = []
    scratch.lu_darcy_z = []
    scratch.lu_darcy_div = []
    scratch.lunitflow = [unitflow.copy()]
    compute_u_darcy(scratch, model.tr_model, model.grid, 0)
    compute_u_darcy_div(scratch, model.grid, 0)
    return FlowFields.from_model(scratch, 0)


def dflow_fields(
    model: ForwardModel,
    time_index: int,
    unitflow: NDArrayFloat,
    d_permeability: NDArrayFloat | None,
    d_head: NDArrayFloat,
) -> FlowFields:
    """
    Return the derivative of the flow fields in one direction.

    Parameters
    ----------
    model : ForwardModel
        The forward model, solved up to ``time_index`` at least.
    time_index : int
        The time index.
    unitflow : NDArrayFloat
        Flow sources before the correction on the constant head grid cells.
    d_permeability : NDArrayFloat | None
        Direction for the permeability with shape (nx, ny, nz). None if null.
    d_head : NDArrayFloat
        Sensitivity of the head with shape (nx, ny, nz).

    Returns
    -------
    FlowFields
        The derivatives of the fields.
    """
    kperm = model.fl_model.permeability
    head = model.fl_model.lhead[time_index]

    def _fields(k: NDArrayFloat, h: NDArrayFloat) -> FlowFields:
        return get_flow_fields(model, k, h, unitflow)

    out = FlowFields(
        np.zeros((model.grid.nx + 1, model.grid.ny, model.grid.nz)),
        np.zeros((model.grid.nx, model.grid.ny + 1, model.grid.nz)),
        np.zeros((model.grid.nx, model.grid.ny, model.grid.nz + 1)),
        np.zeros(model.grid.shape),
        np.zeros(model.grid.shape),
    )

    # Affine in the head
    eps = _rel_step(d_head, head)
    if eps > 0.0:
        plus = _fields(kperm, head + eps * d_head)
        minus = _fields(kperm, head - eps * d_head)
        out = out.shifted(plus, 0.5 / eps).shifted(minus, -0.5 / eps)

    if d_permeability is not None:
        eps = _rel_step_ratio(d_permeability, kperm)
        if eps > 0.0:
            plus = _fields(kperm + eps * d_permeability, head)
            minus = _fields(kperm - eps * d_permeability, head)
            out = out.shifted(plus, 0.5 / eps).shifted(minus, -0.5 / eps)
    return out


# ---------------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------------


def _scratch_flow_for_transport(
    fl_model: FlowModel,
    time_index: int,
    fields_old: FlowFields,
    fields: FlowFields,
) -> FlowModel:
    """
    Return a shallow copy of the flow model with given fields at two successive times.

    Only the times ``time_index - 1`` and ``time_index`` are filled.
    """
    scratch = copy.copy(fl_model)
    pad: list = [None] * (time_index - 1)
    scratch.lu_darcy_x = [*pad, fields_old.u_x, fields.u_x]
    scratch.lu_darcy_y = [*pad, fields_old.u_y, fields.u_y]
    scratch.lu_darcy_z = [*pad, fields_old.u_z, fields.u_z]
    scratch.lu_darcy_div = [*pad, fields_old.div, fields.div]
    scratch.lunitflow = [*pad, fields_old.unitflow, fields.unitflow]
    return scratch


def get_transport_matrices(
    model: ForwardModel,
    time_index: int,
    fields_old: FlowFields,
    fields: FlowFields,
    disp: NDArrayFloat,
) -> tuple[sparse.csc_array, sparse.csc_array]:
    """
    Assemble the transport matrices (without the 1/dt term) for given fields.

    Parameters
    ----------
    model : ForwardModel
        The forward model.
    time_index : int
        The time index (>= 1).
    fields_old : FlowFields
        The flow fields at ``time_index - 1``.
    fields : FlowFields
        The flow fields at ``time_index``.
    disp : NDArrayFloat
        The diffusion/dispersion coefficient with shape (nx, ny, nz).

    Returns
    -------
    tuple[sparse.csc_array, sparse.csc_array]
        The matrices of the terms at the next and at the previous time.
    """
    dim = model.grid.n_grid_cells
    builder_next = SparseMatrixBuilder((dim, dim))
    builder_prev = SparseMatrixBuilder((dim, dim))
    fl_model = _scratch_flow_for_transport(
        model.fl_model, time_index, fields_old, fields
    )
    _assemble_transport_matrices(
        model.grid,
        model.tr_model,
        fl_model,
        time_index,
        builder_next,
        builder_prev,
        disp=disp,
    )
    return builder_next.tocsc(), builder_prev.tocsc()


def get_velocity_norm(
    model: ForwardModel, time_index: int, fields: FlowFields
) -> NDArrayFloat:
    """Return the norm of the darcy velocity at the cell centers for given fields."""
    fl_model = copy.copy(model.fl_model)
    fl_model.lu_darcy_x = [np.empty(0)] * time_index + [fields.u_x]
    fl_model.lu_darcy_y = [np.empty(0)] * time_index + [fields.u_y]
    fl_model.lu_darcy_z = [np.empty(0)] * time_index + [fields.u_z]
    return fl_model.get_u_darcy_norm_sample(time_index)


def get_dispersion_for_fields(
    model: ForwardModel,
    time_index: int,
    fields: FlowFields,
    porosity: NDArrayFloat,
    diffusion: NDArrayFloat,
    dispersivity: NDArrayFloat,
) -> NDArrayFloat:
    """Return the diffusion/dispersion coefficient for the given fields and values."""
    return diffusion * porosity + dispersivity * get_velocity_norm(
        model, time_index, fields
    )


def _mat_diff(
    plus: tuple[sparse.csc_array, sparse.csc_array],
    minus: tuple[sparse.csc_array, sparse.csc_array],
    mob: NDArrayFloat,
    mob_old: NDArrayFloat,
    eps: float,
) -> NDArrayFloat:
    """Return the centered difference of ``q_next @ mob - q_prev @ mob_old``."""
    return ((plus[0] - minus[0]) @ mob.T - (plus[1] - minus[1]) @ mob_old.T).T / (
        2.0 * eps
    )


def dFcds(
    model: ForwardModel,
    time_index: int,
    dt: float,
    fields_old: FlowFields,
    fields: FlowFields,
    dfields_old: FlowFields | None,
    dfields: FlowFields | None,
    d_porosity: NDArrayFloat | None,
    d_diffusion: NDArrayFloat | None,
    d_dispersivity: NDArrayFloat | None,
) -> NDArrayFloat:
    r"""
    Return the product of the derivative of the transport equation with a direction.

    It is the derivative of the left hand side of the transport equation

    .. math::
        Q^n c^n - Q^{n-1} c^{n-1} + \frac{\omega}{\Delta t} (\overline{c}^n
        - \overline{c}^{n-1}) = S

    with respect to the porosity, the diffusion, the dispersivity and the flow
    fields (darcy velocities, divergence, sources), the concentrations and the grades
    being fixed.

    Parameters
    ----------
    model : ForwardModel
        The forward model, solved up to ``time_index``.
    time_index : int
        The time index (>= 1).
    dt : float
        The duration of the timestep.
    fields_old : FlowFields
        The flow fields at ``time_index - 1``.
    fields : FlowFields
        The flow fields at ``time_index``.
    dfields_old : FlowFields | None
        The derivatives of the fields at ``time_index - 1`` in the direction.
        None if null.
    dfields : FlowFields | None
        The derivatives of the fields at ``time_index`` in the direction. None if null.
    d_porosity : NDArrayFloat | None
        Direction for the porosity. None if null.
    d_diffusion : NDArrayFloat | None
        Direction for the diffusion. None if null.
    d_dispersivity : NDArrayFloat | None
        Direction for the dispersivity. None if null.

    Returns
    -------
    NDArrayFloat
        The forcing, with shape (n_sp, nx * ny * nz).
    """
    tr_model = model.tr_model
    n_sp = tr_model.n_sp
    shape = model.grid.shape
    mob = tr_model.lmob[time_index].reshape(n_sp, -1, order="F")
    mob_old = tr_model.lmob[time_index - 1].reshape(n_sp, -1, order="F")
    immob = tr_model.limmob[time_index].reshape(n_sp, -1, order="F")
    immob_old = tr_model.limmob[time_index - 1].reshape(n_sp, -1, order="F")
    porosity = tr_model.porosity
    diffusion = tr_model.diffusion
    dispersivity = tr_model.dispersivity

    disp = get_dispersion_for_fields(
        model, time_index, fields, porosity, diffusion, dispersivity
    )
    out = np.zeros((n_sp, model.grid.n_grid_cells))

    # 1) Derivative with respect to the flow fields (the dispersion is fixed)
    d_disp = np.zeros(shape)
    if dfields is not None and dfields_old is not None:
        scale = max(fields.max_abs_velocity(), fields_old.max_abs_velocity())
        dmax = max(dfields.max_abs_velocity(), dfields_old.max_abs_velocity())
        if dmax > 0.0:
            eps = REL_STEP * (scale if scale > 0.0 else 1.0) / dmax
            plus = get_transport_matrices(
                model,
                time_index,
                fields_old.shifted(dfields_old, eps),
                fields.shifted(dfields, eps),
                disp,
            )
            minus = get_transport_matrices(
                model,
                time_index,
                fields_old.shifted(dfields_old, -eps),
                fields.shifted(dfields, -eps),
                disp,
            )
            out += _mat_diff(plus, minus, mob, mob_old, eps)

            # derivative of the dispersion due to the darcy velocity
            disp_plus = get_dispersion_for_fields(
                model,
                time_index,
                fields.shifted(dfields, eps),
                porosity,
                diffusion,
                dispersivity,
            )
            disp_minus = get_dispersion_for_fields(
                model,
                time_index,
                fields.shifted(dfields, -eps),
                porosity,
                diffusion,
                dispersivity,
            )
            d_disp += (disp_plus - disp_minus) / (2.0 * eps)

    # 2) Dispersion direction
    if d_porosity is not None:
        d_disp += diffusion * d_porosity
    if d_diffusion is not None:
        d_disp += porosity * d_diffusion
    if d_dispersivity is not None:
        d_disp += get_velocity_norm(model, time_index, fields) * d_dispersivity
    eps = _rel_step(d_disp, disp)
    if eps > 0.0:
        plus = get_transport_matrices(
            model, time_index, fields_old, fields, disp + eps * d_disp
        )
        minus = get_transport_matrices(
            model, time_index, fields_old, fields, disp - eps * d_disp
        )
        out += _mat_diff(plus, minus, mob, mob_old, eps)

    # 3) Porosity: diagonal terms (1/dt) of the matrices and chemical source term
    if d_porosity is not None:
        dpor = d_porosity.ravel(order="F") / dt
        out += dpor * ((mob - mob_old) + (immob - immob_old))

    return out
