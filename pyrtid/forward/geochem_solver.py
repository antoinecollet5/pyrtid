# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

r"""
Chemistry step of the reactive transport operator splitting.

The geochemical system is made of one dissolving mineral and two mobile species:

.. math::
    \overline{S}_1 + \nu \, S_2 \rightleftharpoons S_1 + \overline{S}_2

where :math:`S_1` and :math:`S_2` are the mobile species (concentrations ``mob[0]``
and ``mob[1]``) and :math:`\overline{S}_1`, :math:`\overline{S}_2` are the immobile
ones (grades ``immob[0]`` and ``immob[1]``). :math:`\nu` is ``stocoef``. The
dissolution rate reads

.. math::
    \phi = k_v A_s \, \overline{c}_1^{\,n} \, c_2 \left(1 - \dfrac{c_1}{K_s}\right)

with :math:`\overline{c}_1^{\,n}` the mineral grade at the beginning of the timestep.
Since :math:`k_v < 0` for a dissolution, :math:`\phi < 0` means that the mineral is
consumed.

Two formulations are available (see
:attr:`pyrtid.forward.GeochemicalParameters.use_explicit_formulation`):

- an explicit one, :func:`solve_geochem_explicit`, where the rate is evaluated with
  the concentrations given by the transport step;
- an implicit one, :func:`solve_geochem_implicit`, where the (small, non-linear)
  chemical system of every grid cell is solved with a Newton-Raphson algorithm. All
  the grid cells are solved at once (vectorized).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import Literal

import numpy as np
from inv_toolbox.utils.preconditioner import NoTransform, Preconditioner
from quickpaver import RectilinearGrid
from scipy.optimize import OptimizeResult

from pyrtid.forward.geochem_utils import (
    backtracking_linesearch,
    get_polish,
    newton,
    solve_with_svd,
)
from pyrtid.forward.models import (
    ConstantConcentration,
    GeochemicalParameters,
    TimeParameters,
    TransportModel,
)
from pyrtid.utils import NDArrayBool, NDArrayFloat, NDArrayInt

logger = logging.getLogger(__name__)

__all__ = [
    "solve_geochem",
    "solve_geochem_explicit",
    "solve_geochem_implicit",
    "solve_geochem_system",
    "get_dM",
    "get_dM_derivatives",
    "get_dM_pos",
    "get_implicit_dM_derivatives",
    "get_phi",
    "F",
    "Jacobian",
    "get_implicit_extent_analytical",
]


def _get_constant_concentration_mask(
    tr_model: TransportModel, shape: tuple[int, ...]
) -> NDArrayBool:
    """Return a boolean mask (grid shape) of the constant concentration cells."""
    mask = np.zeros(shape, dtype=bool)
    for condition in tr_model.boundary_conditions:
        if isinstance(condition, ConstantConcentration):
            mask[condition.span] = True
    return mask


def solve_geochem(
    grid: RectilinearGrid,
    tr_model: TransportModel,
    gch_params: GeochemicalParameters,
    time_params: TimeParameters,
    time_index: int,
) -> None:
    """
    Solve the chemistry for the current timestep.

    The formulation (explicit or implicit) is selected with
    ``gch_params.use_explicit_formulation``. The mineral grades
    (``tr_model.limmob[time_index]``) are updated in place. With the implicit
    formulation, the mobile concentrations (``tr_model.lmob[time_index]``) are
    updated as well.

    Parameters
    ----------
    grid : RectilinearGrid
        The grid.
    tr_model : TransportModel
        The transport model, which holds the concentrations.
    gch_params : GeochemicalParameters
        The geochemical parameters.
    time_params : TimeParameters
        The time parameters (the current timestep is ``time_params.dt``).
    time_index : int
        Index of the current time (must be >= 1).
    """
    if gch_params.use_explicit_formulation:
        solve_geochem_explicit(tr_model, gch_params, time_params, time_index)
    else:
        solve_geochem_implicit(grid, tr_model, gch_params, time_params, time_index)


def solve_geochem_explicit(
    tr_model: TransportModel,
    gch_params: GeochemicalParameters,
    time_params: TimeParameters,
    time_index: int,
) -> None:
    r"""
    Compute the mineral dissolution with an explicit scheme.

    The mobile concentrations are those computed by the transport step. The grade
    of the mineral (species 1) and of the product (species 2) are updated in place
    in ``tr_model.limmob[time_index]``:

    .. math::
        \overline{c}_{1}^{n+1} &= \overline{c}_{1}^{n} + \Delta M \\
        \overline{c}_{2}^{n+1} &= \overline{c}_{2}^{n} - \nu \Delta M

    where :math:`\Delta M \leq 0` is given by :func:`get_dM`. The grades of the
    constant concentration grid cells are not modified.

    Parameters
    ----------
    tr_model : TransportModel
        The transport model.
    gch_params : GeochemicalParameters
        The geochemical parameters.
    time_params : TimeParameters
        The time parameters.
    time_index : int
        Index of the current time (must be >= 1).

    Raises
    ------
    RuntimeError
        If a negative grade is produced (which should never happen, see
        :func:`get_dM`).
    """
    immob_prev = tr_model.limmob[time_index - 1]

    # The mobile concentration is from the transport
    dM = get_dM(tr_model, gch_params, time_index, time_params.dt)

    for condition in tr_model.boundary_conditions:
        if isinstance(condition, ConstantConcentration):
            dM[condition.span] = 0.0

    new_grade = immob_prev[0] + dM
    if np.any(new_grade < 0.0):
        raise RuntimeError(
            f"Negative mineral grade (min={new_grade.min():.3e}) in the explicit "
            f"geochemistry at time index {time_index}."
        )

    # Species 1 -> mineral being dissolved
    tr_model.limmob[time_index][0, :, :] = new_grade
    # And for species 2 -> species being consumed
    tr_model.limmob[time_index][1, :, :] = immob_prev[1] - gch_params.stocoef * dM


def _get_dM_candidates(
    tr_model: TransportModel,
    gch_params: GeochemicalParameters,
    time_index: int,
    dt: float,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """
    Return the 3 candidate dissolved amounts and the arrays they derive from.

    Returns
    -------
    Tuple[NDArrayFloat, ...]
        ``(kinetic, immob1, reagent, fac, mob2)`` with ``kinetic`` the amount
        dissolved according to the kinetics, ``immob1`` the available mineral,
        ``reagent`` the amount of mineral dissolvable with the available reagent
        and ``fac = 1 - mob1 / Ks`` the saturation factor.
    """
    mob1 = tr_model.lmob[time_index][0]
    mob2 = tr_model.lmob[time_index][1]
    immob1 = tr_model.limmob[time_index - 1][0]

    fac = 1.0 - mob1 / gch_params.Ks
    kinetic = -dt * gch_params.kv * gch_params.As * immob1 * fac * mob2
    return kinetic, immob1, mob2 / gch_params.stocoef, fac, mob2


def get_dM(
    tr_model: TransportModel,
    gch_params: GeochemicalParameters,
    time_index: int,
    dt: float,
) -> NDArrayFloat:
    r"""
    Compute the (explicit) variation of the mineral grade over a timestep.

    The amount of dissolved mineral is the minimum of

    - the amount given by the kinetics
      :math:`-\Delta t \, k_v A_s \overline{c}_1^n (1 - c_1 / K_s) c_2`,
    - the amount of mineral available :math:`\overline{c}_1^n`,
    - the amount which can be dissolved with the available reagent
      :math:`c_2 / \nu`.

    Nothing is dissolved if the solution is saturated (:math:`c_1 \geq K_s`), if
    :math:`c_1 < 0` or if :math:`c_2 \leq 0`. Negative concentrations may indeed
    appear with the transport because of the semi-implicit time scheme for the
    advection.

    Parameters
    ----------
    tr_model : TransportModel
        The transport model.
    gch_params : GeochemicalParameters
        The geochemical parameters (``kv`` is expected to be negative).
    time_index : int
        Index of the current time (must be >= 1).
    dt : float
        The timestep in seconds.

    Returns
    -------
    NDArrayFloat
        The variation of the mineral grade (:math:`\leq 0`), with shape
        ``(nx, ny, nz)``.
    """
    kinetic, immob1, reagent, fac, mob2 = _get_dM_candidates(
        tr_model, gch_params, time_index, dt
    )
    dM = -np.minimum(np.minimum(kinetic, immob1), reagent)

    # Special cases: saturated (fac <= 0), negative mob1 (fac > 1) or no reagent.
    no_dissolution = (fac <= 0.0) | (fac > 1.0) | (mob2 <= 0.0)
    dM[no_dissolution] = 0.0
    return dM


def get_dM_pos(
    tr_model: TransportModel,
    gch_params: GeochemicalParameters,
    time_index: int,
    dt: float,
) -> NDArrayInt:
    """
    Return which limitation controls the dissolution in each grid cell.

    Parameters
    ----------
    tr_model : TransportModel
        The transport model.
    gch_params : GeochemicalParameters
        The geochemical parameters.
    time_index : int
        Index of the current time (must be >= 1).
    dt : float
        The timestep in seconds.

    Returns
    -------
    NDArrayFloat
        Integer array with shape ``(nx, ny, nz)``: 0 if the kinetics is limiting,
        1 if the mineral is exhausted, 2 if the reagent is exhausted. Note that the
        special cases of :func:`get_dM` (saturation, negative concentrations) are
        not taken into account.
    """
    kinetic, immob1, reagent, _, _ = _get_dM_candidates(
        tr_model, gch_params, time_index, dt
    )
    return np.argmin(np.array([kinetic, immob1, reagent]), axis=0)


def get_dM_derivatives(
    tr_model: TransportModel,
    gch_params: GeochemicalParameters,
    time_index: int,
    dt: float,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    r"""
    Return the derivatives of the grade variation :math:`\Delta M` of :func:`get_dM`.

    The derivatives are those of the active limitation (kinetics, available mineral
    or available reagent), and are null where nothing is dissolved. The grades of
    the constant concentration grid cells do not vary, so the derivatives are null
    there as well.

    Parameters
    ----------
    tr_model : TransportModel
        The transport model.
    gch_params : GeochemicalParameters
        The geochemical parameters.
    time_index : int
        Index of the current time (must be >= 1).
    dt : float
        The timestep in seconds.

    Returns
    -------
    tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]
        ``(d_dmob, d_dgrade0, d_dgrade1)`` with shapes ``(2, nx, ny, nz)``,
        ``(nx, ny, nz)`` and ``(nx, ny, nz)``:

        - ``d_dmob[i]`` is the derivative with respect to the mobile concentration
          of the species ``i`` at ``time_index``,
        - ``d_dgrade0`` is the derivative with respect to the grade of the mineral at
          ``time_index - 1``,
        - ``d_dgrade1`` is the derivative with respect to the grade of the product
          at ``time_index - 1``, which is always null.
    """
    kinetic, immob1, reagent, fac, mob2 = _get_dM_candidates(
        tr_model, gch_params, time_index, dt
    )
    mob1 = tr_model.lmob[time_index][0]
    shape = mob1.shape
    d_dmob = np.zeros((2, *shape))
    d_dgrade0 = np.zeros(shape)
    d_dgrade1 = np.zeros(shape)

    # Which limitation is active (see get_dM_pos)
    pos = np.argmin(np.array([kinetic, immob1, reagent]), axis=0)

    # 1) the kinetics: dM = dt * kv * As * grade * (1 - mob1 / Ks) * mob2
    coef = dt * gch_params.kv * gch_params.As
    is_kin = pos == 0
    d_dmob[0] = np.where(is_kin, -coef * immob1 * mob2 / gch_params.Ks, 0.0)
    d_dmob[1] = np.where(is_kin, coef * immob1 * fac, 0.0)
    d_dgrade0 = np.where(is_kin, coef * fac * mob2, 0.0)

    # 2) the mineral is exhausted: dM = -grade
    d_dgrade0 = np.where(pos == 1, -1.0, d_dgrade0)

    # 3) the reagent is exhausted: dM = -mob2 / stocoef
    d_dmob[1] = np.where(pos == 2, -1.0 / gch_params.stocoef, d_dmob[1])

    # Special cases of get_dM: nothing is dissolved
    no_dissolution = (fac <= 0.0) | (fac > 1.0) | (mob2 <= 0.0)
    no_dissolution |= _get_constant_concentration_mask(tr_model, shape)
    d_dmob[:, no_dissolution] = 0.0
    d_dgrade0[no_dissolution] = 0.0

    return d_dmob, d_dgrade0, d_dgrade1


def get_implicit_dM_derivatives(
    tr_model: TransportModel,
    gch_params: GeochemicalParameters,
    time_index: int,
    dt: float,
    tol: float = 1e-3,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayInt, int]:
    r"""
    Return the derivatives of the grade variation of the implicit chemistry.

    The timestep must have been solved, and the fixed point iterations between the
    transport and the chemistry must have converged. In that case, the final
    concentrations are the transported ones, and, as long as the extent
    :math:`\xi` of the dissolution is not bounded, :math:`\Delta M = - \xi` is the
    local function of the concentrations and of the grades at the previous time
    (the same as with the explicit chemistry, but precipitation is allowed).

    Otherwise, the extent is bounded by the physical limits (see
    :func:`solve_geochem_implicit`) and the active limitation is identified from the
    results:

    - the mineral is exhausted: :math:`\Delta M = -\overline{c}_1^{n-1}`,
    - the product is exhausted: :math:`\Delta M = \overline{c}_2^{n-1} / \nu`,
    - the reagent (species 2 for the dissolution, species 1 for the precipitation)
      is exhausted: the concentration of the species is null (pinned), and the
      grade variation is determined by the transport.

    Parameters
    ----------
    tr_model : TransportModel
        The transport model.
    gch_params : GeochemicalParameters
        The geochemical parameters.
    time_index : int
        Index of the current time (must be >= 1).
    dt : float
        The timestep in seconds.
    tol : float, optional
        Relative tolerance used to identify the active limitation, by default 1e-3.

    Returns
    -------
    tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayInt, int]
        ``(d_dmob, d_dgrade0, d_dgrade1, pinned, n_unknown)``:

        - ``d_dmob`` with shape ``(2, nx, ny, nz)``: derivative of :math:`\Delta M`
          with respect to the mobile concentrations at ``time_index``,
        - ``d_dgrade0`` and ``d_dgrade1`` with shape ``(nx, ny, nz)``: derivatives
          with respect to the grades of the species 1 and 2 at ``time_index - 1``,
        - ``pinned`` with shape ``(nx, ny, nz)``: 0 if the grid cell is not pinned,
          1 (resp. 2) if the concentration of the species 1 (resp. 2) is null,
        - ``n_unknown``: number of grid cells where the active limitation could not
          be identified (the unbounded case is assumed there).
    """
    mob = tr_model.lmob[time_index]
    c1, c2 = mob[0], mob[1]
    g1_prev, g2_prev = tr_model.limmob[time_index - 1]
    g1 = tr_model.limmob[time_index][0]
    nu = gch_params.stocoef
    coef = dt * gch_params.kv * gch_params.As
    fac = 1.0 - c1 / gch_params.Ks

    extent = g1_prev - g1  # observed extent
    extent_free = -coef * g1_prev * c2 * fac
    scale = np.maximum(np.abs(extent), np.abs(extent_free))
    scale = np.maximum(scale, 1e-12 * np.maximum(np.abs(g1_prev), 1e-30))

    is_free = np.abs(extent - extent_free) <= tol * scale
    is_mineral = ~is_free & (
        np.abs(extent - g1_prev) <= tol * np.maximum(np.abs(g1_prev), scale)
    )
    is_product = ~is_free & ~is_mineral
    is_product &= np.abs(extent + g2_prev / nu) <= tol * np.maximum(
        np.abs(g2_prev / nu), scale
    )
    c1_zero = np.abs(c1) <= 1e-8 * max(float(np.max(np.abs(c1))), 1e-300)
    c2_zero = np.abs(c2) <= 1e-8 * max(float(np.max(np.abs(c2))), 1e-300)
    other = ~is_free & ~is_mineral & ~is_product
    pinned = np.zeros(c1.shape, dtype=int)
    pinned[other & (extent < 0.0) & c1_zero] = 1
    pinned[other & (extent > 0.0) & c2_zero] = 2
    n_unknown = int(np.count_nonzero(other & (pinned == 0)))

    smooth = ~is_mineral & ~is_product & (pinned == 0)  # free (or unknown)
    d_dmob = np.zeros((2, *c1.shape))
    d_dmob[0] = np.where(smooth, -coef * g1_prev * c2 / gch_params.Ks, 0.0)
    d_dmob[1] = np.where(smooth, coef * g1_prev * fac, 0.0)
    d_dgrade0 = np.where(smooth, coef * fac * c2, 0.0)
    d_dgrade0 = np.where(is_mineral, -1.0, d_dgrade0)
    d_dgrade1 = np.where(is_product, 1.0 / nu, 0.0)

    # The grades of the constant concentration grid cells do not vary
    mask = _get_constant_concentration_mask(tr_model, c1.shape)
    d_dmob[:, mask] = 0.0
    d_dgrade0[mask] = 0.0
    d_dgrade1[mask] = 0.0
    pinned[mask] = 0
    return d_dmob, d_dgrade0, d_dgrade1, pinned, n_unknown


def get_implicit_extent_analytical(
    mob_prev: NDArrayFloat,
    immob_prev: NDArrayFloat,
    gch_params: GeochemicalParameters,
    dt: float,
) -> NDArrayFloat:
    r"""
    Solve the implicit chemical system in closed form.

    With the extent of dissolution :math:`\xi = -\Delta t \, \phi` (mol/kg of mineral
    dissolved), the conservation laws give :math:`c_1 = c_1^0 + \xi` and
    :math:`c_2 = c_2^0 - \nu \xi`. The implicit kinetics then reduces to a scalar
    quadratic equation in :math:`\phi`:

    .. math::
        \dfrac{A \nu \Delta t^2}{K_s} \phi^2
        + \left[A \Delta t \left(\dfrac{c_2^0}{K_s} + \nu w\right) - 1\right] \phi
        + A c_2^0 w = 0

    with :math:`A = k_v A_s \overline{c}_1^{\,n}` and :math:`w = 1 - c_1^0 / K_s`.
    The physical root is the one tending to the explicit solution when
    :math:`\Delta t \rightarrow 0`.

    Parameters
    ----------
    mob_prev : NDArrayFloat
        Mobile concentrations before the reaction (after the transport), with shape
        (2, ...).
    immob_prev : NDArrayFloat
        Mineral grades at the beginning of the timestep, with shape (2, ...). Only
        the first species is used.
    gch_params : GeochemicalParameters
        The geochemical parameters.
    dt : float
        The timestep in seconds.

    Returns
    -------
    NDArrayFloat
        The extent of dissolution :math:`\xi`, with the shape of ``mob_prev[0]``.
        It is NaN where the quadratic equation has no real root.
    """
    s = gch_params.stocoef
    A = gch_params.kv * gch_params.As * immob_prev[0]
    w = 1.0 - mob_prev[0] / gch_params.Ks
    c20 = mob_prev[1]

    a = A * s * dt**2 / gch_params.Ks
    b = A * dt * (c20 / gch_params.Ks + s * w) - 1.0
    c = A * c20 * w

    disc = b * b - 4.0 * a * c
    is_valid = disc >= 0.0
    # numerically stable quadratic formula: the physical root is c / q
    q = -0.5 * (b + np.copysign(np.sqrt(np.where(is_valid, disc, 0.0)), b))
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(q != 0.0, c / q, 0.0)
    return np.where(is_valid, -dt * phi, np.nan)


def get_phi(
    mob_next: NDArrayFloat, immob_prev: NDArrayFloat, gch_params: GeochemicalParameters
) -> NDArrayFloat:
    r"""
    Return the dissolution rate :math:`\phi` (mol/kg/s).

    Parameters
    ----------
    mob_next : NDArrayFloat
        Mobile concentrations at the end of the timestep, with shape (2, ...).
    immob_prev : NDArrayFloat
        Grades at the beginning of the timestep, with shape (2, ...).
    gch_params : GeochemicalParameters
        The geochemical parameters.
    """
    return (
        gch_params.kv
        * gch_params.As
        * immob_prev[0]
        * mob_next[1]
        * (1 - mob_next[0] / gch_params.Ks)
    )


def F(
    mob_next: NDArrayFloat,
    immob_next: NDArrayFloat,
    mob_prev: NDArrayFloat,
    immob_prev: NDArrayFloat,
    gch_params: GeochemicalParameters,
    dt: float,
) -> NDArrayFloat:
    """
    Return the residuals of the implicit chemical system.

    The system is made of the mass conservation of the two species (the sum of the
    mobile and immobile parts is unchanged by the reaction) and of the kinetics of
    the two immobile species.

    All inputs may have trailing dimensions (e.g. one column per grid cell), in
    which case the residuals are computed for all of them at once.

    Parameters
    ----------
    mob_next : NDArrayFloat
        Unknown mobile concentrations, with shape (2, ...).
    immob_next : NDArrayFloat
        Unknown grades, with shape (2, ...).
    mob_prev : NDArrayFloat
        Mobile concentrations *before the reaction*, with shape (2, ...).
    immob_prev : NDArrayFloat
        Grades *before the reaction* (beginning of the timestep), with shape
        (2, ...).
    gch_params : GeochemicalParameters
        The geochemical parameters.
    dt : float
        The timestep in seconds.

    Returns
    -------
    NDArrayFloat
        The residuals, with shape (4, ...).
    """
    # mass conservation for species 0 and species 1
    P1, P2 = immob_next + mob_next - immob_prev - mob_prev
    # kinetics
    phi = get_phi(mob_next, immob_prev, gch_params)
    # Change for species 0
    P3 = immob_next[0] - immob_prev[0] - dt * phi
    # change for species 1
    P4 = immob_next[1] - immob_prev[1] + dt * gch_params.stocoef * phi
    return np.array([P1, P2, P3, P4])


def Jacobian(
    mob_next: NDArrayFloat,
    immob_next: NDArrayFloat,
    mob_prev: NDArrayFloat,
    immob_prev: NDArrayFloat,
    gch_params: GeochemicalParameters,
    dt: float,
) -> NDArrayFloat:
    """
    Return the Jacobian of :func:`F` with respect to ``(mob_next, immob_next)``.

    The arguments are the same as for :func:`F`.

    Returns
    -------
    NDArrayFloat
        The Jacobian, with shape (4, 4, ...): the first two axes are the residuals
        and the unknowns ``(mob_0, mob_1, immob_0, immob_1)``.
    """
    J = np.zeros((4, 4, *np.shape(mob_next)[1:]))
    dtKvAs = dt * gch_params.kv * gch_params.As
    stocoef = gch_params.stocoef
    # d(P3)/d(mob_0), d(P3)/d(mob_1)
    dP3_dm0 = dtKvAs * mob_next[1] * immob_prev[0] / gch_params.Ks
    dP3_dm1 = -dtKvAs * immob_prev[0] * (1 - mob_next[0] / gch_params.Ks)

    # P1: mass conservation of species 0
    J[0, 0] = 1.0
    J[0, 2] = 1.0
    # P2: mass conservation of species 1
    J[1, 1] = 1.0
    J[1, 3] = 1.0
    # P3: kinetics of the immobile species 0
    J[2, 0] = dP3_dm0
    J[2, 1] = dP3_dm1
    J[2, 2] = 1.0
    # P4: kinetics of the immobile species 1
    J[3, 0] = -stocoef * dP3_dm0
    J[3, 1] = -stocoef * dP3_dm1
    J[3, 3] = 1.0
    return J


def _solve_batched(J: NDArrayFloat, rhs: NDArrayFloat) -> NDArrayFloat:
    """Solve ``J[i] x[i] = rhs[i]`` for a stack of matrices ``J`` (n, 4, 4)."""
    try:
        return np.linalg.solve(J, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        # At least one singular Jacobian -> minimum-norm least-squares solution
        return (np.linalg.pinv(J) @ rhs[..., None])[..., 0]


def _backtrack_batch(
    xa: NDArrayFloat,
    dx: NDArrayFloat,
    x_new: NDArrayFloat,
    F_new: NDArrayFloat,
    Fa: NDArrayFloat,
    accept: NDArrayFloat,
    mob_prev: NDArrayFloat,
    immob_prev: NDArrayFloat,
    gch_params: GeochemicalParameters,
    dt: float,
    c1: float = 1e-4,
    max_backtrack: int = 20,
) -> None:
    """
    Cell-wise Armijo backtracking, updating ``x_new`` and ``F_new`` in place.

    The step of a cell is halved until the least-squares objective
    :math:`0.5 \\lVert F \\rVert^2` is sufficiently decreased. Since ``dx`` is the
    Newton direction, the slope is :math:`-\\lVert F \\rVert^2`. Cells flagged in
    ``accept`` (Newton increment already negligible) keep the full step.
    """
    f0 = 0.5 * np.sum(Fa**2, axis=0)
    f_new = 0.5 * np.sum(F_new**2, axis=0)
    alpha = np.ones(xa.shape[1])
    bad = ~(accept | (np.isfinite(f_new) & (f_new <= f0 * (1.0 - 2.0 * c1))))

    for _ in range(max_backtrack):
        if not bad.any():
            break
        idx = np.flatnonzero(bad)
        alpha[idx] *= 0.5
        xt = xa[:, idx] - alpha[idx] * dx[:, idx]
        Ft = F(xt[:2], xt[2:], mob_prev[:, idx], immob_prev[:, idx], gch_params, dt)
        x_new[:, idx] = xt
        F_new[:, idx] = Ft
        ft = 0.5 * np.sum(Ft**2, axis=0)
        ok = np.isfinite(ft) & (ft <= f0[idx] * (1.0 - 2.0 * c1 * alpha[idx]))
        bad[idx[ok]] = False


def _newton_batch(
    x0: NDArrayFloat,
    mob_prev: NDArrayFloat,
    immob_prev: NDArrayFloat,
    gch_params: GeochemicalParameters,
    dt: float,
    atol: float,
    rtol: float,
    max_iter: int,
    is_use_linesearch: bool,
) -> OptimizeResult:
    """
    Newton-Raphson on many independent chemical systems at once.

    Parameters
    ----------
    x0 : NDArrayFloat
        Initial guess ``(mob_0, mob_1, immob_0, immob_1)`` with shape (4, n).
    mob_prev, immob_prev : NDArrayFloat
        Mobile concentrations and grades before the reaction, with shape (2, n).

    Returns
    -------
    OptimizeResult
        With ``x`` the solutions (4, n), ``converged`` the per-cell convergence
        flags, ``nit`` the maximum number of iterations among cells, and ``success``
        whether all the cells converged.
    """
    x = np.array(x0, dtype=float)
    residuals = F(x[:2], x[2:], mob_prev, immob_prev, gch_params, dt)
    n_iter = np.zeros(x.shape[1], dtype=int)
    done = np.max(np.abs(residuals), axis=0) <= atol  # no iteration needed
    converged = done.copy()

    for _ in range(max_iter):
        idx = np.flatnonzero(~done)
        if idx.size == 0:
            break
        xa, Fa = x[:, idx], residuals[:, idx]
        mp, ip = mob_prev[:, idx], immob_prev[:, idx]

        J = np.moveaxis(Jacobian(xa[:2], xa[2:], mp, ip, gch_params, dt), -1, 0)
        dx = _solve_batched(J, Fa.T).T

        # Cells with a non-finite increment can't be solved -> stop for them
        is_finite = np.all(np.isfinite(dx), axis=0)
        dx = np.where(is_finite, dx, 0.0)
        # The tolerance is relative to the largest unknown of the cell: the round-off
        # errors of the residuals scale with the total masses, not with the (possibly
        # much smaller) concentrations. Newton converges quadratically, hence the
        # error after the step is far below the increment itself.
        scale = np.max(np.abs(xa), axis=0)
        small = is_finite & np.all(np.abs(dx) <= atol + rtol * scale, axis=0)

        x_new = xa - dx
        F_new = F(x_new[:2], x_new[2:], mp, ip, gch_params, dt)
        if is_use_linesearch:
            _backtrack_batch(
                xa,
                dx,
                x_new,
                F_new,
                Fa,
                np.logical_or(small, ~is_finite),
                mp,
                ip,
                gch_params,
                dt,
            )
        x[:, idx] = x_new
        residuals[:, idx] = F_new
        n_iter[idx] += 1

        converged[idx] = small
        done[idx] = small | ~is_finite

    return OptimizeResult(
        x=x,
        converged=converged,
        nit=int(n_iter.max(initial=0)),
        success=bool(converged.all()),
    )


def _solve_extent(
    mob_prev: NDArrayFloat,
    immob_prev: NDArrayFloat,
    mob_guess: NDArrayFloat,
    immob_guess: NDArrayFloat,
    gch_params: GeochemicalParameters,
    dt: float,
    method: str,
    atol: float,
    rtol: float,
    max_iter: int,
    is_use_linesearch: bool,
) -> tuple[NDArrayFloat, int, int]:
    """
    Solve the implicit chemistry and return the extent of dissolution.

    Returns
    -------
    Tuple[NDArrayFloat, int, int]
        The extent of dissolution (n,), the maximum number of Newton iterations and
        the number of cells for which the solver failed (their extent is 0).
    """
    n_cells = mob_prev.shape[1]
    extent = np.full(n_cells, np.nan)
    n_iter = 0

    if method == "analytical":
        extent = get_implicit_extent_analytical(mob_prev, immob_prev, gch_params, dt)
    elif method != "newton":
        raise ValueError(f"Unknown method '{method}'. Use 'newton' or 'analytical'.")

    # Newton for all the cells (or only for those without analytical solution)
    todo = np.flatnonzero(np.isnan(extent))
    n_failed = 0
    if todo.size > 0:
        res = _newton_batch(
            np.vstack([mob_guess[:, todo], immob_guess[:, todo]]),
            mob_prev[:, todo],
            immob_prev[:, todo],
            gch_params,
            dt,
            atol,
            rtol,
            max_iter,
            is_use_linesearch,
        )
        n_iter = res.nit
        # the extent is deduced from the grade of the dissolving mineral
        extent[todo] = immob_prev[0, todo] - res.x[2]
        n_failed = int(np.count_nonzero(~res.converged))

    # Safeguard against non-finite values (set no reaction)
    not_finite = ~np.isfinite(extent)
    n_failed += int(np.count_nonzero(not_finite & ~np.isnan(extent)))
    extent[not_finite] = 0.0
    return extent, n_iter, n_failed


def solve_geochem_implicit(
    grid: RectilinearGrid,
    tr_model: TransportModel,
    gch_params: GeochemicalParameters,
    time_params: TimeParameters,
    time_index: int,
    method: Literal["newton", "analytical"] = "newton",
    atol: float = 1e-20,
    rtol: float = 1e-10,
    max_iter: int = 50,
    is_use_linesearch: bool = True,
) -> int:
    r"""
    Compute the mineral dissolution with an implicit scheme.

    The chemical system of every grid cell is solved, all at once, with a
    vectorized Newton-Raphson algorithm. The unknowns are the mobile concentrations
    and the grades at the end of the timestep, which satisfy

    - the conservation of the total mass of each species during the reaction. The
      reference total is the one *after the transport*:
      :math:`c^{n+1} + \overline{c}^{n+1} = c^{T} + \overline{c}^{k}`, with
      :math:`c^T` the concentration computed by the transport (stored in
      ``tr_model.lmob[time_index]``) and :math:`\overline{c}^{k}` the grades of the
      previous fixed-point iteration (stored in ``tr_model.limmob[time_index]``);
    - the kinetics, integrated from the grades at the beginning of the timestep
      :math:`\overline{c}^{n}` (``tr_model.limmob[time_index - 1]``).

    The solution is then limited to physically admissible values: the dissolved
    amount cannot exceed the available mineral nor the available reagent, and the
    precipitated amount cannot exceed the available dissolved species nor the
    available product. This only matters when the timestep is large compared to the
    reaction time. The mass is conserved in any case.

    ``tr_model.lmob[time_index]`` and ``tr_model.limmob[time_index]`` are updated in
    place. The grid cells with a constant concentration are not modified.

    Parameters
    ----------
    grid : RectilinearGrid
        The grid.
    tr_model : TransportModel
        The transport model.
    gch_params : GeochemicalParameters
        The geochemical parameters.
    time_params : TimeParameters
        The time parameters (the current timestep is ``time_params.dt``).
    time_index : int
        Index of the current time (must be >= 1).
    method : {"newton", "analytical"}, optional
        ``"newton"`` solves the system with the Newton-Raphson algorithm.
        ``"analytical"`` uses the closed form solution of this specific system, see
        :func:`get_implicit_extent_analytical` (the Newton-Raphson algorithm is used
        for the cells without real solution). By default ``"newton"``, which is
        the generic approach.
    atol : float, optional
        Absolute tolerance on the Newton increment, by default 1e-20.
    rtol : float, optional
        Relative tolerance on the Newton increment, by default 1e-10.
    max_iter : int, optional
        Maximum number of Newton iterations, by default 50.
    is_use_linesearch : bool, optional
        Whether to damp the Newton steps with a backtracking line search, by
        default True.

    Returns
    -------
    int
        The maximum number of Newton iterations required among the grid cells.

    Warns
    -----
    UserWarning
        If the solver failed for some grid cells. The mass is conserved but the
        kinetics is not satisfied there. Reduce the timestep.
    """
    n_sp = tr_model.n_sp
    dt = time_params.dt
    mob = tr_model.lmob[time_index]
    immob = tr_model.limmob[time_index]
    immob_prev_full = tr_model.limmob[time_index - 1]
    shape = mob.shape[1:]

    free = np.flatnonzero(~_get_constant_concentration_mask(tr_model, shape).ravel())

    mob_flat = mob.reshape(n_sp, -1)[:, free]  # transported concentrations
    immob_flat = immob.reshape(n_sp, -1)[:, free]  # grades of last iteration
    immob_prev = immob_prev_full.reshape(n_sp, -1)[:, free]

    # Concentrations that the cells would have without reaction (after transport)
    mob_prev = mob_flat + immob_flat - immob_prev

    extent, n_iter, n_failed = _solve_extent(
        mob_prev,
        immob_prev,
        mob_flat,
        immob_flat,
        gch_params,
        dt,
        method,
        atol,
        rtol,
        max_iter,
        is_use_linesearch,
    )
    if n_failed > 0:
        warnings.warn(
            f"The implicit geochemistry did not converge in {n_failed} grid cell(s) at "
            f"time index {time_index}. Consider reducing the timestep.",
            stacklevel=2,
        )

    # Physical limits of the extent of dissolution (lo <= 0 <= hi)
    stocoef = gch_params.stocoef
    hi = np.maximum(np.minimum(immob_prev[0], mob_prev[1] / stocoef), 0.0)
    lo = -np.maximum(np.minimum(mob_prev[0], immob_prev[1] / stocoef), 0.0)
    extent = np.clip(extent, lo, hi)

    # Conservative update
    new_mob = mob_prev + np.vstack([extent, -stocoef * extent])
    new_immob = immob_prev + np.vstack([-extent, stocoef * extent])

    out_mob = np.array(mob.reshape(n_sp, -1))
    out_mob[:, free] = new_mob
    out_immob = np.array(immob_prev_full.reshape(n_sp, -1))  # constant conc. cells
    out_immob[:, free] = new_immob
    tr_model.lmob[time_index][...] = out_mob.reshape(n_sp, *shape)
    tr_model.limmob[time_index][...] = out_immob.reshape(n_sp, *shape)

    logger.debug(
        "Implicit geochemistry (%s): max %d Newton iteration(s) at time index %d.",
        method,
        n_iter,
        time_index,
    )
    return n_iter


class _LocalNewtonProblem:
    """
    Residuals and Jacobian of the chemical system of one grid cell.

    The values are memoized: they are only recomputed when ``x`` changes. All the
    methods take the preconditioned unknowns ``x`` and handle the change of variable
    with the preconditioner ``pcd``.
    """

    def __init__(
        self,
        mob_prev: NDArrayFloat,
        immob_prev: NDArrayFloat,
        gch_params: GeochemicalParameters,
        dt: float,
        pcd: Preconditioner,
        is_use_svd: bool,
    ) -> None:
        self.mob_prev = mob_prev
        self.immob_prev = immob_prev
        self.gch_params = gch_params
        self.dt = dt
        self.pcd = pcd
        self.is_use_svd = is_use_svd
        # counters: residuals, jacobians and Newton increments evaluations
        self.nfev = 0
        self.njev = 0
        self.nhev = 0
        self._x: NDArrayFloat | None = None
        self._residuals: NDArrayFloat | None = None
        self._jac: NDArrayFloat | None = None
        self._invjacres: NDArrayFloat | None = None

    def _update_x(self, x: NDArrayFloat) -> None:
        if self._x is not None and np.array_equal(x, self._x):
            return
        # a copy is stored (not a reference), otherwise the memoization is broken
        self._x = np.array(x, dtype=float)
        self._residuals = self._jac = self._invjacres = None

    def get_residuals(self, x: NDArrayFloat) -> NDArrayFloat:
        """Get the residuals vector."""
        self._update_x(x)
        if self._residuals is None:
            self.nfev += 1
            assert self._x is not None
            c = self.pcd.backtransform(self._x)
            self._residuals = F(
                c[:2], c[2:], self.mob_prev, self.immob_prev, self.gch_params, self.dt
            )
        return self._residuals

    def get_jac(self, x: NDArrayFloat) -> NDArrayFloat:
        """Get the Jacobian of the residuals (with respect to the non-preconditioned
        unknowns)."""
        self._update_x(x)
        if self._jac is None:
            self.njev += 1
            assert self._x is not None
            c = self.pcd.backtransform(self._x)
            self._jac = Jacobian(
                c[:2], c[2:], self.mob_prev, self.immob_prev, self.gch_params, self.dt
            )
        return self._jac

    def get_invjacres(self, x: NDArrayFloat) -> NDArrayFloat:
        """Get the Newton increment (non-preconditioned): J^{-1} F."""
        self._update_x(x)
        if self._invjacres is None:
            self.nhev += 1
            jac, res = self.get_jac(x), self.get_residuals(x)
            self._invjacres = (
                solve_with_svd(jac, res)
                if self.is_use_svd
                else np.linalg.solve(jac, res)
            )
        return self._invjacres

    def get_invjacres_pcd(self, x: NDArrayFloat) -> NDArrayFloat:
        """Get the Newton increment for the preconditioned unknowns."""
        return self.pcd.dtransform_vec(self.pcd.backtransform(x), self.get_invjacres(x))

    def objective(self, x: NDArrayFloat) -> float:
        """Sum of squared residuals (halved)."""
        return 0.5 * float(np.sum(self.get_residuals(x) ** 2))

    def gradient(self, x: NDArrayFloat) -> NDArrayFloat:
        """Gradient of :meth:`objective` with respect to the preconditioned x."""
        return self.pcd.dbacktransform_vec(x, self.get_jac(x).T @ self.get_residuals(x))


def solve_geochem_system(
    mob_next: NDArrayFloat,
    immob_next: NDArrayFloat,
    mob_prev: NDArrayFloat,
    immob_prev: NDArrayFloat,
    gch_params: GeochemicalParameters,
    dt: float,
    atol: float = 1e-15,
    is_use_svd: bool = False,
    is_use_ln: bool = False,
    is_use_polish: bool = False,
    pcd: Preconditioner | None = None,
    rtol: float = 1e-12,
    max_iter: int = 100,
) -> OptimizeResult:
    """
    Solve the chemical system of *one* grid cell with a Newton-Raphson algorithm.

    This is the generic (and slower) counterpart of the vectorized solver used by
    :func:`solve_geochem_implicit`. It supports preconditioning (change of
    variable), a truncated-SVD linear solver, a line search and a polishing of the
    Newton step.

    Parameters
    ----------
    mob_next : NDArrayFloat
        Initial guess of the mobile concentrations at the end of the timestep, with
        shape (2,).
    immob_next : NDArrayFloat
        Initial guess of the grades at the end of the timestep, with shape (2,).
    mob_prev : NDArrayFloat
        Mobile concentrations before the reaction, with shape (2,).
    immob_prev : NDArrayFloat
        Grades before the reaction (beginning of the timestep), with shape (2,).
    gch_params : GeochemicalParameters
        The geochemical parameters.
    dt : float
        The timestep in seconds.
    atol : float, optional
        Absolute tolerance on the residuals and on the Newton increment, by default
        1e-15.
    is_use_svd : bool, optional
        Whether to solve the linear systems with a truncated SVD, which is robust to
        singular Jacobians, by default False.
    is_use_ln : bool, optional
        Whether to use a backtracking line search, by default False.
    is_use_polish : bool, optional
        Whether to polish the Newton step (only if ``is_use_ln`` is False), by
        default False.
    pcd : Optional[Preconditioner], optional
        Change of variable applied to the unknowns, by default None (no change).
    rtol : float, optional
        Relative tolerance on the Newton increment, by default 1e-12.
    max_iter : int, optional
        Maximum number of iterations, by default 100.

    Returns
    -------
    OptimizeResult
        The solution ``x = (mob_0, mob_1, immob_0, immob_1)`` (non-preconditioned),
        ``success`` and the counters ``nit``, ``nfev`` (residuals), ``njev``
        (Jacobian) and ``nhev`` (linear solves).
    """
    pcd = NoTransform() if pcd is None else pcd
    x0 = pcd(np.hstack([mob_next, immob_next]))
    problem = _LocalNewtonProblem(mob_prev, immob_prev, gch_params, dt, pcd, is_use_svd)

    def _linesearch(x: NDArrayFloat, dx: NDArrayFloat, n_iterations: int) -> float:
        # The Newton update is x - alpha * dx, hence the descent direction is -dx
        d = -dx
        alpha = backtracking_linesearch(
            problem.objective,
            x,
            d,
            problem.objective(x),
            float(problem.gradient(x) @ d),
            alpha_max=1.0,
        )
        return 1.0 if alpha is None else alpha

    def _polish(x: NDArrayFloat, *args) -> NDArrayFloat:
        return get_polish(problem.get_invjacres(x), pcd.backtransform(x))

    linesearch: Callable | None = None
    if is_use_ln:
        linesearch = _linesearch
    elif is_use_polish:
        linesearch = _polish

    opt_res = newton(
        x0,
        problem.get_residuals,
        problem.get_invjacres_pcd,
        atol,
        linesearch=linesearch,
        rtol=rtol,
        max_iter=max_iter,
    )
    opt_res.x = pcd.backtransform(opt_res.x)
    opt_res.nfev = problem.nfev
    opt_res.njev = problem.njev
    opt_res.nhev = problem.nhev
    return opt_res
