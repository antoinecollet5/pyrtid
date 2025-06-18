"""Provide a reactive transport solver."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.optimize import OptimizeResult

from pyrtid.forward.geochem_utils import (
    get_polish,
    newton,
    solve_with_svd,
    standalone_linesearch,
)
from pyrtid.forward.models import (
    ConstantConcentration,
    GeochemicalParameters,
    TimeParameters,
    TransportModel,
)
from pyrtid.utils import NDArrayFloat, RectilinearGrid
from pyrtid.utils.preconditioner import NoTransform, Preconditioner


def solve_geochem(
    grid: RectilinearGrid,
    tr_model: TransportModel,
    gch_params: GeochemicalParameters,
    time_params: TimeParameters,
    time_index: int,
) -> None:
    r"""
    Compute the mineral dissolution.

    The equation reads:

    .. math::
        \overline{c}_{i}^{n+1} = \overline{c}_{i}^{n} + \Delta t^{n} k_{v} A_{s}
        \overline{c}_{i}^{n} \left( 1 - \dfrac{c_{i}^{n+1}}{K_{s}}\right)
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
    Compute the mineral dissolution.

    The equation reads:

    .. math::
        \overline{c}_{i}^{n+1} = \overline{c}_{i}^{n} + \Delta t^{n} k_{v} A_{s}
        \overline{c}_{i}^{n} \left( 1 - \dfrac{c_{i}^{n+1}}{K_{s}}\right)
    """

    immob1 = tr_model.limmob[time_index - 1][0]
    immob2 = tr_model.limmob[time_index - 1][1]

    # The mobile concentration is from the transport
    dM = get_dM(tr_model, gch_params, time_index, time_params.dt)

    for condition in tr_model.boundary_conditions:
        if isinstance(condition, ConstantConcentration):
            dM[condition.span] = 0.0
        # elif isinstance(condition, ZeroConcGradient):

    assert np.count_nonzero(immob1 + dM < 0) == 0

    # overwrite the grade for species 1
    tr_model.limmob[time_index][0, :, :] = immob1 + dM

    # And for species 2 -> species being consumed
    tr_model.limmob[time_index][1, :, :] = immob2 - gch_params.stocoef * dM


def get_dM(
    tr_model: TransportModel,
    gch_params: GeochemicalParameters,
    time_index: int,
    dt: float,
) -> NDArrayFloat:
    mob1 = tr_model.lmob[time_index][0]
    mob2 = tr_model.lmob[time_index][1]
    immob1 = tr_model.limmob[time_index - 1][0]

    dM = -np.min(
        np.array(
            [
                (
                    -dt
                    * gch_params.kv
                    * gch_params.As
                    * immob1
                    * (1 - mob1 / gch_params.Ks)
                    * mob2
                ),
                immob1,
                mob2 / gch_params.stocoef,
            ]
        ),
        axis=0,
    )

    # Handle special cases: because there might be some negative values in the
    # transport because of the semi-implicit time scheme for advection
    mask = (1 - mob1 / gch_params.Ks) <= 0.0  # (1 - mob1 / Ks) positive: precipitation
    dM[mask] = 0.0
    mask = (1 - mob1 / gch_params.Ks) > 1.0  # (1 - mob1 / Ks) positive: precipitation
    dM[mask] = 0.0
    mask = mob2 <= 0.0
    dM[mask] = 0.0

    return dM


def get_dM_pos(
    tr_model: TransportModel,
    gch_params: GeochemicalParameters,
    time_index: int,
    dt: float,
) -> NDArrayFloat:
    mob1 = tr_model.lmob[time_index][0]
    mob2 = tr_model.lmob[time_index][1]
    immob1 = tr_model.limmob[time_index - 1][0]

    dM_pos = np.argmin(
        np.array(
            [
                (
                    -dt
                    * gch_params.kv
                    * gch_params.As
                    * immob1
                    * (1 - mob1 / gch_params.Ks)
                    * mob2
                ),
                immob1,
                mob2 / gch_params.stocoef,
            ]
        ),
        axis=0,
    )

    return dM_pos


def solve_geochem_implicit(
    grid: RectilinearGrid,
    tr_model: TransportModel,
    gch_params: GeochemicalParameters,
    time_params: TimeParameters,
    time_index: int,
) -> None:
    r"""
    Compute the mineral dissolution.

    The equation reads:

    .. math::
        \overline{c}_{i}^{n+1} = \overline{c}_{i}^{n} + \Delta t^{n} k_{v} A_{s}
        \overline{c}_{i}^{n} \left( 1 - \dfrac{c_{i}^{n+1}}{K_{s}}\right)
    """
    # Mineral grades
    immob_next = tr_model.limmob[time_index]
    immob_prev = tr_model.limmob[time_index - 1]
    # Mobile concentrations
    mob_next = tr_model.lmob[time_index]
    mob_prev = tr_model.lmob[time_index - 1]

    max_chem_iter = 0
    # Loop over x, y, z  = > this is not very efficient and we should find a way to
    # parallelize this or vectorize it at least
    for i in range(grid.nx):
        for j in range(grid.ny):
            for k in range(grid.nz):
                print(i, j, k)

                # Newton-Raphson to solve the geochemical system
                opt_res = solve_geochem_system(
                    mob_next[:, i, j, k],
                    immob_next[:, i, j, k],
                    mob_prev[:, i, j, k],
                    immob_prev[:, i, j, k],
                    atol=1e-20,
                    gch_params=gch_params,
                    is_use_svd=True,
                    is_use_ln=True,
                    dt=time_params.dt,
                )

                tr_model.lmob[time_index][:, i, j, k] = opt_res.x[:2]
                tr_model.limmob[time_index][:, i, j, k] = opt_res.x[2:]

                max_chem_iter = max(max_chem_iter, opt_res.nfev)
    print(max_chem_iter)
    assert np.count_nonzero(mob_next < 0) == 0
    assert np.count_nonzero(immob_next < 0) == 0


def get_phi(
    mob_next: NDArrayFloat, immob_prev: NDArrayFloat, gch_params: GeochemicalParameters
) -> float:
    return (
        gch_params.kv
        * gch_params.As
        * immob_prev[0]
        * mob_next[1]
        * (1 - mob_next[0] / gch_params.Ks)
    )


# Define the vector function P(C)
def F(
    mob_next: NDArrayFloat,
    immob_next: NDArrayFloat,
    mob_prev: NDArrayFloat,
    immob_prev: NDArrayFloat,
    gch_params: GeochemicalParameters,
    dt: float,
) -> NDArrayFloat:
    # mass conservation for species 0 and species 1
    P1, P2 = immob_next + mob_next - immob_prev - mob_prev
    # mass conservation for species 1
    # P2 = immob_next[2] + C_np1[3] - C_n[2] - C_n[3]
    # kinetics
    phi = get_phi(mob_next, immob_prev, gch_params)
    # Change for species 0
    P3 = immob_next[0] - immob_prev[0] - dt * phi
    # change for species 1
    P4 = immob_next[1] - immob_prev[1] + dt * gch_params.stocoef * phi
    return np.array([P1, P2, P3, P4])


# Define the Jacobian matrix of P(C)
def Jacobian(
    mob_next: NDArrayFloat,
    immob_next: NDArrayFloat,
    mob_prev: NDArrayFloat,
    immob_prev: NDArrayFloat,
    gch_params: GeochemicalParameters,
    dt: float,
) -> NDArrayFloat:
    # Initialise a matrix for the Jacobian
    J = np.zeros((4, 4))
    dtKvAs = dt * gch_params.kv * gch_params.As
    # Partial derivatives of P1 with respect to (mob_next, immob_next)
    J[0, 0] = 1.0
    J[0, 1] = 0.0
    J[0, 2] = 1.0
    J[0, 3] = 0.0
    # Partial derivatives of P2 with respect to (mob_next, immob_next)
    J[1, 0] = 0.0
    J[1, 1] = 1.0
    J[1, 2] = 0.0
    J[1, 3] = 1.0
    # Partial derivatives of P3 with respect to (mob_next, immob_next)
    J[2, 0] = dtKvAs * mob_next[1] * immob_prev[0] / gch_params.Ks
    J[2, 1] = -dtKvAs * immob_prev[0] * (1 - mob_next[0] / gch_params.Ks)
    J[2, 2] = 1.0
    J[2, 3] = 0.0
    # Partial derivatives of P4 with respect to (mob_next, immob_next)
    J[3, 0] = -gch_params.stocoef * dtKvAs * mob_next[1] * immob_prev[0] / gch_params.Ks
    J[3, 1] = (
        +dtKvAs
        * gch_params.stocoef
        * (immob_prev[0] * (1 - mob_next[0] / gch_params.Ks))
    )
    J[3, 2] = 0.0
    J[3, 3] = 1.0
    return J


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
    pcd: Preconditioner = NoTransform(),
) -> OptimizeResult:
    # change the variable
    c0 = np.hstack([mob_next, immob_next])
    x0 = pcd(c0)

    def F_wrapper(_C) -> NDArrayFloat:
        return F(_C[0:2], _C[2:], mob_prev, immob_prev, gch_params, dt)

    def jac_wrapper(_C) -> NDArrayFloat:
        return Jacobian(_C[0:2], _C[2:], mob_prev, immob_prev, gch_params, dt)

    class FunctionsWrapper:
        """Wrapper to keep track of the function calls."""

        def __init__(self, x0: NDArrayFloat) -> None:
            """Initialize the instance"""
            # comptors for the number of residuals evaluation and the number
            # of jacobian evaluations
            self.nfev = 0
            self.njev = 0
            self.nhev = 0
            self.x = x0
            self.res_updated = False
            self.jac_updated = False
            self.invjacres_updated = False
            self.residuals = self.get_residuals(self.x)
            self.jac = self.get_jac(self.x)
            self.invjacres = self.get_invjacres(self.x)

        def get_residuals(self, x: NDArrayFloat) -> NDArrayFloat:
            """Get the residuals vector."""
            if not np.array_equal(x, self.x):
                self.update_x(x)

            if self.res_updated:
                # no need to update the residuals since x has not changed
                return self.residuals
            self.nfev += 1
            self.residuals = F_wrapper(pcd.backtransform(x))
            # print(self.residuals)
            self.res_updated = True
            return self.residuals

        def get_jac(self, x) -> NDArrayFloat:
            """
            Get the jacobian matrix of the residuals (for the preconditioned variable).
            """
            if not np.array_equal(x, self.x):
                self.update_x(x)

            if self.jac_updated:
                # no need to update the residuals since x has not changed
                return self.jac

            self.jac = jac_wrapper(pcd.backtransform(x))
            self.njev += 1
            self.jac_updated = True
            return self.jac

        def get_invjacres(self, x: NDArrayFloat) -> NDArrayFloat:
            """
            Get the inverse of the jacobian matrix of the residuals times the residuals.

            This is for the non-preconditioned x.
            """
            if not np.array_equal(x, self.x):
                self.update_x(x)

            if self.invjacres_updated:
                # no need to update the residuals since x has not changed
                return self.invjacres

            # we do not check the update of x because it is done in self.get_jac(x)
            # 1) Compute the Jacobian inverse time the input vector
            if is_use_svd:
                self.invjacres = solve_with_svd(self.get_jac(x), self.get_residuals(x))
            else:
                self.invjacres = np.linalg.inv(self.get_jac(x)).dot(
                    self.get_residuals(x)
                )
            self.nhev += 1
            self.invjacres_updated = True
            return self.invjacres

        def get_invjacvres_pcd(self, x: NDArrayFloat) -> NDArrayFloat:
            """
            Get the inverse of the jacobian matrix of the residuals times the residuals.

            This is for the preconditioned x.
            """
            # Apply the preconditioner
            return pcd.dtransform_vec(pcd.backtransform(x), self.get_invjacres(x))

        def update_x(self, x: NDArrayFloat) -> None:
            """Update the stored x

            Note
            ----
            x is preconditioned.
            """
            # ensure that self.x is a copy of x. Don't store a reference
            # otherwise the memoization doesn't work properly.
            self.x = np.atleast_1d(x).astype(float)
            self.res_updated = False
            self.jac_updated = False
            self.invjacres_updated = False

    # create an instance of the function wrapper
    # TODO: cache system for the residuals
    fw = FunctionsWrapper(x0)

    # define the linesearch or polishing.
    def _ln_obj_func(x: NDArrayFloat) -> float:
        """SSE on residuals."""
        return 0.5 * np.sum(fw.get_residuals(x) ** 2).item()

    def _ln_obj_grad(x: NDArrayFloat) -> NDArrayFloat:
        """Gradient of the above loss function."""
        return pcd.dbacktransform_vec(x, fw.get_jac(x).T @ fw.get_residuals(x))

    def _linesearch_wrapper(
        x: NDArrayFloat, dx: NDArrayFloat, n_iterations: int
    ) -> float:
        # See 9.7.1 LineSearchesandBacktracking in W. H. Press and S. A. Teukolsky.
        # Numerical Recipes 3rd Edition: The Art of Scientific Computing.
        # Cambridge University Press, 2007. isbn: 978-0-521-880688.
        # url: http://nrbook.com.
        # It is the same algorithm used in B.2. Line search strategy from Numerical
        # Recipes. The thesis manuscript of Maxime Jonval.
        _alpha, nfev, ngev, fun, f0, grad = standalone_linesearch(
            x0=x,
            fun=_ln_obj_func,
            grad=_ln_obj_grad,
            d=dx,
            opt_iter=n_iterations,
        )
        if _alpha is not None:
            alpha = _alpha
        else:
            alpha = 1.0

        print(f"alpha = {alpha}")
        print(f"fun = {fun}")
        print(f"f0 = {f0}")

        return alpha

    def _get_polish_wrapper(x: NDArrayFloat, *args) -> float:
        alpha = get_polish(fw.get_invjacres(x), pcd.backtransform(x))
        print(f"alpha = {alpha}")
        return alpha

    def _get_linesearch() -> Optional[Callable]:
        if is_use_ln:
            return _linesearch_wrapper
        elif is_use_polish:
            return _get_polish_wrapper
        return None

    # call newton
    opt_res = newton(
        x0, fw.get_residuals, fw.get_invjacvres_pcd, atol, linesearch=_get_linesearch()
    )
    # apply the preconditionning
    opt_res.x = pcd.backtransform(opt_res.x)
    # compute the residuals
    opt_res.nfev = fw.nfev
    opt_res.njev = fw.njev
    opt_res.hjev = fw.nhev

    return opt_res
