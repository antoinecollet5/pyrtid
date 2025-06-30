from typing import Tuple

import numpy as np
import scipy as sp
from numpy.typing import ArrayLike

from pyrtid.utils import NDArrayFloat


def _get_curvature(
    interp_loss_ls: NDArrayFloat,
    interp_loss_reg: NDArrayFloat,
    is_logspace: bool = False,
) -> NDArrayFloat:
    if is_logspace:
        dx_dt = np.gradient(np.log(interp_loss_ls)[2:-2])
        dy_dt = np.gradient(np.log(interp_loss_reg)[2:-2])
    else:
        dx_dt = np.gradient(interp_loss_ls)
        dy_dt = np.gradient(interp_loss_reg)

    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    curvature = np.zeros_like(interp_loss_ls)
    tmp = (
        np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2)
        / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
    )
    if is_logspace:
        curvature[2:-2] = tmp
        curvature[:2] = tmp[0]
        curvature[-2:] = tmp[-1]
    else:
        curvature = tmp
    return curvature


def _interpolate_reg_weights(
    reg_weights: ArrayLike, loss_ls_list: ArrayLike, interp_loss_ls: NDArrayFloat
) -> NDArrayFloat:
    # reg_weights is an increasing sequence
    # but it is not necessarily the case for loss_ls_list
    # so we eliminate non increasing entries
    valid_rw = [reg_weights[0]]
    valid_ls = [loss_ls_list[0]]

    for i in np.arange(len(reg_weights) - 1) + 1:
        if loss_ls_list[i] > valid_ls[-1]:
            valid_rw.append(reg_weights[i])
            valid_ls.append(loss_ls_list[i])

    # since reg_weights and loss_ls_list are both increasing sequence, using linear
    # interpolation give a monotonic interpolation
    return np.interp(interp_loss_ls, np.asarray(valid_ls), np.asarray(valid_rw))


def _interpolate_lcurve(
    reg_weights: ArrayLike,
    loss_ls_list: ArrayLike,
    loss_reg_list: ArrayLike,
    is_logspace: bool = False,
    target_n: int = 500,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Interpolate the trace with provided function to desired number of points."""
    if is_logspace:
        interp_loss_ls: NDArrayFloat = np.logspace(
            np.log10(np.min(loss_ls_list)),
            np.log10(np.max(loss_ls_list)),
            target_n + 1,
            base=10,
        )
    else:
        interp_loss_ls: NDArrayFloat = np.linspace(
            np.min(loss_ls_list),
            np.max(loss_ls_list),
            target_n + 1,
        )

    # sort by increasing loss_ls
    x_sorted, y_sorted, z_sorted = np.array(
        sorted(zip(loss_ls_list, loss_reg_list, reg_weights))
    ).T

    # Transform the response subspace
    def lcurve(x, a, b) -> NDArrayFloat:
        return np.max(np.log(y_sorted)) + a * (x - np.min(np.log(x_sorted))) ** b

    # fit parameters
    popt, err = sp.optimize.curve_fit(
        lcurve,
        np.log(x_sorted),
        np.log(y_sorted),
        p0=(
            -np.abs(np.min(np.log(y_sorted))),
            0.5,
        ),
        bounds=np.array(
            [
                (-np.inf, 0),
                (1e-10, np.inf),
            ]
        ).T,
    )

    # interpolate loss reg
    interp_loss_reg: NDArrayFloat = np.exp(lcurve(np.log(interp_loss_ls), *popt))

    interp_reg_weights: NDArrayFloat = _interpolate_reg_weights(
        reg_weights, loss_ls_list, interp_loss_ls
    )

    return interp_reg_weights, interp_loss_ls, interp_loss_reg


def get_l_curvature(
    reg_weights: ArrayLike,
    loss_ls_list: ArrayLike,
    loss_reg_list: ArrayLike,
    is_logspace: bool = False,
    nb_interp_points: int = 500,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, int]:
    """
    Interpolate and evaluate the L-curve curvature.

    Parameters
    ----------
    reg_weights : ArrayLike
        List of regularization weights in increasing order.
    loss_ls_list : ArrayLike
        List of least square objective function (or equivalent data fit measure).
    loss_reg_list : ArrayLike
        List of regularization objective function.
    is_logspace : bool, optional
        Whether to use logspace for the fit, it depends on data scaling.
        by default False.
    nb_interp_points : int, optional
        Number of interpolation points, by default 500

    Returns
    -------
    Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, int]
        _description_
    """

    # step 1) Interpolate the functions and the regularization weights to get
    # smoother curves
    interp_reg_weights, interp_loss_ls, interp_loss_reg = _interpolate_lcurve(
        reg_weights, loss_ls_list, loss_reg_list, is_logspace, nb_interp_points
    )

    # evaluate the curvature from the smooth interpolations
    curvature = _get_curvature(interp_loss_ls, interp_loss_reg, is_logspace)

    return (
        interp_reg_weights,
        interp_loss_ls,
        interp_loss_reg,
        curvature,
        int(np.argmax(np.abs(curvature))),
    )
