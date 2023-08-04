r"""
Provide finite difference approxation for the gradient and hessian.

Alternative, see the library
`numdifftools <https://numdifftools.readthedocs.io/en/latest/index.html.>`_

TODO: test and see if the results are the same with numdifftools

grad = finite_gradient(np.array([1, 1]), rosen)

Pour le moment,j'aime bien cette implémentation car on sait ce qu'il s'y passe.
A voir plus tard si on peu accelerer ça avec numba ?

Chapitre très intéressant:
    https://pythonnumericalmethods.berkeley.edu/notebooks/chapter20.02-Finite-Difference-Approximating-Derivatives.html

@author: acollet
"""


import sys
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np


def rosen(x):
    """Rosenbrook function."""
    return (1 - x[0]) ** 2 + 105.0 * (x[1] - x[0] ** 2) ** 2


def gradient(x):
    """Rosenbrook function first derivative."""
    return np.array([1.0, 1.0, 1.0])


def is_gradient_correct(
    x: np.ndarray,
    fm: Callable,
    grad: Callable,
    fm_args: Optional[Tuple[Any]] = None,
    fm_kwargs: Optional[Dict[str, Any]] = None,
    grad_args: Optional[Tuple[Any]] = None,
    grad_kwargs: Optional[Dict[str, Any]] = None,
    accuracy: int = 0,
    eps: float = sys.float_info.epsilon * 1e10,
) -> bool:
    """
    Check by finite difference if the gradient is correct.

    Parameters
    ----------
    x : np.ndarray
        The input parameters vector.
    fm : Callable
        Forward model.
    grad : Callable
        Gradient model.
    fm_args: Tuple[Any]
        Positional arguments for the forward model.
    fm_kwargs : Dict[Any, Any]
        Keyword arguments for the forward model.
    grad_args: Tuple[Any]
        Positional arguments for the gradient model.
    grad_kwargs : Dict[Any, Any]
        Keyword arguments for the gradient model.
    accuracy : int, optional
        Number of points to use for the finite difference approximation.
        Possible values are 0 (2 points), 1 (4 points), 2 (6 points),
        3 (4 points). The default is 0 which corresponds to the central
        difference scheme (2 points).
    eps: float, optional
        The epsilon for the computation (h). The default value has been
        taken from the C++ implementation of
        :cite:`wieschollek2016cppoptimizationlibrary`, and should correspond
        to the optimal h taking into account the roundoff errors due to
        the machine precision. The default is -2.2204e-6.

    Returns
    -------
    bool
        True if the gradient is correct, false otherwise.

    """
    if grad_args is None:
        grad_args = ()
    if grad_kwargs is None:
        grad_kwargs = {}
    actual_grad = grad(x, *grad_args, **grad_kwargs)
    expected_grad = finite_gradient(x, fm, fm_args, fm_kwargs, accuracy, eps)

    return is_all_close(actual_grad, expected_grad)


def is_all_close(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-2) -> bool:
    """Return whether the two vectors are approximately equal."""
    scale = np.maximum(np.maximum(np.abs(v1), np.abs(v2)), 1.0)
    return np.less_equal((np.abs(v1 - v2)), eps * scale).all()


def finite_gradient(
    x: np.ndarray,
    fm: Callable,
    fm_args: Optional[Sequence[Any]] = None,
    fm_kwargs: Optional[Dict[str, Any]] = None,
    accuracy: int = 0,
    eps: Optional[float] = None,
) -> np.ndarray:
    r"""
    Compute the gradient by finite difference.

    The gradient is computed by using Taylor Series. For instance, if
    accurcy = 1, we use 4 points, which means that
    we take the Taylor series of $f$ around $a = x_j$ and compute the series
    at $x = x_{j-2}, x_{j-1}, x_{j+1}, x_{j+2}$.

    .. math::
        \begin{eqnarray*}
        f(x_{j-2}) &=& f(x_j) - 2hf^{\prime}(x_j) + \frac{4h^2f''(x_j)}{2} -
        \frac{8h^3f'''(x_j)}{6} + \frac{16h^4f''''(x_j)}{24}
        - \frac{32h^5f'''''(x_j)}{120} + \cdots\\
        f(x_{j-1}) &=& f(x_j) - hf^{\prime}(x_j) + \frac{h^2f''(x_j)}{2}
        - \frac{h^3f'''(x_j)}{6} + \frac{h^4f''''(x_j)}{24}
        - \frac{h^5f'''''(x_j)}{120} + \cdots\\
        f(x_{j+1}) &=& f(x_j) + hf^{\prime}(x_j) + \frac{h^2f''(x_j)}{2}
        + \frac{h^3f'''(x_j)}{6} + \frac{h^4f''''(x_j)}{24}
        + \frac{h^5f'''''(x_j)}{120} + \cdots\\
        f(x_{j+2}) &=& f(x_j) + 2hf^{\prime}(x_j) + \frac{4h^2f''(x_j)}{2}
        + \frac{8h^3f'''(x_j)}{6} + \frac{16h^4f''''(x_j)}{24}
        + \frac{32h^5f'''''(x_j)}{120} + \cdots
        \end{eqnarray*}

    To get the $h^2, h^3$, and $h^4$ terms to cancel out, we can compute

    .. math::
        f(x_{j-2}) - 8f(x_{j-1}) + 8f(x_{j-1}) - f(x_{j+2})
        = 12hf^{\prime}(x_j) - \frac{48h^5f'''''(x_j)}{120}

    which can be rearranged to

    .. math::
        f^{\prime}(x_j) = \frac{f(x_{j-2}) - 8f(x_{j-1})
        + 8f(x_{j-1}) - f(x_{j+2})}{12h} + O(h^4).

    This formula is a better approximation for the derivative at $x_j$
    than the central difference formula, but requires twice as many
    calculations.

    Parameters
    ----------
    x : np.ndarray
        The input parameters array.
    fm : Callable
        Forward model.
    fm_args: Tuple[Any]
        Positional arguments for the forward model.
    fm_kwargs : Dict[Any, Any]
        Keyword arguments for the forward model.
    accuracy : int, optional
        Number of points to use for the finite difference approximation.
        Possible values are 0 (2 points), 1 (4 points), 2 (6 points),
        3 (4 points). The default is 0 which corresponds to the central
        difference scheme (2 points).
    eps: float, optional
        The epsilon for the computation (h). The default value has been
        taken from the C++ implementation of
        :cite:`wieschollek2016cppoptimizationlibrary`, and should correspond
        to the optimal h taking into account the roundoff errors due to
        the machine precision. The default is 2.2204e-6.

    Returns
    -------
    np.ndarray
        The finite difference gradient vector.

    """
    if eps is None:
        eps = sys.float_info.epsilon * 1e10
    if accuracy not in [0, 1, 2, 3]:
        raise ValueError("The accuracy should be 0, 1, 2 or 3!")
    if fm_args is None:
        fm_args = []
    if fm_kwargs is None:
        fm_kwargs = {}
    shape = x.shape
    x = x.ravel().astype(np.float64)
    grad = np.zeros(x.size)
    coeff = [
        [1.0, -1.0],
        [1, -8.0, 8.0, -1.0],
        [-1, 9, -45.0, 45.0, -9.0, 1.0],
        [3.0, -32.0, 168.0, -672.0, 672.0, -168.0, 32.0, -3.0],
    ]
    coeff2 = [
        [1.0, -1.0],
        [-2.0, -1.0, 1.0, 2.0],
        [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
        [-4, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0],
    ]
    dd = [2.0, 12.0, 60.0, 840.0]

    inner_steps = 2 * (accuracy + 1)
    dd_val = dd[accuracy] * eps

    for i in range(x.size):
        for s in range(inner_steps):
            tmp = x[i].copy()
            x[i] += coeff2[accuracy][s] * eps
            grad[i] += coeff[accuracy][s] * fm(x.reshape(shape), *fm_args, **fm_kwargs)
            x[i] = tmp
        grad[i] /= dd_val
    return grad.reshape(shape)


def is_hessian_correct(
    x: np.ndarray,
    fm: Callable,
    hess: Callable,
    fm_args: Tuple[Any] = (),
    fm_kwargs: Dict[str, Any] = {},
    hess_args: Tuple[Any] = (),
    hess_kwargs: Dict[str, Any] = {},
    accuracy: int = 0,
    eps: float = sys.float_info.epsilon * 1e7,
) -> bool:
    """
    Check by finite difference if the hessian is correct.

    Parameters
    ----------
    x : np.ndarray
        The input parameters vector.
    fm : Callable
        Forward model.
    hess : Callable
        Hessian model.
    fm_args: Tuple[Any]
        Positional arguments for the forward model.
    fm_kwargs : Dict[Any, Any]
        Keyword arguments for the forward model.
    hess_args: Tuple[Any]
        Positional arguments for the hessian model.
    grad_kwargs : Dict[Any, Any]
        Keyword arguments for the hessian model.
    accuracy : int, optional
        Number of points to use for the finite difference approximation.
        Possible values are 0 (2 points), 1 (4 points), 2 (6 points),
        3 (4 points). The default is 0 which corresponds to the central
        difference scheme (2 points).
    eps: float, optional
        The epsilon for the computation (h). The default value has been
        taken from the C++ implementation of
        :cite:`wieschollek2016cppoptimizationlibrary`, and should correspond
        to the optimal h taking into account the roundoff errors due to
        the machine precision. The default is -2.2204e-6.

    Returns
    -------
    bool
        True if the gradient is correct, false otherwise.

    """
    actual_hessian = hess(x, hess_args, hess_kwargs)
    expected_hessian = finite_gradient(x, fm, fm_args, fm_kwargs, accuracy, eps)

    scale = np.maximum(
        np.maximum(np.abs(actual_hessian), np.abs(expected_hessian)), 1.0
    )
    return np.less_equal(
        (np.abs(actual_hessian - expected_hessian)), 1e-1 * scale
    ).all()


def finite_hessian(
    x: np.ndarray,
    fm: Callable,
    fm_args: Tuple[Any] = (),
    fm_kwargs: Dict[str, Any] = {},
    accuracy: int = 0,
    eps: float = sys.float_info.epsilon * 1e7,
) -> np.ndarray:
    r"""
    Compute the hessian by finite difference.


      /*
        \displaystyle{{\frac{\partial^2{f}}{\partial{x}\partial{y}}}\approx
        \frac{1}{600\,h^2} \left[\begin{matrix}
          -63(f_{1,-2}+f_{2,-1}+f_{-2,1}+f_{-1,2})+\\
          63(f_{-1,-2}+f_{-2,-1}+f_{1,2}+f_{2,1})+\\
          44(f_{2,-2}+f_{-2,2}-f_{-2,-2}-f_{2,2})+\\
          74(f_{-1,-1}+f_{1,1}-f_{1,-1}-f_{-1,1})
        \end{matrix}\right] }
      */

    Parameters
    ----------
    x : np.ndarray
        The input parameters vector.
    fm : Callable
        Forward model.
    fm_args: Tuple[Any]
        Positional arguments for the forward model.
    fm_kwargs : Dict[Any, Any]
        Keyword arguments for the forward model.
    accuracy : int, optional
        Number of points to use for the finite difference approximation.
        Possible values are 0 (3 points), 1 (4 points), 2 (6 points),
        3 (4 points). The default is 0 which corresponds to the central
        difference scheme (2 points).
    eps: float, optional
        The epsilon for the computation (h). The default value has been
        taken from the C++ implementation of
        :cite:`wieschollek2016cppoptimizationlibrary`, and should correspond
        to the optimal h taking into account the roundoff errors due to
        the machine precision. The default is 2.2204e-9.

    Returns
    -------
    None.

    """
    hessian = np.zeros((x.size, x.size))

    # Centered scheme
    if accuracy == 0:
        for i, j in np.ndindex(hessian.shape):
            tmp_xi = x[i].copy()
            tmp_xj = x[j].copy()
            f4 = fm(x, *fm_args, **fm_kwargs)
            x[i] += eps
            x[i] += eps
            f1 = fm(x, *fm_args, **fm_kwargs)
            x[i] -= eps
            f2 = fm(x, *fm_args, **fm_kwargs)
            x[j] += eps
            x[i] -= eps
            f3 = fm(x, *fm_args, **fm_kwargs)

            hessian[i, j] = (f1 - f2 - f3 + f4) / (eps * eps)

            x[i] = tmp_xi
            x[j] = tmp_xj
    else:
        for i, j in np.ndindex(hessian.shape):
            tmp_xi = x[i].copy()
            tmp_xj = x[j].copy()

            term_1 = 0
            x[i] = tmp_xi
            x[j] = tmp_xj
            x[i] += 1 * eps
            x[j] += -2 * eps
            term_1 += fm(x, *fm_args, **fm_kwargs)
            x[i] = tmp_xi
            x[j] = tmp_xj
            x[i] += 2 * eps
            x[j] += -1 * eps
            term_1 += fm(x, *fm_args, **fm_kwargs)
            x[i] = tmp_xi
            x[j] = tmp_xj
            x[i] += -2 * eps
            x[j] += 1 * eps
            term_1 += fm(x, *fm_args, **fm_kwargs)
            x[i] = tmp_xi
            x[j] = tmp_xj
            x[i] += -1 * eps
            x[j] += 2 * eps
            term_1 += fm(x, *fm_args, **fm_kwargs)

            term_2 = 0
            x[i] = tmp_xi
            x[j] = tmp_xj
            x[i] += -1 * eps
            x[j] += -2 * eps
            term_2 += fm(x, *fm_args, **fm_kwargs)
            x[i] = tmp_xi
            x[j] = tmp_xj
            x[i] += -2 * eps
            x[j] += -1 * eps
            term_2 += fm(x, *fm_args, **fm_kwargs)
            x[i] = tmp_xi
            x[j] = tmp_xj
            x[i] += 1 * eps
            x[j] += 2 * eps
            term_2 += fm(x, *fm_args, **fm_kwargs)
            x[i] = tmp_xi
            x[j] = tmp_xj
            x[i] += 2 * eps
            x[j] += 1 * eps
            term_2 += fm(x, *fm_args, **fm_kwargs)

            term_3 = 0
            x[i] = tmp_xi
            x[j] = tmp_xj
            x[i] += 2 * eps
            x[j] += -2 * eps
            term_3 += fm(x, *fm_args, **fm_kwargs)
            x[i] = tmp_xi
            x[j] = tmp_xj
            x[i] += -2 * eps
            x[j] += 2 * eps
            term_3 += fm(x, *fm_args, **fm_kwargs)
            x[i] = tmp_xi
            x[j] = tmp_xj
            x[i] += -2 * eps
            x[j] += -2 * eps
            term_3 -= fm(x, *fm_args, **fm_kwargs)
            x[i] = tmp_xi
            x[j] = tmp_xj
            x[i] += 2 * eps
            x[j] += 2 * eps
            term_3 -= fm(x, *fm_args, **fm_kwargs)

            term_4 = 0
            x[i] = tmp_xi
            x[j] = tmp_xj
            x[i] += -1 * eps
            x[j] += -1 * eps
            term_4 += fm(x, *fm_args, **fm_kwargs)
            x[i] = tmp_xi
            x[j] = tmp_xj
            x[i] += 1 * eps
            x[j] += 1 * eps
            term_4 += fm(x, *fm_args, **fm_kwargs)
            x[i] = tmp_xi
            x[j] = tmp_xj
            x[i] += 1 * eps
            x[j] += -1 * eps
            term_4 -= fm(x, *fm_args, **fm_kwargs)
            x[i] = tmp_xi
            x[j] = tmp_xj
            x[i] += -1 * eps
            x[j] += 1 * eps
            term_4 -= fm(x, *fm_args, **fm_kwargs)

            x[i] = tmp_xi
            x[j] = tmp_xj

            hessian[i, j] = (-63 * term_1 + 63 * term_2 + 44 * term_3 + 74 * term_4) / (
                600.0 * eps * eps
            )

    return hessian
