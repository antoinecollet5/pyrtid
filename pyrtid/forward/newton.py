"""
Newton solver for the geochemical model.

https://hplgit.github.io/prog4comp/doc/pub/._p4c-bootstrap-Python031.html

For sparse system, this is actually much better:

https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html#scipy.sparse.linalg.lsqr
"""

import numpy as np


def Newton_system(F, J, x, eps):
    """
    Solve nonlinear system F=0 by Newton's method.
    J is the Jacobian of F. Both F and J must be functions of x.
    At input, x holds the start value. The iteration continues
    until ||F|| < eps.
    """
    F_value = F(x)
    F_norm = np.linalg.norm(F_value, ord=2)  # l2 norm of vector
    iteration_counter = 0
    while abs(F_norm) > eps and iteration_counter < 100:
        delta = np.linalg.solve(J(x), -F_value)
        x = x + delta
        F_value = F(x)
        F_norm = np.linalg.norm(F_value, ord=2)
        iteration_counter += 1

    # Here, either a solution is found, or too many iterations
    if abs(F_norm) > eps:
        iteration_counter = -1
    return x, iteration_counter


def test_Newton_system1():
    from numpy import cos, exp, pi, sin

    def F(x):
        return np.array(
            [
                x[0] ** 2 - x[1] + x[0] * cos(pi * x[0]),
                x[0] * x[1] + exp(-x[1]) - x[0] ** (-1),
            ]
        )

    def J(x):
        return np.array(
            [
                [2 * x[0] + cos(pi * x[0]) - pi * x[0] * sin(pi * x[0]), -1],
                [x[1] + x[0] ** (-2), x[0] - exp(-x[1])],
            ]
        )

    expected = np.array([1, 0])
    tol = 1e-4
    x, n = Newton_system(F, J, x=np.array([2, -1]), eps=0.0001)
    print(n, x)
    error_norm = np.linalg.norm(expected - x, ord=2)
    assert error_norm < tol, "norm of error =%g" % error_norm
    print("norm of error =%g" % error_norm)
