# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""Provide ACA."""

import numpy as np

__all__ = ["GenerateDenseMatrix", "ACA", "ACApp"]


def GenerateDenseMatrix(pts, ind_x, ind_y, kernel):
    """_summary_

    # Compute O(N^2) interactions

    Parameters
    ----------
    pts : _type_
        _description_
    ind_x : _type_
        _description_
    ind_y : _type_
        _description_
    kernel : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    nx = ind_x.size
    ny = ind_y.size

    ptsx = pts[ind_x, :]
    ptsy = pts[ind_y, :]

    if nx == 1:
        ptsx = ptsx[np.newaxis, :]
    if ny == 1:
        ptsy = ptsy[np.newaxis, :]

    dim = np.size(pts, 1)
    R = np.zeros((nx, ny), "d")

    for i in np.arange(dim):
        X, Y = np.meshgrid(ptsx[:, i], ptsy[:, i])
        R += (X.transpose() - Y.transpose()) ** 2.0

    return kernel(np.sqrt(R))


# This is the expensive version, only for debugging purposes
def ACA(pts, ind_x, ind_y, kernel, rkmax, eps):
    """
    Adaptive Cross Approximation

    Parameters:
    -----------
    pts:    (n,dim) ndarray
        all the points

    ind_x:    (nx,) ndarray
        indices of the points

    ind_y:    (ny,) ndarray
        indices of the points

    kernel:    Kernel object
        see covariance/kernel.py for more details

    rkmax:     int
        Maximum rank of the low rank approximation

    eps:    double
        Relative error of the low rank approximation

    Returns:
    --------

    A,B:    (nx,k) and (ny,k) ndarray
        such that Q approx AB^T

    References:
    -----------
    Sergej Rjasanow, Olaf Steinbach, The fast solution of boundary integral equations.
        Mathematical and analytical techniques with applications
        to engineering. Springer 2007, New York.

    """

    # Generate matrix
    R = GenerateDenseMatrix(pts, ind_x, ind_y, kernel)

    # lod
    normR = np.linalg.norm(R, "fro")

    nx = ind_x.size
    ny = ind_y.size

    A = np.zeros((nx, rkmax), "d")
    B = np.zeros((ny, rkmax), "d")

    kmax = rkmax

    for k in np.arange(rkmax):
        # Find largest pivot indices
        ind = np.unravel_index(np.argmax(np.abs(R)), (nx, ny))

        # Largest pivot
        gamma = 1.0 / R[ind]

        u, v = gamma * R[:, ind[1]], R[ind[0], :]

        A[:, k] = np.copy(u)
        B[:, k] = np.copy(v.transpose())

        R -= np.outer(u, v)

        if np.linalg.norm(R, "fro") <= eps * normR:
            kmax = k
            break

    return A[:, :kmax], B[:, :kmax]


# Implementation of the partially pivoted Adaptive Cross Approximation
def ACApp(pts, ind_x, ind_y, kernel, rkmax, eps):
    """
    Partially pivoted Adaptive Cross Approximation.

    Parameters:
    -----------
    pts:    (n,dim) ndarray
            all the points

    ind_x:   (nx,) ndarray
            indices of the points

    ind_y:   (ny,) ndarray
            indices of the points

    kernel: Kernel object
            see covariance/kernel.py for more details

    rkmax:  int
            Maximum rank of the low rank approximation

    eps:    double
            Approximate relative error of the low rank approximation
    Computing frobenius norm is O(n^2) so it is avoided.
    Returns:
    --------

    A,B:    (nx,k) and (ny,k) ndarray
            such that Q approx AB^T

    References:
    -----------
    Sergej Rjasanow, Olaf Steinbach, The fast solution of boundary integral equations.
    Mathematicaical and analytical techniques with applications
    to engineering. Springer 2007, New York.

    """
    nx = np.size(ind_x)
    ny = np.size(ind_y)

    A = np.zeros((nx, rkmax), "d")
    B = np.zeros((ny, rkmax), "d")

    rows = np.zeros((rkmax + 1,), "i")

    # Initialize
    row = np.min(np.arange(nx))

    # Norm
    norm = 0.0

    # Maximum rank
    kmax = rkmax

    for k in np.arange(rkmax):
        # generate row
        b = GenerateDenseMatrix(pts, ind_x[row], ind_y, kernel)
        B[:, k] = np.copy(b.ravel())
        for nu in np.arange(k):
            B[:, k] -= A[row, nu] * B[:, nu]

        # maximum row entry
        col = np.argmax(np.abs(B[:, k]))

        # Compute delta
        delta = B[col, k]
        if np.abs(delta) < 1.0e-16:
            kmax = k
            break

        B[:, k] /= delta

        # Generate column
        a = GenerateDenseMatrix(pts, ind_x, ind_y[col], kernel)

        A[:, k] = np.copy(a.ravel())

        for nu in np.arange(k):
            A[:, k] -= A[:, nu] * B[col, nu]

        # Next row
        diff = np.setdiff1d(np.arange(nx), rows[: k + 1])
        if np.size(diff) == 0:
            break
        row = diff[np.argmin(np.abs(A[diff, k]))]
        rows[k + 1] = row

        # update norm
        for nu in np.arange(k):
            norm += 2.0 * np.dot(A[:, k], A[:, nu]) * np.dot(B[:, k], B[:, nu])

        ukvk = np.linalg.norm(A[:, k]) ** 2.0 * np.linalg.norm(B[:, k]) ** 2.0
        norm += ukvk

        if ukvk <= eps * np.sqrt(norm):
            kmax = k
            break

    return A[:, :kmax], B[:, :kmax]
