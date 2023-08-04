"""
    toeplitz matrix-vector multiplication adapted from Arvind Saibaba's code
"""
from typing import Callable

import numpy as np

from pyrtid.utils.types import NDArrayFloat, NDArrayInt


def get_distances(x, y, theta: NDArrayFloat) -> NDArrayFloat:
    dim = x.shape[1]
    DM = np.zeros(x.shape[0])

    if dim == 1:
        DM = (x[:] - y) ** 2.0 / theta**2.0
    else:
        for i in np.arange(dim):
            DM += (x[:, i] - y[i]) ** 2.0 / theta[i] ** 2.0

    DM = np.sqrt(DM)
    return DM


def create_row(
    mesh_dim: NDArrayInt,
    shape: NDArrayInt,
    kernel: Callable,
    theta: NDArrayFloat,
):
    """
    Create row column of covariance matrix
    """

    xmin: NDArrayFloat = np.array(mesh_dim) / 2.0
    xmax = (np.array(shape) - 0.5) * mesh_dim
    dim = shape.size

    if dim == 1:
        x = np.linspace(xmin[0], xmax[0], shape[0])
        x = x.reshape(-1, 1)  # make it 2D for consistency
        R = get_distances(x, x[0], theta)
    elif dim == 2:
        x1 = np.linspace(xmin[0], xmax[0], shape[0])
        x2 = np.linspace(xmin[1], xmax[1], shape[1])
        xx, yy = np.meshgrid(x1, x2, indexing="ij")
        x = np.vstack((np.ravel(xx, order="F"), np.ravel(yy, order="F"))).transpose()
        R = get_distances(x, x[0, :].transpose(), theta)
    elif dim == 3:
        x1 = np.linspace(xmin[0], xmax[0], shape[0])
        x2 = np.linspace(xmin[1], xmax[1], shape[1])
        x3 = np.linspace(xmin[2], xmax[2], shape[2])
        xx, yy, zz = np.meshgrid(x1, x2, x3, indexing="ij")
        x = np.vstack(
            (np.ravel(xx, order="F"), np.ravel(yy, order="F"), np.ravel(zz, order="F"))
        ).transpose()
        R = get_distances(x, x[0, :].transpose(), theta)

    else:
        raise ValueError("Support 1,2 and 3 dimensions")

    row = kernel(R)

    return row, x


def toeplitz_product(x, row, shape):
    """Toeplitz matrix times x

    :param x: x for Qx
    :param row: from CreateRow
    :param shape: size in each dimension ex) shape = [2,3,4]
    :return: Qx
    """
    dim = shape.size

    if dim == 1:
        circ = np.concatenate((row, row[-2:0:-1])).reshape(-1)
        padded = np.concatenate((x, np.zeros(shape[0] - 2)))
        result = np.fft.ifft(np.fft.fft(circ) * np.fft.fft(padded))
        result = np.real(result[0 : shape[0]])

    elif dim == 2:
        circ = np.reshape(row, (shape[0], shape[1]), order="F")
        circ = np.concatenate((circ, circ[:, -2:0:-1]), axis=1)
        circ = np.concatenate((circ, circ[-2:0:-1, :]), axis=0)

        n = np.shape(circ)
        padded = np.reshape(x, (shape[0], shape[1]), order="F")

        result = np.fft.ifft2(np.fft.fft2(circ) * np.fft.fft2(padded, n))
        result = np.real(result[0 : shape[0], 0 : shape[1]])
        result = np.reshape(result, -1, order="F")

    elif dim == 3:
        circ = np.reshape(row, (shape[0], shape[1], shape[2]), order="F")
        circ = np.concatenate((circ, circ[:, :, -2:0:-1]), axis=2)
        circ = np.concatenate((circ, circ[:, -2:0:-1, :]), axis=1)
        circ = np.concatenate((circ, circ[-2:0:-1, :, :]), axis=0)

        n = np.shape(circ)
        padded = np.reshape(x, shape, order="F")

        result = np.fft.ifftn(np.fft.fftn(circ) * np.fft.fftn(padded, n))
        result = np.real(result[0 : shape[0], 0 : shape[1], 0 : shape[2]])
        result = np.reshape(result, -1, order="F")
    else:
        raise ValueError("Support 1,2 and 3 dimensions")

    return result


def Realizations(row, shape):
    dim = shape.size
    if dim == 1:
        circ = np.concatenate((row, row[-2:0:-1]))
        n = circ.shape

        eps = np.random.normal(0, 1, n) + 1j * np.random.normal(0, 1, n)
        res = np.fft.ifft(np.sqrt(np.fft.fft(circ)) * eps) * np.sqrt(n)

        r1 = np.real(res[0 : shape[0]])
        r2 = np.imag(res[0 : shape[0]])

    elif dim == 2:
        circ = np.reshape(row, (shape[0], shape[1]), order="F")
        circ = np.concatenate((circ, circ[:, -2:0:-1]), axis=1)
        circ = np.concatenate((circ, circ[-2:0:-1, :]), axis=0)

        n = np.shape(circ)
        eps = np.random.normal(0, 1, n) + 1j * np.random.normal(0, 1, n)

        res = np.fft.ifft2(np.sqrt(np.fft.fft2(circ)) * eps) * np.sqrt(n[0] * n[1])
        res = res[0 : shape[0], 0 : shape[1]]
        res = np.reshape(res, -1, order="F")

        r1 = np.real(res)
        r2 = np.imag(res)

    elif dim == 3:
        circ = np.reshape(row, (shape[0], shape[1], shape[2]), order="F")
        circ = np.concatenate((circ, circ[:, :, -2:0:-1]), axis=2)
        circ = np.concatenate((circ, circ[:, -2:0:-1, :]), axis=1)
        circ = np.concatenate((circ, circ[-2:0:-1, :, :]), axis=0)

        n = np.shape(circ)
        eps = np.random.normal(0, 1, n) + 1j * np.random.normal(0, 1, n)

        res = np.fft.ifftn(np.sqrt(np.fft.fftn(circ)) * eps) * np.sqrt(
            n[0] * n[1] * n[2]
        )
        res = res[0 : shape[0], 0 : shape[1], 0 : shape[2]]
        res = np.reshape(res, -1, order="F")

        r1 = np.real(res)
        r2 = np.imag(res)
    else:
        raise ValueError("Support 1,2 and 3 dimensions")

    return r1, r2, eps
