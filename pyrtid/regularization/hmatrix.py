"""Hierarchical matrix."""

import numpy as np

from pyrtid.regularization.tree import BlockCluster, Cluster


class Hmatrix:
    """
    Implementation of the Hierarchical matrix class.

    Parameters
    ----------
    pts: (n,dim) ndarray
        all the points
    kernel: Kernel object
        see covariance/kernel.py for more details
    ind_x: (nx,) ndarray
        indices of the points
    ind_y: (ny,) ndarray, optional.
        indices of the points
    rkmax: int. default = 32.
        Maximum rank of the low rank approximation
    eps: double. default = 1.e-9.
        Approximate relative error of the low rank approximation
        Computing frobenius norm is O(n^2) so it is avoided.


    Methods
    -------
    mult()
    transpmult()

    Attributes
    ----------
    ctreex, ctreey:    Cluster Tree objects
    btree:        Block Cluster Tree


    Notes
    -----
    Details of the implementation including benchmarking is available in [2].

    References
    ----------
    .. [1] Sergej Rjasanow, Olaf Steinbach, The fast solution of boundary integral
    equations. Mathematical and analytical techniques with applications to engineering.
    Springer 2007, New York.
    .. [2]  A.K. Saibaba, Fast solvers for geostatistical inverse problems and
    uncertainty quantification, PhD Thesis 2013, Stanford University.


    """

    def __init__(
        self, pts, kernel, ind_x, ind_y=None, rkmax=32, eps=1.0e-9, verbose=False
    ):
        self.pts = pts
        self.ind_x = ind_x
        self.ind_y = ind_y
        self.kernel = kernel
        self.verbose = verbose

        from time import time

        # Construct cluster tree
        dim = np.size(pts, 1)
        self.ctreex = Cluster(dim=dim, level=0)
        start = time()
        self.ctreex.assign_points_bisection(pts, self.ind_x)
        if verbose:
            print("Time to construct cluster tree is %g " % (time() - start))

        if ind_y is None:
            self.ctreey = self.ctreex
        else:
            self.ctreey = Cluster(dim=dim, level=0)
            start = time()
            self.ctreey.assign_points_bisection(pts, self.ind_y)
            if verbose:
                print("Time to construct cluster tree is %g " % (time() - start))

        # Construct Block Cluster tree
        self.btree = BlockCluster(level=0)
        self.btree.construct_block_tree(self.ctreex, self.ctreey)
        if verbose:
            print("Time to construct block cluster tree is %g " % (time() - start))
        self.btree.construct_low_rank_representation(self.pts, self.kernel, rkmax, eps)
        if verbose:
            print("Time to construct low rank representation is %g " % (time() - start))

        return

    def mult(self, x, y, verbose=False):
        """
        Matrix-vector product with the H-matrix.

        Parameters
        ----------
        x:    (n,) ndarray
        y:    (n,) ndarray
        verbose: bool, False

        """
        from time import time

        start = time()
        self.btree.mult(x, y, self.pts, self.kernel)
        if verbose:
            print("Time for mat-vec is %g" % (time() - start))

        return

    def transpmult(self, x, y, verbose=False):
        """
        Transpose matrix-vector product with the H-matrix.

        Parameters
        ----------
        x:    (n,) ndarray
        y:    (n,) ndarray
        verbose: bool, False

        """

        from time import time

        start = time()
        self.btree.transpmult(x, y, self.pts, self.kernel)
        if verbose:
            print("Time for mat-vec is %g" % (time() - start))

        return

    def _memoryusage(self):
        leaflist = []
        costlist = []
        self.btree.construct_space_filling_curve(leaflist, costlist)

        memory = float(np.sum(costlist) * 8.0) / (1024.0**2.0)
        return memory
