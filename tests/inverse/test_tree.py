from time import time

import numpy as np

from pyrtid.inverse.regularization.tree import Cluster


def test_build_cluster() -> None:
    tree = Cluster()

    N = 10000

    theta = 2.0 * np.pi * np.arange(N) / float(N)

    # Circle
    pts = np.hstack((np.cos(theta)[:, np.newaxis], np.sin(theta)[:, np.newaxis]))

    # Cardoid
    a = 1.0
    x = a * (2.0 * np.cos(theta) - np.cos(2.0 * theta))
    y = a * (2.0 * np.sin(theta) - np.sin(2.0 * theta))
    pts = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))

    indices = np.arange(N)

    start = time()
    tree.assign_points_bisection(pts, indices)
    print("Assigning points to clusters took %g" % (time() - start))
