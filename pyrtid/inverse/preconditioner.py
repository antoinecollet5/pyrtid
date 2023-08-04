#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 14:43:50 2022

@author: acollet
"""

import numpy as np

from pyrtid.utils.types import NDArrayFloat

# let's say we have something varying between 50 mD and 5000 mD.


def logit(x: NDArrayFloat, lbound: float, ubound: float) -> NDArrayFloat:
    return np.log((x - lbound) / (ubound - x))


def expit(x: NDArrayFloat, lbound: float, ubound: float) -> NDArrayFloat:
    return np.log10((x - lbound) / (ubound - x))


# derivative: https://fr.wikipedia.org/wiki/Logit

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example 1
    min1 = 0.51
    max1 = 5049
    x = np.linspace(min1, max1, 50)
    y = logit(x, min1 - 0.1 * min1, max1 + 0.1 * min1)
    ax1 = plt.subplot(111)
    ax1.plot(y, x)
    plt.show()

    # Example 2
    min2 = 1e-9
    max2 = 1e-4

    x = np.linspace(min2, max2, 50)
    y = logit(x, min2 - 0.1 * min2, max2 + 0.1 * min2)
    y2 = expit(x, min2 - 0.1 * min2, max2 + 0.1 * min2)

    plt.plot(y, x)
    plt.show()

    plt.semilogy(y2, x)
    plt.show()
