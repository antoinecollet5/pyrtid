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
