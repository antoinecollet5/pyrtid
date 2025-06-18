from typing import List  # For type annotations

import numdifftools as nd
import numpy as np  # NumPy for numerical operations
from lbfgsb.types import NDArrayFloat
from pyrtid.forward.geochem_solver import F, Jacobian
from pyrtid.forward.models import GeochemicalParameters


def test_jacobian():
    # Parameters
    gch_params = GeochemicalParameters(
        kv=-2.5e-7,  # Constante de vitesse (1/s)
        As=13.5,  # Surface réactive (m²)
        Ks=1e2,  # Constante de saturation (mol/m³)
        stocoef=5.0,  # Coefficient stoechiométrique
    )

    # Initialization
    dt = 36000

    # List of mobile concentrations
    lmob: List[NDArrayFloat] = [np.array([1e-10, 1e0])]
    # List of immobile concentrations (grades)
    limmob: List[NDArrayFloat] = [np.array([1.0e-3, 1e-10])]

    # Start a time step
    time_index = 1
    lmob.append(lmob[-1].copy())
    limmob.append(limmob[-1].copy())

    # # Mineral grades
    immob_next = limmob[time_index]
    immob_prev = limmob[time_index - 1]
    # Mobile concentrations
    mob_next = lmob[time_index]
    mob_prev = lmob[time_index - 1]

    def F_wrapper(_C) -> NDArrayFloat:
        return F(_C[0:2], _C[2:], mob_prev, immob_prev, gch_params, dt)

    C = np.hstack([mob_next, immob_next])

    np.testing.assert_allclose(
        nd.Jacobian(F_wrapper, step=1e-5)(C),
        Jacobian(mob_next, immob_next, mob_prev, immob_prev, gch_params, dt),
    )
