#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: acollet
"""

import pytest
from pyrtid.utils import gen_wells_coordinates


@pytest.mark.parametrize("selection", [(None), ([0, 1])])
def test_gen_well_coordinates(selection) -> None:
    injectors_coords, producers_coords, polygons = gen_wells_coordinates(
        0, 0, 121, 121, radius=60, rotation=45, selection=selection
    )
