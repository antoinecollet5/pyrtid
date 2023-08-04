#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 12:27:40 2022

@author: acollet
"""

from pyrtid.utils import gen_wells_coordinates


def test_gen_well_coordinates() -> None:
    injectors_coords, producers_coords, polygons = gen_wells_coordinates(
        0, 0, 121, 121, radius=60, rotation=45
    )
