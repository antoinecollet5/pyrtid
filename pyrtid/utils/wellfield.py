"""
Provide utilities to create wellfields.

@author: acollet
"""
# pylint: disable=C0103 # doesn't conform to snake_case naming style
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np


def gen_hexagon_vertices(
    startx: float,
    starty: float,
    endx: float,
    endy: float,
    radius: float,
    rotation: float = 0.0,
    selection: Optional[Sequence[int]] = None,
) -> Sequence[Sequence[float]]:
    """
    Generate and hexagon paving and return vertices coordinates of each hexagon.

    Note
    ----
    The first coordinate is repeated (to close the polygon).

    Parameters
    ----------
    startx : float
        Grid x0 coordinate.
    starty : float
        Grid y0 coordinate.
    endx : float
        Grid xmax coordinate.
    endy : float
        Grid ymax coordinate.
    radius : float
        Hexagon radius.
    rotation: float, optional
        The rotation in degrees. The grid center point is used as origin.
        The default 0.0.
    selection: Optional[Sequence[int]]
        List of hexagons to keep.

    Returns
    -------
    polygons : List[List[float]]
        vertices coordinates by hexagon.

    """
    sl = (2 * radius) * math.tan(math.pi / 6)

    # origin for rotation = grid center
    origin = (startx + (endx - startx) / 2, starty + (endy - starty) / 2)

    # calculate coordinates of the hexagon points
    p = sl * 0.5
    b = sl * math.cos(math.radians(30))
    w = b * 2
    h = 2 * sl

    # offsets for moving along and up rows
    xoffset = b
    yoffset = 3 * p

    row = 1

    shifted_xs = []
    straight_xs = []
    shifted_ys = []
    straight_ys = []

    while startx < endx:
        xs = [startx, startx, startx + b, startx + w, startx + w, startx + b, startx]
        straight_xs.append(xs)
        shifted_xs.append([xoffset + x for x in xs])
        startx += w

    while starty < endy:
        ys = [
            starty + p,
            starty + (3 * p),
            starty + h,
            starty + (3 * p),
            starty + p,
            starty,
            starty + p,
        ]
        (straight_ys if row % 2 else shifted_ys).append(ys)
        starty += yoffset
        row += 1

    polygons = [list(zip(xs, ys)) for xs in shifted_xs for ys in shifted_ys] + [
        list(zip(xs, ys)) for xs in straight_xs for ys in straight_ys
    ]

    # Apply the rotation
    polygons = [
        list(map(tuple, rotate(polygon, origin=origin, degrees=rotation)))
        for polygon in polygons
    ]

    polygons = sorted(
        polygons,
        key=lambda polygon: [np.mean(polygon, axis=0)[1], np.mean(polygon, axis=0)[0]],
    )

    if selection is not None:
        return [poly for i, poly in enumerate(polygons) if i in selection]
    return polygons


def rotate(p, origin: Tuple[float, float] = (0, 0), degrees: float = 0.0) -> np.ndarray:
    """
    Apply rotation to the input array of coordinates.

    Parameters
    ----------
    p : TYPE
        DESCRIPTION.
    origin : Tuple[float, float], optional
        DESCRIPTION. The default is (0, 0).
    degrees : float, optional
        DESCRIPTION. The default is 0.0.

    Returns
    -------
    np.ndarray
        Updated coordinates.

    """
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


def gen_wells_coordinates(
    startx: float,
    starty: float,
    endx: float,
    endy: float,
    radius: float,
    rotation: float = 0.0,
    selection: Optional[Sequence[int]] = None,
) -> Tuple[
    List[Tuple[float, float]],
    List[Tuple[float, float]],
    List[List[Tuple[float, float]]],
]:
    """
    Generate sequences of injectors and producers coordinates.

    Parameters
    ----------
    startx : float
        Grid x0 coordinate.
    starty : float
        Grid y0 coordinate.
    endx : float
        Grid xmax coordinate.
    endy : float
        Grid ymax coordinate.
    radius : float
        Hexagon radius.
    rotation: float, optional
        The rotation in degrees. The grid center point is used as origin.
        The default 0.0.
    selection: Optional[Sequence[int]]
        List of hexagons to keep.

    Returns
    -------
    (Tuple[List[Tuple[float]], List[Tuple[float]]])
        Coordinates of injectors and produers

    """
    polygons = gen_hexagon_vertices(
        startx, starty, endx, endy, radius, rotation, selection
    )
    # barycenter = we do not take the last vertices which is a duplicate of the
    # first one to close the polygon
    producers = [tuple(np.mean(polygon[:-1], axis=0)) for polygon in polygons]

    # Extract the vector of (x, y) coordinatesist(map(tuple, a))
    injectors = list(
        map(tuple, np.unique(np.array(polygons).reshape(-1, 2).round(4), axis=0))
    )

    return injectors, producers, polygons
