# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""Provide the directions in which the forward sensitivities are computed."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pyrtid.forward.models import ForwardModel
from pyrtid.inverse.params import AdjustableParameters, ParameterName
from pyrtid.utils import NDArrayFloat, object_or_object_sequence_to_list


@dataclass
class FSMDirections:
    """
    Perturbations of the model parameters, one per column of the vectors.

    All the arrays have shape (nx, ny, nz, ne) with ne the number of vectors. A
    parameter which is not adjusted is None. The values are expressed in the space
    of the non-preconditioned parameters.

    Attributes
    ----------
    permeability : NDArrayFloat | None
        Perturbations of the permeability.
    storage_coefficient : NDArrayFloat | None
        Perturbations of the storage coefficient.
    head : NDArrayFloat | None
        Perturbations of the initial head.
    porosity : NDArrayFloat | None
        Perturbations of the porosity.
    diffusion : NDArrayFloat | None
        Perturbations of the diffusion.
    dispersivity : NDArrayFloat | None
        Perturbations of the dispersivity.
    conc : list[NDArrayFloat | None]
        Perturbations of the initial concentrations, for each species.
    grade : list[NDArrayFloat | None]
        Perturbations of the initial grades, for each species.
    ne : int
        The number of vectors.
    """

    ne: int
    permeability: NDArrayFloat | None = None
    storage_coefficient: NDArrayFloat | None = None
    head: NDArrayFloat | None = None
    porosity: NDArrayFloat | None = None
    diffusion: NDArrayFloat | None = None
    dispersivity: NDArrayFloat | None = None
    conc: list[NDArrayFloat | None] = field(default_factory=lambda: [None, None])
    grade: list[NDArrayFloat | None] = field(default_factory=lambda: [None, None])

    def column(self, idx: int, name: str) -> NDArrayFloat | None:
        """Return the direction ``idx`` of the field ``name`` (None if not adjusted)."""
        arr = getattr(self, name)
        return None if arr is None else arr[..., idx]


_ATTRIBUTES = {
    ParameterName.PERMEABILITY: "permeability",
    ParameterName.STORAGE_COEFFICIENT: "storage_coefficient",
    ParameterName.INITIAL_HEAD: "head",
    ParameterName.POROSITY: "porosity",
    ParameterName.DIFFUSION: "diffusion",
    ParameterName.DISPERSIVITY: "dispersivity",
}


def _add(current: NDArrayFloat | None, new: NDArrayFloat) -> NDArrayFloat:
    return new if current is None else current + new


def get_directions(
    model: ForwardModel,
    parameters_to_adjust: AdjustableParameters,
    vecs: NDArrayFloat,
) -> FSMDirections:
    """
    Convert the vectors in the preconditioned space to perturbations of the model.

    The vectors are the derivatives of the preconditioned parameters, concatenated
    in the order of the adjusted parameters. They are converted to the derivatives
    of the parameters through the derivative of the backtransformation.

    Parameters
    ----------
    model : ForwardModel
        The forward model, which holds the current values of the parameters.
    parameters_to_adjust : AdjustableParameters
        The adjusted parameters.
    vecs : NDArrayFloat
        Vectors with shape (:math:`N_s`, :math:`N_e`).

    Returns
    -------
    FSMDirections
        The perturbations of the model parameters.

    Raises
    ------
    ValueError
        If the first dimension of ``vecs`` does not match the number of adjusted
        values.
    NotImplementedError
        If a parameter is not supported by the forward sensitivity method.
    """
    if vecs.ndim != 2:
        raise ValueError(f"`vecs` must be a 2D array, got {vecs.ndim} dimension(s).")
    shape = model.grid.shape
    directions = FSMDirections(ne=vecs.shape[1])
    idx = 0
    for param in object_or_object_sequence_to_list(parameters_to_adjust):
        size = int(np.prod(param.values.shape))
        block = vecs[idx : idx + size, :]
        if block.shape[0] != size:
            raise ValueError(
                f"`vecs` has {vecs.shape[0]} rows but more values are adjusted."
            )
        idx += size
        # derivative of the backtransformation (diagonal)
        values = param.values.ravel("F")
        factor = param.preconditioner.dbacktransform_vec(
            param.preconditioner(values), np.ones(size)
        )
        arr = (factor[:, np.newaxis] * block).reshape((*shape, -1), order="F")
        if param.name == ParameterName.INITIAL_CONCENTRATION:
            directions.conc[param.sp] = _add(directions.conc[param.sp], arr)
        elif param.name == ParameterName.INITIAL_GRADE:
            directions.grade[param.sp] = _add(directions.grade[param.sp], arr)
        elif param.name in _ATTRIBUTES:
            name = _ATTRIBUTES[param.name]
            setattr(directions, name, _add(getattr(directions, name), arr))
        else:
            raise NotImplementedError(
                f'The parameter "{param.name}" is not supported by the FSM.'
            )
    if idx != vecs.shape[0]:
        raise ValueError(
            f"`vecs` has {vecs.shape[0]} rows but {idx} values are adjusted."
        )
    return directions
