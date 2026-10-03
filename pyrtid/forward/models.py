# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""
Provide the data model of the reactive transport system.

The module gathers

- the parameter classes used to configure a simulation (:class:`TimeParameters`,
  :class:`FlowParameters`, :class:`TransportParameters`,
  :class:`GeochemicalParameters`),
- the source terms and the boundary conditions,
- the flow and transport models, which hold the results of the simulation as lists
  of arrays (one entry per time, ``l*`` attributes) and as read-only properties
  returning the same data as a single array with time as last dimension,
- the :class:`ForwardModel` which aggregates all of the above.

Note
----
- The properties returning arrays (``head``, ``mob``, ``u_darcy_x``, ...) copy the
  whole simulation history. They are meant for post-processing; the solvers use the
  lists (``lhead``, ``lmob``, ...) and the ``*_sample`` methods instead.
- The grid is composed of regular grid cells.
- The timestep is variable (see :class:`TimeParameters`).
"""

from __future__ import annotations

import copy
import types
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from quickpaver import (
    RectilinearGrid,
    rlg_nn_to_idx,
    span_to_node_numbers_3d,
)
from scipy import sparse
from scipy.sparse import lil_array
from scipy.sparse.linalg import LinearOperator, SuperLU

from pyrtid.utils import (
    NDArrayBool,
    NDArrayFloat,
    NDArrayInt,
    StrEnum,
    object_or_object_sequence_to_list,
)

GRAVITY = 9.81
WATER_DENSITY = 997
WATER_MW = 0.01801528  # kg/mol
TDS_LINEAR_COEFFICIENT = 1.0  # -> for the densities calculation
H_PLUS_CONC = 1.00603e-07  # mol/l -> for the densities calculation
SMALL_NUMBER = 1e-10
VERY_SMALL_NUMBER = 1e-30


class TimeParameters:
    """
    Class defining the time parameters used in the simulation.

    It also handles the variable timestep.

    Attributes
    ----------
    duration : float
        Desired duration of the simulation.Duration
    dt : float
        Current timestep in seconds.
    dt_init : float
        Initial timestep in seconds.
    dt_min : Optional[float]
        Minimum timestep in seconds.
    dt_max : Optional[float]
        Maximum timestep in seconds.
    courant_factor: float
        The timestep is generally limited to some maximum value by the flow and
        transport models, to assure numerical stability. The Courant-Friedlichs-
        Lewy-Factor is a relaxation parameter for the maximum timestep.
        Reactive systems often allow to relax the (very restrictive) maximum timestep,
        imposed by the transport model. For values greater than 1, the timestep re-
        striction will be relaxed. On the contrary, the restriction will be tightened
        for values inferior to 1.
        Reactive systems often allow to relax the maximum timestep by a factor 5,
        10 or even 20. Using this option, however, may be dangerous and possibly
        lead to failure of the model. Always test the results obtained against a case
        without this parameter set.
        The default is 1.0.
    ldt: List[float]
        List of successive timesteps (in seconds) used in the forward modelling.
    nts: int
        Number of timesteps in the simulation.
    nt: int
        Number of times in the simulation (nts + 1).
    nfpi: int
        Number of fixed point iterations used in the last time iteration.
    lnfpi:
        List of the number of fixed point iterations used for each time iteration.
        This list should have the same length as `ldt`.
    times: NDArrayFloat
        Array of times in second from 0 to t_max."

    """

    __slots__ = [
        "duration",
        "dt",
        "dt_init",
        "dt_min",
        "dt_max",
        "ldt",
        "nfpi",
        "lnfpi",
        "courant_factor",
    ]

    def __init__(
        self,
        duration: float,
        dt_init: float,
        dt_min: Optional[float] = None,
        dt_max: Optional[float] = None,
        courant_factor: float = 1.0,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        duration : float
            Desired duration of the simulation in seconds.
        dt_init : float
            Initial timestep in seconds. It is clipped to ``[dt_min, dt_max]``.
        dt_min : Optional[float], optional
            Minimum timestep in seconds. If None, it is set to the (clipped) initial
            timestep, which means that the timestep cannot decrease.
        dt_max : Optional[float], optional
            Maximum timestep in seconds. If None, it is set to the (clipped) initial
            timestep, which means that the timestep cannot increase. Hence, if
            neither ``dt_min`` nor ``dt_max`` is given, the timestep is fixed.
        courant_factor : float, optional
            Relaxation of the maximum timestep given by the CFL condition,
            by default 1.0.

        Raises
        ------
        ValueError
            If ``dt_min`` is above ``dt_max``.
        """
        self.duration = duration

        # Check dt_min and dt_max consistency
        if dt_min is not None and dt_max is not None and dt_min > dt_max:
            raise ValueError(f"dt_min ({dt_min}) is above dt_max ({dt_max})!")

        # Apply bounds to the initial timestep
        _dt_init = dt_init
        if dt_max is not None:
            _dt_init = min(_dt_init, dt_max)
        if dt_min is not None:
            _dt_init = max(_dt_init, dt_min)

        self.dt_min: float = dt_min if dt_min is not None else _dt_init
        self.dt_max: float = dt_max if dt_max is not None else _dt_init
        self.courant_factor: float = courant_factor
        self.dt_init: float = _dt_init
        self.dt: float = _dt_init
        self.nfpi: int = 0
        self.ldt: List[float] = []
        self.lnfpi: List[int] = []

    @property
    def time_elapsed(self) -> float:
        """Time elapsed in the simulation (sum of the timesteps), in seconds."""
        return float(np.sum(self.ldt))

    @property
    def nts(self) -> int:
        """
        Number of timesteps (dt).

        It is the number of times (`nt`) - 1.
        """
        return len(self.ldt)

    @property
    def nt(self) -> int:
        """
        Number of times (including t0).

        It is the number of timesteps (`nts`) +1.
        """
        return self.nts + 1

    @property
    def times(self) -> NDArrayFloat:
        """Return all the times in second from 0 to t_max."""
        return np.cumsum([0] + self.ldt)

    def reset_to_init(self) -> None:
        """Empty the list of timesteps and set dt to its initial value."""
        self.dt = self.dt_init
        self.ldt = []
        self.lnfpi = []

    def save_dt(self) -> None:
        "Save the current timestep to the list of timesteps."
        self.ldt.append(self.dt)

    def save_nfpi(self) -> None:
        "Save the current number of fixed point iterations."
        self.lnfpi.append(self.nfpi)

    def update_dt(self, n_iter: float, dt_max_cfl: float, max_fpi: int) -> None:
        """
        Update the timestep.

        The timestep is increased by 2% if the last timestep converged in less than
        ``max_fpi`` iterations, and decreased by 30% otherwise. It is then bounded
        by the CFL condition and by ``[dt_min, dt_max]``.

        Parameters
        ----------
        n_iter: int
            Number of iterations required to solve the last timestep. Pass
            ``max_fpi`` (or more) to force a decrease of the timestep.
        dt_max_cfl: float
            Maximum timestep according to the CFL.
        max_fpi: int
            Maximum number of fixed point iterations per timestep.
        """
        if n_iter < max_fpi:
            # increase dt by 2%
            self.dt *= 1.02
        else:
            # decrease dt by 30%
            self.dt *= 0.7

        # Ensure CFL respect
        if self.dt > dt_max_cfl:
            self.dt = dt_max_cfl

        # Ensure timebounds
        if self.dt < self.dt_min:
            self.dt = self.dt_min
        if self.dt > self.dt_max:
            self.dt = self.dt_max

    def get_dt_max_cfl(self, model: ForwardModel, time_index: int) -> float:
        """
        Get the maximum timestep to respect the CFL condition.

        Parameters
        ----------
        model : ForwardModel
            The model, which holds the velocity field and the porosity.
        time_index : int
            Index of the time at which the velocity field is evaluated.

        Returns
        -------
        float
            The maximum timestep in seconds (the courant factor is applied).
        """
        return float(
            np.min(
                self.courant_factor
                * model.tr_model.porosity
                * model.get_ij_over_u(time_index)
            )
        )


class FlowRegime(StrEnum):
    """Flow regime: stationary (initial equilibrium) or transient."""

    STATIONARY = "stationary"
    TRANSIENT = "transient"


class VerticalAxis(StrEnum):
    """Axis of the grid which is vertical (only matters with the gravity)."""

    X = "x"
    Y = "y"
    Z = "z"

    @property
    def axis_index(self) -> int:
        """Return the index of the axis (0 for x, 1 for y, 2 for z)."""
        return {"x": 0, "y": 1, "z": 2}[self.value]


class FlowParameters:
    """
    Class defining the flow parameters used in the simulation.

    Attributes
    ----------
    permeability: float, optional
        Default permeability in the grid (m/s). The default is 1.e-4 m/s.
    storage_coefficient: float, optional
        The default storage coefficient in the grid ($m^{-1}$).
        The default is 1.0 $m^{-1}$.
    crank_nicolson: float
        The Crank-Nicolson parameter allows to set the temporal resolution
        scheme to explicit, fully implicit or somewhere in between these two
        extremes. The value must be comprised between 0.0 and 1.0, 0.0 being a
        full explicit scheme and 1.0 fully implicit. The default is 1.0.
    regime: FlowRegime
        Whether the initial heads are equilibrated with the sources and the
        boundary conditions (stationary) or not (transient). The default is
        stationary.
    is_gravity: bool, optional
        Whether the gravity is taken into account, i.e. density driven flow.
        The default is False.
    vertical_axis: VerticalAxis
        Define which axis is the vertical one. It only affects if the gravity is
        enabled. The default is the z axis.
    rtol: float, optional
        The relative tolerance of the iterative linear solver (GMRES) used for the
        flow. The default is 1e-8.
    """

    def __init__(
        self,
        permeability: float = 1e-4,
        storage_coefficient: float = 1.0,
        crank_nicolson: float = 1.0,
        regime: FlowRegime = FlowRegime.STATIONARY,
        is_gravity: bool = False,
        vertical_axis: VerticalAxis = VerticalAxis.Z,
        rtol: float = 1e-8,
    ) -> None:
        """
        Initialize the instance.

        See the class docstring for the description of the parameters.
        """
        self.permeability: float = permeability
        self.storage_coefficient: float = storage_coefficient
        self.crank_nicolson: float = crank_nicolson
        self.regime: FlowRegime = regime
        self.is_gravity: bool = is_gravity
        self.vertical_axis: VerticalAxis = vertical_axis
        self.rtol: float = rtol


class TransportParameters:
    """
    Class defining the transport parameters used in the simulation.

    Attributes
    ----------
    diffusion: float, optional
        Default diffusion coefficient in the grid in [m2/s]. The default is 1e-4 m2/s.
    dispersivity: float, optional
        The dispersivity (kinematic and numeric) in meters. The default is 0.1 m.
    porosity: float, optional
        Default porosity in the grid Should be a number between 0 and 1.
        The default is 1.0.
    crank_nicolson_advection: float
        The Crank-Nicholson parameter of the advection term allows to set the
        temporal resolution scheme to explicit, fully implicit or somewhere in
        between these two extremes. The value must be comprised between 0.0 and 1.0,
        0.0 being a full explicit scheme and 1.0 fully implicit. The default is 0.5.
    crank_nicolson_diffusion: float
        Same as above, for the diffusion/dispersion term. The default is 1.0.
    rtol: float, optional
        The relative tolerance of the iterative linear solver (GMRES) used for the
        transport. The default is 1e-8.
    is_numerical_acceleration: bool, optional
        Whether to use the chemical source term from the previous iteration (at t=n-1)
        as a first guess in the transport equation (only apply to the first coupling
        fixed point iteration). In practise it might save one iteration (transport-
        chemistry) or more if the system is in a quasi steady-state and it might also
        reduce the overall coupling error. However if the timestep is large or the
        system unstable (stiff), it might lead to non-convergence as well. For more
        information, refer to
        :cite:`lagneauOperatorsplittingbasedReactiveTransport2010`.
        The default is False.
    is_skip_rt: bool
        Whether to skip the reactive-transport step, considering only the flow problem.
    fpi_eps: float
       Tolerance on the transport-chemistry coupling error. The default value is 1e-5.
    max_fpi: int
        Maximum number of fixed point iterations per timestep. If this number is
        exceeded then the numerical acceleration is temporarily disabled and the
        timestep is solved again. If it is already disabled, the timestep is
        reduced. The default is 20.
    """

    def __init__(
        self,
        diffusion: float = 1e-4,
        dispersivity: float = 0.1,
        porosity: float = 1.0,
        crank_nicolson_advection: float = 0.5,
        crank_nicolson_diffusion: float = 1.0,
        rtol: float = 1e-8,
        is_numerical_acceleration: bool = False,
        is_skip_rt: bool = False,
        fpi_eps: float = 1e-5,
        max_fpi: int = 20,
    ) -> None:
        """
        Initialize the instance.

        See the class docstring for the description of the parameters.
        """
        self.diffusion: float = diffusion
        self.dispersivity: float = dispersivity
        self.porosity: float = porosity
        self.crank_nicolson_advection: float = crank_nicolson_advection
        self.crank_nicolson_diffusion: float = crank_nicolson_diffusion
        self.rtol: float = rtol
        self.is_numerical_acceleration: bool = is_numerical_acceleration
        self.is_skip_rt: bool = is_skip_rt
        self.fpi_eps: float = fpi_eps
        self.max_fpi: int = max_fpi


class GeochemicalParameters:
    """
    Class defining the geochemical parameters used in the simulation.

    The chemical system is made of one mineral (immobile species 1) dissolving into
    the mobile species 1 by consuming ``stocoef`` moles of the mobile species 2,
    which is transformed in the immobile species 2.

    Attributes
    ----------
    conc: float, optional
        Initial concentration of the mobile species 1 (tracer) in the grid in molal.
        The default is 1e-10.
    conc2: float, optional
        Initial concentration of the mobile species 2 (reagent) in the grid in molal.
        The default is 1e-10.
    grade: float, optional
        Default grade of the immobile species 1 (the mineral) in the grid in mol/kg
        (kg of water). The default is 1e-10.
    grade2: float, optional
        Default grade of the immobile species 2 in the grid in mol/kg (kg of water).
        The default is 1e-10.
    kv: float, optional
        The kinetic rate of the mineral in [mol/m2/s]. It is negative for a
        dissolution. The default is -6.9e-9.
    As: float, optional
        Specific area in [m2/mol]. The default is 13.5.
    Ks: float, optional
        Solubility constant (no unit). The default is 6.3e-4.
    Ms: float, optional
        Molar mass of the mobile species 1 in g/mol. The default is 270.
    Ms2: float, optional
        Molar mass of the mobile species 2 in g/mol. The default is 270.
    stocoef: float
        Number of mole of species 2 consumed when dissolving the mineral.
        The default is 1.0.
    use_explicit_formulation: bool
        Whether to use the explicit formulation of the chemistry, otherwise the
        (more expensive) implicit formulation is used. See
        :mod:`pyrtid.forward.geochem_solver`. The default is True.
    """

    def __init__(
        self,
        conc: float = SMALL_NUMBER,
        conc2: float = SMALL_NUMBER,
        grade: float = SMALL_NUMBER,
        grade2: float = SMALL_NUMBER,
        kv: float = -6.9e-9,
        As: float = 13.5,
        Ks: float = 6.3e-4,
        Ms: float = 270,
        Ms2: float = 270,
        stocoef: float = 1.0,
        use_explicit_formulation: bool = True,
    ) -> None:
        """
        Initialize the instance.

        See the class docstring for the description of the parameters.
        """
        self.conc: float = conc
        self.conc2: float = conc2
        self.grade: float = grade
        self.grade2: float = grade2
        self.kv: float = kv
        self.As: float = As
        self.Ks: float = Ks
        self.Ms: float = Ms
        self.Ms2: float = Ms2
        self.stocoef: float = stocoef
        self.use_explicit_formulation: bool = use_explicit_formulation


class SourceTerm:
    """
    Define a source term object.

    A source term (e.g. a well) can pump or inject, with piecewise constant
    flowrates and concentrations.

    Attributes
    ----------
    name: str
        Name of the instance.
    node_ids: NDArrayInt
        Node numbers of the grid cells where the source term applies. The flowrate
        is equally distributed between them.
    times: NDArrayFloat
        Times (in seconds, sorted in ascending order) at which the flowrates and
        concentrations change. Before the first time, the source term is inactive.
    flowrates: NDArrayFloat
        Sequence of flowrates (m3/s), one per time. Positive = injection,
        negative = pumping.
    concentrations: NDArrayFloat
        Concentrations (mol/l) of the injected species, one row per time. Used only
        if the flowrate is positive.
    """

    __slots__ = [
        "name",
        "node_ids",
        "times",
        "flowrates",
        "concentrations",
    ]

    def __init__(
        self,
        name: str,
        node_ids: NDArrayInt,
        times: NDArrayFloat,
        flowrates: NDArrayFloat,
        concentrations: NDArrayFloat,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        name: str
            Name of the instance.
        node_ids: NDArrayInt
            Node numbers of the grid cells where the source term applies.
        times: NDArrayFloat
            Times (s) at which the flowrates and concentrations change, sorted in
            ascending order. With dimension (nt,).
        flowrates: NDArrayFloat
            Sequence of flowrates (m3/s). Positive = injection, negative = pumping.
            With dimension (nt,).
        concentrations: NDArrayFloat
            Concentration, used only if flowrates is positive (mol/l).
            With dimension (nt, n_sp) (or (nt,) for a single species).

        Raises
        ------
        ValueError
            If ``times``, ``flowrates`` and ``concentrations`` do not have the same
            number of times.
        """
        self.name = name
        self.node_ids = np.array(node_ids).reshape(-1)
        self.times = np.array(times).reshape(-1)
        self.flowrates = np.array(flowrates).reshape(-1)
        _conc = np.array(concentrations)
        self.concentrations = _conc.reshape(_conc.shape[0], -1)

        if (
            self.concentrations.shape[0] != self.times.size
            or self.flowrates.size != self.times.size
        ):
            raise ValueError(
                "Times, flowrates and concentrations must have the same dimension !"
            )

    def get_node_indices(self, grid: RectilinearGrid) -> NDArrayInt:
        """Return the node indices."""
        return np.array(rlg_nn_to_idx(self.node_ids, nx=grid.nx, ny=grid.ny)).reshape(
            3, -1
        )

    @property
    def n_nodes(self) -> int:
        """Return the number of nodes."""
        return np.size(self.node_ids)

    def get_values(self, time: float) -> Tuple[float, NDArrayFloat]:
        """
        Return the flowrate and the concentrations for a given time.

        The values are piecewise constant: those of the last ``times`` entry which is
        lower or equal to ``time`` are returned. This is matching the "modify"
        process behavior of HYTEC.

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        Tuple[float, NDArrayFloat]
            The flowrate (m3/s) and the concentrations (mol/l, one per species). Both
            are zero before the first time.
        """
        if time < self.times[0]:
            return 0.0, 0.0
        time_index = int(np.searchsorted(self.times, time, side="right")) - 1
        return self.flowrates[time_index], self.concentrations[time_index]


@dataclass
class BoundaryCondition(ABC):
    """
    Represent a boundary condition.

    Parameters
    ----------
    span: slice
        The span over which the condition applies.
    """

    span: Union[NDArrayInt, Tuple[slice, slice, slice]]


@dataclass
class ConstantHead(BoundaryCondition):
    """
    Represent a constant head condition (Dirichlet).

    Parameters
    ----------
    span: slice
        The span over which the condition applies.
    values: Union[float, NDArrayFloat]
        The values to set.
    """

    span: Union[NDArrayInt, Tuple[slice, slice, slice], slice]
    values: Union[float, NDArrayFloat]


@dataclass
class ConstantConcentration(BoundaryCondition):
    """
    Represent a constant conentration boundary condition (Dirichlet).

    Parameters
    ----------
    span: slice
        The span over which the condition applies.
    values: Union[float, NDArrayFloat]
        The values to set.
    """

    span: Union[NDArrayInt, Tuple[slice, slice, slice], slice]
    values: Union[float, NDArrayFloat]


@dataclass
class ZeroConcGradient(BoundaryCondition):
    """
    Represent a zero conentration gradient boundary condition (Neumann).

    Parameters
    ----------
    span: slice
        The span over which the condition applies.
    """

    span: Union[NDArrayInt, Tuple[slice, slice, slice], slice]


def _get_a_not_in_b_1d(a: NDArrayInt, b: NDArrayInt) -> NDArrayInt:
    """Return the elements of a not found in b sorted by ascending order."""
    # handle the case with an empty b
    if b.size == 0 or a.size == 0:
        return a
    return np.sort(a[np.isin(a, b, invert=True)])


def _get_free_node_numbers(n_nodes: int, constant_nn: NDArrayInt) -> NDArrayInt:
    """
    Return the node numbers (sorted) which are not in ``constant_nn``.

    Parameters
    ----------
    n_nodes : int
        Total number of nodes (grid cells).
    constant_nn : NDArrayInt
        Node numbers of the nodes with a constant value (boundary conditions).
    """
    is_free = np.ones(n_nodes, dtype=bool)
    is_free[constant_nn] = False
    return np.flatnonzero(is_free).astype(np.int32)


class FlowModel(ABC):
    """
    Represent a flow model.

    The simulated fields are stored in lists with one entry per time (``lhead``,
    ``lpressure``, ``lu_darcy_x``, ...) and are also available as read-only
    properties (``head``, ``pressure``, ``u_darcy_x``, ...) returning arrays with
    time as last dimension.

    The faces velocities (``u_darcy_*``) have one more value than the grid has cells
    along their axis.
    """

    __slots__ = [
        "_vertical_pos",
        "vertical_axis",
        "vertical_mesh_size",
        "crank_nicolson",
        "storage_coefficient",
        "permeability",
        "lhead",
        "lpressure",
        "lu_darcy_x",
        "lu_darcy_y",
        "lu_darcy_z",
        "lu_darcy_div",
        "lunitflow",
        "boundary_conditions",
        "cst_head_nn",
        "regime",
        "q_prev_no_dt",
        "q_next_no_dt",
        "q_prev",
        "q_next",
        "rtol",
        "west_boundary_idx",
        "east_boundary_idx",
        "north_boundary_idx",
        "south_boundary_idx",
        "top_boundary_idx",
        "bottom_boundary_idx",
        "is_save_spmats",
        "l_q_next",
        "l_q_prev",
        "is_save_spilu",
        "super_ilu",
        "preconditioner",
    ]

    def __init__(
        self,
        grid: RectilinearGrid,
        time_params: TimeParameters,
        fl_params: FlowParameters,
    ) -> None:
        """Initialize the instance."""
        self.crank_nicolson: float = fl_params.crank_nicolson
        self.storage_coefficient: NDArrayFloat = (
            np.ones(grid.shape, dtype=np.float64) * fl_params.storage_coefficient
        )
        self.regime: FlowRegime = fl_params.regime
        self.permeability: NDArrayFloat = (
            np.ones(grid.shape, dtype=np.float64) * fl_params.permeability
        )

        self.lu_darcy_x: List[NDArrayFloat] = []
        self.lu_darcy_y: List[NDArrayFloat] = []
        self.lu_darcy_z: List[NDArrayFloat] = []
        self.lu_darcy_div: List[NDArrayFloat] = []
        self.lunitflow: List[NDArrayFloat] = []

        self.boundary_conditions: List[BoundaryCondition] = []
        self.q_prev_no_dt = lil_array((grid.n_grid_cells, grid.n_grid_cells))
        self.q_next_no_dt = lil_array((grid.n_grid_cells, grid.n_grid_cells))
        self.q_prev = lil_array((grid.n_grid_cells, grid.n_grid_cells))
        self.q_next = lil_array((grid.n_grid_cells, grid.n_grid_cells))
        self.cst_head_nn: NDArrayInt = np.array([], dtype=np.int64)
        self.rtol = fl_params.rtol
        self.vertical_axis = fl_params.vertical_axis
        self.vertical_mesh_size = {
            VerticalAxis.X: grid.dx,
            VerticalAxis.Y: grid.dy,
            VerticalAxis.Z: grid.dz,
        }[fl_params.vertical_axis]

        # Indices of the constant head cells located on the domain borders (empty by
        # default). Arrays with shape (2, n), which are updated with
        # `set_constant_head_indices`.
        self.west_boundary_idx: NDArrayInt = np.empty((2, 0), dtype=np.int64)
        self.east_boundary_idx: NDArrayInt = np.empty((2, 0), dtype=np.int64)
        self.south_boundary_idx: NDArrayInt = np.empty((2, 0), dtype=np.int64)
        self.north_boundary_idx: NDArrayInt = np.empty((2, 0), dtype=np.int64)
        self.bottom_boundary_idx: NDArrayInt = np.empty((2, 0), dtype=np.int64)
        self.top_boundary_idx: NDArrayInt = np.empty((2, 0), dtype=np.int64)

        # Cache of the vertical position of the grid cell centers
        self._vertical_pos: Optional[NDArrayFloat] = None

        # These are list of ndarrays
        self.lhead: List[NDArrayFloat] = [np.zeros(grid.shape, dtype=np.float64)]

        # TODO: provide the initial density
        self.lpressure: List[NDArrayFloat] = [
            (
                np.zeros(grid.shape, dtype=np.float64)
                - self._get_mesh_center_vertical_pos()
            )
            * GRAVITY
            * WATER_DENSITY
        ]

        # List to store the successive stiffness matrices
        # This is mostly for development purposes.
        # only activated with the adjoint state or specific devs.
        self.is_save_spmats: bool = False
        self.l_q_next: List[lil_array] = []
        self.l_q_prev: List[lil_array] = []

        # preconditioner (LU) for q_next, only useful to store with the forward
        # sensivitiy approach.
        self.is_save_spilu: bool = False
        self.super_ilu: Optional[SuperLU] = None
        self.preconditioner: Optional[LinearOperator] = None

    @property
    def head(self) -> NDArrayFloat:
        """
        Return head [m] as array with dimension (nx, ny, nz, nt).

        This is read-only.
        """
        return np.transpose(np.array(self.lhead), axes=(1, 2, 3, 0))

    @property
    def pressure(self) -> NDArrayFloat:
        """
        Return pressure [Pa] as array with dimension (nx, ny, nz, nt).

        This is read-only.
        """
        return np.transpose(np.array(self.lpressure), axes=(1, 2, 3, 0))

    @property
    def u_darcy_x(self) -> NDArrayFloat:
        """
        Return x-darcy velocities as array with dimension (nx, ny, nz, nt + 1).

        This is read-only.
        """
        return np.transpose(np.array(self.lu_darcy_x), axes=(1, 2, 3, 0))

    @property
    def u_darcy_y(self) -> NDArrayFloat:
        """
        Return y-darcy velocities as array with dimension (nx, ny, nz, nt + 1).

        This is read-only.
        """
        return np.transpose(np.array(self.lu_darcy_y), axes=(1, 2, 3, 0))

    @property
    def u_darcy_z(self) -> NDArrayFloat:
        """
        Return z-darcy velocities as array with dimension (nx, ny, nz, nt + 1).

        This is read-only.
        """
        return np.transpose(np.array(self.lu_darcy_z), axes=(1, 2, 3, 0))

    @property
    def u_darcy_div(self) -> NDArrayFloat:
        """
        Return darcy divergence as array with dimension (nx, ny, nz, nt + 1).

        This is read-only.
        """
        return np.transpose(np.array(self.lu_darcy_div), axes=(1, 2, 3, 0))

    @property
    def unitflow(self) -> NDArrayFloat:
        """
        Return flow sources sources as array with dimension (nx, ny, nz, nt + 1).

        This is read-only.
        """
        return np.transpose(np.array(self.lunitflow), axes=(1, 2, 3, 0))

    def add_boundary_conditions(self, condition: BoundaryCondition) -> None:
        """
        Add a boundary condition to the flow model.

        The head (and the pressure) of the initial state is set to the value of the
        condition over its span.

        Raises
        ------
        ValueError
            If the condition is not a :class:`ConstantHead`.
        """
        if not isinstance(condition, ConstantHead):
            raise ValueError(
                f"{condition} is not a valid boundary condition for the flow model !"
            )
        self.boundary_conditions.append(condition)

        # Set the values (both the head and the pressure, which are used for the
        # constant head cells when the gravity is considered). The constant head node
        # numbers are updated by `set_constant_head_indices`.
        self.set_initial_head(condition.values, condition.span)

    def set_constant_head_indices(self) -> None:
        """
        Set the indices of nodes with constant head.

        It also identifies the constant head cells which are on the borders of the
        domain (``west_boundary_idx``, ``east_boundary_idx``, ...).
        """
        node_numbers = np.array([], dtype=np.int32)
        nx, ny, nz = self.lhead[0].shape  # type: ignore

        _west_bidx: Set[Tuple[int, int]] = set()
        _east_bidx: Set[Tuple[int, int]] = set()
        _south_bidx: Set[Tuple[int, int]] = set()
        _north_bidx: Set[Tuple[int, int]] = set()
        _bottom_bidx: Set[Tuple[int, int]] = set()
        _top_bidx: Set[Tuple[int, int]] = set()

        for condition in self.boundary_conditions:
            if isinstance(condition, ConstantHead):
                # 1) Get the new constant head node numbers
                new_nn: NDArrayInt = span_to_node_numbers_3d(condition.span, nx, ny, nz)
                # 2) add the new nn to the global list of nn
                node_numbers = np.hstack([node_numbers, new_nn])

                # 3) determine if the segment is along one of the 5 borders of the
                # domain. First we start by getting the indices in the grid
                _ix, _iy, _iz = rlg_nn_to_idx(new_nn, nx, ny)
                # The span must be continuous (rectangular group of grid cells),
                # so we can estimate the direction of constant head segment:
                # must be more than 2 values on one of the borders
                # X
                non_zero_west: int = np.count_nonzero(_ix == 0)
                if non_zero_west > 1 or (non_zero_west == 1 and ny == 1 and nz == 1):
                    for iy, iz in zip(_iy, _iz):
                        _west_bidx.add((iy, iz))
                non_zero_east: int = np.count_nonzero(_ix == nx - 1)
                if non_zero_east > 1 or (non_zero_east == 1 and ny == 1 and nz == 1):
                    for iy, iz in zip(_iy, _iz):
                        _east_bidx.add((iy, iz))

                # Y
                non_zero_south: int = np.count_nonzero(_iy == 0)
                if non_zero_south > 1 or (non_zero_south == 1 and nx == 1 and nz == 1):
                    for ix, iz in zip(_ix, _iz):
                        _south_bidx.add((ix, iz))

                non_zero_north: int = np.count_nonzero(_iy == ny - 1)
                if non_zero_north > 1 or (non_zero_north == 1 and nx == 1 and nz == 1):
                    for ix, iz in zip(_ix, _iz):
                        _north_bidx.add((ix, iz))

                # Z
                non_zero_bottom: int = np.count_nonzero(_iz == 0)
                if non_zero_bottom > 1 or (
                    non_zero_bottom == 1 and nx == 1 and ny == 1
                ):
                    for ix, iy in zip(_ix, _iy):
                        _bottom_bidx.add((ix, iy))

                non_zero_top: int = np.count_nonzero(_iz == nz - 1)
                if non_zero_top > 1 or (non_zero_top == 1 and nx == 1 and ny == 1):
                    for ix, iy in zip(_ix, _iy):
                        _top_bidx.add((ix, iy))

        # domain boundary indices to numpy => easier indexing
        self.west_boundary_idx = np.atleast_2d(
            np.array(list(_west_bidx), dtype=np.int64).T
        )
        self.east_boundary_idx = np.atleast_2d(
            np.array(list(_east_bidx), dtype=np.int64).T
        )
        self.south_boundary_idx = np.atleast_2d(
            np.array(list(_south_bidx), dtype=np.int64).T
        )
        self.north_boundary_idx = np.atleast_2d(
            np.array(list(_north_bidx), dtype=np.int64).T
        )
        self.bottom_boundary_idx = np.atleast_2d(
            np.array(list(_bottom_bidx), dtype=np.int64).T
        )
        self.top_boundary_idx = np.atleast_2d(
            np.array(list(_top_bidx), dtype=np.int64).T
        )

        # remove duplicates from the global list
        self.cst_head_nn: NDArrayInt = np.unique(node_numbers.flatten())

    @property
    def cst_head_indices(self) -> NDArrayInt:
        """Return the indices (array with shape (3, n)) of the constant head cells."""
        nx, ny = self.lhead[0].shape[:2]
        return np.array(rlg_nn_to_idx(self.cst_head_nn, nx=nx, ny=ny))

    @property
    def free_head_nn(self) -> NDArrayInt:
        """Return the free head node numbers."""
        return _get_free_node_numbers(self.lhead[0].size, self.cst_head_nn)

    @property
    def free_head_indices(self) -> NDArrayInt:
        """Return the indices (array with shape (3, n)) of the free head cells."""
        nx, ny = self.lhead[0].shape[:2]
        return np.array(rlg_nn_to_idx(self.free_head_nn, nx=nx, ny=ny))

    def reinit(self) -> None:
        """Reset all the results, but keep the initial conditions (first time)."""
        self.lhead = self.lhead[:1]
        self.lpressure = self.lpressure[:1]
        self.lu_darcy_x = []
        self.lu_darcy_y = []
        self.lu_darcy_z = []
        self.lu_darcy_div = []
        self.lunitflow = []
        self.set_constant_head_indices()
        self.l_q_next = []
        self.l_q_prev = []
        self.super_ilu = None
        self.preconditioner = None

    def _get_center_weights(self, axis: int) -> NDArrayFloat:
        """
        Return the weights to average the face velocities at the cell centers.

        The velocity at a cell center along ``axis`` is the sum of the velocities of
        its two faces multiplied by these weights, with shape (nx, ny, nz):

        - 1/2 for the cells which are not on the border of the domain,
        - 1 for the cells on the border of the domain: one of their faces has no
          flow, except for the constant head cells, for which the evacuation flow is
          reported on the border face. The weight is then 1/2 there as well.
        """
        weights = np.ones(self.lhead[0].shape, dtype=np.float64)
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = 0
        upper[axis] = -1
        interior = [slice(None)] * 3
        interior[axis] = slice(1, -1)
        weights[tuple(interior)] = 0.5

        # (low border, high border) indices of the constant head cells
        borders = {
            0: (self.west_boundary_idx, self.east_boundary_idx),
            1: (self.south_boundary_idx, self.north_boundary_idx),
            2: (self.bottom_boundary_idx, self.top_boundary_idx),
        }[axis]
        for border_slicer, idx in zip((lower, upper), borders):
            if idx.size == 0:
                continue
            # idx holds the indices along the two other axes (in increasing order)
            slicer = list(border_slicer)
            other_axes = [a for a in range(3) if a != axis]
            slicer[other_axes[0]] = idx[0]
            slicer[other_axes[1]] = idx[1]
            weights[tuple(slicer)] *= 0.5
        return weights

    def _get_u_darcy_center_sample(self, axis: int, time_index: int) -> NDArrayFloat:
        """
        Return the darcy velocity along ``axis`` at the cell centers for one time.

        Parameters
        ----------
        axis : int
            0 for x, 1 for y, 2 for z.
        time_index : int
            Index of the time.

        Returns
        -------
        NDArrayFloat
            Array with shape (nx, ny, nz).
        """
        u_faces = (self.lu_darcy_x, self.lu_darcy_y, self.lu_darcy_z)[axis][time_index]
        low = [slice(None)] * 3
        high = [slice(None)] * 3
        low[axis] = slice(None, -1)
        high[axis] = slice(1, None)
        return (u_faces[tuple(low)] + u_faces[tuple(high)]) * self._get_center_weights(
            axis
        )

    @property
    def u_darcy_x_center(self) -> NDArrayFloat:
        """The darcy x-velocities estimated at the mesh centers (nx, ny, nz, nt)."""
        return np.stack(
            [self._get_u_darcy_center_sample(0, t) for t in range(len(self.lhead))],
            axis=-1,
        )

    @property
    def u_darcy_y_center(self) -> NDArrayFloat:
        """The darcy y-velocities estimated at the mesh centers (nx, ny, nz, nt)."""
        return np.stack(
            [self._get_u_darcy_center_sample(1, t) for t in range(len(self.lhead))],
            axis=-1,
        )

    @property
    def u_darcy_z_center(self) -> NDArrayFloat:
        """The darcy z-velocities estimated at the mesh centers (nx, ny, nz, nt)."""
        return np.stack(
            [self._get_u_darcy_center_sample(2, t) for t in range(len(self.lhead))],
            axis=-1,
        )

    @property
    def u_darcy_norm(self) -> NDArrayFloat:
        """The norm of the darcy velocity estimated at the center of the mesh."""
        return self.get_u_darcy_norm()

    def get_u_darcy_norm_sample(self, time_index: int) -> NDArrayFloat:
        """
        The norm of the darcy velocity estimated at the center of the grid cell.

        Parameters
        ----------
        time_index : int
            Index of the time.

        Returns
        -------
        NDArrayFloat
            Array with shape (nx, ny, nz).
        """
        return np.sqrt(
            self._get_u_darcy_center_sample(0, time_index) ** 2
            + self._get_u_darcy_center_sample(1, time_index) ** 2
            + self._get_u_darcy_center_sample(2, time_index) ** 2
        )

    def get_du_darcy_norm_sample(
        self, time_index: int
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """
        Return the derivatives of the cell-center velocity norm for one time.

        Returns
        -------
        Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]
            The derivatives of the norm of the velocity at the grid cell centers with
            respect to the *face* velocities, ``(d|U|/dUx, d|U|/dUy, d|U|/dUz)``,
            each with shape (nx, ny, nz). The derivatives are null where the norm is
            null. They are the same for the two faces of a grid cell, up to the
            weights ``_get_center_weights``.
        """
        weights = [self._get_center_weights(axis) for axis in range(3)]
        centers = [
            self._get_u_darcy_center_sample(axis, time_index) for axis in range(3)
        ]
        norm = np.sqrt(centers[0] ** 2 + centers[1] ** 2 + centers[2] ** 2)

        # inverse of the norm -> avoid division by zero
        inv_norm = np.zeros_like(norm)
        mask = norm > 0.0
        inv_norm[mask] = 1.0 / norm[mask]

        # return (d|U|/dUx , d|U|/dUy, d|U|/dUz)
        return tuple(inv_norm * c * w for c, w in zip(centers, weights))  # type: ignore

    def get_u_darcy_norm(self) -> NDArrayFloat:
        """
        The norm of the darcy velocity estimated at the center of the grid cells.

        Returns
        -------
        NDArrayFloat
            Array with shape (nx, ny, nz, nt).
        """
        return np.stack(
            [self.get_u_darcy_norm_sample(t) for t in range(len(self.lhead))], axis=-1
        )

    @property
    def vertical_axis_index(self) -> int:
        """Return the index of the vertical axis (0 for x, 1 for y, 2 for z)."""
        return VerticalAxis(self.vertical_axis).axis_index

    def get_vertical_dim(self) -> int:
        """Return the number of voxel along the vertical_axis axis."""
        return self.lhead[0].shape[self.vertical_axis_index]

    def _get_mesh_center_vertical_pos(self) -> NDArrayFloat:
        """
        Return the vertical position of the grid cells centers.

        The result is cached (a copy is returned).
        """
        if self._vertical_pos is None:
            axis = self.vertical_axis_index
            shape = self.lhead[0].shape
            pos = (np.arange(shape[axis]) + 0.5) * self.vertical_mesh_size
            bshape = [1, 1, 1]
            bshape[axis] = shape[axis]
            self._vertical_pos = np.broadcast_to(pos.reshape(bshape), shape).copy()
        return self._vertical_pos.copy()

    def get_pressure_pa(self) -> NDArrayFloat:
        """Return the pressure in Pa."""
        return self.pressure

    def get_pressure_bar(self) -> NDArrayFloat:
        """Return the pressure in bar."""
        return self.get_pressure_pa() / 1e5

    def pressure_to_head(
        self,
        pressure: NDArrayFloat,
    ) -> NDArrayFloat:
        """Convert pressure [Pa] to head [m]."""
        return pressure / GRAVITY / WATER_DENSITY + self._get_mesh_center_vertical_pos()

    def head_to_pressure(
        self,
        head: NDArrayFloat,
    ) -> NDArrayFloat:
        """Convert head [m] to pressure [Pa]."""
        return (head - self._get_mesh_center_vertical_pos()) * GRAVITY * WATER_DENSITY

    def set_initial_head(
        self,
        values: Union[float, int, NDArrayInt, NDArrayFloat],
        span: Union[NDArrayInt, Tuple[slice, slice, slice], NDArrayBool] = (
            slice(None),
            slice(None),
            slice(None),
        ),
    ) -> None:
        """Set the initial head field."""
        self.lhead[0][span] = values
        self.lpressure[0][span] = self.head_to_pressure(self.lhead[0])[span]

    def set_initial_pressure(
        self,
        values: Union[float, int, NDArrayInt, NDArrayFloat],
        span: Union[NDArrayInt, Tuple[slice, slice, slice], NDArrayBool] = (
            slice(None),
            slice(None),
            slice(None),
        ),
    ) -> None:
        """Set the initial pressure field in Pa."""
        self.lpressure[0][span] = values
        self.lhead[0][span] = self.pressure_to_head(self.lpressure[0])[span]

    @property
    @abstractmethod
    def is_gravity(self) -> bool:
        """Return False because the gravity effect is ignored with saturated flow."""
        ...


# TODO: make the link with the initial density for the pressure
class SaturatedFlowModel(FlowModel):
    """Flow model of a saturated medium, where the density effects are ignored."""

    __slots__ = [
        "_head",
    ]

    def __init__(
        self,
        grid: RectilinearGrid,
        time_params: TimeParameters,
        fl_params: FlowParameters,
    ) -> None:
        """Initialize the instance."""
        super().__init__(grid, time_params, fl_params)

    @property
    def is_gravity(self) -> bool:
        """Return False because the gravity effect is ignored with saturated flow."""
        return False


class DensityFlowModel(FlowModel):
    """Flow model of a saturated medium with density driven flow (gravity)."""

    __slots__ = ["_pressure", "density"]

    def __init__(
        self,
        grid: RectilinearGrid,
        time_params: TimeParameters,
        fl_params: FlowParameters,
    ) -> None:
        """Initialize the instance."""
        super().__init__(grid, time_params, fl_params)

    @property
    def is_gravity(self) -> bool:
        """Return True because the gravity effect is considered with density flow."""
        return True


class TransportModel:
    """
    Represent a transport (and chemistry) model.

    The simulated fields are stored in lists with one entry per time (``lmob``,
    ``limmob``, ``ldensity``, ...) and are also available as read-only properties
    returning arrays with time as last dimension.

    The first axis of the concentrations arrays is the species. Two mobile species
    are simulated (``n_sp``): the tracer/product (species 0) and the reagent
    (species 1). The immobile species are the minerals (grades).
    """

    __slots__ = [
        "crank_nicolson_diffusion",
        "crank_nicolson_advection",
        "diffusion",
        "dispersivity",
        "porosity",
        "lmob",
        "limmob",
        "lsources",  # this is needed for the adjoint state
        "ldensity",
        "immob_prev",
        "boundary_conditions",
        "cst_conc_nn",
        "q_prev",
        "q_next",
        "rtol",
        "is_numerical_acceleration",
        "is_num_acc_for_timestep",
        "fpi_eps",
        "max_fpi",
        "molar_mass",
        "is_skip_rt",
        "is_save_spmats",
        "l_q_next",
        "l_q_prev",
        "is_save_spilu",
        "super_ilu",
        "preconditioner",
    ]

    def __init__(
        self,
        grid: RectilinearGrid,
        time_params: TimeParameters,
        tr_params: TransportParameters,
        gch_params: GeochemicalParameters,
    ) -> None:
        """Initialize the instance."""
        self.crank_nicolson_diffusion: float = tr_params.crank_nicolson_diffusion
        self.crank_nicolson_advection: float = tr_params.crank_nicolson_advection
        self.diffusion = np.ones(grid.shape, dtype=np.float64) * tr_params.diffusion
        self.dispersivity = (
            np.ones(grid.shape, dtype=np.float64) * tr_params.dispersivity
        )
        self.porosity = np.ones(grid.shape, dtype=np.float64) * tr_params.porosity
        self.lmob: List[NDArrayFloat] = [
            np.zeros((self.n_sp, grid.nx, grid.ny, grid.nz), dtype=np.float64)
        ]
        self.lmob[0][0, :, :, :] = gch_params.conc
        self.lmob[0][1, :, :, :] = gch_params.conc2

        self.limmob: List[NDArrayFloat] = [
            np.zeros((self.n_sp, grid.nx, grid.ny, grid.nz), dtype=np.float64)
        ]
        # For now, only on mineral
        self.limmob[0][0, :, :, :] = gch_params.grade
        self.limmob[0][1, :, :, :] = gch_params.grade2

        self.ldensity: List[NDArrayFloat] = []
        self.lsources: List[NDArrayFloat] = []
        self.immob_prev = np.zeros(
            (self.n_sp, grid.nx, grid.ny, grid.nz), dtype=np.float64
        )
        self.boundary_conditions: List[BoundaryCondition] = []
        # Stiffness matrices of the transport (lil when built, csc once solved)
        self.q_prev: Union[lil_array, sparse.csc_array] = lil_array(
            (grid.n_grid_cells, grid.n_grid_cells)
        )
        self.q_next: Union[lil_array, sparse.csc_array] = lil_array(
            (grid.n_grid_cells, grid.n_grid_cells)
        )
        self.cst_conc_nn: NDArrayInt = np.array([], dtype=np.int64)
        self.rtol: float = tr_params.rtol
        self.is_numerical_acceleration: bool = tr_params.is_numerical_acceleration
        # The numerical acceleration can be temporarily disabled
        self.is_num_acc_for_timestep: bool = self.is_numerical_acceleration
        self.fpi_eps: float = tr_params.fpi_eps
        self.max_fpi: int = tr_params.max_fpi
        self.molar_mass: float = gch_params.Ms
        self.is_skip_rt: bool = tr_params.is_skip_rt

        # List to store the successive stiffness matrices
        # This is mostly for development purposes.
        # only activated with the adjoint state or specific devs.
        self.is_save_spmats: bool = False
        self.l_q_next: List[lil_array] = []
        self.l_q_prev: List[lil_array] = []

        # preconditioner (LU) for q_next, only useful to store with the forward
        # sensivitiy approach.
        self.is_save_spilu: bool = False
        self.super_ilu: Optional[SuperLU] = None
        self.preconditioner: Optional[LinearOperator] = None

    @property
    def mob(self) -> NDArrayFloat:
        """
        Return mobile concentrations as array with dimension (nsp, nx, ny, nz, nt+1).

        This is read-only.
        """
        return np.transpose(np.array(self.lmob), axes=(1, 2, 3, 4, 0))

    @property
    def immob(self) -> NDArrayFloat:
        """
        Return immobile concentrations as array with dimension (nsp, nx, ny, nz, nt+1).

        This is read-only.
        """
        return np.transpose(np.array(self.limmob), axes=(1, 2, 3, 4, 0))

    @property
    def conc(self) -> NDArrayFloat:
        """
        Return the first mobile species as array with dimension (nx, ny, nz, nt + 1).

        This is read-only. Alias for mob[0].
        """
        return self.mob[0]

    @property
    def conc2(self) -> NDArrayFloat:
        """
        Return the second mobile species as array with dimension (nx, ny, nz, nt + 1).

        This is read-only. Alias for mob[1].
        """
        return self.mob[1]

    @property
    def grade(self) -> NDArrayFloat:
        """
        Return the first immobile species as array with dimension (nx, ny, nz, nt + 1).

        This is read-only. Alias for immob[0].
        """
        return self.immob[0]

    @property
    def grade2(self) -> NDArrayFloat:
        """
        Return the second immobile species as array with dimension
        (nx, ny, nz, nt + 1).

        This is read-only. Alias for immob[1].
        """
        return self.immob[1]

    @property
    def density(self) -> NDArrayFloat:
        """
        Return densities in kg/m3 as array with dimension (nx, ny, nz, nt + 1).

        This is read-only. It is empty if the simulation has not been run.
        """
        if len(self.ldensity) == 0:
            return np.array([])
        return np.transpose(np.array(self.ldensity), axes=(1, 2, 3, 0))

    @property
    def sources(self) -> NDArrayFloat:
        """
        Return concentration sources as array with dimension (2, nx, ny, nz, nt + 1).

        This is read-only.
        """
        return np.transpose(np.array(self.lsources), axes=(1, 2, 3, 4, 0))

    @property
    def effective_diffusion(self) -> NDArrayFloat:
        """Return the effective diffusion (diffusion * porosity)."""
        return self.diffusion * self.porosity

    @property
    def n_sp(self) -> int:
        """
        Return the number of mobile species in the system.

        This is hard-coded for now.
        """
        return 2

    def set_initial_grade(
        self,
        values: Union[float, int, NDArrayInt, NDArrayFloat],
        sp: Optional[int] = 0,
        span: Union[NDArrayInt, Tuple[slice, slice, slice], NDArrayBool] = (
            slice(None),
            slice(None),
            slice(None),
        ),
    ) -> None:
        """Set the initial grades."""
        self.limmob[0][sp][span] = values

    def set_initial_conc(
        self,
        values: Union[float, int, NDArrayInt, NDArrayFloat],
        sp: int = 0,
        span: Union[NDArrayInt, Tuple[slice, slice, slice], NDArrayBool] = (
            slice(None),
            slice(None),
            slice(None),
        ),
    ) -> None:
        """Set the initial concentrations."""
        self.lmob[0][sp][span] = values

    def add_boundary_conditions(self, condition: BoundaryCondition) -> None:
        """
        Add a boundary condition to the transport model.

        Note
        ----
        The grid cells of a :class:`ConstantConcentration` condition keep the
        concentration of the initial state (set it with :meth:`set_initial_conc`):
        the ``values`` of the condition are not applied to the initial state.

        Raises
        ------
        ValueError
            If the condition is neither a :class:`ConstantConcentration` nor a
            :class:`ZeroConcGradient`.
        """
        if not isinstance(condition, ConstantConcentration) and not isinstance(
            condition, ZeroConcGradient
        ):
            raise ValueError(
                f"{condition} is not a valid boundary condition for the "
                "transport model !"
            )
        self.boundary_conditions.append(condition)

    @property
    def _grid_shape(self) -> Tuple[int, int, int]:
        """Shape (nx, ny, nz) of the grid."""
        return self.lmob[0].shape[1:]  # type: ignore

    def set_constant_conc_indices(self) -> None:
        """Set the node numbers of the grid cells with a constant concentration."""
        nx, ny, nz = self._grid_shape
        node_numbers = np.array([], dtype=np.int32)
        for condition in self.boundary_conditions:
            if isinstance(condition, ConstantConcentration):
                node_numbers = np.hstack(
                    [
                        node_numbers,
                        span_to_node_numbers_3d(condition.span, nx, ny, nz),
                    ]
                )
        self.cst_conc_nn: NDArrayInt = np.unique(node_numbers.flatten())

    @property
    def cst_conc_indices(self) -> NDArrayInt:
        """Return the indices (array with shape (3, n)) of the constant conc cells."""
        nx, ny, _ = self._grid_shape
        return np.array(rlg_nn_to_idx(self.cst_conc_nn, nx=nx, ny=ny))

    @property
    def free_conc_nn(self) -> NDArrayInt:
        """Return the free conc node numbers."""
        return _get_free_node_numbers(int(np.prod(self._grid_shape)), self.cst_conc_nn)

    @property
    def free_conc_indices(self) -> NDArrayInt:
        """Return the indices (array with shape (3, n)) of the free conc cells."""
        nx, ny, _ = self._grid_shape
        return np.array(rlg_nn_to_idx(self.free_conc_nn, nx=nx, ny=ny))

    def reinit(self) -> None:
        """Reset all the results, but keep the initial conditions (first time)."""
        self.lmob = self.lmob[:1]
        self.limmob = self.limmob[:1]
        self.immob_prev = self.limmob[0]
        # There is no initial condition for the density. It is all computed.
        self.ldensity.clear()
        self.lsources.clear()
        self.set_constant_conc_indices()
        self.l_q_next = []
        self.l_q_prev = []
        self.super_ilu = None
        self.preconditioner = None


class ForwardModel:
    """
    Class representing the reactive transport model.

    It aggregates the grid, the parameters, the flow and transport models (which hold
    the results) and the source terms. It is solved by
    :class:`~pyrtid.forward.ForwardSolver`.

    Attributes
    ----------
    grid: RectilinearGrid
        The grid.
    time_params: TimeParameters
        The time parameters.
    gch_params: GeochemicalParameters
        The geochemical parameters.
    fl_model: FlowModel
        The flow model (:class:`DensityFlowModel` if the gravity is enabled in the
        flow parameters, :class:`SaturatedFlowModel` otherwise).
    tr_model: TransportModel
        The transport model.
    source_terms: Dict[str, SourceTerm]
        The source terms, by name.
    """

    def __init__(
        self,
        grid: RectilinearGrid,
        time_params: TimeParameters,
        fl_params: Optional[FlowParameters] = None,
        tr_params: Optional[TransportParameters] = None,
        gch_params: Optional[GeochemicalParameters] = None,
        source_terms: Optional[Union[SourceTerm, Sequence[SourceTerm]]] = None,
        boundary_conditions: Optional[
            Union[BoundaryCondition, Sequence[BoundaryCondition]]
        ] = None,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        grid : RectilinearGrid
            The grid.
        time_params : TimeParameters
            The time parameters.
        fl_params : Optional[FlowParameters], optional
            The flow parameters. By default None, which means default
            :class:`FlowParameters`.
        tr_params : Optional[TransportParameters], optional
            The transport parameters. By default None, which means default
            :class:`TransportParameters`.
        gch_params : Optional[GeochemicalParameters], optional
            The geochemical parameters. By default None, which means default
            :class:`GeochemicalParameters`.
        source_terms : Optional[Union[SourceTerm, Sequence[SourceTerm]]], optional
            One or several source terms (wells, ...). By default None.
        boundary_conditions : BoundaryCondition or Sequence of, optional
            One or several boundary conditions, for the flow
            (:class:`ConstantHead`) or for the transport
            (:class:`ConstantConcentration`, :class:`ZeroConcGradient`).
            By default None.
        """
        # Parameters instances are created here, rather than in the signature, so
        # that they are not shared between models.
        fl_params = FlowParameters() if fl_params is None else fl_params
        tr_params = TransportParameters() if tr_params is None else tr_params
        gch_params = GeochemicalParameters() if gch_params is None else gch_params

        self.grid: RectilinearGrid = grid
        self.time_params: TimeParameters = time_params
        self.gch_params: GeochemicalParameters = gch_params
        # Two possible flowmodels
        self.fl_model: FlowModel
        if fl_params.is_gravity:
            self.fl_model = DensityFlowModel(grid, time_params, fl_params)
        else:
            self.fl_model = SaturatedFlowModel(grid, time_params, fl_params)

        self.tr_model: TransportModel = TransportModel(
            grid, time_params, tr_params, gch_params
        )
        self.source_terms: Dict[str, SourceTerm] = {}
        if source_terms is not None:
            self.source_terms = {
                v.name: v for v in object_or_object_sequence_to_list(source_terms)
            }
        if boundary_conditions is not None:
            for condition in object_or_object_sequence_to_list(boundary_conditions):
                self.add_boundary_conditions(condition)
            self.fl_model.set_constant_head_indices()

    def get_sources(
        self, time: float, grid: RectilinearGrid
    ) -> Tuple[NDArrayFloat, NDArrayFloat]:
        """
        Get the flow sources and sink terms at a given time.

        Parameters
        ----------
        time : float
            Time in seconds.
        grid : RectilinearGrid
            The grid.

        Returns
        -------
        Tuple[NDArrayFloat, NDArrayFloat]
            The flow sources (1/s, with shape (nx, ny, nz)) and the concentration
            sources (mol/l/s, with shape (n_sp, nx, ny, nz)). Positive flow values
            are injections, negative ones are pumpings. The concentration sources
            only account for the injections. They are null in the constant head
            (flow) and constant concentration (concentration) grid cells.
        """

        _unitflw_src = np.zeros(grid.shape)
        _conc_src = np.zeros((self.tr_model.n_sp, grid.nx, grid.ny, grid.nz))

        # iterate the source terms
        for source in self.source_terms.values():
            # identify the source term applying
            _flw, _conc = source.get_values(time)
            nids = source.get_node_indices(grid)

            # Add the flowrates contribution
            _unitflw_src[nids[0], nids[1], nids[2]] += _flw / source.n_nodes

            # Keep only non negative flowrates (remove sink terms)
            if _flw > 0:
                # A source term may define the concentration of the first species only
                for sp in range(min(self.tr_model.n_sp, np.size(_conc))):
                    _conc_src[sp, nids[0], nids[1], nids[2]] += (
                        _flw * _conc[sp] / source.n_nodes
                    )
        for condition in self.fl_model.boundary_conditions:
            if isinstance(condition, ConstantHead):
                # Set zero where there constant head
                _unitflw_src[condition.span] = 0.0

        for condition in self.tr_model.boundary_conditions:
            if isinstance(condition, ConstantConcentration):
                # Set zero where there constant concentration
                for sp in range(self.tr_model.n_sp):
                    _conc_src[sp][condition.span] = 0.0

        return (
            _unitflw_src / self.grid.grid_cell_volume_m3,  # /s
            _conc_src / self.grid.grid_cell_volume_m3,  # mol/L
        )

    def add_src_term(self, source_term: SourceTerm) -> None:
        """
        Add a source term.

        A warning is raised and the existing source term is overwritten if one with
        the same name exists.
        """
        if self.source_terms.get(source_term.name) is not None:
            warnings.warn(
                f"{source_term.name} is already among the source terms"
                " and has been overwritten!"
            )
        self.source_terms[source_term.name] = source_term

    def add_boundary_conditions(self, condition: BoundaryCondition) -> None:
        """
        Add a boundary condition to the flow or the transport model.

        Raises
        ------
        ValueError
            If the type of the condition is not supported.
        """
        # TODO: add a check to see if the given condition is on a border of the grid or
        # not.
        if isinstance(condition, ConstantHead):
            self.fl_model.add_boundary_conditions(condition)
            return
        if isinstance(condition, ConstantConcentration) or isinstance(
            condition, ZeroConcGradient
        ):
            self.tr_model.add_boundary_conditions(condition)
            return
        raise ValueError(f"{condition} is not a valid boundary condition !")

    def reinit(self) -> None:
        """Reset all the results, but keep the initial conditions (first time)."""
        self.fl_model.reinit()
        self.tr_model.reinit()
        self.time_params.reset_to_init()

    def get_ij_over_u(self, time_index: int) -> NDArrayFloat:
        """
        Get the ij/Unorm for the CFL condition.

        Parameters
        ----------
        time_index : int
            Index of the time at which the velocity field is evaluated.

        Returns
        -------
        NDArrayFloat
            The smallest grid cell size divided by the norm of the velocity at the
            grid cell centers, with shape (nx, ny, nz).
        """
        num = 1e300
        if self.grid.nx > 1:
            num = min(self.grid.dx, num)
        if self.grid.ny > 1:
            num = min(self.grid.dy, num)
        if self.grid.nz > 1:
            num = min(self.grid.dz, num)
        den = self.fl_model.get_u_darcy_norm_sample(time_index)
        # VERY_SMALL_NUMBER to avoid division by zero.
        den = np.where(den < VERY_SMALL_NUMBER, VERY_SMALL_NUMBER, den)
        return num / den

    def __deepcopy__(self, memo):
        """
        Deep copy the model.

        The SuperLU factorizations and the preconditioners are not copied
        (they can't be pickled): the copy shares them with the original.
        """
        deepcopy_method = self.__deepcopy__
        self.__deepcopy__ = None

        # Handle non pickebeable objects
        tmp_fl_spilu = self.fl_model.super_ilu
        tmp_fl_pcd = self.fl_model.preconditioner
        tmp_tr_spilu = self.tr_model.super_ilu
        tmp_tr_pcd = self.tr_model.preconditioner
        self.fl_model.super_ilu = None
        self.fl_model.preconditioner = None
        self.tr_model.super_ilu = None
        self.tr_model.preconditioner = None

        try:
            cp = copy.deepcopy(self, memo)
        finally:
            # always restore the original object, even if the copy failed
            self.__deepcopy__ = deepcopy_method
            self.fl_model.super_ilu = tmp_fl_spilu
            self.fl_model.preconditioner = tmp_fl_pcd
            self.tr_model.super_ilu = tmp_tr_spilu
            self.tr_model.preconditioner = tmp_tr_pcd

        # Bind to cp by types.MethodType
        cp.__deepcopy__ = types.MethodType(deepcopy_method.__func__, cp)

        # restore the attributes in the copy
        cp.fl_model.super_ilu = tmp_fl_spilu
        cp.fl_model.preconditioner = tmp_fl_pcd
        cp.tr_model.super_ilu = tmp_tr_spilu
        cp.tr_model.preconditioner = tmp_tr_pcd

        return cp


class SparseMatrixBuilder:
    """
    Accumulate the entries of a sparse matrix as ``(row, col, value)`` triplets.

    Assembling a matrix by incrementing the entries of a ``lil_array`` is slow
    (python loops), especially for the diagonal. This builder only stores the
    triplets, and the duplicated entries are summed when converting to a sparse
    format, which is fast and vectorized.

    Use :func:`add_entries` and :func:`add_to_diagonal` to fill either a builder or a
    ``lil_array`` with the same code.
    """

    __slots__ = ["shape", "_rows", "_cols", "_values"]

    def __init__(self, shape: Tuple[int, int]) -> None:
        """Initialize an empty matrix with the given shape."""
        self.shape = shape
        self._rows: List[NDArrayInt] = []
        self._cols: List[NDArrayInt] = []
        self._values: List[NDArrayFloat] = []

    def add(self, rows: NDArrayInt, cols: NDArrayInt, values: NDArrayFloat) -> None:
        """
        Add ``values`` to the entries ``(rows[i], cols[i])``.

        ``values`` can be a scalar. The same entry can be added several times.
        """
        rows = np.asarray(rows).ravel()
        self._rows.append(rows)
        self._cols.append(np.asarray(cols).ravel())
        self._values.append(np.broadcast_to(values, rows.shape).astype(np.float64))

    def tocsc(self) -> sparse.csc_array:
        """Return the matrix in csc format (duplicated entries are summed)."""
        if len(self._rows) == 0:
            return sparse.csc_array(self.shape, dtype=np.float64)
        return sparse.coo_array(
            (
                np.concatenate(self._values),
                (np.concatenate(self._rows), np.concatenate(self._cols)),
            ),
            shape=self.shape,
        ).tocsc()

    def tolil(self) -> lil_array:
        """Return the matrix in lil format (duplicated entries are summed)."""
        return self.tocsc().tolil()


def add_entries(
    matrix: Union[lil_array, SparseMatrixBuilder],
    rows: NDArrayInt,
    cols: NDArrayInt,
    values: Union[float, NDArrayFloat],
) -> None:
    """
    Add ``values`` to the entries ``(rows[i], cols[i])`` of a matrix, in place.

    The pairs ``(rows[i], cols[i])`` must be unique.
    """
    if isinstance(matrix, SparseMatrixBuilder):
        matrix.add(rows, cols, values)
    else:
        matrix[rows, cols] += values  # type: ignore


def add_to_diagonal(
    matrix: Union[lil_array, SparseMatrixBuilder], values: Union[float, NDArrayFloat]
) -> None:
    """Add ``values`` to the diagonal of a square matrix, in place."""
    if isinstance(matrix, SparseMatrixBuilder):
        idx = np.arange(matrix.shape[0])
        matrix.add(idx, idx, values)
    else:
        matrix.setdiag(matrix.diagonal() + values)


def remove_cst_bound_indices(
    indices_owner: NDArrayInt, indices_neigh: NDArrayInt, indices_to_remove: NDArrayInt
) -> Tuple[NDArrayInt, NDArrayInt]:
    """
    Remove the owner/neighbor pairs whose owner is a boundary condition node.

    Parameters
    ----------
    indices_owner : NDArrayInt
        Indices of owner grid cells.
    indices_neigh : NDArrayInt
        Indices of neighbor grid cells.
    indices_to_remove : NDArrayInt
        Owner indices to remove.

    Returns
    -------
    Tuple[NDArrayInt, NDArrayInt]
        The remaining owner and neighbor indices.
    """
    is_kept = ~np.isin(indices_owner, indices_to_remove)
    return indices_owner[is_kept], indices_neigh[is_kept]


def keep_a_b_if_c_in_a(
    a: NDArrayInt, b: NDArrayInt, c: NDArrayInt
) -> Tuple[NDArrayInt, NDArrayInt]:
    """Keep the pairs ``(a[i], b[i])`` for which ``a[i]`` is in ``c``."""
    is_kept = np.isin(a, c)
    return a[is_kept], b[is_kept]


# TODO cache
def get_owner_neigh_indices(
    grid: RectilinearGrid,
    span_owner: Tuple[slice, slice, slice],
    span_neigh: Tuple[slice, slice, slice],
    owner_indices_to_keep: Optional[NDArrayInt] = None,
    neigh_indices_to_keep: Optional[NDArrayInt] = None,
) -> Tuple[NDArrayInt, NDArrayInt]:
    """
    Return the node numbers of the pairs of neighbor grid cells.

    The two spans must select the same number of grid cells: the i-th owner is paired
    with the i-th neighbor.

    Parameters
    ----------
    grid : RectilinearGrid
        The grid.
    span_owner : Tuple[slice, slice, slice]
        Span of the owner grid cells.
    span_neigh : Tuple[slice, slice, slice]
        Span of the neighbor grid cells.
    owner_indices_to_keep : Optional[NDArrayInt], optional
        If given, only the pairs whose owner node number is in this array are kept.
        By default None.
    neigh_indices_to_keep : Optional[NDArrayInt], optional
        If given, only the pairs whose neighbor node number is in this array are
        kept. By default None.

    Returns
    -------
    Tuple[NDArrayInt, NDArrayInt]
        The node numbers of the owners and of the neighbors.
    """
    # Get indices
    indices_owner: NDArrayInt = span_to_node_numbers_3d(
        span_owner, nx=grid.nx, ny=grid.ny, nz=grid.nz
    )
    indices_neigh: NDArrayInt = span_to_node_numbers_3d(
        span_neigh, nx=grid.nx, ny=grid.ny, nz=grid.nz
    )

    if owner_indices_to_keep is not None:
        indices_owner, indices_neigh = keep_a_b_if_c_in_a(
            indices_owner, indices_neigh, owner_indices_to_keep
        )
    if neigh_indices_to_keep is not None:
        indices_neigh, indices_owner = keep_a_b_if_c_in_a(
            indices_neigh, indices_owner, neigh_indices_to_keep
        )
    return indices_owner, indices_neigh
