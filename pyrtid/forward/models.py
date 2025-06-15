"""Provide a model representing the reactive transport system.

Note: this part is handle with numba so it can be used in functions later on.

Note :
- The timestep is fixed.
- The grid is composed of regular grid cells
"""

from __future__ import annotations

import copy
import types
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from scipy.sparse import lil_array
from scipy.sparse.linalg import LinearOperator, SuperLU

from pyrtid.utils import (
    NDArrayBool,
    NDArrayFloat,
    NDArrayInt,
    RectilinearGrid,
    StrEnum,
    get_a_not_in_b_1d,
    node_number_to_indices,
    object_or_object_sequence_to_list,
    span_to_node_numbers_3d,
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
        """Initialize the instance."""
        self.duration = duration

        # First pass on the min/max
        if dt_min is not None:
            _dt_min: float = dt_min
        else:
            _dt_min = dt_init
        if dt_max is not None:
            _dt_max: float = dt_max
        else:
            _dt_max = dt_init

        _dt_init = max(min(_dt_max, dt_init), _dt_min, dt_init)

        # Second pass on the min/max
        if dt_min is not None:
            self.dt_min: float = dt_min
        else:
            self.dt_min = _dt_init
        if dt_max is not None:
            self.dt_max: float = dt_max
        else:
            self.dt_max = _dt_init

        # Check dt_min and dt_max consistency
        if self.dt_min > self.dt_max:
            raise ValueError(f"dt_min ({self.dt_min}) is above dt_max ({self.dt_max})!")

        self.courant_factor: float = courant_factor
        # Apply bounds
        self.dt_init: float = _dt_init
        self.dt = _dt_init
        self.nfpi: int = 0
        self.ldt: List[float] = []
        self.lnfpi: List[int] = []

    @property
    def time_elapsed(self) -> float:
        """Time elapsed in the simulation."""
        return np.sum(self.ldt)

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

        Parameters
        ----------
        n_iter: int
            Number of iterations required to solve the last timestep.
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
        """Get the maximum timestep to respect the CFL condition."""
        dt_cfl = np.min(
            self.courant_factor
            * model.tr_model.porosity
            * model.get_ij_over_u(time_index)
        )
        return float(np.min(dt_cfl))


class FlowRegime(StrEnum):
    STATIONARY = "stationary"
    TRANSIENT = "transient"


class VerticalAxis(StrEnum):
    X = "x"
    Y = "y"
    Z = "z"


class FlowParameters:
    """
    Class defining the time parameters used in the simulation.

    Attributes
    ----------
    k0: float, optional
        Default permeability in the grid (m/s). The default is 1.e-4 m/s.
    storage_coefficient: float, optional
        The default storage coefficient in the grid ($m^{-1}$).
        The default is 1e-3 $m^{-1}$.
    crank_nicolson: float
        The Crank-Nicolson parameter allows to set the temporal resolution
        scheme to explicit, fully implicit or somewhere in between these two
        extremes. The value must be comprised between 0.0 and 1.0, 0.0 being a
        full explicit scheme and 1.0 fully implicit. The default is 1.0.
    is_gravity: bool, optional
        Whether the gravity is taken into account, i.e. density driven flow.
    vertical_axis: VerticalAxis
        Define which axis is the vertical one. It only affects if the gravity is
        enabled. The default is the z axis.
    tolerance: float, optional
        The tolerance on the flow. The default is 1e-8.
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
        """Initialize the instance."""
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
    dispercivity: float, optional
        The dispersivity (kinematic and numeric) in meters. The default is 0.1 m.
    porosity: float, optional
        Default porosity in the grid Should be a number between 0 and 1.
        The default is 1.0.
    crank_nicolson: float
        The Crank-Nicholson parameter allows to set the temporal resolution
        scheme to explicit, fully implicit or somewhere in between these two
        extremes. The value must be comprised between 0.0 and 1.0, 0.0 being a
        full explicit scheme and 1.0 fully implicit. The default is 0.5.
    tolerance: float, optional
        The tolerance on the transport. The default is 1e-8.
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
        Maximum number of fixed point iterations per timestep. If this number is reached
        then the numerical acceleration is temporarily disabled, otherwise, the
        timestep is reduced.
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
        """Initialize the instance."""
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
    Class defining the geocgemical parameters used in the simulation.

    Attributes
    ----------
    conc: float, optional
        Initial tracer concentration in the grid in molal. The default is 0.0.
    conc2: float, optional
        Initial reagent concentration in the grid in molal. The default is 0.0.
    grade: float, optional
        Default mineral grade in the grid in mol/kg (kg of water).
        The default is 0.0.
    kv: float, optional
        The kinetic rate of the mineral in [mol/m2/s]. The default is -6.9e-9.
    As: float, optional
        Specific area in [m2/mol]. The default is 13.5.
    Ks: float, optional
        Solubility constant (no unit). The default is 6.3e-4.
    Ms: float, optional
        Molar mass in g/mol.
    stocoef: float
        Number of mole of species 2 consumed when dissolving the mineral.
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
    ) -> None:
        """Initialize the instance."""
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


class SourceTerm:
    """
    Define a source term object.

    A well object can pump or inject.

    Attributes
    ----------
    name: str
        Name of the instance.
    x_coord: float
        x coordinate of the well.
    y_coord: float
        y_coordinate of the well.
    flowrates: NDArrayFloat
        Sequence of flowrates of the well. Positive = injection, negative = pumping.
    concentrations: NDArrayFloat
        Concentrations of the first species, used only if flowrates is positive.
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
        x_coord: float
            x coordinate of the well.
        y_coord: float
            y_coordinate of the well.
        flowrates: NDArrayFloat
            Sequence of flowrates of the well (m3/s).
            Positive = injection, negative = pumping.
        concentrations: NDArrayFloat
            Concentration, used only if flowrates is positive (mol/l).
            With dimension (nt, n_sp).

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
        return np.array(
            node_number_to_indices(self.node_ids, nx=grid.nx, ny=grid.ny)
        ).reshape(3, -1)

    @property
    def n_nodes(self) -> int:
        """Return the number of nodes."""
        return np.size(self.node_ids)

    def get_values(self, time: float) -> Tuple[float, float]:
        """Return the concentrations and the flowrates for a given time."""
        if time < self.times[0]:
            return 0.0, 0.0
        time_index = 0  # index in times

        # This is matching the "modify" process behavior of HYTEC
        for time_index, _time in enumerate(self.times):
            if _time > time:
                if time_index > 0:
                    time_index -= 1
                break
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
    value: Union[float, NDArrayFloat]


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


class FlowModel(ABC):
    """Represent a flow model."""

    __slots__ = [
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

        # Empty arrays
        self.west_boundary_idx: NDArrayInt = np.zeros([], dtype=np.int64)
        self.east_boundary_idx: NDArrayInt = np.zeros([], dtype=np.int64)
        self.south_boundary_idx: NDArrayInt = np.zeros([], dtype=np.int64)
        self.north_boundary_idx: NDArrayInt = np.zeros([], dtype=np.int64)
        self.bottom_boundary_idx: NDArrayInt = np.zeros([], dtype=np.int64)
        self.top_boundary_idx: NDArrayInt = np.zeros([], dtype=np.int64)

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
        """Add a boundary condition to the flow model."""
        if not isinstance(condition, ConstantHead):
            raise ValueError(
                f"{condition} is not a valid boundary condition for the flow model !"
            )
        self.boundary_conditions.append(condition)

        if isinstance(condition, ConstantHead):
            # 0) Set the values
            self.lhead[0][condition.span] = condition.values
            # 1) Get the new constant head node numbers

    def set_constant_head_indices(self) -> None:
        """Set the indices of nodes with constant head."""
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
                _ix, _iy, _iz = node_number_to_indices(new_nn, nx, ny)
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
        """Return the indices (array) of the constant head grid cells."""
        # [:2] to ignore the z axis
        return np.array(
            node_number_to_indices(
                self.cst_head_nn, nx=self.head.shape[0], ny=self.head.shape[1]
            )
        )

    @property
    def free_head_nn(self) -> NDArrayInt:
        """Return the free head node numbers."""
        return get_a_not_in_b_1d(
            np.arange(np.prod(self.lhead[0].shape), dtype=np.int32),  # type: ignore
            self.cst_head_nn,
        )

    @property
    def free_head_indices(self) -> NDArrayInt:
        """Return the indices (array) of the free head grid cells."""
        # [:2] to ignore the z axis
        return np.array(
            node_number_to_indices(
                self.free_head_nn, nx=self.head.shape[0], ny=self.head.shape[1]
            )
        )

    def reinit(self) -> None:
        """Set all arrays to zero except for the initial conditions(first time)."""
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

    @property
    def u_darcy_x_center(self) -> NDArrayFloat:
        """The darcy x-velocities estimated at the mesh centers."""
        # Compute the average velocity
        tmp = np.zeros((self.head.shape))
        tmp += self.u_darcy_x[:-1, :, :, :]
        tmp += self.u_darcy_x[1:, :, :, :]
        # All nodes have 2 boundaries along the y axis, except for the
        # borders grid cells
        tmp[1:-1, :, :, :] /= 2
        # for the borders we need to check if a boundary (flow) exist or not
        # this is a consequence of constant head and imposed flux
        if self.west_boundary_idx.size != 0:
            tmp[0, self.west_boundary_idx[0], self.west_boundary_idx[1], :] /= 2
        if self.east_boundary_idx.size != 0:
            tmp[-1, self.east_boundary_idx[0], self.east_boundary_idx[1], :] /= 2
        return tmp

    @property
    def u_darcy_y_center(self) -> NDArrayFloat:
        """The darcy y-velocities estimated at the mesh centers."""
        # Compute the average velocity
        tmp = np.zeros((self.head.shape))
        tmp += self.u_darcy_y[:, :-1, :, :]
        tmp += self.u_darcy_y[:, 1:, :, :]
        tmp[:, 1:-1, :, :] /= 2
        # All nodes have 2 boundaries along the x axis, except for the
        # borders grid cells
        # for the borders we need to check if a boundary (flow) exist or not
        # this is a consequence of constant head and imposed flux
        if self.south_boundary_idx.size != 0:
            tmp[self.south_boundary_idx[0], 0, self.south_boundary_idx[1], :] /= 2
        if self.north_boundary_idx.size != 0:
            tmp[self.north_boundary_idx[0], -1, self.north_boundary_idx[1], :] /= 2

        return tmp

    @property
    def u_darcy_z_center(self) -> NDArrayFloat:
        """The darcy Z-velocities estimated at the mesh centers."""
        # Compute the average velocity
        tmp = np.zeros((self.head.shape))
        tmp += self.u_darcy_z[:, :, :-1, :]
        tmp += self.u_darcy_z[:, :, 1:, :]
        tmp[:, :, 1:-1, :] /= 2
        # All nodes have 2 boundaries along the z axis, except for the
        # borders grid cells
        # for the borders we need to check if a boundary (flow) exist or not
        # this is a consequence of constant head and imposed flux
        if self.bottom_boundary_idx.size != 0:
            tmp[self.bottom_boundary_idx[0], self.bottom_boundary_idx[1], 0, :] /= 2
        if self.top_boundary_idx.size != 0:
            tmp[self.top_boundary_idx[0], self.top_boundary_idx[1], -1, :] /= 2

        return tmp

    @property
    def u_darcy_norm(self) -> NDArrayFloat:
        """The norm of the darcy velocity estimated at the center of the mesh."""
        return np.sqrt(self.u_darcy_x_center**2 + self.u_darcy_y_center**2)

    def get_u_darcy_norm_sample(self, time_index: int) -> NDArrayFloat:
        """The norm of the darcy velocity estimated at the center of the grid cell."""
        # for x
        tmp_x = np.zeros_like(self.lhead[time_index])
        tmp_x += (
            self.lu_darcy_x[time_index][:-1, :, :]
            + self.lu_darcy_x[time_index][1:, :, :]
        )
        # All nodes have 2 boundaries along the y axis, except for the
        # borders grid cells
        tmp_x[1:-1, :, :] /= 2
        # for the borders we need to check if a boundary (flow) exist or not
        # this is a consequence of constant head and imposed flux
        if self.west_boundary_idx.size != 0:
            tmp_x[0, self.west_boundary_idx[0], self.west_boundary_idx[1]] /= 2
        if self.east_boundary_idx.size != 0:
            tmp_x[-1, self.east_boundary_idx[0], self.east_boundary_idx[1]] /= 2

        # for y
        tmp_y = np.zeros((self.lhead[time_index].shape))
        tmp_y += (
            self.lu_darcy_y[time_index][:, :-1, :]
            + self.lu_darcy_y[time_index][:, 1:, :]
        )
        # All nodes have 2 boundaries along the y axis, except for the
        # borders grid cells
        tmp_y[:, 1:-1, :] /= 2
        # for the borders we need to check if a boundary (flow) exist or not
        # this is a consequence of constant head and imposed flux
        if self.south_boundary_idx.size != 0:
            tmp_y[self.south_boundary_idx[0], 0, self.south_boundary_idx[1]] /= 2
        if self.north_boundary_idx.size != 0:
            tmp_y[self.north_boundary_idx[0], -1, self.north_boundary_idx[1]] /= 2

        # for z
        tmp_z = np.zeros((self.lhead[time_index].shape))
        tmp_z += (
            self.lu_darcy_z[time_index][:, :, :-1]
            + self.lu_darcy_z[time_index][:, :, 1:]
        )
        # All nodes have 2 boundaries along the y axis, except for the
        # borders grid cells
        tmp_z[:, :, 1:-1] /= 2
        # for the borders we need to check if a boundary (flow) exist or not
        # this is a consequence of constant head and imposed flux
        if self.bottom_boundary_idx.size != 0:
            tmp_z[self.bottom_boundary_idx[0], self.bottom_boundary_idx[1], 0] /= 2
        if self.top_boundary_idx.size != 0:
            tmp_z[self.top_boundary_idx[0], self.top_boundary_idx[1], -1] /= 2

        # norm
        return np.sqrt(tmp_x**2 + tmp_y**2 + tmp_z**2)

    def get_du_darcy_norm_sample(
        self, time_index: int
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """The norm of the darcy velocity estimated at the center of the grid cell."""
        # for x
        tmp_x = np.zeros_like(self.lhead[time_index])
        tmp_x += (
            self.lu_darcy_x[time_index][:-1, :, :]
            + self.lu_darcy_x[time_index][1:, :, :]
        )
        # All nodes have 2 boundaries along the y axis, except for the
        # borders grid cells
        divx = np.ones_like(tmp_x)
        divx[1:-1, :, :] /= 2
        # for the borders we need to check if a boundary (flow) exist or not
        # this is a consequence of constant head and imposed flux
        if self.west_boundary_idx.size != 0:
            divx[0, self.west_boundary_idx[0], self.west_boundary_idx[1]] /= 2
        if self.east_boundary_idx.size != 0:
            divx[-1, self.east_boundary_idx[0], self.east_boundary_idx[1]] /= 2

        tmp_x *= divx

        # for y
        tmp_y = np.zeros((self.lhead[time_index].shape))
        tmp_y += (
            self.lu_darcy_y[time_index][:, :-1, :]
            + self.lu_darcy_y[time_index][:, 1:, :]
        )
        # All nodes have 2 boundaries along the y axis, except for the
        # borders grid cells
        divy = np.ones_like(tmp_y)
        divy[:, 1:-1, :] /= 2

        # for the borders we need to check if a boundary (flow) exist or not
        # this is a consequence of constant head and imposed flux
        if self.south_boundary_idx.size != 0:
            divy[self.south_boundary_idx[0], 0, self.south_boundary_idx[1]] /= 2
        if self.north_boundary_idx.size != 0:
            divy[self.north_boundary_idx[0], -1, self.north_boundary_idx[1]] /= 2

        tmp_y *= divy

        # for z
        tmp_z = np.zeros((self.lhead[time_index].shape))
        tmp_z += (
            self.lu_darcy_z[time_index][:, :, :-1]
            + self.lu_darcy_z[time_index][:, :, 1:]
        )
        # All nodes have 2 boundaries along the y axis, except for the
        # borders grid cells
        divz = np.ones_like(tmp_z)
        divz[:, :, 1:-1] /= 2

        # for the borders we need to check if a boundary (flow) exist or not
        # this is a consequence of constant head and imposed flux
        if self.bottom_boundary_idx.size != 0:
            divz[self.bottom_boundary_idx[0], self.bottom_boundary_idx[1], 0] /= 2
        if self.top_boundary_idx.size != 0:
            divz[self.top_boundary_idx[0], self.top_boundary_idx[1], -1] /= 2

        tmp_z *= divz

        # norm
        norm = np.sqrt(tmp_x**2 + tmp_y**2, tmp_z**2)

        # inverse of the norm -> avoid division by zero
        inv_norm = np.zeros_like(norm)
        mask = norm > 0.0
        inv_norm[mask] = 1.0 / norm[mask]

        # return (d|U|/dUx , d|U|/dUy, d|U|/dUz)
        return inv_norm * tmp_x * divx, inv_norm * tmp_y * divy, inv_norm * tmp_z * divz

    def get_u_darcy_norm(self) -> NDArrayFloat:
        """The norm of the darcy velocity estimated at the center of the grid cell."""
        # for x
        tmp_x = np.zeros_like(self.head)
        tmp_x += self.u_darcy_x[:-1, :, :] + self.u_darcy_x[1:, :, :]
        # All nodes have 2 boundaries along the y axis, except for the
        # borders grid cells
        tmp_x[1:-1, :, :] /= 2
        # for the borders we need to check if a boundary (flow) exist or not
        # this is a consequence of constant head and imposed flux

        if self.west_boundary_idx.size != 0:
            tmp_x[0, self.west_boundary_idx[0], self.west_boundary_idx[1]] /= 2
        if self.east_boundary_idx.size != 0:
            tmp_x[-1, self.east_boundary_idx[0], self.east_boundary_idx[1]] /= 2

        # for y
        tmp_y = np.zeros((self.head.shape))
        tmp_y += self.u_darcy_y[:, :-1, :] + self.u_darcy_y[:, 1:, :]
        # All nodes have 2 boundaries along the y axis, except for the
        # borders grid cells
        tmp_y[:, 1:-1, :] /= 2
        # for the borders we need to check if a boundary (flow) exist or not
        # this is a consequence of constant head and imposed flux
        if self.south_boundary_idx.size != 0:
            tmp_y[self.south_boundary_idx[0], 0, self.south_boundary_idx[1]] /= 2
        if self.north_boundary_idx.size != 0:
            tmp_y[self.north_boundary_idx[0], -1, self.north_boundary_idx[1]] /= 2

        # for z
        tmp_z = np.zeros((self.head.shape))
        tmp_z += self.u_darcy_z[:, :, :-1] + self.u_darcy_z[:, :, 1:]
        # All nodes have 2 boundaries along the y axis, except for the
        # borders grid cells
        tmp_z[:, :, 1:-1] /= 2
        # for the borders we need to check if a boundary (flow) exist or not
        # this is a consequence of constant head and imposed flux
        if self.bottom_boundary_idx.size != 0:
            tmp_z[self.bottom_boundary_idx[0], self.bottom_boundary_idx[1], 0] /= 2
        if self.top_boundary_idx.size != 0:
            tmp_z[self.top_boundary_idx[0], self.top_boundary_idx[1], -1] /= 2

        # norm
        return np.sqrt(tmp_x**2 + tmp_y**2 + tmp_z**2)

    def get_vertical_dim(self) -> int:
        """Return the number of voxel along the vertical_axis axis."""
        if self.vertical_axis == VerticalAxis.X:
            return self.lhead[0].shape[0]
        elif self.vertical_axis == VerticalAxis.Y:
            return self.lhead[0].shape[1]
        else:
            return self.lhead[0].shape[2]

    # TODO: cache
    def _get_mesh_center_vertical_pos(self) -> NDArrayFloat:
        """Return the vertical position of the grid cells centers."""
        xv, yv, zv = np.meshgrid(
            range(self.lhead[0].shape[0]),
            range(self.lhead[0].shape[1]),
            range(self.lhead[0].shape[2]),
            indexing="ij",
        )
        if self.vertical_axis == VerticalAxis.X:
            return (xv + 0.5) * self.vertical_mesh_size
        elif self.vertical_axis == VerticalAxis.Y:
            return (yv + 0.5) * self.vertical_mesh_size
        else:
            return (zv + 0.5) * self.vertical_mesh_size

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
    """Represent a flow model."""

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
        self.q_prev: lil_array = lil_array((grid.n_grid_cells, grid.n_grid_cells))
        self.q_next: lil_array = lil_array((grid.n_grid_cells, grid.n_grid_cells))
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
        Return mobile concentrations as array with dimension (2, nx, ny, nz, nt + 1).

        This is read-only. Alias for mob.
        """
        return self.mob[0]

    @property
    def conc2(self) -> NDArrayFloat:
        """
        Return mobile concentrations as array with dimension (2, nx, ny, nz, nt + 1).

        This is read-only. Alias for mob.
        """
        return self.mob[1]

    @property
    def grade(self) -> NDArrayFloat:
        """
        Return immobile concentrations as array with dimension (nx, ny, nz, nt + 1).

        This is read-only. Alias for immob[0].
        """
        return self.immob[0]

    @property
    def grade2(self) -> NDArrayFloat:
        """
        Return immobile concentrations as array with dimension (nx, ny, nz, nt + 1).

        This is read-only. Alias for immob[1].
        """
        return self.immob[1]

    @property
    def density(self) -> NDArrayFloat:
        """
        Return densities in g/l as array with dimension (nx, ny, nz, nt + 1).

        This is read-only.
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
        """Add a boundary condition to the transport model."""
        if not isinstance(condition, ConstantConcentration) and not isinstance(
            condition, ZeroConcGradient
        ):
            raise ValueError(
                f"{condition} is not a valid boundary condition for the "
                "transport model !"
            )
        self.boundary_conditions.append(condition)

        if isinstance(condition, ConstantConcentration):
            # 0) Set the values
            # TODO
            pass

    def set_constant_conc_indices(self) -> None:
        """Set the indices of nodes with constant head."""
        node_numbers = np.array([], dtype=np.int32)
        for condition in self.boundary_conditions:
            if isinstance(condition, ConstantConcentration):
                node_numbers = np.hstack(
                    [
                        node_numbers,
                        span_to_node_numbers_3d(
                            condition.span,
                            self.mob.shape[0],
                            self.mob.shape[1],
                            self.mob.shape[2],
                        ),
                    ]
                )
        self.cst_conc_nn: NDArrayInt = np.unique(node_numbers.flatten())

    @property
    def cst_conc_indices(self) -> NDArrayInt:
        """Return the indices (array) of the constant conc grid cells."""
        # [:2] to ignore the z axis
        return np.array(
            node_number_to_indices(
                self.cst_conc_nn, nx=self.mob.shape[0], ny=self.mob.shape[1]
            )
        )

    @property
    def free_conc_nn(self) -> NDArrayInt:
        """Return the free conc node numbers."""
        return get_a_not_in_b_1d(
            np.arange(np.prod(self.lmob[0].shape), dtype=np.int32),  # type: ignore
            self.cst_conc_nn,
        )

    @property
    def free_conc_indices(self) -> NDArrayInt:
        """Return the indices (array) of the free conc grid cells."""
        # [:2] to ignore the z axis
        return np.array(
            node_number_to_indices(
                self.free_conc_nn, nx=self.mob.shape[0], ny=self.mob.shape[1]
            )
        )

    def reinit(self) -> None:
        """Set all arrays to zero except for the initial conditions(first time)."""
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

    wadv: float
        Advection weight (for testing between 0.0 and 1.0). The default is 1.0.

    """

    slots = [
        "grid",
        "time_params",
        "fl_params",
        "tr_params",
        "gch_params",
        "source_terms",
        "boundary_conditions",
    ]

    def __init__(
        self,
        grid: RectilinearGrid,
        time_params: TimeParameters,
        fl_params: FlowParameters = FlowParameters(),
        tr_params: TransportParameters = TransportParameters(),
        gch_params: GeochemicalParameters = GeochemicalParameters(),
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
            _description_
        time_params : TimeParameters
            _description_
        fl_params : FlowParameters
            _description_
        tr_params : TransportParameters
            _description_
        gch_params : GeochemicalParameters
            _description_
        wells : Sequence[Well], optional
            _description_, by default default_field([])
        """
        self.grid: RectilinearGrid = grid
        self.time_params: TimeParameters = time_params
        self.gch_params: GeochemicalParameters = gch_params
        # Two possible flowmodels
        if fl_params.is_gravity:
            self.fl_model: FlowModel = DensityFlowModel(grid, time_params, fl_params)
        else:
            self.fl_model: FlowModel = SaturatedFlowModel(grid, time_params, fl_params)

        self.tr_model: TransportModel = TransportModel(
            grid, time_params, tr_params, gch_params
        )
        if source_terms is not None:
            self.source_terms: Dict[str, SourceTerm] = {
                v.name: v for v in object_or_object_sequence_to_list(source_terms)
            }
        else:
            self.source_terms: Dict[str, SourceTerm] = {}
        if boundary_conditions is None:
            return
        for condition in object_or_object_sequence_to_list(boundary_conditions):
            self.add_boundary_conditions(condition)
        self.fl_model.set_constant_head_indices()

    def get_sources(
        self, time: float, grid: RectilinearGrid
    ) -> Tuple[NDArrayFloat, NDArrayFloat]:
        """Get the flow sources and sink terms."""

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
                for sp in range(self.tr_model.n_sp):
                    try:
                        _conc_src[sp, nids[0], nids[1], nids[2]] += (
                            _flw * _conc[sp] / source.n_nodes
                        )
                    except IndexError:
                        pass
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
            _unitflw_src / self.grid.grid_cell_volume,  # /s
            _conc_src / self.grid.grid_cell_volume,  # mol/L
        )

    def add_src_term(self, source_term: SourceTerm) -> None:
        """Add a source term."""
        if self.source_terms.get(source_term.name) is not None:
            warnings.warn(
                f"{source_term.name} is already among the source terms"
                " and has been overwritten!"
            )
        self.source_terms[source_term.name] = source_term

    def add_boundary_conditions(self, condition: BoundaryCondition) -> None:
        """Add a boundary condition to the flow or the transport model."""
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
        """Set all arrays to zero except for the initial conditions(first time)."""
        self.fl_model.reinit()
        self.tr_model.reinit()
        self.time_params.reset_to_init()

    def get_ij_over_u(self, time_index: int) -> NDArrayFloat:
        """Get the ij/Unorm for the CFL condition."""
        num = 1e300
        if self.grid.nx > 1:
            num = min(self.grid.dx, num)
        if self.grid.ny > 1:
            num = min(self.grid.dy, num)
        if self.grid.nz > 1:
            num = min(self.grid.dz, num)
        den = np.sqrt(
            self.fl_model.u_darcy_x_center[:, :, :, time_index] ** 2
            + self.fl_model.u_darcy_y_center[:, :, :, time_index] ** 2
            + self.fl_model.u_darcy_z_center[:, :, :, time_index] ** 2
        )
        # VERY_SMALL_NUMBER to avoid division by zero.
        den = np.where(den < VERY_SMALL_NUMBER, VERY_SMALL_NUMBER, den)
        return num / den

    def __deepcopy__(self, memo):
        deepcopy_method = self.__deepcopy__
        self.__deepcopy__ = None

        # Handle non pickebeable objects
        tmp_fl_spilu = self.fl_model.super_ilu
        self.fl_model.super_ilu = None
        tmp_fl_pcd = self.fl_model.preconditioner
        self.fl_model.preconditioner = None

        tmp_tr_spilu = self.tr_model.super_ilu
        self.tr_model.super_ilu = None
        tmp_tr_pcd = self.tr_model.preconditioner
        self.tr_model.preconditioner = None

        cp = copy.deepcopy(self, memo)
        self.__deepcopy__ = deepcopy_method

        # Bind to cp by types.MethodType
        cp.__deepcopy__ = types.MethodType(deepcopy_method.__func__, cp)

        # custom treatments
        # restore the attributes
        cp.fl_model.super_ilu = tmp_fl_spilu
        cp.fl_model.preconditioner = tmp_fl_pcd
        cp.tr_model.super_ilu = tmp_tr_spilu
        cp.tr_model.preconditioner = tmp_tr_pcd

        return cp


def remove_cst_bound_indices(
    indices_owner: NDArrayInt, indices_neigh: NDArrayInt, indices_to_remove: NDArrayInt
) -> Tuple[NDArrayInt, NDArrayInt]:
    """
    Remove the indices because of the boundary condition.

    Parameters
    ----------
    indices_owner : _type_
        Indices of owner grid cells.
    indices_neigh : _type_
        Indices of neighbor grid cells.
    indices_to_remove : DArrayInt
        Indices to remove.

    Returns
    -------
    Tuple[NDArrayInt, NDArrayInt]
        _description_
    """
    is_kept = ~np.isin(indices_owner, indices_to_remove)
    return indices_owner[is_kept], indices_neigh[is_kept]


def keep_a_b_if_c_in_a(
    a: NDArrayInt, b: NDArrayInt, c: NDArrayInt
) -> Tuple[NDArrayInt, NDArrayInt]:
    """Keep values in a and b if c in a."""
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
    """_summary_

    Parameters
    ----------
    grid : RectilinearGrid
        _description_
    span_owner : Tuple[slice, slice, slice]
        _description_
    span_neigh : Tuple[slice, slice, slice]
        _description_
    indices_to_remove : NDArrayInt
        _description_

    Returns
    -------
    Tuple[NDArrayInt, NDArrayInt]
        _description_
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
