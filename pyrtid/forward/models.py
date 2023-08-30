"""Provide a model representing the reactive transport system.

Note: this part is handle with numba so it can be used in functions later on.

Note :
- The timestep is fixed.
- The grid is composed of regular meshes
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.sparse import lil_array

from pyrtid.utils import StrEnum, node_number_to_indices, span_to_node_numbers_2d
from pyrtid.utils.types import (
    NDArrayBool,
    NDArrayFloat,
    NDArrayInt,
    object_or_object_sequence_to_list,
)

GRAVITY = 9.81
WATER_DENSITY = 997


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
    ldt: List[float]
        List of successive timesteps (in seconds) used in the forward modelling.
    nts: int
        Number of timesteps in the simulation.
    nt: int
        Number of times in the simulation (nt + 1).
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
    ]

    def __init__(
        self,
        duration: float,
        dt_init: float,
        dt_min: Optional[float] = None,
        dt_max: Optional[float] = None,
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

    def update_dt(self, n_iter: int) -> None:
        """
        Update the timestep.

        Parameters
        ----------
        n_iter: int
            Number of iterations required to solve the last timestep.
        """
        if n_iter < 20:
            # increase dt by 5%
            self.dt *= 1.05
        else:
            # decrease dt by 30%
            self.dt *= 0.7
        # Ensure timebounds
        if self.dt < self.dt_min:
            self.dt = self.dt_min
        if self.dt > self.dt_max:
            self.dt = self.dt_max


class FlowRegime(StrEnum):
    STATIONARY = "stationary"
    TRANSIENT = "transient"


class VerticalAxis(StrEnum):
    DX = "dx"
    DY = "dy"
    DZ = "dz"


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
        vertical_axis: VerticalAxis = VerticalAxis.DZ,
        tolerance: float = 1e-8,
    ) -> None:
        """Initialize the instance."""
        self.permeability: float = permeability
        self.storage_coefficient: float = storage_coefficient
        self.crank_nicolson: float = crank_nicolson
        self.regime: FlowRegime = regime
        self.is_gravity: bool = is_gravity
        self.vertical_axis: VerticalAxis = vertical_axis
        self.tolerance: float = tolerance


class TransportParameters:
    """
    Class defining the transport parameters used in the simulation.

    Attributes
    ----------
    diffusion: float, optional
        Default diffusion coefficient in the grid in [m2/s]. The default is 1e-4 m2/s.
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
        :cite:`lagneauOperatorsplittingbasedReactiveTransport2010a`.. .
        The default is False.
    fpi_eps: float
       Tolerance on the transport-chemistry coupling error. The default value is 1e-5.
    """

    def __init__(
        self,
        diffusion: float = 1e-4,
        porosity: float = 1.0,
        crank_nicolson_advection: float = 0.5,
        crank_nicolson_diffusion: float = 1.0,
        tolerance: float = 1e-8,
        is_numerical_acceleration: bool = False,
        fpi_eps: float = 1e-5,
    ) -> None:
        """Initialize the instance."""
        self.diffusion: float = diffusion
        self.porosity: float = porosity
        self.crank_nicolson_advection: float = crank_nicolson_advection
        self.crank_nicolson_diffusion: float = crank_nicolson_diffusion
        self.tolerance: float = tolerance
        self.is_numerical_acceleration: bool = is_numerical_acceleration
        self.fpi_eps = fpi_eps


class GeochemicalParameters:
    """
    Class defining the geocgemical parameters used in the simulation.

    Attributes
    ----------
    conc: float, optional
        Initial tracer concentration in the grid in molal. The default is 0.0.
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
    """

    def __init__(
        self,
        conc: float = 0.0,
        grade: float = 0.0,
        kv: float = -6.9e-9,
        As: float = 13.5,
        Ks: float = 6.3e-4,
        Ms: float = 270,
    ) -> None:
        """Initialize the instance."""
        self.conc: float = conc
        self.grade: float = grade
        self.kv: float = kv
        self.As: float = As
        self.Ks: float = Ks
        self.Ms: float = Ms


class Geometry:
    """
    Class defining the grid geometry used in the simulation.

    Attributes
    ----------
    nx : int
        Number of voxels along the x axis.
    ny : int
        Number of voxels along the y axis.
    dx : float
        Voxel dimension along the x axis.
    dy : float
        Voxel dimension along the y axis.
    dz : float
        Voxel dimension along the z axis. This is only taken into account for
        the voxel volume computation.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        dz: float = 1.0,
    ) -> None:
        """Initialize the class instance."""
        self._nx = 1
        self._ny = 1
        self.nx = int(nx)
        self.ny = int(ny)
        self.dx: float = dx
        self.dy: float = dy
        self.dz: float = dz

        if self.nx < 3 and self.ny < 3:
            raise (ValueError("At least one of (nx, ny) should be of dimension 3"))

    @property
    def nx(self) -> int:
        """Return the number of meshes along the x axis."""
        return self._nx

    @nx.setter
    def nx(self, value: int) -> None:
        if value < 1:
            raise (ValueError("nx should be > 1!)"))
        self._nx = value

    @property
    def ny(self) -> int:
        """Return the number of meshes along the y axis."""
        return self._ny

    @ny.setter
    def ny(self, value: int) -> None:
        if value < 1:
            raise (ValueError("ny should be > 1!)"))
        # if value > 1 and self.nx == 1:
        #     raise (
        #         ValueError("For a 1D case, set nx different "
        # "from 1 and ny equal to 1!")
        #     )
        self._ny = value

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the grid as (nx, ny)."""
        return (self.nx, self.ny)

    @property
    def mesh_area(self) -> float:
        """Return the area of a voxel in m2."""
        return self.dx * self.dy

    @property
    def mesh_volume(self) -> float:
        """Return the volume of a voxel in m3."""
        return self.mesh_area * self.dz


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
        Concentration, used only if flowrates is positive.
    """

    __slots__ = ["name", "node_ids", "times", "flowrates", "concentrations"]

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

        """
        self.name = name
        self.node_ids = np.array(node_ids).reshape(-1)
        self.times = np.array(times).reshape(-1)
        self.flowrates = np.array(flowrates).reshape(-1)
        self.concentrations = np.array(concentrations).reshape(-1)

        if (
            self.concentrations.size != self.times.size
            or self.flowrates.size != self.times.size
        ):
            raise ValueError(
                "Times, flowrates and concentrations must have the same dimension !"
            )

    def get_node_indices(self, geometry: Geometry) -> NDArrayInt:
        """Return the node indices."""
        return np.array(
            node_number_to_indices(self.node_ids, nx=geometry.nx, ny=geometry.ny)
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
        while time > self.times[time_index]:
            time_index += 1
            if time_index == len(self.times):
                break
        if time != self.times[min(time_index, len(self.times) - 1)]:
            time_index -= 1
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

    span: Union[NDArrayInt, Tuple[slice, slice]]


@dataclass
class ConstantHead(BoundaryCondition):
    """
    Represent a constant head condition (Dirichlet).

    Parameters
    ----------
    span: slice
        The span over which the condition applies.
    """

    span: Union[NDArrayInt, Tuple[slice, slice], slice]


@dataclass
class ConstantConcentration(BoundaryCondition):
    """
    Represent a constant conentration boundary condition (Dirichlet).

    Parameters
    ----------
    span: slice
        The span over which the condition applies.
    """

    span: Union[NDArrayInt, Tuple[slice, slice], slice]


@dataclass
class ZeroConcGradient(BoundaryCondition):
    """
    Represent a zero conentration gradient boundary condition (Neumann).

    Parameters
    ----------
    span: slice
        The span over which the condition applies.
    """

    span: Union[NDArrayInt, Tuple[slice, slice], slice]


class FlowModel:
    """Represent a flow model."""

    __slots__ = [
        "vertical_axis",
        "vertical_mesh_size",
        "crank_nicolson",
        "storage_coefficient",
        "permeability",
        "lhead",
        "lu_darcy_x",
        "lu_darcy_y",
        "lu_darcy_div",
        "boundary_conditions",
        "cst_head_nn",
        "regime",
        "q_prev",
        "q_next",
        "tolerance",
    ]

    def __init__(
        self, geometry: Geometry, time_params: TimeParameters, fl_params: FlowParameters
    ) -> None:
        """Initialize the instance."""
        self.crank_nicolson: float = fl_params.crank_nicolson
        self.storage_coefficient: float = fl_params.storage_coefficient
        self.regime: FlowRegime = fl_params.regime
        self.permeability = (
            np.ones((geometry.nx, geometry.ny), dtype=np.float64)
            * fl_params.permeability
        )

        # These are list of ndarrays
        self.lhead: List[NDArrayFloat] = [
            np.zeros((geometry.nx, geometry.ny), dtype=np.float64)
        ]
        self.lu_darcy_x: List[NDArrayFloat] = []
        self.lu_darcy_y: List[NDArrayFloat] = []
        self.lu_darcy_div: List[NDArrayFloat] = []

        self.boundary_conditions: List[BoundaryCondition] = []
        self.q_prev = lil_array(geometry.nx * geometry.ny)
        self.q_next = lil_array(geometry.nx * geometry.ny)
        self.cst_head_nn: NDArrayInt = np.array([], dtype=np.int32)
        self.tolerance = fl_params.tolerance
        self.vertical_axis = fl_params.vertical_axis
        self.vertical_mesh_size = {
            VerticalAxis.DX: geometry.dx,
            VerticalAxis.DY: geometry.dy,
            VerticalAxis.DZ: geometry.dz,
        }[fl_params.vertical_axis]

    @property
    def head(self) -> NDArrayFloat:
        """
        Return head as array with dimension (nx, ny, nz, nt).

        This is read-only.
        """
        return np.transpose(np.array(self.lhead), axes=(1, 2, 0))

    @property
    def u_darcy_x(self) -> NDArrayFloat:
        """
        Return x-darcy velocities as array with dimension (nx, ny, nz, nt + 1).

        This is read-only.
        """
        return np.transpose(np.array(self.lu_darcy_x), axes=(1, 2, 0))

    @property
    def u_darcy_y(self) -> NDArrayFloat:
        """
        Return y-darcy velocities as array with dimension (nx, ny, nz, nt + 1).

        This is read-only.
        """
        return np.transpose(np.array(self.lu_darcy_y), axes=(1, 2, 0))

    @property
    def u_darcy_div(self) -> NDArrayFloat:
        """
        Return darcy divergence as array with dimension (nx, ny, nz, nt + 1).

        This is read-only.
        """
        return np.transpose(np.array(self.lu_darcy_div), axes=(1, 2, 0))

    @property
    def pressure(self) -> NDArrayFloat:
        """
        Return the pressure in pascals (Pa).

        This is read-only (for now).
        """
        return self.head

    def set_initial_head(
        self,
        values: Union[float, int, NDArrayInt, NDArrayFloat],
        span: Union[NDArrayInt, Tuple[slice, slice], NDArrayBool] = (
            slice(None),
            slice(None),
        ),
    ) -> None:
        """Set the initial head field."""
        self.lhead[0][span] = values

    def set_initial_pressure(
        self,
        values: Union[float, int, NDArrayInt, NDArrayFloat],
        span: Union[NDArrayInt, Tuple[slice, slice], NDArrayBool] = (
            slice(None),
            slice(None),
        ),
    ) -> None:
        """Set the initial pressure field in Pa."""
        self.lhead[0][span] = values

    def add_boundary_conditions(self, condition: BoundaryCondition) -> None:
        """Add a boundary condition to the flow model."""
        if not isinstance(condition, ConstantHead):
            raise ValueError(
                f"{condition} is not a valid boundary condition for the flow model !"
            )
        self.boundary_conditions.append(condition)

    def set_constant_head_indices(self) -> None:
        """Set the indices of nodes with constant head."""
        node_numbers = np.array([], dtype=np.int32)
        for condition in self.boundary_conditions:
            if isinstance(condition, ConstantHead):
                node_numbers = np.hstack(
                    [
                        node_numbers,
                        span_to_node_numbers_2d(
                            condition.span,
                            self.lhead[0].shape[0],
                            self.lhead[0].shape[1],
                        ),
                    ]
                )
        self.cst_head_nn: NDArrayInt = np.unique(node_numbers.flatten())

    @property
    def cst_head_indices(self) -> NDArrayInt:
        """Return the indices (array) of the constant head meshes."""
        # [:2] to ignore the z axis
        return np.array(
            node_number_to_indices(
                self.cst_head_nn, nx=self.head.shape[0], ny=self.head.shape[1]
            )[:2]
        )

    def reinit(self) -> None:
        """Set all arrays to zero execpt for the initial conditions(first time)."""
        self.lhead = self.lhead[:1]
        self.lu_darcy_x = []
        self.lu_darcy_y = []
        self.lu_darcy_div = []
        self.set_constant_head_indices()

    @property
    def u_darcy_x_center(self) -> NDArrayFloat:
        """The darcy x-velocities estimated at the mesh centers."""
        tmp = np.zeros((self.head.shape))
        tmp += self.u_darcy_x[:-1, :, :]
        tmp += self.u_darcy_x[1:, :, :]
        tmp /= 2
        return tmp

    @property
    def u_darcy_y_center(self) -> NDArrayFloat:
        """The darcy y-velocities estimated at the mesh centers."""
        tmp = np.zeros((self.head.shape))
        tmp += self.u_darcy_y[:, :-1, :]
        tmp += self.u_darcy_y[:, 1:, :]
        tmp /= 2
        return tmp

    def get_vertical_dim(self) -> int:
        """Return the number of voxel along the vertical_axis axis."""
        if self.vertical_axis == VerticalAxis.DX:
            return self.lhead[0].shape[0]
        elif self.vertical_axis == VerticalAxis.DY:
            return self.lhead[0].shape[1]
        else:
            return 1

    def _get_mesh_center_vertical_pos(self) -> NDArrayFloat:
        """Return the vertical position of the meshes centers."""
        xv, yv = np.meshgrid(range(self.head.shape[0]), range(self.head.shape[1]))
        if self.vertical_axis == VerticalAxis.DX:
            return (xv + 0.5) * self.vertical_mesh_size
        elif self.vertical_axis == VerticalAxis.DY:
            return (yv + 0.5) * self.vertical_mesh_size
        else:
            return np.array([[0.5 * self.vertical_mesh_size]])

    def get_pressure_pa(self) -> NDArrayFloat:
        """Return the pressure in Pa."""
        return (
            (self.head - self._get_mesh_center_vertical_pos().T[:, :, np.newaxis])
            * GRAVITY
            * WATER_DENSITY
        )

    def get_pressure_bar(self) -> NDArrayFloat:
        """Return the pressure in bar."""
        return self.get_pressure_pa() / 1e5


class SaturatedFlowModel(FlowModel):
    __slots__ = [
        "_head",
    ]

    def __init__(
        self, geometry: Geometry, time_params: TimeParameters, fl_params: FlowParameters
    ) -> None:
        """Initialize the instance."""
        super().__init__(geometry, time_params, fl_params)


class DensityFlowModel(FlowModel):
    __slots__ = ["_pressure", "density"]

    def __init__(
        self, geometry: Geometry, time_params: TimeParameters, fl_params: FlowParameters
    ) -> None:
        """Initialize the instance."""
        super().__init__(geometry, time_params, fl_params)

    # Need acceccors for density etc.


class TransportModel:
    """Represent a flow model."""

    __slots__ = [
        "crank_nicolson_diffusion",
        "crank_nicolson_advection",
        "diffusion",
        "porosity",
        "lconc",
        "lgrade",
        "ldensity",
        "grade_prev",
        "boundary_conditions",
        "cst_conc_indices",
        "q_prev_diffusion",
        "q_next_diffusion",
        "q_prev",
        "q_next",
        "tolerance",
        "is_numerical_acceleration",
        "fpi_eps",
        "molar_mass",
    ]

    def __init__(
        self,
        geometry: Geometry,
        time_params: TimeParameters,
        tr_params: TransportParameters,
        gch_params: GeochemicalParameters,
    ) -> None:
        """Initialize the instance."""
        self.crank_nicolson_diffusion: float = tr_params.crank_nicolson_diffusion
        self.crank_nicolson_advection: float = tr_params.crank_nicolson_advection
        self.diffusion = (
            np.ones((geometry.nx, geometry.ny), dtype=np.float64) * tr_params.diffusion
        )
        self.porosity = (
            np.ones((geometry.nx, geometry.ny), dtype=np.float64) * tr_params.porosity
        )
        self.lconc: List[NDArrayFloat] = [
            np.ones((geometry.nx, geometry.ny), dtype=np.float64) * gch_params.conc
        ]

        self.lgrade: List[NDArrayFloat] = [
            np.ones((geometry.nx, geometry.ny), dtype=np.float64) * gch_params.grade
        ]

        self.ldensity: List[NDArrayFloat] = [
            np.ones((geometry.nx, geometry.ny), dtype=np.float64) * gch_params.grade
        ]

        self.grade_prev = np.zeros((geometry.nx, geometry.ny), dtype=np.float64)
        self.boundary_conditions: List[BoundaryCondition] = []
        # q_prev is composed of q_prev_diffusion + advection term
        self.q_prev_diffusion = lil_array(geometry.nx * geometry.ny)
        self.q_next_diffusion = lil_array(geometry.nx * geometry.ny)
        self.q_prev = lil_array(geometry.nx * geometry.ny)
        self.q_next = lil_array(geometry.nx * geometry.ny)
        self.cst_conc_indices: NDArrayInt = np.array([], dtype=np.int32)
        self.tolerance = tr_params.tolerance
        self.is_numerical_acceleration = tr_params.is_numerical_acceleration
        self.fpi_eps = tr_params.fpi_eps
        self.molar_mass = gch_params.Ms

    @property
    def conc(self) -> NDArrayFloat:
        """
        Return mobile concentrations as array with dimension (nx, ny, nz, nt + 1).

        This is read-only.
        """
        return np.transpose(np.array(self.lconc), axes=(1, 2, 0))

    @property
    def grade(self) -> NDArrayFloat:
        """
        Return immobile concentrations as array with dimension (nx, ny, nz, nt + 1).

        This is read-only.
        """
        return np.transpose(np.array(self.lgrade), axes=(1, 2, 0))

    @property
    def density(self) -> NDArrayFloat:
        """
        Return densities in g/l as array with dimension (nx, ny, nz, nt + 1).

        This is read-only.
        """
        return np.transpose(np.array(self.ldensity), axes=(1, 2, 0))

    @property
    def effective_diffusion(self) -> NDArrayFloat:
        """Return the effective diffusion (diffusion * porosity)."""
        return self.diffusion * self.porosity

    def set_initial_grade(
        self,
        values: Union[float, int, NDArrayInt, NDArrayFloat],
        span: Union[NDArrayInt, Tuple[slice, slice], NDArrayBool] = (
            slice(None),
            slice(None),
        ),
    ) -> None:
        """Set the initial grades."""
        self.lgrade[0][span] = values

    def set_initial_conc(
        self,
        values: Union[float, int, NDArrayInt, NDArrayFloat],
        span: Union[NDArrayInt, Tuple[slice, slice], NDArrayBool] = (
            slice(None),
            slice(None),
        ),
    ) -> None:
        """Set the initial concentrations."""
        self.lconc[0][span] = values

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

    def set_constant_conc_indices(self) -> None:
        """Set the indices of nodes with constant head."""
        node_numbers = np.array([], dtype=np.int32)
        for condition in self.boundary_conditions:
            if isinstance(condition, ConstantConcentration):
                node_numbers = np.hstack(
                    [
                        node_numbers,
                        span_to_node_numbers_2d(
                            condition.span, self.conc.shape[0], self.conc.shape[1]
                        ),
                    ]
                )
        self.cst_conc_indices: NDArrayInt = np.unique(node_numbers.flatten())

    def reinit(self) -> None:
        """Set all arrays to zero execpt for the initial conditions(first time)."""
        self.lconc = self.lconc[:1]
        self.lgrade = self.lgrade[:1]
        self.grade_prev = self.lgrade[0]
        self.ldensity = self.ldensity[:1]
        self.set_constant_conc_indices()


class ForwardModel:
    """
    Class representing the reactive transport model.

    wadv: float
        Advection weight (for testing between 0.0 and 1.0). The default is 1.0.

    """

    slots = [
        "geometry",
        "time_params",
        "fl_params",
        "tr_params",
        "gch_params",
        "source_terms",
        "boundary_conditions",
    ]

    def __init__(
        self,
        geometry: Geometry,
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
        geometry : Geometry
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
        self.geometry: Geometry = geometry
        self.time_params: TimeParameters = time_params
        self.gch_params: GeochemicalParameters = gch_params
        self.fl_model: FlowModel = FlowModel(geometry, time_params, fl_params)
        self.tr_model: TransportModel = TransportModel(
            geometry, time_params, tr_params, gch_params
        )
        if source_terms is not None:
            self.source_terms: List[SourceTerm] = object_or_object_sequence_to_list(
                source_terms
            )
        else:
            self.source_terms: List[SourceTerm] = []
        if boundary_conditions is None:
            return
        for condition in object_or_object_sequence_to_list(boundary_conditions):
            self.add_boundary_conditions(condition)

    def get_sources(
        self, time: float, geometry: Geometry
    ) -> Tuple[NDArrayFloat, NDArrayFloat]:
        """Get the flow sources and sink terms."""

        _fl_src = np.zeros((geometry.nx, geometry.ny))
        _conc_src = np.zeros((geometry.nx, geometry.ny))

        # iterate the source terms
        for source in self.source_terms:
            # identify the source term applying
            _fl, _conc = source.get_values(time)
            nids = source.get_node_indices(geometry)

            # Add the flowrates contribution
            _fl_src[nids[0], nids[1]] += _fl / source.n_nodes

            # Keep only non negative flowrates (remove sink terms)
            if _fl > 0:
                # if flw < 0
                _conc_src[nids[0], nids[1]] += _fl * _conc / source.n_nodes

        for condition in self.fl_model.boundary_conditions:
            if isinstance(condition, ConstantHead):
                # Set zero where there constant head
                _fl_src[condition.span] = 0.0

        for condition in self.tr_model.boundary_conditions:
            if isinstance(condition, ConstantConcentration):
                # Set zero where there constant concentration
                _conc_src[condition.span] = 0.0

        return (
            _fl_src / self.geometry.mesh_volume,
            _conc_src / self.geometry.mesh_volume,
        )

    def add_src_term(self, source_term: SourceTerm) -> None:
        """Add a source term."""
        self.source_terms.append(source_term)

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
        """Set all arrays to zero execpt for the initial conditions(first time)."""
        self.fl_model.reinit()
        self.tr_model.reinit()
        self.time_params.reset_to_init()


def remove_cst_bound_indices(
    indices_owner: NDArrayInt, indices_neigh: NDArrayInt, indices_to_remove: NDArrayInt
) -> Tuple[NDArrayInt, NDArrayInt]:
    """
    Remove the indices because of the boundary condition.

    Parameters
    ----------
    indices_owner : _type_
        Indices of owner meshes.
    indices_neigh : _type_
        Indices of neighbor meshes.
    indices_to_remove : DArrayInt
        Indices to remove.

    Returns
    -------
    Tuple[NDArrayInt, NDArrayInt]
        _description_
    """
    is_kept = ~np.isin(indices_owner, indices_to_remove)
    return indices_owner[is_kept], indices_neigh[is_kept]


def get_owner_neigh_indices(
    geometry: Geometry,
    span_owner: Tuple[slice, slice],
    span_neigh: Tuple[slice, slice],
    indices_to_remove: NDArrayInt,
) -> Tuple[NDArrayInt, NDArrayInt]:
    """_summary_

    Parameters
    ----------
    geometry : Geometry
        _description_
    span_owner : Tuple[slice, slice]
        _description_
    span_neigh : Tuple[slice, slice]
        _description_
    indices_to_remove : NDArrayInt
        _description_

    Returns
    -------
    Tuple[NDArrayInt, NDArrayInt]
        _description_
    """
    # Get indices
    indices_owner: NDArrayInt = span_to_node_numbers_2d(
        span_owner, nx=geometry.nx, ny=geometry.ny
    )
    indices_neigh: NDArrayInt = span_to_node_numbers_2d(
        span_neigh, nx=geometry.nx, ny=geometry.ny
    )
    # Remove constant meshes
    return remove_cst_bound_indices(indices_owner, indices_neigh, indices_to_remove)
