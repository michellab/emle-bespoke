"""Base class for OpenMM-based samplers."""

import numpy as _np
import openmm as _mm

from ._base import Sampler


class OpenMMSampler(Sampler):
    """
    Base class for OpenMM-based samplers.

    This class extends the base Sampler class with OpenMM-specific functionality.
    It provides methods to handle OpenMM-specific operations and state management.

    Parameters
    ----------
    system : simtk.openmm.System
        OpenMM system.
    context : simtk.openmm.Context
        OpenMM context.
    integrator : simtk.openmm.Integrator
        OpenMM integrator.
    topology : simtk.openmm.app.Topology
        OpenMM topology.
    energy_scale : float, optional
        Energy scale to convert from the QM calculator energy units to kJ/mol.
    length_scale : float, optional
        Length scale to convert from the QM calculator length units to Angstrom.

    Attributes
    ----------
    Inherits all attributes from Sampler class.

    _system : simtk.openmm.System
        OpenMM system.
    _context : simtk.openmm.Context
        OpenMM context.
    _integrator : simtk.openmm.Integrator
        OpenMM integrator.
    _topology : simtk.openmm.app.Topology
        OpenMM topology.
    """

    def __init__(
        self,
        system: _mm.System,
        context: _mm.Context,
        integrator: _mm.Integrator,
        topology: _mm.app.Topology,
        energy_scale: float = 1.0,
        length_scale: float = 1.0,
    ):
        """Initialize the OpenMMSampler."""
        # Call the parent class's __init__
        super().__init__(energy_scale=energy_scale, length_scale=length_scale)

        # Set the OpenMM objects
        self._system = system
        self._context = context
        self._integrator = integrator
        self._topology = topology

    def get_state(self) -> _mm.State:
        """
        Get the current state of the OpenMM system.

        Returns
        -------
        state : simtk.openmm.State
            Current state of the OpenMM system.
        """
        return self._context.getState(
            getPositions=True, getVelocities=True, getEnergy=True
        )

    def set_state(self, state: _mm.State) -> None:
        """
        Set the state of the OpenMM system.

        Parameters
        ----------
        state : simtk.openmm.State
            State to set the system to.
        """
        self._context.setState(state)

    def get_positions(self) -> _np.ndarray:
        """
        Get the current positions of all atoms in the system.

        Returns
        -------
        positions : numpy.ndarray
            Current positions of all atoms in the system.
        """
        state = self.get_state()
        return state.getPositions(asNumpy=True)

    def get_velocities(self) -> _np.ndarray:
        """
        Get the current velocities of all atoms in the system.

        Returns
        -------
        velocities : numpy.ndarray
            Current velocities of all atoms in the system.
        """
        state = self.get_state()
        return state.getVelocities(asNumpy=True)

    def get_potential_energy(self) -> float:
        """
        Get the current potential energy of the system.

        Returns
        -------
        energy : float
            Current potential energy of the system.
        """
        state = self.get_state()
        return state.getPotentialEnergy().value_in_unit(_mm.unit.kilojoules_per_mole)

    def get_kinetic_energy(self) -> float:
        """
        Get the current kinetic energy of the system.

        Returns
        -------
        energy : float
            Current kinetic energy of the system.
        """
        state = self.get_state()
        return state.getKineticEnergy().value_in_unit(_mm.unit.kilojoules_per_mole)

    def get_total_energy(self) -> float:
        """
        Get the current total energy of the system.

        Returns
        -------
        energy : float
            Current total energy of the system.
        """
        return self.get_potential_energy() + self.get_kinetic_energy()

    def step(self, steps: int = 1) -> None:
        """
        Take a specified number of integration steps.

        Parameters
        ----------
        steps : int, optional
            Number of integration steps to take.
        """
        self._integrator.step(steps)

    def _get_point_charges(self) -> _np.ndarray:
        """
        Get the point charges from the system.

        Returns
        -------
        point_charges: _np.ndarray(NATOMS)
        """
        assert self._system, "The system must be set before getting the point charges."
        non_bonded_force = [
            f for f in self._system.getForces() if isinstance(f, _mm.NonbondedForce)
        ][0]
        point_charges = _np.zeros(self._topology.getNumAtoms(), dtype=_np.float64)
        for i in range(non_bonded_force.getNumParticles()):
            # charge, sigma, epsilon
            charge, _, _ = non_bonded_force.getParticleParameters(i)
            point_charges[i] = charge._value
        return point_charges
