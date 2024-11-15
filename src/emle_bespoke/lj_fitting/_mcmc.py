from copy import deepcopy as _deepcopy

import numpy as np
import openmm.unit as _unit
from loguru import logger as _logger


class MonteCarloSampler:
    def __init__(self, log_frequency=100):
        self.configurations = []
        self.energies = []
        self._log_frequency = log_frequency

    @staticmethod
    def random_rotation_matrix():
        """
        Generate a random axis and angle for rotation of the water coordinates (using the
        method used for this in the ProtoMS source code (www.protoms.org), and then return
        a rotation matrix produced from these

        Returns
        -------
        rot_matrix : numpy.ndarray
            Rotation matrix generated
        """
        # First generate a random axis about which the rotation will occur
        rand1 = rand2 = 2.0

        while (rand1**2 + rand2**2) >= 1.0:
            rand1 = np.random.rand()
            rand2 = np.random.rand()
        rand_h = 2 * np.sqrt(1.0 - (rand1**2 + rand2**2))
        axis = np.array(
            [rand1 * rand_h, rand2 * rand_h, 1 - 2 * (rand1**2 + rand2**2)]
        )
        axis /= np.linalg.norm(axis)

        # Get a random angle
        theta = np.pi * (2 * np.random.rand() - 1.0)

        # Simplify products & generate matrix
        x, y, z = axis[0], axis[1], axis[2]
        x2, y2, z2 = axis[0] * axis[0], axis[1] * axis[1], axis[2] * axis[2]
        xy, xz, yz = axis[0] * axis[1], axis[0] * axis[2], axis[1] * axis[2]
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        rot_matrix = np.array(
            [
                [
                    cos_theta + x2 * (1 - cos_theta),
                    xy * (1 - cos_theta) - z * sin_theta,
                    xz * (1 - cos_theta) + y * sin_theta,
                ],
                [
                    xy * (1 - cos_theta) + z * sin_theta,
                    cos_theta + y2 * (1 - cos_theta),
                    yz * (1 - cos_theta) - x * sin_theta,
                ],
                [
                    xz * (1 - cos_theta) - y * sin_theta,
                    yz * (1 - cos_theta) + x * sin_theta,
                    cos_theta + z2 * (1 - cos_theta),
                ],
            ]
        )

        return rot_matrix

    def move_water(self, positions, sphere_radius, sphere_centre, atom_indices):
        """ """

        rand_nums = np.random.randn(3)
        insert_point = sphere_centre + (
            sphere_radius * np.power(np.random.rand(), 1.0 / 3) * rand_nums
        ) / np.linalg.norm(rand_nums)
        #  Generate a random rotation matrix
        R = MonteCarloSampler.random_rotation_matrix()
        new_positions = _deepcopy(positions)
        for i, index in enumerate(atom_indices):
            #  Translate coordinates to an origin defined by the oxygen atom, and normalise
            atom_position = positions[index] - positions[atom_indices[0]]
            # Rotate about the oxygen position
            if i != 0:
                vec_length = np.linalg.norm(atom_position)
                atom_position = atom_position / vec_length
                # Rotate coordinates & restore length
                atom_position = vec_length * np.dot(R, atom_position) * _unit.nanometer
            # Translate to new position
            new_positions[index] = atom_position + insert_point

        return new_positions

    def reset(self):
        """
        Reset the configurations and energies lists.
        """
        self.configurations = []
        self.energies = []

    def sample(
        self, context, n_samples=1000, temperature=298.15, **move_water_kwargs
    ) -> None:
        """
        Perform a Monte Carlo simulation using the Boltzmann distribution to sample
        states of a system based on their energy.

        Parameters
        ----------
        context : OpenMM Context
            The OpenMM Context object for the system to simulate.
        n_samples : int, optional, default=1000
            The number of samples to generate.
        temperature : float, optional, default=298.15
            The temperature of the system in Kelvin.
        move_water_kwargs : dict
            Keyword arguments to pass to the `move_water` method.
        """
        if isinstance(temperature, float):
            temperature = _unit.Quantity(temperature, _unit.kelvin)
        elif not isinstance(temperature, _unit.Quantity):
            raise TypeError("Temperature must be a float or Quantity object.")

        current_state = context.getState(getPositions=True).getPositions(asNumpy=True)
        current_energy = context.getState(getEnergy=True).getPotentialEnergy()

        _logger.debug(
            f"{'Step':<10} | {'Current Energy':<15} | {'New Energy':<15} | {'DeltaE':<15}"
        )

        for _ in range(n_samples):
            # Propose a new water positions
            new_state = self.move_water(positions=current_state, **move_water_kwargs)
            context.setPositions(new_state)

            # Get the energy of the new state
            new_energy = context.getState(getEnergy=True).getPotentialEnergy()

            # Calculate the change in energy
            delta_E = new_energy - current_energy

            if _ % self._log_frequency == 0:
                _logger.debug(
                    f"{_:<10} | {current_energy._value:<15.2f} | {new_energy._value:<15.2f} | {delta_E._value:<15.2f}"
                )

            # Metropolis acceptance criterion
            if delta_E < 0.0 * _unit.kilojoules_per_mole:
                current_state = new_state
                current_energy = new_energy
            elif np.random.rand() < np.exp(
                -delta_E
                / (
                    _unit.BOLTZMANN_CONSTANT_kB
                    * _unit.AVOGADRO_CONSTANT_NA
                    * temperature
                )
            ):
                current_state = new_state
                current_energy = new_energy

            self.configurations.append(
                current_state.in_units_of(_unit.nanometer)._value
            )
            self.energies.append(
                current_energy.in_units_of(_unit.kilojoules_per_mole)._value
            )

        _logger.info(f"Finished MC sampling of {n_samples} configurations.")
