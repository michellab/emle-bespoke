"""Module for generating dimer configurations and dimer curves."""

import numpy as _np
import openmm.unit as _unit
from loguru import logger as _logger

from .._constants import ATOMIC_NUMBERS_TO_SYMBOLS as _ATOMIC_NUMBERS_TO_SYMBOLS
from ..reference_data import ReferenceData as _ReferenceData
from ._base import BaseSampler as _BaseSampler


class DimerSampler(_BaseSampler):
    def __init__(
        self,
        system,
        context,
        integrator,
        topology,
        qm_region,
        qm_calculator,
        reference_data=None,
        energy_scale=1.0,
        length_scale=1.0,
    ):
        super().__init__(
            system=system,
            context=context,
            integrator=integrator,
            topology=topology,
            reference_data=reference_data if reference_data else _ReferenceData(),
            qm_calculator=qm_calculator,
            horton_calculator=None,
            energy_scale=energy_scale,
            length_scale=length_scale,
        )
        # Specific to the dimer sampler
        self._atomic_numbers = _np.array(
            [a.element.atomic_number for a in topology.atoms()],
            dtype=_np.int64,
        )
        self._qm_region = _np.array(qm_region, dtype=_np.int64)

        # Get the point charges
        self._point_charges = self._get_point_charges()

        # Initialize lists to store dimer configurations and energies
        self.configurations = []
        self.energies = []

    @staticmethod
    def generate_dimer_curve(ref_positions, solute_mask, solvent_mask):
        from scipy.spatial import distance

        distances = distance.cdist(
            XA=ref_positions[solute_mask], XB=ref_positions[solvent_mask]
        )

        # Find the minimum distance indices
        solute_index, solvent_index = _np.unravel_index(
            _np.argmin(distances), distances.shape
        )
        solvent_index = solvent_mask.nonzero()[0][solvent_index]

        # Displace the solvent molecule along the vector connecting
        # the closest solute and solvent atoms
        solute_position = ref_positions[solute_index]
        solvent_position = ref_positions[solvent_index]
        dist_vec = solvent_position - solute_position

        # Displace the solvent molecule along the vector
        dimer_curve = []
        distances = _np.linspace(0.7, 1.5, 3)
        for dist in distances:
            new_pos = ref_positions.copy()
            new_pos[solvent_mask] += (dist - 1.0) * dist_vec
            dimer_curve.append(new_pos)

        return dimer_curve

    def generate_dimers(
        self,
        n_samples: int,
        n_lowest: int,
        temperature: _unit.Quantity,
        sphere_radius: _unit.Quantity,
        delta=0.5,
    ):
        from openmm import LocalEnergyMinimizer as _LocalEnergyMinimizer

        from ._mcmc import MonteCarloSampler
        from ._utils import get_unique_atoms as _get_unique_atoms
        from ._utils import get_water_mapping as _get_water_mapping
        from ._utils import sort_two_lists as _sort_two_lists
        from ._utils import unique_with_delta as _unique_with_delta

        # Convert OpenFF Topology to OpenMM Topology and get positions
        positions_omm = self._context.getState(getPositions=True).getPositions(
            asNumpy=True
        )

        # Get atom indices for O, H1, and H2 atoms in a water molecule
        water_mapping = _get_water_mapping(self._topology)

        # Create Monte Carlo sampler
        mc = MonteCarloSampler()

        # Get atoms with unique chemical environments in the target molecule
        unique_atoms = _get_unique_atoms(self._topology)
        atom_indices = [water_mapping["O"], water_mapping["H1"], water_mapping["H2"]]

        # Sample dimers
        _logger.info("Sampling dimers.")
        configurations_list = []
        energies_list = []
        for atom in unique_atoms:
            _logger.info(f"Sampling dimers for atom {atom}.")
            mc.sample(
                self._context,
                n_samples=n_samples,
                temperature=temperature,
                sphere_radius=sphere_radius,
                sphere_centre=positions_omm[atom],
                atom_indices=atom_indices,
            )

            # Get unique dimers with the lowest energy
            energies_unique, mask = _unique_with_delta(mc.energies, delta=delta)
            configurations_unique = [mc.configurations[i] for i in mask]
            energies_unique, configurations_unique = _sort_two_lists(
                energies_unique, configurations_unique
            )

            # Append the unique dimers to the list
            configurations_list.extend(configurations_unique[:n_lowest])
            energies_list.extend(energies_unique[:n_lowest])

            # Reset the Monte Carlo sampler
            mc.reset()

        # Optimize the dimers with the lowest energy
        optimised_energies = []
        optimised_configurations = []
        for config in configurations_list:
            _logger.info("Optimizing dimer configuration.")
            self._context.setPositions(config)
            _LocalEnergyMinimizer.minimize(self._context, tolerance=1.0)
            optimised_configurations.append(
                self._context.getState(getPositions=True)
                .getPositions(asNumpy=True)
                ._value
            )
            optimised_energies.append(
                self._context.getState(getEnergy=True).getPotentialEnergy()._value
            )

        # Get unique dimers with the lowest energy
        energies_unique, mask = _unique_with_delta(optimised_energies, delta=delta)
        configurations_unique = [optimised_configurations[i] for i in mask]
        energies_unique, configurations_unique = _sort_two_lists(
            energies_unique, configurations_unique
        )

        return energies_unique, configurations_unique

    def sample(
        self,
        n_samples: int = 2500,
        n_lowest: int = 50,
        temperature: float = 1000.0,
        sphere_radius=0.5 * _unit.nanometer,
        delta: float = 0.5,
    ) -> _ReferenceData:
        energies_dimers, configs_dimers = self.generate_dimers(
            n_samples=n_samples,
            n_lowest=n_lowest,
            temperature=temperature,
            sphere_radius=sphere_radius,
            delta=delta,
        )

        # Get the masks for the solute and solvent atoms
        natoms = self._topology.getNumAtoms()
        solute_mask = _np.zeros(natoms, dtype=bool)
        solute_mask[self._qm_region] = True
        solvent_mask = ~solute_mask

        # Generate dimer curves
        curves = [
            self.generate_dimer_curve(config, solute_mask, solvent_mask)
            for config in configs_dimers
        ]

        # Calculate the interaction energy for each curve
        symbols_dimer = [_ATOMIC_NUMBERS_TO_SYMBOLS[an] for an in self._atomic_numbers]
        symbols_solute = [symbols_dimer[i] for i in solute_mask.nonzero()[0]]
        symbols_solvent = [symbols_dimer[i] for i in solvent_mask.nonzero()[0]]

        for i, curve in enumerate(curves):
            _logger.info(f"Calculating interaction energy for dimer curve {i + 1}.")
            _logger.debug("Calculating QM energy of solute.")
            solute_energy = self._qm_calculator.get_potential_energy(
                elements=symbols_solute,
                positions=curve[0][solute_mask] * 10.0,
                directory="solute_vacuum",
                orca_simple_input="! b3lyp cc-pvtz TightSCF D3BJ",
                orca_blocks="%MaxCore 1024\n%pal\nnprocs 8\nend\n",
            )
            _logger.debug("Calculating QM energy of solute.")
            solvent_energy = self._qm_calculator.get_potential_energy(
                elements=symbols_solvent,
                positions=curve[0][solvent_mask] * 10.0,
                directory="solvent_vacuum",
                orca_simple_input="! b3lyp cc-pvtz TightSCF D3BJ",
                orca_blocks="%MaxCore 1024\n%pal\nnprocs 8\nend\n",
            )
            for j, config in enumerate(curve):
                _logger.info(
                    f"Calculating interaction energy for configuration {j + 1} / {len(curve)}."
                )
                dimer_energy = self._qm_calculator.get_potential_energy(
                    elements=symbols_dimer,
                    positions=config * 10.0,
                    directory="dimer_vacuum",
                    orca_simple_input="! b3lyp cc-pvtz TightSCF D3BJ",
                    orca_blocks="%MaxCore 1024\n%pal\nnprocs 8\nend\n",
                )

                interaction_energy = dimer_energy - solute_energy - solvent_energy

                self._reference_data.add_data_to_key("e_int", interaction_energy)
                self._reference_data.add_data_to_key("e_dimer", dimer_energy)
                self._reference_data.add_data_to_key("e_solute", solute_energy)
                self._reference_data.add_data_to_key("e_solvent", solvent_energy)
                self._reference_data.add_data_to_key(
                    "z", self._atomic_numbers[solute_mask]
                )
                self._reference_data.add_data_to_key("solute_mask", solute_mask)
                self._reference_data.add_data_to_key("solvent_mask", solvent_mask)
                self._reference_data.add_data_to_key("xyz_qm", config[solute_mask])
                self._reference_data.add_data_to_key("xyz_mm", config[solvent_mask])

        return self._reference_data
