"""Module for generating dimer configurations and dimer curves."""
from typing import List, Union

import numpy as _np
import openmm.unit as _unit
from loguru import logger as _logger
from openff.interchange import Interchange as _Interchange
from openff.toolkit import ForceField as _ForceField
from openff.toolkit import Topology as _Topology
from openmm import LocalEnergyMinimizer as _LocalEnergyMinimizer

from ..cli._sample_train import create_simulation as _create_simulation
from ..cli._sample_train import remove_constraints as _remove_constraints
from ._mcmc import MonteCarloSampler
from ._utils import get_unique_atoms as _get_unique_atoms
from ._utils import get_water_mapping as _get_water_mapping
from ._utils import sort_two_lists as _sort_two_lists
from ._utils import unique_with_delta as _unique_with_delta


class DimerGenerator:
    def __init__(self):
        self.configurations = []
        self.energies = []

    @staticmethod
    def create_dimer_topology(solute_smiles: str, solvent_smiles: str):
        from openff.toolkit import Molecule as _Molecule
        from openff.toolkit import Topology as _Topology

        # Convert the SMILES strings to _Molecule objects
        solute = _Molecule.from_smiles(solute_smiles)
        solvent = _Molecule.from_mapped_smiles(solvent_smiles)

        # Assign residue names
        for atom in solute.atoms:
            atom.metadata["residue_name"] = "LIG"

        for atom in solvent.atoms:
            atom.metadata["residue_name"] = "HOH"

        # Generate conformers
        solute.generate_conformers(n_conformers=1)
        solvent.generate_conformers(n_conformers=1)

        # Create the topology
        topology = _Topology.from_molecules([solute, solvent])

        return topology

    def generate_dimers(
        self,
        topology_off: _Topology,
        n_samples: int = 2500,
        n_lowest: int = 50,
        temperature: float = 1000.0,
        sphere_radius=0.5 * _unit.nanometer,
        forcefields: Union[str, List[str]] = ["openff_unconstrained-2.0.0.offxml"],
    ):
        # Convert OpenFF Topology to OpenMM Topology and get positions
        topology = topology_off.to_openmm()
        positions_omm = topology_off.get_positions().to_openmm()
        qm_region = [atom.index for atom in list(topology.chains())[0].atoms()]

        # Get atom indices for O, H1, and H2 atoms in a water molecule
        water_mapping = _get_water_mapping(topology)

        # Initialize force field and interchange object
        forcefields = forcefields if isinstance(forcefields, list) else [forcefields]
        forcefield = _ForceField(*forcefields)
        interchange = _Interchange.from_smirnoff(
            force_field=forcefield, topology=topology_off
        )

        # Create simulation instance
        simulation = _create_simulation(
            interchange, temperature=temperature, pressure=None
        )

        # Remove constraints involving alchemical atoms
        _remove_constraints(simulation.system, qm_region)
        simulation.context.reinitialize(preserveState=True)

        # Create Monte Carlo sampler
        mc = MonteCarloSampler()

        # Get atoms with unique chemical environments in the target molecule
        unique_atoms = _get_unique_atoms(topology)
        atom_indices = [water_mapping["O"], water_mapping["H1"], water_mapping["H2"]]

        # Sample dimers
        _logger.info("Sampling dimers.")
        for atom in unique_atoms:
            _logger.info(f"Sampling dimers for atom {atom}.")
            mc.sample(
                simulation.context,
                n_samples=n_samples,
                temperature=temperature,
                sphere_radius=sphere_radius,
                sphere_centre=positions_omm[atom],
                atom_indices=atom_indices,
            )

            # Get unique dimers with the lowest energy
            energies_unique, mask = _unique_with_delta(mc.energies, delta=0.5)
            configurations_unique = [mc.configurations[i] for i in mask]
            energies_unique, configurations_unique = _sort_two_lists(
                energies_unique, configurations_unique
            )

            # Append the unique dimers to the list
            self.configurations.extend(configurations_unique[:n_lowest])
            self.energies.extend(energies_unique[:n_lowest])

            # Reset the Monte Carlo sampler
            mc.reset()

        # Optimize the dimers with the lowest energy
        optimised_energies = []
        optimised_configurations = []
        for config in self.configurations:
            _logger.info("Optimizing dimer configuration.")
            simulation.context.setPositions(config)
            _LocalEnergyMinimizer.minimize(simulation.context, tolerance=1.0)
            optimised_configurations.append(
                simulation.context.getState(getPositions=True)
                .getPositions(asNumpy=True)
                ._value
            )
            optimised_energies.append(
                simulation.context.getState(getEnergy=True).getPotentialEnergy()._value
            )

        # Get unique dimers with the lowest energy
        energies_unique, mask = _unique_with_delta(optimised_energies, delta=0.5)
        configurations_unique = [optimised_configurations[i] for i in mask]
        energies_unique, configurations_unique = _sort_two_lists(
            energies_unique, configurations_unique
        )

        return energies_unique, configurations_unique

    @staticmethod
    def generate_dimer_curve(ref_positions, solute_indices, solvent_indices):
        from scipy.spatial import distance

        distances = distance.cdist(
            XA=ref_positions[solute_indices], XB=ref_positions[solvent_indices]
        )

        # Find the minimum distance indices
        solute_index, solvent_index = _np.unravel_index(
            _np.argmin(distances), distances.shape
        )
        solvent_index = solvent_indices[solvent_index]

        # Displace the solvent molecule along the vector connecting
        # the closest solute and solvent atoms
        solute_position = ref_positions[solute_index]
        solvent_position = ref_positions[solvent_index]
        dist_vec = solvent_position - solute_position

        # Displace the solvent molecule along the vector
        dimer_curve = []
        distances = _np.linspace(0.7, 1.5, 11)
        for dist in distances:
            new_pos = ref_positions.copy()
            new_pos[solvent_indices] += (dist - 1.0) * dist_vec
            dimer_curve.append(new_pos)

        return dimer_curve
