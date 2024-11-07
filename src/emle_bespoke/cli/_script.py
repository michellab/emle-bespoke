"""Main script for generating reference data and training a bespoke EMLE model."""
# General imports
import argparse
import re
from typing import Any, Dict, Optional


# OpenMM imports
import openmm as _mm
import openmm.app as _app
import openmm.unit as _unit

# OpenFF imports
from openff.toolkit import ForceField, Molecule, unit
from openff.interchange import Interchange
from openff.interchange.components._packmol import  _pack_box
from openff.interchange.components._packmol import UNIT_CUBE
from openff.units import unit as offunit

# Imports from the emle-bespoke package
from .. import EMLEBespoke as EMLEBespoke
from .. import ReferenceDataSampler as ReferenceDataSampler
from ..calculators import HortonCalculator as HortonCalculator
from ..calculators import ORCACalculator as ORCACalculator


PACKMOL_KWARGS = {
    "box_shape": UNIT_CUBE,
    "mass_density": 1.0 * offunit.gram / offunit.milliliter,
}


def is_mapped_smiles(smiles: str) -> bool:
    """
    Check if the given SMILES string is a mapped SMILES.

    Parameters
    ----------
    smiles : str
        The SMILES string to check.

    Returns
    -------
    bool
        Whether the SMILES string contains mapping information (e.g., ":1", ":2").
    """
    pattern = re.compile(r":[0-9]+")  # Regular expression to detect mapped atom indices
    return bool(pattern.search(smiles))


def create_molecule(smiles: str) -> Molecule:
    """
    Create an OpenFF molecule from a SMILES string.

    Parameters
    ----------
    smiles : str
        The SMILES string of the molecule.

    Returns
    -------
    Molecule
        The OpenFF molecule.
    """
    if is_mapped_smiles(smiles):
        return Molecule.from_mapped_smiles(smiles)
    return Molecule.from_smiles(smiles)


def create_off_topology(
    n_solute: int,
    n_solvent: int,
    solvent_smiles: str,
    solute_smiles: str,
    packmol_kwargs: Optional[Dict[str, Any]] = None
) -> openff.interchange.Topology:
    """
    Create an OpenFF topology for a system with solute and solvent molecules.

    Parameters
    ----------
    n_solute : int
        The number of solute molecules.
    n_solvent : int
        The number of solvent molecules.
    solvent_smiles : str
        The SMILES string for the solvent.
    solute_smiles : str
        The SMILES string for the solute.
    packmol_kwargs : dict, optional
        Additional parameters for Packmol packaging, by default None.

    Returns
    -------
    openff.interchange.Topology
        The created OpenFF topology.
    """
    solute = create_molecule(solute_smiles)
    if n_solvent == 0 or solvent_smiles is None:
        return solute.to_topology()
    
    solvent = create_molecule(solvent_smiles)
    if solvent_smiles == "[H:2][O:1][H:3]":  # Handle water molecule case
        for atom in solvent.atoms:
            atom.metadata["residue_name"] = "HOH"

    return _pack_box(
        molecules=[solute, solvent],
        number_of_copies=[n_solute, n_solvent],
        **PACKMOL_KWARGS,
    )


def create_simulation(
    interchange: Interchange,
    temperature: float = 300.0,
    pressure: float = 1.0,
    timestep: float = 1.0,
    friction_coefficient: float = 1.0
) -> openmm.app.Simulation:
    """
    Create an OpenMM simulation from the provided interchange object.

    Parameters
    ----------
    interchange : Interchange
        The Interchange object containing the system topology and parameters.
    temperature : float, optional
        The temperature in Kelvin, by default 300.0.
    pressure : float, optional
        The pressure in bar, by default 1.0.
    timestep : float, optional
        The timestep in femtoseconds, by default 1.0.
    friction_coefficient : float, optional
        The Langevin friction coefficient in picoseconds^-1, by default 1.0.

    Returns
    -------
    openmm.app.Simulation
        The OpenMM simulation object.
    """
    integrator = openmm.LangevinIntegrator(
        temperature * openmm.unit.kelvin,
        friction_coefficient / openmm.unit.picosecond,
        timestep * openmm.unit.femtoseconds,
    )

    barostat = openmm.MonteCarloBarostat(
        pressure * openmm.unit.bar,
        temperature * openmm.unit.kelvin,
    )

    simulation = interchange.to_openmm_simulation(
        combine_nonbonded_forces=True,
        integrator=integrator,
        additional_forces=[barostat],
    )

    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature * openmm.unit.kelvin)
    simulation.context.computeVirtualSites()

    return simulation


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference data and train a bespoke EMLE model."
    )

    # Command-line arguments
    parser.add_argument("--n_solute", type=int, default=1, help="Number of ligands in the system.")
    parser.add_argument("--n_solvent", type=int, default=1000, help="Number of water molecules in the system.")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to generate.")
    parser.add_argument("--n_steps", type=int, default=1000, help="Number of simulation steps to run.")
    parser.add_argument("--solute", type=str, default="c1ccccc1", help="The ligand SMILES string.")
    parser.add_argument("--solvent", type=str, default="[H:2][O:1][H:3]", help="The solvent SMILES string.")
    parser.add_argument("--forcefield", type=str, default="openff-2.0.0.offxml", help="The force field to use.")
    parser.add_argument("--temperature", type=float, default=298.15, help="The simulation temperature in Kelvin.")
    parser.add_argument("--pressure", type=float, default=1.0, help="The simulation pressure in bar.")
    parser.add_argument("--timestep", type=float, default=1.0, help="The simulation timestep in femtoseconds.")
    parser.add_argument("--friction_coefficient", type=float, default=1.0, help="The Langevin friction coefficient in picoseconds^-1.")

    # Parse the arguments
    args = parser.parse_args()

    # Create the ligand and solvate the system
    ligand = create_molecule(args.solute)
    topology_off = create_off_topology(args.n_solute, args.n_solvent, args.solvent, args.solute)

    print(f"Topology: {topology_off.n_molecules}, {topology_off.box_vectors}, {topology_off.get_positions().shape}")

    # Create the Interchange object
    force_field = ForceField(args.forcefield)
    interchange = Interchange.from_smirnoff(
        force_field=force_field, topology=topology_off
    )
    print(f"Interchange: {interchange.topology.n_atoms}, {interchange.box}, {interchange.positions.shape}")

    # Create and run the simulation
    simulation = create_simulation(
        interchange=interchange,
        temperature=args.temperature,
        pressure=args.pressure,
        timestep=args.timestep,
        friction_coefficient=args.friction_coefficient,
    )

    # Create the EMLE bespoke object and train the model
    ref_sampler = ReferenceDataSampler(
        system=simulation.system,
        context=simulation.context,
        integrator=simulation.integrator,
        topology=topology_off.to_openmm(),
        qm_region=None,
        qm_calculator=ORCACalculator(),
        horton_calculator=HortonCalculator(),
    )

    emle_bespoke = EMLEBespoke(ref_sampler)
    emle_bespoke.train_model(n_samples=args.n_samples, n_steps=args.n_steps)


if __name__ == "__main__":
    main()
