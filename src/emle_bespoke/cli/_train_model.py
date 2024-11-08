"""Main script for generating reference data and training a bespoke EMLE model."""

# General imports
import argparse
import re
import logging
from typing import Any, Dict, Optional

# OpenMM imports
import openmm as _mm
import openmm.app as _app
import openmm.unit as _unit
from openff.interchange import Interchange as _Interchange
from openff.interchange.components._packmol import UNIT_CUBE as _UNIT_CUBE
from openff.interchange.components._packmol import pack_box as _pack_box

# OpenFF imports
from openff.toolkit import ForceField as _ForceField
from openff.toolkit import Molecule as _Molecule
from openff.toolkit import Topology as _Topology
from openff.units import unit as _offunit

# Imports from the emle-bespoke package
from .. import EMLEBespoke as _EMLEBespoke
from .. import ReferenceDataSampler as _ReferenceDataSampler
from ..calculators import HortonCalculator as _HortonCalculator
from ..calculators import ORCACalculator as _ORCACalculator

PACKMOL_KWARGS = {
    "box_shape": _UNIT_CUBE,
    "target_density": 1.0 * _offunit.gram / _offunit.milliliter,
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_mapped_smiles(smiles: str) -> bool:
    """Check if the given SMILES string is a mapped SMILES."""
    pattern = re.compile(r":[0-9]+")
    return bool(pattern.search(smiles))


def create_molecule(smiles: str) -> _Molecule:
    """Create an OpenFF molecule from a SMILES string."""
    try:
        if is_mapped_smiles(smiles):
            return _Molecule.from_mapped_smiles(smiles)
        return _Molecule.from_smiles(smiles)
    except Exception as e:
        logger.error(f"Failed to create molecule from SMILES '{smiles}': {e}")
        raise


def create_off_topology(
    n_solute: int,
    n_solvent: int,
    solvent_smiles: str,
    solute_smiles: str,
    packmol_kwargs: Optional[Dict[str, Any]] = None,
) -> _Topology:
    """Create an OpenFF topology for a system with solute and solvent molecules."""
    try:
        logger.info("Creating OpenFF topology.")
        logger.info(f"Number of solute molecules: {n_solute}")
        logger.info(f"Number of solvent molecules: {n_solvent}")
        logger.info(f"Solute SMILES: {solute_smiles}")
        logger.info(f"Solvent SMILES: {solvent_smiles}")
        for key, value in (packmol_kwargs or PACKMOL_KWARGS).items():
            logger.info(f"Packmol kwarg: {key} = {value}")
            
        solute = create_molecule(solute_smiles)
        if n_solvent == 0 or solvent_smiles is None:
            return solute.to_topology()

        solvent = create_molecule(solvent_smiles)
        if solvent_smiles == "[H:2][O:1][H:3]":  # Handle water case
            for atom in solvent.atoms:
                atom.metadata["residue_name"] = "HOH"

        return _pack_box(
            molecules=[solute, solvent],
            number_of_copies=[n_solute, n_solvent],
            **(packmol_kwargs or PACKMOL_KWARGS),
        )
    except Exception as e:
        logger.error(f"Failed to create OpenFF topology: {e}")
        raise RuntimeError(f"Failed to create OpenFF topology: {e}")


def create_simulation(
    interchange: _Interchange,
    temperature: float = 300.0,
    pressure: float = 1.0,
    timestep: float = 1.0,
    friction_coefficient: float = 1.0,
) -> _app.Simulation:
    """Create an OpenMM simulation from the provided interchange object."""
    try:
        logger.info("Creating OpenMM simulation.")
        logger.info(f"Temperature: {temperature} K")
        logger.info(f"Pressure: {pressure} bar")
        logger.info(f"Timestep: {timestep} fs")
        logger.info(f"Friction coefficient: {friction_coefficient} ps^-1")
        
        integrator = _mm.LangevinIntegrator(
            temperature * _unit.kelvin,
            friction_coefficient / _mm.unit.picosecond,
            timestep * _mm.unit.femtoseconds,
        )

        barostat = _mm.MonteCarloBarostat(
            pressure * _unit.bar,
            temperature * _unit.kelvin,
        )

        simulation = interchange.to_openmm_simulation(
            combine_nonbonded_forces=True,
            integrator=integrator,
            additional_forces=[barostat],
        )

        simulation.minimizeEnergy()
        simulation.context.setVelocitiesToTemperature(temperature * _unit.kelvin)
        simulation.context.computeVirtualSites()

        return simulation
    except Exception as e:
        logger.error(f"Failed to create OpenMM simulation: {e}")
        raise RuntimeError(f"Failed to create OpenMM simulation: {e}")


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
    parser.add_argument("--forcefield", type=str, default="openff-2.0.0.offxml", help="The force field(s) to use, separated by commas.")
    parser.add_argument("--temperature", type=float, default=298.15, help="Simulation temperature in Kelvin.")
    parser.add_argument("--pressure", type=float, default=1.0, help="Simulation pressure in bar.")
    parser.add_argument("--timestep", type=float, default=1.0, help="Simulation timestep in femtoseconds.")
    parser.add_argument("--friction_coefficient", type=float, default=1.0, help="Langevin friction coefficient (ps^-1).")

    # Parse the arguments
    args = parser.parse_args()

    # Validation of critical arguments
    if args.n_solvent < 0 or args.n_solute < 0:
        logger.error("n_solvent and n_solute must be non-negative.")
        return

    # Create topology
    topology_off = create_off_topology(
        n_solute=args.n_solute, 
        n_solvent=args.n_solvent, 
        solvent_smiles=args.solvent, 
        solute_smiles=args.solute
    )

    # Initialize force field and interchange object
    force_field = _ForceField(*args.forcefield.split(","))
    interchange = _Interchange.from_smirnoff(force_field=force_field, topology=topology_off)

    # Create simulation object
    simulation = create_simulation(
        interchange=interchange,
        temperature=args.temperature,
        pressure=args.pressure,
        timestep=args.timestep,
        friction_coefficient=args.friction_coefficient,
    )

    # Define QM region and train model
    topology = topology_off.to_openmm()
    qm_region = [atom.index for atom in list(topology.chains())[0].atoms()]

    ref_sampler = _ReferenceDataSampler(
        system=simulation.system,
        context=simulation.context,
        integrator=simulation.integrator,
        topology=topology,
        qm_region=qm_region,
        qm_calculator=_ORCACalculator(),
        horton_calculator=_HortonCalculator(),
    )

    emle_bespoke = _EMLEBespoke(ref_sampler)
    emle_bespoke.train_model(n_samples=args.n_samples, n_steps=args.n_steps)


if __name__ == "__main__":
    main()
