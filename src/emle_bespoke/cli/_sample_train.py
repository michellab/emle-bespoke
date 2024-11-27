"""Main script for generating reference data and training a bespoke EMLE model."""

# General imports
import argparse

# OpenMM imports
import openmm.unit as _unit
from loguru import logger as _logger
from openff.interchange import Interchange as _Interchange

# OpenFF imports
from openff.toolkit import ForceField as _ForceField

from .._log import log_banner as _log_banner

# Imports from the emle-bespoke package
from ..bespoke import BespokeModelTrainer as _BespokeModelTrainer
from ..calculators import HortonCalculator as _HortonCalculator
from ..calculators import ORCACalculator as _ORCACalculator
from ..samplers._md import MDSampler as _MDSampler
from ..utils import create_mixed_system as _create_mixed_system
from ..utils import create_simulation as _create_simulation
from ..utils import create_simulation_box_topology as _create_simulation_box_topology
from ..utils import remove_constraints as _remove_constraints


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference data and train a bespoke EMLE model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Command-line arguments
    parser.add_argument(
        "--n_solute", type=int, default=1, help="Number of ligands in the system."
    )
    parser.add_argument(
        "--n_solvent",
        type=int,
        default=1000,
        help="Number of water molecules in the system.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=100, help="Number of samples to generate."
    )
    parser.add_argument(
        "--n_steps", type=int, default=1000, help="Number of simulation steps to run."
    )

    parser.add_argument(
        "--filename-prefix",
        type=str,
        default="ligand",
        help="Prefix for the output files.",
    )

    parser.add_argument(
        "--n_equilibration",
        type=int,
        default=1000,
        help="Number of equilibration steps to run.",
    )

    parser.add_argument(
        "--solute", type=str, default="c1ccccc1", help="The ligand SMILES string."
    )
    parser.add_argument(
        "--solvent",
        type=str,
        default="[H:2][O:1][H:3]",
        help="The solvent SMILES string.",
    )
    parser.add_argument(
        "--forcefields",
        type=str,
        default="openff_unconstrained-2.0.0.offxml,tip3p.offxml",
        help="The force field(s) to use, separated by commas.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=298.15,
        help="Simulation temperature in Kelvin.",
    )
    parser.add_argument(
        "--pressure", type=float, default=1.0, help="Simulation pressure in bar."
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=1.0,
        help="Simulation timestep in femtoseconds.",
    )
    parser.add_argument(
        "--friction_coefficient",
        type=float,
        default=1.0,
        help="Langevin friction coefficient (ps^-1).",
    )
    parser.add_argument(
        "--ml_model",
        type=str,
        default=None,
        help="The machine learning model to use for the solute.",
    )

    # Parse the arguments
    args = parser.parse_args()

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        _logger.error(f"Error: {e}")
    except SystemExit as e:
        _logger.error(f"Unrecognized argument(s) detected: {e}")
        exit(1)

    # Validation of critical arguments
    if args.n_solvent < 0 or args.n_solute < 0:
        _logger.error("n_solvent and n_solute must be non-negative.")
        return

    _log_banner()

    # Print all arguments
    msg = r"""
╔════════════════════════════════════════════════════════════╗
║                      Input CLI Arguments                   ║
╚════════════════════════════════════════════════════════════╝
"""
    for line in msg.split("\n"):
        _logger.info(line)
    for arg in vars(args):
        _logger.info(f"{arg}: {getattr(args, arg)}")
    _logger.info("══════════════════════════════════════════════════════════════\n")

    # Create topology
    topology_off = _create_simulation_box_topology(
        n_solute=args.n_solute,
        n_solvent=args.n_solvent,
        solvent_smiles=args.solvent,
        solute_smiles=args.solute,
    )

    # Initialize force field and interchange object
    force_field = _ForceField(*args.forcefields.split(","))
    interchange = _Interchange.from_smirnoff(
        force_field=force_field, topology=topology_off
    )

    # Create simulation object
    simulation = _create_simulation(
        interchange=interchange,
        temperature=args.temperature,
        pressure=args.pressure,
        timestep=args.timestep,
        friction_coefficient=args.friction_coefficient,
    )

    # Define QM region and train model
    topology = topology_off.to_openmm()
    qm_region = [atom.index for atom in list(topology.chains())[0].atoms()]

    # Remove constraints involving alchemical atoms
    _remove_constraints(simulation.system, qm_region)
    simulation.context.reinitialize(preserveState=True)

    # Create mixed system
    system, context, integrator = _create_mixed_system(
        args.ml_model, qm_region, simulation
    )

    if args.n_equilibration:
        _logger.info(f"Running {args.n_equilibration} equilibration steps.")
        context.setVelocitiesToTemperature(args.temperature * _unit.kelvin)
        integrator.step(args.n_equilibration)

    ref_sampler = _MDSampler(
        system=system,
        context=context,
        integrator=integrator,
        topology=topology,
        qm_region=qm_region,
        qm_calculator=_ORCACalculator(),
        horton_calculator=_HortonCalculator(),
    )

    emle_bespoke = _BespokeModelTrainer(
        ref_sampler, filename_prefix=args.filename_prefix
    )
    emle_bespoke.sample_and_train_model(n_samples=args.n_samples, n_steps=args.n_steps)

    msg = r"""
╔════════════════════════════════════════════════════════════╗
║              emle-bespoke terminated normally!             ║
╚════════════════════════════════════════════════════════════╝"""
    for line in msg.split("\n"):
        _logger.info(line)


if __name__ == "__main__":
    main()
