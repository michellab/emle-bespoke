"""Main script for fitting LJ parameters."""

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
from ..calculators import ORCACalculator as _ORCACalculator
from ..lj_fitting import LennardJonesPotential as _LennardJonesPotential
from ..samplers._dimers import DimerSampler as _DimerSampler
from ..utils import (
    add_emle_force as _add_emle_force,
    create_dimer_topology as _create_dimer_topology,
    create_mixed_system as _create_mixed_system,
    create_simulation as _create_simulation,
    remove_constraints as _remove_constraints,
    write_system_to_xml as _write_system_to_xml,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference data and train a bespoke EMLE model."
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
        "--ml_model",
        type=str,
        default=None,
        help="The machine learning model to use for the solute.",
    )

    parser.add_argument(
        "--emle_model",
        type=str,
        default="default",
        help="The EMLE model to use for the solute.",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug.")

    # Parse the arguments
    args = parser.parse_args()

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        _logger.error(f"Error: {e}")
    except SystemExit as e:
        _logger.error(f"Unrecognized argument(s) detected: {e}")
        exit(1)

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

    if args.debug:
        import torch as _torch

        _torch.autograd.set_detect_anomaly(True)

    # Create topology
    topology_off = _create_dimer_topology(
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
        pressure=None,
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

    """
    # Add the EMLE force
    system, context = _add_emle_force(
        args.emle_model, qm_region, system, context, topology
    )

    # Write the system to an XML file
    _write_system_to_xml(system, "initial_dimer.xml")
    """
    # Create the reference dimer sampler
    ref_sampler = _DimerSampler(
        system=system,
        context=context,
        integrator=integrator,
        topology=topology,
        qm_region=qm_region,
        qm_calculator=_ORCACalculator(),
    )

    # Create the Lennard-Jones potential
    lj_potential = _LennardJonesPotential(
        topology_off=topology_off,
        forcefield=force_field,
        parameters_to_fit={
            #"n7": ["sigma", "epsilon"],
            #"n14": ["sigma", "epsilon"],
            #"n16": ["sigma", "epsilon"],
            #"n19": ["sigma", "epsilon"],
            #"n3": ["sigma", "epsilon"],
            # "n12": ["sigma", "epsilon"],
            "n-tip3p-O": ["sigma", "epsilon"],
            #"n-tip3p-H": ["sigma", "epsilon"],
        },
    )

    # Fit the Lennard-Jones potential
    emle_bespoke = _BespokeModelTrainer(
        ref_sampler, filename_prefix=args.filename_prefix
    )
    """
    emle_bespoke.sample_dimer_curves()
    """
    ref_sampler.reference_data.read("/home/joaomorado/test/a/water-benzene-sapt.pkl")
    # ref_sampler.reference_data.read("/home/joaomorado/test/a/water-methanol.pkl")
    # ref_sampler.reference_data.read("/home/joaomorado/test/a/solvator/data_solvator.pkl")

    ni = 0
    nf = 16
    emle_bespoke.fit_lj(
        lj_potential=lj_potential,
        xyz_qm=ref_sampler.reference_data["xyz_qm"][ni:nf],
        xyz_mm=ref_sampler.reference_data["xyz_mm"][ni:nf],
        atomic_numbers=ref_sampler.reference_data["z"][ni:nf],
        charges_mm=ref_sampler.reference_data["charges_mm"][ni:nf],
        e_int_target=ref_sampler.reference_data["sapt_all"][ni:nf],
        solute_mask=ref_sampler.reference_data["solute_mask"][ni:nf],
        solvent_mask=ref_sampler.reference_data["solvent_mask"][ni:nf],
    )

    msg = r"""
╔════════════════════════════════════════════════════════════╗
║             emle-bespoke-lj terminated normally!           ║
╚════════════════════════════════════════════════════════════╝"""
    for line in msg.split("\n"):
        _logger.info(line)


if __name__ == "__main__":
    main()
