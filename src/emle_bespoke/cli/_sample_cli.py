"""Main script for sampling configurations to train a bespoke EMLE model."""

# General imports
import argparse

from loguru import logger as _logger

# Imports from the emle-bespoke package
from .._log import log_banner as _log_banner
from .._log import log_cli_args as _log_cli_args
from .._log import log_termination as _log_termination


def main():
    """Main function for sampling configurations to train a bespoke EMLE model."""
    _log_banner()

    parser = argparse.ArgumentParser(
        description="Generate reference data and train a bespoke EMLE model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--n-solute", type=int, default=1, help="Number of ligands in the system."
    )
    parser.add_argument(
        "--n-solvent",
        type=int,
        default=1000,
        help="Number of water molecules in the system.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=100, help="Number of samples to generate."
    )
    parser.add_argument(
        "--n-steps", type=int, default=1000, help="Number of simulation steps to run."
    )

    parser.add_argument(
        "--filename-prefix",
        type=str,
        default="ligand",
        help="Prefix for the output files.",
    )

    parser.add_argument(
        "--n-equilibration",
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
        "--friction-coefficient",
        type=float,
        default=1.0,
        help="Langevin friction coefficient (ps^-1).",
    )

    parser.add_argument(
        "--cutoff",
        type=float,
        default=12.0,
        help="Cutoff distance in Angstroms. For EMLE, this defines the group-based cutoff between QM and MM regions. For OpenMM, this sets the non-bonded interaction cutoff distance.",
    )

    parser.add_argument(
        "--ml-model",
        type=str,
        default=None,
        help="The machine learning model to use for the solute.",
    )

    parser.add_argument(
        "--emle-model",
        type=str,
        default=None,
        help="The EMLE model to use for the solute.",
    )

    parser.add_argument(
        "--alpha-mode",
        type=str,
        default="species",
        help="Alpha mode of the EMLE model.",
    )

    parser.add_argument(
        "--qm-calculator",
        type=str,
        default="orca",
        help="The QM calculator to use. Options are 'orca'.",
    )

    parser.add_argument(
        "--mbis-calculator",
        type=str,
        default="horton",
        help="The MBIS calculator to use. Options are 'horton'.",
    )

    parser.add_argument(
        "--sampler",
        type=str,
        default="md",
        help="The sampler to use. Options are 'md'. Must be an OpenMM-compatible sampler.",
    )

    parser.add_argument(
        "--skip-static",
        action="store_true",
        help="Skip the static energy calculation (calculated by default).",
    )

    parser.add_argument(
        "--skip-induction",
        action="store_true",
        help="Skip the induction energy calculation (calculated by default).",
    )

    parser.add_argument(
        "--skip-mbis",
        action="store_true",
        help="Skip the MBIS partitioning calculation (calculated by default).",
    )

    parser.add_argument(
        "--skip-polarizability",
        action="store_true",
        help="Skip the polarizability calculation (calculated by default).",
    )

    # Parse the arguments
    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        _logger.error(f"Error: {e}")
        return
    except SystemExit as e:
        if e.code == 0:
            return
        _logger.error(f"Error parsing arguments: {e}")
        exit(1)

    # Validation of critical arguments
    if args.n_solvent < 0 or args.n_solute < 0:
        _logger.error("n_solvent and n_solute must be non-negative.")
        return

    _log_cli_args(args)

    # OpenMM imports
    import openmm.unit as _unit
    from openff.interchange import Interchange as _Interchange

    # OpenFF imports
    from openff.toolkit import ForceField as _ForceField

    # Imports from the emle-bespoke package
    from ..calculators import ReferenceDataCalculator as _ReferenceDataCalculator
    from ..reference_data import ReferenceData as _ReferenceData
    from ..samplers._md import MDSampler as _MDSampler
    from ..utils import add_emle_force as _add_emle_force
    from ..utils import create_mixed_system as _create_mixed_system
    from ..utils import create_simulation as _create_simulation
    from ..utils import (
        create_simulation_box_topology as _create_simulation_box_topology,
    )
    from ..utils import remove_constraints as _remove_constraints

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

    # ------------------------------------------------------------------------- #
    #                              Simulation                                   #
    # ------------------------------------------------------------------------- #
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

    # ------------------------------------------------------------------------- #
    #                               EMLE force                                  #
    # ------------------------------------------------------------------------- #
    system, context = _add_emle_force(
        args.emle_model,
        qm_region,
        system,
        context,
        topology,
        alpha_mode=args.alpha_mode,
    )

    # ------------------------------------------------------------------------- #
    #                               Equilibration                               #
    # ------------------------------------------------------------------------- #
    if args.n_equilibration:
        _logger.info(f"Running {args.n_equilibration} equilibration steps.")
        context.setVelocitiesToTemperature(args.temperature * _unit.kelvin)
        integrator.step(args.n_equilibration)

    # ------------------------------------------------------------------------- #
    #                                  Sampler                                  #
    # ------------------------------------------------------------------------- #
    if args.sampler == "md":
        ref_sampler = _MDSampler(
            system=system,
            context=context,
            integrator=integrator,
            topology=topology,
            qm_region=qm_region,
            cutoff=args.cutoff,
        )
    else:
        raise ValueError(
            f"Sampler {args.sampler} not supported. Check the --sampler argument."
        )

    # ------------------------------------------------------------------------- #
    #                              Calculators                                  #
    # ------------------------------------------------------------------------- #
    if args.qm_calculator == "orca":
        from ..calculators import ORCACalculator as _ORCACalculator

        qm_calculator = _ORCACalculator()
    else:
        raise ValueError(
            f"QM calculator {args.qm_calculator} not supported. Check the --qm-calculator argument."
        )

    if args.mbis_calculator == "horton":
        from ..calculators import HortonCalculator as _HortonCalculator

        mbis_calculator = _HortonCalculator()
    else:
        raise ValueError(
            f"MBIS calculator {args.mbis_calculator} not supported. Check the --mbis-calculator argument."
        )

    ref_data_calculator = _ReferenceDataCalculator(
        qm_calculator=qm_calculator,
        mbis_calculator=mbis_calculator,
    )

    ref_data = _ReferenceData()

    # ------------------------------------------------------------------------- #
    #                Sample configurations and calculate reference data         #
    # ------------------------------------------------------------------------- #
    for i in range(args.n_samples):
        _logger.info(f"Sampling configuration {i + 1} / {args.n_samples}.")
        sampler_output = ref_sampler.sample(n_steps=args.n_steps)
        ref_data_output = ref_data_calculator.get_reference_data(
            directory_vacuum=args.filename_prefix + f"_vacuum_{i}",
            directory_pc=args.filename_prefix + f"_pc_{i}",
            calc_static=not args.skip_static,
            calc_induction=not args.skip_induction,
            calc_polarizability=not args.skip_polarizability,
            calc_mbis=not args.skip_mbis,
            **sampler_output,
        )
        ref_data.add(ref_data_output)

    # Save the reference data
    ref_data.write(args.filename_prefix)

    _log_termination()


if __name__ == "__main__":
    main()
