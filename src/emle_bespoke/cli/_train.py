"""Main script for generating reference data and training a bespoke EMLE model."""

# General imports
import argparse
import os
import pickle

# OpenMM imports
from loguru import logger as _logger

from .._log import log_banner as _log_banner


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference data and train a bespoke EMLE model."
    )

    # Command-line arguments
    parser.add_argument(
        "--reference-data",
        type=str,
        required=True,
        help="Path to the reference data file.",
    )

    parser.add_argument(
        "--filename-prefix",
        type=str,
        default="ligand",
        help="Prefix for the output files.",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=0.001,
        help="The kernel uncertainty parameter.",
    )

    parser.add_argument(
        "--ivm-thr",
        type=float,
        default=0.05,
        help="The IVM threshold parameter.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="The number of epochs to train the model.",
    )

    parser.add_argument(
        "--lr-qeq",
        type=float,
        default=0.05,
        help="The learning rate for the QEq model.",
    )

    parser.add_argument(
        "--lr-thole",
        type=float,
        default=0.05,
        help="The learning rate for the Thole model.",
    )

    parser.add_argument(
        "--lr-sqrtk",
        type=float,
        default=0.05,
        help="The learning rate for the sqrtk model.",
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print the loss every N epochs.",
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

    # Imports from the emle-bespoke package
    from ..bespoke import BespokeModelTrainer as _BespokeModelTrainer

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

    if not os.path.exists(args.reference_data):
        _logger.error(f"Reference data file not found: {args.reference_data}")
        raise FileNotFoundError(f"Reference data file not found: {args.reference_data}")
    else:
        _logger.info(f"Reference data file found: {args.reference_data}")
        with open(args.reference_data, "rb") as f:
            reference_data = pickle.load(f)

    # Train the bespoke EMLE model
    emle_bespoke = _BespokeModelTrainer(
        reference_data=args.reference_data, filename_prefix=args.filename_prefix
    )
    emle_bespoke.train_model(
        z=reference_data["z"],
        xyz=reference_data["xyz_qm"],
        s=reference_data["s"],
        q_core=reference_data["q_core"],
        q_val=reference_data["q_val"],
        alpha=reference_data["alpha"],
        train_mask=None,
        model_filename=args.filename_prefix + "_bespoke.mat",
        plot_data_filename=args.filename_prefix + "_bespoke_plot_data.mat",
        sigma=args.sigma,
        ivm_thr=args.ivm_thr,
        epochs=args.epochs,
        lr_qeq=args.lr_qeq,
        lr_thole=args.lr_thole,
        lr_sqrtk=args.lr_sqrtk,
        print_every=args.print_every,
    )

    emle_bespoke.get_mbis_static_predictions(
        charges_mm=reference_data["charges_mm"],
        xyz_qm=reference_data["xyz_qm"],
        xyz_mm=reference_data["xyz_mm"],
        q_core=reference_data["q_core"],
        q_val=reference_data["q_val"],
        s=reference_data["s"],
    )

    msg = r"""
╔════════════════════════════════════════════════════════════╗
║             emle-bespoke-train terminated normally!        ║
╚════════════════════════════════════════════════════════════╝"""
    for line in msg.split("\n"):
        _logger.info(line)


if __name__ == "__main__":
    main()
