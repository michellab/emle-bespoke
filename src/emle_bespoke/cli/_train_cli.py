"""Main script for training a bespoke EMLE model."""

import argparse
import sys

from loguru import logger as _logger

from .._log import log_banner as _log_banner
from .._log import log_cli_args as _log_cli_args
from .._log import log_termination as _log_termination


def main() -> None:
    """Main function for training and patching a bespoke EMLE model."""
    _log_banner()

    parser = argparse.ArgumentParser(
        description="Generate reference data and train a bespoke EMLE model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--reference-data",
        type=str,
        required=True,
        help="Path to the reference data file.",
    )

    parser.add_argument(
        "--alpha-mode",
        type=str,
        default="species",
        choices=["species", "reference"],
        help="The mode for the polarizabilities.",
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

    parser.add_argument(
        "--train-mask",
        type=str,
        default=None,
        help="The mask to use for training.",
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training the model. Useful if only patching is desired.",
    )

    parser.add_argument(
        "--patch",
        action="store_true",
        help="Patch the model after training.",
    )

    parser.add_argument(
        "--lr-patch",
        type=float,
        default=0.001,
        help="The learning rate for patching the model.",
    )

    parser.add_argument(
        "--epochs-patch",
        type=int,
        default=1000,
        help="The number of epochs to patch the model.",
    )

    parser.add_argument(
        "--emle-model-filename",
        type=str,
        default=None,
        help="The filename of the EMLE model to patch.",
    )

    parser.add_argument(
        "--skip-e-static",
        action="store_true",
        help="Skip fitting the static energy (fitted by default).",
    )

    parser.add_argument(
        "--skip-e-ind",
        action="store_true",
        help="Skip fitting the induced energy (fitted by default).",
    )

    parser.add_argument(
        "--fit-e-total",
        action="store_true",
        help="Fit the total energy.",
    )

    parser.add_argument(
        "--e-static-param",
        type=lambda x: x.split(","),
        default="a_QEq,ref_values_chi",
        help="The parameter to fit for the static energy. Provide as comma-separated values.",
    )

    parser.add_argument(
        "--e-ind-param",
        type=lambda x: x.split(","),
        default="a_Thole,k_Z,sqrtk_ref",
        help="The parameters to fit for the induced energy, provided as comma-separated values.",
    )

    parser.add_argument(
        "--l2-reg-alpha",
        type=float,
        default=1.0,
        help="The L2 regularization parameter for alpha. Only used if e_ind is fitted during patching.",
    )

    parser.add_argument(
        "--l2-reg-s",
        type=float,
        default=1.0,
        help="The L2 regularization parameter for s. Only used if e_static is fitted during patching.",
    )

    parser.add_argument(
        "--l2-reg-q",
        type=float,
        default=1.0,
        help="The L2 regularization parameter for q. Only used if e_static is fitted during patching.",
    )

    parser.add_argument(
        "--n-batches",
        type=int,
        default=32,
        help="The number of batches to use for calculating the EMLE predictions when patching the model.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="The device to use for training.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="The data type to use for training.",
    )

    # Parse the arguments
    try:
        args = parser.parse_args()
    except SystemExit as e:
        if e.code == 0:  # Help was requested
            sys.exit(0)
        _logger.error("Unrecognized argument(s) detected.")
        sys.exit(1)
    except Exception as e:
        _logger.error(f"Error parsing arguments: {e}")
        sys.exit(1)

    _log_cli_args(args)

    # Only import required modules after argument parsing
    import os as _os
    import pickle as _pkl

    import torch as _torch
    from emle.models import EMLEBase as _EMLEBase
    from emle.train._loss import QEqLoss as _QEqLoss
    from emle.train._loss import TholeLoss as _TholeLoss

    from ..train import EMLEPatched as _EMLEPatched
    from ..train import EMLETrainer as _EMLETrainer
    from ..train import PatchingLoss as _PatchingLoss
    from ..utils import (
        convert_reference_data_to_tensors as _convert_reference_data_to_tensors,
    )

    # Load reference data
    if not _os.path.exists(args.reference_data):
        _logger.error(f"Reference data file not found: {args.reference_data}")
        raise FileNotFoundError(f"Reference data file not found: {args.reference_data}")

    try:
        with open(args.reference_data, "rb") as f:
            reference_data = _pkl.load(f)
    except Exception as e:
        _logger.error(f"Error loading reference data: {e}")
        raise

    # Set up device and dtype
    try:
        device = _torch.device(args.device)
        dtype = _torch.float64 if args.dtype == "float64" else _torch.float32
    except Exception as e:
        _logger.error(f"Error setting up device/dtype: {e}")
        raise

    # Convert reference data to tensors
    reference_data = _convert_reference_data_to_tensors(reference_data, device, dtype)

    if args.emle_model_filename:
        args.skip_training = True
        _logger.info(
            f"Loading EMLE model from {args.emle_model_filename} and patching... "
            "Default training will be skipped."
        )

        if not _os.path.exists(args.emle_model_filename):
            _logger.error(f"EMLE model file not found: {args.emle_model_filename}")
            raise FileNotFoundError(
                f"EMLE model file not found: {args.emle_model_filename}"
            )

        model_filename = args.emle_model_filename

    # Initialize trainer
    trainer = _EMLETrainer(
        emle_base=_EMLEBase,  # EMLE base reference for emle-engine
        qeq_loss=_QEqLoss,  # QEq loss reference for emle-engine
        thole_loss=_TholeLoss,  # Thole loss reference for emle-engine
        patch_loss=_PatchingLoss,  # Patching loss reference for emle-bespoke
    )

    # Train the model
    if not args.skip_training:
        msg = r"""
╔════════════════════════════════════════════════════════════╗
║         Starting training of bespoke EMLE model...         ║
╚════════════════════════════════════════════════════════════╝
"""
        for line in msg.split("\n"):
            _logger.info(line)

        model_filename = args.filename_prefix + "_bespoke.mat"

        trainer.train(
            z=reference_data["z"],
            xyz=reference_data["xyz_qm"],
            s=reference_data["s"],
            q_core=reference_data["q_core"],
            q_val=reference_data["q_val"],
            alpha=reference_data["alpha"],
            train_mask=None,
            model_filename=model_filename,
            plot_data_filename=args.filename_prefix + "_bespoke_plot_data.mat",
            sigma=args.sigma,
            ivm_thr=args.ivm_thr,
            epochs=args.epochs,
            lr_qeq=args.lr_qeq,
            lr_thole=args.lr_thole,
            lr_sqrtk=args.lr_sqrtk,
            print_every=args.print_every,
            device=device,
            dtype=dtype,
        )

    # Patch the model if requested
    if args.patch:
        msg = r"""
╔════════════════════════════════════════════════════════════╗
║         Starting patching of bespoke EMLE model...         ║
╚════════════════════════════════════════════════════════════╝
"""
        for line in msg.split("\n"):
            _logger.info(line)

        if args.e_ind_param:
            param_to_remove = "k_Z" if args.alpha_mode == "reference" else "sqrtk_ref"
            try:
                args.e_ind_param.remove(param_to_remove)
            except ValueError:
                pass

        if args.fit_e_total:
            opt_param_names = args.e_static_param + args.e_ind_param

            trainer.patch(
                opt_param_names=opt_param_names,
                emle_model_filename=model_filename,
                alpha_mode=args.alpha_mode,
                lr=args.lr_patch,
                epochs=args.epochs_patch,
                e_static_target=reference_data.get("e_static", None),
                e_ind_target=reference_data.get("e_ind", None),
                atomic_numbers=reference_data.get("z", None),
                charges_mm=reference_data.get("charges_mm", None),
                xyz_qm=reference_data.get("xyz_qm", None),
                xyz_mm=reference_data.get("xyz_mm", None),
                q_core=reference_data.get("q_core", None),
                q_val=reference_data.get("q_val", None),
                s=reference_data.get("s", None),
                alpha=reference_data.get("alpha", None),
                l2_reg_alpha=args.l2_reg_alpha,
                l2_reg_s=args.l2_reg_s,
                l2_reg_q=args.l2_reg_q,
                n_batches=args.n_batches,
                filename_prefix=args.filename_prefix
                + "_patched"
                + f"_{args.alpha_mode}",
                device=device,
                dtype=dtype,
            )
        else:
            if not args.skip_e_static:
                opt_param_names = args.e_static_param

                trainer.patch(
                    opt_param_names=opt_param_names,
                    emle_model_filename=model_filename,
                    alpha_mode=args.alpha_mode,
                    lr=args.lr_patch,
                    epochs=args.epochs_patch,
                    e_static_target=reference_data.get("e_static", None),
                    e_ind_target=None,
                    atomic_numbers=reference_data.get("z", None),
                    charges_mm=reference_data.get("charges_mm", None),
                    xyz_qm=reference_data.get("xyz_qm", None),
                    xyz_mm=reference_data.get("xyz_mm", None),
                    q_core=reference_data.get("q_core", None),
                    q_val=reference_data.get("q_val", None),
                    s=reference_data.get("s", None),
                    alpha=None,
                    l2_reg_alpha=0.0,
                    l2_reg_s=args.l2_reg_s,
                    l2_reg_q=args.l2_reg_q,
                    n_batches=args.n_batches,
                    filename_prefix=args.filename_prefix
                    + "_patched"
                    + f"_{args.alpha_mode}",
                    device=device,
                    dtype=dtype,
                )

            if not args.skip_e_ind:
                opt_param_names = args.e_ind_param

                trainer.patch(
                    opt_param_names=opt_param_names,
                    emle_model_filename=model_filename,
                    alpha_mode=args.alpha_mode,
                    lr=args.lr_patch,
                    epochs=args.epochs_patch,
                    e_static_target=None,
                    e_ind_target=reference_data.get("e_ind", None),
                    atomic_numbers=reference_data.get("z", None),
                    charges_mm=reference_data.get("charges_mm", None),
                    xyz_qm=reference_data.get("xyz_qm", None),
                    xyz_mm=reference_data.get("xyz_mm", None),
                    q_core=None,
                    q_val=None,
                    s=None,
                    alpha=reference_data.get("alpha", None),
                    l2_reg_alpha=args.l2_reg_alpha,
                    l2_reg_s=0.0,
                    l2_reg_q=0.0,
                    n_batches=args.n_batches,
                    filename_prefix=args.filename_prefix
                    + "_patched"
                    + f"_{args.alpha_mode}",
                    device=device,
                    dtype=dtype,
                )

    _log_termination()


if __name__ == "__main__":
    main()
