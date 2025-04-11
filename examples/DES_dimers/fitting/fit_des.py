import argparse as _argparse
import os as _os
import sys

import torch as _torch

from emle_bespoke._log import _logger
from emle_bespoke._log import log_banner as _log_banner
from emle_bespoke._log import log_cli_args as _log_cli_args

sys.path.append("/home/joaomorado/repos/emle-bespoke/examples/DES_dimers/fitting")


def _evaluate_energies(emle_model, lj_model, dataloader, device):
    """Evaluate all energy components for the given dataset."""
    import numpy as _np
    from tqdm import tqdm as _tqdm

    from emle_bespoke._constants import HARTREE_TO_KJ_MOL as _HARTREE_TO_KJ_MOL

    energies = {"e_static": [], "e_ind": [], "e_lj": []}
    _logger.info("Evaluating energy components...")
    progress_bar = _tqdm(dataloader, desc="Evaluating energies")

    emle_model.eval()
    lj_model.eval()

    for batch in progress_bar:
        batch_data = {
            k: v.to(device)
            for k, v in batch.items()
            if hasattr(v, "to") and hasattr(v, "device") and v.device != device
        }
        batch_data.update(
            {
                k: v
                for k, v in batch.items()
                if not (
                    hasattr(v, "to") and hasattr(v, "device") and v.device != device
                )
            }
        )

        with _torch.no_grad():
            # EMLE forward pass
            e_static, e_ind = emle_model(
                atomic_numbers=batch_data["atomic_numbers"],
                charges_mm=batch_data["charges_mm"],
                xyz_qm=batch_data["xyz_qm"],
                xyz_mm=batch_data["xyz_mm"],
            )
            # LJ forward pass
            e_lj = lj_model(
                xyz=batch_data["xyz"],
                solute_mask=batch_data["solute_mask"],
                solvent_mask=batch_data["solvent_mask"],
                indices=batch_data["indices"],
            )

        energies["e_static"].append(
            e_static.detach().cpu().numpy() * _HARTREE_TO_KJ_MOL
        )
        energies["e_ind"].append(e_ind.detach().cpu().numpy() * _HARTREE_TO_KJ_MOL)
        energies["e_lj"].append(e_lj.detach().cpu().numpy())

    for key in energies:
        if energies[key]:
            energies[key] = _np.concatenate(energies[key])
        else:
            energies[key] = _np.array([])
    _logger.info("Finished evaluating energy components.")
    return energies


def _save_final_parameters(trained_loss_instance, model_type, output_dir):
    """Save final parameters from the trained loss instance."""

    if model_type == "lj" and hasattr(trained_loss_instance, "lj_potential"):
        lj_model_to_save = trained_loss_instance.lj_potential
        _logger.info(f"Saving final LJ parameters to {output_dir}")
        lj_model_to_save.print_lj_parameters()
        lj_model_to_save.write_lj_parameters(
            _os.path.join(output_dir, "final_lj_parameters.txt")
        )
    elif model_type == "emle" and hasattr(trained_loss_instance, "emle_model"):
        emle_model_to_save = trained_loss_instance.emle_model
        _logger.info(f"Saving final EMLE parameters to {output_dir}")
        emle_state_dict = emle_model_to_save.state_dict()
        # Ensure state dict tensors are on CPU before saving
        emle_state_dict_cpu = {k: v.cpu() for k, v in emle_state_dict.items()}
        _torch.save(
            emle_state_dict_cpu,
            _os.path.join(output_dir, "final_emle_parameters.pt"),
        )
        _logger.info(
            f"Saved EMLE parameters to {_os.path.join(output_dir, 'final_emle_parameters.pt')}"
        )
    else:
        _logger.warning(
            f"Could not save parameters for model type '{model_type}'. Loss instance might be incorrect or lack expected attributes ('lj_potential' or 'emle_model')."
        )


def _run_lj_optimization(
    lj_params_to_fit,
    lj_model,
    emle_model,
    fixed_emle_energies,
    args,
    dataloader,
    device,
):
    """Runs an optimization step focusing on LJ parameters."""

    from _training_utils import train_model

    from emle_bespoke.loss import InteractionEnergyLoss as _InteractionEnergyLoss

    if not lj_params_to_fit:
        _logger.info("Skipping LJ optimization step as no parameters were specified.")
        return lj_model

    _logger.info(f"Starting LJ optimization ({lj_params_to_fit})")
    loss_class_kwargs_lj = {
        "lj_potential": lj_model,
        "emle_model": emle_model,
        "weighting_method": args.weighting_method,
        "l2_reg": args.l2_reg,
        "weights_fudge": 1.0,
        "e_static_emle": _torch.tensor(
            fixed_emle_energies["e_static"], device=device, dtype=_torch.float64
        ),
        "e_ind_emle": _torch.tensor(
            fixed_emle_energies["e_ind"], device=device, dtype=_torch.float64
        ),
    }
    trained_loss_instance = train_model(
        loss_class=_InteractionEnergyLoss,
        opt_param_names=lj_params_to_fit,
        lr=args.lr,
        epochs=args.epochs,
        dataloader=dataloader,
        print_every=args.print_every,
        loss_class_kwargs=loss_class_kwargs_lj,
        n_iterations_stop=args.patience,
        loss_threshold=args.loss_threshold,
        output_dir=args.output_dir,
        save_checkpoints=args.save_checkpoints,
        accumulation_steps=args.accumulation_steps,
    )
    if trained_loss_instance and hasattr(trained_loss_instance, "lj_potential"):
        lj_model = trained_loss_instance.lj_potential
        _save_final_parameters(trained_loss_instance, "lj", args.output_dir)
        _logger.info("LJ optimization step completed.")
    else:
        _logger.warning(
            "LJ optimization step did not return a valid model. LJ parameters may not be updated."
        )

    return lj_model


def _run_emle_optimization(
    emle_params_to_fit,
    lj_model,
    emle_model,
    fixed_lj_energies,
    args,
    dataloader,
    device,
):
    """Runs an optimization step focusing on EMLE parameters."""

    from _training_utils import train_model

    from emle_bespoke.loss import InteractionEnergyLoss as _InteractionEnergyLoss

    if not emle_params_to_fit:
        _logger.info("Skipping EMLE optimization step as no parameters were specified.")
        return emle_model

    _logger.info(f"Starting EMLE optimization ({emle_params_to_fit})")
    loss_class_kwargs_emle = {
        "lj_potential": lj_model,
        "emle_model": emle_model,
        "weighting_method": args.weighting_method,
        "l2_reg": args.l2_reg,
        "weights_fudge": 1.0,
        "e_lj": _torch.tensor(
            fixed_lj_energies["e_lj"], device=device, dtype=_torch.float64
        ),
    }
    trained_loss_instance = train_model(
        loss_class=_InteractionEnergyLoss,
        opt_param_names=emle_params_to_fit,
        lr=args.lr,
        epochs=args.epochs,
        dataloader=dataloader,
        print_every=args.print_every,
        loss_class_kwargs=loss_class_kwargs_emle,
        n_iterations_stop=args.patience,
        loss_threshold=args.loss_threshold,
        output_dir=args.output_dir,
        save_checkpoints=args.save_checkpoints,
        accumulation_steps=args.accumulation_steps,
    )
    if trained_loss_instance and hasattr(trained_loss_instance, "emle_model"):
        emle_model = trained_loss_instance.emle_model
        _save_final_parameters(trained_loss_instance, "emle", args.output_dir)
        _logger.info("EMLE optimization step completed.")
    else:
        _logger.warning(
            "EMLE optimization step did not return a valid model. EMLE parameters may not be updated."
        )

    return emle_model


def _run_simultaneous_optimization(
    lj_params_to_fit, emle_params_to_fit, lj_model, emle_model, args, dataloader
):
    """Runs a simultaneous optimization step for both LJ and EMLE parameters."""

    from _training_utils import train_model

    from emle_bespoke.loss import InteractionEnergyLoss as _InteractionEnergyLoss

    opt_params_all = list(set(lj_params_to_fit + emle_params_to_fit))
    if not opt_params_all:
        _logger.warning(
            "Simultaneous optimization selected, but no parameters specified."
        )
        return lj_model, emle_model

    _logger.info(f"Running simultaneous optimization ({opt_params_all})")
    loss_class_kwargs = {
        "lj_potential": lj_model,
        "emle_model": emle_model,
        "weighting_method": args.weighting_method,
        "l2_reg": args.l2_reg,
        "weights_fudge": 1.0,
    }
    trained_loss_instance = train_model(
        loss_class=_InteractionEnergyLoss,
        opt_param_names=opt_params_all,
        lr=args.lr,
        epochs=args.epochs,
        dataloader=dataloader,
        print_every=args.print_every,
        loss_class_kwargs=loss_class_kwargs,
        n_iterations_stop=args.patience,
        loss_threshold=args.loss_threshold,
        output_dir=args.output_dir,
        save_checkpoints=args.save_checkpoints,
        accumulation_steps=args.accumulation_steps,
    )
    if trained_loss_instance:
        updated_lj = False
        updated_emle = False
        if hasattr(trained_loss_instance, "lj_potential"):
            lj_model = trained_loss_instance.lj_potential
            updated_lj = True
        if hasattr(trained_loss_instance, "emle_model"):
            emle_model = trained_loss_instance.emle_model
            updated_emle = True

        if lj_params_to_fit and updated_lj:
            _save_final_parameters(trained_loss_instance, "lj", args.output_dir)
        elif lj_params_to_fit and not updated_lj:
            _logger.warning(
                "LJ parameters were intended to be optimized simultaneously, but the model instance was not updated."
            )

        if emle_params_to_fit and updated_emle:
            _save_final_parameters(trained_loss_instance, "emle", args.output_dir)
        elif emle_params_to_fit and not updated_emle:
            _logger.warning(
                "EMLE parameters were intended to be optimized simultaneously, but the model instance was not updated."
            )
        _logger.info("Simultaneous optimization step completed.")
    else:
        _logger.warning(
            "Simultaneous optimization step did not return a valid model. Parameters may not be updated."
        )

    return lj_model, emle_model


def main():
    """Main function to run the parameter fitting process."""

    _log_banner()

    parser = _argparse.ArgumentParser(
        description="Fit EMLE and/or LJ parameters for DES dimers.",
        formatter_class=_argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data and model configuration
    data_group = parser.add_argument_group("Data and Model Configuration")
    data_group.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the reference dataset file (.pkl)",
    )
    data_group.add_argument(
        "--force-field",
        type=str,
        default="openff-2.0.0.offxml",
        help="Base OpenFF force field to use for LJ parameters",
    )
    data_group.add_argument(
        "--emle-model",
        type=str,
        default=None,
        help="Path to pre-trained EMLE model file (optional)",
    )
    data_group.add_argument(
        "--alpha-mode",
        type=str,
        default="species",
        choices=["species", "atom"],
        help="Alpha mode used in the EMLE model",
    )

    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training (cuda or cpu)",
    )
    train_group.add_argument(
        "--n-batches",
        type=int,
        default=32,
        help="Number of batches to divide the dataset into for training",
    )
    train_group.add_argument(
        "--accumulation-steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    train_group.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Optimizer learning rate",
    )
    train_group.add_argument(
        "--epochs", type=int, default=100, help="Maximum number of training epochs"
    )
    train_group.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Frequency of printing training progress (epochs)",
    )

    # Early stopping configuration
    early_group = parser.add_argument_group("Early Stopping Configuration")
    early_group.add_argument(
        "--patience",
        type=int,
        default=1000,
        help="Number of epochs to wait for improvement before early stopping",
    )
    early_group.add_argument(
        "--loss-threshold",
        type=float,
        default=1e-5,
        help="Minimum change in loss to qualify as improvement for early stopping",
    )

    # Loss function configuration
    loss_group = parser.add_argument_group("Loss Function Configuration")
    loss_group.add_argument(
        "--weighting-method",
        type=str,
        default="uniform",
        choices=["uniform", "boltzmann", "non-boltzmann", "openff"],
        help="Method for weighting samples in the loss function",
    )
    loss_group.add_argument(
        "--l2-reg",
        type=float,
        default=1.0,
        help="L2 regularization strength on parameters",
    )

    # Parameter fitting configuration
    param_group = parser.add_argument_group("Parameter Fitting Configuration")
    param_group.add_argument(
        "--lj-params",
        type=str,
        default="epsilon,sigma",
        help="Comma-separated list of LJ parameters to optimize. Default is 'epsilon,sigma'.",
    )
    param_group.add_argument(
        "--emle-params",
        type=str,
        default="a_QEq,ref_values_chi",
        help="Comma-separated list of EMLE parameters to optimize. Default is 'a_QEq,ref_values_chi'.",
    )
    param_group.add_argument(
        "--opt-strategy",
        type=str,
        default="lj-only",
        choices=[
            "simultaneous",
            "lj-then-emle",
            "emle-then-lj",
            "lj-only",
            "emle-only",
        ],
        help="Strategy for parameter optimization. Default is 'lj-only'.",
    )

    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="output",  # Changed default back
        help="Directory to save output files (optimized parameters, checkpoints)",
    )
    output_group.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save model checkpoints (best model state) during training",
    )

    args = parser.parse_args()
    _log_cli_args(args)

    import pickle as _pkl

    import torch as _torch
    from emle.models import EMLE as _EMLE
    from openff.toolkit import ForceField as _ForceField
    from torch.utils.data import DataLoader

    from emle_bespoke.lj import LennardJonesPotential as _LJPotential
    from emle_bespoke.reference_data import ReferenceDataset as _ReferenceDataset

    _os.makedirs(args.output_dir, exist_ok=True)
    try:
        # Load processed data
        _logger.info(f"Loading reference dataset from {args.data_path}")
        with open(args.data_path, "rb") as f:
            reference_data_dict = _pkl.load(f)

        _logger.info(f"Loading base force field from {args.force_field}")
        force_field = _ForceField(args.force_field)

        if args.device == "cuda" and _torch.cuda.is_available():
            device = _torch.device("cuda")
            _logger.info("Using CUDA device.")
        else:
            device = _torch.device("cpu")
            if args.device == "cuda":
                _logger.warning("CUDA not available, falling back to CPU.")
            else:
                _logger.info("Using CPU device.")

        lj_params_to_fit = []
        if args.lj_params:
            lj_params_to_fit = [param.strip() for param in args.lj_params.split(",")]

        emle_params_to_fit = []
        if args.emle_params:
            emle_params_to_fit = [
                param.strip() for param in args.emle_params.split(",")
            ]

        # Create parameters_to_fit dictionary for LJ potential
        parameters_to_fit_lj = {
            "all": {
                "epsilon": "epsilon" in lj_params_to_fit,
                "sigma": "sigma" in lj_params_to_fit,
            }
        }

        lj_strategies = ["simultaneous", "lj-then-emle", "emle-then-lj", "lj-only"]
        if args.opt_strategy in lj_strategies and not any(
            parameters_to_fit_lj["all"].values()
        ):
            _logger.warning(
                f"Strategy '{args.opt_strategy}' includes LJ optimization, but no LJ parameters were specified via --lj-params."
            )

        emle_strategies = ["simultaneous", "lj-then-emle", "emle-then-lj", "emle-only"]
        if args.opt_strategy in emle_strategies and not emle_params_to_fit:
            _logger.warning(
                f"Strategy '{args.opt_strategy}' includes EMLE optimization, but no EMLE parameters were specified via --emle-params."
            )

        # Initialize LJ model
        _logger.info("Initializing LJ potential model.")
        lj_model = _LJPotential(
            topology_off=reference_data_dict["topology"],
            forcefield=force_field,
            parameters_to_fit=parameters_to_fit_lj,
            device=device,
        )

        # Prepare dataset
        # TODO: remove this
        topology = reference_data_dict["topology"]
        del reference_data_dict["topology"]
        reference_dataset = _ReferenceDataset(
            reference_data_dict, device=device
        )  # Renamed dataset -> reference_dataset
        _logger.info(f"Dataset contains {len(reference_dataset)} configurations.")

        # Initialize EMLE model
        _logger.info("Initializing EMLE model.")
        emle_model = _EMLE(
            device=device,
            dtype=_torch.float64,
            model=args.emle_model,
            alpha_mode=args.alpha_mode,
        )

        # TODO: remove this
        # Prepare data (move to device, convert to tensors)
        if "zzz" in reference_dataset._data:  # Clean up potential leftover keys
            del reference_dataset._data["zzz"]
        reference_dataset.to_tensors()

        # Create DataLoader
        batch_size = max(1, len(reference_dataset) // args.n_batches + 1)
        dataloader = DataLoader(
            reference_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cpu",
        )
        _logger.info(
            f"Using batch size: {batch_size}, resulting in {len(dataloader)} batches."
        )

        # Evaluate initial EMLE and LJ energies
        initial_energies = _evaluate_energies(emle_model, lj_model, dataloader, device)

        # Execute the optimization strategy
        _logger.info(f"Executing optimization strategy: {args.opt_strategy}")

        fixed_emle_energies = {
            "e_static": initial_energies["e_static"],
            "e_ind": initial_energies["e_ind"],
        }
        fixed_lj_energies = {"e_lj": initial_energies["e_lj"]}

        # --- Optimization Strategy Logic (Refactored) ---
        if args.opt_strategy == "simultaneous":
            lj_model, emle_model = _run_simultaneous_optimization(
                lj_params_to_fit,
                emle_params_to_fit,
                lj_model,
                emle_model,
                args,
                dataloader,
                device,
            )

        elif args.opt_strategy == "lj-then-emle":
            # 1. Optimize LJ
            lj_model = _run_lj_optimization(
                lj_params_to_fit,
                lj_model,
                emle_model,
                fixed_emle_energies,
                args,
                dataloader,
                device,
            )
            # Re-evaluate energies if LJ model was updated
            updated_energies = _evaluate_energies(
                emle_model, lj_model, dataloader, device
            )
            fixed_lj_energies = {"e_lj": updated_energies["e_lj"]}

            # 2. Optimize EMLE
            emle_model = _run_emle_optimization(
                emle_params_to_fit,
                lj_model,
                emle_model,
                fixed_lj_energies,
                args,
                dataloader,
                device,
            )

        elif args.opt_strategy == "emle-then-lj":
            # 1. Optimize EMLE
            emle_model = _run_emle_optimization(
                emle_params_to_fit,
                lj_model,
                emle_model,
                fixed_lj_energies,
                args,
                dataloader,
                device,
            )
            # Re-evaluate energies if EMLE model was updated
            updated_energies = _evaluate_energies(
                emle_model, lj_model, dataloader, device
            )
            fixed_emle_energies = {
                "e_static": updated_energies["e_static"],
                "e_ind": updated_energies["e_ind"],
            }

            # 2. Optimize LJ
            lj_model = _run_lj_optimization(
                lj_params_to_fit,
                lj_model,
                emle_model,
                fixed_emle_energies,
                args,
                dataloader,
                device,
            )

        elif args.opt_strategy == "lj-only":
            lj_model = _run_lj_optimization(
                lj_params_to_fit,
                lj_model,
                emle_model,
                fixed_emle_energies,
                args,
                dataloader,
                device,
            )

        elif args.opt_strategy == "emle-only":
            emle_model = _run_emle_optimization(
                emle_params_to_fit,
                lj_model,
                emle_model,
                fixed_lj_energies,
                args,
                dataloader,
                device,
            )

        _logger.info(f"Fitting process completed. Results saved in {args.output_dir}")

    except Exception as e:
        _logger.exception(f"An error occurred during the fitting process: {str(e)}")
        raise


if __name__ == "__main__":
    main()
