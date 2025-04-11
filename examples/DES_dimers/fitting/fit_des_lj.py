import argparse as _argparse
import os as _os
import pickle as _pkl
from pathlib import Path as _Path

import numpy as _np
import torch as _torch
from emle.models import EMLE as _EMLE
from loguru import logger as _logger
from openff.toolkit import (
    ForceField as _ForceField,
)
from torch.utils.data import DataLoader
from tqdm import tqdm as _tqdm

from emle_bespoke._constants import HARTREE_TO_KJ_MOL as _HARTREE_TO_KJ_MOL
from emle_bespoke._log import _logger
from emle_bespoke._log import log_banner as _log_banner
from emle_bespoke._log import log_cli_args as _log_cli_args
from emle_bespoke.lj import LennardJonesPotential as _LJPotential

# from emle_bespoke.lj import LennardJonesPotentialEfficient as _LJPotentialEfficient
from emle_bespoke.loss import InteractionEnergyLoss as _InteractionEnergyLoss
from emle_bespoke.reference_data import ReferenceDataset as _ReferenceDataset


def save_results(results: dict, output_dir: str, prefix: str = "") -> None:
    """Save training results to files.

    Parameters
    ----------
    results : dict
        Dictionary containing results to save
    output_dir : str
        Directory to save results in
    prefix : str, optional
        Prefix for output filenames, by default ""
    """
    _os.makedirs(output_dir, exist_ok=True)

    # Save energy plots
    import matplotlib.pyplot as _plt

    _plt.figure(figsize=(10, 6))
    mask = results["e_int_target"] < 0
    _plt.plot(results["e_int_target"][mask], label="Target")
    _plt.plot(results["e_int"][mask], label="Total")
    _plt.plot(results["e_static"][mask], label="Static")
    _plt.plot(results["e_ind"][mask], label="Induced")
    _plt.plot(results["e_lj"][mask], label="LJ")
    _plt.xlabel("Sample Index")
    _plt.ylabel("Energy (kJ/mol)")
    _plt.legend()
    _plt.title("Energy Components")
    _plt.tight_layout()
    _plt.savefig(_os.path.join(output_dir, f"{prefix}energies.png"))
    _plt.close()

    # Save numerical results
    _np.savez(
        _os.path.join(output_dir, f"{prefix}results.npz"),
        **{k: _np.array(v) for k, v in results.items()},
    )


def train_model(
    loss_class,
    opt_param_names,
    lr,
    epochs,
    dataloader,
    print_every=10,
    loss_class_kwargs=None,
    n_iterations_stop=1000,
    loss_threshold=1e-5,
    output_dir=None,
    save_checkpoints=False,
    *args,
    **kwargs,
):
    """Train a model with an early stopping criterion.

    Parameters
    ----------
    loss_class: class
        Loss class.
    opt_param_names: list of str
        List of parameter names to optimize.
    lr: float
        Learning rate.
    epochs: int
        Number of training epochs.
    dataloader: DataLoader
        DataLoader for input data.
    print_every: int
        How often to print training progress
    loss_class_kwargs: dict
        Keyword arguments to pass to the loss class.
    n_iterations_stop: int
        Number of iterations to check for early stopping.
    loss_threshold: float
        Threshold for the change in loss to trigger early stopping.
    output_dir: str, optional
        Directory to save checkpoints and results
    save_checkpoints: bool, optional
        Whether to save model checkpoints

    Returns
    -------
    model
        Trained model.
    """

    def _train_loop(
        loss_instance,
        optimizer,
        epochs,
        dataloader,
        print_every=10,
        n_iterations_stop=10,
        loss_threshold=1e-5,
        output_dir=None,
        save_checkpoints=False,
        *args,
        **kwargs,
    ):
        """Perform the training loop with early stopping."""
        loss_history = []
        best_loss = float("inf")
        best_model_state = None

        # Precompute weights
        e_int_target = _torch.cat([batch["e_int_target"] for batch in dataloader])
        loss_instance.precompute_weights(
            e_int_target=e_int_target,
            e_int_predicted=None,
        )

        for epoch in range(epochs):
            loss_instance.train()
            loss_total = 0
            rmse_total = 0
            max_error_total = []

            optimizer.zero_grad()

            for batch in dataloader:
                loss, rmse, max_error = loss_instance(**batch)
                loss.backward()
                loss_total += loss.item()
                rmse_total += rmse.item()
                max_error_total.append(max_error.item())

            optimizer.step()

            loss = loss_total / len(dataloader)
            rmse = rmse_total / len(dataloader)
            max_error = max(max_error_total)

            loss_history.append(loss)

            # Save best model
            if loss < best_loss:
                best_loss = loss
                best_model_state = {
                    "epoch": epoch,
                    "model_state_dict": loss_instance.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "rmse": rmse,
                    "max_error": max_error,
                }

                if save_checkpoints and output_dir:
                    _torch.save(
                        best_model_state, _os.path.join(output_dir, "best_model.pt")
                    )

            # Check for early stopping condition
            if len(loss_history) > n_iterations_stop:
                recent_loss_changes = [
                    abs(loss_history[-i] - loss_history[-(i + 1)])
                    for i in range(1, n_iterations_stop)
                ]
                if all(change < loss_threshold for change in recent_loss_changes):
                    _logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                    break

            if (epoch + 1) % print_every == 0 or epoch <= 1:
                _logger.info(
                    f"Epoch {epoch + 1}: Loss ={loss:9.4f}    "
                    f"RMSE ={rmse:9.4f}    "
                    f"Max Error ={max_error:9.4f}"
                )

        # Restore best model
        if best_model_state is not None:
            loss_instance.load_state_dict(best_model_state["model_state_dict"])

        return loss

    # Initialize model and optimizer
    model = loss_class(**loss_class_kwargs)

    opt_parameters = [
        param
        for name, param in model.named_parameters()
        if any(opt_param in name.split(".", 1)[1] for opt_param in opt_param_names)
        or any(opt_param in name for opt_param in opt_param_names)
    ]
    _logger.info(f"Optimizing parameters: {opt_param_names}")

    optimizer = _torch.optim.Adam(opt_parameters, lr=lr)
    _train_loop(
        model,
        optimizer,
        epochs,
        dataloader,
        print_every,
        n_iterations_stop,
        loss_threshold,
        output_dir,
        save_checkpoints,
        *args,
        **kwargs,
    )
    return model


def main():
    _log_banner()
    parser = _argparse.ArgumentParser(
        description="Fit DES dimers with enhanced control and output options",
        formatter_class=_argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data and model configuration
    data_group = parser.add_argument_group("Data and Model Configuration")
    data_group.add_argument(
        "--data-path", type=str, required=True, help="Path to the data file"
    )
    data_group.add_argument(
        "--force-field",
        type=str,
        default="openff-2.0.0.offxml",
        help="OpenFF force field to use",
    )
    data_group.add_argument(
        "--emle-model", type=str, default=None, help="EMLE model to use"
    )
    data_group.add_argument(
        "--alpha-mode",
        type=str,
        default="species",
        choices=["species", "atom"],
        help="Alpha mode to use in EMLE",
    )

    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training",
    )
    train_group.add_argument(
        "--batch-size", type=int, default=5000, help="Batch size for training"
    )
    train_group.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for optimization",
    )
    train_group.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    train_group.add_argument(
        "--print-every", type=int, default=10, help="Print progress every N epochs"
    )

    # Early stopping configuration
    early_group = parser.add_argument_group("Early Stopping Configuration")
    early_group.add_argument(
        "--patience",
        type=int,
        default=1000,
        help="Number of iterations to wait for improvement before early stopping",
    )
    early_group.add_argument(
        "--loss-threshold",
        type=float,
        default=1e-5,
        help="Minimum change in loss to qualify as an improvement",
    )

    # Loss function configuration
    loss_group = parser.add_argument_group("Loss Function Configuration")
    loss_group.add_argument(
        "--weighting-method",
        type=str,
        default="uniform",
        choices=["uniform", "boltzmann", "non-boltzmann", "openff"],
        help="Method for weighting the loss function",
    )
    loss_group.add_argument(
        "--l2-reg", type=float, default=1.0, help="L2 regularization strength"
    )

    # Parameter fitting configuration
    param_group = parser.add_argument_group("Parameter Fitting Configuration")
    param_group.add_argument(
        "--fit-epsilon", action="store_true", help="Fit epsilon parameters"
    )
    param_group.add_argument(
        "--fit-sigma", action="store_true", help="Fit sigma parameters"
    )

    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save output files",
    )
    output_group.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save model checkpoints during training",
    )

    args = parser.parse_args()
    _log_cli_args(args)

    # Create output directory
    _os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Load processed data
        with open(args.data_path, "rb") as f:
            reference_data = _pkl.load(f)

        force_field = _ForceField(args.force_field)

        if args.device == "cuda" and _torch.cuda.is_available():
            device = _torch.device("cuda")
        else:
            device = _torch.device("cpu")
            if args.device == "cuda":
                _logger.warning("CUDA not available, falling back to CPU")

        # Determine parameters to fit
        parameters_to_fit = {
            "all": {
                "epsilon": args.fit_epsilon,
                "sigma": args.fit_sigma,
            }
        }

        if not any(parameters_to_fit["all"].values()):
            _logger.warning("No parameters selected for fitting. Using defaults.")
            parameters_to_fit["all"] = {"epsilon": True, "sigma": True}

        # Initialize LJ potential
        lj_potential = _LJPotential(
            topology_off=reference_data["topology"],
            forcefield=force_field,
            parameters_to_fit=parameters_to_fit,
            device=device,
        )

        # Fit LJ potential
        topology = reference_data["topology"]
        del reference_data["topology"]
        reference_data = _ReferenceDataset(reference_data)

        # Initialize EMLE model
        emle = _EMLE(
            device=device,
            dtype=_torch.float64,
            model=args.emle_model,
            alpha_mode=args.alpha_mode,
        )

        # Ensure EMLE model is in evaluation mode
        emle.eval()

        # Ensure reference data is on the correct device
        if "zzz" in reference_data._data:
            del reference_data._data["zzz"]
        reference_data.to_tensors()

        # Create DataLoader for batch processing
        dataloader = DataLoader(
            reference_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cpu",
        )

        # Evaluate initial EMLE and LJ energies
        results = {"e_static": [], "e_ind": [], "e_lj": [], "e_int_target": []}
        _logger.info("Evaluating initial EMLE and LJ energies...")
        progress_bar = _tqdm(dataloader, desc="Processing batches")

        for batch_idx, batch in enumerate(progress_bar):
            with _torch.no_grad():
                # EMLE forward pass
                e_static, e_ind = emle(
                    atomic_numbers=batch["atomic_numbers"],
                    charges_mm=batch["charges_mm"],
                    xyz_qm=batch["xyz_qm"],
                    xyz_mm=batch["xyz_mm"],
                )
                # LJ forward pass
                e_lj = lj_potential(
                    xyz=batch["xyz"],
                    solute_mask=batch["solute_mask"],
                    solvent_mask=batch["solvent_mask"],
                    indices=batch["indices"],
                )

            progress_bar.set_description(
                f"Processing batch {batch_idx + 1}/{len(dataloader)}"
            )

            results["e_static"].append(
                e_static.detach().cpu().numpy() * _HARTREE_TO_KJ_MOL
            )
            results["e_ind"].append(e_ind.detach().cpu().numpy() * _HARTREE_TO_KJ_MOL)
            results["e_lj"].append(e_lj.detach().cpu().numpy())
            results["e_int_target"].append(batch["e_int_target"].detach().cpu().numpy())

        # Convert results to numpy arrays
        for key in results:
            results[key] = _np.concatenate(results[key])

        results["e_int"] = results["e_static"] + results["e_ind"] + results["e_lj"]

        # Save initial results
        # save_results(results, args.output_dir, prefix="initial_")

        # Convert EMLE energies to tensors
        e_static_emle = _torch.tensor(
            results["e_static"], device=device, dtype=_torch.float64
        )
        e_ind_emle = _torch.tensor(
            results["e_ind"], device=device, dtype=_torch.float64
        )

        # For memory management, delete EMLE model and empty cache
        del emle
        _torch.cuda.empty_cache()
        emle = _EMLE(
            device=device,
            dtype=_torch.float64,
            model=args.emle_model,
            alpha_mode=args.alpha_mode,
        )

        from collections import Counter

        weights_fudge = Counter(topology)
        weights_fudge = _np.array([1.0 / weights_fudge[top] for top in topology])
        weights_fudge = weights_fudge / weights_fudge.sum()

        # Define loss class kwargs
        loss_class_kwargs = {
            "lj_potential": lj_potential,
            "weighting_method": args.weighting_method,
            "e_static_emle": e_static_emle,
            "e_ind_emle": e_ind_emle,
            "emle_model": emle,
            "l2_reg": args.l2_reg,
            "weights_fudge": 1.0,  # weights_fudge,
        }

        # Determine parameters to optimize
        opt_param_names = []
        if args.fit_epsilon:
            opt_param_names.append("epsilon")
        if args.fit_sigma:
            opt_param_names.append("sigma")

        # Fit LJ potential
        trained_model = train_model(
            loss_class=_InteractionEnergyLoss,
            opt_param_names=opt_param_names,
            lr=args.learning_rate,
            epochs=args.epochs,
            dataloader=dataloader,
            print_every=args.print_every,
            loss_class_kwargs=loss_class_kwargs,
            n_iterations_stop=args.patience,
            loss_threshold=args.loss_threshold,
            output_dir=args.output_dir,
            save_checkpoints=args.save_checkpoints,
        )

        # Save final parameters
        lj_potential.print_lj_parameters()
        lj_potential.write_lj_parameters(
            _os.path.join(args.output_dir, "final_parameters.txt")
        )

        _logger.info(f"Training completed. Results saved in {args.output_dir}")

    except Exception as e:
        _logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
