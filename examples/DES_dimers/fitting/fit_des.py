import argparse as _argparse
import pickle as _pkl

import numpy as _np
import torch as _torch
from emle.models import EMLE as _EMLE
from loguru import logger as _logger
from openff.toolkit import (
    ForceField as _ForceField,
)
from torch.utils.data import DataLoader
from tqdm import tqdm as _tqdm

from emle_bespoke._log import _logger
from emle_bespoke._log import log_banner as _log_banner
from emle_bespoke._log import log_cli_args as _log_cli_args
from emle_bespoke.lj import LennardJonesPotential as _LJPotential
from emle_bespoke.lj._loss import InteractionEnergyLoss as _InteractionEnergyLoss
from emle_bespoke.reference_data import ReferenceDataset as _ReferenceDataset


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
    *args,
    **kwargs,
):
    """
    Train a model with an early stopping criterion.

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
        *args,
        **kwargs,
    ):
        """
        Perform the training loop with early stopping.
        """
        loss_history = []

        for epoch in range(epochs):
            loss_instance.train()
            loss_total = 0
            rmse_total = 0
            max_error_total = []

            optimizer.zero_grad()

            for batch in dataloader:
                print("batch keys: ", batch.keys())
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

        return loss

    # Initialize model and optimizer
    model = loss_class(**loss_class_kwargs)
    opt_parameters = [
        param
        for name, param in model.named_parameters()
        if any(opt_param in name.split(".", 1)[1] for opt_param in opt_param_names)
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
        *args,
        **kwargs,
    )
    return model


def main():
    _log_banner()
    parser = _argparse.ArgumentParser(description="Fit DES dimers")
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to the data file"
    )
    parser.add_argument(
        "--force-field",
        type=str,
        default="openff-2.0.0.offxml",
        help="OpenFF force field to use",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for training"
    )
    parser.add_argument(
        "--emle-model", type=str, default=None, help="EMLE model to use"
    )
    parser.add_argument(
        "--alpha-mode", type=str, default="species", help="Alpha mode to use in EMLE"
    )
    parser.add_argument(
        "--batch-size", type=int, default=5000, help="Batch size for training"
    )
    # parser.add_argument("--output_path", type=str, required=True, help="Path to save the fitted model")
    args = parser.parse_args()

    _log_cli_args(args)

    # Load processed data
    with open(args.data_path, "rb") as f:
        reference_data = _pkl.load(f)

    force_field = _ForceField(args.force_field)

    if args.device == "cuda" and _torch.cuda.is_available():
        device = _torch.device("cuda")
    else:
        device = _torch.device("cpu")

    # Initialize LJ potential
    lj_potential = _LJPotential(
        topology_off=reference_data["topology"],
        forcefield=force_field,
        parameters_to_fit={"all": {"epsilon": True, "sigma": True, "alpha": True}},
        device=device,
    )

    # Fit LJ potential
    del reference_data["topology"]
    reference_data = _ReferenceDataset(reference_data)
    # reference_data.write("reference_data.h5")

    # Initialize EMLE model
    emle = _EMLE(
        device=device,
        dtype=_torch.float64,
        model=args.emle_model,
        alpha_mode=args.alpha_mode,
    )

    # Ensure reference data is on the correct device
    reference_data.to_tensors()  # Convert to tensors if not already
    reference_data.rename_key("e_int", "e_int_target")

    # Create DataLoader for batch processing
    dataloader = DataLoader(
        reference_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 workers for now to debug
        pin_memory=device.type == "cpu",  # Only pin memory for CPU tensors  `
    )

    # Evaluate initial EMLE and LJ energies
    results = {"e_static": [], "e_ind": [], "e_lj": [], "total_energy": []}
    print("Evaluating initial EMLE and LJ energies...")
    progress_bar = _tqdm(dataloader, desc="Processing batches")
    for batch_idx, batch in enumerate(progress_bar):
        atomic_numbers = batch["atomic_numbers"] * batch["solute_mask"]
        atomic_numbers = atomic_numbers[:, : batch["xyz_qm"].shape[1]]
        charges_mm = batch["charges_mm"]
        xyz_qm = batch["xyz_qm"]
        xyz_mm = batch["xyz_mm"]

        with _torch.no_grad():
            # EMLE forward pass
            e_static, e_ind = emle(
                atomic_numbers=atomic_numbers,
                charges_mm=charges_mm,
                xyz_qm=xyz_qm,
                xyz_mm=xyz_mm,
            )

            # LJ forward pass
            e_lj = lj_potential(
                xyz=batch["xyz"] * 0.1,
                solute_mask=batch["solute_mask"],
                solvent_mask=batch["solvent_mask"],
                start_idx=batch["indices"][0],
                end_idx=batch["indices"][-1] + 1,
            )

        progress_bar.set_description(
            f"Processing batch {batch_idx + 1}/{len(dataloader)}"
        )

        results["e_static"].append(e_static.cpu().numpy())
        results["e_ind"].append(e_ind.cpu().numpy())
        results["e_lj"].append(e_lj.cpu().numpy())
        results["total_energy"].append((e_static + e_ind + e_lj).cpu().numpy())

    # Convert results to numpy arrays
    for key in results:
        results[key] = _np.concatenate(results[key])

    # Convert EMLE energies to tensors
    e_static_emle = _torch.tensor(
        results["e_static"], device=device, dtype=_torch.float64
    )
    e_ind_emle = _torch.tensor(results["e_ind"], device=device, dtype=_torch.float64)

    # Define loss class kwargs
    loss_class_kwargs = {
        "lj_potential": lj_potential,
        "weighting_method": "uniform",
        "e_static_emle": e_static_emle,
        "e_ind_emle": e_ind_emle,
        "emle_model": emle,
    }

    # Fit LJ potential
    train_model(
        loss_class=_InteractionEnergyLoss,
        opt_param_names=["sigma", "epsilon"],
        lr=0.001,
        epochs=100,
        dataloader=dataloader,
        print_every=10,
        loss_class_kwargs=loss_class_kwargs,
    )


if __name__ == "__main__":
    main()
