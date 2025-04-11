import os as _os

import torch as _torch
from loguru import logger as _logger

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
    accumulation_steps=1,
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
    accumulation_steps: int, optional
        Number of steps to accumulate gradients over.

    Returns
    -------
    model
        Trained model (the loss instance).
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
        accumulation_steps=1,
        *args,
        **kwargs,
    ):
        """Perform the training loop with early stopping."""
        loss_history = []
        best_loss = float("inf")
        best_model_state = None

        if save_checkpoints and output_dir:
            _os.makedirs(output_dir, exist_ok=True)

        if hasattr(loss_instance, "precompute_weights"):
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

            for batch_idx, batch in enumerate(dataloader):
                loss, rmse, max_error = loss_instance(**batch)
                loss = loss / accumulation_steps  
                loss.backward(retain_graph=True)
                loss_total += (
                    loss.item() * accumulation_steps
                )  
                rmse_total += rmse.item()
                max_error_total.append(max_error.item())

                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            if (batch_idx + 1) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = loss_total / len(dataloader)
            avg_rmse = rmse_total / len(dataloader)
            max_error_epoch = max(max_error_total)

            loss_history.append(avg_loss)

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = {
                    "epoch": epoch,
                    "model_state_dict": {
                        k: v.detach().cpu()
                        for k, v in loss_instance.state_dict().items()
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "rmse": avg_rmse,
                    "max_error": max_error_epoch,
                }

                if save_checkpoints and output_dir:
                    checkpoint_path = _os.path.join(output_dir, "best_model.pt")
                    _torch.save(best_model_state, checkpoint_path)
                    _logger.debug(
                        f"Saved checkpoint at epoch {epoch + 1} to {checkpoint_path}"
                    )

            # Check for early stopping condition
            if len(loss_history) > n_iterations_stop:
                recent_loss_changes = [
                    abs(loss_history[-i] - loss_history[-(i + 1)])
                    for i in range(1, n_iterations_stop + 1)
                ]
                if all(change < loss_threshold for change in recent_loss_changes):
                    _logger.info(
                        f"Early stopping triggered at epoch {epoch + 1}. No improvement in the last {n_iterations_stop} epochs."
                    )
                    break

            if (epoch + 1) % print_every == 0 or epoch == 0:
                _logger.info(
                    f"Epoch {epoch + 1: >4}: Loss ={avg_loss:9.4f}    "
                    f"RMSE ={avg_rmse:9.4f}    "
                    f"Max Error ={max_error_epoch:9.4f}"
                )

        # Restore best model state
        if best_model_state is not None:
            _logger.info(
                f"Restoring best model from epoch {best_model_state['epoch'] + 1} with loss {best_model_state['loss']:.4f}"
            )
            loss_instance.load_state_dict(best_model_state["model_state_dict"])
        else:
            _logger.warning("No best model state found to restore. Using final state.")

        return best_loss  

    loss_instance = loss_class(**(loss_class_kwargs or {}))

    device = next(loss_instance.parameters()).device
    _logger.info(f"Training will proceed on device: {device}")
    loss_instance.to(device)

    # Filter parameters to optimize
    opt_parameters = []
    all_param_names = [name for name, _ in loss_instance.named_parameters()]
    _logger.debug(f"All available parameters: {all_param_names}")

    for name, param in loss_instance.named_parameters():
        is_optimizable = False
        for opt_name in opt_param_names:
            # Direct match (e.g., 'a_QEq') or specific LJ/EMLE parameter name, or part of the name
            # TODO: improve this
            name_parts = name.split(".", 1)[-1]
            if name == opt_name or name.endswith(f"{opt_name}") or opt_name in name_parts:
                is_optimizable = True
                break

        if is_optimizable:
            if param.requires_grad:
                opt_parameters.append(param)
                _logger.debug(f"Including parameter for optimization: {name}")
            else:
                _logger.warning(
                    f"Parameter {name} specified for optimization but requires_grad=False. Skipping."
                )
        else:
            param.requires_grad_(False)  
            _logger.debug(f"Freezing parameter: {name}")

    if not opt_parameters:
        _logger.error(
            "No parameters found for optimization based on the provided names. Stopping training."
        )
        return loss_instance  

    _logger.info(f"Optimizing parameters: {opt_param_names}")

    optimizer = _torch.optim.Adam(opt_parameters, lr=lr)

    _train_loop(
        loss_instance,
        optimizer,
        epochs,
        dataloader,
        print_every,
        n_iterations_stop,
        loss_threshold,
        output_dir,
        save_checkpoints,
        accumulation_steps,
        *args,
        **kwargs,  
    )


    loss_instance.eval()

    return loss_instance  
