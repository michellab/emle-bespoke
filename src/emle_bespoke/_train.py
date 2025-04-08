import torch as _torch
from loguru import logger as _logger


def train_model(
    loss_class,
    opt_param_names,
    lr,
    epochs,
    emle_model,
    dataloader,
    print_every=10,
    loss_class_kwargs=None,
    n_iterations_stop=1000,  # New parameter for early stopping (number of iterations)
    loss_threshold=1e-5,  # Threshold for early stopping
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
    emle: EMLE
        EMLE model instance.
    dataloader: DataLoader
        DataLoader for input data.
    print_every: int
        How often to print training progress
    loss_class_kwargs: dict
        Keyword arguments to pass to the loss class besides the EMLE model.
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

        Parameters
        ----------
        loss_instance: nn.Module
            Loss instance.
        optimizer: torch.optim.Optimizer
            Optimizer.
        epochs: int
            Number of training epochs.
        print_every: int
            How often to print training progress.
        dataloader: DataLoader
            The DataLoader for input data.
        n_iterations_stop: int
            Number of iterations to check for early stopping.
        loss_threshold: float
            Threshold for the change in loss to trigger early stopping.
        args: list
            Positional arguments to pass to the forward method.
        kwargs: dict
            Keyword arguments to pass to the forward method.

        Returns
        -------
        loss
            Forward loss.
        """
        loss_instance.l2_reg_calc = True

        all_e_int_targets = _torch.cat(
            [batch["e_int_target"] for batch in dataloader], dim=0
        )
        loss_instance.calulate_weights(all_e_int_targets, None, "boltzmann")
        loss_instance._weights.to("cuda")

        # Pass the whole dataset by forming just one batch
        loss_total = 0
        for batch in dataloader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            loss, _, _ = loss_instance(**batch)
            loss_total += loss.item()

        # loss_instance._loss._normalization = 1.0 / loss_total
        # Calculate the loss for the whole dataset
        all_e_int_targets = all_e_int_targets[all_e_int_targets < 5 * 4.184]
        loss_instance._loss._normalization = 1.0  # / all_e_int_targets.std() ** 2

        _logger.info(f"Initial Loss: {loss_total}")

        # Early stopping variables
        loss_history = []

        # Preload all batches into memory
        all_batches = [batch for batch in dataloader]

        for epoch in range(epochs):
            loss_instance.train()

            loss_total = 0
            rmse_total = 0
            max_error_total = []
            loss_instance.l2_reg_calc = True
            optimizer.zero_grad()

            for batch in all_batches:  # Iterate over preloaded batches
                # Optionally move batch data to GPU beforehand if needed
                # batch = {k: v.to("cuda") for k, v in batch.items()}

                loss, rmse, max_error = loss_instance(**batch)  # Calculate loss
                loss.backward(retain_graph=False)

                loss_total += loss.item()
                rmse_total += rmse.item()
                max_error_total.append(max_error.item())

                loss_instance.l2_reg_calc = False

            print(f"Epoch {epoch + 1}: Loss = {loss_total}, RMSE = {rmse_total}")

            optimizer.step()

            loss = loss_total
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

    # Additonal kwargs for the loss class
    loss_class_kwargs = loss_class_kwargs or {}

    model = loss_class(emle_model, **loss_class_kwargs)
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
