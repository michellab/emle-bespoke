import torch as _torch
from loguru import logger as _logger


def train_model(
    loss_class,
    opt_param_names,
    lr,
    epochs,
    emle_model,
    print_every=10,
    loss_class_kwargs=None,
    *args,
    **kwargs,
):
    """
    Train a model.

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
    print_every: int
        How often to print training progress
    loss_class_kwargs: dict
            Keyword arguments to pass to the loss class besides the EMLE model.

    Returns
    -------
    model
        Trained model.
    """

    def _train_loop(loss_instance, optimizer, epochs, print_every=10, *args, **kwargs):
        """
        Perform the training loop.

        Parameters
        ----------
        loss_instance: nn.Module
            Loss instance.
        optimizer: torch.optim.Optimizer
            Optimizer.
        epochs: int
            Number of training epochs.
        print_every: int
            How often to print training progress
        args: list
            Positional arguments to pass to the forward method.
        kwargs: dict
            Keyword arguments to pass to the forward method.

        Returns
        -------
        loss
            Forward loss.
        """
        for epoch in range(epochs):
            loss_instance.train()
            optimizer.zero_grad()
            loss, rmse, max_error = loss_instance(*args, **kwargs)
            loss.backward(retain_graph=True)
            optimizer.step()
            if (epoch + 1) % print_every == 0:
                _logger.info(
                    f"Epoch {epoch+1}: Loss ={loss.item():9.4f}    "
                    f"RMSE ={rmse.item():9.4f}    "
                    f"Max Error ={max_error.item():9.4f}"
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
    _train_loop(model, optimizer, epochs, print_every, *args, **kwargs)
    return model
