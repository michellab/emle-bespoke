"""Weighted Mean Squared Error loss."""

import torch as _torch


class WeightedMSELoss(_torch.nn.Module):
    """
    Weighted Mean Squared Error loss.

    This loss function computes a weighted MSE where each sample can have a different
    weight. The weights can be used to focus the training on specific regions of the
    energy landscape or to balance different types of interactions.
    """

    def __init__(self) -> None:
        """Initialize the weighted MSE loss."""
        super().__init__()

    def forward(
        self, inputs: _torch.Tensor, targets: _torch.Tensor, weights: _torch.Tensor
    ) -> _torch.Tensor:
        """
        Compute the weighted MSE loss.

        Parameters
        ----------
        inputs : _torch.Tensor
            Predicted values.
        targets : _torch.Tensor
            Target values.
        weights : _torch.Tensor
            Weights for each sample.

        Returns
        -------
        _torch.Tensor
            The weighted MSE loss.

        Raises
        ------
        ValueError
            If the shapes of inputs, targets, and weights do not match.
        """
        if not (inputs.shape == targets.shape == weights.shape):
            raise ValueError(
                "Inputs, targets, and weights must have the same shape. "
                f"Got shapes: inputs={inputs.shape}, targets={targets.shape}, "
                f"weights={weights.shape}"
            )

        diff = targets - inputs
        squared_error = diff**2
        weighted_squared_error = squared_error * weights

        return weighted_squared_error.sum()
