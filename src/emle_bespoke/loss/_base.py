"""Extend emle-engine base loss class to add more features."""

from typing import Optional, Tuple

import numpy as _np
import openmm.unit as _unit
import torch as _torch
from emle.train._loss import _BaseLoss
from loguru import logger as _logger


class BaseLoss(_BaseLoss):
    """Extend emle-engine base loss class to add more features."""

    def __init__(
        self,
        temperature: float = 300.0,
        weighting_method: str = "uniform",
        weights_fudge: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if not isinstance(weighting_method, str):
            raise TypeError("weighting_method must be a string")
        weighting_method = weighting_method.lower()
        if weighting_method not in ["boltzmann", "uniform", "non-boltzmann"]:
            raise ValueError(
                'weighting_method must be one of "boltzmann", "uniform", or "non-boltzmann"'
            )
        self._weighting_method = weighting_method

        if not isinstance(temperature, (int, float)):
            raise TypeError("temperature must be a number")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self._temperature = temperature * _unit.kelvin
        self._kBT = (
            _unit.BOLTZMANN_CONSTANT_kB
            * _unit.AVOGADRO_CONSTANT_NA
            * self._temperature
            / _unit.kilojoules_per_mole
        )

        if isinstance(weights_fudge, (int, float)):
            if weights_fudge <= 0:
                raise ValueError("weights_fudge must be positive")
            self._weights_fudge = weights_fudge
        elif isinstance(weights_fudge, (list, tuple, _np.ndarray)):
            self._weights_fudge = _torch.tensor(weights_fudge)
        elif isinstance(weights_fudge, _torch.Tensor):
            self._weights_fudge = weights_fudge
        else:
            raise TypeError(
                "weights_fudge must be a number or a list/tuple/array/tensor of numbers"
            )

        self._weights = None
        self._weights_normalization = None

    def forward(self, *args, **kwargs) -> _torch.Tensor:
        """Forward pass."""
        raise NotImplementedError("This method should be implemented by the subclass.")

    def precompute_weights(
        self,
        e_int_target: _torch.Tensor,
        e_int_predicted: Optional[_torch.Tensor] = None,
    ) -> Tuple[_torch.Tensor, float]:
        """Precompute weights for energy fitting."""
        _logger.info(
            f"Precomputing weights for {self._weighting_method} weighting method."
        )
        if self._weighting_method == "non-boltzmann":
            raise ValueError(
                "Non-boltzmann weighting is not supported for precomputation."
            )

        self._weights = self._calculate_weights(
            e_int_target, e_int_predicted, self._weighting_method
        )
        self._weights_normalization = self._weights.sum()

        return self._weights, self._weights_normalization

    def _calculate_weights(
        self, e_int_target: _torch.Tensor, e_int_predicted: _torch.Tensor, method: str
    ) -> _torch.Tensor:
        """
        Calculate weights for energy fitting based on the specified method.

        Parameters
        ----------
        e_int_target : _torch.Tensor
            Target interaction energies.
        e_int_predicted : _torch.Tensor
            Predicted interaction energies.
        method : str
            Weighting method to use. Options: "boltzmann", "uniform", "non-boltzmann".

        Returns
        -------
        _torch.Tensor
            Normalized weights for the loss function.

        Raises
        ------
        ValueError
            If an invalid weighting method is specified.
        """
        if method == "boltzmann":
            return self._calculate_boltzmann_weights(e_int_target)
        elif method == "uniform":
            return self._calculate_uniform_weights(e_int_target)
        elif method == "non-boltzmann":
            return self._calculate_non_boltzmann_weights(e_int_target, e_int_predicted)
        else:
            raise ValueError(f"Invalid weighting method: {method}")

    def _calculate_uniform_weights(self, e_int_target: _torch.Tensor) -> _torch.Tensor:
        """
        Calculate uniform weights for energy fitting.

        Parameters
        ----------
        e_int_target : _torch.Tensor
            Target interaction energies.

        Returns
        -------
        _torch.Tensor
            Unnormalized uniform weights.
        """
        return _torch.ones_like(e_int_target)

    def _calculate_boltzmann_weights(
        self, e_int_target: _torch.Tensor
    ) -> _torch.Tensor:
        """
        Calculate Boltzmann weights for energy fitting.

        Parameters
        ----------
        e_int_target : _torch.Tensor
            Target interaction energies.

        Returns
        -------
        _torch.Tensor
            Unnormalized Boltzmann weights.
        """
        return _torch.exp(-e_int_target / self._kBT)

    def _calculate_non_boltzmann_weights(
        self, e_int_target: _torch.Tensor, e_int_predicted: _torch.Tensor
    ) -> _torch.Tensor:
        """
        Calculate non-Boltzmann weights for energy fitting.

        Parameters
        ----------
        e_int_target : _torch.Tensor
            Target interaction energies.
        e_int_predicted : _torch.Tensor
            Predicted interaction energies.

        Returns
        -------
        _torch.Tensor
            Unnormalized non-Boltzmann weights.
        """
        return _torch.exp(-(e_int_target - e_int_predicted) / self._kBT)
