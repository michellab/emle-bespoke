"""Loss functions for EMLE patched model training.

This module provides loss functions for training the EMLE patched model, including:
- Weighted MSE loss with customizable normalization
- Interaction energy loss combining EMLE and Lennard-Jones components
- Various weighting schemes for energy fitting
"""

from typing import Optional, Tuple

import numpy as _np
import openmm.unit as _unit
import torch as _torch
from emle.models import EMLE as _EMLE
from emle.train._loss import _BaseLoss
from loguru import logger as _logger

from .._constants import HARTREE_TO_KJ_MOL
from ._lj_potential import LennardJonesPotential as _LennardJonesPotential
from ._lj_potential_efficient import (
    LennardJonesPotentialEfficient as _LennardJonesPotentialEfficient,
)


class WeightedMSELoss(_torch.nn.Module):
    """Weighted Mean Squared Error loss with optional normalization.

    This loss function computes a weighted MSE where each sample can have a different
    weight. The weights can be used to focus the training on specific regions of the
    energy landscape or to balance different types of interactions.

    Attributes
    ----------
    _normalization : float or None
        Normalization factor for the loss. If None, no normalization is applied.
    """

    def __init__(self, normalization: float = 1.0) -> None:
        """
        Initialize the weighted MSE loss.

        Parameters
        ----------
        normalization : float, optional
            Initial normalization factor for the loss. Default is 1.0.
        """
        super().__init__()
        self._normalization = normalization

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

        return weighted_squared_error.sum() * self._normalization


class InteractionEnergyLoss(_BaseLoss):
    """
    Loss function for fitting interaction energy curves.

    This loss function combines EMLE and Lennard-Jones potential energies to fit
    interaction energy curves. It supports various weighting schemes and includes
    L2 regularization for the LJ parameters.

    Attributes
    ----------
    _emle_model : _EMLE
        The EMLE model for computing static and induced energies.
    _lj_potential : _LennardJonesPotential
        The Lennard-Jones potential for computing LJ energies.
    _loss : _torch.nn.Module
        The base loss function (e.g., WeightedMSELoss).
    _weighting_method : str
        The method used for weighting the energy contributions.
    _e_static_emle : _torch.Tensor or None
        Cached static EMLE energies.
    _e_ind_emle : _torch.Tensor or None
        Cached induced EMLE energies.
    _l2_reg : float
        L2 regularization strength.
    _weights : _torch.Tensor or None
        Cached weights for the loss function.
    """

    def __init__(
        self,
        emle_model: _EMLE,
        lj_potential: _LennardJonesPotential,
        loss: _torch.nn.Module = WeightedMSELoss(),
        weighting_method: str = "uniform",
        temperature: float = 300.0,
        e_static_emle: Optional[_torch.Tensor] = None,
        e_ind_emle: Optional[_torch.Tensor] = None,
        l2_reg: float = 1.0,
    ) -> None:
        """
        Initialize the LJ patching loss.

        Parameters
        ----------
        emle_model : _EMLE
            The EMLE model for computing static and induced energies.
        lj_potential : _LennardJonesPotential
            The Lennard-Jones potential for computing LJ energies.
        loss : _torch.nn.Module, optional
            The base loss function. Default is WeightedMSELoss().
        weighting_method : str, optional
            The method used for weighting the energy contributions.
            Options: "boltzmann", "uniform", "non-boltzmann". Default is "uniform".
        temperature : float, optional
            Temperature in Kelvin for Boltzmann weighting. Default is 300.0.
        e_static_emle : _torch.Tensor, optional
            Static EMLE energies.
        e_ind_emle : _torch.Tensor, optional
            Induced EMLE energies.
        l2_reg : float
            L2 regularization strength. Default is 1.0.
        """
        super().__init__()

        if not isinstance(emle_model, _EMLE):
            raise TypeError("emle_model must be an instance of EMLE")
        self._emle_model = emle_model

        if not isinstance(
            lj_potential, (_LennardJonesPotential, _LennardJonesPotentialEfficient)
        ):
            raise TypeError(
                "lj_potential must be an instance of LennardJonesPotential or LennardJonesPotentialEfficient"
            )
        self._lj_potential = lj_potential

        if not isinstance(loss, _torch.nn.Module):
            raise TypeError("loss must be an instance of torch.nn.Module")
        self._loss = loss

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
        if not isinstance(l2_reg, (int, float)):
            raise TypeError("l2_reg must be a number")
        if l2_reg < 0:
            raise ValueError("l2_reg must be non-negative")
        self._l2_reg = l2_reg

        self._weights = None

        if e_static_emle is not None:
            if isinstance(e_static_emle, (list, _np.ndarray)):
                e_static_emle = _torch.tensor(e_static_emle)
            self._e_static_emle = e_static_emle
        else:
            self._e_static_emle = None

        if e_ind_emle is not None:
            if isinstance(e_ind_emle, (list, _np.ndarray)):
                e_ind_emle = _torch.tensor(e_ind_emle)
            self._e_ind_emle = e_ind_emle
        else:
            self._e_ind_emle = None

    def forward(
        self,
        e_int_target: _torch.Tensor,
        atomic_numbers: _torch.Tensor,
        charges_mm: _torch.Tensor,
        xyz_qm: _torch.Tensor,
        xyz_mm: _torch.Tensor,
        xyz: _torch.Tensor,
        solute_mask: _torch.Tensor,
        solvent_mask: _torch.Tensor,
        indices: Optional[_torch.Tensor] = None,
    ) -> Tuple[_torch.Tensor, float, float]:
        """
        Calculate the total loss including energy fitting and regularization.

        Parameters
        ----------
        e_int_target : _torch.Tensor
            Target interaction energy in kJ/mol.
        atomic_numbers : _torch.Tensor
            Atomic numbers of QM atoms.
        charges_mm : _torch.Tensor
            MM point charges in atomic units.
        xyz_qm : _torch.Tensor
            QM atom positions in Angstrom.
        xyz_mm : _torch.Tensor
            MM atom positions in Angstrom.
        xyz : _torch.Tensor
            Combined atom positions in Angstrom.
        solute_mask : _torch.Tensor
            Mask for solute atoms.
        solvent_mask : _torch.Tensor
            Mask for solvent atoms.
        indices : Optional[_torch.Tensor], optional
            Indices of the topologies for which to compute the energy. If None, processes all data.

        Returns
        -------
        Tuple[_torch.Tensor, float, float]
            Tuple containing:
            - Total loss (energy fitting + regularization)
            - RMSE of the energy prediction
            - Maximum absolute error
        """
        # Calculate predicted energies
        e_static, e_ind, e_lj = self._compute_predictions(
            atomic_numbers=atomic_numbers,
            charges_mm=charges_mm,
            xyz_qm=xyz_qm,
            xyz_mm=xyz_mm,
            xyz=xyz,
            solute_mask=solute_mask,
            solvent_mask=solvent_mask,
            indices=indices,
        )

        # Calculate total predicted energy
        values = e_static + e_ind + e_lj
        target = e_int_target

        # Calculate base loss
        if isinstance(self._loss, WeightedMSELoss):
            weights = self._calculate_weights(
                target, values, self._weighting_method
            ).to(target.device)
            loss = self._loss(values, target, weights)
        elif isinstance(self._loss, _torch.nn.MSELoss):
            loss = self._loss(values, target)
        else:
            raise NotImplementedError(f"Loss function {self._loss} not implemented")

        # Add L2 regularization if enabled
        if self._l2_reg > 0:
            loss += self._calculate_l2_regularization(self._l2_reg)

        # Calculate error metrics
        rmse = self._get_rmse(values, target)
        max_error = self._get_max_error(values, target)

        return loss, rmse, max_error

    def _compute_predictions(
        self,
        atomic_numbers: _torch.Tensor,
        charges_mm: _torch.Tensor,
        xyz_qm: _torch.Tensor,
        xyz_mm: _torch.Tensor,
        xyz: _torch.Tensor,
        solute_mask: _torch.Tensor,
        solvent_mask: _torch.Tensor,
        indices: _torch.Tensor,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
        """
        Calculate predicted interaction energies from EMLE and LJ components.

        Parameters
        ----------
        atomic_numbers : _torch.Tensor
            Atomic numbers of QM atoms.
        charges_mm : _torch.Tensor
            MM point charges in atomic units.
        xyz_qm : _torch.Tensor
            QM atom positions in Angstrom.
        xyz_mm : _torch.Tensor
            MM atom positions in Angstrom.
        xyz : _torch.Tensor
            Combined atom positions in Angstrom.
        solute_mask : _torch.Tensor
            Mask for solute atoms.
        solvent_mask : _torch.Tensor
            Mask for solvent atoms.
        indices : _torch.Tensor
            Indices of the topologies for which to compute the energy.

        Returns
        -------
        Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]
            Tuple containing:
            - Static EMLE energy
            - Induced EMLE energy
            - LJ potential energy
        """
        # Ensure input tensors have batch dimension
        if atomic_numbers.ndim == 1:
            atomic_numbers = atomic_numbers.unsqueeze(0)
            charges_mm = charges_mm.unsqueeze(0)
            xyz_qm = xyz_qm.unsqueeze(0)
            xyz_mm = xyz_mm.unsqueeze(0)
            xyz = xyz.unsqueeze(0)
            solvent_mask = solvent_mask.unsqueeze(0)
            solute_mask = solute_mask.unsqueeze(0)

        # Calculate or retrieve EMLE energies.
        if self._e_static_emle is None or self._e_ind_emle is None:
            e_static, e_ind = self._emle_model.forward(
                atomic_numbers,
                charges_mm,
                xyz_qm,
                xyz_mm,
            )
            e_static = e_static * HARTREE_TO_KJ_MOL
            e_ind = e_ind * HARTREE_TO_KJ_MOL
        else:
            e_static = self._e_static_emle[indices]
            e_ind = self._e_ind_emle[indices]

        # Calculate LJ potential energy
        e_lj = self._lj_potential.forward(
            xyz,
            solute_mask=solute_mask,
            solvent_mask=solvent_mask,
            indices=indices,
        )

        return e_static, e_ind, e_lj

    def _calculate_l2_regularization(self, l2_reg: float) -> _torch.Tensor:
        """
        Calculate L2 regularization term for LJ parameters.

        Parameters
        ----------
        l2_reg : float
            L2 regularization strength.

        Returns
        -------
        _torch.Tensor
            L2 regularization term.
        """
        # Get current parameter values
        atom_types = _torch.arange(
            self._lj_potential._num_atom_types + 1, device=self._lj_potential._device
        )
        epsilon = self._lj_potential._epsilon_embedding(atom_types)
        sigma = self._lj_potential._sigma_embedding(atom_types)

        # Calculate parameter differences
        epsilon_diff = epsilon - self._lj_potential._epsilon_init
        sigma_diff = sigma - self._lj_potential._sigma_init

        # Calculate regularization term
        return l2_reg * (epsilon_diff.square().sum() + sigma_diff.square().sum())

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
            # if self._weights is None:
            self._weights = self._calculate_boltzmann_weights(e_int_target)
            return self._weights
        elif method == "uniform":
            # if self._weights is None:
            self._weights = self._calculate_uniform_weights(e_int_target)
            return self._weights
        elif method == "non-boltzmann":
            return self._calculate_non_boltzmann_weights(e_int_target, e_int_predicted)
        else:
            raise ValueError(f"Invalid weighting method: {method}")

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
            Normalized Boltzmann weights.
        """
        weights = _torch.exp(-e_int_target / self._kBT)
        return weights / weights.sum()

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
            Normalized uniform weights.
        """
        weights = _torch.ones_like(e_int_target)
        return weights / weights.sum()

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
            Non-Boltzmann weights.
        """
        weights = _torch.exp(-(e_int_target - e_int_predicted) / self._kBT)
        return weights / weights.sum()
