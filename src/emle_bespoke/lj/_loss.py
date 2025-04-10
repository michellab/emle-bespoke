"""Loss functions for EMLE patched model training.

This module provides loss functions for training the EMLE patched model, including:
- Weighted MSE loss with customizable normalization
- Interaction energy loss combining EMLE and Lennard-Jones components
- Various weighting schemes for energy fitting
"""

from typing import Optional, Tuple

import numpy as _np
import torch as _torch
from emle.models import EMLE as _EMLE
from emle.train._loss import _BaseLoss
from loguru import logger as _logger

from .._constants import ANGSTROM_TO_NANOMETER, HARTREE_TO_KJ_MOL
from ._lj_potential import LennardJonesPotential as _LennardJonesPotential


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
        e_static_emle: _torch.Tensor = None,
        e_ind_emle: _torch.Tensor = None,
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

        if not isinstance(lj_potential, _LennardJonesPotential):
            raise TypeError("lj_potential must be an instance of LennardJonesPotential")
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
        indices: Optional[Tuple[int, int]] = None,
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
            QM atom positions in nanometers.
        xyz_mm : _torch.Tensor
            MM atom positions in nanometers.
        xyz : _torch.Tensor
            Combined atom positions in nanometers.
        solute_mask : _torch.Tensor
            Mask for solute atoms.
        solvent_mask : _torch.Tensor
            Mask for solvent atoms.
        indices : Optional[Tuple[int, int]], optional
            Start and end indices for batch processing. If None, processes all data.

        Returns
        -------
        Tuple[_torch.Tensor, float, float]
            Tuple containing:
            - Total loss (energy fitting + regularization)
            - RMSE of the energy prediction
            - Maximum absolute error
        """
        # Determine batch indices
        if indices is not None:
            start_idx, end_idx = indices[0], indices[-1] + 1
        else:
            start_idx, end_idx = 0, None

        # Calculate predicted energies
        e_static, e_ind, e_lj = self._compute_predictions(
            atomic_numbers=atomic_numbers,
            charges_mm=charges_mm,
            xyz_qm=xyz_qm,
            xyz_mm=xyz_mm,
            xyz=xyz,
            solute_mask=solute_mask,
            solvent_mask=solvent_mask,
            start_idx=start_idx,
            end_idx=end_idx,
        )

        # Calculate total predicted energy
        values = e_static + e_ind + e_lj
        target = e_int_target

        # Calculate base loss
        if isinstance(self._loss, WeightedMSELoss):
            weights = self._calculate_weights(
                e_int_target, values, self._weighting_method
            ).to(e_int_target.device)
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
        start_idx: int = 0,
        end_idx: Optional[int] = None,
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
            QM atom positions in nanometers.
        xyz_mm : _torch.Tensor
            MM atom positions in nanometers.
        xyz : _torch.Tensor
            Combined atom positions in nanometers.
        solute_mask : _torch.Tensor
            Mask for solute atoms.
        solvent_mask : _torch.Tensor
            Mask for solvent atoms.
        start_idx : int, optional
            Start index for batch processing. Default is 0.
        end_idx : Optional[int], optional
            End index for batch processing. If None, processes until the end.

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

        # Calculate or retrieve EMLE energies
        if self._e_static_emle is None or self._e_ind_emle is None:
            e_static, e_ind = self._emle_model.forward(
                atomic_numbers,
                charges_mm,
                xyz_qm / ANGSTROM_TO_NANOMETER,
                xyz_mm / ANGSTROM_TO_NANOMETER,
            )
            e_static = e_static * HARTREE_TO_KJ_MOL
            e_ind = e_ind * HARTREE_TO_KJ_MOL
        else:
            e_static = self._e_static_emle[start_idx:end_idx]
            e_ind = self._e_ind_emle[start_idx:end_idx]

        # Calculate LJ potential energy
        e_lj = self._lj_potential.forward(
            xyz,
            solute_mask=solute_mask,
            solvent_mask=solvent_mask,
            start_idx=start_idx,
            end_idx=end_idx,
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
            if self._weights is None:
                self._weights = self._calculate_boltzmann_weights(e_int_target)
            return self._weights
        elif method == "uniform":
            if self._weights is None:
                self._weights = self._calculate_uniform_weights(e_int_target)
            return self._weights
        elif method == "non-boltzmann":
            return self._calculate_non_boltzmann_weights(e_int_target, e_int_predicted)
        else:
            raise ValueError(f"Invalid weighting method: {method}")

    def _calculate_boltzmann_weights(
        self, e_int_target: _torch.Tensor, temperature: float = 500.0
    ) -> _torch.Tensor:
        """
        Calculate Boltzmann weights for energy fitting.

        Parameters
        ----------
        e_int_target : _torch.Tensor
            Target interaction energies.
        temperature : float, optional
            Temperature in Kelvin for Boltzmann weighting. Default is 500.0.

        Returns
        -------
        _torch.Tensor
            Normalized Boltzmann weights.
        """
        import openmm.unit as _unit

        kBT = (
            _unit.BOLTZMANN_CONSTANT_kB
            * _unit.AVOGADRO_CONSTANT_NA
            * temperature
            * _unit.kelvin
            / _unit.kilojoules_per_mole
        )

        weights = _torch.zeros_like(e_int_target)
        window_sizes = self._lj_potential._windows
        frame = 0

        for size in window_sizes:
            window_end = frame + size
            e_int_target_window = e_int_target[frame:window_end]
            window_weights = _torch.ones_like(e_int_target_window)

            # Define energy thresholds
            mask_uniform = e_int_target_window < 4.184  # 1 kcal/mol
            mask_filter = e_int_target_window > 5 * 4.184  # 5 kcal/mol
            mask_middle = ~mask_uniform & ~mask_filter

            # Apply weighting scheme
            window_weights[mask_filter] = 0.0
            window_weights[mask_middle] = 1.0 / _torch.sqrt(
                1 + (e_int_target_window[mask_middle] / 4.184 - 1) ** 2
            )

            # Normalize weights
            total_weight = window_weights.sum()
            if total_weight > 0:
                window_weights /= total_weight

            weights[frame:window_end] = window_weights
            frame = window_end

        return weights

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
        return _torch.ones_like(e_int_target)

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
            Normalized non-Boltzmann weights.
        """
        import openmm.unit as _unit

        temperature = 300.0 * _unit.kelvin
        kBT = (
            _unit.BOLTZMANN_CONSTANT_kB
            * _unit.AVOGADRO_CONSTANT_NA
            * temperature
            / _unit.kilojoules_per_mole
        )
        weights = _torch.exp(-(e_int_target - e_int_predicted) / kBT)
        return weights / weights.sum()
