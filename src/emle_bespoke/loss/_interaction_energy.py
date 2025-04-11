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

from .._constants import HARTREE_TO_KJ_MOL
from ..lj._lj_potential import LennardJonesPotential as _LennardJonesPotential
from ..lj._lj_potential_efficient import (
    LennardJonesPotentialEfficient as _LennardJonesPotentialEfficient,
)
from ._base import BaseLoss as _BaseLoss
from ._weighted_mse import WeightedMSELoss as _WeightedMSELoss


class InteractionEnergyLoss(_BaseLoss):
    """
    Loss function for fitting interaction energy curves.

    This loss function combines EMLE and Lennard-Jones potential energies to fit
    interaction energy curves. It supports various weighting schemes and includes
    L2 regularization for the LJ parameters. It can be used to fit LJ or EMLE
    parameters, or both.

    Attributes
    ----------
    emle_model : _EMLE
        The EMLE model for computing static and induced energies.
    lj_potential : _LennardJonesPotential
        The Lennard-Jones potential for computing LJ energies.
    loss : _torch.nn.Module
        The base loss function (e.g., WeightedMSELoss).
    _e_static_emle : _torch.Tensor or None
        Cached static EMLE energies.
    _e_ind_emle : _torch.Tensor or None
        Cached induced EMLE energies.
    _l2_reg : float
        L2 regularization strength.
    """

    def __init__(
        self,
        emle_model: _EMLE,
        lj_potential: _LennardJonesPotential,
        loss: _torch.nn.Module = _WeightedMSELoss(),
        e_static_emle: Optional[_torch.Tensor] = None,
        e_ind_emle: Optional[_torch.Tensor] = None,
        e_lj: Optional[_torch.Tensor] = None,
        l2_reg: float = 1.0,
        *args,
        **kwargs,
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
        e_static_emle : _torch.Tensor, optional
            Static EMLE energies.
        e_ind_emle : _torch.Tensor, optional
            Induced EMLE energies.
        l2_reg : float
            L2 regularization strength. Default is 1.0.
        """
        super().__init__(
            *args,
            **kwargs,
        )

        if not isinstance(emle_model, _EMLE):
            raise TypeError("emle_model must be an instance of EMLE")
        self.emle_model = emle_model

        if not isinstance(
            lj_potential, (_LennardJonesPotential, _LennardJonesPotentialEfficient)
        ):
            raise TypeError(
                "lj_potential must be an instance of LennardJonesPotential or LennardJonesPotentialEfficient"
            )
        self.lj_potential = lj_potential

        if not isinstance(loss, _torch.nn.Module):
            raise TypeError("loss must be an instance of torch.nn.Module")
        self.loss = loss

        if not isinstance(l2_reg, (int, float)):
            raise TypeError("l2_reg must be a number")
        if l2_reg < 0:
            raise ValueError("l2_reg must be non-negative")
        self._l2_reg = l2_reg

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

        if e_lj is not None:
            if isinstance(e_lj, (list, _np.ndarray)):
                e_lj = _torch.tensor(e_lj)
            self._e_lj = e_lj
        else:
            self._e_lj = None

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
        if isinstance(self.loss, _WeightedMSELoss):
            if self._weights_fudge is not None:
                if isinstance(self._weights_fudge, _torch.Tensor):
                    weights_fudge = self._weights_fudge.to(target.device)[indices]
                else:
                    weights_fudge = self._weights_fudge
            else:
                weights_fudge = 1.0

            if self._weights_normalization is None:
                # If there's no global normalization, it means weights were not pre-computed
                # so we need to compute them here. Note that the normalization is done here, so
                # it will not be a global normalization. Should only be used for non-boltzmann weighting.
                weights = self._calculate_weights(
                    target, values, self._weighting_method
                ).to(target.device)
                weights = weights * weights_fudge
                weights = weights / weights.sum()
            else:
                weights = self._weights[indices] / self._weights_normalization
                weights = weights * weights_fudge
            loss = self.loss(values, target, weights)
        elif isinstance(self.loss, _torch.nn.MSELoss):
            loss = self.loss(values, target)
        else:
            raise NotImplementedError(f"Loss function {self.loss} not implemented")

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
            e_static, e_ind = self.emle_model.forward(
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

        # Calculate or retrieve LJ potential energy
        if self._e_lj is None:
            e_lj = self.lj_potential.forward(
                xyz,
                solute_mask=solute_mask,
                solvent_mask=solvent_mask,
                indices=indices,
            )
        else:
            e_lj = self._e_lj[indices]

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
            self.lj_potential._num_atom_types + 1, device=self.lj_potential._device
        )
        epsilon = self.lj_potential._epsilon_embedding(atom_types)
        sigma = self.lj_potential._sigma_embedding(atom_types)

        # Calculate parameter differences
        epsilon_diff = epsilon - self.lj_potential._epsilon_init
        sigma_diff = sigma - self.lj_potential._sigma_init

        # Calculate regularization term
        return l2_reg * (epsilon_diff.square().sum() + sigma_diff.square().sum())
