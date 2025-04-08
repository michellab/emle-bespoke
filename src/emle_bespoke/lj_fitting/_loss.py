"""Loss function for the EMLE patched model."""

import torch as _torch
from emle.models import EMLE as _EMLE
from emle.train._loss import _BaseLoss

from .._constants import ANGSTROM_TO_NANOMETER, HARTREE_TO_KJ_MOL
from ._lj_potential import LennardJonesPotential as _LennardJonesPotential


class WeightedMSELoss(_torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._normalization = None

    def forward(self, inputs, targets, weights):
        """
        assert (
            inputs.shape == targets.shape
        ), "Inputs and targets must have the same shape"
        assert (
            inputs.shape == weights.shape
        ), "Inputs and weights must have the same shape"
        """

        if self._normalization is None:
            self._normalization = 1.0

        diff = targets - inputs
        squared_error = (diff) ** 2
        weighted_squared_error = squared_error * weights

        return weighted_squared_error.sum() * self._normalization


class InteractionEnergyLoss(_BaseLoss):
    """Loss function for fitting the interaction energy curve to the Lennard-Jones potential."""

    def __init__(
        self,
        emle_model,
        lj_potential,
        loss=WeightedMSELoss(),
        weighting_method="boltzmann",
    ):
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

        self._weighting_method = weighting_method

        self._e_static_emle = None
        self._e_ind_emle = None
        self._weights = None
        self.l2_reg_calc = True

    def calulate_weights(self, e_int_target, e_int_predicted, method):
        import openmm.unit as _unit

        if method.lower() == "boltzmann":
            if self._weights is not None:
                weights = self._weights
            else:
                temperature = 500.0 * _unit.kelvin
                kBT = (
                    _unit.BOLTZMANN_CONSTANT_kB
                    * _unit.AVOGADRO_CONSTANT_NA
                    * temperature
                    / _unit.kilojoules_per_mole
                )
                weights = _torch.zeros_like(e_int_target)

                window_sizes = self._lj_potential._windows
                frame = 0
                for i in window_sizes:
                    size = i
                    window_end = frame + size

                    # Slice only once and reuse
                    e_int_target_window = e_int_target[frame:window_end]
                    # e_int_predicted_window = e_int_predicted[frame:window_end].detach()
                    window = weights[frame:window_end]
                    """
                    # Calculate the mask once
                    delta = 5.0 * 4.184
                    mask_filter = e_int_target_window > delta
                    mask_boltzmann = ~mask_filter

                    # Update weights with in-place operations
                    window[mask_boltzmann] = e_int_target_window[mask_boltzmann]
                    window_weights = _torch.exp(-window / kBT)
                    window_weights[mask_filter] = 0.0
                    """
                    window_weights = _torch.ones_like(e_int_target_window)

                    mask_uniform = e_int_target_window < 4.184
                    mask_filter = e_int_target_window > 5 * 4.184
                    mask_middle = ~mask_uniform & ~mask_filter

                    # Apply mask_filter directly to zero weights
                    window_weights[mask_filter] = 0.0
                    window_weights[mask_middle] = 1.0 / _torch.sqrt(
                        1 + (e_int_target_window[mask_middle] / 4.184 - 1) ** 2
                    )

                    # Normalize weights if sum is non-zero
                    total_weight = window_weights.sum() * float(mask_filter.sum())
                    if total_weight > 0:
                        window_weights /= total_weight

                    # Assign updated weights back to the original array
                    weights[frame:window_end] = window_weights

                    frame = window_end

                # Process in windows
                self._weights = weights
        elif method.lower() == "uniform":
            if self._weights is not None:
                weights = self._weights
            else:
                weights = _torch.zeros_like(e_int_target)

                window_sizes = self._lj_potential._windows
                frame = 0
                for i in window_sizes:
                    size = i
                    window_end = frame + size

                    # Slice only once and reuse
                    e_int_target_window = e_int_target[frame:window_end]
                    window = weights[frame:window_end]

                    # Calculate the mask once
                    mask_filter = e_int_target_window > 50.0
                    mask_boltzmann = ~mask_filter

                    window_weights = _torch.ones_like(e_int_target_window)
                    # Apply mask_filter directly to zero weights
                    window_weights[mask_filter] = 0.0

                    # Normalize weights if sum is non-zero
                    total_weight = float(mask_boltzmann.sum())
                    if total_weight > 0:
                        window_weights /= total_weight

                    # Assign updated weights back to the original array
                    weights[frame:window_end] = window_weights

                    frame = window_end

                # Process in windows
                self._weights = weights
        elif method.lower() == "non-boltzmann":
            temperature = 300.0 * _unit.kelvin
            kBT = (
                _unit.BOLTZMANN_CONSTANT_kB
                * _unit.AVOGADRO_CONSTANT_NA
                * temperature
                / _unit.kilojoules_per_mole
            )
            weights = _torch.exp(-(e_int_target - e_int_predicted) / kBT)
        else:
            raise ValueError(f"Invalid weighting method: {method}")

        return weights / weights.sum()

    def calculate_predicted_interaction_energy(
        self,
        atomic_numbers,
        charges_mm,
        xyz_qm,
        xyz_mm,
        xyz,
        solute_mask,
        solvent_mask,
        start_idx,
        end_idx,
    ):
        if atomic_numbers.ndim == 1:
            atomic_numbers = atomic_numbers.unsqueeze(0)
            charges_mm = charges_mm.unsqueeze(0)
            xyz_qm = xyz_qm.unsqueeze(0)
            xyz_mm = xyz_mm.unsqueeze(0)
            xyz = xyz.unsqueeze(0)
            solvent_mask = solvent_mask.unsqueeze(0)
            solute_mask = solute_mask.unsqueeze(0)

        if (
            self._lj_potential._e_static_emle is None
            or self._lj_potential._e_ind_emle is None
        ):
            # Calculate EMLE predictions for static and induced components
            e_static, e_ind = self._emle_model.forward(
                atomic_numbers,
                charges_mm,
                xyz_qm / ANGSTROM_TO_NANOMETER,
                xyz_mm / ANGSTROM_TO_NANOMETER,
            )
            e_static = e_static * HARTREE_TO_KJ_MOL
            e_ind = e_ind * HARTREE_TO_KJ_MOL
        else:
            e_static = self._lj_potential._e_static_emle.to(atomic_numbers.device)[
                start_idx:end_idx
            ]
            e_ind = self._lj_potential._e_ind_emle.to(atomic_numbers.device)[
                start_idx:end_idx
            ]

        # Calculate Lennard-Jones potential energy
        e_lj = self._lj_potential.forward(
            xyz,
            solute_mask=solute_mask,
            solvent_mask=solvent_mask,
            start_idx=start_idx,
            end_idx=end_idx,
        )

        return e_static, e_ind, e_lj

    def forward(
        self,
        e_int_target,
        atomic_numbers,
        charges_mm,
        xyz_qm,
        xyz_mm,
        xyz,
        solute_mask,
        solvent_mask,
        l2_reg=1.0,
        indices=None,
    ):
        """
        Forward pass.

        Parameters
        ----------
        e_int_target: torch.Tensor (NBATCH,)
            Target interaction energy in kJ/mol.
        atomic_numbers: torch.Tensor (NBATCH, N_QM_ATOMS)
            Atomic numbers of QM atoms.
        charges_mm: torch.Tensor (NBATCH, max_mm_atoms)
            MM point charges in atomic units.
        xyz_qm: torch.Tensor (NBATCH, N_QM_ATOMS, 3)
            QM atom positions in nanometers.
        xyz_mm: torch.Tensor (NBATCH, N_MM_ATOMS, 3)
            MM atom positions in nanometers.
        solute_mask: torch.Tensor (NBATCH, N_MM_ATOMS)
            Mask for the solute atoms.
        solute_mask: torch.Tensor (N_MM_ATOMS,)
            Mask for the solute atoms.
        solvent_mask: torch.Tensor (N_MM_ATOMS,)
            Mask for the solvent atoms.
        l2_reg: float or None
            L2 regularization strength. If None, no regularization is applied.
        """
        if indices is not None:
            start_idx, end_idx = indices[0], indices[-1] + 1
        else:
            start_idx, end_idx = 0, None

        e_static, e_ind, e_lj = self.calculate_predicted_interaction_energy(
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

        target = e_int_target
        values = e_static + e_ind + e_lj

        if isinstance(self._loss, WeightedMSELoss):
            weights = self.calulate_weights(
                e_int_target, values, self._weighting_method
            )[start_idx:end_idx].to(e_int_target.device)
            loss = self._loss(values, target, weights)
        elif isinstance(self._loss, _torch.nn.MSELoss):
            loss = self._loss(values, target)
        else:
            raise NotImplementedError(f"Loss function {self._loss} not implemented")

        if l2_reg is not None and self.l2_reg_calc:
            epsilon_std = (
                self._lj_potential._epsilon_init.std()
                if self._lj_potential._epsilon_init.shape[0] > 1
                else 1.0
            )
            sigma_std = (
                self._lj_potential._sigma_init.std()
                if self._lj_potential._sigma_init.shape[0] > 1
                else 1.0
            )

            # Get the current epsilon and sigma values
            atom_types = _torch.arange(
                self._lj_potential._num_atom_types + 1, device=xyz.device
            )
            epsilon = self._lj_potential._epsilon_embedding(atom_types)
            sigma = self._lj_potential._sigma_embedding(atom_types)

            # Calculate the regularization term
            epsilon_diff = (
                epsilon - self._lj_potential._epsilon_init
            )  # / (0.1*4.184) # / epsilon_std #/ self._lj_potential._epsilon_init.mean()# / epsilon_std
            sigma_diff = (
                sigma - self._lj_potential._sigma_init
            )  # / 0.1 # / sigma_std #/ self._lj_potential._sigma_init.mean()# / sigma_std
            reg = l2_reg * (epsilon_diff.square().sum() + sigma_diff.square().sum())

            loss += reg
        else:
            loss = loss

        return (
            loss,
            self._get_rmse(values, target),
            self._get_max_error(values, target),
        )
