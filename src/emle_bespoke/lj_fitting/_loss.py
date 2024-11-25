"""Loss function for the EMLE patched model."""
import torch as _torch
from emle.models import EMLE as _EMLE
from emle.train._loss import _BaseLoss

from .._constants import ANGSTROM_TO_NANOMETER, HARTREE_TO_KJ_MOL
from ._lj_potential import LennardJonesPotential as _LennardJonesPotential


class WeightedMSELoss(_torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, weights):
        assert (
            inputs.shape == targets.shape
        ), "Inputs and targets must have the same shape"
        assert (
            inputs.shape == weights.shape
        ), "Inputs and weights must have the same shape"
        squared_error = (inputs - targets) ** 2
        weighted_squared_error = squared_error * weights
        return weighted_squared_error.sum()


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

    @staticmethod
    def calulate_weights(e_int_target, e_int_predicted, method):
        import openmm.unit as _unit

        if method.lower() == "boltzmann":
            temperature = 300.0 * _unit.kelvin
            kBT = (
                _unit.BOLTZMANN_CONSTANT_kB
                * _unit.AVOGADRO_CONSTANT_NA
                * temperature
                / _unit.kilojoules_per_mole
            )
            weights = _torch.exp(-e_int_target / kBT)
        elif method.lower() == "uniform":
            n_samples = len(e_int_target)
            weights = _torch.ones(
                n_samples, device=e_int_target.device, dtype=e_int_target.dtype
            )
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
        self, atomic_numbers, charges_mm, xyz_qm, xyz_mm, solute_mask, solvent_mask
    ):
        if atomic_numbers.ndim == 1:
            atomic_numbers = atomic_numbers.unsqueeze(0)
            charges_mm = charges_mm.unsqueeze(0)
            xyz_qm = xyz_qm.unsqueeze(0)
            xyz_mm = xyz_mm.unsqueeze(0)
            solvent_mask = solvent_mask.unsqueeze(0)
            solute_mask = solute_mask.unsqueeze(0)

        # Calculate EMLE predictions for static and induced components
        e_static, e_ind = self._emle_model.forward(
            atomic_numbers,
            charges_mm,
            xyz_qm / ANGSTROM_TO_NANOMETER,
            xyz_mm / ANGSTROM_TO_NANOMETER,
        )
        e_static = e_static * HARTREE_TO_KJ_MOL
        e_ind = e_ind * HARTREE_TO_KJ_MOL

        # Calculate Lennard-Jones potential energy
        e_lj = self._lj_potential.forward(
            _torch.cat([xyz_qm, xyz_mm], dim=1),
            solute_mask=solute_mask,
            solvent_mask=solvent_mask,
        )

        return e_static, e_ind, e_lj

    def forward(
        self,
        e_int_target,
        atomic_numbers,
        charges_mm,
        xyz_qm,
        xyz_mm,
        solute_mask,
        solvent_mask,
        l2_reg=1.0,
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
        solute_mask: torch.Tensor (N_MM_ATOMS,)
            Mask for the solute atoms.
        solvent_mask: torch.Tensor (N_MM_ATOMS,)
            Mask for the solvent atoms.
        l2_reg: float or None
            L2 regularization strength. If None, no regularization is applied.
        """
        # Calculate the predicted interaction energy
        e_static, e_ind, e_lj = self.calculate_predicted_interaction_energy(
            atomic_numbers=atomic_numbers,
            charges_mm=charges_mm,
            xyz_qm=xyz_qm,
            xyz_mm=xyz_mm,
            solute_mask=solute_mask,
            solvent_mask=solvent_mask,
        )

        target = e_int_target
        values = e_static + e_ind + e_lj

        if isinstance(self._loss, WeightedMSELoss):
            weights = self.calulate_weights(
                e_int_target, values, self._weighting_method
            )
            loss = self._loss(values, target, weights)
        elif isinstance(self._loss, _torch.nn.MSELoss):
            loss = self._loss(values, target)
        else:
            raise NotImplementedError(f"Loss function {self._loss} not implemented")

        if l2_reg is not None:
            epsilon_std = self._lj_potential._epsilon_tensor_initial.std()
            sigma_std = self._lj_potential._sigma_tensor_initial.std()

            epsilon_diff = (
                self._lj_potential._epsilon_tensor
                - self._lj_potential._epsilon_tensor_initial
            )
            epsilon_diff = epsilon_diff / epsilon_std
            sigma_diff = (
                self._lj_potential._sigma_tensor
                - self._lj_potential._sigma_tensor_initial
            )
            sigma_diff = sigma_diff / sigma_std

            reg = l2_reg * (epsilon_diff.square().sum() + sigma_diff.square().sum())
            loss = loss / target.std() + reg

        return (
            loss,
            self._get_rmse(values, target),
            self._get_max_error(values, target),
        )
