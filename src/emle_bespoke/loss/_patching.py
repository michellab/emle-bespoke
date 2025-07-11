"""Loss function for the EMLE patched model."""

import torch as _torch
from emle.models import EMLE as _EMLE

from .._constants import HARTREE_TO_KJ_MOL
from ._base import BaseLoss as _BaseLoss


class PatchingLoss(_BaseLoss):
    def __init__(self, emle_model, loss=_torch.nn.MSELoss()):
        """
        Initialize the PatchingLoss class.

        Parameters
        ----------
        emle_model: EMLE
            An instance of the EMLE model.
        loss: torch.nn.Module, optional
            Loss function to use (default: MSELoss).
        """
        super().__init__()
        if not isinstance(emle_model, _EMLE):
            raise TypeError("emle_model must be an instance of EMLE")

        self.emle_model = emle_model

        if not isinstance(loss, _torch.nn.Module):
            raise TypeError("loss must be an instance of torch.nn.Module")
        self.loss = loss
        self._epsilon = 1e-16

    def forward(
        self,
        e_static_target=None,
        e_ind_target=None,
        atomic_numbers=None,
        charges_mm=None,
        xyz_qm=None,
        xyz_mm=None,
        q_core=None,
        q_val=None,
        s=None,
        alpha=None,
        l2_reg_alpha=1.0,
        l2_reg_s=1.0,
        l2_reg_q=1.0,
        n_batches=32,
    ):
        """
        Forward pass.

        Parameters
        ----------
        e_static_target: torch.Tensor (NBATCH,), optional
            Target static energy component in kJ/mol.
        e_ind_target: torch.Tensor (NBATCH,), optional
            Target induced energy component in kJ/mol.
        atomic_numbers: torch.Tensor (NBATCH, N_QM_ATOMS)
            Atomic numbers of QM atoms.
        charges_mm: torch.Tensor (NBATCH, max_mm_atoms)
            MM point charges in atomic units.
        xyz_qm: torch.Tensor (NBATCH, N_QM_ATOMS, 3)
            Positions of QM atoms in Angstrom.
        xyz_mm: torch.Tensor (NBATCH, N_MM_ATOMS, 3)
            Positions of MM atoms in Angstrom.
        q_core: torch.Tensor, optional
            Core charges (default: None).
        q_val: torch.Tensor, optional
            Valence charges (default: None).
        s: torch.Tensor, optional
            Static component for regularization (default: None).
        n_batches: int
            Maximum number of batches for splitting input.
        """
        if e_static_target is None and e_ind_target is None:
            raise ValueError(
                "At least one of e_static_target or e_ind_target must be provided"
            )

        self._update_emle_model_parameters()

        # Calculate EMLE predictions in batches
        e_static_all, e_ind_all = self._compute_emle_predictions(
            atomic_numbers, charges_mm, xyz_qm, xyz_mm, n_batches
        )

        # Static component regularization
        l2_reg = _torch.tensor(0.0, device=atomic_numbers.device)
        s_pred, q_core_pred, q_val_pred, a_Thole_pred = self._predict_emle_parameters(
            atomic_numbers, xyz_qm
        )

        # Add regularization based on which components are being fit
        if e_static_target is not None:
            l2_reg += (
                self._calculate_static_regularization(
                    atomic_numbers, q_core, q_val, q_core_pred, q_val_pred
                )
                * l2_reg_q
            )
            if s is not None:
                l2_reg += (
                    self._calculate_s_regularization(atomic_numbers, s, s_pred)
                    * l2_reg_s
                )

        if e_ind_target is not None:
            alpha_pred = self._get_alpha_mol(a_Thole_pred, atomic_numbers > 0)
            l2_reg += (
                self._calculate_alpha_regularization(alpha, alpha_pred) * l2_reg_alpha
            )

        # Concatenate all batch results
        e_static = _torch.cat(e_static_all, dim=0)
        e_ind = _torch.cat(e_ind_all, dim=0)

        # Prepare targets and values for loss calculation
        target, values, target_orig, values_orig = self._prepare_targets_and_values(
            e_static_target, e_ind_target, e_static, e_ind
        )

        # Compute loss
        loss = self.loss(values, target)
        # print(f"Loss: {loss.item()}, L2 Reg: {l2_reg.item()}")
        loss += l2_reg

        # Return metrics
        return (
            loss,
            self._get_rmse(values_orig, target_orig),
            self._get_max_error(values_orig, target_orig),
        )

    def _update_emle_model_parameters(self):
        """Update EMLE model parameters."""
        base = self.emle_model._emle_base
        self._update_s_gpr(base)
        self._update_chi_gpr(base)
        if self.emle_model._alpha_mode == "reference":
            self._update_sqrtk_gpr(base)

    def _compute_emle_predictions(
        self, atomic_numbers, charges_mm, xyz_qm, xyz_mm, n_batches
    ):
        """Compute EMLE predictions in batches."""
        n_samples = atomic_numbers.shape[0]
        batch_size = max(1, n_samples // n_batches)
        n_batches = n_samples // batch_size

        e_static_all, e_ind_all = [], []
        for i in range(n_batches):
            batch_start = i * batch_size
            batch_end = n_samples if i == n_batches - 1 else (i + 1) * batch_size
            e_static, e_ind = self.emle_model(
                atomic_numbers[batch_start:batch_end],
                charges_mm[batch_start:batch_end],
                xyz_qm[batch_start:batch_end],
                xyz_mm[batch_start:batch_end],
            )

            e_static_all.append(e_static * HARTREE_TO_KJ_MOL)
            e_ind_all.append(e_ind * HARTREE_TO_KJ_MOL)

        return e_static_all, e_ind_all

    def _predict_emle_parameters(self, atomic_numbers, xyz_qm):
        """Predict q_core and q_val components."""
        q_mol = _torch.zeros(
            len(atomic_numbers), dtype=_torch.float64, device=atomic_numbers.device
        )
        s_pred, q_core_pred, q_val_pred, a_Thole_pred = self.emle_model._emle_base(
            atomic_numbers, xyz_qm, q_mol
        )
        return s_pred, q_core_pred, q_val_pred, a_Thole_pred

    def _calculate_static_regularization(
        self, atomic_numbers, q_core, q_val, q_core_pred, q_val_pred
    ):
        """Calculate regularization for static components."""
        q_target = q_core + q_val
        mask = atomic_numbers > 0
        diff = q_target[mask] - (q_core_pred + q_val_pred)[mask]

        # avg_diff, max_diff, min_diff = diff.abs().mean(), diff.max(), diff.min()
        # print(
        #   f"Q Regularization - Avg diff: {avg_diff:.4f}, Max diff: {max_diff:.4f}, Min diff: {min_diff:.4f}"
        # )

        return diff.square().mean() / q_target.std() ** 2

    def _calculate_s_regularization(self, atomic_numbers, s, s_pred):
        """Calculate regularization for the static (s) component."""
        mask = atomic_numbers > 0
        diff = s[mask] - s_pred[mask]

        # avg_diff, max_diff, min_diff = diff.abs().mean(), diff.max(), diff.min()
        # print(
        #   f"S Regularization - Avg diff: {avg_diff:.4f}, Max diff: {max_diff:.4f}, Min diff: {min_diff:.4f}"
        # )

        return diff.square().mean() / s.std() ** 2

    def _calculate_alpha_regularization(self, alpha, alpha_pred):
        """Calculate regularization for the induced component."""
        triu_row, triu_col = _torch.triu_indices(3, 3, offset=0)
        alpha_triu = alpha[:, triu_row, triu_col]
        alpha_pred_triu = alpha_pred[:, triu_row, triu_col]

        diff = alpha_triu - alpha_pred_triu
        # avg_diff, max_diff, min_diff = diff.abs().mean(), diff.max(), diff.min()
        # print(
        #   f"Alpha Regularization - Avg diff: {avg_diff:.4f}, Max diff: {max_diff:.4f}, Min diff: {min_diff:.4f}"
        # )

        return diff.square().mean() / alpha_triu.std() ** 2

    def _predict_s_component(self, atomic_numbers, xyz_qm, q_mol):
        """Helper function to predict s component using the EMLE base model."""
        s_pred, _, _, _ = self.emle_model._emle_base(atomic_numbers, xyz_qm, q_mol)
        return s_pred

    def _prepare_targets_and_values(
        self, e_static_target, e_ind_target, e_static, e_ind
    ):
        """Prepare target and predicted values for loss calculation."""
        target = (e_static_target if e_static_target is not None else 0) + (
            e_ind_target if e_ind_target is not None else 0
        )
        values = (e_static if e_static_target is not None else 0) + (
            e_ind if e_ind_target is not None else 0
        )
        std = target.std() + self._epsilon
        return target / std, values / std, target, values

    @staticmethod
    def _get_alpha_mol(A_thole, mask):
        """Calculates molecular dipolar polarizability tensor from the A_thole matrix."""
        n_atoms = mask.shape[1]

        mask_mat = (
            (mask[:, :, None] * mask[:, None, :])
            .repeat_interleave(3, dim=1)
            .repeat_interleave(3, dim=2)
        )

        A_thole_inv = _torch.where(mask_mat, _torch.linalg.inv(A_thole), 0.0)
        return _torch.sum(A_thole_inv.reshape((-1, n_atoms, 3, n_atoms, 3)), dim=(1, 3))

    @staticmethod
    def _update_sqrtk_gpr(emle_base):
        emle_base._ref_mean_sqrtk, emle_base._c_sqrtk = emle_base._get_c(
            emle_base._n_ref, emle_base.ref_values_sqrtk, emle_base._Kinv
        )

    @staticmethod
    def _update_chi_gpr(emle_base):
        emle_base._ref_mean_chi, emle_base._c_chi = emle_base._get_c(
            emle_base._n_ref, emle_base.ref_values_chi, emle_base._Kinv
        )

    @staticmethod
    def _update_s_gpr(emle_base):
        emle_base._ref_mean_s, emle_base._c_s = emle_base._get_c(
            emle_base._n_ref, emle_base.ref_values_s, emle_base._Kinv
        )
