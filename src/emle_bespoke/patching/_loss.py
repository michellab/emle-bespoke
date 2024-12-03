"""Loss function for the EMLE patched model."""
import torch as _torch
from emle.models import EMLE
from emle.train._loss import _BaseLoss

from .._constants import HARTREE_TO_KJ_MOL


class PatchingLoss(_BaseLoss):
    def __init__(self, emle_model, loss=_torch.nn.MSELoss()):
        super().__init__()

        if not isinstance(emle_model, EMLE):
            raise TypeError("emle_model must be an instance of EMLE")

        self._emle_model = emle_model

        if not isinstance(loss, _torch.nn.Module):
            raise TypeError("loss must be an instance of torch.nn.Module")

        self._loss = loss

    def forward(
        self,
        e_static_target,
        e_ind_target,
        atomic_numbers,
        charges_mm,
        xyz_qm,
        xyz_mm,
        fit_e_static=True,
        fit_e_ind=True,
        n_batches=8,
    ):
        """
        Forward pass.

        Parameters
        ----------
        e_static_target: torch.Tensor (NBATCH,)
            Target static energy component in kJ/mol.
        e_ind_target: torch.Tensor (NBATCH,)
            Target induced energy component in kJ/mol.
        atomic_numbers: torch.Tensor (NBATCH, N_QM_ATOMS)
            Atomic numbers of QM atoms.
        charges_mm: torch.Tensor (NBATCH, max_mm_atoms)
            MM point charges in atomic units.
        xyz_qm: torch.Tensor (NBATCH, N_QM_ATOMS, 3)
            Positions of QM atoms in Angstrom.
        xyz_mm: torch.Tensor (NBATCH, N_MM_ATOMS, 3)
            Positions of MM atoms in Angstrom.
        fit_e_static: bool
            Whether to fit the static component.
        fit_e_ind: bool
            Whether to fit the induced component.
        """

        # Update the EMLE model with the current parameters
        self._update_s_gpr(self._emle_model._emle_base)
        self._update_chi_gpr(self._emle_model._emle_base)
        self._update_sqrtk_gpr(self._emle_model._emle_base)

        # Calculate EMLE predictions for static and induced components in a batched manner
        n_samples = atomic_numbers.shape[0]
        batch_size = n_samples // n_batches
        e_static_all = []
        e_ind_all = []

        for i in range(n_batches):
            batch_start = int(i * batch_size)
            batch_end = (
                int((i + 1) * batch_size) if i < n_batches - 1 else int(n_samples)
            )

            e_static, e_ind = self._emle_model(
                atomic_numbers[batch_start:batch_end],
                charges_mm[batch_start:batch_end],
                xyz_qm[batch_start:batch_end],
                xyz_mm[batch_start:batch_end],
            )

            e_static = e_static * HARTREE_TO_KJ_MOL
            e_ind = e_ind * HARTREE_TO_KJ_MOL

            e_static_all.append(e_static)
            e_ind_all.append(e_ind)

        e_static = _torch.cat(e_static_all, dim=0)
        e_ind = _torch.cat(e_ind_all, dim=0)

        target = (e_static_target if fit_e_static else 0) + (
            e_ind_target if fit_e_ind else 0
        )
        values = (e_static if fit_e_static else 0) + (e_ind if fit_e_ind else 0)

        if not fit_e_static and not fit_e_ind:
            raise ValueError("At least one of fit_e_static or fit_e_ind must be True")

        return (
            self._loss(values, target),
            self._get_rmse(values, target),
            self._get_max_error(values, target),
        )

    @staticmethod
    def _update_sqrtk_gpr(emle_base):
        emle_base._ref_mean_sqrtk, emle_base._c_sqrtk = emle_base._get_c(
            emle_base._n_ref,
            emle_base.ref_values_sqrtk,
            emle_base._Kinv,
        )

    @staticmethod
    def _update_chi_gpr(emle_base):
        emle_base._ref_mean_chi, emle_base._c_chi = emle_base._get_c(
            emle_base._n_ref,
            emle_base.ref_values_chi,
            emle_base._Kinv,
        )

    @staticmethod
    def _update_s_gpr(emle_base):
        emle_base._ref_mean_s, emle_base._c_s = emle_base._get_c(
            emle_base._n_ref,
            emle_base.ref_values_s,
            emle_base._Kinv,
        )
