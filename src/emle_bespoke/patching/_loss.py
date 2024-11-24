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
        self, e_static_target, e_ind_target, atomic_numbers, charges_mm, xyz_qm, xyz_mm
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
        """
        # Calculate EMLE predictions for static and induced components in a batched manner
        e_static, e_ind = self._emle_model(atomic_numbers, charges_mm, xyz_qm, xyz_mm)
        e_static = e_static * HARTREE_TO_KJ_MOL
        e_ind = e_ind * HARTREE_TO_KJ_MOL

        target = e_static_target + e_ind_target
        values = e_static + e_ind

        return (
            self._loss(values, target),
            self._get_rmse(values, target),
            self._get_max_error(values, target),
        )
