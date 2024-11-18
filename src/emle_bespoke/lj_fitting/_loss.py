"""Loss function for the EMLE patched model."""
import torch as _torch
from emle.models import EMLE as _EMLE
from emle.train._loss import _BaseLoss

from .._constants import HARTREE_TO_KJ_MOL
from ._lj_potential import LennardJonesPotential as _LennardJonesPotential


class InteractionEnergyLoss(_BaseLoss):
    """Loss function for fitting the interaction energy curve to the Lennard-Jones potential."""

    def __init__(self, emle_model, lj_potential, loss=_torch.nn.MSELoss()):
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

    def calculate_predicted_interaction_energy(
        self, atomic_numbers, charges_mm, pos, solvent_indices, solute_indices
    ):
        # Calculate EMLE predictions for static and induced components and LJ potential energy
        e_static, e_ind = self._emle_model(
            atomic_numbers, charges_mm, pos[solute_indices], pos[solvent_indices]
        )
        e_static = e_static * HARTREE_TO_KJ_MOL
        e_ind = e_ind * HARTREE_TO_KJ_MOL
        e_lj = self._lj_potential.forward(pos, solute_indices, solvent_indices)

        return e_static, e_ind, e_lj

    def forward(
        self,
        e_int_target,
        atomic_numbers,
        charges_mm,
        pos,
        solvent_indices,
        solute_indices,
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
        pos: torch.Tensor (NBATCH, max_atoms, 3)
            Cartesian coordinates in nm.
        solvent_indices: torch.Tensor (NBATCH, N_MM_ATOMS)
            Indices of solvent atoms.
        solute_indices: torch.Tensor (NBATCH, N_QM_ATOMS)
            Indices of solute atoms.
        """
        # Calculate EMLE predictions for static and induced components
        e_static_list = []
        e_ind_list = []
        e_lj = []
        for i in range(len(atomic_numbers)):
            e_static, e_ind, e_lj = self.calculate_predicted_interaction_energy(
                atomic_numbers=atomic_numbers[i],
                charges_mm=charges_mm[i],
                pos=pos[i],
                solvent_indices=solvent_indices,
                solute_indices=solute_indices,
            )
            e_static_list.append(e_static)
            e_ind_list.append(e_ind)
            e_lj.append(e_lj)

        e_static = _torch.stack(e_static_list) * HARTREE_TO_KJ_MOL
        e_ind = _torch.stack(e_ind_list) * HARTREE_TO_KJ_MOL
        e_lj = self._lj_potential.forward(pos, solute_indices, solvent_indices)

        target = e_int_target
        values = e_static + e_ind + e_lj

        return (
            self._loss(values, target),
            self._get_rmse(values, target),
            self._get_max_error(values, target),
        )
