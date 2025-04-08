"""Loss function for fitting Lennard-Jones parameters to experimental data with reweighting."""

from typing import List as List

import torch as _torch
from emle.train._loss import _BaseLoss

from ._lj_potential import LennardJonesPotential as _LennardJonesPotential


class ReweightingLoss(_BaseLoss):
    """Loss function for fitting Lenard-Jones parameters to experimental data with reweighting."""

    def __init__(
        self,
        lj_potential=_LennardJonesPotential,
        loss=_torch.nn.MSELoss(),
        temperature=300.0,
    ):
        super().__init__()

        if not isinstance(lj_potential, _LennardJonesPotential):
            raise TypeError("lj_potential must be an instance of LennardJonesPotential")

        self._lj_potential = lj_potential

        if not isinstance(loss, _torch.nn.Module):
            raise TypeError("loss must be an instance of torch.nn.Module")

        self._loss = loss

        self._e_lj_unperturbed = None
        self._e_lj_perturbed = None
        self._beta = 1 / (8.31446261815324e-3 * temperature)

    def _calculate_lj_energy(
        self, xyz, solute_mask, solvent_mask, start_idx, end_idx, checkpoint
    ):
        """
        Calculate the Lennard-Jones energy for various potentials.

        Parameters
        ----------
        xyz: torch.Tensor (N_MOLECULES, N_BATCH, N_MM_ATOMS, 3)
            MM atom positions in nanometers.
        solute_mask: torch.Tensor (N_MOLECULES, N_BATCH, N_MM_ATOMS)
            Mask for the solute atoms.
        solvent_mask: torch.Tensor (N_MOLECULES, N_BATCH, N_MM_ATOMS)
            Mask for the solvent atoms.
        start_idx: int
            Start index of the atoms to consider.
        end_idx: int or None
            End index of the atoms to consider.
        """
        e_lj = self._lj_potential.forward(
            xyz,
            solute_mask=solute_mask,
            solvent_mask=solvent_mask,
            start_idx=start_idx,
            end_idx=end_idx,
            checkpoint=checkpoint,
        )
        return e_lj

    def calculate_lj(
        self,
        xyz,
        solute_mask,
        solvent_mask,
        indices=None,
        checkpoint=True,
        **kwargs,
    ):
        """
        Calculate the perturbed Lennard-Jones energy.

        Parameters
        ----------
        xyz : torch.Tensor (N_MOLECULES, N_BATCH, N_MM_ATOMS, 3)
            Molecular mechanics (MM) atom positions in nanometers.
        solute_mask : torch.Tensor (N_MOLECULES, N_BATCH, N_MM_ATOMS)
            Mask identifying solute atoms.
        solvent_mask : torch.Tensor (N_MOLECULES, N_BATCH, N_MM_ATOMS)
            Mask identifying solvent atoms.
        indices : list of int or None, optional
            Indices specifying the range of atoms to consider in the loss calculation.
            If None, all atoms are considered.
        unperturbed : bool, optional
            Whether to calculate the unperturbed energy.
        **kwargs : dict
            Additional parameters for flexibility in calculations.

        Returns
        -------
        loss : torch.Tensor (1,)
            The computed loss value.
        rmse : torch.Tensor (1,)
            Root mean squared error.
        max_error : torch.Tensor (1,)
            Maximum error observed in the predictions.
        """
        # Determine the range of indices to process
        if indices is not None:
            start_idx, end_idx = indices[0], indices[-1] + 1
        else:
            start_idx, end_idx = 0, None

        # Calculate the perturbed Lennard-Jones energy
        e_lj = self._calculate_lj_energy(
            xyz, solute_mask, solvent_mask, start_idx, end_idx, checkpoint=checkpoint
        )

        return e_lj

    def forward(
        self,
        xyz,
        solute_mask,
        solvent_mask,
        dG_exp,
        dG_calc,
        indices=None,
        **kwargs,
    ):
        """
        Forward pass.

        Parameters
        ----------
        dG_exp: torch.Tensor (N_MOLECULES,)
            Experimental free energy in kJ/mol, i.e., the target.
        dG_calc: torch.Tensor (N_MOLECULES,)
            Calculated free energy in kJ/mol, i.e., the unperturbed prediction.
        indices: list of int or None
            Indices of the atoms to consider in the loss calculation. If None, all atoms are considered.

        Returns
        -------
        loss: torch.Tensor (1,)
            Loss value.
        rmse: torch.Tensor (1,)
            Root mean squared error.
        max_error: torch.Tensor (1,)
            Maximum error.
        """
        if indices is not None:
            start_idx, end_idx = indices[0], indices[-1] + 1
        else:
            start_idx, end_idx = 0, None

        e_lj_perturbed = self._calculate_lj_energy(
            xyz, solute_mask, solvent_mask, start_idx, end_idx, checkpoint=True
        ).float()

        # Calculate the perturbation term
        """
        dG_calc_perturbation = (
            -1
            / self._beta
            * _torch.log(
                _torch.mean(
                    _torch.exp(-self._beta * (e_lj_perturbed - e_lj_unperturbed)),
                    dim=0,
                )
            )
        )

        mapping = {
            i: [i*1000, (i+1)*1000] for i in range(20)
        }
        mol = next(i for i in range(20) if mapping[i][0] <= indices[0] <= mapping[i][1])
        # Calculate the reweighted free energy
        dG_calc_reweighted = dG_calc_perturbation
        # Calculate the loss (minimize the difference between the reweighted free energy and the experimental free energy)
        #loss = self._loss(dG_calc_reweighted, dG_exp)
        """
        return (
            e_lj_perturbed,
            1,
            1,
            # self._get_rmse(dG_calc_reweighted, dG_exp),
            # self._get_max_error(dG_calc_reweighted, dG_exp),
        )
