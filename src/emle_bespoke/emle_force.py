from typing import Optional

import torch as _torch

from ._constants import ANGSTROM_TO_NANOMETER, HARTREE_TO_KJ_MOL


class EMLEForce(_torch.nn.Module):
    """
    OpenMM-Torch force implementation for the EMLE model.

    This class provides a PyTorch-based module to compute interaction energies between solute
    (QM region) and solvent (MM region) atoms using the EMLE model. It integrates with OpenMM
    via the TorchForce class, enabling hybrid quantum mechanics/molecular mechanics (QM/MM)
    simulations with electrostatic embedding.

    Parameters
    ----------
    model: EMLE model
        The instance of the EMLE model.
    atomic_numbers: torch.Tensor(N_ML_ATOMS)
        Atomic numbers of the atoms in the system.
    charges_mm: torch.Tensor(N_MM_ATOMS)
        Charges of the atoms in the MM region.
    qm_mask: torch.Tensor(N_ATOMS)
        Mask to separate the QM and MM atoms.
    cutoff: float or None
        Cutoff distance for the QM/MM interaction in nanometers. If None, the cutoff is disabled.
    device: torch.device
        Device to run the calculations.
    dtype: torch.dtype
        Data type for the calculations.

    Notes
    -----
    Usage:
        emle_module = torch.jit.script(EMLEForce(model, atomic_numbers, charges_mm, solvent_mask, solute_mask))
        emle_force = TorchForce(emle_module)
        system.addForce(emle_force)
    """

    def __init__(
        self, model, atomic_numbers, charges_mm, qm_mask, cutoff, device, dtype
    ):
        super().__init__()
        self.model = model
        self.atomic_numbers = atomic_numbers
        self.charges_mm = charges_mm
        self.qm_mask = qm_mask
        self.mm_mask = ~qm_mask
        self.cutoff = cutoff

        self._energy_scale = HARTREE_TO_KJ_MOL
        self._lenght_scale = 1.0 / ANGSTROM_TO_NANOMETER
        self.device = device
        self.dtype = dtype

    @staticmethod
    def _wrap_positions(
        positions: _torch.Tensor, boxvectors: _torch.Tensor
    ) -> _torch.Tensor:
        """
        Wrap the positions to the main box.

        Parameters
        ----------
        positions: torch.Tensor(NATOMS, 3)
            Atomic positions.
        boxvectors: torch.Tensor(3, 3)
            Box vectors.

        Returns
        -------
        positions: torch.Tensor(NATOMS, 3)
            Wrapped atomic positions.
        """
        for i in range(3):
            positions = positions - _torch.outer(
                _torch.floor(positions[:, i] / boxvectors[i, i]), boxvectors[i]
            )
        return positions

    @staticmethod
    def _center_molecule(
        positions: _torch.Tensor,
        boxvectors: _torch.Tensor,
        molecule_mask: _torch.Tensor,
    ) -> _torch.Tensor:
        """
        Center the molecule in the box (Lx/2, Ly/2, Lz/2).

        Parameters
        ----------
        positions: torch.Tensor(NATOMS, 3)
            Atomic positions.
        boxvectors: torch.Tensor(3, 3)
            Box vectors.
        molecule_mask: torch.Tensor(NATOMS)
            Molecule mask.

        Returns
        -------
        positions: torch.Tensor(NATOMS, 3)
            Centered atomic positions.
        """
        mol_positions = positions[molecule_mask]
        com = mol_positions.mean(dim=0)
        box_center = 0.5 * _torch.diag(boxvectors)
        translation_vector = box_center - com
        positions += translation_vector

        # Wrap the positions to the main box
        positions = EMLEForce._wrap_positions(positions, boxvectors)

        return positions

    @staticmethod
    def _distance_to_molecule(
        positions: _torch.Tensor,
        boxvectors: _torch.Tensor,
        molecule_mask: _torch.Tensor,
    ) -> _torch.Tensor:
        """
        Calculate the R matrix for the molecule.

        Parameters
        ----------
        positions: torch.Tensor(NATOMS, 3)
            Atomic positions.
        boxvectors: torch.Tensor(3, 3)
            Box vectors.
        molecule_mask: torch.Tensor(N_ATOMS)
            Molecule mask.

        Returns
        -------
        R: torch.Tensor(NATOMS, NATOMS)
            Distance matrix.
        """
        ml_positions = positions[molecule_mask]
        mm_positions = positions[~molecule_mask]
        R = _torch.cdist(ml_positions, mm_positions)
        return R

    def forward(self, positions, boxvectors: Optional[_torch.Tensor] = None):
        positions = positions.to(device=self.device, dtype=self.dtype)

        if boxvectors is not None or self._cutoff is not None:
            boxvectors = boxvectors.to(device=self.device, dtype=self.dtype)
            positions = self._center_molecule(positions, boxvectors, self.qm_mask)
            R = self._distance_to_molecule(positions, boxvectors, self.qm_mask)
            R_cutoff = _torch.any(R < self._cutoff, dim=0)
        else:
            R_cutoff = slice(None)

        # Get the positions of the QM and MM atoms
        xyz_qm = positions[self.qm_mask] * self._lenght_scale
        xyz_mm = positions[self.mm_mask][R_cutoff] * self._lenght_scale

        # Get the charges of the MM atoms
        charges_mm = self.charges_mm[self.mm_mask][R_cutoff]

        # Calculate the static and induced components of the interaction energy
        e_emle = self.model.forward(self.atomic_numbers, charges_mm, xyz_qm, xyz_mm)
        e_final = (e_emle * self._energy_scale).sum()

        return e_final
