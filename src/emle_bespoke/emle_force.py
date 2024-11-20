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

    Usage:
        emle_module = torch.jit.script(EMLEForce(model, atomic_numbers, charges_mm, solvent_mask, solute_mask))
        emle_force = TorchForce(emle_module)
        system.addForce(emle_force)
    """

    def __init__(
        self, model, atomic_numbers, charges_mm, qm_mask, mm_mask, device, dtype
    ):
        super().__init__()
        self.model = model
        self.atomic_numbers = atomic_numbers
        self.charges_mm = charges_mm
        self.qm_mask = qm_mask
        self.mm_mask = mm_mask

        self.energy_scale = HARTREE_TO_KJ_MOL
        self.lenght_scale = 1.0 / ANGSTROM_TO_NANOMETER
        self.device = device
        self.dtype = dtype

    def forward(self, positions, boxvectors: Optional[_torch.Tensor] = None):
        positions = positions.to(device=self.device, dtype=self.dtype)
        positions *= self.lenght_scale

        # Get the positions of the QM and MM atoms
        xyz_qm = positions[self.qm_mask]
        xyz_mm = positions[self.mm_mask]

        # Calculate the static and induced components of the interaction energy
        e_emle = self.model.forward(
            self.atomic_numbers, self.charges_mm, xyz_qm, xyz_mm
        )
        e_final = (e_emle * self.energy_scale).sum()

        return e_final
