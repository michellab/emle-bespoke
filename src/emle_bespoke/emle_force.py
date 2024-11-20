import torch as _torch
from ._constants import HARTREE_TO_KJ_MOL, ANGSTROM_TO_NANOMETER
from typing import Optional

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
    def __init__(self, model, atomic_numbers, charges_mm, solvent_mask, solute_mask):
        super().__init__()
        self.model = model
        self.atomic_numbers = atomic_numbers
        self.charges_mm = charges_mm
        self.solvent_mask = solvent_mask
        self.solute_mask = solute_mask

        self.energyScale = HARTREE_TO_KJ_MOL
        self.lenght_scale = 1.0 / ANGSTROM_TO_NANOMETER

    def forward(self, positions, boxvectors: Optional[_torch.Tensor] = None):
        positions = positions.to(_torch.float32)

        # Get the positions of the QM and MM atoms
        xyz_qm = positions[:, self.solute_mask]
        xyz_mm = positions[:, self.solvent_mask]

        # Calculate the static and induced components of the interaction energy
        e_static, e_ind = self.model(self.atomic_numbers, self.charges_mm, xyz_qm, xyz_mm)

        return self.energy_scale * (e_static + e_ind)
    