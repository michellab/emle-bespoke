"""Patched EMLE model with alpha*static and beta*induced energy components."""
import torch as _torch
from emle.models import EMLE as _EMLE


class EMLEPatched(_EMLE):
    """EMLE model with alpha*static and beta*induced energy components."""

    def __init__(self, alpha_static=1.0, beta_induced=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha_static = _torch.nn.Parameter(_torch.tensor(alpha_static))
        self.beta_induced = _torch.nn.Parameter(_torch.tensor(beta_induced))

    def forward(self, atomic_numbers, charges_mm, xyz_qm, xyz_mm):
        """
        Computes the patched alpha*static and beta*induced EMLE energy components.

        Parameters
        ----------

        atomic_numbers: torch.Tensor (N_QM_ATOMS,)
            Atomic numbers of QM atoms.

        charges_mm: torch.Tensor (max_mm_atoms,)
            MM point charges in atomic units.

        xyz_qm: torch.Tensor (N_QM_ATOMS, 3)
            Positions of QM atoms in Angstrom.

        xyz_mm: torch.Tensor (N_MM_ATOMS, 3)
            Positions of MM atoms in Angstrom.

        Returns
        -------

        result: torch.Tensor (2,)
            The static and induced EMLE energy components in Hartree.
        """
        e_static, e_ind = super().forward(atomic_numbers, charges_mm, xyz_qm, xyz_mm)

        return self.alpha_static * e_static, self.beta_induced * e_ind
