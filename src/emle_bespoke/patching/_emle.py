"""Patched EMLE model with alpha*static and beta*induced energy components."""
from emle.models import EMLE as _EMLE
import torch as _torch

class EMLEPatched(_EMLE):
    """EMLE model with alpha*static and beta*induced energy components."""
    def __init__(self, alpha=1.0, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._static_alpha = _torch.nn.Parameter(_torch.tensor(alpha))
        self._induced_beta = _torch.nn.Parameter(_torch.tensor(beta))

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

        return self._static_alpha * e_static + self._induced_beta * e_ind

