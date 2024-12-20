import torch as _torch
from torch.utils.data import Dataset as _Dataset


class MolecularDataset(_Dataset):
    def __init__(
        self,
        xyz_qm,
        xyz_mm,
        xyz,
        atomic_numbers,
        charges_mm,
        e_int_target,
        solute_mask,
        solvent_mask,
    ):
        self.data = {
            "xyz_qm": _torch.tensor(xyz_qm, dtype=_torch.float64),
            "xyz_mm": _torch.tensor(xyz_mm, dtype=_torch.float64),
            "xyz": _torch.tensor(xyz, dtype=_torch.float64),
            "atomic_numbers": _torch.tensor(atomic_numbers, dtype=_torch.long),
            "charges_mm": _torch.tensor(charges_mm, dtype=_torch.float64),
            "e_int_target": _torch.tensor(e_int_target, dtype=_torch.float64),
            "solute_mask": _torch.tensor(solute_mask, dtype=_torch.float64),
            "solvent_mask": _torch.tensor(solvent_mask, dtype=_torch.float64),
            "indices": _torch.arange(len(e_int_target), dtype=_torch.long),
        }

        # Send to GPU if available
        if _torch.cuda.is_available():
            for key in self.data:
                self.data[key] = self.data[key].cuda()

    def __len__(self):
        return len(self.data["e_int_target"])

    def __getitem__(self, indices):
        # Use slicing directly without creating a new dictionary
        return {key: self.data[key][indices] for key in self.data}
