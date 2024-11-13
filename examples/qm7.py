"""
### Data Required for This Example:
1. Download the following files:
    - QM7_B3LYP_cc-pVTZ_horton.tgz
      wget https://zenodo.org/record/7051785/files/QM7_B3LYP_cc-pVTZ_horton.tgz
    - QM7_B3LYP_cc-pVTZ.tgz
      wget https://zenodo.org/record/7051785/files/QM7_B3LYP_cc-pVTZ.tgz
    - qm7.mat
      wget http://quantum-machine.org/data/qm7.mat

### References:
- Github: https://github.com/emedio/embedding
- Dataset: https://zenodo.org/records/7051785
"""
import io
import tarfile

import h5py
import numpy as np
import scipy.io
import torch
from emle.models import EMLEBase
from emle.train import EMLETrainer


def seek_to(f, value):
    while next(f) != value:
        pass


def skip(f, n):
    for _ in range(n):
        next(f)


def parse_orca_out(f):
    seek_to(f, b"THE POLARIZABILITY TENSOR\n")
    skip(f, 3)
    alpha = [list(map(float, next(f).split())) for _ in range(3)]
    return np.array(alpha)


def get_alpha_by_id(id_, orca_tgz):
    return parse_orca_out(orca_tgz.extractfile(f"{id_}.out"))


_HORTON_KEYS = (
    "cartesian_multipoles",
    "core_charges",
    "valence_charges",
    "valence_widths",
)


def parse_horton_out(f):
    data = {key: f[key][:] for key in _HORTON_KEYS}
    q = data["core_charges"] + data["valence_charges"]
    q_shift = np.sum(np.round(q) - q) / len(q)
    return {
        "s": data["valence_widths"],
        "q_core": data["core_charges"],
        "q_val": data["valence_charges"] + q_shift,
        "mu": data["cartesian_multipoles"][:, 1:4],
    }


def get_horton_data_by_file(t, tgz):
    f = io.BytesIO(horton_tgz.extractfile(t).read())
    return parse_horton_out(h5py.File(f, "r"))


def pad_to_shape(array, shape, value=0):
    pad = [(0, n_max - n) for n_max, n in zip(shape, array.shape)]
    return np.pad(array, pad, constant_values=value)


def pad_to_max(arrays, value=0):
    # Takes arrays with different shapes, but same number of dimensions
    # and pads them to the size of the largest array along each axis
    shape = np.max([_.shape for _ in arrays], axis=0)
    return np.array([pad_to_shape(_, shape, value) for _ in arrays])


def zip_list_of_dicts(l):
    # Converts list of dicts to dict of lists
    return {key: [_[key] for _ in l] for key in l[0].keys()}


if __name__ == "__main__":
    READ_MAT = True

    orca_tgz = "QM7_B3LYP_cc-pVTZ.tgz"
    horton_tgz = "QM7_B3LYP_cc-pVTZ_horton.tgz"

    # Load QM7 data
    qm7_data = scipy.io.loadmat("qm7.mat", squeeze_me=True)

    # Get the coordinates and atomic numbers
    BOHR_TO_ANGSTROM = 0.52917721067
    xyz = qm7_data["R"] * BOHR_TO_ANGSTROM
    z = qm7_data["Z"]
    n_mols = len(z)

    if READ_MAT:
        alpha = scipy.io.loadmat("alpha.mat", squeeze_me=True)["alpha"]
        horton_data = scipy.io.loadmat("horton.mat", squeeze_me=True)
    else:
        orca_tgz = tarfile.open(orca_tgz, mode="r:gz")
        alpha = np.array([get_alpha_by_id(i + 1, orca_tgz) for i in range(n_mols)])
        scipy.io.savemat("alpha.mat", {"alpha": alpha})

        horton_tgz = tarfile.open(horton_tgz, mode="r|gz")
        horton_dicts = [None for _ in range(n_mols)]
        for t in horton_tgz:
            i = int(t.name.split(".")[0]) - 1
            horton_dicts[i] = get_horton_data_by_file(t, horton_tgz)

        # Store each MBIS property as an np.array with shape (n_mols, n_atoms)
        # Since molecules have different n_mols, pad with zeros when needed
        horton_data = {
            k: pad_to_max(v) for k, v in zip_list_of_dicts(horton_dicts).items()
        }
        scipy.io.savemat("horton.mat", horton_data)

    trainer = EMLETrainer(EMLEBase)
    trainer.train(
        z=z,
        xyz=xyz,
        s=horton_data["s"],
        q_core=horton_data["q_core"],
        q_val=horton_data["q_val"],
        alpha=alpha,
        train_mask=torch.arange(n_mols) % 2 == 0,
        device=torch.device("cuda"),
        epochs=250,
    )
