import io
import tarfile
import torch
import h5py
import numpy as np
import scipy.io

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


def parse_horton_out(f):
    q_shift = np.mean(f["cartesian_multipoles"][:, 0])
    return {
        "s": f["valence_widths"][:],
        "q_core": f["core_charges"][:],
        "q": f["cartesian_multipoles"][:, 0] - q_shift,
        "v": f["radial_moments"][:, 3],
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

    orca_tgz = (
        "/home/joaomorado/repos/emle-engine-kirill/test_folder/QM7_B3LYP_cc-pVTZ.tgz"
    )
    horton_tgz = "/home/joaomorado/repos/emle-engine-kirill/test_folder/QM7_B3LYP_cc-pVTZ_horton.tgz"

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
        q=horton_data["q"],
        alpha=alpha,
        train_mask=torch.arange(n_mols) % 5 == 0,
        device=torch.device("cuda"),
        alpha_mode="reference",
        epochs=100,
        ivm_thr=0.025,
        sigma=1e-4,
    )
