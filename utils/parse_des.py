"""
        e_int_target,
        atomic_numbers,
        charges_mm,
        xyz_qm,
        xyz_mm,
        solute_mask,
        solvent_mask,
"""


def parse_des_dataset(path: str) -> None:
    """
    Parse the dataset from the DES paper.

    Parameters
    ----------
    path: str
        Path to the dataset.
    """
    import pandas as _pd

    df = _pd.read_csv(path)
