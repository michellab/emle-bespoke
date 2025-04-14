"""Derive Lennard-Jones parameters from ab initio/MBIS data."""

import argparse as _argparse
import os as _os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as _np
import torch as _torch
from emle.models import EMLE as _EMLE
from emle_bespoke._constants import ANGSTROM_TO_BOHR as _ANGSTROM_TO_BOHR

from emle_bespoke._log import _logger
from emle_bespoke._log import log_banner as _log_banner
from emle_bespoke._log import log_cli_args as _log_cli_args
from emle_bespoke.lj import AILennardJones as _AILennardJones
from openff.toolkit.topology import Molecule as _Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField as _ForceField
from rdkit import Chem as _Chem
from rdkit.Chem import rdDetermineBonds as _rdDetermineBonds


def load_data(input_file: str) -> Tuple[_np.ndarray, _np.ndarray]:
    """
    Load reference data.
    """
    import pickle as _pkl

    _logger.info(f"Loading data from {input_file}")
    with open(input_file, "rb") as f:
        ref_data = _pkl.load(f)
    return ref_data["z"], ref_data["xyz_qm"]


def load_qm7_data(input_file: str) -> Tuple[_np.ndarray, _np.ndarray]:
    """
    Load and preprocess QM7 dataset.

    Notes
    -----
    Adds TIP3P water to the dataset.

    Parameters
    ----------
    input_file : str
        Path to the QM7 dataset file (.mat format)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing:
        - Atomic numbers array (n_molecules, max_atoms)
        - Coordinates array (n_molecules, max_atoms, 3) in Angstrom
    """
    _logger.info(f"Loading QM7 data from {input_file}")
    try:
        from scipy.io import loadmat as _loadmat

        qm7_data = _loadmat(input_file)
        _logger.info(
            f"Successfully loaded QM7 data with {qm7_data['Z'].shape[0]} molecules"
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file {input_file} not found")
    except Exception as e:
        raise RuntimeError(f"Error loading input file: {e}")

    max_atoms = qm7_data["Z"].shape[1]
    _logger.debug(f"Maximum number of atoms per molecule: {max_atoms}")

    # Convert coordinates to Angstrom
    qm7_data["R"] = qm7_data["R"] / _ANGSTROM_TO_BOHR
    _logger.debug("Converted coordinates from Bohr to Angstrom")

    # Add TIP3P water to the dataset (Atoms are ordered as OHH, and units are in Angstrom)
    _logger.info("Adding TIP3P water to the dataset")
    tip3p = {
        "Z": _np.array([[8, 1, 1] + [0] * (max_atoms - 3)]),
        "R": _np.array(
            [
                [
                    [0.00000, -0.06556, 0.00000],
                    [0.75695, 0.52032, 0.00000],
                    [-0.75695, 0.52032, 0.00000],
                ]
                + [[0.0, 0.0, 0.0]] * (max_atoms - 3)
            ]
        ),
    }
    qm7_data["Z"] = _np.vstack((tip3p["Z"], qm7_data["Z"]))
    qm7_data["R"] = _np.vstack((tip3p["R"], qm7_data["R"]))
    _logger.info(f"Final dataset size: {qm7_data['Z'].shape[0]} molecules")

    return qm7_data["Z"], qm7_data["R"]


def create_rdkit_molecule(z: _np.ndarray, xyz: _np.ndarray) -> _Chem.Mol:
    """
    Create an RDKit molecule from atomic numbers and coordinates.

    Parameters
    ----------
    z : np.ndarray
        Array of atomic numbers
    xyz : np.ndarray
        Array of atomic coordinates in Angstrom

    Returns
    -------
    Chem.Mol
        RDKit molecule with atoms and coordinates
    """
    _logger.debug(f"Creating RDKit molecule with {len(z)} atoms")
    mol = _Chem.Mol()
    mol = _Chem.EditableMol(mol)

    # Add atoms
    for atomic_num in z:
        atom = _Chem.Atom(int(atomic_num))
        mol.AddAtom(atom)

    mol = mol.GetMol()
    # Iterate over atoms and set atom map numbers
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i + 1)

    # Add coordinates
    conf = _Chem.Conformer(len(z))
    for i, (atom, pos) in enumerate(zip(mol.GetAtoms(), xyz)):
        conf.SetAtomPosition(i, pos)
    mol.AddConformer(conf)
    _logger.debug("Added coordinates to molecule")

    # Determine bonds
    try:
        _rdDetermineBonds.DetermineBonds(mol, charge=0)
        _logger.debug(f"Determined {mol.GetNumBonds()} bonds")
    except Exception as e:
        raise ValueError(f"Failed to determine bonds: {e}")

    return mol


def process_molecules(
    z: _np.ndarray, xyz: _np.ndarray, n_molecules: Optional[int] = None
) -> Tuple[List[_Chem.Mol], List[str], List[_Molecule]]:
    """
    Process molecules and create RDKit and OpenFF representations.

    Parameters
    ----------
    z : np.ndarray
        Array of atomic numbers for all molecules
    xyz : np.ndarray
        Array of atomic coordinates for all molecules in Angstrom
    n_molecules : int, optional
        Number of molecules to process, by default all.

    Returns
    -------
    Tuple[List[Chem.Mol], List[str], List[Molecule], List[int]]
        Tuple containing:
        - List of RDKit molecules
        - List of OpenFF molecules
        - List of indices of molecules processed
    """
    _logger.info(f"Processing molecules (n_molecules={n_molecules})")
    rdkit_mols = []
    openff_mols = []
    indices_mols = []

    if n_molecules is None:
        n_molecules = len(z)
    _logger.info(f"Will process {n_molecules} molecules")

    for i in range(n_molecules):
        mask = z[i] > 0
        mol_xyz = xyz[i][mask]
        mol_z = z[i][mask]
        _logger.debug(
            f"Processing molecule {i + 1}/{n_molecules} with {len(mol_z)} atoms"
        )

        try:
            # Create RDKit molecule
            mol = create_rdkit_molecule(mol_z, mol_xyz)

            # Set atom map numbers
            for k, atom in enumerate(mol.GetAtoms()):
                atom.SetAtomMapNum(k + 1)

            # Create OpenFF molecule (mapped SMILES is essential, otherwise ordering will be wrong!)
            smiles = _Chem.MolToSmiles(mol)
            openff_mol = _Molecule.from_mapped_smiles(smiles)
            openff_mol.generate_conformers(n_conformers=1)

            rdkit_mols.append(mol)
            openff_mols.append(openff_mol)
            indices_mols.append(i)
            _logger.debug(
                f"Successfully processed molecule {i + 1} with smiles {smiles}"
            )
        except Exception as e:
            _logger.warning(f"Skipping molecule {i} due to error: {e}")
            continue

    _logger.info(
        f"Successfully processed {len(rdkit_mols)} out of {n_molecules} molecules"
    )
    return rdkit_mols, openff_mols, indices_mols


def get_atom_types(
    openff_mols: List[_Molecule], forcefield: _ForceField
) -> List[List[str]]:
    """
    Get atom types for each molecule using the forcefield.

    Parameters
    ----------
    openff_mols : List[Molecule]
        List of OpenFF molecules
    forcefield : ForceField
        OpenFF force field to use for atom typing

    Returns
    -------
    List[List[str]]
        List of atom types for each molecule
    """
    _logger.info("Getting atom types for molecules")
    mols_atom_types = []
    i = 0
    for mol in openff_mols:
        topology = mol.to_topology()
        labels = forcefield.label_molecules(topology)
        atom_types = [val.id for _, val in labels[0]["vdW"].items()]
        mols_atom_types.append(atom_types)
        i += 1
        if i % 100 == 0:
            _logger.debug(f"Processed {i} molecules for atom typing")
    _logger.info(f"Completed atom typing for {len(mols_atom_types)} molecules")
    return mols_atom_types


def average_lj_parameters(
    sigma: _torch.Tensor, epsilon: _torch.Tensor, atom_types: List[List[str]]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Average LJ parameters for each atom type.

    Parameters
    ----------
    sigma : torch.Tensor
        LJ sigma parameters for all atoms
    epsilon : torch.Tensor
        LJ epsilon parameters for all atoms
    atom_types : List[List[str]]
        List of atom types for each molecule

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, float], Dict[str, List[float]], Dict[str, List[float]]]
        Tuple containing:
        - Dictionary of averaged sigma parameters by atom type
        - Dictionary of averaged epsilon parameters by atom type
        - Dictionary of sigma parameters by atom type
        - Dictionary of epsilon parameters by atom type
    """
    _logger.info("Averaging LJ parameters by atom type")
    from collections import defaultdict

    sigma_params = defaultdict(list)
    epsilon_params = defaultdict(list)

    for i, mol in enumerate(atom_types):
        for j, atom_type in enumerate(mol):
            sigma_params[atom_type].append(sigma[i][j].item())
            epsilon_params[atom_type].append(epsilon[i][j].item())
        if i % 100 == 0:
            _logger.debug(f"Processed {i} molecules for parameter averaging")

    # Average the parameters
    sigma_avg = {k: sum(v) / len(v) for k, v in sigma_params.items()}
    epsilon_avg = {k: sum(v) / len(v) for k, v in epsilon_params.items()}
    _logger.info(f"Averaged parameters for {len(sigma_avg)} unique atom types")

    return sigma_avg, epsilon_avg, sigma_params, epsilon_params


def plot_parameters_distributions(
    sigma_params: Dict[str, List[float]],
    epsilon_params: Dict[str, List[float]],
    forcefield: _ForceField,
    output_dir: str,
) -> None:
    """
    Plot distributions of Lennard-Jones parameters (sigma and epsilon) using seaborn.

    Parameters
    ----------
    sigma_params : Dict[str, List[float]]
        Dictionary of sigma parameters by atom type.
    epsilon_params : Dict[str, List[float]]
        Dictionary of epsilon parameters by atom type.
    forcefield : _ForceField
        The force field containing parameter information.
    output_dir : str
        Directory to save the plot files.
    """
    import matplotlib.pyplot as _plt
    import seaborn as sns
    from openff.units import unit as _offunit

    _logger.info("Plotting LJ parameters distributions")

    vdw_handler = forcefield.get_parameter_handler("vdW")

    sns.set(style="whitegrid", context="paper")

    # Plot epsilon parameters
    n_col = 4
    n_row = (_np.ceil(len(epsilon_params) / n_col)).astype(int)
    fig, axs = _plt.subplots(n_row, n_col, figsize=(15, 4 * n_row))
    axs = axs.flatten()

    for i, (k, v) in enumerate(epsilon_params.items()):
        ax = axs[i]
        sns.histplot(v, bins=100, kde=True, ax=ax)
        ax.axvline(
            vdw_handler.get_parameter({"id": k})[0]
            .epsilon.to(_offunit.kilojoules_per_mole)
            .magnitude,
            color="red",
            linestyle="--",
            label="OpenFF",
        )
        ax.set_title(f"Atom type: {k}")
        ax.set_xlabel(r"$\epsilon$ (kJ/mol)")
        ax.set_ylabel("Frequency")
        ax.legend()

    # Remove empty subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    _plt.tight_layout()
    epsilon_plot_path = _os.path.join(output_dir, "epsilon_distribution.png")
    _plt.savefig(epsilon_plot_path)
    _logger.info(f"Epsilon distribution plot saved to {epsilon_plot_path}")

    # Plot sigma parameters
    fig, axs = _plt.subplots(n_row, n_col, figsize=(15, 4 * n_row))
    axs = axs.flatten()

    for i, (k, v) in enumerate(sigma_params.items()):
        ax = axs[i]
        sns.histplot(v, bins=100, kde=True, ax=ax)
        ax.axvline(
            vdw_handler.get_parameter({"id": k})[0]
            .sigma.to(_offunit.angstrom)
            .magnitude,
            color="red",
            linestyle="--",
            label="OpenFF",
        )
        ax.set_title(f"Atom type: {k}")
        ax.set_xlabel(r"$\sigma$ (Å)")
        ax.set_ylabel("Frequency")
        ax.legend()

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    _plt.tight_layout()
    sigma_plot_path = _os.path.join(output_dir, "sigma_distribution.png")
    _plt.savefig(sigma_plot_path)
    _logger.info(f"Sigma distribution plot saved to {sigma_plot_path}")


def save_parameters(
    sigma_avg: Dict[str, float],
    epsilon_avg: Dict[str, float],
    output_dir: str,
    prefix: str,
) -> None:
    """
    Save or print LJ parameters.

    Parameters
    ----------
    sigma_avg : Dict[str, float]
        Dictionary of averaged sigma parameters by atom type
    epsilon_avg : Dict[str, float]
        Dictionary of averaged epsilon parameters by atom type
    output_dir : str
        Directory to save the parameter file.
    prefix : str
        Prefix for the output parameter file name.
    """
    _logger.info("Saving LJ parameters")
    output = []
    output.append("Averaged LJ parameters:")
    output.append("\nSigma (Å):")
    for atom_type, value in sigma_avg.items():
        output.append(f"{atom_type}: {value:.4f}")

    output.append("\nEpsilon (kJ/mol):")
    for atom_type, value in epsilon_avg.items():
        output.append(f"{atom_type}: {value:.4f}")

    output_file_path = _os.path.join(output_dir, f"{prefix}_averaged_lj_parameters.txt")
    with open(output_file_path, "w") as f:
        f.write("\n".join(output))
    _logger.info(f"Averaged parameters saved to {output_file_path}")


def update_forcefield(
    forcefield: _ForceField,
    sigma: Dict[str, float],
    epsilon: Dict[str, float],
    output_dir: str,
    prefix: str,
) -> None:
    """
    Update the force field with the averaged LJ parameters.

    Parameters
    ----------
    forcefield : _ForceField
        The force field object to update.
    sigma : Dict[str, float]
        Dictionary of averaged sigma parameters by atom type.
    epsilon : Dict[str, float]
        Dictionary of averaged epsilon parameters by atom type.
    output_dir : str
        Directory to save the updated force field file.
    prefix : str
        Prefix for the output force field file name.
    """
    _logger.info("Updating force field with new LJ parameters")
    # Update sigma and epsilon values in the force field
    from openff.units import unit as _off_unit

    updated_params = 0
    for param in forcefield.get_parameter_handler("vdW").parameters:
        if param.id not in sigma:
            continue

        param.sigma = sigma[param.id] * _off_unit.angstrom
        param.epsilon = epsilon[param.id] * _off_unit.kilojoules_per_mole
        updated_params += 1

    # Save the updated force field
    forcefield_file = _os.path.join(output_dir, f"{prefix}_updated.offxml")
    forcefield.to_file(forcefield_file)
    _logger.info(f"Updated {updated_params} parameters in force field")
    return forcefield_file


def main() -> None:
    """
    Main function to derive Lennard-Jones parameters from ab initio data.

    This function:
    1. Parses command line arguments
    2. Loads and preprocesses the QM7 dataset
    3. Processes molecules to create RDKit and OpenFF representations
    4. Computes EMLE properties and LJ parameters
    5. Averages parameters by atom type
    6. Saves or prints the results
    """
    _log_banner()

    parser = _argparse.ArgumentParser(
        description="Derive Lennard-Jones parameters from ab initio data for the QM7 dataset.",
        formatter_class=_argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "-i",
        "--input",
        type=str,
        default="qm7",
        help="Input file with reference data. If not provided, the QM7 dataset will be used and TIP3P water will be added.",
    )
    data_group.add_argument(
        "-n",
        "--num-molecules",
        type=int,
        default=None,
        help="Number of molecules to process. By default all.",
    )
    data_group.add_argument(
        "--forcefield",
        type=str,
        default="openff-2.0.0.offxml",
        help="Force field file to use",
    )

    # EMLE configuration
    emle_group = parser.add_argument_group("EMLE Configuration")
    emle_group.add_argument(
        "--emle-model",
        type=str,
        default=None,
        help="Path to EMLE model. If not provided, the default model will be used.",
    )
    emle_group.add_argument(
        "--alpha-mode",
        type=str,
        default="species",
        help="Alpha mode to use",
    )

    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "-o",
        "--output-prefix",
        type=str,
        default="mbis_lj",
        help="Directory name to save outputs and prefix for output filenames within that directory (default: mbis_lj).",
    )
    output_group.add_argument("--plot", action="store_true", help="Produce plots.")

    args = parser.parse_args()
    _log_cli_args(args)

    from emle.train._utils import pad_to_max

    try:
        # Create output directory
        output_dir = args.output_prefix
        _os.makedirs(output_dir, exist_ok=True)
        _logger.info(f"Output directory set to: {output_dir}")

        # Load and preprocess data
        if args.input == "qm7":
            _logger.info("Loading and preprocessing QM7 data")
            z, xyz = load_qm7_data(args.input)
        else:
            _logger.info("Loading and preprocessing reference data")
            z, xyz = load_data(args.input)

        # Process molecules
        _logger.info("Processing molecules")
        rdkit_mols, openff_mols, indices_mols = process_molecules(
            z, xyz, args.num_molecules
        )

        if not rdkit_mols:
            raise RuntimeError("No molecules were successfully processed")

        # Convert to tensors
        _logger.info("Converting data to tensors")
        z_tensor = pad_to_max(z).to(_torch.long)[indices_mols]
        xyz_tensor = pad_to_max(xyz).to(_torch.float)[indices_mols]
        q_total = _torch.zeros(len(z_tensor), dtype=_torch.float)

        # Compute EMLE properties
        _logger.info("Computing EMLE properties")
        emle = _EMLE(
            model=args.emle_model,
            alpha_mode=args.alpha_mode,
        )
        s, q_core, q_val, A_thole = emle._emle_base(z_tensor, xyz_tensor, q_total)
        rcubed = -60 * q_val * s**3 / _ANGSTROM_TO_BOHR**3  # Convert to Angstrom^3

        # Extract rcubed values
        """
        _logger.info("Extracting rcubed values")
        import h5py
        from emle.train._utils import pad_to_max
        rcubed = []
        for i in range(len(z)):
            with h5py.File(f'/home/joaomorado/EMLE_WATER_MODEL/mbis_lj/HORTON/{i}.h5', 'r') as f:
                rcubed.append(f['radial_moments'][:, 3] / _ANGSTROM_TO_BOHR**3)
        rcubed = pad_to_max(rcubed)[indices_mols]
        """
        # Compute LJ parameters
        _logger.info("Computing LJ parameters")
        ai_lj = _AILennardJones()
        alpha = ai_lj.compute_isotropic_polarizabilities(A_thole)
        sigma, epsilon = ai_lj.get_lj_parameters(z_tensor, rcubed, alpha)

        # Get atom types and average parameters
        _logger.info("Getting atom types and averaging parameters")
        forcefield = _ForceField(args.forcefield)
        atom_types = get_atom_types(openff_mols, forcefield)
        sigma_avg, epsilon_avg, sigma_params, epsilon_params = average_lj_parameters(
            sigma, epsilon, atom_types
        )

        # Plot parameters distributions
        if args.plot:
            plot_parameters_distributions(
                sigma_params, epsilon_params, forcefield, output_dir
            )

        # Save or print results
        _logger.info("Saving results")
        save_parameters(sigma_avg, epsilon_avg, output_dir, args.output_prefix)

        # Update force field
        _logger.info("Updating force field")
        forcefield_file = update_forcefield(
            forcefield, sigma_avg, epsilon_avg, output_dir, args.output_prefix
        )
        _logger.info(f"Updated force field saved to {forcefield_file}")
        _logger.info("LJ parameter derivation completed successfully")

    except Exception as e:
        _logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
