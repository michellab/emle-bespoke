"""Derive Lennard-Jones parameters from ab initio data for the QM7 dataset."""

import argparse as _argparse
import os as _os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as _np
import torch as _torch
from emle.models import EMLE as _EMLE
from openff.toolkit.topology import Molecule as _Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField as _ForceField
from rdkit import Chem as _Chem
from rdkit.Chem import rdDetermineBonds as _rdDetermineBonds

from emle_bespoke._constants import ANGSTROM_TO_BOHR as _ANGSTROM_TO_BOHR
from emle_bespoke._constants import (
    ATOMIC_NUMBERS_TO_SYMBOLS as _ATOMIC_NUMBERS_TO_SYMBOLS,
)
from emle_bespoke._log import _logger
from emle_bespoke._log import log_banner as _log_banner
from emle_bespoke._log import log_cli_args as _log_cli_args
from emle_bespoke.lj import AILennardJones as _AILennardJones


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
    try:
        from scipy.io import loadmat as _loadmat

        qm7_data = _loadmat(input_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file {input_file} not found")
    except Exception as e:
        raise RuntimeError(f"Error loading input file: {e}")

    max_atoms = qm7_data["Z"].shape[1]

    # Convert coordinates to Angstrom
    qm7_data["R"] = qm7_data["R"] / _ANGSTROM_TO_BOHR

    # Add TIP3P water to the dataset (Atoms are ordered as OHH, and units are in Angstrom)
    tip3p = {
        "Z": _np.array([[8.0, 1.0, 1.0] + [0.0] * (max_atoms - 3)]),
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
    mol = _Chem.Mol()
    mol = _Chem.EditableMol(mol)

    # Add atoms
    for atomic_num in z:
        atom = _Chem.Atom(int(atomic_num))
        mol.AddAtom(atom)

    mol = mol.GetMol()

    # Add coordinates
    conf = _Chem.Conformer(len(z))
    for i, (atom, pos) in enumerate(zip(mol.GetAtoms(), xyz)):
        conf.SetAtomPosition(i, pos)
    mol.AddConformer(conf)

    # Determine bonds
    try:
        _rdDetermineBonds.DetermineBonds(mol, charge=0)
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
        - List of SMILES strings
        - List of OpenFF molecules
        - List of indices of molecules processed
    """
    rdkit_mols = []
    smiles_mols = []
    openff_mols = []
    indices_mols = []

    if n_molecules is None:
        n_molecules = len(z)

    for i in range(n_molecules):
        mask = z[i] > 0
        mol_xyz = xyz[i][mask]
        mol_z = z[i][mask]

        try:
            # Create RDKit molecule
            mol = create_rdkit_molecule(mol_z, mol_xyz)

            # Set atom map numbers
            for k, atom in enumerate(mol.GetAtoms()):
                atom.SetAtomMapNum(k + 1)

            # Convert to SMILES and create OpenFF molecule
            smiles = _Chem.MolToSmiles(mol)
            openff_mol = _Molecule.from_smiles(smiles)
            openff_mol.generate_conformers(n_conformers=1)

            rdkit_mols.append(mol)
            smiles_mols.append(smiles)
            openff_mols.append(openff_mol)
            indices_mols.append(i)
        except Exception as e:
            _logger.warning(f"Skipping molecule {i} due to error: {e}")
            continue

    return rdkit_mols, smiles_mols, openff_mols, indices_mols


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
    mols_atom_types = []
    i = 0
    for mol in openff_mols:
        topology = mol.to_topology()
        labels = forcefield.label_molecules(topology)
        atom_types = [val.id for _, val in labels[0]["vdW"].items()]
        mols_atom_types.append(atom_types)
        i += 1
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
    Tuple[Dict[str, float], Dict[str, float]]
        Tuple containing:
        - Dictionary of averaged sigma parameters by atom type
        - Dictionary of averaged epsilon parameters by atom type
    """
    from collections import defaultdict

    sigma_params = defaultdict(list)
    epsilon_params = defaultdict(list)

    for i, mol in enumerate(atom_types):
        for j, atom_type in enumerate(mol):
            sigma_params[atom_type].append(sigma[i][j].item())
            epsilon_params[atom_type].append(epsilon[i][j].item())

    # Average the parameters
    sigma_avg = {k: sum(v) / len(v) for k, v in sigma_params.items()}
    epsilon_avg = {k: sum(v) / len(v) for k, v in epsilon_params.items()}

    return sigma_avg, epsilon_avg


def save_parameters(
    sigma_avg: Dict[str, float],
    epsilon_avg: Dict[str, float],
    output_file: Optional[str] = None,
) -> None:
    """
    Save or print LJ parameters.

    Parameters
    ----------
    sigma_avg : Dict[str, float]
        Dictionary of averaged sigma parameters by atom type
    epsilon_avg : Dict[str, float]
        Dictionary of averaged epsilon parameters by atom type
    output_file : Optional[str]
        Path to output file. If None, parameters are printed to stdout.
    """
    output = []
    output.append("Averaged LJ parameters:")
    output.append("\nSigma (Å):")
    for atom_type, value in sigma_avg.items():
        output.append(f"{atom_type}: {value:.4f}")

    output.append("\nEpsilon (kJ/mol):")
    for atom_type, value in epsilon_avg.items():
        output.append(f"{atom_type}: {value:.4f}")

    if output_file:
        with open(output_file, "w") as f:
            f.write("\n".join(output))
        _logger.info(f"Parameters saved to {output_file}")
    else:
        print("\n".join(output))


def update_forcefield(
    forcefield: _ForceField, sigma: Dict[str, float], epsilon: Dict[str, float]
) -> None:
    """
    Update the force field with the averaged LJ parameters.
    """
    # Update sigma and epsilon values in the force field
    from openff.units import unit as _off_unit

    for param in forcefield.get_parameter_handler("vdW").parameters:
        if param.id not in sigma:
            continue

        param.sigma = sigma[param.id] * _off_unit.angstrom
        param.epsilon = epsilon[param.id] * _off_unit.kilojoules_per_mole

    # Save the updated force field
    forcefield_file = "openff-2.0.0-lj-mbis.offxml"
    forcefield.to_file(forcefield_file)
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

    Raises
    ------
    RuntimeError
        If no molecules are successfully processed
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
        default="qm7.mat",
        help="Input QM7 dataset file",
    )
    data_group.add_argument(
        "-n",
        "--num-molecules",
        type=int,
        default=None,
        help="Number of molecules to process. By default all.",
    )
    data_group.add_argument(
        "-f",
        "--forcefield",
        type=str,
        default="openff-2.0.0.offxml",
        help="Force field file to use",
    )

    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file to save parameters (default: print to stdout)",
    )

    args = parser.parse_args()
    _log_cli_args(args)

    try:
        # Load and preprocess data
        z, xyz = load_qm7_data(args.input)

        # Process molecules
        rdkit_mols, smiles_mols, openff_mols, indices_mols = process_molecules(
            z, xyz, args.num_molecules
        )

        if not rdkit_mols:
            raise RuntimeError("No molecules were successfully processed")

        # Convert to tensors
        xyz_tensor = _torch.tensor(xyz, dtype=_torch.float)[indices_mols]
        z_tensor = _torch.tensor(z, dtype=_torch.long)[indices_mols]
        q_total = _torch.zeros(len(z_tensor), dtype=_torch.float)

        # Compute EMLE properties
        emle = _EMLE()
        s, q_core, q_val, A_thole = emle._emle_base(z_tensor, xyz_tensor, q_total)
        rcubed = -60 * q_val * s**3 / _ANGSTROM_TO_BOHR**3  # Convert to Angstrom^3
        """
        # Extract rcubed values
        import h5py
        rcubed = []
        for i in range(1, len(z) + 1):
            with h5py.File(f'/home/joaomorado/EMLE_WATER_MODEL/mbis_lj/HORTON/{i}.h5', 'r') as f:
                rcubed.append(f['radial_moments'][:, 3] / _ANGSTROM_TO_BOHR**3)
        
        from emle.train._utils import pad_to_max
        rcubed = pad_to_max(rcubed)[indices_mols]
     
        #rcubed = [item / (angstrom ** 3)  for sublist in rcubed for item in sublist]
        #rcubed=s**3
        """
        # Compute LJ parameters
        ai_lj = _AILennardJones()
        alpha = ai_lj.compute_isotropic_polarizabilities(A_thole)
        sigma, epsilon = ai_lj.get_lj_parameters(z_tensor, rcubed, alpha)

        # Get atom types and average parameters
        forcefield = _ForceField(args.forcefield)
        atom_types = get_atom_types(openff_mols, forcefield)
        sigma_avg, epsilon_avg = average_lj_parameters(sigma, epsilon, atom_types)

        # Save or print results
        save_parameters(sigma_avg, epsilon_avg, args.output)

        # Update force field
        forcefield_file = update_forcefield(forcefield, sigma_avg, epsilon_avg)
        _logger.info(f"Updated force field saved to {forcefield_file}")

    except Exception as e:
        _logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
