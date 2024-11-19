"""Miscellaneous utility functions for LJ fitting."""
from typing import Any, List, Tuple, Union


def get_unique_atoms(topology, res_name="LIG"):
    """
    Get indices of atoms with unique environments in a topology.

    Parameters
    ----------
    topology : simtk.openmm.app.Topology
        OpenMM Topology object
    res_name : str
        Name of residue to consider.

    Returns
    -------
    unique_atoms : list of int
        List of atom indices with unique chemical environments.
    """
    from collections import defaultdict

    chemical_envs = defaultdict(list)

    # Get chemical environments for each atom in the topology
    # Creates a dictionary with atom indices as keys and lists of chemical environments as values
    # Chemical environments are represented as lists of [sorted[element 1, element 2], bond type, bond order]
    for chain in topology.chains():
        for residue in chain.residues():
            if residue.name == res_name:
                for bond in residue.internal_bonds():
                    atom1 = bond.atom1
                    atom2 = bond.atom2

                    # Sort atoms to ensure consistent representation of bond
                    bonded_atoms = sorted([atom1.element.symbol, atom2.element.symbol])
                    bond_info = [bonded_atoms, bond.type, bond.order]
                    chemical_envs[atom1.index].append(bond_info)
                    chemical_envs[atom2.index].append(bond_info)

    # Identify unique environments
    chemical_envs = dict(chemical_envs)
    unique_atoms, seen_envs = [], []

    for atom, env in chemical_envs.items():
        env = sorted(env)

        if env not in seen_envs:
            unique_atoms.append(atom)
            seen_envs.append(env)
    return unique_atoms


def get_water_mapping(topology, res_name="HOH"):
    """
    Get atom indices for O, H1, and H2 atoms in a water molecule.

    Parameters
    ----------
    topology : simtk.openmm.app.Topology
        OpenMM Topology object
    res_name : str
        Name of residue to consider.

    Returns
    -------
    water_mapping : dict
        Dictionary with atom indices for O, H1, and H2 atoms in water.
    """
    water_mapping = {}
    for chain in topology.chains():
        for residue in chain.residues():
            if residue.name == res_name:
                for atom in residue.atoms():
                    if "O" in atom.name:
                        water_mapping["O"] = atom.index
                    elif "H1" in atom.name:
                        water_mapping["H1"] = atom.index
                    elif "H2" in atom.name:
                        water_mapping["H2"] = atom.index
                    else:
                        raise ValueError(
                            f"Invalid atom name in water molecule: {atom.name}"
                        )
    return water_mapping


def unique_with_delta(lst: list, delta: float):
    """
    Get unique values in a list with a specified tolerance.

    Parameters
    ----------
    lst : list
        List of values to check for uniqueness.
    delta : float
        Tolerance for uniqueness.

    Returns
    -------
    unique_values : list
        List of unique values.
    unique_idx : list
        List of indices of unique values.
    """
    unique_values = []
    unique_idx = []
    for i, value in enumerate(lst):
        if not any(abs(value - unique) <= delta for unique in unique_values):
            unique_values.append(value)
            unique_idx.append(i)
    return unique_values, unique_idx


def sort_two_lists(list1: List[Any], list2: List[Any]) -> Tuple[List[Any], List[Any]]:
    """
    Sort two lists based on the values in the first list.

    Parameters
    ----------
    list1 : list
        List of values to sort by.
    list2 : list
        List to sort.

    Returns
    -------
    sorted_list1 : list
        Sorted list1.
    """
    paired_lists = list(zip(list1, list2))
    sorted_pairs = sorted(paired_lists, key=lambda x: x[0])
    sorted_list1, sorted_list2 = zip(*sorted_pairs)
    return list(sorted_list1), list(sorted_list2)
