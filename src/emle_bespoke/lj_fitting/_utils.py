"""Miscellaneous utility functions for LJ fitting."""
from openff.toolkit import Molecule as _Molecule
from openff.toolkit.topology import Topology as _Topology


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


def get_water_mapping(topology, res_name="LIG"):
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
            if residue.name == "HOH":
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


def create_dimer_topology(ligand_smiles, water_smiles):
    # Convert the SMILES strings to _Molecule objects
    ligand = _Molecule.from_smiles(ligand_smiles)
    water = _Molecule.from_mapped_smiles(water_smiles)

    # Assign residue names
    for atom in ligand.atoms:
        atom.metadata["residue_name"] = "LIG"

    for atom in water.atoms:
        atom.metadata["residue_name"] = "HOH"

    # Generate conformers
    ligand.generate_conformers(n_conformers=1)
    water.generate_conformers(n_conformers=1)

    # Create the topology
    topology = _Topology.from_molecules([ligand, water])

    return topology
