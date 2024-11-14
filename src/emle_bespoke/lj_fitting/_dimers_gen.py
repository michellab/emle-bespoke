import numpy as _np
import openmm.unit as _unit

from ..cli._sample_train import create_off_topology

"""
1. Generate Initial Dimers
   - For each atom in the target molecule, generate N initial configurations of dimers for both orientations:
     - Water-oxygen facing the molecule.
     - Water-hydrogen facing the molecule.
   - By default, N = 50 for each orientation.

2. Set Up Initial 3D Mesh Grid
   - Around each atom of interest in the molecule, generate an initial 3D mesh grid.
   - Calculate an ideal pairwise distance between the water probe atom and the target molecule atom based on the van der Waals (vdW) radii using a combining rule.

3. Probe Placement on 3D Grid
   - The water probe molecule searches along the 3D mesh grid to find points that:
     - Are within the target distance of the probed atom of interest.
     - Maintain a reasonable distance from other atoms in the molecule to avoid steric clashes.

4. Selection of Points for Minimization
   - Select the first N points from this grid to undergo a minimization process (using Tinker minimization).

5. Tinker Minimization
   - Minimize each of the selected N dimer configurations to identify the most favorable (lowest energy) minimized dimer structure.

6. Selection of Favorable Dimer for QM Computations
   - Choose the minimized dimer with the most favorable energy as the starting point for further quantum mechanical (QM) computations.

7. Translation of Minimized Structure
   - Translate the minimized dimer structure along the direction of the pairwise distance by increments to sample points along the potential energy surface.
   - Default sampling points are at 80%, 90%, 100%, 110%, and 120% of the minimized pairwise distance.

8. Energy Computations Using Tinker ANALYZE
   - Compute the total energy for each of these translated structures using Tinker's ANALYZE tool.

9. Single Point Energy Computations
   - Perform single point energy calculations for each translated structure using the ωB97X-D/aug-cc-pVDZ level of theory.

My protocol:

1. Identify atoms with unique chemical environments in the target molecule.
2. Generate dimer configurations for each unique atom.
"""


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
    # Chemical environments are represented as lists of [element_bonded, bond type, bond order]
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


def generate_water_dimer(pos, mesh_point, water_mapping, orientation="O"):
    """
    Generate TIP3P water dimer coordinates based on a specified mesh point and orientation.

    Notes
    -----
    Assumes units of Angstroms for positions but returns positions in nanometers.

    Parameters
    ----------
    pos : np.ndarray
        Array with positions for atoms in ligand-water dimer
    mesh_point : np.ndarray
        3D coordinates of mesh point to place either O or H atom of water, depending on orientation.
    water_mapping : dict
        Dictionary with atom indices for O, H1, and H2 atoms in water.
    orientation : str
        Orientation of water molecule relative to ligand atom. Choose "O" or "H".

    Returns
    -------
    dimer_pos : np.ndarray
        Array with positions for atoms in water dimer in nanometers.
    """
    # Constants for TIP3P water model
    OH_distance = 0.9572  # O-H bond length in Angstroms
    HOH_angle = 104.52  # H-O-H angle in degrees
    half_angle = _np.radians(HOH_angle / 2)  # Half angle in radians

    # Set water positions based on orientation
    if orientation == "O":
        pos[water_mapping["O"]] = mesh_point

        pos[water_mapping["H1"]] = mesh_point + _np.array(
            [OH_distance * _np.cos(half_angle), OH_distance * _np.sin(half_angle), 0]
        )

        pos[water_mapping["H2"]] = mesh_point + _np.array(
            [OH_distance * _np.cos(half_angle), -OH_distance * _np.sin(half_angle), 0]
        )

    elif orientation == "H":
        pos[water_mapping["H1"]] = mesh_point

        pos[water_mapping["H2"]] = mesh_point + _np.array(
            [OH_distance * _np.cos(half_angle), OH_distance * _np.sin(half_angle), 0]
        )

        pos[water_mapping["O"]] = mesh_point + _np.array([OH_distance, 0, 0])
    else:
        raise ValueError("Invalid orientation. Choose 'O' or 'H'.")

    return pos * 0.1  # Convert to nanometers


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
                    print(atom.name)
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


def generate_mesh_grid(center, n_points=100, spacing=0.1):
    """
    Generate a 3D mesh grid around a specified position.

    Parameters
    ----------
    center : np.ndarray
        3D coordinates of position to center mesh grid around.
    n_points : int
        Number of points to generate in each dimension.
    spacing : float
        Spacing between points in the grid.

    Returns
    -------
    mesh_points : np.ndarray
        3D mesh grid points around the specified position.
    """
    x = _np.linspace(
        center[0] - n_points * spacing / 2, center[0] + n_points * spacing / 2, n_points
    )
    y = _np.linspace(
        center[1] - n_points * spacing / 2, center[1] + n_points * spacing / 2, n_points
    )
    z = _np.linspace(
        center[2] - n_points * spacing / 2, center[2] + n_points * spacing / 2, n_points
    )
    mesh_points = _np.array(_np.meshgrid(x, y, z)).T.reshape(-1, 3)
    return mesh_points


def generate_initial_dimers(pos, atom_id, n_samples=50):
    """
    Generate initial dimer configurations for each unique atom in a topology.

    Parameters
    ----------
    pos : np.ndarray
        Array with positions for atoms in ligand-water dimer.
    n_samples : int
        Number of initial dimer configurations to generate for each atom.

    Returns
    -------
    dimers : list of openmm.Topology
        List of OpenMM Topology objects representing initial dimer configurations.
    """
    mesh_grid = generate_mesh_grid(center=pos[atom_id], n_points=n_samples, spacing=0.5)

    # Check there are no atoms within 2 Angstroms of the mesh grid points to avoid steric clashes
    # Create mask
    mask = _np.all(
        _np.linalg.norm(mesh_grid[:, None, :] - pos[:12], axis=2) > 2, axis=1
    )
    mesh_grid = mesh_grid[mask]

    dimers = []
    for mesh_point in mesh_grid:
        # Generate water dimers for both orientations
        dimer_O = generate_water_dimer(pos, mesh_point, water_mapping, orientation="O")
        dimer_H = generate_water_dimer(pos, mesh_point, water_mapping, orientation="H")
        dimers.append(dimer_O)
        dimers.append(dimer_H)
    return dimers


topology_off = create_off_topology(
    n_solute=1,
    n_solvent=1,
    solvent_smiles="[H:2][O:1][H:3]",
    solute_smiles="c1ccccc1",
)


# Convert OpenFF Topology to OpenMM Topology and get positions
topology = topology_off.to_openmm()
positions_omm = topology_off.get_positions().to_openmm()

# Get atoms with unique chemical environments in the target molecule
unique_atoms = get_unique_atoms(topology)

# Get atom indices for O, H1, and H2 atoms in a water molecule
water_mapping = get_water_mapping(topology)

# Generate initial dimer configurations for each unique atom
for atom in unique_atoms:
    positions = positions_omm.value_in_unit(_unit.angstroms)
    dimers = generate_initial_dimers(positions, atom, n_samples=10)


import openmm.app as app

for i in range(len(dimers)):
    with open(f"dimers_{i}.pdb", "w") as f:
        app.PDBFile.writeFile(topology, dimers[i] * 10, f)
