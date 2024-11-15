"""LJ fitting"""
import numpy as _np
import openmm as _mm
import openmm.unit as _unit
from openff.interchange import Interchange as _Interchange
from openff.toolkit import ForceField as _ForceField
from openmm import LocalEnergyMinimizer as _LocalEnergyMinimizer

from ..cli._sample_train import create_mixed_system as _create_mixed_system
from ..cli._sample_train import create_simulation as _create_simulation
from ..cli._sample_train import remove_constraints as _remove_constraints
from ._mcmc import MonteCarloSampler
from ._utils import create_dimer_topology as _create_dimer_topology
from ._utils import get_unique_atoms as _get_unique_atoms
from ._utils import get_water_mapping as _get_water_mapping

# Create OpenFF Topology
topology_off = _create_dimer_topology("c1ccccc1", "[H:2][O:1][H:3]")

# Convert OpenFF Topology to OpenMM Topology and get positions
topology = topology_off.to_openmm()
positions_omm = topology_off.get_positions().to_openmm()

# Get atom indices for O, H1, and H2 atoms in a water molecule
water_mapping = _get_water_mapping(topology)

# Initialize force field and interchange object
ffs = ["openff_unconstrained-2.0.0.offxml"]
force_field = _ForceField(*ffs)
interchange = _Interchange.from_smirnoff(force_field=force_field, topology=topology_off)

# Create simulation object
simulation = _create_simulation(interchange, pressure=None)


# Define QM region and train model
topology = topology_off.to_openmm()
qm_region = [atom.index for atom in list(topology.chains())[0].atoms()]

# Remove constraints involving alchemical atoms
_remove_constraints(simulation.system, qm_region)
simulation.context.reinitialize(preserveState=True)

# Create mixed system
# system, context, integrator = _create_mixed_system(
#    "ani2x", qm_region, simulation
#    )

system = simulation.system
context = simulation.context
integrator = simulation.integrator

# Create Monte Carlo sampler
mc = MonteCarloSampler()

# Get atoms with unique chemical environments in the target molecule
unique_atoms = _get_unique_atoms(topology)
atom_indices = [water_mapping["O"], water_mapping["H1"], water_mapping["H2"]]

opt_pos = []
opt_energies = []
for i, atom in enumerate(unique_atoms):
    print("***")
    sphere_centre = positions_omm[atom]
    mc.sample(
        context,
        n_samples=50000,
        temperature=1000.0,
        sphere_radius=0.5 * _unit.nanometer,
        sphere_centre=sphere_centre,
        atom_indices=atom_indices,
    )

    # Get the energies and positions of the sampled dimers
    energies = _np.asarray([en._value for en in mc.energies])
    configurations = _np.asarray([pos._value for pos in mc.configurations])

    # Get unique dimers with the lowest energy
    _, unique_mask = _np.unique(_np.round(energies, decimals=0), return_index=True)
    energies_unique = energies[unique_mask]
    configurations = configurations[unique_mask]
    min_energy_idx = _np.argsort(energies_unique)[:50]

    # Optimize the dimers with the lowest energy
    for idx in min_energy_idx:
        context.setPositions(configurations[idx])
        en_before = context.getState(getEnergy=True).getPotentialEnergy()
        _LocalEnergyMinimizer.minimize(context, tolerance=1)
        print(en_before, context.getState(getEnergy=True).getPotentialEnergy())
        opt_pos.append(context.getState(getPositions=True).getPositions()._value)
        opt_energies.append(
            context.getState(getEnergy=True).getPotentialEnergy()._value
        )

    mc.reset()


#
opt_energies = _np.asarray(opt_energies)
opt_pos = _np.asarray(opt_pos)
_, unique_mask = _np.unique(_np.round(opt_energies, decimals=0), return_index=True)
energies_unique = opt_energies[unique_mask]
opt_pos = opt_pos[unique_mask]
print(opt_pos.shape)
print(unique_mask)


# Write the dimers with the lowest energy to PDB files
for i, pos in enumerate(opt_pos):
    with open(f"pdb_files/opt_dimer_{i}.pdb", "w") as f:
        print(pos.shape)
        _mm.app.PDBFile.writeFile(topology, pos * 10, f)


"""
for i in range(len(opt_pos)):
    with open(f"pdb_files/dimers_{i}.pdb", "w") as f:
        app.PDBFile.writeFile(topology, dimers[i] * 10, f)
"""
