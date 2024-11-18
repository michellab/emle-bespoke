"""LJ fitting"""
from typing import List, Union

import numpy as _np
import openmm as _mm
import openmm.unit as _unit
from emle.models import EMLE
from loguru import logger as _logger
from openff.interchange import Interchange as _Interchange
from openff.toolkit import ForceField as _ForceField
from openmm import LocalEnergyMinimizer as _LocalEnergyMinimizer

from .._constants import HARTREE_TO_KJ_MOL
from ..cli._sample_train import create_mixed_system as _create_mixed_system
from ..cli._sample_train import create_simulation as _create_simulation
from ..cli._sample_train import remove_constraints as _remove_constraints
from ._lj_potential import LennardJonesPotential as _LennardJonesPotential
from ._mcmc import MonteCarloSampler
from ._utils import get_unique_atoms as _get_unique_atoms
from ._utils import get_water_mapping as _get_water_mapping
from ._utils import sort_two_lists as _sort_two_lists
from ._utils import unique_with_delta as _unique_with_delta




dimer_gen = DimerGenerator()
energies, config = dimer_gen.generate_dimers(
    solute_smiles="c1ccccc1", solvent_smiles="[H:2][O:1][H:3]"
)


lj_pot = _LennardJonesPotential(
    topology_off=dimer_gen._topology_off,
    forcefield=dimer_gen._forcefield,
    parameters_to_fit={"n-tip3p-O": ["sigma", "epsilon"]},
)


curves = []
for pos in config:
    curves.append(
        dimer_gen.generate_dimer_curve(pos, list(range(12)), list(range(12, 15)))
    )

import torch as _torch

device = _torch.device("cuda") if _torch.cuda.is_available() else _torch.device("cpu")
dtype = _torch.float64

atomic_types = _torch.tensor([6] * 6 + [1] * 6, device=device, dtype=_torch.int64)
emle_model = EMLE(device=device, dtype=dtype)
charges_mm = _torch.tensor([-0.834, 0.417, 0.417], device=device, dtype=dtype)

for c, curve in enumerate(curves):
    for i, pos in enumerate(curve):
        pos = _torch.tensor(pos)
        energy_lj = lj_pot.forward(pos, list(range(12)), list(range(12, 15)))
        e_static, e_ind = emle_model.forward(
            atomic_numbers=atomic_types,
            charges_mm=charges_mm,
            xyz_qm=pos[:12].to(device=device, dtype=dtype) * 10,
            xyz_mm=pos[12:].to(device=device, dtype=dtype) * 10,
        )
        e_static = e_static * HARTREE_TO_KJ_MOL
        e_ind = e_ind * HARTREE_TO_KJ_MOL
        energy = energy_lj + e_static + e_ind
        print(energy.item(), energy_lj.item(), e_static.item(), e_ind.item())

exit()

from ..calculators import ORCACalculator as _ORCACalculator
from ..sampler import ReferenceDataSampler as _ReferenceDataSampler

_logger.debug("Running the single point QM energy calculation.")
calc = _ORCACalculator()

# Write the dimers with the lowest energy to PDB files
for c, curve in enumerate(curves):
    for i, pos in enumerate(curve):
        with open(f"pdb_files/dimer_{c}_{i}.pdb", "w") as f:
            _mm.app.PDBFile.writeFile(dimer_gen._topology, pos * 10, f)

solute_energy = calc.get_potential_energy(
    elements=[atom.element.symbol for atom in list(dimer_gen._topology.atoms())[:12]],
    positions=pos[:12] * 10,
    directory="solute_fitting",
    orca_simple_input="! b3lyp cc-pvtz TightSCF D3BJ",
    orca_blocks="%MaxCore 1024\n%pal\nnprocs 8\nend\n",
)
solvent_energy = calc.get_potential_energy(
    elements=[atom.element.symbol for atom in list(dimer_gen._topology.atoms())[12:]],
    positions=pos[12:] * 10,
    directory="solvent_fitting",
    orca_simple_input="! b3lyp cc-pvtz TightSCF D3BJ",
    orca_blocks="%MaxCore 1024\n%pal\nnprocs 8\nend\n",
)


# Write the dimers with the lowest energy to PDB files
for c, curve in enumerate(curves):
    print("Curve", c)
    for i, pos in enumerate(curve):
        pot_energy = calc.get_potential_energy(
            elements=[atom.element.symbol for atom in dimer_gen._topology.atoms()],
            positions=pos * 10,
            directory="lj_fitting",
            orca_simple_input="! b3lyp cc-pvtz TightSCF D3BJ",
            orca_blocks="%MaxCore 1024\n%pal\nnprocs 8\nend\n",
        )
        print(pot_energy - solute_energy - solvent_energy)
