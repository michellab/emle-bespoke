import logging as _logging
from sys import stdout

_logging.getLogger().setLevel(_logging.INFO)

_logging.info("Starting the EMLE Bespoke example")

import openmm as mm
import openmm.app as app
import openmm.unit as unit
import torch as torch
from openmmml import MLPotential

from emle_bespoke import EMLEBespoke, ReferenceDataSampler
from emle_bespoke.calculators import HortonCalculator, ORCACalculator

# Load PDB file and set the FFs
prmtop = app.AmberPrmtopFile("benzene_sage_water.prm7")
inpcrd = app.AmberInpcrdFile("benzene_sage_water.rst7")

# Create the OpenMM MM System and ML potential
mmSystem = prmtop.createSystem(
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.0 * unit.nanometers,
    constraints=app.HBonds,
    rigidWater=True,
    ewaldErrorTolerance=0.0005,
)
potential = MLPotential("ani2x")
# Choose the ML atoms
mlAtoms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Create the mixed ML/MM system (we're using the nnpops implementation for performance)
mixedSystem = potential.createMixedSystem(
    prmtop.topology, mmSystem, mlAtoms, interpolate=False, implementation="nnpops"
)
# Choose to run on a GPU (CUDA), with the LangevinMiddleIntegrator (NVT) and create the context
platform = mm.Platform.getPlatformByName("CUDA")
integrator = mm.LangevinMiddleIntegrator(
    300 * unit.kelvin, 1 / unit.picosecond, 0.001 * unit.picoseconds
)
context = mm.Context(mixedSystem, integrator, platform)
context.setPositions(inpcrd.positions)

# Create the reference data calculator
ref_calculator = ReferenceDataSampler(
    system=mixedSystem,
    context=context,
    integrator=integrator,
    topology=prmtop.topology,
    qm_calculator=ORCACalculator(),
    horton_calculator=HortonCalculator(),
    qm_region=mlAtoms,
    dtype=torch.float64,
    device=torch.device("cuda"),
)

# Create the EMLE bespoke trainer
trainer = EMLEBespoke(ref_calculator)
trainer.train_model(
    n_samples=2,
    n_steps=100,
)
