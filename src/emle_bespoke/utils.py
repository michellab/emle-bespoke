import re
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import openmm as _mm
import openmm.app as _app
import openmm.unit as _unit
from loguru import logger as _logger
from openff.interchange import Interchange as _Interchange
from openff.interchange.components._packmol import UNIT_CUBE as _UNIT_CUBE
from openff.interchange.components._packmol import pack_box as _pack_box
from openff.toolkit import Molecule as _Molecule
from openff.toolkit import Topology as _Topology
from openff.units import unit as _offunit
from openmmml import MLPotential as _MLPotential

PACKMOL_KWARGS = {
    "box_shape": _UNIT_CUBE,
    "target_density": 1.0 * _offunit.gram / _offunit.milliliter,
}


def is_mapped_smiles(smiles: str) -> bool:
    """Check if the given SMILES string is a mapped SMILES."""
    pattern = re.compile(r":[0-9]+")
    return bool(pattern.search(smiles))


def create_molecule(smiles: str) -> _Molecule:
    """Create an OpenFF molecule from a SMILES string."""
    try:
        if is_mapped_smiles(smiles):
            return _Molecule.from_mapped_smiles(smiles)
        return _Molecule.from_smiles(smiles)
    except Exception as e:
        _logger.error(f"Failed to create molecule from SMILES '{smiles}': {e}")
        raise RuntimeError(f"Failed to create molecule from SMILES '{smiles}': {e}")


def create_dimer_topology(
    solute_smiles: str, solvent_smiles: str, n_solvent: int = 1
) -> _Topology:
    try:
        _logger.info("Creating OpenFF dimer topology.")
        _logger.info(f"Solute SMILES: {solute_smiles}")
        _logger.info(f"Solvent SMILES: {solvent_smiles}")

        solute = create_molecule(solute_smiles)
        for atom in solute.atoms:
            atom.metadata["residue_name"] = "LIG"

        solvent = create_molecule(solvent_smiles)
        if solvent_smiles == "[H:2][O:1][H:3]":
            for atom in solvent.atoms:
                atom.metadata["residue_name"] = "HOH"

        # Generate conformers
        solute.generate_conformers(n_conformers=1)
        solvent.generate_conformers(n_conformers=1)

        # Create the topology
        topology = _Topology.from_molecules([solute] + [solvent] * n_solvent)
    except Exception as e:
        _logger.error(f"Failed to create OpenFF dimer topology: {e}")
        raise RuntimeError(f"Failed to create OpenFF dimer topology: {e}")

    return topology


def create_simulation_box_topology(
    n_solute: int,
    n_solvent: int,
    solvent_smiles: str,
    solute_smiles: str,
    packmol_kwargs: Optional[Dict[str, Any]] = None,
) -> _Topology:
    """Create an OpenFF topology for a system with solute and solvent molecules."""
    try:
        _logger.info("Creating OpenFF topology.")
        _logger.info(f"Number of solute molecules: {n_solute}")
        _logger.info(f"Number of solvent molecules: {n_solvent}")
        _logger.info(f"Solute SMILES: {solute_smiles}")
        _logger.info(f"Solvent SMILES: {solvent_smiles}")
        for key, value in (packmol_kwargs or PACKMOL_KWARGS).items():
            _logger.info(f"Packmol kwarg: {key} = {value}")

        solute = create_molecule(solute_smiles)
        if n_solute > 0:
            for atom in solute.atoms:
                atom.metadata["residue_name"] = "LIG"

        if n_solvent == 0 or solvent_smiles is None:
            return solute.to_topology()

        solvent = create_molecule(solvent_smiles)
        if solvent_smiles == "[H:2][O:1][H:3]" and n_solvent > 0:
            for atom in solvent.atoms:
                atom.metadata["residue_name"] = "HOH"

        return _pack_box(
            molecules=[solute, solvent],
            number_of_copies=[n_solute, n_solvent],
            **(packmol_kwargs or PACKMOL_KWARGS),
        )
    except Exception as e:
        _logger.error(f"Failed to create OpenFF topology: {e}")
        raise RuntimeError(f"Failed to create OpenFF topology: {e}")


def create_simulation(
    interchange: _Interchange,
    temperature: float = 298.15,
    pressure: float = 1.0,
    timestep: float = 1.0,
    friction_coefficient: float = 1.0,
) -> _app.Simulation:
    """Create an OpenMM simulation from the provided interchange object."""
    try:
        _logger.info("Creating OpenMM simulation.")
        _logger.info(f"Temperature: {temperature} K")
        _logger.info(f"Pressure: {pressure} bar")
        _logger.info(f"Timestep: {timestep} fs")
        _logger.info(f"Friction coefficient: {friction_coefficient} ps^-1")

        integrator = _mm.LangevinIntegrator(
            temperature * _unit.kelvin,
            friction_coefficient / _mm.unit.picosecond,
            timestep * _mm.unit.femtoseconds,
        )

        if pressure:
            barostat = _mm.MonteCarloBarostat(
                pressure * _unit.bar,
                temperature * _unit.kelvin,
            )

            additional_forces = [barostat]
        else:
            additional_forces = []

        simulation = interchange.to_openmm_simulation(
            combine_nonbonded_forces=True,
            integrator=integrator,
            additional_forces=additional_forces,
        )

        # simulation.minimizeEnergy()
        simulation.context.setVelocitiesToTemperature(temperature * _unit.kelvin)
        simulation.context.computeVirtualSites()

        return simulation
    except Exception as e:
        _logger.error(f"Failed to create OpenMM simulation: {e}")
        raise RuntimeError(f"Failed to create OpenMM simulation: {e}")


def create_mixed_system(
    ml_model: str, ml_atoms, simulation: _app.Simulation
) -> Tuple[_mm.System, _mm.Context, _mm.Integrator]:
    """
    Create a mixed system with the provided ML model and simulation object.

    Parameters
    ----------
    ml_model : str
        The machine learning model to use for the solute.
    ml_atoms : list
        The indices of the atoms to be treated with the ML model.
    simulation : _app.Simulation
        The OpenMM simulation object.

    Returns
    -------
    _mm.System, _mm.Context, _mm.Integrator
        The OpenMM system, context, and integrator instances.
    """
    if ml_model:
        _logger.info(f"Creating mixed system with ML model '{ml_model}'.")
        potential = _MLPotential(ml_model)
        integrator = deepcopy(simulation.integrator)
        system = potential.createMixedSystem(
            simulation.topology, simulation.system, ml_atoms
        )
        context = _mm.Context(system, integrator)
        context.setPositions(
            simulation.context.getState(getPositions=True).getPositions()
        )
    else:
        _logger.info("No ML model provided. Using the original MM system.")
        system = simulation.system
        context = simulation.context
        integrator = simulation.integrator

    return system, context, integrator


def add_emle_force(
    emle_model, qm_region, system, context, topology, cutoff=None, *args, **kwargs
):
    import torch as _torch
    from emle.models import EMLE as _EMLE
    from openmmtorch import TorchForce as _TorchForce

    from .emle_force import EMLEForce as _EMLEForce

    if emle_model:
        _logger.info(f"Adding EMLE force with model '{emle_model}'.")
        if emle_model == "default":
            emle_model = None

        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        dtype = _torch.float64

        # Create the EMLE model
        model = _EMLE(model=emle_model, device=device, dtype=dtype, *args, **kwargs)

        # Create the QM and MM masks
        qm_mask = _torch.zeros(topology.getNumAtoms(), dtype=_torch.bool, device=device)
        qm_mask[qm_region] = True

        # Get the atomic numbers
        atomic_numbers = _torch.tensor(
            [atom.element.atomic_number for atom in topology.atoms()],
            dtype=_torch.int64,
            device=device,
        )[qm_mask]

        # Charges
        non_bonded_force = [
            force
            for force in system.getForces()
            if isinstance(force, _mm.NonbondedForce)
        ][0]

        charges = _torch.tensor(
            [
                non_bonded_force.getParticleParameters(i)[0]._value
                for i in range(non_bonded_force.getNumParticles())
            ]
        ).to(device=device, dtype=dtype)

        # Create the EMLE force
        emle_force = _EMLEForce(
            model=model,
            atomic_numbers=atomic_numbers,
            charges=charges,
            qm_mask=qm_mask,
            cutoff=cutoff,
            device=device,
            dtype=dtype,
        )

        # Add the EMLE force to the system
        emle_module = _torch.jit.script(emle_force)
        emle_force = _TorchForce(emle_module)
        emle_force.setUsesPeriodicBoundaryConditions(True)
        system.addForce(emle_force)

        # In order to ensure that OpenMM doesn’t perform mechanical embedding,
        # we next need to zero the charges of the QM atoms in the MM system
        # See: https://sire.openbiosim.org/tutorial/part08/02_emle.html
        for i in qm_region:
            _, sigma, epsilon = non_bonded_force.getParticleParameters(i)
            non_bonded_force.setParticleParameters(i, 0, sigma, epsilon)

        context.reinitialize(preserveState=True)
    else:
        _logger.info("No EMLE model provided. Skipping EMLE force.")

    return system, context


def remove_constraints(system: _mm.System, atoms: list[int]) -> _mm.System:
    """
    Remove constraints involving chosen atoms from the system.

    Parameters
    ----------
    system : openmm.System
        The OpenMM system.
    atoms : list of int
        The list of atoms to remove constraints from.

    Returns
    -------
    openmm.System
        The modified OpenMM system.
    """
    # Remove constraints involving chosen atoms
    for i in range(system.getNumConstraints() - 1, -1, -1):
        p1, p2, _ = system.getConstraintParameters(i)
        if p1 in atoms or p2 in atoms:
            system.removeConstraint(i)

    return system


def write_dict_to_file(dict_to_write, filename):
    """
    Write the trained model to a file.

    Parameters
    ----------

    emle_model: dict
        Trained EMLE model.

    model_filename: str
        Filename to save the trained model.
    """
    import scipy.io
    import torch as _torch

    # Deatch the tensors, convert to numpy arrays and save the model.
    dict_filtered = {
        k: v.cpu().detach().numpy() if isinstance(v, _torch.Tensor) else v
        for k, v in dict_to_write.items()
        if v is not None
    }
    scipy.io.savemat(filename, dict_filtered)


def write_system_to_xml(system: _mm.System, filename: str) -> None:
    """
    Write the System to an XML file.

    Parameters
    ----------
    system : openmm.System
        The System to write.
    filename : str
        The name of the file to write.
    """
    with open(filename, "w") as outfile:
        outfile.write(_mm.XmlSerializer.serialize(system))
