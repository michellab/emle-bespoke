import time
from typing import Tuple

import numpy as np
import openmm.unit as unit
import torch as _torch

from ._constants import ATOMIC_NUMBERS_TO_SYMBOLS as _ATOMIC_NUMBERS_TO_SYMBOLS
from ._constants import HARTREE_TO_KJ_MOL as _HARTREE_TO_KJ_MOL


class ReferenceDataCalculator:
    """
    Class to calculate the reference QM(/MM) data.

    Parameters
    ----------
    system: simtk.openmm.System
        OpenMM system.
    context: simtk.openmm.Context
        OpenMM context.
    integrator: simtk.openmm.Integrator
        OpenMM integrator.
    topology: simtk.openmm.app.Topology
        OpenMM topology.
    energy_scale: float
        Energy scale.
    length_scale: float
        Lengt scale.
    """

    def __init__(
        self,
        system,
        context,
        integrator,
        topology,
        qm_calculator,
        qm_region,
        energy_scale=_HARTREE_TO_KJ_MOL,
        length_scale=1.0,
        dtype=_torch.float64,
        device=_torch.device("cuda"),
    ):
        self._system = system
        self._context = context
        self._integrator = integrator
        self._topology = topology
        self._cutoff = 12.0
        self._atomic_numbers = _torch.tensor(
            [a.element.atomic_number for a in topology.atoms()],
            dtype=_torch.int64,
            device=device,
        )

        # QM settings
        self._qm_region = _torch.tensor(qm_region, dtype=_torch.int64, device=device)
        self._qm_calculator = qm_calculator

        # Energy and length scales to convert the units for/from the QM/MM calculations
        self._energy_scale = energy_scale
        self._length_scale = length_scale

        # Reference data lists
        self._ref_pos_qm = []
        self._ref_pos_mm = []
        self._ref_charges_mm = []
        self._ref_atomic_numbers = []

        # Device and dtype
        self._dtype = dtype
        self._device = device

        # Get the point charges
        self._point_charges = self._get_point_charges()

    def _get_point_charges(self):
        """
        Get the point charges from the system.

        Returns
        -------
        point_charges: _torch.Tensor(NATOMS)
        """
        non_bonded_force = [
            f for f in self._system.getForces() if isinstance(f, mm.NonbondedForce)
        ][0]
        point_charges = _torch.zeros(
            self._topology.getNumAtoms(), dtype=_torch.float64, device=self._device
        )
        for i in range(non_bonded_force.getNumParticles()):
            _, charge, _ = non_bonded_force.getParticleParameters(i)
            point_charges[i] = charge._value
        return point_charges

    def _wrap_positions(
        self, positions: _torch.Tensor, boxvectors: _torch.Tensor
    ) -> _torch.Tensor:
        """
        Wrap the positions to the main box.

        Parameters
        ----------
        positions: _torch.Tensor(NATOMS, 3)
            Atomic positions.
        boxvectors: _torch.Tensor(3, 3)
            Box vectors.

        Returns
        -------
        positions: _torch.Tensor(NATOMS, 3)
            Wrapped atomic positions.
        """
        positions = positions - _torch.outer(
            _torch.floor(positions[:, 2] / boxvectors[2, 2]), boxvectors[2]
        )
        positions = positions - _torch.outer(
            _torch.floor(positions[:, 1] / boxvectors[1, 1]), boxvectors[1]
        )
        positions = positions - _torch.outer(
            _torch.floor(positions[:, 0] / boxvectors[0, 0]), boxvectors[0]
        )
        return positions

    def _center_molecule(
        self,
        positions: _torch.Tensor,
        boxvectors: _torch.Tensor,
        molecule_mask: _torch.Tensor,
    ) -> _torch.Tensor:
        """
        Center the molecule in the box (Lx/2, Ly/2, Lz/2).

        Parameters
        ----------
        positions: _torch.Tensor(NATOMS, 3)
            Atomic positions.
        boxvectors: _torch.Tensor(3, 3)
            Box vectors.
        mol_indices: _torch.Tensor(NATOMS)
            Molecule indices.

        Returns
        -------
        positions: _torch.Tensor(NATOMS, 3)
            Centered atomic positions.
        """
        mol_positions = positions[molecule_mask]
        com = mol_positions.mean(dim=0)
        box_center = 0.5 * boxvectors.diagonal()
        translation_vector = box_center - com
        positions += translation_vector

        # Wrap the positions to the main box
        self._wrap_positions(positions, boxvectors)

        return positions

    def _distance_to_molecule(
        self,
        positions: _torch.Tensor,
        boxvectors: _torch.Tensor,
        molecule_mask: _torch.Tensor,
    ) -> _torch.Tensor:
        """
        Calculate the R matrix for the molecule.

        Parameters
        ----------
        positions: _torch.Tensor(NATOMS, 3)
            Atomic positions.
        boxvectors: _torch.Tensor(3, 3)
            Box vectors.
        molecule_mask: _torch.Tensor(NATOMS)
            Molecule mask.

        Returns
        -------
        R: _torch.Tensor(NATOMS, NATOMS)
            Distance matrix.
        """
        R = _torch.cdist(positions[molecule_mask], positions[~molecule_mask], p=2)
        return R

    def _write_xyz(
        self, positions: _torch.Tensor, elements: list[str], filename: str
    ) -> None:
        """
        Write the XYZ file.

        Parameters
        ----------
        positions: _torch.Tensor(NATOMS, 3)
            Atomic positions.
        elements: _torch.Tensor(NATOMS)
            Atomic elements.
        filename: str
            Filename.
        """
        with open(filename, "w") as f:
            f.write(f"{positions.shape[0]}\n")
            f.write(f"Written by emle-spoke on {time.strftime('%Y-%m-%d %H:%M:%S')} \n")
            for element, position in zip(elements, positions):
                f.write(f"{element} {position[0]} {position[1]} {position[2]}\n")

    def sample(
        self,
        steps: int,
        calc_static: bool = True,
        calc_induction: bool = True,
        calc_horton: bool = True,
        calc_polarizability: bool = True,
    ) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
        """
        Sample the system for a given number of steps and calculate the necessary properties.

        Parameters
        ----------
        steps: int
            Number of steps to sample the system.


        Returns
        -------
        e_static: float
            Static energy from the QM/MM calculation.
        """
        if calc_polarizability or calc_horton:
            calc_static = True

        # Integrate for a given number of steps
        self._integrator.step(steps)

        # Get the positions, box vectors, and energy before EMLE to ensure correct positions
        state = self._context.getState(getPositions=True, getEnergy=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        pbc = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)

        # Convert the positions and box vectors to torch tensors
        pos = _torch.from_numpy(positions).to(self._device, dtype=self._dtype)
        box = _torch.from_numpy(pbc).to(self._device, dtype=self._dtype)

        # Get the molecule mask
        molecule_mask = _torch.zeros(
            pos.shape[0], dtype=_torch.bool, device=self._device
        )
        molecule_mask[self._qm_region] = True

        # Center the molecule
        pos = self._center_molecule(pos, box, molecule_mask)

        # Calculate the distance matrix
        R = self._distance_to_molecule(pos, box, molecule_mask)
        R_cutoff = _torch.any(R < self._cutoff, dim=0)

        # Split the positions into QM and MM regions
        pos_qm = pos[molecule_mask]
        pos_mm = pos[~molecule_mask][R_cutoff]

        # Get the atomic numbers
        z_qm = self._atomic_numbers[molecule_mask]
        z_mm = self._atomic_numbers[~molecule_mask][R_cutoff]

        # Get the atomic symbols
        symbols_qm = [_ATOMIC_NUMBERS_TO_SYMBOLS[an.item()] for an in z_qm]
        symbols_mm = [_ATOMIC_NUMBERS_TO_SYMBOLS[an.item()] for an in z_mm]
        symbols = symbols_qm + symbols_mm

        orca_blocks = "%MaxCore 1024\n%pal\nnprocs 1\nend\n"
        if calc_polarizability:
            orca_blocks += "%elprop\nPolar 1\ndipole true\nquadrupole true\nend\n"

        if calc_polarizability or calc_horton or calc_static:
            vacuum_energy = self._qm_calculator.get_potential_energy(
                elements=symbols_qm,
                positions=pos_qm,
                directory="vacuum",
                orca_blocks=orca_blocks,
            )

        if calc_static:
            vacuum_pot = self._qm_calculator.get_vpot(
                mesh=pos_mm,
                directory="vacuum",
            )
            e_static = (
                _torch.sum(vacuum_pot * self._point_charges[~molecule_mask][R_cutoff])
                * self._energy_scale
            )

        if calc_horton:
            self._qm_calculator.get_mkl(
                directory="vacuum",
            )

        if calc_induction:
            charges_mm = self._point_charges[~molecule_mask][R_cutoff]
            external_potentials = _torch.hstack(
                [_torch.unsqueeze(charges_mm, dim=1), pos_mm]
            )
            qm_mm_energy = self._qm_calculator.get_potential_energy(
                elements=symbols,
                positions=pos_qm,
                orca_external_potentials=external_potentials,
                directory="qm_mm",
            )

            e_int = (qm_mm_energy - vacuum_energy) * self._energy_scale
            e_ind = e_int - e_static

        # Add the reference data to the lists
        self._ref_pos_qm.append(pos_qm)
        self._ref_pos_mm.append(pos_mm)
        self._ref_charges_mm.append(charges_mm)
        self._ref_atomic_numbers.append(z_qm)

        return e_static, e_ind, vacuum_energy, vacuum_pot, pos_qm, pos_mm


if __name__ == "__main__":
    from sys import stdout

    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    from openmmml import MLPotential

    from .parsers import ORCACalculator as _ORCACalculator

    # Load PDB file and set the FFs
    prmtop = app.AmberPrmtopFile(
        "/home/joaomorado/repos/emle-bespoke/src/benzene_sage_water.prm7"
    )
    inpcrd = app.AmberInpcrdFile(
        "/home/joaomorado/repos/emle-bespoke/src/benzene_sage_water.rst7"
    )

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
    mlAtoms = [a.index for a in next(prmtop.topology.chains()).atoms()]

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

    ref_calculator = ReferenceDataCalculator(
        system=mixedSystem,
        context=context,
        integrator=integrator,
        topology=prmtop.topology,
        qm_calculator=_ORCACalculator(),
        qm_region=mlAtoms,
        energy_scale=_HARTREE_TO_KJ_MOL,
        length_scale=1.0,
        dtype=_torch.float64,
        device=_torch.device("cuda"),
    )

    context.setPositions(inpcrd.positions)

    ref_calculator.sample(10)
