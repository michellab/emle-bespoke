import time
from typing import Any, Tuple

import openmm as _mm
import openmm.unit as _unit
import torch as _torch
from loguru import logger as _logger

from ._constants import ATOMIC_NUMBERS_TO_SYMBOLS as _ATOMIC_NUMBERS_TO_SYMBOLS
from .reference_data import ReferenceData as _ReferenceData


class ReferenceDataSampler:
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
    """

    def __init__(
        self,
        system,
        context,
        integrator,
        topology,
        qm_region,
        qm_calculator,
        reference_data=None,
        cutoff=12.0,
        horton_calculator=None,
        energy_scale=1.0,
        length_scale=1.0,
        dtype=_torch.float64,
        device=_torch.device("cuda"),
    ):
        self._system = system
        self._context = context
        self._integrator = integrator
        self._topology = topology
        self._cutoff = cutoff
        self._atomic_numbers = _torch.tensor(
            [a.element.atomic_number for a in topology.atoms()],
            dtype=_torch.int64,
            device=device,
        )
        self._qm_region = _torch.tensor(qm_region, dtype=_torch.int64, device=device)
        self._energy_scale = energy_scale
        self._length_scale = length_scale

        # Calculators
        self._qm_calculator = qm_calculator
        self._horton_calculator = horton_calculator

        # Reference data lists
        self._reference_data = reference_data if reference_data else _ReferenceData()

        # Device and dtype
        self._dtype = dtype
        self._device = device

        # Get the point charges
        self._point_charges = self._get_point_charges()

    @property
    def reference_data(self):
        return self._reference_data

    def _get_point_charges(self):
        """
        Get the point charges from the system.

        Returns
        -------
        point_charges: _torch.Tensor(NATOMS)
        """
        non_bonded_force = [
            f for f in self._system.getForces() if isinstance(f, _mm.NonbondedForce)
        ][0]
        point_charges = _torch.zeros(
            self._topology.getNumAtoms(), dtype=_torch.float64, device=self._device
        )
        for i in range(non_bonded_force.getNumParticles()):
            # charge, sigma, epsilon
            charge, _, _ = non_bonded_force.getParticleParameters(i)
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
        for i in range(3):
            positions = positions - _torch.outer(
                _torch.floor(positions[:, i] / boxvectors[i, i]), boxvectors[i]
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

    def get_static_energy(
        self,
        pos_mm: _torch.Tensor,
        charges_mm: _torch.Tensor,
        directory_vacuum: str,
        calc_static: bool,
    ) -> float:
        """
        Get the static energy.

        Parameters
        ----------
        pos_mm: _torch.Tensor(NATOMS, 3)
            Atomic positions in the MM region in Angstrom.
        charges_mm: _torch.Tensor(NATOMS)
            Point charges in the MM region.
        directory_vacuum: str
            Directory for the vacuum calculation.
        calc_static: bool
            Whether to calculate the static energy.

        Returns
        -------
        e_static: float or None
            Static energy in kJ/mol or None if not calculated.

        Notes
        -----
        This method first gets the vacuum potential at the MM positions and then calculates the static energy.
        """
        if calc_static:
            _logger.debug("Getting the static energy.")
            vacuum_pot = self._qm_calculator.get_vpot(
                mesh=pos_mm,
                directory=directory_vacuum,
            ).to(self._device, dtype=self._dtype)
            e_static = _torch.sum(vacuum_pot * charges_mm) * self._energy_scale
        else:
            e_static = None

        return e_static

    def get_polarizability(
        self, directory_vacuum: str, calc_polarizability: bool
    ) -> _torch.Tensor:
        """
        Get the polarizability.

        Parameters
        ----------
        directory_vacuum: str
            Directory for the vacuum calculation.
        calc_polarizability: bool
            Whether to get the polarizability.

        Returns
        -------
        polarizability: _torch.Tensor(3, 3) or None
            Polarizability tensor in Bohr^3 or None if not calculated.
        """
        if calc_polarizability:
            _logger.debug("Getting the polarizability.")
            polarizability = self._qm_calculator.get_polarizability(
                directory=directory_vacuum,
            )
        else:
            polarizability = None

        return polarizability

    def get_horton_partitioning(self, directory_vacuum: str, calc_horton: bool) -> dict:
        """
        Get the horton partitioning.

        Parameters
        ----------
        directory_vacuum: str
            Directory for the vacuum calculation.
        calc_horton: bool
            Whether to get the horton partitioning.

        Returns
        -------
        dict
            Dictionary containing the horton partitioning data.
        """
        if calc_horton:
            _logger.debug("Getting the horton partitioning.")
            self._qm_calculator.get_mkl(
                directory=directory_vacuum,
            )

            horton_data = self._horton_calculator.get_horton_partitioning(
                input_file=self._qm_calculator.name_prefix + ".molden.input",
                directory=directory_vacuum,
                scheme="mbis",
                lmax=3,
            )
        else:
            horton_data = {
                "s": None,
                "q_core": None,
                "q_val": None,
                "alpha": None,
                "mu": None,
            }

        return horton_data

    def get_induction_energy(
        self,
        pos_qm: _torch.Tensor,
        symbols_qm: list[str],
        pos_mm: _torch.Tensor,
        charges_mm: _torch.Tensor,
        directory_pc: str,
        orca_blocks: str,
        vacuum_energy: float,
        e_static: float,
        calc_induction: bool,
    ):
        """
        Get the induction energy.

        Parameters
        ----------
        pos_qm: _torch.Tensor(NATOMS, 3)
            Atomic positions in the QM region in Angstrom.
        symbols_qm: list[str]
            Atomic symbols in the QM region.
        pos_mm: _torch.Tensor(NATOMS, 3)
            Atomic positions in the MM region in Angstrom.
        charges_mm: _torch.Tensor(NATOMS)
            Point charges in the MM region.
        directory_pc: str
            Directory for the QM/MM calculation.
        orca_blocks: str
            ORCA blocks.
        vacuum_energy: float
            Vacuum energy.
        e_static: float
            Static energy.
        calc_induction: bool
            Whether to calculate the induction energy.

        Returns
        -------
        e_ind: float or None
            Induction energy in kJ/mol or None if not calculated.
        """
        if calc_induction:
            _logger.debug("Getting the induction energy.")
            external_potentials = _torch.hstack(
                [_torch.unsqueeze(charges_mm, dim=1), pos_mm]
            )
            qm_mm_energy = self._qm_calculator.get_potential_energy(
                elements=symbols_qm,
                positions=pos_qm,
                orca_external_potentials=external_potentials,
                directory=directory_pc,
                orca_blocks=orca_blocks,
            )

            e_int = (qm_mm_energy - vacuum_energy) * self._energy_scale
            e_ind = e_int - e_static
        else:
            e_ind = None

        return e_ind

    def get_reference_data(
        self,
        pos_qm: _torch.Tensor,
        symbols_qm: list[str],
        pos_mm: _torch.Tensor,
        charges_mm: _torch.Tensor,
        directory_vacuum: str,
        directory_pc: str,
        orca_blocks: str,
        calc_static: bool = True,
        calc_induction: bool = True,
        calc_horton: bool = True,
        calc_polarizability: bool = True,
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Get the reference data.

        Parameters
        ----------
        pos_qm: _torch.Tensor(NATOMS, 3)
            Atomic positions in the QM region in Angstrom.
        symbols_qm: list[str]
            Atomic symbols in the QM region.
        pos_mm: _torch.Tensor(NATOMS, 3)
            Atomic positions in the MM region in Angstrom.
        charges_mm: _torch.Tensor(NATOMS)
            Point charges in the MM region.
        directory_vacuum: str
            Directory for the vacuum calculation.
        directory_pc: str
            Directory for the QM/MM calculation.
        orca_blocks: str
            ORCA blocks.
        calc_static: bool
            Whether to calculate the static energy.
        calc_induction: bool
            Whether to calculate the induction energy.
        calc_horton: bool
            Whether to calculate the horton partitioning.
        calc_polarizability: bool
            Whether to calculate the polarizability.

        Returns
        -------
        e_static, polarizability, horton_data, e_ind: float, _torch.Tensor(3, 3), dict, float
            Static energy, polarizability, horton partitioning data, and induction energy.
        """
        _logger.debug("Running the single point QM energy calculation.")
        vacuum_energy = self._qm_calculator.get_potential_energy(
            elements=symbols_qm,
            positions=pos_qm,
            directory=directory_vacuum,
            orca_blocks=orca_blocks
            if not calc_polarizability
            else orca_blocks + "%elprop\nPolar 1\ndipole true\nquadrupole true\nend\n",
        )

        e_static = self.get_static_energy(
            pos_mm=pos_mm,
            charges_mm=charges_mm,
            directory_vacuum=directory_vacuum,
            calc_static=calc_static,
        )

        # Get the polarizability
        polarizability = self.get_polarizability(
            calc_polarizability=calc_polarizability,
            directory_vacuum=directory_vacuum,
        )

        # Get the horton partitioning
        horton_data = self.get_horton_partitioning(
            calc_horton=calc_horton,
            directory_vacuum=directory_vacuum,
        )

        # Get the induction energy
        e_ind = self.get_induction_energy(
            pos_qm=pos_qm,
            symbols_qm=symbols_qm,
            pos_mm=pos_mm,
            charges_mm=charges_mm,
            directory_pc=directory_pc,
            orca_blocks=orca_blocks,
            vacuum_energy=vacuum_energy,
            e_static=e_static,
            calc_induction=calc_induction,
        )

        return e_static, polarizability, horton_data, e_ind

    def sample(
        self,
        n_steps: int,
        calc_static: bool = True,
        calc_induction: bool = True,
        calc_horton: bool = True,
        calc_polarizability: bool = True,
    ) -> dict:
        """
        Sample the system for a given number of steps and calculate the necessary properties.

        Parameters
        ----------
        n_steps: int
            Number of steps to sample the system.

        Returns
        -------
        e_static: float
            Static energy from the QM/MM calculation.
        """
        assert (
            self._horton_calculator is not None if calc_horton else True
        ), "The horton calculator must be provided if the horton partitioning is to be calculated."

        _logger.debug("Sampling a new configuration.")
        _logger.debug(f"Number of integration steps: {n_steps}")

        # Integrate for a given number of steps
        self._integrator.step(n_steps)

        # Get the positions, box vectors, and energy before EMLE to ensure correct positions
        state = self._context.getState(getPositions=True, getEnergy=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(_unit.angstrom)
        pbc = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(_unit.angstrom)

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
        # symbols_mm = [_ATOMIC_NUMBERS_TO_SYMBOLS[an.item()] for an in z_mm]
        # symbols = symbols_qm + symbols_mm

        # Define the directory for vacuum calculations and QM/MM calculations
        directory_vacuum = "vacuum"
        directory_pc = "pc"

        orca_blocks = "%MaxCore 1024\n%pal\nnprocs 8\nend\n"

        # Run the single point QM energy calculation
        charges_mm = self._point_charges[~molecule_mask][R_cutoff]

        # Get the reference data
        e_static, polarizability, horton_data, e_ind = self.get_reference_data(
            pos_qm=pos_qm,
            symbols_qm=symbols_qm,
            pos_mm=pos_mm,
            charges_mm=charges_mm,
            directory_vacuum=directory_vacuum,
            directory_pc=directory_pc,
            orca_blocks=orca_blocks,
            calc_static=calc_static,
            calc_induction=calc_induction,
            calc_horton=calc_horton,
            calc_polarizability=calc_polarizability,
        )

        _logger.debug(f"E(static) = {e_static}")
        _logger.debug(f"E(induced) = {e_ind}")

        # Add the reference data to the lists
        self._reference_data.add_data_to_key("z", z_qm)
        self._reference_data.add_data_to_key("xyz_qm", pos_qm * self._length_scale)
        self._reference_data.add_data_to_key("alpha", polarizability)
        self._reference_data.add_data_to_key("xyz_mm", pos_mm * self._length_scale)
        self._reference_data.add_data_to_key("charges_mm", charges_mm)
        self._reference_data.add_data_to_key("s", horton_data["s"])
        self._reference_data.add_data_to_key("mu", horton_data["mu"])
        self._reference_data.add_data_to_key("q_core", horton_data["q_core"])
        self._reference_data.add_data_to_key("q_val", horton_data["q_val"])
        self._reference_data.add_data_to_key("e_static", e_static)
        self._reference_data.add_data_to_key("e_ind", e_ind)

        return self._reference_data
