import time

import numpy as _np
import openmm as _mm
import openmm.unit as _unit
from loguru import logger as _logger

from .._constants import ATOMIC_NUMBERS_TO_SYMBOLS as _ATOMIC_NUMBERS_TO_SYMBOLS
from ._openmm_sampler import OpenMMSampler as _OpenMMSampler


class MDSampler(_OpenMMSampler):
    """
    Simple OpenMM-based plain MD sampler.

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
    qm_region: _np.ndarray
        An array of indices of the QM region.
    cutoff: float
        Cutoff for interaction between QM and MM region (group-based cutoff).
    energy_scale: float
        Energy scale.
    length_scale: float
        Length scale.

    Attributes
    ----------
    Inherits all attributes from OpenMMSampler class.

    _cutoff: float
        Cutoff for interaction between QM and MM region (group-based cutoff).
    _atomic_numbers: _np.ndarray(NATOMS)
        Atomic numbers.
    _qm_region: _np.ndarray(NATOMS)
        Indices of the QM region.
    _point_charges: _np.ndarray(NATOMS)
        Point charges.
    """

    def __init__(
        self,
        system: _mm.System,
        context: _mm.Context,
        integrator: _mm.Integrator,
        topology: _mm.app.Topology,
        qm_region: _np.ndarray,
        cutoff: float = 12.0,
        energy_scale: float = 1.0,
        length_scale: float = 1.0,
    ):
        super().__init__(
            system=system,
            context=context,
            integrator=integrator,
            topology=topology,
            energy_scale=energy_scale,
            length_scale=length_scale,
        )

        self._cutoff = cutoff
        self._atomic_numbers = _np.array(
            [atom.element.atomic_number for atom in topology.atoms()],
            dtype=_np.int64,
        )
        self._qm_region = _np.array(qm_region, dtype=_np.int64)
        self._point_charges = self._get_point_charges()

    def _wrap_positions(
        self, positions: _np.ndarray, boxvectors: _np.ndarray
    ) -> _np.ndarray:
        """
        Wrap the positions to the main box.

        Parameters
        ----------
        positions: _np.ndarray(NATOMS, 3)
            Atomic positions.
        boxvectors: _np.ndarray(3, 3)
            Box vectors.

        Returns
        -------
        positions: _np.ndarray(NATOMS, 3)
            Wrapped atomic positions.
        """
        for i in range(3):
            positions = positions - _np.outer(
                _np.floor(positions[:, i] / boxvectors[i, i]), boxvectors[i]
            )

        return positions

    def _center_molecule(
        self,
        positions: _np.ndarray,
        boxvectors: _np.ndarray,
        molecule_mask: _np.ndarray,
    ) -> _np.ndarray:
        """
        Center the molecule in the box (Lx/2, Ly/2, Lz/2).

        Parameters
        ----------
        positions: _np.ndarray(NATOMS, 3)
            Atomic positions.
        boxvectors: _np.ndarray(3, 3)
            Box vectors.
        mol_indices: _np.ndarray(NATOMS)
            Molecule indices.

        Returns
        -------
        positions: _np.ndarray(NATOMS, 3)
            Centered atomic positions.
        """
        mol_positions = positions[molecule_mask]
        com = mol_positions.mean(axis=0)
        box_center = 0.5 * boxvectors.diagonal()
        translation_vector = box_center - com
        positions += translation_vector

        # Wrap the positions to the main box
        positions = self._wrap_positions(positions, boxvectors)

        return positions

    def _distance_to_molecule(
        self,
        positions: _np.ndarray,
        boxvectors: _np.ndarray,
        molecule_mask: _np.ndarray,
    ) -> _np.ndarray:
        """
        Calculate the R matrix for the molecule.

        Parameters
        ----------
        positions: _np.ndarray(NATOMS, 3)
            Atomic positions.
        boxvectors: _np.ndarray(3, 3)
            Box vectors.
        molecule_mask: _np.ndarray(NATOMS)
            Molecule mask.

        Returns
        -------
        R: _np.ndarray(NATOMS, NATOMS)
            Distance matrix.
        """
        R = _np.linalg.norm(
            positions[molecule_mask][:, None, :]
            - positions[~molecule_mask][None, :, :],
            axis=-1,
        )
        return R

    def _write_xyz(
        self, positions: _np.ndarray, elements: list[str], filename: str
    ) -> None:
        """
        Write the XYZ file.

        Parameters
        ----------
        positions: _np.ndarray(NATOMS, 3)
            Atomic positions.
        elements: _np.ndarray(NATOMS)
            Atomic elements.
        filename: str
            Filename.
        """
        with open(filename, "w") as f:
            f.write(f"{positions.shape[0]}\n")
            f.write(
                f"Written by emle-bespoke on {time.strftime('%Y-%m-%d %H:%M:%S')} \n"
            )
            for element, position in zip(elements, positions):
                f.write(f"{element} {position[0]} {position[1]} {position[2]}\n")

    def sample(
        self,
        n_steps: int,
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
        _logger.debug("Sampling a new configuration.")
        _logger.debug(f"Number of integration steps: {n_steps}")

        # Integrate for a given number of steps
        self._integrator.step(n_steps)

        # Get the positions, box vectors, and energy before EMLE to ensure correct positions
        state = self._context.getState(getPositions=True, getEnergy=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(_unit.angstrom)
        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(_unit.angstrom)

        # Get the molecule mask
        molecule_mask = _np.zeros(pos.shape[0], dtype=bool)
        molecule_mask[self._qm_region] = True

        # Center the molecule
        pos = self._center_molecule(pos, box, molecule_mask)

        # Calculate the distance matrix
        R = self._distance_to_molecule(pos, box, molecule_mask)
        R_cutoff = _np.any(R < self._cutoff, axis=0)

        # Split the positions into QM and MM regions
        pos_qm = pos[molecule_mask]
        pos_mm = pos[~molecule_mask][R_cutoff]

        # Get the atomic numbers
        z_qm = self._atomic_numbers[molecule_mask]
        # z_mm = self._atomic_numbers[~molecule_mask][R_cutoff]

        # Get the atomic symbols
        symbols_qm = [_ATOMIC_NUMBERS_TO_SYMBOLS[an.item()] for an in z_qm]
        # symbols_mm = [_ATOMIC_NUMBERS_TO_SYMBOLS[an.item()] for an in z_mm]
        # symbols = symbols_qm + symbols_mm

        # Run the single point QM energy calculation
        charges_mm = self._point_charges[~molecule_mask][R_cutoff]

        # Output dictionary
        output_dict = {
            "pos_qm": pos_qm,
            "symbols_qm": symbols_qm,
            "pos_mm": pos_mm,
            "charges_mm": charges_mm,
        }

        return output_dict
