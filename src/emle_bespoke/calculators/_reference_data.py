"""Module containing the ReferenceDataCalculator class."""

from typing import TYPE_CHECKING, Tuple

import numpy as _np
from loguru import logger as _logger

from .._constants import SYMBOLS_TO_ATOMIC_NUMBERS as _SYMBOLS_TO_ATOMIC_NUMBERS
from ._base import MBISCalculator as _MBISCalculator
from ._base import QMCalculator as _QMCalculator
from ._horton import HortonCalculator as _HortonCalculator
from ._orca import ORCACalculator as _ORCACalculator


class ReferenceDataCalculator:
    """
    Base class for reference data calculators.

    Attributes
    ----------
    qm_calculator: _QMCalculator
        QM calculator.
    mbis_calculator: _MBISCalculator
        MBIS calculator.
    energy_scale: float
        Energy scaling factor to convert from the QM calculator energy units to kJ/mol.
    length_scale: float
        Length scaling factor to convert from Angstrom to the QM calculator length units.
    """

    def __init__(
        self,
        qm_calculator: _QMCalculator = _ORCACalculator(),
        mbis_calculator: _MBISCalculator = _HortonCalculator(),
    ):
        self._qm_calculator = qm_calculator
        self._mbis_calculator = mbis_calculator

        self._energy_scale = 1.0
        self._length_scale = 1.0

    def get_static_energy(
        self,
        pos_mm: _np.ndarray,
        charges_mm: _np.ndarray,
        directory_vacuum: str,
        calc_static: bool = True,
    ) -> float:
        """
        Get the static energy.

        Parameters
        ----------
        pos_mm: _np.ndarray(NATOMS, 3)
            Atomic positions in the MM region in Angstrom.
        charges_mm: _np.ndarray(NATOMS)
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
            )
            e_static = _np.sum(vacuum_pot * charges_mm) * self._energy_scale
        else:
            e_static = None

        return e_static

    def get_polarizability(
        self, directory_vacuum: str, calc_polarizability: bool
    ) -> _np.ndarray:
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
        polarizability: _np.ndarray(3, 3) or None
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

    def get_mbis_partitioning(self, directory_vacuum: str, calc_mbis: bool) -> dict:
        """
        Get the MBIS partitioning.

        Parameters
        ----------
        directory_vacuum: str
            Directory for the vacuum calculation.
        calc_mbis: bool
            Whether to get the MBIS partitioning.

        Returns
        -------
        dict
            Dictionary containing the MBIS partitioning data.
        """
        if calc_mbis:
            _logger.debug("Getting the MBIS partitioning.")
            self._qm_calculator.get_mkl(
                directory=directory_vacuum,
            )

            mbis_data = self._mbis_calculator.get_mbis_partitioning(
                input_file=self._qm_calculator.name_prefix + ".molden.input",
                directory=directory_vacuum,
                scheme="mbis",
                lmax=3,
            )
        else:
            mbis_data = {
                "s": None,
                "q_core": None,
                "q_val": None,
                "alpha": None,
                "mu": None,
            }

        return mbis_data

    def get_induction_energy(
        self,
        pos_qm: _np.ndarray,
        symbols_qm: list[str],
        pos_mm: _np.ndarray,
        charges_mm: _np.ndarray,
        directory_pc: str,
        vacuum_energy: float,
        e_static: float,
        calc_induction: bool,
    ):
        """
        Get the induction energy.

        Parameters
        ----------
        pos_qm: _np.ndarray(NATOMS, 3)
            Atomic positions in the QM region in Angstrom.
        symbols_qm: list[str]
            Atomic symbols in the QM region.
        pos_mm: _np.ndarray(NATOMS, 3)
            Atomic positions in the MM region in Angstrom.
        charges_mm: _np.ndarray(NATOMS)
            Point charges in the MM region.
        directory_pc: str
            Directory for the QM/MM calculation.
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
            external_potentials = _np.hstack(
                [_np.expand_dims(charges_mm, axis=1), pos_mm]
            )
            qm_mm_energy = self._qm_calculator.get_potential_energy(
                elements=symbols_qm,
                positions=pos_qm,
                orca_external_potentials=external_potentials, # TODO: generalise
                directory=directory_pc,
            )

            e_int = (qm_mm_energy - vacuum_energy) * self._energy_scale
            e_ind = e_int - e_static
        else:
            e_ind = None

        return e_ind

    def get_single_point_energy(
        self,
        pos: _np.ndarray,
        symbols: list[str],
        directory: str,
        calc_polarizability: bool = False,
    ):
        """
        Get the single point energy.

        Parameters
        ----------
        pos: _np.ndarray(NATOMS, 3)
            Atomic positions in Angstrom.
        symbols: list[str]
            Atomic symbols.
        directory: str
            Directory for the calculation.
        calc_polarizability: bool
            Whether to calculate the polarizability.

        Returns
        -------
        vacuum_energy: float
            Vacuum energy.
        """

        _logger.debug("Running the single point QM energy calculation.")
        vacuum_energy = self._qm_calculator.get_potential_energy(
            elements=symbols,
            positions=pos,
            directory=directory,
            calc_polarizability=calc_polarizability,
        )

        return vacuum_energy

    def get_reference_data(
        self,
        pos_qm: _np.ndarray,
        symbols_qm: list[str],
        pos_mm: _np.ndarray,
        charges_mm: _np.ndarray,
        directory_vacuum: str,
        directory_pc: str,
        calc_static: bool = True,
        calc_induction: bool = True,
        calc_mbis: bool = True,
        calc_polarizability: bool = True,
    ) -> Tuple[float, float, _np.ndarray, dict, float]:
        """
        Get the reference data.

        Parameters
        ----------
        pos_qm: _np.ndarray(NATOMS, 3)
            Atomic positions in the QM region in Angstrom.
        symbols_qm: list[str]
            Atomic symbols in the QM region.
        pos_mm: _np.ndarray(NATOMS, 3)
            Atomic positions in the MM region in Angstrom.
        charges_mm: _np.ndarray(NATOMS)
            Point charges in the MM region.
        directory_vacuum: str
            Directory for the vacuum calculation.
        directory_pc: str
            Directory for the QM/MM calculation.
        calc_static: bool
            Whether to calculate the static energy.
        calc_induction: bool
            Whether to calculate the induction energy.
        calc_mbis: bool
            Whether to calculate the MBIS partitioning.
        calc_polarizability: bool
            Whether to calculate the polarizability.

        Returns
        -------
        dict
            Dictionary containing the reference data.
        """
        if pos_mm is None or charges_mm is None:
            if calc_induction or calc_static:
                raise ValueError(
                    "pos_mm and charges_mm must be provided if calc_induction or calc_static is True."
                )
        
        # Get the vacuum energy
        vacuum_energy = self.get_single_point_energy(
            pos=pos_qm,
            symbols=symbols_qm,
            directory=directory_vacuum, 
            calc_polarizability=calc_polarizability,
        )

        # Get the static energy
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
        mbis_data = self.get_mbis_partitioning(
            calc_mbis=calc_mbis,
            directory_vacuum=directory_vacuum,
        )

        # Get the induction energy
        e_ind = self.get_induction_energy(
            pos_qm=pos_qm,
            symbols_qm=symbols_qm,
            pos_mm=pos_mm,
            charges_mm=charges_mm,
            directory_pc=directory_pc,            
            vacuum_energy=vacuum_energy,
            e_static=e_static,
            calc_induction=calc_induction,
        )

        reference_dada_dict = {
            "z": _np.array([_SYMBOLS_TO_ATOMIC_NUMBERS[s] for s in symbols_qm]),
            "xyz_qm": pos_qm,
            "alpha": polarizability,
            "xyz_mm": pos_mm,
            "charges_mm": charges_mm,
            "s": mbis_data["s"],
            "mu": mbis_data["mu"],
            "q_core": mbis_data["q_core"],
            "q_val": mbis_data["q_val"],
            "e_static": e_static,
            "e_ind": e_ind,
        }

        return reference_dada_dict
