"""Base class for samplers."""
from abc import ABC, abstractmethod
import numpy as _np
from loguru import logger as _logger
from typing import Any, Tuple, Union
from ..reference_data import ReferenceData as _ReferenceData
from ..calculators import (
    HortonCalculator as _HortonCalculator,
    OrcaCalculator as _OrcaCalculator,
)

class BaseSampler(ABC):
    """Base class for samplers."""
    def __init__(self, 
                 reference_data: Union[_ReferenceData, None],
                 qm_calculator: Union[_OrcaCalculator, None],
                 horton_calculator: Union[_HortonCalculator, None] = None,
                 energy_scale: float = 1.0
                ):
        
        self._refence_data = reference_data or _ReferenceData()
        self._qm_calculator = qm_calculator
        self._horton_calculator = horton_calculator
        self._energy_scale = energy_scale

    @abstractmethod
    def sample(self, *args, **kwargs):
        raise NotImplementedError("This method must be implemented in the derived sampler class.")

    # ------------------------------------------------------------------------- #
    #                            Concrete methods                               #
    # ------------------------------------------------------------------------- #    
    @property   
    def reference_data(self):
        return self._reference_data

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
        pos_qm: _np.ndarray,
        symbols_qm: list[str],
        pos_mm: _np.ndarray,
        charges_mm: _np.ndarray,
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
            external_potentials = _np.hstack(
                [_np.expand_dims(charges_mm, axis=1), pos_mm]
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
    
    def get_single_point_energy(self, pos: _np.ndarray, symbols: list[str], directory: str, orca_blocks:str, calc_polarizability: bool=False):
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
        orca_blocks: str
            ORCA blocks.
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
            orca_blocks=orca_blocks
            if not calc_polarizability
            else orca_blocks + "%elprop\nPolar 1\ndipole true\nquadrupole true\nend\n",
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
        orca_blocks: str,
        calc_static: bool = True,
        calc_induction: bool = True,
        calc_horton: bool = True,
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
        vacuum_energy, e_static, polarizability, horton_data, e_ind: float, _np.ndarray(3, 3), dict, float
            Vacuum, energy, static energy, polarizability, horton partitioning data, and induction energy.
        """
        # Get the vacuum energy
        vacuum_energy = self.get_single_point_energy(
            pos=pos_qm,
            symbols=symbols_qm,
            directory=directory_vacuum,
            orca_blocks=orca_blocks,
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

        return vacuum_energy, e_static, polarizability, horton_data, e_ind