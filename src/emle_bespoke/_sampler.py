import time
import numpy as np
from typing import Tuple
from ._constants import HARTREE_TO_KJ_MOL as _HARTREE_TO_KJ_MOL
from ._constants import ATOMIC_NUMBERS_TO_SYMBOLS as _ATOMIC_NUMBERS_TO_SYMBOLS


class ReferenceDataCalculator:
    def __init__(
        self,
        parser,
        system,
        context,
        integrator,
        emle_calculator,
        energy_scale=_HARTREE_TO_KJ_MOL,
        length_scale=1.0,
    ):
        self.parser = parser
        self.system = system
        self.context = context
        self.integrator = integrator
        self.emle_base = emle_calculator

        self._energy_scale = 1.0

        #

    def _get_qm_mm_static_energy(self, symbols, xyz_mm, charges_mm, xyz_qm):
        vacuum_energy = self.parser.get_potential_energy(
            elements=symbols,
            positions=xyz_qm,
            directory="vacuum",
        )

        vacuum_pot = self.parser.get_vpot(
            mesh=xyz_mm,
            directory="vacuum",
        )

        e_static = np.sum(vacuum_pot * charges_mm) * self._energy_scale

        return e_static

    def _get_qm_mm_induction_energy(self, symbols, xyz_mm, charges_mm, xyz_qm):
        vacuum_energy = self.parser.get_potential_energy(
            elements=symbols,
            positions=xyz_qm,
            directory="vacuum",
        )

        vacuum_pot = self.parser.get_vpot(
            mesh=xyz_mm,
            directory="vacuum",
        )

        e_static = np.sum(vacuum_pot * charges_mm) * self._energy_scale
        qm_mm_energy = self.parser.get_potential_energy(
            elements=symbols,
            positions=xyz_qm,
            orca_external_potentials=np.hstack(
                [np.expand_dims(charges_mm, axis=1), xyz_mm]
            ),
            directory="qm_mm",
        )

        e_int = (qm_mm_energy - vacuum_energy) * self._energy_scale
        e_ind = e_int - e_static

        return e_ind

    def sample(
        self, steps: int
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
        # Integrate for a given number of steps
        self.integrator.step(steps)

        # Get the positions, box vectors, and energy before EMLE to ensure correct positions
        state = self.context.getState(getPositions=True, getEnergy=True)
        pos = state.getPositions(asNumpy=True)
        pbc = state.getPeriodicBoxVectors(asNumpy=True)
        en = state.getPotentialEnergy()

        # EMLE data extraction
        xyz_mm = self.emle_calculator._jm_xyz_mm
        xyz_qm = self.emle_calculator._jm_xyz_qm
        charges_mm = self.emle_calculator._jm_charges_mm
        atomic_numbers = self.emle_calculator._jm_atomic_numbers
        emle_static = self.emle_calculator._jm_e_static.cpu().item()
        emle_ind = self.emle_calculator._jm_e_ind.cpu().item()

        # Convert atomic numbers to symbols
        symbols = [_ATOMIC_NUMBERS_TO_SYMBOLS[an] for an in atomic_numbers]
        external_potentials = np.hstack([np.expand_dims(charges_mm, axis=1), xyz_mm])

        # Vacuum energy and potentials
        vacuum_energy = self.parser.get_potential_energy(
            elements=symbols,
            positions=xyz_qm,
            directory="vacuum",
        )

        vacuum_pot = self.parser.get_vpot(
            mesh=xyz_mm,
            directory="vacuum",
        )
        e_static = np.sum(vacuum_pot * charges_mm) * HARTREE_TO_KJ_MOL

        # QM/MM energy and derived interaction energies
        qm_mm_energy = self.parser.get_potential_energy(
            elements=symbols,
            positions=xyz_qm,
            orca_external_potentials=external_potentials,
            directory="qm_mm",
        )

        e_int = (qm_mm_energy - vacuum_energy) * HARTREE_TO_KJ_MOL
        e_ind = e_int - e_static

        # Logging for debugging
        print(f"Vacuum energy: {vacuum_energy}")
        print(f"QM/MM energy: {qm_mm_energy}")
        print(f"Interaction energy: {e_int}")
        print(f"Static energy: {e_static}")
        print(f"Induction energy: {e_ind}")
        print(f"EMLE static energy: {emle_static}")
        print(f"EMLE induction energy: {emle_ind}")
        print(f"Time: {time.time() - t0}")

        return e_static, e_ind, emle_static, emle_ind, pos, pbc
