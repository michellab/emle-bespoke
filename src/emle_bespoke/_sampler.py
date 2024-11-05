import time
import numpy as np
import torch as _torch
from typing import Tuple
from ._constants import HARTREE_TO_KJ_MOL as _HARTREE_TO_KJ_MOL
from ._constants import ATOMIC_NUMBERS_TO_SYMBOLS as _ATOMIC_NUMBERS_TO_SYMBOLS

import openmm.unit as unit


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
        self._atomic_numbers = _torch.tensor([a.element.atomic_number for a in topology.atoms()], dtype=_torch.int64, device=device)

        # QM settings
        self._qm_region = _torch.tensor(qm_region, dtype=_torch.int64, device=device)
        self._qm_calculator = qm_calculator
        
        # Energy and length scales to convert the units for/from the QM/MM calculations
        self._energy_scale = energy_scale
        self._length_scale = length_scale

        # Reference data lists
        self._qm_energies = []
        self._qm_mm_energies = []

        # Device and dtype
        self._dtype = dtype
        self._device = device

        # Get the point charges
        self._point_charges = self._get_point_charges()


    def _get_point_charges(self):
        non_bonded_force = [f for f in self._system.getForces() if isinstance(f, mm.NonbondedForce)][0]
        point_charges = _torch.zeros(self._topology.getNumAtoms(), dtype=_torch.float64, device=self._device)
        for i in range(non_bonded_force.getNumParticles()):
            _, charge, _ = non_bonded_force.getParticleParameters(i)
            point_charges[i] = charge._value
        return point_charges
        
    '''
    def _get_vacuum_energy(self, symbols, xyz_mm, charges_mm, xyz_qm):
        """
        Get the static energy from the QM/MM calculation.

        Parameters
        ----------
        symbols: list
            List of atomic symbols.
        xyz_mm: np.ndarray
            Array of MM atomic positions.
        charges_mm: np.ndarray
            Array of MM atomic charges.
        xyz_qm: np.ndarray
            Array of QM atomic positions.

        Returns
        -------
        e_static: float
            Static energy from the QM/MM calculation 
        """
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
    '''

    def _wrap_positions(self, positions: _torch.Tensor, boxvectors: _torch.Tensor) -> _torch.Tensor:
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
        positions = positions - _torch.outer(_torch.floor(positions[:,2]/boxvectors[2,2]), boxvectors[2])
        positions = positions - _torch.outer(_torch.floor(positions[:,1]/boxvectors[1,1]), boxvectors[1])
        positions = positions - _torch.outer(_torch.floor(positions[:,0]/boxvectors[0,0]), boxvectors[0])
        return positions
    
    def _center_molecule(self, positions: _torch.Tensor, boxvectors: _torch.Tensor, molecule_mask: _torch.Tensor) -> _torch.Tensor:
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

    def _distance_to_molecule(self, positions: _torch.Tensor, boxvectors: _torch.Tensor, molecule_mask: _torch.Tensor) -> _torch.Tensor:
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

    
    def _write_xyz(self, positions: _torch.Tensor, elements: list[str], filename: str) -> None:
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
        calc_polarizability: bool = True
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

        # Convert the positions to a torch tensor
        pos = _torch.from_numpy(positions).to(self._device, dtype=self._dtype)
        box = _torch.from_numpy(pbc).to(self._device, dtype=self._dtype)

        # Get the molecule mask
        molecule_mask = _torch.zeros(pos.shape[0], dtype=_torch.bool, device=self._device)
        molecule_mask[self._qm_region] = True

        # Center the molecule
        pos = self._center_molecule(pos, box, molecule_mask)

        # Calculate the distance matrix
        R = self._distance_to_molecule(pos, box, molecule_mask)
        R_cutoff = _torch.any(R < 12.0, dim=0)
    
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
            orca_blocks += "%elprop\n    Polar 1\ndipole true\nquadrupole true\nend\n"
        
        vacuum_energy = self._qm_calculator.get_potential_energy(
                elements=symbols,
                positions=pos_qm,
                directory="vacuum",
                orca_blocks=orca_blocks,)
        
        






            
        """
        if calc_induction:
            # Get the point charges and construct the external potential tensor
            charges_mm = self._point_charges[~molecule_mask][R_cutoff]
            external_potentials = _torch.hstack([_torch.unsqueeze(charges_mm, dim=1), pos_mm])
        """

        """
        # Write the XYZ file
        self._write_xyz(_torch.vstack([pos_qm, pos_mm]).cpu().numpy(), 
                        symbols,
                        "test.xyz")
        """
        print(external_potentials)

        """


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
        """


if __name__ == "__main__":

    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    from openmmml import MLPotential
    from sys import stdout
    from .parsers import ORCACalculator as _ORCACalculator
    # Load PDB file and set the FFs
    pdb = app.PDBFile('alanine-dipeptide-explicit.pdb')
    ff = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    # Create the OpenMM MM System and ML potential
    mmSystem = ff.createSystem(pdb.topology, nonbondedMethod=app.PME)
    potential = MLPotential('ani2x')

    # Choose the ML atoms
    mlAtoms = [a.index for a in next(pdb.topology.chains()).atoms()]

    # Create the mixed ML/MM system (we're using the nnpops implementation for performance)
    mixedSystem = potential.createMixedSystem(pdb.topology, mmSystem, mlAtoms, interpolate=False, implementation="nnpops")

    # Choose to run on a GPU (CUDA), with the LangevinMiddleIntegrator (NVT) and create the context
    platform = mm.Platform.getPlatformByName("CUDA")
    integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.001*unit.picoseconds)
    context = mm.Context(mixedSystem, integrator, platform)

    ref_calculator = ReferenceDataCalculator(
        system=mixedSystem,
        context=context,
        integrator=integrator,
        topology=pdb.topology,
        qm_calculator=_ORCACalculator(),
        qm_region=mlAtoms,
        energy_scale=_HARTREE_TO_KJ_MOL,
        length_scale=1.0,
        dtype=_torch.float64,
        device=_torch.device("cuda"),
    )

    context.setPositions(pdb.positions)

    ref_calculator.sample(10)
