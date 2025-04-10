#!/usr/bin/env python3
"""Process DES dimer dataset."""

import argparse
import os as _os
import time as _time
from typing import Dict

import numpy as _np
import openmm as _mm
import pandas as pd
from openff.interchange import Interchange as _Interchange
from openff.toolkit import (
    ForceField as _ForceField,
)
from openff.toolkit import (
    Molecule as _Molecule,
)
from openff.toolkit import (
    Topology as _Topology,
)
from rdkit import Chem as _Chem

from emle_bespoke.reference_data import ReferenceDataset as _ReferenceDataset


class DESDimerProcessor:
    """Process DES dimer data with support for water molecule selection and order reversal."""

    ALLOWED_ELEMENTS = {"H", "C", "N", "O", "S"}
    WATER_SMILES = "O"

    def __init__(
        self,
        sdf_dir: str = "SDFS",
        force_field: str = "openff-2.0.0.offxml",
        water_only: bool = True,
        reverse_order: bool = False,
        water_as_mol1: bool = False,
    ):
        """Initialize the processor.

        Parameters
        ----------
        sdf_dir : str, optional
            Directory containing SDF files. Default is 'SDFS'.
        force_field : str, optional
            Path to the force field file. Default is 'openff-2.0.0.offxml'.
        water_only : bool, optional
            If True, only process dimers containing water. Default is True.
        reverse_order : bool, optional
            If True, reverse the order of molecules in the dimer. Default is False.
        water_as_mol1 : bool, optional
            If True, ensure water is always the second molecule (mol1) when present.
            Default is False.
        """
        self.sdf_dir = sdf_dir
        self.force_field = _ForceField(force_field)
        self.water_only = water_only
        self.reverse_order = reverse_order
        self.water_as_mol1 = water_as_mol1
        self.smiles_to_molecule = self._load_sdf_files()
        self.processed_ds = _ReferenceDataset()

    def _load_sdf_files(self) -> Dict[str, _Chem.Mol]:
        """Load SDF files into a dictionary mapping SMILES to RDKit molecules.

        Returns
        -------
        Dict[str, _Chem.Mol]
            Dictionary mapping SMILES to RDKit molecules.
        """
        smiles_to_molecule = {}
        for filename in _os.listdir(self.sdf_dir):
            if not filename.endswith(".sdf"):
                continue
            smiles = filename[:-4]
            supp = _Chem.SDMolSupplier(
                f"{self.sdf_dir}/{filename}", sanitize=False, removeHs=False
            )
            smiles_to_molecule[smiles] = list(supp)[0]
        return smiles_to_molecule

    def _validate_elements(self, elements: str) -> bool:
        """Validate that all elements are in the allowed set.

        Parameters
        ----------
        elements : str
            Space-separated string of element symbols.

        Returns
        -------
        bool
            True if all elements are allowed, False otherwise.
        """
        return all(elem in self.ALLOWED_ELEMENTS for elem in set(elements.split()))

    def _validate_charges(self, charge0: int, charge1: int) -> bool:
        """Validate that both molecules are neutral.

        Parameters
        ----------
        charge0, charge1 : int
            Charges of the molecules.

        Returns
        -------
        bool
            True if both molecules are neutral, False otherwise.
        """
        return charge0 == 0 and charge1 == 0

    def _validate_water_presence(self, smiles0: str, smiles1: str) -> bool:
        """Validate that at least one molecule is water if water_only is True.

        Parameters
        ----------
        smiles0, smiles1 : str
            SMILES strings of the molecules.

        Returns
        -------
        bool
            True if water is present (when required), False otherwise.
        """
        if not self.water_only:
            return True
        return smiles0 == self.WATER_SMILES or smiles1 == self.WATER_SMILES

    def _process_dimer(
        self,
        df_filtered: pd.DataFrame,
        system_id: str,
        smiles0: str,
        smiles1: str,
        natoms0: int,
        natoms1: int,
        charge0: int,
        charge1: int,
        elements: str,
    ) -> None:
        """Process a single dimer system.

        Parameters
        ----------
        df_filtered : pd.DataFrame
            Filtered DataFrame containing system data.
        system_id : str
            System identifier.
        smiles0, smiles1 : str
            SMILES strings of the molecules.
        natoms0, natoms1 : int
            Number of atoms in each molecule.
        charge0, charge1 : int
            Charges of the molecules.
        elements : str
            Space-separated string of element symbols.
        """
        if not self._validate_elements(elements):
            print(
                f"Skipping system {system_id} due to unsupported elements: {set(elements.split())}"
            )
            return

        if not self._validate_charges(charge0, charge1):
            print(
                f"Skipping system {system_id} due to non-zero charges: {charge0}, {charge1}"
            )
            return

        if not self._validate_water_presence(smiles0, smiles1):
            print(f"Skipping system {system_id} as it doesn't contain water")
            return

        # Determine which molecule should be solute (QM) and which should be solvent (MM)
        # By default, mol0 is solute (QM) and mol1 is solvent (MM)
        solute_smiles = smiles0
        solvent_smiles = smiles1
        solute_natoms = natoms0
        solvent_natoms = natoms1
        solute_charge = charge0
        solvent_charge = charge1
        needs_coord_swap = False  # Track if we need to swap coordinates

        # Handle water position if requested
        if self.water_as_mol1:
            if smiles0 == self.WATER_SMILES:
                # Swap to make water the solvent (MM)
                solute_smiles, solvent_smiles = solvent_smiles, solute_smiles
                solute_natoms, solvent_natoms = solvent_natoms, solute_natoms
                solute_charge, solvent_charge = solvent_charge, solute_charge
                needs_coord_swap = True
            elif smiles1 == self.WATER_SMILES:
                # Already correct, water is solvent (MM)
                pass
            else:
                print(
                    f"Skipping system {system_id} as water position cannot be determined"
                )
                return

        # Handle reverse order if requested
        if self.reverse_order:
            solute_smiles, solvent_smiles = solvent_smiles, solute_smiles
            solute_natoms, solvent_natoms = solvent_natoms, solute_natoms
            solute_charge, solvent_charge = solvent_charge, solute_charge
            needs_coord_swap = not needs_coord_swap

        if needs_coord_swap:
            elements = elements.split()
            elements = " ".join(elements[natoms0:]) + " " + " ".join(elements[:natoms0])

        # Get RDKit molecules
        try:
            solute_rdmol = self.smiles_to_molecule[solute_smiles]
            solvent_rdmol = self.smiles_to_molecule[solvent_smiles]
        except KeyError as e:
            print(f"Skipping system {system_id} due to missing molecule: {e}")
            return

        # Check element consistency
        symbols = [a.GetSymbol() for a in solute_rdmol.GetAtoms()] + [
            a.GetSymbol() for a in solvent_rdmol.GetAtoms()
        ]
        expected_elements = " ".join(symbols)
        if str(elements) != expected_elements:
            print(
                f"Skipping system {system_id} due to element mismatch: expected {expected_elements}, got {elements}"
            )
            exit()
            return

        # Create OpenFF molecules
        try:
            solute_mol = _Molecule.from_rdkit(
                solute_rdmol, allow_undefined_stereo=True, hydrogens_are_explicit=True
            )
            solvent_mol = _Molecule.from_rdkit(
                solvent_rdmol, allow_undefined_stereo=True, hydrogens_are_explicit=True
            )
        except Exception as e:
            print(f"Skipping system {system_id} due to molecule creation error: {e}")
            return

        # Create topology and interchange
        try:
            off_topology = _Topology.from_molecules([solute_mol, solvent_mol])
            interchange = _Interchange.from_smirnoff(self.force_field, off_topology)
        except Exception as e:
            print(
                f"Skipping system {system_id} due to topology/interchange creation error: {e}"
            )
            return

        # Pre-compute atomic numbers and masks
        solute_z = _np.array([a.GetAtomicNum() for a in solute_rdmol.GetAtoms()])
        solvent_z = _np.array([a.GetAtomicNum() for a in solvent_rdmol.GetAtoms()])
        z = _np.concatenate([solute_z, solvent_z])

        solute_mask = _np.zeros(len(z), dtype=bool)
        solute_mask[: len(solute_z)] = True
        solvent_mask = _np.zeros(len(z), dtype=bool)
        solvent_mask[len(solute_z) :] = True

        # Get charges from OpenMM system
        omm_system = interchange.to_openmm()
        nonbonded_force = next(
            f for f in omm_system.getForces() if isinstance(f, _mm.NonbondedForce)
        )
        charges = _np.array(
            [
                nonbonded_force.getParticleParameters(i)[0].value_in_unit(
                    _mm.unit.elementary_charge
                )
                for i in range(nonbonded_force.getNumParticles())
            ]
        )

        # Process each configuration
        for _, row in df_filtered.iterrows():
            # Get coordinates in original order (smiles0, smiles1)
            coords = _np.array([float(f) for f in row.xyz.split()]).reshape(
                len(symbols), 3
            )
            energy = row["cbs_CCSD(T)_all"] * _mm.unit.kilocalories_per_mole

            # Reorder coordinates if needed to match solute/solvent ordering
            if needs_coord_swap:
                coords = _np.concatenate([coords[natoms0:], coords[:natoms0]])

            # Add data to ReferenceDataset
            self.processed_ds.append(
                {
                    "e_int_target": energy.value_in_unit(_mm.unit.kilojoule_per_mole),
                    "xyz_mm": coords[solvent_mask],  # Solvent (MM) coordinates
                    "xyz_qm": coords[solute_mask],  # Solute (QM) coordinates
                    "xyz": coords,
                    "atomic_numbers": solute_z,  # Solute (QM) atomic numbers
                    "zzz": z,
                    "charges_mm": charges[solvent_mask],  # Solvent (MM) charges
                    "solute_mask": solute_mask,  # Solute (QM) mask
                    "solvent_mask": solvent_mask,  # Solvent (MM) mask
                    "topology": off_topology,
                }
            )

    def write_xyz(self, processed_ds: _ReferenceDataset, filename: str) -> None:
        """Write processed dataset to XYZ file.

        Parameters
        ----------
        processed_ds : _ReferenceDataset
            The processed dataset containing molecular configurations
        filename : str
            Path to the output XYZ file

        Notes
        -----
        Writes molecular configurations in XYZ format with atomic coordinates and symbols.
        Each frame includes the number of atoms and a comment line with the system index.
        """
        # Map atomic numbers to element symbols
        ATOMIC_NUMBERS = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S", 9: "F", 17: "Cl"}
        with open(filename, "w") as f:
            for i in range(len(processed_ds)):
                xyz = processed_ds._data["xyz"][i]
                atomic_numbers = processed_ds._data["zzz"][i]

                # Write number of atoms and comment line
                f.write(f"{len(xyz)}\n")
                f.write(
                    f"System {i + 1}, Energy: {processed_ds._data['e_int_target'][i]:.4f} kJ/mol\n"
                )

                # Write atomic coordinates with element symbols
                for atom_id, (x, y, z) in enumerate(xyz):
                    element = ATOMIC_NUMBERS.get(atomic_numbers[atom_id].item(), "X")
                    f.write(f"{element:2s} {x:10.6f} {y:10.6f} {z:10.6f}\n")

    def process_csv(self, csv_file: str) -> None:
        """Process a CSV file containing DES dimer data.

        Parameters
        ----------
        csv_file : str
            Path to the CSV file.
        """
        df = pd.read_csv(csv_file)
        unique_ids = df["system_id"].unique()

        for system_id in unique_ids:
            print("--------------------------------")
            print(f"Processing system {system_id}")
            df_filtered = df[df["system_id"] == system_id]
            row = df_filtered.iloc[0]

            self._process_dimer(
                df_filtered=df_filtered,
                system_id=system_id,
                smiles0=row["smiles0"],
                smiles1=row["smiles1"],
                natoms0=row["natoms0"],
                natoms1=row["natoms1"],
                charge0=row["charge0"],
                charge1=row["charge1"],
                elements=row["elements"],
            )

    def save_results(self, output_file: str) -> None:
        """Save processed results to a file.

        Parameters
        ----------
        output_file : str
            Path to save the output file.
        """
        print("Saving results to", output_file)
        self.processed_ds.write(output_file)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Process DES dimer data.")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="DES370K.h5",
        help="Path to save the output file.",
    )
    parser.add_argument(
        "--sdf-dir",
        type=str,
        default="SDFS",
        help="Directory containing SDF files.",
    )
    parser.add_argument(
        "--force-field",
        type=str,
        default="openff-2.0.0.offxml",
        help="Path to the force field file.",
    )
    parser.add_argument(
        "--water-only",
        action="store_true",
        help="Only process dimers containing water.",
    )
    parser.add_argument(
        "--reverse-order",
        action="store_true",
        help="Reverse the order of molecules in the dimer. Performed after --water_as_mol1 if both are specified.",
    )
    parser.add_argument(
        "--water-as-mol1",
        action="store_true",
        help="Ensure water is always the second molecule (mol1) when present.",
    )

    args = parser.parse_args()

    time0 = _time.time()

    processor = DESDimerProcessor(
        sdf_dir=args.sdf_dir,
        force_field=args.force_field,
        water_only=args.water_only,
        reverse_order=args.reverse_order,
        water_as_mol1=args.water_as_mol1,
    )

    processor.process_csv(args.csv)
    processor.save_results(args.output)
    processor.write_xyz(processor.processed_ds, args.output.replace(".pkl", ".xyz"))

    print("\nProcessed data summary:")
    print(f"Total frames: {len(processor.processed_ds)}")
    print(f"Time taken: {_time.time() - time0:.2f} seconds")


if __name__ == "__main__":
    main()
