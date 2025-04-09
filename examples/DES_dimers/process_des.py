#!/usr/bin/env python3
"""Process DES dimer dataset."""

import argparse
import os as _os
import pickle as _pkl
from collections import defaultdict
from typing import Dict
import time as _time

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
        self.processed_ds = defaultdict(list)

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

        # Get RDKit molecules
        try:
            rdmol0 = self.smiles_to_molecule[smiles0]
            rdmol1 = self.smiles_to_molecule[smiles1]
        except KeyError as e:
            print(f"Skipping system {system_id} due to missing molecule: {e}")
            return

        # Check element consistency
        symbols = [a.GetSymbol() for a in rdmol0.GetAtoms()] + [
            a.GetSymbol() for a in rdmol1.GetAtoms()
        ]
        expected_elements = " ".join(symbols)
        if str(elements) != expected_elements:
            print(
                f"Skipping system {system_id} due to element mismatch: expected {expected_elements}, got {elements}"
            )
            return

        # Create OpenFF molecules
        try:
            mol0 = _Molecule.from_rdkit(
                rdmol0, allow_undefined_stereo=True, hydrogens_are_explicit=True
            )
            mol1 = _Molecule.from_rdkit(
                rdmol1, allow_undefined_stereo=True, hydrogens_are_explicit=True
            )
        except Exception as e:
            print(f"Skipping system {system_id} due to molecule creation error: {e}")
            return

        # Store original order for coordinate handling
        original_order = True
        should_swap = False

        # Handle water position if requested
        if self.water_as_mol1:
            if smiles0 == self.WATER_SMILES:
                should_swap = True
            elif smiles1 == self.WATER_SMILES:
                should_swap = False
            else:
                print(
                    f"Skipping system {system_id} as water position cannot be determined"
                )
                return

        # Handle reverse order if requested
        if self.reverse_order:
            should_swap = not should_swap

        # Swap molecules if needed
        if should_swap:
            original_order = False
            mol0, mol1 = mol1, mol0
            smiles0, smiles1 = smiles1, smiles0
            natoms0, natoms1 = natoms1, natoms0
            charge0, charge1 = charge1, charge0

        # Create topology and interchange
        try:
            off_topology = _Topology.from_molecules([mol0, mol1])
            interchange = _Interchange.from_smirnoff(self.force_field, off_topology)
        except Exception as e:
            print(
                f"Skipping system {system_id} due to topology/interchange creation error: {e}"
            )
            return

        # Pre-compute atomic numbers and masks
        mol0_z = _np.array([a.GetAtomicNum() for a in rdmol0.GetAtoms()])
        mol1_z = _np.array([a.GetAtomicNum() for a in rdmol1.GetAtoms()])
        z = _np.concatenate([mol0_z, mol1_z])

        mol0_mask = _np.zeros(len(z), dtype=bool)
        mol0_mask[: len(mol0_z)] = True
        mol1_mask = _np.zeros(len(z), dtype=bool)
        mol1_mask[len(mol0_z) :] = True

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
            coords = _np.array([float(f) for f in row.xyz.split()]).reshape(
                len(symbols), 3
            )
            energy = row["cbs_CCSD(T)_all"] * _mm.unit.kilocalories_per_mole

            # Reorder coordinates if needed
            if not original_order:
                coords = _np.concatenate([coords[natoms1:], coords[:natoms1]])

            self.processed_ds["e_int"].append(
                energy.value_in_unit(_mm.unit.kilojoule_per_mole)
            )
            self.processed_ds["xyz_mm"].append(coords[mol0_mask])
            self.processed_ds["xyz_qm"].append(coords[mol1_mask])
            self.processed_ds["xyz"].append(coords)
            self.processed_ds["atomic_numbers"].append(z)
            self.processed_ds["charges_mm"].append(charges[mol0_mask])
            self.processed_ds["solute_mask"].append(mol0_mask)
            self.processed_ds["solvent_mask"].append(mol1_mask)
            self.processed_ds["topology"].append(off_topology)

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
            print(f"Processing system {system_id} out of {len(unique_ids)}")
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
        """Save processed results to a pickle file.

        Parameters
        ----------
        output_file : str
            Path to save the output pickle file.
        """
        with open(output_file, "wb") as f:
            _pkl.dump(self.processed_ds, f)


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
        default="DES370K.pkl",
        help="Path to save the output pickle file.",
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
        help="Reverse the order of molecules in the dimer.",
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

    print("\nProcessed data summary:")
    print(f"Total frames: {len(processor.processed_ds['e_int'])}")
    print(f"Time taken: {_time.time() - time0:.2f} seconds")

if __name__ == "__main__":
    main()
