# DES370K Lennard-Jones Fitting

## Instructions

1. Process the 


## Note

The SDF files in the `SDFS` directory were downloaded from https://github.com/openmm/spice-dataset/tree/main/des370k. This is required to solve the ambiguity regarding the mapping from smiles to atomic coordinates in the order given in the `.csv` files. See the comment:

    The dataset describes each sample using SMILES strings for the monomers, and lists the full atomic coordinates for each conformation.  Because the SMILES strings have implicit hydrogens, the exact mapping of coordinates to atoms is ambiguous.  To resolve this, the SDFS directory contains SDF files provided by Alexander Donchev, one for each monomer, listing all atoms in the order used by the dataset.

## DES Dimer Processing

The script `process_des.py` processes the DES370K dataset from an CSV file.


### Input Files

1. CSV file containing dimer data with the following columns:
   - `system_id`: Unique identifier for each dimer system
   - `smiles0`, `smiles1`: SMILES strings for the two molecules
   - `natoms0`, `natoms1`: Number of atoms in each molecule
   - `charge0`, `charge1`: Charges of the molecules
   - `elements`: Space-separated string of element symbols
   - `xyz`: Cartesian coordinates in Angstrom
   - `cbs_CCSD(T)_all`: Interaction energy in kcal/mol

2. SDF files:
   - Directory containing SDF files for each molecule
   - Filename should match the SMILES string (e.g., "O.sdf" for water)

### Usage

```bash
python final.py --csv <input_csv> [options]
```

### Required Arguments

- `--csv`: Path to the input CSV file containing DES dimer data

### Optional Arguments

- `--output`: Path to save the output pickle file (default: "DES370K_processed.pkl")
- `--sdf-dir`: Directory containing SDF files (default: "SDFS")
- `--force-field`: Path to the force field file (default: "openff-2.0.0.offxml")
- `--water-only`: Only process dimers containing water
- `--reverse-order`: Reverse the order of monomers in the dimer (`mol0` and `mol1` become the solvent and solute, respectively)
- `--water-as-mol1`: Ensure water is always the solvent (second monomer, `mol1`) when present

### Examples

1. Process all dimers:
```bash
python final.py --csv DES370K.csv
```

2. Process only water-containing dimers:
```bash
python final.py --csv DES370K.csv --water-only
```

3. Ensure water is always the solvent (second monomer, `mol1`)
```bash
python final.py --csv DES370K.csv --water-as-mol1
```

4. Reverse monomer order (`mol0` and `mol1` become the solvent and solute, respectively)
```bash
python final.py --csv DES370K.csv --reverse-order
```

5. Process only water-containing dimers and ensure water is always the solvent (second monomer, `mol1`):
```bash
python final.py --csv DES370K.csv --water-only --water-as-mol1
```

## Output

The script generates a pickle file containing the processed data with the following keys:
- `e_int`: Dimer interaction energies [kj/mol]
- `xyz_mm`: Coordinates of the first monomer [Angstrom]
- `xyz_qm`: Coordinates of the second monomer [Angstrom]
- `xyz`: Combined coordinates [Angstrom]
- `atomic_numbers`: Atomic numbers [integer] 
- `charges_mm`: Charges of the first monomer [elementary charge]
- `solute_mask`: Mask for the first monomer [boolean]
- `solvent_mask`: Mask for the second monomer [boolean]
- `topology`: OpenFF topology objects [OpenFF Topolgy]

## Filtering

The following filters are applied:

1. Only dimers with H, C, N, O, S are allowed (EMLE-supported elements)
2. Both molecules must be neutral
