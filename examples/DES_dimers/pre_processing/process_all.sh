#!/bin/bash

# Base command
BASE_CMD="python process_des.py --csv DES370K.csv"

# Process all dimers (default)
echo "Processing all dimers..."
$BASE_CMD --output DES370K_all.pkl > DES370K_all.log 2> DES370K_all.err

# Process only water-containing dimers
echo "Processing water-containing dimers..."
$BASE_CMD --water-only --output DES370K_water_only.pkl > DES370K_water_only.log 2> DES370K_water_only.err

# Process with water as mol1
echo "Processing with water as mol1..."
$BASE_CMD --water-as-mol1 --output DES370K_water_mol1.pkl > DES370K_water_mol1.log 2> DES370K_water_mol1.err

# Process with reversed order
echo "Processing with reversed order..."
$BASE_CMD --reverse-order --output DES370K_reversed.pkl > DES370K_reversed.log 2> DES370K_reversed.err

# Process water-containing dimers with water as mol1
echo "Processing water-containing dimers with water as mol1..."
$BASE_CMD --water-only --water-as-mol1 --output DES370K_water_only_mol1.pkl > DES370K_water_only_mol1.log 2> DES370K_water_only_mol1.err

# Process water-containing dimers with reversed order
echo "Processing water-containing dimers with reversed order..."
$BASE_CMD --water-only --reverse-order --output DES370K_water_only_reversed.pkl > DES370K_water_only_reversed.log 2> DES370K_water_only_reversed.err

# Process with water as mol1 and reversed order
echo "Processing with water as mol1 and reversed order..."
$BASE_CMD --water-as-mol1 --reverse-order --output DES370K_water_mol1_reversed.pkl > DES370K_water_mol1_reversed.log 2> DES370K_water_mol1_reversed.err

# Process water-containing dimers with water as mol1 and reversed order
echo "Processing water-containing dimers with water as mol1 and reversed order..."
$BASE_CMD --water-only --water-as-mol1 --reverse-order --output DES370K_water_only_mol1_reversed.pkl > DES370K_water_only_mol1_reversed.log 2> DES370K_water_only_mol1_reversed.err

echo "All processing complete!"
