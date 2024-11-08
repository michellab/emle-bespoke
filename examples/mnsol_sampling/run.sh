#!/bin/bash
set -e  

N_SAMPLES=1000
ML_MODEL=mace-off23-small
N_EQUILIBRATION=100000

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <csv_file>"
    exit 1
fi

csv_file=$1

if [ ! -f "$csv_file" ]; then
    echo "File not found: $csv_file"
    exit 1
fi

# Open the file with file descriptor 9
exec 9< "$csv_file"

while IFS=, read -u 9 NAME SMILES; do
    SMILES=$(echo "$SMILES" | tr -d '[:space:]')
    echo "Processing: ${NAME} ${SMILES}"

    # Remove the previous files
    rm -rf pc vacuum

    emle-bespoke --filename-prefix "${NAME}" --solute "${SMILES}" --n_sample "${N_SAMPLES}" --ml_model "${ML_MODEL}" --n_equilibration "${N_EQUILIBRATION}" > "${NAME}.log" 2> "${NAME}.err" || { echo "Error processing ${NAME}"; continue; }
done

# Close the file descriptor 9
exec 9>&-
