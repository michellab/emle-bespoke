#!/bin/bash

set -e

N_SAMPLES=1
ML_MODEL=mace-off23-small
N_EQUILIBRATION=10

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <csv_file>"
    exit 1
fi

csv_file=$1

if [ ! -f "$csv_file" ]; then
    echo "File not found: $csv_file"
    exit 1
fi

# Read the CSV file line by line
while IFS=, read -r NAME SMILES; do
    SMILES=$(echo "$SMILES" | tr -d '[:space:]')
    echo "${NAME} ${SMILES}"
    emle-bespoke --filename-prefix "${NAME}" --solute "${SMILES}" --n_sample "${N_SAMPLES}" --ml_model "${ML_MODEL}" --n_equilibration "${N_EQUILIBRATION}" > "${NAME}.log" 2> "${NAME}.err"
done < "$csv_file"
