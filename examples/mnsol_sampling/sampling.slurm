#!/bin/bash
#SBATCH -p main
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time 7-00:00:00

# Capture first positional argumnet
NAME=$1
SMILES=$2
N_SAMPLES=$3
ML_MODEL=$4
N_EQUILIBRATION=$5

# Export environmental variables
emle-bespoke --filename-prefix "${NAME}" --solute "${SMILES}" --n_sample "${N_SAMPLES}" --ml_model "${ML_MODEL}" --n_equilibration "${N_EQUILIBRATION}" > "${NAME}.log" 2> "${NAME}.err"
