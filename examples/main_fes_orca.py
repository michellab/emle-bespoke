import os
import time
from typing import Tuple
import pandas as pd
import numpy as np
import openff.units as offunit
import openmm.unit as unit
from fes_ml.alchemical.modifications import _EMLE_CALCULATORS
from fes_ml.fes import FES
import shutil
from scipy.optimize import minimize

from _constants import ATOMIC_NUMBERS_TO_SYMBOLS, HARTREE_TO_KJ_MOL
from _orca_parser import ORCACalculator
import torch

torch.inverse(torch.ones((1, 1), device="cuda:0"))

def get_model_data(context, emle_calculator, positions, pbc, *args, **kwargs):
    e_static_emle = []
    e_ind_emle = []

    for frame, box_vectors in zip(positions, pbc):
        context.setPeriodicBoxVectors(*box_vectors)
        context.setPositions(frame)
        context.getState(getEnergy=True).getPotentialEnergy()

        e_static_emle.append(emle_calculator._jm_e_static.cpu().item())
        e_ind_emle.append(emle_calculator._jm_e_ind.cpu().item())

    return np.array(e_static_emle) + np.array(e_ind_emle), e_static_emle, e_ind_emle


def set_model_parameters(x, *args, **kwargs):
    # Set environment variable EMLE_SCREENING_FACTOR equal to x
    """
    if isinstance(x, np.ndarray) or isinstance(x, list):
        x = float(x[0])
        x = round(x, 2)

    os.environ["EMLE_SCREENING_TYPE"] = "dielectric"
    os.environ["EMLE_SCREENING_FACTOR"] = str(x)
    """

    os.environ["EMLE_SCREENING_TYPE"] = "dielectric"
    os.environ["EMLE_SCREENING_FACTOR"] = str(1.0)
    os.environ["EMLE_STATIC_FACTOR"] = str(x[0])
    os.environ["EMLE_IND_FACTOR"] = str(x[1])


def _loss_function(
    x, ref_data: np.ndarray, context, emle_calculator, positions, pbc, *args, **kwargs
) -> float:
    set_model_parameters(x, *args, **kwargs)
    model_data, _, _ = get_model_data(
        context, emle_calculator, positions, pbc, *args, **kwargs
    )
 
    return np.mean((ref_data - model_data) ** 2) / np.std(ref_data)

def loss_function(
    x, 
    e_total_ref: np.ndarray, 
    e_static_ref: np.ndarray, 
    e_ind_ref: np.ndarray, 
    context, 
    emle_calculator, 
    positions, 
    pbc, *args, **kwargs
) -> float:
    
    set_model_parameters(x, *args, **kwargs)
    e_total_emle, e_static_emle, e_ind_emle = get_model_data(
        context, emle_calculator, positions, pbc, *args, **kwargs
    )

    loss_static = np.mean((e_static_ref - e_static_emle) ** 2) / np.std(e_static_ref)
    loss_ind = np.mean((e_ind_ref - e_ind_emle) ** 2) / np.std(e_ind_ref)
    loss = loss_static + loss_ind
    print("-" * 50)
    print(f"Parameters: {x}")
    print(f"Loss: {loss} | Static: {loss_static} | Ind: {loss_ind}")
    return loss


def get_reference_data_inner_loop(
    steps: int, parser: ORCACalculator, system, context, integrator, emle_calculator
) -> Tuple[float, float, float, float]:
    t0 = time.time()
    integrator.step(steps)

    # Get the positions, box vectors, and energy before EMLE
    # Otherwise EMLE will have the wrong positions
    state = context.getState(getPositions=True, getEnergy=True)
    pos = state.getPositions(asNumpy=True)
    pbc = state.getPeriodicBoxVectors(asNumpy=True)
    en = state.getPotentialEnergy()

    xyz_mm = emle_calculator._jm_xyz_mm
    xyz_qm = emle_calculator._jm_xyz_qm
    charges_mm = emle_calculator._jm_charges_mm
    atomic_numbers = emle_calculator._jm_atomic_numbers
    emle_static = emle_calculator._jm_e_static.cpu().item()
    emle_ind = emle_calculator._jm_e_ind.cpu().item()

    symbols = [ATOMIC_NUMBERS_TO_SYMBOLS[an] for an in atomic_numbers]
    external_potentials = np.hstack([np.expand_dims(charges_mm, axis=1), xyz_mm])

    vacuum_energy = parser.get_potential_energy(
        elements=symbols,
        positions=xyz_qm,
        directory="vacuum",
    )

    vacuum_pot = parser.get_vpot(
        mesh=xyz_mm,
        directory="vacuum",
    )
    e_static = np.sum(vacuum_pot * charges_mm) * HARTREE_TO_KJ_MOL

    qm_mm_energy = parser.get_potential_energy(
        elements=symbols,
        positions=xyz_qm,
        orca_external_potentials=external_potentials,
        directory="qm_mm",
    )

    e_int = (qm_mm_energy - vacuum_energy) * HARTREE_TO_KJ_MOL
    e_ind = e_int - e_static
    
    print(f"Vacuum energy: {vacuum_energy}")
    print(f"QM/MM energy: {qm_mm_energy}")
    print(f"Interaction energy: {e_int}")
    print(f"Static energy: {e_static}")
    print(f"Induction energy: {e_ind}")
    print(f"EMLE static energy: {emle_static}")
    print(f"EMLE induction energy: {emle_ind}")
    print(f"Time: {time.time() - t0}")

    return e_static, e_ind, emle_static, emle_ind, pos, pbc


def optimize_screening_factor(
    system, context, integrator, emle_calculator, n_samples, n
):
    pass


def main(mol_name, smiles):
    print(f"Running optimization for {mol_name} with SMILES {smiles}")
    lambda_schedule = {"EMLEPotential": [1.0], "MLInterpolation": [1.0]}
    modifications_kwargs = {"MLPotential": {"name": "mace-off23-small"}}

    temperature = 298.15 * unit.kelvin
    dt = 1.0 * unit.femtosecond

    mdconfig_dict = {
        "periodic": True,
        "constraints": "h-bonds",
        "vdw_method": "cutoff",
        "vdw_cutoff": offunit.Quantity(12.0, "angstrom"),
        "mixing_rule": "lorentz-berthelot",
        "switching_function": True,
        "switching_distance": offunit.Quantity(11.0, "angstrom"),
        "coul_method": "pme",
        "coul_cutoff": offunit.Quantity(12.0, "angstrom"),
    }

    # Define the kwargs for the creation of the alchemical states
    # Alternatively, these kwargs can be passed directly to the create_alchemical_states method
    # This uses OpenMM units
    create_alchemical_states_kwargs = {
        "smiles_ligand": smiles,
        "smiles_solvent": "[H:2][O:1][H:3]",
        "integrator": None,
        "forcefields": ["openff_unconstrained-2.0.0.offxml", "tip3p.offxml"],
        "temperature": temperature,
        "timestep": dt,  # ignored if integrator is passed
        "pressure": None,
        "hydrogen_mass": 1.007947 * unit.amu,  # use this for HMR
        "mdconfig_dict": mdconfig_dict,
        "modifications_kwargs": modifications_kwargs,
    }

    # --------------------------------------------------------------- #
    # Prepare and run the simulations
    # --------------------------------------------------------------- #
    # Create the FES object and add the alchemical states
    fes = FES()
    fes.create_alchemical_states(
        strategy_name="openff",
        lambda_schedule=lambda_schedule,
        **create_alchemical_states_kwargs,
    )

    window = 0


    # Get the system
    omm_system = fes.alchemical_states[window].system
    omm_context = fes.alchemical_states[window].context
    omm_integrator = fes.alchemical_states[window].integrator

    from fes_ml.alchemical.modifications import _EMLE_CALCULATORS

    # Get the EMLE calculator
    emle_calculator = _EMLE_CALCULATORS[0]
    emle_calculator._is_interpolate = False

    e_static_ref = []
    e_ind_ref = []
    e_static_emle = []
    e_ind_emle = []
    pos_ref = []
    pbc_ref = []

    # Create the ORCA calculator
    parser = ORCACalculator()
    sample = True

    pairs = {
        "e_static_ref": e_static_ref,
        "e_ind_ref": e_ind_ref,
        "e_static_emle": e_static_emle,
        "e_ind_emle": e_ind_emle,
        "pos_ref": pos_ref,
        "pbc_ref": pbc_ref,
    }

    if sample:
        # Equilibrate the state of interest
        fes.equilibrate(100000, window=window)

        # Set initial velocities
        fes.set_velocities(temperature=temperature, window=window)
        for i in range(100):
            print(f"Step {i}")
            (
                e_static,
                e_ind,
                emle_static,
                emle_ind,
                pos,
                pbc,
            ) = get_reference_data_inner_loop(
                1000, parser, omm_system, omm_context, omm_integrator, emle_calculator
            )

            e_static_ref.append(e_static)
            e_ind_ref.append(e_ind)
            e_static_emle.append(emle_static)
            e_ind_emle.append(emle_ind)
            pos_ref.append(pos)
            pbc_ref.append(pbc)

        # Save to pickles all ref data
        import pickle

        for key, value in pairs.items():
            with open(f"{mol_name}_{key}.pkl", "wb") as f:
                pickle.dump(value, f)


    if not sample:
        # Load from pickles all ref data
        import pickle

        for key in pairs.keys():
            with open(f"{mol_name}_{key}.pkl", "rb") as f:
                pairs[key] = pickle.load(f)

        e_static_ref = pairs["e_static_ref"][1:]
        e_ind_ref = pairs["e_ind_ref"][1:]
        e_static_emle = pairs["e_static_emle"][1:]
        e_ind_emle = pairs["e_ind_emle"][1:]
        pos_ref = pairs["pos_ref"][1:]
        pbc_ref = pairs["pbc_ref"][1:]


    e_total_tmp, e_static_tmp, e_ind_tmp = get_model_data(
        omm_context, emle_calculator, pos_ref, pbc_ref
    )

    print("E_ind", e_ind_tmp)
    exit()


    import matplotlib.pyplot as plt
    plt.plot(e_ind_ref)
    plt.savefig("e_ind_ref.png") 


    e_static_ref = np.array(e_static_ref)
    e_ind_ref = np.array(e_ind_ref)
    total_energy_ref = e_static_ref + e_ind_ref

    set_model_parameters([1, 1])
    e_total_emle, e_static_emle, e_ind_emle = get_model_data(
        omm_context, emle_calculator, pos_ref, pbc_ref
    )
   
    print("Minimizing...")
    res = minimize(
        loss_function,
        [1, 1],
        args=(total_energy_ref, e_static_ref, e_ind_ref, omm_context, emle_calculator, pos_ref, pbc_ref),
        method="cobyla",
        options={"disp": True, "maxiter": 1000},
    )
    print("Minimization done.")

    # Plot and save the results as stacked subplots
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 2, figsize=(12, 15))

    # x=y line
    x = np.linspace(-500, 500, 50)

    # convert to numpy arrays
    e_static_ref = np.array(e_static_ref)
    e_ind_ref = np.array(e_ind_ref)
    e_static_emle = np.array(e_static_emle)
    e_ind_emle = np.array(e_ind_emle)

    # Auto limits
    lim_low = np.min(e_static_ref) - 10
    lim_max = np.max(e_static_ref) + 10

    # Static energy before optimization
    axs[0, 0].plot(x, x, color="black")
    axs[0, 0].scatter(e_static_emle, e_static_ref)
    axs[0, 0].set_ylabel("Reference static energy (kJ/mol)")
    axs[0, 0].set_xlabel("EMLE static energy (kJ/mol)")
    axs[0, 0].set_xlim(lim_low, lim_max)
    axs[0, 0].set_ylim(lim_low, lim_max)

    # RMSE on plot
    rmse = np.sqrt(np.mean((e_static_emle - e_static_ref) ** 2))
    axs[0, 0].set_title(f"Static Energy Before Optimization, RMSE: {rmse:.2f}")

    # Auto limits
    lim_low = np.min(e_ind_ref) - 10
    lim_max = np.max(e_ind_ref) + 10
    # Induction energy before optimization
    axs[0, 1].plot(x, x, color="black")
    axs[0, 1].scatter(e_ind_emle, e_ind_ref)
    axs[0, 1].set_ylabel("Reference induction energy (kJ/mol)")
    axs[0, 1].set_xlabel("EMLE induction energy (kJ/mol)")
    axs[0, 1].set_xlim(lim_low, lim_max)
    axs[0, 1].set_ylim(lim_low, lim_max)

    # RMSE on plot
    rmse = np.sqrt(np.mean((e_ind_emle - e_ind_ref) ** 2))
    axs[0, 1].set_title(f"Induction Energy Before Optimization, RMSE: {rmse:.2f}")

    set_model_parameters(res.x)
    e_total_opt, e_static_opt, e_ind_opt = get_model_data(
        omm_context, emle_calculator, pos_ref, pbc_ref
    )

    # convert to numpy arrays
    e_static_opt = np.array(e_static_opt)
    e_ind_opt = np.array(e_ind_opt)

    # Auto limits
    lim_low = np.min(e_static_ref) - 10
    lim_max = np.max(e_static_ref) + 10

    # Static energy after optimization
    axs[1, 0].plot(x, x, color="black")
    axs[1, 0].scatter(e_static_opt, e_static_ref)
    axs[1, 0].set_ylabel("Reference static energy (kJ/mol)")
    axs[1, 0].set_xlabel("EMLE static energy (kJ/mol)")
    axs[1, 0].set_xlim(lim_low, lim_max)
    axs[1, 0].set_ylim(lim_low, lim_max)

    rmse = np.sqrt(np.mean((e_static_opt - e_static_ref) ** 2))

    axs[1, 0].set_title(f"Static Energy After Optimization, RMSE: {rmse:.2f}")

    # Auto limits
    lim_low = np.min(e_ind_ref) - 10
    lim_max = np.max(e_ind_ref) + 10

    # Induction energy after optimization
    axs[1, 1].plot(x, x, color="black")
    axs[1, 1].scatter(e_ind_opt, e_ind_ref)
    axs[1, 1].set_ylabel("Reference induction energy (kJ/mol)")
    axs[1, 1].set_xlabel("EMLE induction energy (kJ/mol)")
    axs[1, 1].set_xlim(lim_low, lim_max)
    axs[1, 1].set_ylim(lim_low, lim_max)

    rmse = np.sqrt(np.mean((e_ind_opt - e_ind_ref) ** 2))
    axs[1, 1].set_title(f"Induction Energy After Optimization, RMSE: {rmse:.2f}")

    # Auto limits
    lim_low = np.min(total_energy_ref) - 10
    lim_max = np.max(total_energy_ref) + 10

    # Total energy before optimization
    axs[2, 0].plot(x, x, color="black")
    axs[2, 0].scatter(e_total_emle, total_energy_ref)
    axs[2, 0].set_ylabel("Reference total energy (kJ/mol)")
    axs[2, 0].set_xlabel("EMLE total energy (kJ/mol)")
    axs[2, 0].set_xlim(lim_low, lim_max)
    axs[2, 0].set_ylim(lim_low, lim_max)

    rmse = np.sqrt(np.mean((e_total_emle - total_energy_ref) ** 2))
    axs[2, 0].set_title(f"Total Energy Before Optimization, RMSE: {rmse:.2f}")

    # Auto limits
    lim_low = np.min(total_energy_ref) - 10
    lim_max = np.max(total_energy_ref) + 10

    # Total energy after optimization
    axs[2, 1].plot(x, x, color="black")
    axs[2, 1].scatter(e_total_opt, total_energy_ref)
    axs[2, 1].set_ylabel("Reference total energy (kJ/mol)")
    axs[2, 1].set_xlabel("EMLE total energy (kJ/mol)")
    axs[2, 1].set_xlim(lim_low, lim_max)
    axs[2, 1].set_ylim(lim_low, lim_max)

    rmse = np.sqrt(np.mean((e_total_opt - total_energy_ref) ** 2))
    axs[2, 1].set_title(f"Total Energy After Optimization, RMSE: {rmse:.2f}")    

    plt.tight_layout()
    plt.savefig(f"{mol_name}_energy_comparison.png")

    print(f"{mol_name} optimized parameters:", res.x)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run EMLE optimization.")
    parser.add_argument("mol_name", type=str, help="The name of the molecule.")
    parser.add_argument("smiles", type=str, help="The SMILES string of the molecule.")
    args = parser.parse_args()
    main(args.mol_name, args.smiles)

    shutil.rmtree("vacuum")
    shutil.rmtree("qm_mm")
