import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
import pandas as pd
import seaborn as sns
import torch
from emle.models import EMLE
from openff.interchange import Interchange as _Interchange
from openff.toolkit import ForceField as _ForceField
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from emle_bespoke._constants import HARTREE_TO_KJ_MOL
from emle_bespoke.bespoke import BespokeModelTrainer as _BespokeModelTrainer
from emle_bespoke.lj_fitting import LennardJonesPotential as _LennardJonesPotential
from emle_bespoke.utils import create_dimer_topology

force_field = _ForceField("openff-2.0.0.offxml")
read = True
if read:
    with open("/home/joaomorado/repos/emle-bespoke/utils/DES370K-water.pkl", "rb") as f:
        results = pickle.load(f)
    with open("/home/joaomorado/repos/emle-bespoke/utils/dataset-water.pkl", "rb") as f:
        dataset = pickle.load(f)
    rg = 1
else:
    # df = pd.read_csv('/home/joaomorado/repos/emle-bespoke/utils/DESS66x8.csv')
    # df = pd.read_csv('/home/joaomorado/repos/emle-bespoke/utils/DESS66x8.csv')
    df = pd.read_csv("/home/joaomorado/repos/emle-bespoke/utils/DES370K.csv")
    # Pre-allocate lists for results
    results = defaultdict(list)

    # Initialize force field and interchange object
    force_field = _ForceField("openff-2.0.0.offxml")

    # Get unique system IDs
    unique_ids = df["system_id"].unique()
    allowed_elements = ["H", "C", "N", "O", "S"]

    dataset = {}

    # Loop over unique IDs
    frame = 0
    for system_id in unique_ids:
        print(f"Processing system {system_id} out of {len(unique_ids)}")
        # Filter DataFrame for the current system ID
        df_filtered = df[df["system_id"] == system_id]
        smiles0, smiles1 = df_filtered.iloc[0][["smiles0", "smiles1"]]
        natoms0, natoms1 = df_filtered.iloc[0][["natoms0", "natoms1"]]
        charge0, charge1 = df_filtered.iloc[0][["charge0", "charge1"]]
        elements = df_filtered.iloc[0]["elements"]

        # Create xyz
        # xyz = np.fromstring(df_filtered.iloc[0]["xyz"], sep=" ").reshape(-1, 3)

        # if smiles0 != "O" and smiles1 != "O":
        #    continue

        if not all(elem in allowed_elements for elem in set(elements.split())):
            print(
                f"Skipping system {system_id} due to unsupported elements: {set(elements.split())}"
            )
            continue
        if charge0 != 0 or charge1 != 0:
            print(
                f"Skipping system {system_id} due to non-zero charges: {charge0}, {charge1}"
            )
            continue

        rg = 2
        for i in range(rg):
            if rg == 1:
                if smiles0 == "O":
                    i = 1
                else:
                    i = 0

            if i == 0:
                solute, solvent = smiles0, smiles1
                solute_idx = list(range(natoms0))
                solvent_idx = list(range(natoms0, natoms0 + natoms1))
            else:
                solute, solvent = smiles1, smiles0
                solute_idx = list(range(natoms0, natoms0 + natoms1))
                solvent_idx = list(range(natoms0))

            # Debugging print statement (optional)
            print(
                f"Processing system: {system_id}, Solute: {solute}, Solvent: {solvent}"
            )

            xyz = np.fromstring(df_filtered.iloc[0]["xyz"], sep=" ").reshape(-1, 3)
            with open("solute.xyz", "w") as f:
                f.write(f"{len(solute_idx)}\n\n")
                for idx in solute_idx:
                    f.write(
                        f"{elements.split()[idx]} {xyz[idx][0]:.6f} {xyz[idx][1]:.6f} {xyz[idx][2]:.6f}\n"
                    )

            with open("solvent.xyz", "w") as f:
                f.write(f"{len(solvent_idx)}\n\n")
                for idx in solvent_idx:
                    f.write(
                        f"{elements.split()[idx]} {xyz[idx][0]:.6f} {xyz[idx][1]:.6f} {xyz[idx][2]:.6f}\n"
                    )

            try:
                solute_mol = Chem.MolFromXYZFile("solute.xyz")
                mol = Chem.Mol(solute_mol)
                rdDetermineBonds.DetermineBonds(mol, charge=0)
                for i, atom in enumerate(mol.GetAtoms()):
                    atom.SetAtomMapNum(i + 1)
                solute = Chem.MolToSmiles(mol)
            except:
                print(f"MolFromXYZFile Skipping system {system_id} due to solute error")
                continue

            try:
                solvent_mol = Chem.MolFromXYZFile("solvent.xyz")
                mol = Chem.Mol(solvent_mol)
                rdDetermineBonds.DetermineBonds(mol, charge=0)
                for i, atom in enumerate(mol.GetAtoms()):
                    atom.SetAtomMapNum(i + 1)
                solvent = Chem.MolToSmiles(mol)
            except:
                print(
                    f"MolFromXYZFile Skipping system {system_id} due to solvent error"
                )
                continue

            print(f"Solute: {solute}, Solvent: {solvent}")

            # Create topology and extract molecules
            try:
                topology = create_dimer_topology(solute, solvent)
            except Exception as e:
                print(
                    f"Skipping system {system_id} due to topology creation error: {e}"
                )
                continue
            mols = list(topology.molecules)

            # Precompute atomic numbers and masks
            solute_atomic_numbers = [atom.atomic_number for atom in mols[0].atoms]
            solvent_atomic_numbers = [atom.atomic_number for atom in mols[1].atoms]
            atomic_numbers = np.array(
                solute_atomic_numbers + solvent_atomic_numbers, dtype=int
            )

            # Create solute and solvent masks
            solute_mask = np.arange(len(atomic_numbers)) < len(solute_atomic_numbers)
            solvent_mask = ~solute_mask

            # Create interchange and extract charges
            try:
                interchange = _Interchange.from_smirnoff(
                    force_field=force_field, topology=topology
                )
            except Exception as e:
                print(
                    f"Skipping system {system_id} due to Interchange creation error: {e}"
                )
                continue
            system = interchange.to_openmm()
            nonbonded_force = next(
                f for f in system.getForces() if isinstance(f, mm.NonbondedForce)
            )
            charges = np.array(
                [
                    nonbonded_force.getParticleParameters(i)[0].value_in_unit(
                        nonbonded_force.getParticleParameters(i)[0].unit
                    )
                    for i in range(system.getNumParticles())
                ]
            )
            charges_mm = charges[solvent_mask]

            # Process each row in the filtered DataFrame
            for _, row in df_filtered.iterrows():
                # Parse XYZ coordinates
                xyz = np.fromstring(row["xyz"], sep=" ").reshape(-1, 3)
                energy = row["cbs_CCSD(T)_all"] * 4.184

                # Calculate pairwise distance
                r = np.linalg.norm(xyz[solute_idx][:, None] - xyz[solvent_idx], axis=-1)
                # if np.any(r < 1.8):
                #    print(f"Skipping system {system_id} due to close contacts: {r.min()}")
                #    print(solute_atomic_numbers)
                #    continue

                # Append computed results to respective lists
                results["e_int_list"].append(energy)
                results["xyz_mm_list"].append(xyz[solvent_idx])
                results["xyz_qm_list"].append(xyz[solute_idx])
                results["xyz_list"].append(
                    np.concatenate([xyz[solute_idx], xyz[solvent_idx]])
                )
                results["atomic_numbers_list"].append(np.asarray(solute_atomic_numbers))
                results["charges_mm_list"].append(charges_mm)
                results["solute_mask_list"].append(solute_mask)
                results["solvent_mask_list"].append(solvent_mask)
                results["topology_list"].append(topology)

                if system_id not in dataset:
                    dataset[system_id] = {"frames": [], "name": []}

                dataset[system_id]["frames"].append(frame)
                dataset[system_id]["name"].append(smiles0)
                dataset[system_id]["name"].append(smiles1)
                frame += 1

    # Save to pickle file
    with open("/home/joaomorado/repos/emle-bespoke/utils/DES370K.pkl", "wb") as f:
        pickle.dump(results, f)

    with open("dataset-370K.pkl", "wb") as f:
        pickle.dump(dataset, f)

# Convert results dictionary to individual lists if necessary
e_int_list = results["e_int_list"]
xyz_mm_list = results["xyz_mm_list"]
xyz_qm_list = results["xyz_qm_list"]
xyz_list = results["xyz_list"]
atomic_numbers_list = results["atomic_numbers_list"]
charges_mm_list = results["charges_mm_list"]
solute_mask_list = results["solute_mask_list"]
solvent_mask_list = results["solvent_mask_list"]

emle_model = "/home/joaomorado/mnsol_sampling/run_fixed/ml_mm_electrostatic_iter_2_no_patching/training/ligand_bespoke.mat"
emle_model = None

# Create the Lennard-Jones potential
lj_potential = _LennardJonesPotential(
    topology_off=results["topology_list"],
    forcefield=force_field,
    parameters_to_fit={
        "all": {"sigma": True, "epsilon": True},
    },
    device=torch.device("cuda"),
)


window_sizes = []
for k, v in dataset.items():
    window_sizes.append(len(dataset[k]["frames"]))

window_sizes = [int(int(size) / int(rg)) for size in window_sizes]
lj_potential._windows = window_sizes


print(sum(window_sizes))

from emle.train._utils import pad_to_max

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

solvent_mask = pad_to_max(solvent_mask_list).to(device=device)
solute_mask = pad_to_max(solute_mask_list).to(device=device)
xyz = pad_to_max(xyz_list).to(device=device, dtype=dtype) * 0.1
xyz_qm = pad_to_max(xyz_qm_list).to(device=device, dtype=dtype) * 0.1
xyz_mm = pad_to_max(xyz_mm_list, value=1e32).to(device=device, dtype=dtype) * 0.1
atomic_numbers = pad_to_max(atomic_numbers_list).to(device=device, dtype=torch.int64)
charges_mm = pad_to_max(charges_mm_list).to(device=device, dtype=dtype)
e_int = torch.tensor(e_int_list).to(device=device, dtype=dtype)

emle = EMLE(device=device, dtype=dtype, model=emle_model, alpha_mode="species")
# do these in batches
batch_size = 5000
n_batches = len(e_int) // batch_size
e_static_emle = []
e_ind_emle = []
e_lj_initial = []

for i in range(n_batches + 1):
    print(f"Processing batch {i + 1} out of {n_batches}")
    e_static, e_ind = emle(
        atomic_numbers[i * batch_size : (i + 1) * batch_size],
        charges_mm[i * batch_size : (i + 1) * batch_size],
        xyz_qm[i * batch_size : (i + 1) * batch_size] * 10.0,
        xyz_mm[i * batch_size : (i + 1) * batch_size] * 10.0,
    )
    e_static_emle.append(e_static.detach().cpu())
    e_ind_emle.append(e_ind.detach().cpu())
    e_lj_initial.append(
        lj_potential(
            xyz[i * batch_size : (i + 1) * batch_size],
            solute_mask[i * batch_size : (i + 1) * batch_size],
            solvent_mask[i * batch_size : (i + 1) * batch_size],
            i * batch_size,
            (i + 1) * batch_size,
        )
        .detach()
        .cpu()
    )

e_static_emle = torch.cat(e_static_emle)
e_ind_emle = torch.cat(e_ind_emle)
e_lj_initial = torch.cat(e_lj_initial)

e_int_predicted = e_static_emle + e_ind_emle
e_int_predicted = e_int_predicted * HARTREE_TO_KJ_MOL
e_int_predicted = e_int_predicted.detach().cpu().numpy()


lj_potential._e_static_emle = e_static_emle
lj_potential._e_ind_emle = e_ind_emle


e_lj_initial = e_lj_initial.detach().cpu().numpy()


e_lj_initial = e_lj_initial
diff = e_int_predicted + e_lj_initial - e_int_list
x = np.arange(0, len(e_lj_initial))
outliers = x[diff > 600000]
out_list = []

offset = 0
seen = []
for out in outliers:
    for k, v in dataset.items():
        if out > v["frames"][0] and out < v["frames"][-1]:
            if v["name"][0] + "_" + v["name"][1] in seen:
                continue
            else:
                seen.append(v["name"][0] + "_" + v["name"][1])
                offset += 1000
            print(k, v["name"][0] + "_" + v["name"][1], out)
            out_list.append([out, 1000 + offset, v["name"][0] + "_" + v["name"][1]])

# Set the Seaborn style for a clean look
sns.set(style="ticks", palette="muted", font_scale=1.2)

# Create a figure and axis
plt.figure(figsize=(12, 7))

# Plot the energy values as a line plot
# plt.plot(e_int_predicted + e_lj_initial, label='Initial', color='green', linewidth=2)
plt.plot(diff, label="Target", color="red", linewidth=2)

# Highlight the outliers with scatter points
outlier_x = [o[0] for o in out_list]
outlier_y = [o[1] for o in out_list]
plt.scatter(
    outlier_x, outlier_y, color="blue", marker="o", s=0.1, label="Outliers", zorder=5
)

# Add labels and title
plt.xlabel("Frame Index", fontsize=14)
plt.ylabel("EMLE+LJ - CCSD(T)/CBS / kJ·mol$^{-1}$", fontsize=14)
plt.title("Outliers For Which Interaction Energy Difference > 500 kJ/mol", fontsize=16)

# Add text annotations with a bounding box for visibility
for out in out_list:
    plt.text(
        out[0],
        out[1],
        f"{out[2]:s}",
        fontsize=10,
        color="black",
        ha="center",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
    )

# Add grid lines for better readability
plt.grid(True, linestyle="--", alpha=0.7)

# Add a legend to the plot
# plt.legend(loc='best', fontsize=12)

# Show the plot with adjusted layout
plt.tight_layout(pad=2.0)
plt.savefig("outliers.png")


torch.autograd.detect_anomaly(True)
from emle_bespoke.bespoke import BespokeModelTrainer as _BespokeModelTrainer

# Fit the Lennard-Jones potential
emle_bespoke = _BespokeModelTrainer(device=device, dtype=dtype)

emle_bespoke.fit_lj(
    lj_potential=lj_potential,
    xyz_qm=xyz_qm,
    xyz_mm=xyz_mm,
    xyz=xyz,
    atomic_numbers=atomic_numbers,
    charges_mm=charges_mm,
    e_int_target=e_int,
    solute_mask=solute_mask,
    solvent_mask=solvent_mask,
    lr=0.001,
    epochs=1000,
)


lj_potential.print_lj_parameters(lj_potential._lj_params)
lj_potential.write_lj_parameters("des370k_uniform")


# do these in batches
batch_size = 5000
n_batches = len(e_int) // batch_size
e_lj_final = []

for i in range(n_batches + 1):
    print(f"Processing batch {i + 1} out of {n_batches}")
    e_lj_final.append(
        lj_potential(
            xyz[i * batch_size : (i + 1) * batch_size],
            solute_mask[i * batch_size : (i + 1) * batch_size],
            solvent_mask[i * batch_size : (i + 1) * batch_size],
            i * batch_size,
            (i + 1) * batch_size,
        )
        .detach()
        .cpu()
    )


e_lj_final = torch.cat(e_lj_final)
e_lj_final = e_lj_final.detach().cpu().numpy()


# Identify outliers
diff = e_int_predicted + e_lj_final - e_int_list
x = np.arange(0, len(e_lj_final))
outliers = x[diff > 20000]
out_list = []

offset = 0
seen = []
for out in outliers:
    for k, v in dataset.items():
        if out > v["frames"][0] and out < v["frames"][-1]:
            if v["name"][0] + "_" + v["name"][1] in seen:
                continue
            else:
                seen.append(v["name"][0] + "_" + v["name"][1])
                offset += 200
            print(k, v["name"][0] + "_" + v["name"][1], out)
            out_list.append([out, 400 + offset, v["name"][0] + "_" + v["name"][1]])

# Set the Seaborn style for a clean look
sns.set(style="ticks", palette="muted", font_scale=1.2)

# Create a figure and axis
plt.figure(figsize=(12, 7))

# Plot the energy values as a line plot
# plt.plot(e_int_predicted + e_lj_initial, label='Initial', color='green', linewidth=2)
plt.plot(diff, label="Target", color="red", linewidth=2)

# Highlight the outliers with scatter points
outlier_x = [o[0] for o in out_list]
outlier_y = [o[1] for o in out_list]
plt.scatter(
    outlier_x, outlier_y, color="blue", marker="o", s=0.1, label="Outliers", zorder=5
)

# Add labels and title
plt.xlabel("Frame Index", fontsize=14)
plt.ylabel("EMLE+LJ − CCSD(T)/CBS / kJ·mol$^{-1}$", fontsize=14)
plt.title(
    "Outliers For Which Interaction Energy Difference > 20000 kJ/mol", fontsize=16
)

# Add text annotations with a bounding box for visibility
for out in out_list:
    plt.text(
        out[0],
        out[1],
        f"{out[2]:s}",
        fontsize=10,
        color="black",
        ha="center",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
    )

# Add grid lines for better readability
plt.grid(True, linestyle="--", alpha=0.7)

# Add a legend to the plot
# plt.legend(loc='best', fontsize=12)

# Show the plot with adjusted layout
plt.tight_layout(pad=2.0)
plt.savefig("outliers.png")


# In[ ]:


sigma_init = lj_potential._sigma_init.detach().cpu().numpy().squeeze()
sigma_final = lj_potential._sigma.detach().cpu().numpy().squeeze()
epsilon_init = lj_potential._epsilon_init.detach().cpu().numpy().squeeze()
epsilon_final = lj_potential._epsilon.detach().cpu().numpy().squeeze()

# Example data (replace with your data)
atom_indices = np.arange(len(sigma_init))
delta_sigma = sigma_final - sigma_init  # Change in sigma
delta_epsilon = epsilon_final - epsilon_init  # Change in epsilon

# Create a DataFrame for Seaborn
data = pd.DataFrame(
    {
        "Atom Index": np.tile(atom_indices, 2),
        "Δ": np.concatenate([delta_sigma, delta_epsilon]),
        "Parameter": ["Δσ"] * len(atom_indices) + ["Δε"] * len(atom_indices),
    }
)

plt.figure(figsize=(12, 8))
sns.barplot(
    data=data,
    x="Atom Index",
    y="Δ",
    hue="Parameter",
    palette="viridis",
    edgecolor="black",
)
plt.xlabel("Atom Type", fontsize=14)
plt.ylabel("Δ (kJ/mol or nm)", fontsize=14)

atom_type_to_index = lj_potential._atom_type_to_index
index_to_atom_type = {v: k for k, v in atom_type_to_index.items()}
atom_types = [index_to_atom_type[i + 1] for i in range(len(index_to_atom_type))]
plt.xticks(
    ticks=np.arange(len(atom_types)) + 1, labels=atom_types, fontsize=12, rotation=45
)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="Parameter", fontsize=12)
plt.ylim(-0.10, 0.10)
plt.show()


# In[ ]:


df = pd.read_csv("/home/joaomorado/repos/emle-bespoke/utils/DES370K.csv")
subset = dataset
num_systems = len(subset)
cols = 3  # Number of columns
rows = int(np.ceil(num_systems / cols))

# Set the Seaborn style for a clean look
sns.set(style="whitegrid", palette="muted")

# Create a figure and axes for the 3xN grid
fig, axes = plt.subplots(
    rows, cols, figsize=(5 * cols, 4 * rows), sharex=False, sharey=False
)

# Flatten axes array for easier iteration
axes = axes.flatten()
# Loop through systems in the dataset
for i, (system_id, values) in enumerate(subset.items()):
    # Get the corresponding axis
    ax = axes[i]

    # Define x limits based on frame IDs
    x_min = values["frames"][0]
    x_max = values["frames"][-1]

    # Define y limits based on e_int_list values for these frames
    y_min = min(e_int_predicted[x_min:x_max] + e_lj_initial[x_min:x_max]) - 15
    y_max = max(e_int_predicted[x_min:x_max] + e_lj_initial[x_min:x_max]) + 5

    # Plot the energy values as a line plot
    ax.plot(
        e_int_predicted[x_min:x_max] + e_lj_final[x_min:x_max],
        label="After fit",
        color="green",
        linewidth=2,
    )
    ax.plot(
        e_int_predicted[x_min:x_max] + e_lj_initial[x_min:x_max],
        label="Before fit",
        color="red",
        linewidth=2,
    )
    ax.plot(e_int_list[x_min:x_max], label="CCSD(T)/CBS", color="black", linewidth=2)

    # Add labels and title
    ax.set_xlabel("Index", fontsize=10)
    ax.set_ylabel("Energy", fontsize=10)
    try:
        title = df[df["system_id"] == system_id]["system_name"].values[0]
    except:
        title = ""

    ax.set_title(f"{title}-{system_id}", fontsize=12)

    # Add grid lines for better readability
    ax.grid(True, linestyle="--", alpha=0.7)
    # ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Add a legend to the plot
    ax.legend(fontsize=8, loc="upper left")

# Hide unused subplots if any
for j in range(num_systems, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout for better spacing
plt.tight_layout()

# Show the grid of plots
plt.show()


# In[ ]:


e_lj_final = e_lj_final.cpu().numpy()
e_lj_initial = e_lj_initial.cpu().numpy()
e_int = e_int.cpu().numpy()


x = 500

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid", context="talk")

# Create the plot
plt.figure(figsize=(8, 7))
plt.plot([-1000, 1000], [-1000, 1000], color="black", linestyle="--", zorder=-10)
e_final = e_int_predicted + e_lj_final
e_initial = e_int_predicted + e_lj_initial
mask = e_int < 50
plt.scatter(
    e_int[mask], e_final[mask], color="blue", alpha=0.2, label="Before Fitting", s=5
)
plt.scatter(
    e_int[mask], e_initial[mask], color="red", alpha=0.2, label="After Fitting", s=5
)


# Define RMSE, MSE, and R^2 functions
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mean_signed_error(y_true, y_pred):
    return np.mean(y_pred - y_true)


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


# Calculate metrics for initial and final LJ energies
rmse_initial = rmse(e_int, e_int_predicted + e_lj_initial)
rmse_final = rmse(e_int, e_int_predicted + e_lj_final)
mse_initial = mean_signed_error(e_int, e_int_predicted + e_lj_initial)
mse_final = mean_signed_error(e_int, e_int_predicted + e_lj_final)
r2_initial = r_squared(e_int, e_int_predicted + e_lj_initial)
r2_final = r_squared(e_int, e_int_predicted + e_lj_final)

print(f"RMSE Initial LJ: {rmse_initial:.2f} kJ/mol")
print(f"RMSE Final LJ: {rmse_final:.2f} kJ/mol")
print(f"MSE Initial LJ: {mse_initial:.2f} kJ/mol")
print(f"MSE Final LJ: {mse_final:.2f} kJ/mol")
print(f"R^2 Initial LJ: {r2_initial:.2f}")
print(f"R^2 Final LJ: {r2_final:.2f}")

# Add identity line for reference

# Add RMSE, MSE, and R^2 values to the plot
plt.text(
    -x + 5,
    x - 10,
    f"Before Fitting:\nRMSE: {rmse_initial:.2f}\nMSE: {mse_initial:.2f}\nR²: {r2_initial:.2f}",
    fontsize=12,
    color="blue",
    ha="left",
    va="top",
)
plt.text(
    -x + 5,
    x - 250,
    f"After Fitting:\nRMSE: {rmse_final:.2f}\nMSE: {mse_final:.2f}\nR²: {r2_final:.2f}",
    fontsize=12,
    color="red",
    ha="left",
    va="top",
)

# Labels and title
plt.xlabel("Target Energy (kJ/mol)", fontsize=14)
plt.ylabel("Predicted Energy (kJ/mol)", fontsize=14)

# Legend and grid
plt.legend(fontsize=12, loc="best")
plt.grid(alpha=0.3)

# Adjust ticks for readability
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(-x, x)
plt.ylim(-x, x)

# Show plot
plt.tight_layout()
plt.savefig("scatter.png")
