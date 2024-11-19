from typing import Any

import openmm.unit as _unit
import torch as _torch
from emle.models import EMLE as _EMLE

from .._constants import ATOMIC_NUMBERS_TO_SYMBOLS as _ATOMIC_NUMBERS_TO_SYMBOLS
from ..calculators import ORCACalculator as _ORCACalculator
from ..reference_data import ReferenceData as _ReferenceData
from ._dimers_gen import DimerGenerator
from ._lj_potential import LennardJonesPotential as _LennardJonesPotential
from ._loss import InteractionEnergyLoss as _InteractionEnergyLoss


class LJFitting:
    """
    Class to train bespoke EMLE models.

    Parameters
    ----------
    emle_model : EMLE
        The EMLE model.
    refence_data : ReferenceData
        The reference data instance.
    emle_base : EMLEBase, optional
        The EMLE base model to use. Default is EMLEBase.
    filename_prefix : str, optional
        The prefix to use for filenames. Default is "lj_fitting".
    """

    def __init__(
        self,
        emle_model: _EMLE,
        qm_calculator: Any,
        reference_data=None,
        filename_prefix: str = "lj_fitting",
    ):
        self.reference_data = reference_data or _ReferenceData()
        self._qm_calculator = qm_calculator
        self._emle_model = emle_model
        self._filename_prefix = filename_prefix

    def fit_lj(self):
        """
        Fit Lennard-Jones parameters.
        """
        pass

    def get_reference_interaction_energy_curve(
        self, dimer_configurations, solute_indices, solvent_indices, atomic_numbers
    ):
        symbols_dimer = [_ATOMIC_NUMBERS_TO_SYMBOLS[an.item()] for an in atomic_numbers]
        symbols_qm = [symbols_dimer[i] for i in solute_indices]
        symbols_mm = [symbols_dimer[i] for i in solvent_indices]

        solute_energy = self._qm_calculator.get_potential_energy(
            elements=symbols_qm,
            positions=dimer_configurations[0][solute_indices] * 10.0,
            directory="solute_vacuum",
            orca_simple_input="! b3lyp cc-pvtz TightSCF D3BJ",
            orca_blocks="%MaxCore 1024\n%pal\nnprocs 8\nend\n",
        )

        solvent_energy = self._qm_calculator.get_potential_energy(
            elements=symbols_mm,
            positions=dimer_configurations[0][solvent_indices] * 10.0,
            directory="solvent_vacuum",
            orca_simple_input="! b3lyp cc-pvtz TightSCF D3BJ",
            orca_blocks="%MaxCore 1024\n%pal\nnprocs 8\nend\n",
        )

        for config in dimer_configurations:
            vacuum_energy = self._qm_calculator.get_potential_energy(
                elements=symbols_dimer,
                positions=config * 10.0,
                directory="dimer_vacuum",
                orca_simple_input="! b3lyp cc-pvtz TightSCF D3BJ",
                orca_blocks="%MaxCore 1024\n%pal\nnprocs 8\nend\n",
            )

            interaction_energy = vacuum_energy - solute_energy - solvent_energy

            self.reference_data.add_data_to_key("e_int", interaction_energy)
            self.reference_data.add_data_to_key("e_dimer", vacuum_energy)
            self.reference_data.add_data_to_key("e_solute", solute_energy)
            self.reference_data.add_data_to_key("e_solvent", solvent_energy)
            self.reference_data.add_data_to_key("z", atomic_numbers[solute_indices])
            self.reference_data.add_data_to_key("solvent_indices", solvent_indices)
            self.reference_data.add_data_to_key("solute_indices", solute_indices)
            self.reference_data.add_data_to_key("xyz", config)

        return self.reference_data

    def get_reference_data(self):
        return self.reference_data


n_lowest = 50
n_samples = 2500
temperature = 1000.0
sphere_radius = 0.5 * _unit.nanometer
forcefields = ["openff_unconstrained-2.0.0.offxml"]

dimer_gen = DimerGenerator()
topology_off = dimer_gen.create_dimer_topology(
    solute_smiles="c1ccccc1", solvent_smiles="[H:2][O:1][H:3]"
)

energies_dimers, configs_dimers = dimer_gen.generate_dimers(
    topology_off=topology_off,
    n_samples=n_samples,
    n_lowest=n_lowest,
    temperature=temperature,
    sphere_radius=sphere_radius,
    forcefields=forcefields,
)


device = _torch.device("cuda") if _torch.cuda.is_available() else _torch.device("cpu")
dtype = _torch.float64

emle_model = _EMLE(
    device=device,
    dtype=dtype,
    model="/home/joaomorado/repos/emle-bespoke/src/emle_bespoke/models/emle_qm7_new_ivm0.1.mat",
)
lj_fitting = LJFitting(
    emle_model=emle_model, qm_calculator=_ORCACalculator(), reference_data=None
)

#
solute_indices = list(range(12))
solvent_indices = list(range(12, 15))
atomic_numbers = _torch.tensor([6] * 6 + [1] * 6)

# Calculate the reference interaction energy curve
curves = []
for config in configs_dimers:
    curves.append(
        dimer_gen.generate_dimer_curve(config, list(range(12)), list(range(12, 15)))
    )


# Fit the Lennard-Jones parameters
lj_potential = _LennardJonesPotential(
    topology_off=topology_off,
    forcefield=forcefields,
    parameters_to_fit={"n-tip3p-O": ["sigma", "epsilon"]},
)

lj_fitting_loss = _InteractionEnergyLoss(
    emle_model=emle_model, lj_potential=lj_potential
)

curves = _torch.tensor(curves, dtype=dtype, device=device)
atomic_numbers = _torch.tensor(atomic_numbers, dtype=_torch.int64, device=device)

for pos in curves[0]:
    e_static, e_ind, e_lj = lj_fitting_loss.calculate_predicted_interaction_energy(
        atomic_numbers=atomic_numbers,
        charges_mm=_torch.tensor([-0.834, 0.417, 0.417], dtype=dtype, device=device),
        pos=pos * 10,
        solvent_indices=solvent_indices,
        solute_indices=solute_indices,
    )
    final_energy = e_static + e_ind + e_lj


for curve in curves:
    lj_fitting.get_reference_interaction_energy_curve(
        dimer_configurations=curve,
        solute_indices=solute_indices,
        solvent_indices=solvent_indices,
        atomic_numbers=_torch.tensor([6] * 6 + [1] * 6 + [8, 1, 1], dtype=_torch.int64),
    )


ref_data = lj_fitting.get_reference_data()
ref_data.write("lj_ref_data.pkl")
