"""Bespoke model trainer containg various thin wrappers."""
import numpy as _np
import torch as _torch
from emle.models import EMLEBase as _EMLEBase
from emle.train._utils import pad_to_max
from loguru import logger as _logger

from .reference_data import ReferenceData as _ReferenceData
from .utils import write_dict_to_file as _write_dict_to_file


class BespokeModelTrainer:
    """
    Class to train bespoke EMLE models.

    Parameters
    ----------
    reference_sampler : ReferenceDataSampler
        The reference data sampler.
    refence_data : ReferenceData
        The reference data instance.
    emle_base : EMLEBase, optional
        The EMLE base model to use. Default is EMLEBase.
    filename_prefix : str, optional
        The prefix to use for filenames. Default is "bespoke".
    """

    def __init__(
        self,
        reference_sampler=None,
        reference_data=None,
        emle_base: _EMLEBase = _EMLEBase,
        filename_prefix: str = "bespoke",
        alpha_static: float = 1.0,
        beta_induced: float = 1.0,
        dtype=_torch.float64,
        device=_torch.device("cuda")
        if _torch.cuda.is_available()
        else _torch.device("cpu"),
    ):
        self.reference_sampler = reference_sampler
        self.reference_data = (
            reference_data or _ReferenceData()
            if not reference_sampler
            else reference_sampler.reference_data
        )

        # Ensure the reference data instance is consistent
        if (
            self.reference_sampler
            and self.reference_data is not self.reference_sampler.reference_data
        ):
            _logger.warning(
                "Reference data instances are not the same in ReferenceDataSampler and ReferenceData."
            )

        self._emle_base = emle_base
        self._filename_prefix = filename_prefix

        # Set the alpha and beta values
        self._alpha_static = alpha_static
        self._beta_induced = beta_induced

        # Set the device and dtype
        self._device = device
        self._dtype = dtype

    def sample_and_train_model(
        self,
        n_samples=100,
        n_steps=1000,
        calc_static=True,
        calc_induction=True,
        calc_horton=True,
        calc_polarizability=True,
        filename_prefix=None,
        train_mask=None,
        **train_model_kwargs,
    ) -> None:
        """
        Sample reference data and train a bespoke model.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        n_steps : int
            The number of steps between each sample.
        calc_static : bool, optional
            Whether to calculate static electrostatic energy. Default is True.
        calc_induction : bool, optional
            Whether to calculate induction energy. Default is True.
        calc_horton : bool, optional
            Whether to calculate Horton properties. Default is True.
        calc_polarizability : bool, optional
            Whether to get polarizability. Default is True.
        filename_prefix : str, optional
            The prefix to use for filenames. Default is "bespoke".
        train_mask : torch.Tensor, optional
            The mask to use for training. Default is None.
        train_model_kwargs : dict
            Additional keyword arguments (apart from reference data) to pass be to the train_model method.
            See https://github.com/kzinovjev/emle-engine/blob/emle-train/emle/train/_trainer.py#L242-L340
        """
        # Sample reference data
        self.sample_reference_data(
            n_samples=n_samples,
            n_steps=n_steps,
            calc_static=calc_static,
            calc_induction=calc_induction,
            calc_horton=calc_horton,
            calc_polarizability=calc_polarizability,
            ref_data_filename=f"{filename_prefix}_ref_data.pkl"
            if filename_prefix
            else f"{self._filename_prefix}_ref_data.pkl",
        )

        # Train the bespoke model
        self.train_model(
            z=self.reference_data["z"],
            xyz=self.reference_data["xyz_qm"],
            s=self.reference_data["s"],
            q_core=self.reference_data["q_core"],
            q_val=self.reference_data["q_val"],
            alpha=self.reference_data["alpha"],
            train_mask=train_mask,
            model_filename=f"{filename_prefix}_model.mat"
            if filename_prefix
            else f"{self._filename_prefix}_model.mat",
            plot_data_filename=f"{filename_prefix}_plot_data.mat"
            if filename_prefix
            else f"{self._filename_prefix}_plot_data.mat",
            **train_model_kwargs,
        )

        # Patch the model
        self.patch_model(
            e_static_target=self.reference_data["e_static"],
            e_ind_target=self.reference_data["e_ind"],
            atomic_numbers=self.reference_data["z"],
            charges_mm=self.reference_data["charges_mm"],
            xyz_qm=self.reference_data["xyz_qm"],
            xyz_mm=self.reference_data["xyz_mm"],
        )

    def sample_reference_data(
        self,
        n_samples=100,
        n_steps=1000,
        calc_static: bool = True,
        calc_induction: bool = True,
        calc_horton: bool = True,
        calc_polarizability: bool = True,
        ref_data_filename=None,
    ) -> _ReferenceData:
        """
        Sample reference data.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        n_steps : int
            The number of steps between each sample.
        calc_static : bool, optional
            Whether to calculate static electrostatic energy. Default is True.
        calc_induction : bool, optional
            Whether to calculate induction energy. Default is True.
        calc_horton : bool, optional
            Whether to calculate Horton properties. Default is True.
        calc_polarizability : bool, optional
            Whether to get polarizability. Default is True.
        ref_data_filename : str, optional
            The filename to save the reference data to. Default is None.

        Returns
        -------
        dict
            The reference data.
        """
        assert self.reference_sampler is not None, "Reference sampler is not set."
        msg = r"""
╔════════════════════════════════════════════════════════════╗
║    Starting sampling of reference data model training...   ║
╚════════════════════════════════════════════════════════════╝
"""
        for line in msg.split("\n"):
            _logger.info(line)
        _logger.info(f"Number of samples to sample: {n_samples}")
        _logger.info(f"Number of steps per sample: {n_steps}")
        for i in range(n_samples):
            _logger.info(f"Sampling {i + 1}/{n_samples} configurations.")
            self.reference_sampler.sample(
                n_steps=n_steps,
                calc_static=calc_static,
                calc_induction=calc_induction,
                calc_horton=calc_horton,
                calc_polarizability=calc_polarizability,
            )

        _logger.info("Finished sampling reference data.")

        # Write the reference data to a file
        ref_data_filename = ref_data_filename or f"{self._filename_prefix}_ref_data.pkl"
        self.reference_sampler.reference_data.write(filename=ref_data_filename)

        return self.reference_sampler.reference_data

    def train_model(
        self,
        z,
        xyz,
        s,
        q_core,
        q_val,
        alpha,
        train_mask,
        model_filename,
        plot_data_filename,
        *args,
        **kwargs,
    ):
        """
        Train a bespoke EMLE model.

        Parameters
        ----------
        z : torch.Tensor or np.ndarray or list of tensors/arrays
            The atomic numbers for every sample.
        xyz : torch.Tensor or np.ndarray or list of tensors/arrays
            The Cartesian coordinates for every sample in Angstrom.
        s : torch.Tensor or np.ndarray or list of tensors/arrays
            The atomic widths for every sample in Angstrom.
        q_core : torch.Tensor or np.ndarray or list of tensors/arrays
            The core charges for every sample.
        q_val : torch.Tensor or np.ndarray or list of tensors/arrays
            The valence charges for every sample.
        alpha : torch.Tensor or np.ndarray or list of tensors/arrays
            The polarizability for every sample.
        train_mask : torch.Tensor or np.ndarray or list of tensors/arrays
            The mask to use for training.
        model_filename : str
            The filename to save the model.
        plot_data_filename : str, optional
            The filename to save the plot data.
        args : list
            Additional positional arguments to pass to the trainer.
        kwargs : dict
            Additional keyword arguments to pass to the trainer.

        Returns
        -------
        EMLETrainer
            The trainer instance.
        """
        from emle.train import EMLETrainer as _EMLETrainer

        msg = r"""
╔════════════════════════════════════════════════════════════╗
║         Starting training of bespoke EMLE model...         ║
╚════════════════════════════════════════════════════════════╝
"""
        for line in msg.split("\n"):
            _logger.info(line)

        trainer = _EMLETrainer(self._emle_base)

        trainer.train(
            z=z,
            xyz=xyz,
            s=s,
            q_core=q_core,
            q_val=q_val,
            alpha=alpha,
            train_mask=train_mask,
            model_filename=model_filename,
            plot_data_filename=plot_data_filename,
            *args,
            **kwargs,
        )

        return trainer

    def patch_model(
        self,
        e_static_target,
        e_ind_target,
        atomic_numbers,
        charges_mm,
        xyz_qm,
        xyz_mm,
        model=None,
        lr=0.0001,
        epochs=100,
        print_every=10,
        alpha_static=None,
        beta_induced=None,
    ):
        """
        Patch the model by finding optimal alpha and beta values.

        Parameters
        ----------
        e_static_target: list of torch.Tensor (NBATCH,)
            Target static energy component in kJ/mol.
        e_ind_target: list of torch.Tensor (NBATCH,)
            Target induced energy component in kJ/mol.
        atomic_numbers: torch.Tensor (NBATCH, N_QM_ATOMS)
            Atomic numbers of QM atoms.
        charges_mm: torch.Tensor (NBATCH, max_mm_atoms)
            MM point charges in atomic units.
        xyz_qm: torch.Tensor (NBATCH, N_QM_ATOMS, 3)
            Positions of QM atoms in Angstrom.
        xyz_mm: torch.Tensor (NBATCH, N_MM_ATOMS, 3)
            Positions of MM atoms in Angstrom.
        model: str
            Filepath to the EMLE model to patch.
        lr : float, optional
            The learning rate. Default is 0.01.
        epochs : int, optional
            The number of epochs. Default is 100.
        print_every : int, optional
            The number of steps between each print. Default is 10.
        alpha_static : float, optional
            The initial guess of alpha. Default is 1.0.
        beta_induced : float, optional
            The initial guess of beta. Default is 1.0.

        Returns
        -------
        EMLEPatched
            The patched EMLE model.
        float
            The optimal alpha_static value.
        float
            The optimal beta_induced value.
        """
        import torch as _torch

        from ._train import train_model
        from .patching import EMLEPatched, PatchingLoss

        msg = r"""
╔════════════════════════════════════════════════════════════╗
║                 Patching the EMLE model...                 ║
╚════════════════════════════════════════════════════════════╝
"""
        for line in msg.split("\n"):
            _logger.info(line)

        self._alpha_static = (
            alpha_static if alpha_static is not None else self._alpha_static
        )
        self._beta_induced = (
            beta_induced if beta_induced is not None else self._beta_induced
        )

        _logger.info(f"Initial alpha_static: {self._alpha_static}")
        _logger.info(f"Initial beta_induced: {self._beta_induced}")

        # Create the patched model
        patched_model = EMLEPatched(
            model="ligand_bespoke.mat",
            alpha_static=self._alpha_static,
            beta_induced=self._beta_induced,
            device=self._device,
            dtype=self._dtype,
        )

        # Convert numpy arrays to tensors
        atomic_numbers = pad_to_max(atomic_numbers).to(
            device=self._device, dtype=_torch.int64
        )
        charges_mm = pad_to_max(charges_mm).to(device=self._device, dtype=self._dtype)
        xyz_qm = pad_to_max(xyz_qm).to(device=self._device, dtype=self._dtype)
        xyz_mm = pad_to_max(xyz_mm).to(device=self._device, dtype=self._dtype)
        e_static_target = _torch.tensor(e_static_target).to(
            device=self._device, dtype=self._dtype
        )
        e_ind_target = _torch.tensor(e_ind_target).to(
            device=self._device, dtype=self._dtype
        )

        # Patch the model
        train_model(
            loss_class=PatchingLoss,
            # opt_param_names=["alpha_static", "beta_induced"],
            opt_param_names=["a_QEq", "ref_values_chi"],
            lr=lr,
            epochs=epochs,
            print_every=print_every,
            emle_model=patched_model,
            e_static_target=e_static_target,
            e_ind_target=e_ind_target,
            atomic_numbers=atomic_numbers,
            charges_mm=charges_mm,
            xyz_qm=xyz_qm,
            xyz_mm=xyz_mm,
            fit_e_static=True,
            fit_e_ind=False,
        )

        # Patch the model
        train_model(
            loss_class=PatchingLoss,
            opt_param_names=["a_Thole", "k_Z"],
            lr=lr,
            epochs=epochs,
            print_every=print_every,
            emle_model=patched_model,
            e_static_target=e_static_target,
            e_ind_target=e_ind_target,
            atomic_numbers=atomic_numbers,
            charges_mm=charges_mm,
            xyz_qm=xyz_qm,
            xyz_mm=xyz_mm,
            fit_e_static=False,
            fit_e_ind=True,
        )

        self._alpha_static = patched_model.alpha_static.item()
        self._beta_induced = patched_model.beta_induced.item()

        _logger.info(f"Optimal alpha_static: {self._alpha_static:.4f}")
        _logger.info(f"Optimal beta_induced: {self._beta_induced:.4f}")
        _logger.info("Finished patching the model.")

        return patched_model, alpha_static, beta_induced

    def sample_dimer_curves(
        self, ref_data_filename=None, *args, **kwargs
    ) -> _ReferenceData:
        """
        Sample reference data.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        n_steps : int
            The number of steps between each sample.
        ref_data_filename : str
            The filename to save the reference data.

        Returns
        -------
        dict
            The reference data.
        """
        assert self.reference_sampler is not None, "Reference sampler is not set."
        msg = r"""
╔════════════════════════════════════════════════════════════╗
║             Starting sampling of dimer curves...           ║
╚════════════════════════════════════════════════════════════╝
"""
        for line in msg.split("\n"):
            _logger.info(line)

        self.reference_sampler.sample()

        _logger.info("Finished sampling of dimer curves.")

        # Write the reference data to a file
        ref_data_filename = ref_data_filename or f"{self._filename_prefix}_ref_data.pkl"
        self.reference_sampler.reference_data.write(filename=ref_data_filename)

        return self.reference_sampler.reference_data

    def fit_lj(
        self,
        e_int_target,
        atomic_numbers,
        charges_mm,
        xyz_qm,
        xyz_mm,
        solute_mask,
        solvent_mask,
        lj_potential,
        model=None,
        lr=5e-4,
        epochs=500,
        print_every=10,
    ):
        from ._train import train_model
        from .lj_fitting import InteractionEnergyLoss as _InteractionEnergyLoss
        from .patching import EMLEPatched

        msg = r"""
╔════════════════════════════════════════════════════════════╗
║            Starting fitting of LJ parameters...            ║
╚════════════════════════════════════════════════════════════╝
"""
        for line in msg.split("\n"):
            _logger.info(line)

        # Create an EMLE patched model to allow for LJ fitting
        # with custom alpha and beta values
        patched_model = EMLEPatched(
            model=model,
            alpha_static=self._alpha_static,
            beta_induced=self._beta_induced,
            device=self._device,
            dtype=self._dtype,
        )

        # Convert reference data to numpy arrays
        atomic_numbers = _torch.tensor(atomic_numbers)
        charges_mm = _torch.tensor(charges_mm)
        xyz_qm = _torch.tensor(xyz_qm)
        xyz_mm = _torch.tensor(xyz_mm)
        solute_mask = _torch.tensor(solute_mask)
        solvent_mask = _torch.tensor(solvent_mask)
        e_int_target = _torch.tensor(e_int_target)

        # Convert reference data to tensors
        atomic_numbers = pad_to_max(atomic_numbers).to(
            device=self._device, dtype=_torch.int64
        )
        charges_mm = pad_to_max(charges_mm).to(device=self._device, dtype=self._dtype)
        xyz_qm = pad_to_max(xyz_qm).to(device=self._device, dtype=self._dtype)
        xyz_mm = pad_to_max(xyz_mm).to(device=self._device, dtype=self._dtype)
        solute_mask = pad_to_max(solute_mask).to(device=self._device, dtype=_torch.bool)
        solvent_mask = pad_to_max(solvent_mask).to(
            device=self._device, dtype=_torch.bool
        )
        e_int_target = pad_to_max(e_int_target).to(
            device=self._device, dtype=self._dtype
        )

        e_int_loss = _InteractionEnergyLoss(
            emle_model=patched_model,
            lj_potential=lj_potential,
        )

        # Store the data for plotting
        plot_data = {
            "e_int_target": e_int_target.cpu().numpy(),
        }

        e_int_predicted = []
        for i in range(len(atomic_numbers)):
            e_static, e_ind, e_lj = e_int_loss.calculate_predicted_interaction_energy(
                atomic_numbers=atomic_numbers[i],
                charges_mm=charges_mm[i],
                xyz_qm=xyz_qm[i],
                xyz_mm=xyz_mm[i],
                solute_mask=solute_mask[i],
                solvent_mask=solvent_mask[i],
            )
            e_int = e_static + e_ind + e_lj
            e_int_predicted.append(e_int.item())
        plot_data["e_int_predicted"] = _np.array(e_int_predicted)

        # Fit the LJ parameters
        loss_class_kwargs = {"lj_potential": lj_potential}
        loss_model = train_model(
            loss_class=_InteractionEnergyLoss,
            opt_param_names=["sigma", "epsilon"],
            lr=lr,
            epochs=epochs,
            print_every=print_every,
            emle_model=patched_model,
            loss_class_kwargs=loss_class_kwargs,
            e_int_target=e_int_target,
            atomic_numbers=atomic_numbers,
            charges_mm=charges_mm,
            xyz_qm=xyz_qm,
            xyz_mm=xyz_mm,
            solute_mask=solute_mask,
            solvent_mask=solvent_mask,
        )

        # Store the fitted data for plotting
        e_int_fitted = []
        e_static_fitted = []
        e_ind_fitted = []
        e_lj_fitted = []
        for i in range(len(atomic_numbers)):
            e_static, e_ind, e_lj = e_int_loss.calculate_predicted_interaction_energy(
                atomic_numbers=atomic_numbers[i],
                charges_mm=charges_mm[i],
                xyz_qm=xyz_qm[i],
                xyz_mm=xyz_mm[i],
                solute_mask=solute_mask[i],
                solvent_mask=solvent_mask[i],
            )
            e_int = e_static + e_ind + e_lj
            e_static_fitted.append(e_static.item())
            e_ind_fitted.append(e_ind.item())
            e_lj_fitted.append(e_lj.item())
            e_int_fitted.append(e_int.item())

        plot_data["e_int_fitted"] = _np.array(e_int_fitted)
        plot_data["e_static_fitted"] = _np.array(e_static_fitted)
        plot_data["e_ind_fitted"] = _np.array(e_ind_fitted)
        plot_data["e_lj_fitted"] = _np.array(e_lj_fitted)

        # Write the plot data to a file
        plot_data_filename = f"{self._filename_prefix}_plot_data.mat"
        _write_dict_to_file(plot_data, plot_data_filename)

        _logger.info("Fitted LJ parameters:")
        # Get all Parameters
        for name, param in loss_model.named_parameters():
            if "lj" in name:
                _logger.info(
                    f"Optimal {name.split('.', 1)[-1][1:]}: {param.item():.8f}"
                )

        lj_potential.print_lj_parameters()
        _logger.info("Finished fitting LJ parameters.")

        return

    def get_mbis_static_predictions(
        self,
        charges_mm,
        xyz_qm,
        xyz_mm,
        q_core,
        q_val,
        s,
    ):
        from emle.models import EMLE as _EMLE

        from ._constants import ANGSTROM_TO_BOHR as _ANGSTROM_TO_BOHR
        from ._constants import HARTREE_TO_KJ_MOL as _HARTREE_TO_KJ_MOL

        # Create the EMLE base model
        emle_base = _EMLE(
            device=self._device,
            dtype=self._dtype,
        )._emle_base

        # Convert reference MBIS data to tensors
        xyz_qm = pad_to_max(xyz_qm).to(device=self._device, dtype=self._dtype)
        xyz_mm = pad_to_max(xyz_mm).to(device=self._device, dtype=self._dtype)
        charges_mm = pad_to_max(charges_mm).to(device=self._device, dtype=self._dtype)
        q_core = pad_to_max(q_core).to(device=self._device, dtype=self._dtype)
        q_val = pad_to_max(q_val).to(device=self._device, dtype=self._dtype)
        s = pad_to_max(s).to(device=self._device, dtype=self._dtype)

        xyz_qm_bohr = xyz_qm * _ANGSTROM_TO_BOHR
        xyz_mm_bohr = xyz_mm * _ANGSTROM_TO_BOHR

        # Get the mesh data
        mesh_data = emle_base._get_mesh_data(xyz_qm_bohr, xyz_mm_bohr, s)

        # Calculate the static and induced energy components
        e_static = (
            emle_base.get_static_energy(q_core, q_val, charges_mm, mesh_data)
            * _HARTREE_TO_KJ_MOL
        )

        # Store the data for plotting
        plot_data = {
            "e_static_mbis": e_static.cpu().numpy(),
        }

        # Write the plot data to a file
        plot_data_filename = f"{self._filename_prefix}_mbis_plot_data.mat"
        _write_dict_to_file(plot_data, plot_data_filename)

        return e_static
