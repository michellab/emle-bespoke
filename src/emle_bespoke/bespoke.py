import logging as _logging

from emle.models import EMLEBase as _EMLEBase

from .reference_data import ReferenceData as _ReferenceData
from .sampler import ReferenceDataSampler as _ReferenceDataSampler

logger = _logging.getLogger(__name__)


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
            logger.warning(
                "Reference data instances are not the same in ReferenceDataSampler and ReferenceData."
            )

        self._emle_base = emle_base
        self._filename_prefix = filename_prefix

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
    ):
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
            xyz=self.reference_data["xyz"],
            s=self.reference_data["s"],
            q_core=self.reference_data["q_core"],
            q_val=self.reference_data["q_val"],
            alpha=self.reference_data["alpha"],
            train_mask=train_mask,
            model_filename=f"{filename_prefix}_model.mat"
            if filename_prefix
            else f"{self._filename_prefix}_model.mat",
            **train_model_kwargs,
        )

        # Patch the model
        self.patch_model()

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
            logger.info(line)
        logger.info(f"Number of samples to sample: {n_samples}")
        logger.info(f"Number of steps per sample: {n_steps}")
        for i in range(n_samples):
            logger.info(f"Sampling {i + 1}/{n_samples} configurations.")
            self.reference_sampler.sample(
                n_steps=n_steps,
                calc_static=calc_static,
                calc_induction=calc_induction,
                calc_horton=calc_horton,
                calc_polarizability=calc_polarizability,
            )

        logger.info("Finished sampling reference data.")

        # Write the reference data to a file
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
║         Starting training of bespoke EMLE model....        ║
╚════════════════════════════════════════════════════════════╝
"""
        for line in msg.split("\n"):
            logger.info(line)

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
            *args,
            **kwargs,
        )

        return trainer

    def patch_model(self):
        pass

    '''
    def patch_model(self, lr=0.01, epochs=100, print_every=10):
        """
        Patch the model by finding optimal alpha and beta values.
        """
        from ._train import train_model
        from .patching import EMLEPatched, PatchingLoss

        model = EMLEPatched(
            alpha=1.0,
            beta=1.0,
            emle_model=self._emle_base,
        )

        loss = PatchingLoss

        train_model(
            loss_class=loss,
            opt_param_names=["static_alpha", "induced_beta"],
            lr=lr,
            epochs=epochs,
            print_every=print_every,
            emle_model=model,
            e_static_target,
            e_ind_target,
            atomic_numbers,
            charges_mm,
            xyz_qm,
            xyz_mm
        )
    '''
