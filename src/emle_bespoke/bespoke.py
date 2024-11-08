import logging as _logging

import torch as _torch
from emle.models import EMLEBase as _EMLEBase
from emle.train import EMLETrainer as _EMLETrainer

logger = _logging.getLogger(__name__)


class BespokeModelTrainer:
    """
    Class to train bespoke EMLE models.

    Parameters
    ----------
    ref_sampler : ReferenceDataSampler
        The reference data sampler.
    emle_base : EMLEBase, optional
        The EMLE base model to use. Default is EMLEBase.
    """

    def __init__(self, ref_sampler, emle_base: _EMLEBase = _EMLEBase, filename_prefix: str ="ref"):
        self._ref_sampler = ref_sampler
        self._emle_base = emle_base
        self._filename_prefix = filename_prefix

    def train_model(
        self,
        n_samples=100,
        n_steps=1000,
        train_mask=None,
        alpha_mode="species",
        sigma=1e-3,
        ivm_thr=0.02,
        epochs=100,
        lr_qeq=0.01,
        lr_thole=0.01,
        lr_sqrtk=0.01,
        ref_data_filename=None,
        model_filename=None,
        device=_torch.device("cuda"),
        dtype=_torch.float64,
    ):
        """
        Sample reference data and train a bespoke model.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        n_steps : int
            The number of steps between each sample.
        train_mask : torch.Tensor, optional
            The training mask. Default is None, which means all data is used.
        alpha_mode : str, optional
            The alpha mode.
        sigma : float, optional
            The sigma value.
        ivm_thr : float, optional
            The ivm threshold.
        epochs : int, optional
            The number of epochs.
        lr_qeq : float, optional
            The learning rate for qeq.
        lr_thole : float, optional
            The learning rate for thole.
        lr_sqrtk : float, optional
            The learning rate for sqrtk.
        model_filename : str, optional
            The model filename.
        device : torch.device, optional
            The device. Default is cuda.
        dtype : torch.dtype, optional
            The data type. Default is float64.
        """
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
            ref_data = self._ref_sampler.sample(n_steps)
            logger.info(f"Sampled {i + 1}/{n_samples} configurations.")

        logger.info("Finished sampling reference data.")

        ref_data_filename = ref_data_filename or f"{self._filename_prefix}_ref_data.mat"
        self._ref_sampler.write_data(filename=ref_data_filename)

        msg = r"""
╔════════════════════════════════════════════════════════════╗
║         Starting training of bespoke EMLE model....        ║
╚════════════════════════════════════════════════════════════╝
"""
        for line in msg.split("\n"):
            logger.info(line)

        self._train(
            z=ref_data["z"],
            xyz=ref_data["xyz_qm"],
            s=ref_data["s"],
            q_core=ref_data["q_core"],
            q_val=ref_data["q_val"],
            alpha=ref_data["alpha"],
            train_mask=train_mask,
            alpha_mode=alpha_mode,
            sigma=sigma,
            ivm_thr=ivm_thr,
            epochs=epochs,
            lr_qeq=lr_qeq,
            lr_thole=lr_thole,
            lr_sqrtk=lr_sqrtk,
            model_filename=model_filename or f"{self._filename_prefix}_bespoke_model.mat",
            device=device,
            dtype=dtype,
        )

        logger.info("Finished training bespoke model.")

    def patch_model(self):
        """
        Patch the model by finding optimal alpha and beta values.
        """
        pass

    def _train(
        self,
        z,
        xyz,
        s,
        q_core,
        q_val,
        alpha,
        train_mask=None,
        alpha_mode="reference",
        sigma=1e-3,
        ivm_thr=0.02,
        epochs=100,
        lr_qeq=0.01,
        lr_thole=0.01,
        lr_sqrtk=0.01,
        model_filename="bespoke_model.mat",
        device=_torch.device("cuda"),
        dtype=_torch.float64,
    ):
        """
        Train a bespoke model.

        Parameters
        ----------
        """
        trainer = _EMLETrainer(self._emle_base)

        trainer.train(
            z=z,
            xyz=xyz,
            s=s,
            q_core=q_core,
            q_val=q_val,
            alpha=alpha,
            train_mask=train_mask,
            alpha_mode=alpha_mode,
            sigma=sigma,
            ivm_thr=ivm_thr,
            epochs=epochs,
            lr_qeq=lr_qeq,
            lr_thole=lr_thole,
            lr_sqrtk=lr_sqrtk,
            model_filename=model_filename,
            device=device,
            dtype=dtype,
        )

        return trainer
