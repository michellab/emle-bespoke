"""Module containg various thin wrappers for performing sampling and training tasks."""

from typing import TYPE_CHECKING, Any

import numpy as _np
import torch as _torch
from emle.models import EMLEBase as _EMLEBase
from emle.train._utils import pad_to_max
from loguru import logger as _logger
from torch.utils.data import DataLoader as _DataLoader

from .dataset import MolecularDataset as _MolecularDataset
from .reference_data import ReferenceData as _ReferenceData
from .utils import write_dict_to_file as _write_dict_to_file

if TYPE_CHECKING:
    pass


def sample_reference_data(
    sampler: Any,
    n_samples=100,
    n_steps=1000,
    calc_static: bool = True,
    calc_induction: bool = True,
    calc_horton: bool = True,
    calc_polarizability: bool = True,
) -> _ReferenceData:
    """
    Sample reference data using a provided sampler.

    Parameters
    ----------
    sampler : Any
        The sampler to use for sampling reference data.
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
    _logger.info("══════════════════════════════════════════════════════════════\n")
    _logger.info("Sampling reference data.")
    _logger.info(f"Number of samples to sample: {n_samples}")
    _logger.info(f"Number of steps per sample: {n_steps}")
    for i in range(n_samples):
        _logger.info(f"Sampling {i + 1}/{n_samples} configurations.")
        reference_data = sampler.sample(
            n_steps=n_steps,
            calc_static=calc_static,
            calc_induction=calc_induction,
            calc_horton=calc_horton,
            calc_polarizability=calc_polarizability,
            label=str(i),
        )
    _logger.info("Reference data sampled.")
    _logger.info("══════════════════════════════════════════════════════════════\n")
    return reference_data


def train_model(
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

    trainer = _EMLETrainer()

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
