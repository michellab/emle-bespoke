"""Child class of EMLETrainer that adds patching functionality."""

from typing import Optional, Type, Union

import torch as _torch
from emle.models import EMLE as _EMLE
from emle.models import EMLEBase as _EMLEBase
from emle.train import EMLETrainer as _EMLETrainer
from emle.train._loss import QEqLoss as _QEqLoss
from emle.train._loss import TholeLoss as _TholeLoss
from loguru import logger as _logger

from ..loss._patching import PatchingLoss as _PatchingLoss
from ._emle import EMLEPatched as _EMLEPatched


class EMLETrainer(_EMLETrainer):
    """
    Child class of EMLETrainer that adds patching functionality.

    Parameters
    ----------
    emle_base : EMLEBase
        The EMLEBase object to use for the training.
    qeq_loss : QEqLoss
        The QEqLoss object to use for the training.
    thole_loss : TholeLoss
        The TholeLoss object to use for the training.
    patch_loss : PatchLoss
        The PatchLoss object to use for the training.
    log_level : int
        The log level to use for the training.
    log_file : str
        The log file to use for the training.

    Attributes
    ----------
    Inherits all attributes from EMLETrainer.

    _patch_loss : PatchLoss
        The PatchLoss object to use for the training.
    """

    def __init__(
        self,
        emle_base: _EMLEBase,
        qeq_loss: _QEqLoss,
        thole_loss: _TholeLoss,
        patch_loss: Type[_PatchingLoss],
        log_level: Optional[int] = None,
        log_file: Optional[str] = None,
    ) -> None:
        super().__init__(emle_base, qeq_loss, thole_loss, log_level, log_file)
        if patch_loss is not _PatchingLoss:
            raise TypeError("patch_loss must be a reference to PatchingLoss")
        self.patch_loss = patch_loss

        self.emle_model = None

    @staticmethod
    def _patch_model(
        loss_class,
        opt_param_names,
        lr,
        epochs,
        emle_model,
        print_every=10,
        loss_class_kwargs=None,
        *args,
        **kwargs,
    ):
        """
        Train a model.

        Parameters
        ----------
        loss_class: class
            Loss class.
        opt_param_names: list of str
            List of parameter names to optimize.
        lr: float
            Learning rate.
        epochs: int
            Number of training epochs.
        emle: EMLE
            EMLE model instance.
        print_every: int
            How often to print training progress
        loss_class_kwargs: dict
                Keyword arguments to pass to the loss class besides the EMLE model.

        Returns
        -------
        model
            Trained model.
        """

        def _train_loop(
            loss_instance, optimizer, epochs, print_every=10, *args, **kwargs
        ):
            """
            Perform the training loop.

            Parameters
            ----------
            loss_instance: nn.Module
                Loss instance.
            optimizer: torch.optim.Optimizer
                Optimizer.
            epochs: int
                Number of training epochs.
            print_every: int
                How often to print training progress
            args: list
                Positional arguments to pass to the forward method.
            kwargs: dict
                Keyword arguments to pass to the forward method.

            Returns
            -------
            loss
                Forward loss.
            """
            for epoch in range(epochs):
                loss_instance.train()
                optimizer.zero_grad()
                loss, rmse, max_error = loss_instance(*args, **kwargs)
                loss.backward(retain_graph=False)
                optimizer.step()
                if (epoch + 1) % print_every == 0:
                    _logger.info(
                        f"Epoch {epoch + 1}: Loss ={loss.item():9.4f}    "
                        f"RMSE ={rmse.item():9.4f}    "
                        f"Max Error ={max_error.item():9.4f}"
                    )

            return loss

        # Additonal kwargs for the loss class
        loss_class_kwargs = loss_class_kwargs or {}

        model = loss_class(emle_model, **loss_class_kwargs)
        opt_parameters = [
            param
            for name, param in model.named_parameters()
            if any(opt_param in name.split(".", 1)[-1] for opt_param in opt_param_names)
        ]
        _logger.info(f"Optimizing parameters: {opt_param_names}")

        optimizer = _torch.optim.Adam(opt_parameters, lr=lr)
        _train_loop(model, optimizer, epochs, print_every, *args, **kwargs)
        return model

    def set_emle_model(self, emle_model: _EMLE) -> None:
        """
        Set the EMLE model to use for patching.

        Parameters
        ----------
        emle_model : EMLE
            The EMLE model to use.
        """
        self.emle_model = emle_model

    def get_emle_model(self) -> Optional[_EMLE]:
        """
        Get the current EMLE model.

        Returns
        -------
        EMLE or None
            The current EMLE model, or None if no model has been set.
        """
        return self.emle_model

    def patch(
        self,
        opt_param_names: list[str],
        emle_model: Optional[_EMLE] = None,
        emle_model_filename: str = None,
        alpha_mode: str = None,
        lr: float = 1e-3,
        epochs: int = 1000,
        e_static_target: Optional[_torch.Tensor] = None,
        e_ind_target: Optional[_torch.Tensor] = None,
        atomic_numbers: Optional[_torch.Tensor] = None,
        charges_mm: Optional[_torch.Tensor] = None,
        xyz_qm: Optional[_torch.Tensor] = None,
        xyz_mm: Optional[_torch.Tensor] = None,
        q_core: Optional[_torch.Tensor] = None,
        q_val: Optional[_torch.Tensor] = None,
        s: Optional[_torch.Tensor] = None,
        alpha: Optional[_torch.Tensor] = None,
        l2_reg_alpha: float = 1.0,
        l2_reg_s: float = 1.0,
        l2_reg_q: float = 1.0,
        n_batches: int = 32,
        filename_prefix: str = "patched_model",
        device: _torch.device = _torch.device("cuda"),
        dtype: _torch.dtype = _torch.float64,
    ) -> None:
        """
        Patch the EMLE model by training on target energies.

        Parameters
        ----------
        opt_param_names : list[str]
            The names of the parameters to optimize.
        emle_model : EMLE, optional
            An existing EMLE model to use. If provided, emle_model_filename and alpha_mode are ignored.
        emle_model_filename : str, optional
            The filename of the EMLE model to patch. Required if emle_model is not provided and no model exists.
        alpha_mode : str, optional
            The alpha mode to use for the EMLE model. Required if emle_model is not provided and no model exists.
        lr : float, default=1e-3
            Learning rate for the patching.
        epochs : int, default=1000
            Number of epochs for the patching.
        e_static_target : torch.Tensor, optional
            Target static energy values, shape (n_samples,)
        e_ind_target : torch.Tensor, optional
            Target induced energy values, shape (n_samples,)
        atomic_numbers : torch.Tensor, optional
            Atomic numbers of QM atoms, shape (n_samples, n_qm_atoms)
        charges_mm : torch.Tensor, optional
            MM point charges, shape (n_samples, n_mm_atoms)
        xyz_qm : torch.Tensor, optional
            QM atom coordinates in Angstrom, shape (n_samples, n_qm_atoms, 3)
        xyz_mm : torch.Tensor, optional
            MM atom coordinates in Angstrom, shape (n_samples, n_mm_atoms, 3)
        q_core : torch.Tensor, optional
            Core charges for regularization, shape (n_samples, n_qm_atoms)
        q_val : torch.Tensor, optional
            Valence charges for regularization, shape (n_samples, n_qm_atoms)
        s : torch.Tensor, optional
            Static component for regularization, shape (n_samples, n_qm_atoms)
        alpha : torch.Tensor, optional
            Polarizability for regularization, shape (n_samples, n_qm_atoms)
        l2_reg_alpha : float, default=1.0
            L2 regularization weight for alpha
        l2_reg_s : float, default=1.0
            L2 regularization weight for s
        l2_reg_q : float, default=1.0
            L2 regularization weight for charges
        n_batches : int, default=32
            Number of batches to split data into
        filename_prefix : str, default="patched_model"
            Filename prefix to save patched model
        """
        # Use provided model, existing model, or create new one
        if emle_model is not None:
            self.emle_model = emle_model
        elif self.emle_model is None:
            if emle_model_filename is None or alpha_mode is None:
                raise ValueError(
                    "emle_model_filename and alpha_mode are required when no EMLE model exists"
                )
            self.emle_model = _EMLE(
                model=emle_model_filename,
                alpha_mode=alpha_mode,
                device=device,
                dtype=dtype,
            )

        self._patch_model(
            loss_class=self.patch_loss,
            opt_param_names=opt_param_names,
            lr=lr,
            epochs=epochs,
            emle_model=self.emle_model,
            e_static_target=e_static_target,
            e_ind_target=e_ind_target,
            atomic_numbers=atomic_numbers,
            charges_mm=charges_mm,
            xyz_qm=xyz_qm,
            xyz_mm=xyz_mm,
            q_core=q_core,
            q_val=q_val,
            s=s,
            alpha=alpha,
            l2_reg_alpha=l2_reg_alpha,
            l2_reg_s=l2_reg_s,
            l2_reg_q=l2_reg_q,
            n_batches=n_batches,
        )

        # Update the EMLE model with the new parameters
        self.patch_loss._update_s_gpr(self.emle_model._emle_base)
        self.patch_loss._update_chi_gpr(self.emle_model._emle_base)
        if self.emle_model._alpha_mode == "reference":
            self.patch_loss._update_sqrtk_gpr(self.emle_model._emle_base)

        # Write the model to a file
        self._write_model_to_file(self.get_emle_model_dict(), f"{filename_prefix}.mat")

    def get_emle_model_dict(self) -> dict:
        """
        Recreate the EMLE model dictionary from the EMLEBase object.

        Returns
        -------
        emle_model : dict
            The EMLE model dictionary.
        """
        emle_model = {
            "q_core": self.emle_model._emle_base._q_core,
            "a_QEq": self.emle_model._emle_base.a_QEq,
            "a_Thole": self.emle_model._emle_base.a_Thole,
            "s_ref": self.emle_model._emle_base.ref_values_s,
            "chi_ref": self.emle_model._emle_base.ref_values_chi,
            "k_Z": self.emle_model._emle_base.k_Z,
            "sqrtk_ref": self.emle_model._emle_base.ref_values_sqrtk
            if self.emle_model._emle_base._alpha_mode == "reference"
            else None,
            "species": [
                i
                for i, val in enumerate(self.emle_model._emle_base._species_map)
                if val != -1
            ],
            "n_ref": self.emle_model._emle_base._n_ref,
            "ref_aev": self.emle_model._emle_base._ref_features,
            "aev_mask": self.emle_model._emle_base._emle_aev_computer._mask,
            "zid_map": self.emle_model._emle_base._emle_aev_computer._zid_map,
            "computer_n_species": len(self.emle_model._emle_base._n_ref),
        }

        return emle_model
