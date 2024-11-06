# from emle.train import EMLETrainer as _EMLETrainer
import torch as _torch
from emle.models import EMLEBase as _EMLEBase
from emle.train import EMLETrainer as _EMLETrainer


class EMLEBespoke:
    """Class to train bespoke EMLE models."""

    def __init__(self, ref_calculator, emle_base: _EMLEBase = None):
        self._ref_calculator = ref_calculator
        self._emle_base = emle_base

    def train_model(
        self,
        n_samples=100,
        n_steps=1000,
        train_mask=None,
        alpha_mode="species",
        sigma=1e-3,
        ivm_thr=0.02,
        epochs=10000,
        lr_qeq=0.01,
        lr_thole=0.01,
        lr_sqrtk=0.01,
        model_filename="mymodel.mat",
        device=_torch.device("cuda"),
        dtype=_torch.float64,
    ):
        # Sample reference data
        for i in n_samples:
            self._ref_calculator.sample(n_steps)

        # Train bespoke model
        self._train(
            z=self._ref_calculator.ref_data["z"],
            xyz=self._ref_calculator.ref_data["xyz"],
            s=self._ref_calculator.ref_data["s"],
            q_core=self._ref_calculator.ref_data["q_core"],
            q=self._ref_calculator.ref_data["q"],
            alpha=self._ref_calculator.ref_data["alpha"],
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

    def patch_model(self, model_filename="mymodel.mat"):
        pass

    def _train(
        self,
        z,
        xyz,
        s,
        q_core,
        q,
        alpha,
        train_mask=None,
        alpha_mode="species",
        sigma=1e-3,
        ivm_thr=0.02,
        epochs=10000,
        lr_qeq=0.01,
        lr_thole=0.01,
        lr_sqrtk=0.01,
        model_filename="mymodel.mat",
        device=_torch.device("cuda"),
        dtype=_torch.float64,
    ):
        trainer = _EMLETrainer(self._emle_base)
        trainer.train(
            z=z,
            xyz=xyz,
            s=s,
            q_core=q_core,
            q=q,
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
