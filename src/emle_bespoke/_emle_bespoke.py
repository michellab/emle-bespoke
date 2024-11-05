# from emle.train import EMLETrainer as _EMLETrainer
from emle.models import EMLEBase as _EMLEBase

from ._sampler import ReferenceDataCalculator as _ReferenceDataCalculator
import torch as _torch

class EMLEBespoke:
    """Class to train bespoke EMLE models."""
    def __init__(self, emle_base=_EMLEBase):
        self._emle_base = emle_base

    def train(
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

    def run(n_samples, n_steps):
        """
        
        
        """
        ref_calculator = _ReferenceDataCalculator()

        for i in(n_samples):
            ref_calculator.sample(n_steps)

        train

