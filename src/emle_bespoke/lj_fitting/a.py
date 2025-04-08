import openmm.unit as _unit
import torch as _torch
from loguru import logger as _logger


class LennardJonesPotential(_torch.nn.Module):
    def __init__(
        self, topology_off, forcefield, parameters_to_fit, device=None, dtype=None
    ):
        super().__init__()
        self._forcefield = forcefield
        self._topology_off = (
            topology_off if isinstance(topology_off, list) else [topology_off]
        )
        self._parameters_to_fit = parameters_to_fit

        # Initialize device and dtype
        self._device = device or (
            _torch.device("cuda")
            if _torch.cuda.is_available()
            else _torch.device("cpu")
        )
        self._dtype = dtype or _torch.float64

        # Precompute LJ parameters and atom type mapping
        (
            self._atom_type_to_index,
            self._sigma,
            self._epsilon,
            self._atom_type_ids,
            self._lj_params,
        ) = self._build_lj_param_lookup()

        self._num_atom_types = len(self._atom_type_to_index)

        # Store initial values for sigma and epsilon without requiring gradients
        self._sigma_init = self._sigma.clone().detach()
        self._epsilon_init = self._epsilon.clone().detach()

        # Create trainable embedding layers for sigma and epsilon
        self._sigma_embedding = _torch.nn.Embedding.from_pretrained(
            self._sigma, freeze=False
        )
        self._epsilon_embedding = _torch.nn.Embedding.from_pretrained(
            self._epsilon, freeze=False
        )

        # Create frozen embedding layers for sigma and epsilon
        self._sigma_embedding_frozen = _torch.nn.Embedding.from_pretrained(
            self._sigma_init, freeze=True
        )
        self._epsilon_embedding_frozen = _torch.nn.Embedding.from_pretrained(
            self._epsilon_init, freeze=True
        )

        # Create embedding masks
        (
            self._sigma_embedding_mask,
            self._epsilon_embedding_mask,
        ) = self._build_embedding_masks()

        self._sigma_mask = self._sigma_embedding_mask(self._atom_type_ids).squeeze(-1)
        self._epsilon_mask = self._epsilon_embedding_mask(self._atom_type_ids).squeeze(
            -1
        )

    @staticmethod
    def print_lj_parameters(lj_params):
        _logger.debug("")
        _logger.debug("Lennard-Jones Parameters")
        _logger.debug("-" * 40)
        _logger.debug(
            f"{'Atom Type':16s} | {'\u03c3 (nm)':>8s} | {'\u03b5 (kJ/mol)':>12s}"
        )
        _logger.debug("-" * 40)
        for atom in lj_params:
            sigma = lj_params[atom]["sigma"]
            epsilon = lj_params[atom]["epsilon"]
            _logger.debug(f"{atom:16s} | {sigma:8.4f} | {epsilon:12.4f}")
        _logger.debug("-" * 40)

    def update_lj_parameters(self):
        for atom_type, index in self._atom_type_to_index.items():
            self._lj_params[atom_type]["sigma"] = self._sigma[index].item()
            self._lj_params[atom_type]["epsilon"] = self._epsilon[index].item()

    def _build_embedding_masks(self):
        sigma_mask = _torch.zeros_like(
            self._sigma_init, dtype=_torch.bool, device=self._device
        )
        epsilon_mask = _torch.zeros_like(
            self._epsilon_init, dtype=_torch.bool, device=self._device
        )

        for atype, params in self._parameters_to_fit.items():
            if atype == "all":
                if "sigma" in params:
                    sigma_mask.fill_(True)
                if "epsilon" in params:
                    epsilon_mask.fill_(True)
                continue

            index = self._atom_type
