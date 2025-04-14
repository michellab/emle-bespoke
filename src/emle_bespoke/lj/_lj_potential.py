"""Lennard-Jones potential energy calculation module."""

from typing import Dict, List, Optional, Tuple, Union

import openmm.unit as _unit
import torch as _torch
from loguru import logger as _logger
from openff.toolkit import ForceField as _ForceField
from openff.toolkit import Topology as _Topology
from torch.utils.checkpoint import checkpoint as _checkpoint


class LennardJonesPotential(_torch.nn.Module):
    """
    Lennard-Jones potential energy calculation.

    This class implements a differentiable Lennard-Jones potential that can be used
    for parameter optimization. It supports fitting both sigma and epsilon parameters
    for specific atom types.

    Parameters
    ----------
    topology_off : Union[_Topology, List[_Topology]]
        The OpenFF Topology object(s).
    forcefield : _ForceField
        The OpenFF ForceField object.
    parameters_to_fit : Dict[str, Dict[str, bool]]
        A dictionary specifying which parameters to fit for each atom type.
        Example: {"n-tip3p-O": {"sigma": True, "epsilon": True}}
    device : Optional[_torch.device]
        The device to use for calculations.
    dtype : Optional[_torch.dtype]
        The data type to use for calculations.
    """

    def __init__(
        self,
        topology_off: Union[_Topology, List[_Topology]],
        forcefield: _ForceField,
        parameters_to_fit: Dict[str, Dict[str, bool]],
        device: Optional[_torch.device] = None,
        dtype: Optional[_torch.dtype] = None,
    ) -> None:
        _logger.debug("Initializing Lennard-Jones potential")
        super().__init__()
        self._forcefield = forcefield
        self._topology_off = (
            topology_off if isinstance(topology_off, list) else [topology_off]
        )
        self._parameters_to_fit = parameters_to_fit

        self._device = device or (
            _torch.device("cuda")
            if _torch.cuda.is_available()
            else _torch.device("cpu")
        )
        self._dtype = dtype or _torch.float64

        # Initialize LJ parameters and mappings
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """
        Initialize all LJ parameters and mappings.

        This method sets up the core components needed for LJ potential calculation:
        1. Builds parameter lookup tables
        2. Stores initial parameter values
        3. Initializes embedding layers
        4. Sets up gradient masks

        The initialization process ensures that:
        - Atom types are properly mapped to indices
        - Initial parameter values are stored for regularization
        - Embedding layers are ready for training
        - Gradient masks are properly applied
        """
        # Build parameter lookup tables
        (
            self._atom_type_to_index,
            self._sigma,
            self._epsilon,
            self._atom_type_ids,
            self._lj_params,
        ) = self._build_lj_param_lookup()

        self._num_atom_types = len(self._atom_type_to_index)

        # Store initial values, must be detached because they are used for regularization
        self._sigma_init = self._sigma.clone().detach()
        self._epsilon_init = self._epsilon.clone().detach()

        # Initialize embedding layers
        self._initialize_embeddings()

        # Build and apply gradient masks
        self._initialize_masks()

    def _initialize_embeddings(self) -> None:
        """
        Initialize embedding layers for sigma and epsilon parameters.

        Creates trainable embedding layers that map atom type indices to their
        corresponding LJ parameters. These layers are initialized with the
        initial parameter values but are allowed to be modified during training.

        The embeddings are used to efficiently look up parameters during
        energy calculations while maintaining differentiability for optimization.
        """
        self._sigma_embedding = _torch.nn.Embedding.from_pretrained(
            self._sigma, freeze=False
        )
        self._epsilon_embedding = _torch.nn.Embedding.from_pretrained(
            self._epsilon, freeze=False
        )

    def _initialize_masks(self) -> None:
        """
        Initialize and apply gradient masks for parameter optimization.

        This method:
        1. Builds masks indicating which parameters should be trainable
        2. Creates embedding masks for efficient parameter lookup
        3. Registers gradient hooks to enforce the masks during backpropagation

        The masks ensure that only specified parameters are updated during training,
        while others remain fixed at their initial values.
        """
        (
            self._sigma_grad_mask,
            self._epsilon_grad_mask,
            self._sigma_embedding_mask,
            self._epsilon_embedding_mask,
        ) = self._build_embedding_masks()

        self._sigma_mask = self._sigma_embedding_mask(self._atom_type_ids).squeeze(-1)
        self._epsilon_mask = self._epsilon_embedding_mask(self._atom_type_ids).squeeze(
            -1
        )

        # Register gradient hooks
        self._sigma_embedding.weight.register_hook(self._apply_sigma_gradient_mask)
        self._epsilon_embedding.weight.register_hook(self._apply_epsilon_gradient_mask)

    def _apply_sigma_gradient_mask(self, grad: _torch.Tensor) -> _torch.Tensor:
        """
        Apply gradient mask for sigma parameters during backpropagation.

        Parameters
        ----------
        grad : _torch.Tensor
            The gradient tensor for sigma parameters.

        Returns
        -------
        _torch.Tensor
            The masked gradient tensor where non-trainable parameters have zero gradient.
        """
        return grad * self._sigma_grad_mask

    def _apply_epsilon_gradient_mask(self, grad: _torch.Tensor) -> _torch.Tensor:
        """
        Apply gradient mask for epsilon parameters during backpropagation.

        Parameters
        ----------
        grad : _torch.Tensor
            The gradient tensor for epsilon parameters.

        Returns
        -------
        _torch.Tensor
            The masked gradient tensor where non-trainable parameters have zero gradient.
        """
        return grad * self._epsilon_grad_mask

    def print_lj_parameters(self, lj_params: Optional[Dict] = None) -> None:
        """
        Print the Lennard-Jones parameters in a formatted table.

        Parameters
        ----------
        lj_params : Dict
            A dictionary of Lennard-Jones parameters.
            The keys are atom types and the values are dictionaries with the parameters to fit.
            For example:
            {
                "n-tip3p-O": {"sigma": 0.315, "epsilon": 0.63},
                "n-tip3p-H": {"sigma": 0.235, "epsilon": 0.0},
            }
        """
        if lj_params is None:
            lj_params = self._lj_params

        _logger.debug("-" * 50)
        _logger.debug("Lennard-Jones Parameters")
        _logger.debug("-" * 50)
        _logger.debug(f"{'Atom Type':16s} | {'σ (Å)':>8s} | {'ε (kJ/mol)':>12s}")
        _logger.debug("-" * 50)
        for atom, params in lj_params.items():
            sigma = params["sigma"]
            epsilon = params["epsilon"]
            _logger.debug(f"{atom:16s} | {sigma:8.4f} | {epsilon:12.4f}")
        _logger.debug("-" * 50)

    def update_lj_parameters(self) -> None:
        """Update LJ parameters in both internal storage and ForceField object."""
        from openff.units import unit as _offunit

        # Update internal parameter dictionary
        for atom_type, index in self._atom_type_to_index.items():
            self._lj_params[atom_type]["sigma"] = _torch.abs(self._sigma[index]).item()
            self._lj_params[atom_type]["epsilon"] = _torch.abs(
                self._epsilon[index]
            ).item()

        # Update ForceField object
        for param in self._forcefield["vdW"].parameters:
            if param.id in self._lj_params:
                param.sigma = _offunit.Quantity(
                    self._lj_params[param.id]["sigma"], _offunit.angstrom
                )
                param.epsilon = _offunit.Quantity(
                    self._lj_params[param.id]["epsilon"], _offunit.kilojoule_per_mole
                )

    def write_lj_parameters(self, filename_prefix: str = "lj_parameters") -> None:
        """
        Write LJ parameters to files.

        Parameters
        ----------
        filename_prefix : str
            The prefix for the output files.
        """
        # Write to .dat file
        with open(f"{filename_prefix}.dat", "w") as f:
            for atom_type, params in self._lj_params.items():
                f.write(f"{atom_type} {params['sigma']} {params['epsilon']}\n")

        # Write to OFFXML file
        self._forcefield.to_file(f"{filename_prefix}.offxml")

    def _build_embedding_masks(
        self,
    ) -> Tuple[_torch.Tensor, _torch.Tensor, _torch.nn.Embedding, _torch.nn.Embedding]:
        """
        Build masks for trainable embedding layers.

        Creates boolean masks indicating which parameters should be trainable
        based on the parameters_to_fit dictionary. These masks are used to:
        1. Control which parameters are updated during training
        2. Create efficient embedding layers for parameter lookup

        Returns
        -------
        Tuple[_torch.Tensor, _torch.Tensor, _torch.nn.Embedding, _torch.nn.Embedding]
            A tuple containing:
            - sigma_mask: Boolean tensor for trainable sigma parameters
            - epsilon_mask: Boolean tensor for trainable epsilon parameters
            - sigma_embedding_mask: Embedding layer for sigma mask lookup
            - epsilon_embedding_mask: Embedding layer for epsilon mask lookup
        """
        sigma_mask = _torch.zeros_like(
            self._sigma_init, dtype=_torch.bool, device=self._device
        )
        epsilon_mask = _torch.zeros_like(
            self._epsilon_init, dtype=_torch.bool, device=self._device
        )

        for atype, params in self._parameters_to_fit.items():
            if atype == "all":
                if "sigma" in params:
                    sigma_mask[1:] = True
                if "epsilon" in params:
                    epsilon_mask[1:] = True
                continue

            index = self._atom_type_to_index.get(atype)
            if index is not None:
                if params.get("sigma", False):
                    sigma_mask[index] = True
                if params.get("epsilon", False):
                    epsilon_mask[index] = True

        sigma_mask_embedding = _torch.nn.Embedding.from_pretrained(
            sigma_mask, freeze=True
        )
        epsilon_mask_embedding = _torch.nn.Embedding.from_pretrained(
            epsilon_mask, freeze=True
        )

        return sigma_mask, epsilon_mask, sigma_mask_embedding, epsilon_mask_embedding

    def _build_lj_param_lookup(
        self,
    ) -> Tuple[Dict[str, int], _torch.Tensor, _torch.Tensor, _torch.Tensor, Dict]:
        """
        Build lookup tables for Lennard-Jones parameters.

        This method:
        1. Extracts LJ parameters from the ForceField object
        2. Creates mappings between atom types and indices
        3. Initializes parameter tensors with proper device and dtype
        4. Builds atom type ID tensors for efficient parameter lookup

        Note
        ----
        Atom types are mapped to indices starting from 1.
        The null atom type is mapped to index 0, which is used for padding.

        Returns
        -------
        Tuple[Dict[str, int], _torch.Tensor, _torch.Tensor, _torch.Tensor, Dict]
            A tuple containing:
            - atom_type_to_index: Dictionary mapping atom types to indices
            - sigma_init: Initial sigma values tensor
            - epsilon_init: Initial epsilon values tensor
            - atom_type_ids: Tensor of atom type IDs for each configuration
            - lj_params: Dictionary of LJ parameters for each atom type
        """
        lj_params = {}
        atom_type_to_index = {}
        sigma_init = [1.0]  # Index 0 is padding
        epsilon_init = [0.0]  # Index 0 is padding
        atom_type_ids = []

        prev_topology = None
        for i, topology in enumerate(self._topology_off):
            if i % 1000 == 0:
                _logger.debug(f"Processing topology {i} / {len(self._topology_off)}")

            if topology == prev_topology:
                atom_type_ids.append(atom_type_ids[-1])
                continue

            prev_topology = topology
            atom_type_ids_topology = []
            labels = self._forcefield.label_molecules(topology)

            for mol in labels:
                for _, val in mol["vdW"].items():
                    if val.id not in lj_params:
                        sigma = val.sigma.to_openmm().in_units_of(_unit.angstrom)._value
                        epsilon = (
                            val.epsilon.to_openmm()
                            .in_units_of(_unit.kilojoule_per_mole)
                            ._value
                        )

                        lj_params[val.id] = {"sigma": sigma, "epsilon": epsilon}
                        atom_type_to_index[val.id] = len(atom_type_to_index) + 1
                        sigma_init.append(sigma)
                        epsilon_init.append(epsilon)

                    atom_type_ids_topology.append(atom_type_to_index[val.id])

            atom_type_ids.append(
                _torch.tensor(atom_type_ids_topology, dtype=_torch.int64)
            )

        # Convert to tensors and move to device
        atom_type_ids = _torch.nn.utils.rnn.pad_sequence(
            atom_type_ids, batch_first=True, padding_value=0
        ).to(device=self._device, dtype=_torch.int64)

        sigma_init = _torch.tensor(
            sigma_init, dtype=self._dtype, device=self._device, requires_grad=True
        ).unsqueeze(-1)

        epsilon_init = _torch.tensor(
            epsilon_init, dtype=self._dtype, device=self._device, requires_grad=True
        ).unsqueeze(-1)

        self.print_lj_parameters(lj_params)

        return atom_type_to_index, sigma_init, epsilon_init, atom_type_ids, lj_params

    def _compute_energy(
        self,
        atom_type_ids: _torch.Tensor,
        xyz: _torch.Tensor,
        solute_mask: _torch.Tensor,
        solvent_mask: _torch.Tensor,
    ) -> _torch.Tensor:
        """
        Compute Lennard-Jones energy for a batch of configurations.

        This method calculates the LJ potential energy between solute and solvent
        atoms using the current parameter values. The calculation is performed
        efficiently using vectorized operations and proper masking.

        Parameters
        ----------
        atom_type_ids : _torch.Tensor
            Tensor of atom type IDs for each atom in each configuration.
        xyz : _torch.Tensor
            Particle positions of shape (batch, natoms, 3).
        solute_mask : _torch.Tensor
            Boolean mask for solute atoms of shape (batch, natoms).
        solvent_mask : _torch.Tensor
            Boolean mask for solvent atoms of shape (batch, natoms).

        Returns
        -------
        _torch.Tensor
            Total LJ energy for each configuration in the batch.
        """
        sigma = self._sigma_embedding(atom_type_ids).squeeze(-1)
        epsilon = self._epsilon_embedding(atom_type_ids).squeeze(-1)

        # Apply masks
        solute_sigma = sigma * solute_mask
        solvent_sigma = sigma * solvent_mask
        solute_epsilon = _torch.abs(epsilon * solute_mask) + 1e-16
        solvent_epsilon = _torch.abs(epsilon * solvent_mask) + 1e-16

        xyz_qm = xyz * solute_mask.unsqueeze(-1)
        xyz_mm = xyz * solvent_mask.unsqueeze(-1)

        # Compute pairwise distances
        distances = _torch.cdist(xyz_mm, xyz_qm)
        distances = _torch.where(distances > 0, distances, 1e32)

        # Compute LJ parameters
        sigma_ij = 0.5 * (solvent_sigma[:, :, None] + solute_sigma[:, None, :])
        epsilon_ij = _torch.sqrt(
            solvent_epsilon[:, :, None] * solute_epsilon[:, None, :]
        )

        # Compute LJ potential
        inv_r = sigma_ij / distances
        inv_r6 = inv_r**6
        inv_r12 = inv_r6 * inv_r6

        # Compute energy matrix
        energy_matrix = (
            4
            * epsilon_ij
            * (inv_r12 - inv_r6)
            * solvent_mask[:, :, None]
            * solute_mask[:, None, :]
        )

        return _torch.sum(energy_matrix, dim=(1, 2))

    def forward(
        self,
        xyz: _torch.Tensor,
        solute_mask: _torch.Tensor,
        solvent_mask: _torch.Tensor,
        indices: _torch.Tensor,
        checkpoint: bool = True,
    ) -> _torch.Tensor:
        """
        Calculate the Lennard-Jones potential energy.

        This is the main forward pass method that computes the LJ energy
        for a batch of configurations. It supports:
        1. Batch processing with start and end indices
        2. Gradient checkpointing for memory efficiency
        3. Automatic parameter updates after energy calculation

        Parameters
        ----------
        xyz : _torch.Tensor
            Particle positions of shape (batch, natoms, 3).
        solute_mask : _torch.Tensor
            Boolean mask for solute atoms of shape (batch, natoms).
        solvent_mask : _torch.Tensor
            Boolean mask for solvent atoms of shape (batch, natoms).
        indices : _torch.Tensor
            Indices of the topologies for which to compute the energy.
        checkpoint : bool, optional
            Whether to use gradient checkpointing. Default is True.

        Returns
        -------
        _torch.Tensor
            Total LJ energy for each configuration in the batch.
        """
        atom_type_ids = self._atom_type_ids[indices]

        if checkpoint:
            total_energy = _checkpoint(
                self._compute_energy,
                atom_type_ids,
                xyz,
                solute_mask,
                solvent_mask,
                use_reentrant=False,
            )
        else:
            total_energy = self._compute_energy(
                atom_type_ids, xyz, solute_mask, solvent_mask
            )

        self.update_lj_parameters()

        return total_energy
