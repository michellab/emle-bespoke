"""Memory-efficient Lennard-Jones potential energy calculation module."""

from typing import Dict, List, Optional, Tuple, Union

import openmm.unit as _unit
import torch as _torch
from loguru import logger as _logger
from openff.toolkit import ForceField as _ForceField
from openff.toolkit import Topology as _Topology
from torch.utils.checkpoint import checkpoint as _checkpoint


class LennardJonesPotentialEfficient(_torch.nn.Module):
    """
    Painfully slow, but highly memory-efficient Lennard-Jones potential energy calculation.

    This class implements a differentiable Lennard-Jones potential that can be used
    for parameter optimization. It supports fitting both sigma and epsilon parameters
    for specific atom types while minimizing memory usage.

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
    print_every : int
        The number of configurations to process before printing a progress update.
    """

    def __init__(
        self,
        topology_off: Union[_Topology, List[_Topology]],
        forcefield: _ForceField,
        parameters_to_fit: Dict[str, Dict[str, bool]],
        device: Optional[_torch.device] = None,
        dtype: Optional[_torch.dtype] = None,
        print_every: int = 1000,
    ) -> None:
        _logger.debug("Initializing efficient Lennard-Jones potential")
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

        self._print_every = print_every

        # Initialize LJ parameters and mappings
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """
        Initialize all LJ parameters and mappings.
        """
        # Build parameter lookup tables
        (
            self._atom_type_to_index,
            self._sigma,
            self._epsilon,
            self._lj_params,
        ) = self._build_lj_param_lookup()

        self._num_atom_types = len(self._atom_type_to_index)

        # Store initial values for regularization
        self._sigma_init = self._sigma.clone().detach()
        self._epsilon_init = self._epsilon.clone().detach()

        # Sigma and epsilon should be optimizable
        self._sigma = _torch.nn.Parameter(self._sigma)
        self._epsilon = _torch.nn.Parameter(self._epsilon)

        # Build and apply gradient masks
        self._initialize_masks()

    def _initialize_masks(self) -> None:
        """
        Initialize and apply gradient masks for parameter optimization.
        """
        self._sigma_grad_mask, self._epsilon_grad_mask = self._build_gradient_masks()

        # Register gradient hooks
        self._sigma.register_hook(self._apply_sigma_gradient_mask)
        self._epsilon.register_hook(self._apply_epsilon_gradient_mask)

    def _apply_sigma_gradient_mask(self, grad: _torch.Tensor) -> _torch.Tensor:
        """
        Apply gradient mask for sigma parameters during backpropagation.
        """
        return grad * self._sigma_grad_mask

    def _apply_epsilon_gradient_mask(self, grad: _torch.Tensor) -> _torch.Tensor:
        """
        Apply gradient mask for epsilon parameters during backpropagation.
        """
        return grad * self._epsilon_grad_mask

    def _build_gradient_masks(self) -> Tuple[_torch.Tensor, _torch.Tensor]:
        """
        Build masks for trainable parameters.
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

        return sigma_mask, epsilon_mask

    def _build_lj_param_lookup(
        self,
    ) -> Tuple[Dict[str, int], _torch.Tensor, _torch.Tensor, Dict]:
        """
        Build lookup tables for Lennard-Jones parameters.
        """
        lj_params = {}
        atom_type_to_index = {}
        sigma_init = [1.0]  # Index 0 is padding
        epsilon_init = [0.0]  # Index 0 is padding

        # Process all topologies to get all unique atom types
        prev_topology = None
        for i, topology in enumerate(self._topology_off):
            if i % 1000 == 0:
                _logger.debug(f"Processing topology {i} / {len(self._topology_off)}")

            if topology == prev_topology:
                continue

            prev_topology = topology
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

        # Convert to tensors and move to device
        sigma_init = _torch.tensor(
            sigma_init, dtype=self._dtype, device=self._device, requires_grad=True
        ).unsqueeze(-1)

        epsilon_init = _torch.tensor(
            epsilon_init, dtype=self._dtype, device=self._device, requires_grad=True
        ).unsqueeze(-1)

        self.print_lj_parameters(lj_params)

        return atom_type_to_index, sigma_init, epsilon_init, lj_params

    def _get_atom_type_ids(self, topology: _Topology) -> _torch.Tensor:
        """
        Get atom type IDs for a single topology.
        """
        atom_type_ids = []
        labels = self._forcefield.label_molecules(topology)

        for mol in labels:
            for _, val in mol["vdW"].items():
                atom_type_ids.append(self._atom_type_to_index[val.id])

        return _torch.tensor(atom_type_ids, dtype=_torch.int64, device=self._device)

    def _compute_energy(
        self,
        atom_type_ids: _torch.Tensor,
        xyz: _torch.Tensor,
        solute_mask: _torch.Tensor,
        solvent_mask: _torch.Tensor,
    ) -> _torch.Tensor:
        """
        Compute Lennard-Jones energy for a batch of configurations.
        """
        # Directly index parameters instead of using embeddings
        natoms = len(atom_type_ids)
        sigma = self._sigma[atom_type_ids].squeeze(-1)
        epsilon = self._epsilon[atom_type_ids].squeeze(-1)

        # We need to truncate the masks and positions to the number of atoms in the topology
        solvent_mask = solvent_mask[:, :natoms]
        solute_mask = solute_mask[:, :natoms]
        xyz = xyz[:, :natoms, :]

        # Apply masks and compute parameters in one go
        solute_sigma = sigma * solute_mask
        solvent_sigma = sigma * solvent_mask
        solute_epsilon = _torch.abs(epsilon * solute_mask) + 1e-16
        solvent_epsilon = _torch.abs(epsilon * solvent_mask) + 1e-16

        # Compute masked positions
        xyz_qm = xyz * solute_mask.unsqueeze(-1)
        xyz_mm = xyz * solvent_mask.unsqueeze(-1)

        # Compute pairwise distances efficiently
        distances = _torch.cdist(xyz_mm, xyz_qm)
        distances = _torch.where(distances > 0, distances, 1e32)

        # Compute LJ parameters efficiently
        sigma_ij = 0.5 * (solvent_sigma[:, :, None] + solute_sigma[:, None, :])
        epsilon_ij = _torch.sqrt(
            solvent_epsilon[:, :, None] * solute_epsilon[:, None, :]
        )

        # Compute LJ potential efficiently
        inv_r = sigma_ij / distances
        inv_r6 = inv_r**6
        inv_r12 = inv_r6 * inv_r6

        # Compute energy matrix with proper masking
        energy_matrix = (
            4
            * epsilon_ij
            * (inv_r12 - inv_r6)
            * solvent_mask[:, :, None]
            * solute_mask[:, None, :]
        )

        # Sum over dimensions efficiently
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

        Parameters
        ----------
        xyz : _torch.Tensor
            Particle positions of shape (batch, natoms, 3).
        solute_mask : _torch.Tensor
            Boolean mask for solute atoms of shape (batch, natoms).
        solvent_mask : _torch.Tensor
            Boolean mask for solvent atoms of shape (batch, natoms).
        indices : _torch.Tensor
            Indices of the topologies to use for each configuration in the batch.
        checkpoint : bool, optional
            Whether to use gradient checkpointing. Default is True.

        Returns
        -------
        _torch.Tensor
            Total LJ energy for each configuration in the batch.
        """
        # Pre-allocate output tensor
        total_energy = _torch.zeros(
            len(indices), device=self._device, dtype=self._dtype
        )

        # Process each configuration in the batch
        for i, idx in enumerate(indices):
            if i % self._print_every == 0:
                _logger.debug(
                    f"Computing LJ energy for configuration {i} / {len(indices)}"
                )
            # Get atom type IDs for this topology
            atom_type_ids = self._get_atom_type_ids(self._topology_off[idx])

            # Get the current configuration
            xyz_i = xyz[i : i + 1]  # Keep batch dimension
            solute_mask_i = solute_mask[i : i + 1]
            solvent_mask_i = solvent_mask[i : i + 1]

            if checkpoint:
                energy = _checkpoint(
                    self._compute_energy,
                    atom_type_ids,
                    xyz_i,
                    solute_mask_i,
                    solvent_mask_i,
                    use_reentrant=False,
                )
            else:
                energy = self._compute_energy(
                    atom_type_ids, xyz_i, solute_mask_i, solvent_mask_i
                )

            # Store energy directly in pre-allocated tensor
            total_energy[i] = energy[0]

        self.update_lj_parameters()

        return total_energy

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

    def print_lj_parameters(self, lj_params: Optional[Dict] = None) -> None:
        """
        Print the Lennard-Jones parameters in a formatted table.
        """
        if lj_params is None:
            lj_params = self._lj_params

        _logger.debug("")
        _logger.debug("Lennard-Jones Parameters")
        _logger.debug("-" * 40)
        _logger.debug(f"{'Atom Type':16s} | {'σ (nm)':>8s} | {'ε (kJ/mol)':>12s}")
        _logger.debug("-" * 40)
        for atom, params in lj_params.items():
            sigma = params["sigma"]
            epsilon = params["epsilon"]
            _logger.debug(f"{atom:16s} | {sigma:8.4f} | {epsilon:12.4f}")
        _logger.debug("-" * 40)

    def write_lj_parameters(self, filename_prefix: str = "lj_parameters") -> None:
        """
        Write LJ parameters to files.
        """
        # Write to .dat file
        with open(f"{filename_prefix}.dat", "w") as f:
            for atom_type, params in self._lj_params.items():
                f.write(f"{atom_type} {params['sigma']} {params['epsilon']}\n")

        # Write to OFFXML file
        self._forcefield.to_file(f"{filename_prefix}.offxml")
