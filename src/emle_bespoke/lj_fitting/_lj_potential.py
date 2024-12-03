import openmm.unit as _unit
import torch as _torch
from loguru import logger as _logger


class LennardJonesPotential(_torch.nn.Module):
    """
    Lennard-Jones potential energy calculation.

    Parameters
    ----------
    topology_off : list of openff.toolkit.topology.Topology or openff.toolkit.topology.Molecule
        The OpenFF Topology object.
    forcefield : openff.toolkit.typing.engines.smirnoff.ForceField
        The OpenFF ForceField object.
    parameters_to_fit : dict of dict
        A dictionary of atom types and parameters to fit.
        The keys are atom types and the values are dictionaries with the parameters to fit.
        For example:
        {
            "n-tip3p-O": {"sigma": True, "epsilon": True}
        }
    device : torch.device, optional
        The device to use for the calculation.
    dtype : torch.dtype, optional
        The data type to use for the calculation.
    """

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
        """Print the Lennard-Jones parameters."""
        _logger.debug("")
        _logger.debug("Lennard-Jones Parameters")
        _logger.debug("-" * 40)
        _logger.debug(f"{'Atom Type':16s} | {'σ (nm)':>8s} | {'ε (kJ/mol)':>12s}")
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
        """
        Build masks for trainable embedding layers.

        Returns
        -------
        sigma_mask : torch.Tensor(N_ATOM_TYPES)
            A boolean mask for trainable sigma parameters.
        epsilon_mask : torch.Tensor(N_ATOM_TYPES)
            A boolean mask for trainable epsilon parameters.
        """
        sigma_mask = _torch.zeros_like(
            self._sigma_init, dtype=_torch.bool, device=self._device
        )
        epsilon_mask = _torch.zeros_like(
            self._epsilon_init, dtype=_torch.bool, device=self._device
        )

        for atype, params in self._parameters_to_fit.items():
            if atype == "all":
                # If 'all' atom types are trainable, set entire masks to True
                if "sigma" in params:
                    sigma_mask.fill_(True)
                if "epsilon" in params:
                    epsilon_mask.fill_(True)
                continue

            # Handle specific atom types
            index = self._atom_type_to_index.get(atype)
            if index is not None:
                if "sigma" in params:
                    sigma_mask[index] = True
                if "epsilon" in params:
                    epsilon_mask[index] = True

        sigma_mask_embedding = _torch.nn.Embedding.from_pretrained(
            sigma_mask, freeze=True
        )
        epsilon_mask_embedding = _torch.nn.Embedding.from_pretrained(
            epsilon_mask, freeze=True
        )

        return sigma_mask_embedding, epsilon_mask_embedding

    def _build_lj_param_lookup(self):
        """
        Build a lookup table for Lennard-Jones parameters.

        Note
        ----
        Atom types are mapped to indices starting from 1.
        The null atom type is mapped to index 0, which is used for padding.

        Returns
        -------
        atom_type_to_index : dict[str, int]
            A dictionary mapping atom types to indices.
        sigma_init : torch.Tensor(N_ATOM_TYPES)
            Initial values for sigma.
        epsilon_init : torch.Tensor(N_ATOM_TYPES)
            Initial values for epsilon.
        atom_type_ids : torch.Tensor(BATCH, NATOMS)
            Tensor of atom type IDs for each particle in each configuration.
        lj_params : dict
            A dictionary of Lennard-Jones parameters.
        """
        lj_params = {}
        atom_type_to_index = {}
        sigma_init = [1.0]
        epsilon_init = [0.0]
        atom_type_ids = []

        for topology in self._topology_off:
            atom_type_ids_topology = []
            labels = self._forcefield.label_molecules(topology)
            for mol in labels:
                for _, val in mol["vdW"].items():
                    if val.id not in lj_params:
                        # Extract sigma and epsilon in consistent units
                        sigma = (
                            val.sigma.to_openmm().in_units_of(_unit.nanometers)._value
                        )
                        epsilon = (
                            val.epsilon.to_openmm()
                            .in_units_of(_unit.kilojoule_per_mole)
                            ._value
                        )

                        lj_params[val.id] = {
                            "sigma": sigma,
                            "epsilon": epsilon,
                        }
                        atom_type_to_index[val.id] = (
                            len(atom_type_to_index) + 1
                        )  # 1-indexed
                        sigma_init.append(sigma)
                        epsilon_init.append(epsilon)

                    atom_type_ids_topology.append(atom_type_to_index[val.id])

            atom_type_ids.append(
                _torch.tensor(atom_type_ids_topology, dtype=_torch.int64)
            )

        # Pad atom type IDs to the same length
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

    def forward(self, xyz, solute_mask, solvent_mask):
        """
        Calculate the Lennard-Jones potential energy for a set of positions.

        Parameters
        ----------
        xyz : torch.Tensor(BATCH, NATOMS, 3)
            The positions of the particles.
        solute_mask : torch.Tensor(BATCH, NATOMS)
            The mask for the solute atoms.
        solvent_mask : torch.Tensor(BATCH, NATOMS)
            The mask for the solvent atoms.

        Returns
        -------
        torch.Tensor
            The total Lennard-Jones potential energy for each batch.
        """
        # Extract trainable and frozen components
        trainable_sigma = self._sigma_embedding(self._atom_type_ids).squeeze(-1)
        frozen_sigma = self._sigma_embedding_frozen(self._atom_type_ids).squeeze(-1)

        trainable_epsilon = self._epsilon_embedding(self._atom_type_ids).squeeze(-1)
        frozen_epsilon = self._epsilon_embedding_frozen(self._atom_type_ids).squeeze(-1)

        # Combine using masks
        sigma = trainable_sigma * self._sigma_mask + frozen_sigma * ~self._sigma_mask
        epsilon = (
            trainable_epsilon * self._epsilon_mask
            + frozen_epsilon * ~self._epsilon_mask
        )

        # Apply masks
        solute_sigma = sigma * solute_mask
        solvent_sigma = sigma * solvent_mask
        solute_epsilon = _torch.abs(epsilon * solute_mask) + 1e-16
        solvent_epsilon = _torch.abs(epsilon * solvent_mask) + 1e-16

        xyz_qm = xyz * solute_mask.unsqueeze(-1)
        xyz_mm = xyz * solvent_mask.unsqueeze(-1)

        # Compute pairwise distances
        distances = _torch.cdist(xyz_mm, xyz_qm)
        distances = _torch.where(distances > 0, distances, 1e16)

        # Reshape parameters for broadcasting
        sigma_ij = 0.5 * (solvent_sigma[:, :, None] + solute_sigma[:, None, :])
        epsilon_ij = _torch.sqrt(
            solvent_epsilon[:, :, None] * solute_epsilon[:, None, :]
        )

        # Lennard-Jones potential
        inv_r = sigma_ij / distances
        inv_r6 = inv_r**6
        inv_r12 = inv_r6 * inv_r6
        energy_matrix = (
            4
            * epsilon_ij
            * (inv_r12 - inv_r6)
            * solvent_mask[:, :, None]
            * solute_mask[:, None, :]
        )

        # Sum over all pairwise interactions
        total_energy = _torch.sum(energy_matrix, dim=(1, 2))

        self.update_lj_parameters()

        return total_energy
