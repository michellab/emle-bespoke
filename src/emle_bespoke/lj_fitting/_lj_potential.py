"""Lennard-Jones potential class."""
import openmm.unit as _unit
import torch as _torch


class LennardJonesPotential(_torch.nn.Module):
    """
    Lennard-Jones potential energy calculation for a set of particles.

    Parameters
    ----------
    topology_off : openff.toolkit.topology.Topology
        The OpenFF Topology object.
    forcefield : openff.toolkit.typing.engines.smirnoff.ForceField
        The OpenFF ForceField object.
    parameters_to_fit : dict
        A dictionary of the parameters to fit.
        The keys are the atom types and the values are lists of the parameters to fit.
        For example, to fit the sigma and epsilon parameters for the "n-tip3p-O" atom type:
        {
            "n-tip3p-O": ["sigma", "epsilon"],
        }
    device : torch.device, optional
        The device to use for the calculation.
    dtype : torch.dtype, optional
        The data type to use for the calculation.

    Attributes
    ----------
    _forcefield : openff.toolkit.typing.engines.smirnoff.ForceField
        The OpenFF ForceField object.
    _topology_off : openff.toolkit.topology.Topology
        The OpenFF Topology object.
    _parameters_to_fit : dict
        A dictionary of the parameters to fit.
    _lj_params : dict
        A dictionary of the Lennard-Jones parameters.
        The keys are the atom types and the values are dictionaries with the "sigma" and "epsilon" parameters.
    _atoms_types : list
        A list of the atom types.
    _sigma : list
        A list of the sigma parameters.
    _epsilon : list
        A list of the epsilon parameters.
    """

    def __init__(
        self, topology_off, forcefield, parameters_to_fit, device=None, dtype=None
    ):
        super().__init__()
        self._forcefield = forcefield
        self._topology_off = topology_off
        self._parameters_to_fit = parameters_to_fit

        # Initialize the Lennard-Jones parameters
        self._lj_params = {}
        self._atoms_types = []

        # Get the Lennard-Jones parameters and dynamically register them
        self._get_lennard_jones_parameters()

        # Initialize device and dtype
        self._device = (
            device or _torch.device("cuda")
            if _torch.cuda.is_available()
            else _torch.device("cpu")
        )
        self._dtype = dtype or _torch.float64

        # Create sigma and epsilon tensors
        self._sigma_tensor = _torch.stack(
            [self._lj_params[atom]["sigma"] for atom in self._atoms_types]
        ).to(self._device, self._dtype)

        self._epsilon_tensor = _torch.stack(
            [self._lj_params[atom]["epsilon"] for atom in self._atoms_types]
        ).to(self._device, self._dtype)

    def _get_lennard_jones_parameters(self):
        ff_params = self._forcefield.label_molecules(self._topology_off)

        # Use nn.ParameterDict for dynamic registration
        self._sigma = _torch.nn.ParameterDict()
        self._epsilon = _torch.nn.ParameterDict()

        for mol in ff_params:
            for _, val in mol["vdW"].items():
                if val.id not in self._lj_params:
                    # Create tensors for sigma and epsilon
                    sigma = _torch.tensor(
                        val.sigma.to_openmm().in_units_of(_unit.nanometers)._value,
                        dtype=_torch.float64,
                    )
                    epsilon = _torch.tensor(
                        val.epsilon.to_openmm()
                        .in_units_of(_unit.kilojoule_per_mole)
                        ._value,
                        dtype=_torch.float64,
                    )

                    # Dynamically register trainable parameters
                    if val.id in self._parameters_to_fit:
                        if "sigma" in self._parameters_to_fit[val.id]:
                            sigma = _torch.nn.Parameter(sigma)
                            self._sigma[val.id] = sigma
                        if "epsilon" in self._parameters_to_fit[val.id]:
                            epsilon = _torch.nn.Parameter(epsilon)
                            self._epsilon[val.id] = epsilon

                    # Store in _lj_params dictionary for reference
                    self._lj_params[val.id] = {"sigma": sigma, "epsilon": epsilon}

                self._atoms_types.append(val.id)

        return self._lj_params, self._atoms_types

    @staticmethod
    def _calculate_lennard_jones_energy(
        position1, position2, sigma1, sigma2, epsilon1, epsilon2
    ):
        """
        Calculate the Lennard-Jones potential energy between two particles.

        Parameters
        ----------
        position1 : torch.Tensor
            The position of the first particle.
        position2 : torch.Tensor
            The position of the second particle.
        sigma1 : float
            The sigma parameter of the first particle.
        sigma2 : float
            The sigma parameter of the second particle.
        epsilon1 : float
            The epsilon parameter of the first particle.
        epsilon2 : float
            The epsilon parameter of the second particle.

        Returns
        -------
        torch.Tensor
            The Lennard-Jones potential energy between the two particles.
        """
        r = _torch.norm(position1 - position2)

        # Lorentz-Berthelot mixing rules
        sigma = 0.5 * (sigma1 + sigma2)
        epsilon = _torch.sqrt(epsilon1 * epsilon2)

        return (
            4 * epsilon * ((_torch.div(sigma, r)) ** 12 - (_torch.div(sigma, r)) ** 6)
        )

    def forward(self, xyz, solute_mask, solvent_mask):
        """
        Calculate the Lennard-Jones potential energy for a set of positions.

        Parameters
        ----------
        xyz: torch.Tensor
            The positions of the particles.
        solute_mask : torch.Tensor(N_MM_ATOMS)
            The mask for the solute atoms.
        solvent_mask : torch.Tensor(N_MM_ATOMS)
            The mask for the solvent atoms.

        Returns
        -------
        torch.Tensor
            The total Lennard-Jones potential energy.
        """
        # Get the sigma and epsilon parameters
        xyz_mm = xyz[solvent_mask]
        xyz_qm = xyz[solute_mask]
        solute_sigma = self._sigma_tensor[solute_mask]
        solvent_sigma = self._sigma_tensor[solvent_mask]
        solute_epsilon = self._epsilon_tensor[solute_mask]
        solvent_epsilon = self._epsilon_tensor[solvent_mask]

        # Calculate pairwise distances
        distances = _torch.cdist(xyz_mm, xyz_qm)

        # Lorentz-Berthelot mixing rules
        sigma = 0.5 * (solvent_sigma[:, None] + solute_sigma[None, :])
        epsilon = _torch.sqrt(solvent_epsilon[:, None] * solute_epsilon[None, :])

        # Lennard-Jones potential
        r6 = (sigma / distances) ** 6
        r12 = r6**2
        energy_matrix = 4 * epsilon * (r12 - r6)

        # Sum over all pairwise interactions
        total_energy = energy_matrix.sum()

        return total_energy
