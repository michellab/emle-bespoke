import torch as _torch
import openmm as _mm
import openmm.unit as _unit

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
    def __init__(self, topology_off, forcefield, parameters_to_fit):
        self._forcefield = forcefield
        self._topology_off = topology_off
        self._parameters_to_fit = parameters_to_fit
        
        # Initialize the Lennard-Jones parameters
        self._lj_params = {}
        self._atoms_types = []

        # Get the Lennard-Jones parameters
        self._get_lennard_jones_parameters()

        # Create tensors for the Lennard-Jones parameters
        self._sigma = [self._lj_params[atom]["sigma"] for atom in self._atoms_types]
        self._epsilon = [self._lj_params[atom]["epsilon"] for atom in self._atoms_types]

    def _get_lennard_jones_parameters(self):
        ff_params = self._forcefield.label_molecules(self._topology_off)
        for mol in ff_params:
            for _, val in mol["vdW"].items():
                if val.id not in self._lj_params:
                    sigma = _torch.tensor(val.sigma.to_openmm().in_units_of(_unit.nanometers)._value, dtype=_torch.float64)
                    epsilon = _torch.tensor(val.epsilon.to_openmm().in_units_of(_unit.kilojoule_per_mole)._value, dtype=_torch.float64)

                    if val.id in self._parameters_to_fit:
                        if "sigma" in self._parameters_to_fit[val.id]:
                            sigma = _torch.nn.Parameter(sigma)
                        if "epsilon" in self._parameters_to_fit[val.id]:
                            epsilon = _torch.nn.Parameter(epsilon)

                    self._lj_params[val.id] = {
                        "sigma": sigma,
                        "epsilon": epsilon
                    }
                
                self._atoms_types.append(val.id)

        return self._lj_params, self._atoms_types
    
    @staticmethod
    def _calculate_lennard_jones_energy(position1, position2, sigma1, sigma2, epsilon1, epsilon2):
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

        return 4 * epsilon * ((_torch.div(sigma, r)) ** 12 - (_torch.div(sigma, r)) ** 6)
    
    def forward(self, positions, solute_indices, solvent_indices):
        """
        Calculate the Lennard-Jones potential energy for a set of positions.

        Parameters
        ----------
        positions : torch.Tensor
            The positions for which to calculate the energy.

        Returns
        -------
        torch.Tensor
            The Lennard-Jones potential energy.
        """
        energy = _torch.tensor(0.0, dtype=_torch.float64)

        for i in solvent_indices:
            for j in solute_indices:
                energy += self._calculate_lennard_jones_energy(
                    positions[i], 
                    positions[j],
                    self._sigma[i],
                    self._sigma[j],
                    self._epsilon[i],
                    self._epsilon[j]
                )

        return energy