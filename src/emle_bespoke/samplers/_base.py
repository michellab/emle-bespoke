"""Base class for samplers."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class Sampler(dict, ABC):
    """
    Base class for samplers.

    This class inherits from dict to provide a simple key-value store for sampled data.
    Samplers should inherit from this class and implement the `sample` method.

    The `sample` method should populate the dictionary with the following keys:
    - `pos_qm`: _np.ndarray(NATOMS, 3)
        Atomic positions in the QM region in Angstrom.
    - `symbols_qm`: list[str]
        Atomic symbols in the QM region.
    - `pos_mm`: _np.ndarray(NATOMS, 3)
        Atomic positions in the MM region in Angstrom.
    - `charges_mm`: _np.ndarray(NATOMS), optional
        Point charges in the MM region.
    - `charges_qm`: _np.ndarray(NATOMS), optional
        Point charges in the QM region.
    - `symbols_mm`: list[str], optional
        Atomic symbols in the MM region.

    Parameters
    ----------
    energy_scale : float, optional
        Energy scale to convert from the QM calculator energy units to kJ/mol.
    length_scale : float, optional
        Length scale to convert from the QM calculator length units to Angstrom.

    Attributes
    ----------
    _energy_scale : float
        Energy scale to convert from the QM calculator energy units to kJ/mol.
    _length_scale : float
        Length scale to convert from the QM calculator length units to Angstrom.
    """

    def __init__(self, energy_scale: float = 1.0, length_scale: float = 1.0):
        """Initialize the sampler."""
        super().__init__()  # Initialize the dict
        self._energy_scale = energy_scale
        self._length_scale = length_scale

    @abstractmethod
    def sample(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Sample data and store it in the dictionary.

        Returns
        -------
        Dict[str, Any]
            The sampled data. The sampler itself is also updated with this data.
        """
        raise NotImplementedError(
            "This method must be implemented in the derived sampler class."
        )

    def update_data(self, data: Dict[str, Any]) -> None:
        """
        Update the sampler's data with new values.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing the data to update.
        """
        self.update(data)

    def clear_data(self) -> None:
        """Clear all sampled data."""
        self.clear()

    def get_data(self) -> Dict[str, Any]:
        """
        Get all sampled data.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all sampled data.
        """
        return dict(self)
