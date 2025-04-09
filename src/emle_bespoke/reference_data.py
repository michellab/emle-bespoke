"""Module defining the ReferenceData class, the basic data structure for reference data in the emle-bespoke package."""

import pickle
from typing import Dict

from loguru import logger as _logger


class ReferenceData(dict):
    def __init__(self, data: Dict = None):
        """
        Initialize the ReferenceData object with optional initial data.

        Parameters
        ----------
        data: dict, optional
            Initial reference data to load.
        """
        super().__init__(data if data else {})

    def append(self, data: Dict) -> None:
        """
        Append data to the reference data.
        """
        for key, value in data.items():
            self.setdefault(key, []).append(value)

    def write(self, filename: str = "ref_data.pkl") -> None:
        """
        Write the reference data to a file in pickle format.

        Parameters
        ----------
        filename: str
            Filename to save the reference data.
        """
        _logger.info(f"Writing reference data to {filename}.")
        if not filename.endswith(".pkl"):
            filename = filename + ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(dict(self), f)

    def read(self, filename: str = "ref_data.pkl", overwrite: bool = True) -> Dict:
        """
        Read the reference data from a pickle file.

        Parameters
        ----------
        filename: str
            The filename from which to read the data.
        overwrite: bool
            Whether to overwrite the existing reference data with the file contents.

        Returns
        -------
        dict
            The loaded reference data.
        """
        _logger.debug(
            f"Reading reference data from {filename} with overwrite={overwrite}."
        )

        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)

            if overwrite or not self:
                self.clear()
                self.update(data)
            else:
                self._extend_data(data)

            return dict(self)

        except FileNotFoundError:
            _logger.warning(f"File {filename} not found. Returning existing data.")
            return dict(self)
        except Exception as e:
            _logger.error(f"Error reading data from {filename}: {e}")
            raise

    def _extend_data(self, new_data: Dict) -> None:
        """
        Extend the current reference data with new data.

        Parameters
        ----------
        new_data: dict
            The data to merge with the existing reference data.
        """
        for key, value in new_data.items():
            self.setdefault(key, []).extend(value)

    def get_data(self) -> Dict:
        """
        Get the current reference data.

        Returns
        -------
        dict
            The current reference data.
        """
        return dict(self)

    def clear_data(self) -> None:
        """
        Clear the reference data.
        """
        self.clear()

    def add_data_to_key(self, key: str, data: Dict) -> Dict:
        """
        Add data to a specific key in the reference data.

        Parameters
        ----------
        key: str
            The key to which to add the data.
        data: dict
            The data to add to the key.

        Returns
        -------
        dict
            The reference data with the added data.
        """
        self.setdefault(key, []).append(data)
        return dict(self)

    def add(self, data: Dict) -> Dict:
        """
        Add data to the reference data.

        Parameters
        ----------
        data: dict
            The data to add to the reference data.

        Returns
        -------
        dict
            The reference data with the added data.
        """
        for key, value in data.items():
            self.setdefault(key, []).extend(
                [value] if not isinstance(value, list) else value
            )
        return dict(self)

    def to_tensors(self) -> Dict:
        """
        Get the reference data in tensor format.

        This method converts the reference data dictionary into tensors by padding
        sequences to the same length. Each value in the dictionary is padded to match
        the length of the longest sequence.

        Returns
        -------
        Dict
            Dictionary containing the reference data converted to padded tensors.
            Keys are the same as the original data, values are padded tensors.
        """
        import torch as _torch
        from emle.train._utils import pad_to_max

        data = {}
        for key, value in self.items():
            if not value:  # Skip empty lists
                continue
            try:
                if isinstance(value[0], (int, float)):
                    data[key] = _torch.tensor(value)
                else:
                    data[key] = pad_to_max(value, max_length=None)
            except Exception as e:
                raise ValueError(f"Failed to convert {key} to tensor format: {str(e)}")

        return data
