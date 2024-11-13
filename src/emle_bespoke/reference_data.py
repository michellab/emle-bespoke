import logging
import pickle
from typing import Any, Dict

_logger = logging.getLogger(__name__)


class ReferenceData:
    def __init__(self, data: Dict = None):
        """
        Initialize the ReferenceData object with optional initial data.

        Parameters
        ----------
        data: dict, optional
            Initial reference data to load.
        """
        self._reference_data = data if data else {}

    def __getitem__(self, key: str) -> Any:
        """
        Allows subscript access to the reference data.

        Parameters
        ----------
        key : str
            The key to retrieve from the reference data.

        Returns
        -------
        Any
            The value associated with the given key.

        Raises
        ------
        KeyError
            If the key is not found in the reference data.
        """
        return self._reference_data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Allows setting values in the reference data via subscript.

        Parameters
        ----------
        key : str
            The key to set in the reference data.
        value : Any
            The value to associate with the key.
        """
        self._reference_data[key] = value

    def __delitem__(self, key: str) -> None:
        """
        Allows deleting items from the reference data via subscript.

        Parameters
        ----------
        key : str
            The key to delete from the reference data.
        """
        del self._reference_data[key]

    def __contains__(self, key: str) -> bool:
        """
        Allows checking if a key exists in the reference data using 'in' keyword.

        Parameters
        ----------
        key : str
            The key to check for existence in the reference data.

        Returns
        -------
        bool
            True if the key is present in the reference data, otherwise False.
        """
        return key in self._reference_data

    def __repr__(self) -> str:
        """
        Provides a string representation of the ReferenceData object.

        Returns
        -------
        str
            A string representation of the ReferenceData object.
        """
        return f"ReferenceData({self._reference_data})"

    def write(self, filename: str = "ref_data.pkl") -> None:
        """
        Write the reference data to a file in pickle format.

        Parameters
        ----------
        filename: str
            Filename to save the reference data.
        """
        _logger.info(f"Writing reference data to {filename}.")
        with open(filename, "wb") as f:
            pickle.dump(self._reference_data, f)

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

            if overwrite or not self._reference_data:
                self._reference_data = data
            else:
                self._extend_data(data)

            return self._reference_data

        except FileNotFoundError:
            _logger.warning(f"File {filename} not found. Returning existing data.")
            return self._reference_data
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
            self._reference_data.setdefault(key, []).extend(value)

    def get_data(self) -> Dict:
        """
        Get the current reference data.

        Returns
        -------
        dict
            The current reference data.
        """
        return self._reference_data

    def clear_data(self) -> None:
        """
        Clear the reference data.
        """
        self._reference_data.clear()

    def add_data_to_key(self, key: str, data: Dict) -> None:
        """
        Add data to a specific key in the reference data.

        Parameters
        ----------
        key: str
            The key to which to add the data.
        data: dict
            The data to add to the key.
        """
        self._reference_data.setdefault(key, []).append(data)
