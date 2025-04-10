"""Module defining the ReferenceDataset class, the basic data structure for reference data in the emle-bespoke package."""

import os as _os
import pickle as _pkl
from typing import Dict, Optional

import h5py as _h5py
import numpy as _np
import torch as _torch
from loguru import logger as _logger
from torch.utils.data import Dataset as _Dataset


class ReferenceDataset(_Dataset):
    def __init__(
        self,
        data: Optional[Dict] = None,
        device: Optional[_torch.device] = None,
        dtype: _torch.dtype = _torch.float64,
        load_file: Optional[str] = None,
    ):
        """
        Initialize the ReferenceDataset object with optional initial data.

        Parameters
        ----------
        data: dict, optional
            Initial reference data to load.
        device: torch.device, optional
            Device to place tensors on. Default is "cuda" if available, else "cpu"
        dtype: torch.dtype, optional
            Data type for floating point tensors. Default is torch.float64
        load_file: str, optional
            Path to file to load data from. Supports both .h5 and .pkl formats.
            If provided, data parameter is ignored.
        """
        super().__init__()

        if device is None:
            device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

        self.device = device
        self.dtype = dtype
        self._data = data if data else {}
        self._tensors = None

        if load_file is not None:
            self.load(load_file)

    def rename_key(self, old_key: str, new_key: str) -> None:
        """
        Rename a key in the dataset.

        Parameters
        ----------
        old_key: str
            The current key name to rename.
        new_key: str
            The new key name.

        Raises
        ------
        KeyError
            If old_key does not exist in the dataset.
        ValueError
            If new_key already exists in the dataset.
        """
        if old_key not in self._data:
            raise KeyError(f"Key '{old_key}' not found in dataset")
        if new_key in self._data:
            raise ValueError(f"Key '{new_key}' already exists in dataset")

        # Rename in data dictionary
        self._data[new_key] = self._data.pop(old_key)

        # Rename in tensors if they exist
        if self._tensors is not None and old_key in self._tensors:
            self._tensors[new_key] = self._tensors.pop(old_key)

        _logger.info(f"Renamed key '{old_key}' to '{new_key}'")

    def load(self, filename: str, overwrite: bool = True) -> Dict:
        """
        Load data from a file. Supports both HDF5 (.h5) and pickle (.pkl) formats.

        Parameters
        ----------
        filename: str
            Path to the file to load. Must have .h5 or .pkl extension.
        overwrite: bool, optional
            Whether to overwrite existing data. Default is True.

        Returns
        -------
        dict
            The loaded reference data.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        """
        if not _os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found")

        ext = _os.path.splitext(filename)[1].lower()

        if ext == ".h5":
            return self._load_from_h5(filename, overwrite)
        elif ext == ".pkl":
            return self._load_from_pickle(filename, overwrite)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Must be .h5 or .pkl")

    def _load_from_h5(self, h5_file: str, overwrite: bool = True) -> Dict:
        """Load data from HDF5 file."""
        try:
            with _h5py.File(h5_file, "r") as f:
                if overwrite or not self._data:
                    self._data.clear()
                for key in f.keys():
                    data = f[key][:]
                    self._data[key] = data
                    _logger.info(f"Loaded tensor '{key}' with shape {data.shape}")
            _logger.info(f"Loaded {len(self._data)} tensors from HDF5 file: {h5_file}")
            self._tensors = None  # Reset tensors to force reconversion
            return dict(self._data)
        except Exception as e:
            _logger.error(f"Error loading HDF5 file {h5_file}: {e}")
            raise

    def _load_from_pickle(self, pickle_file: str, overwrite: bool = True) -> Dict:
        """Load data from pickle file."""
        try:
            with open(pickle_file, "rb") as f:
                data = _pkl.load(f)

            if overwrite or not self._data:
                self._data.clear()
                self._data.update(data)
            else:
                self._extend_data(data)

            # Log shapes of loaded data
            for key, value in self._data.items():
                if isinstance(value, (list, _np.ndarray)):
                    shape = _np.array(value).shape
                    _logger.info(f"Loaded tensor '{key}' with shape {shape}")
                elif isinstance(value, _torch.Tensor):
                    _logger.info(f"Loaded tensor '{key}' with shape {value.shape}")

            self._tensors = None  # Reset tensors to force reconversion
            _logger.info(
                f"Loaded {len(self._data)} tensors from pickle file: {pickle_file}"
            )
            return dict(self._data)
        except Exception as e:
            _logger.error(f"Error loading pickle file {pickle_file}: {e}")
            raise

    def write(self, filename: str = "ref_data.h5") -> None:
        """
        Write the reference data to a file. Supports both HDF5 (.h5) and pickle (.pkl) formats.

        Parameters
        ----------
        filename: str
            Filename to save the reference data. Extension determines format (.h5 or .pkl).
        """
        ext = _os.path.splitext(filename)[1].lower()

        if ext == ".h5":
            self.write_h5(filename)
        elif ext == ".pkl":
            self.write_pickle(filename)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Must be .h5 or .pkl")

    def write_h5(self, filename: str = "ref_data.h5") -> None:
        """
        Write the reference data to an HDF5 file.
        Ensures all data is converted to tensors and padded before writing.

        Parameters
        ----------
        filename: str
            Filename to save the reference data.
        """
        _logger.info(f"Writing reference data to HDF5 file: {filename}")
        if not filename.endswith(".h5"):
            filename = filename + ".h5"

        try:
            # Ensure all data is converted to tensors and padded
            if self._tensors is None:
                self._convert_to_tensors()

            with _h5py.File(filename, "w") as f:
                for key, tensor in self._tensors.items():
                    # Convert tensor to numpy array for HDF5 storage
                    numpy_array = tensor.cpu().numpy()
                    f.create_dataset(key, data=numpy_array)
                    _logger.info(f"Wrote array '{key}' with shape {numpy_array.shape}")

            _logger.info(
                f"Successfully wrote {len(self._tensors)} tensors to HDF5 file '{filename}'"
            )
        except Exception as e:
            _logger.error(f"Error writing to HDF5 file {filename}: {e}")
            raise

    def write_pickle(self, filename: str = "ref_data.pkl") -> None:
        """
        Write the reference data to a pickle file.

        Parameters
        ----------
        filename: str
            Filename to save the reference data.
        """
        _logger.info(f"Writing reference data to pickle file: {filename}")
        if not filename.endswith(".pkl"):
            filename = filename + ".pkl"

        try:
            # Ensure all data is converted to tensors and padded
            if self._tensors is None:
                self._convert_to_tensors()

            # Log shapes before writing
            for key, tensor in self._tensors.items():
                _logger.info(f"Writing tensor '{key}' with shape {tensor.shape}")

            with open(filename, "wb") as f:
                _pkl.dump(self._tensors, f)

            _logger.info(
                f"Successfully wrote {len(self._tensors)} tensors to pickle file"
            )
        except Exception as e:
            _logger.error(f"Error writing to pickle file {filename}: {e}")
            raise

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if not self._data:
            return 0
        # Get the length of the first non-empty list
        for value in self._data.values():
            if value:
                return len(value)
        return 0

    def __getitem__(self, idx: int) -> Dict[str, _torch.Tensor]:
        """
        Get a sample from the dataset.

        Parameters
        ----------
        idx: int
            Index of the sample to get

        Returns
        -------
        dict
            Dictionary of tensors for the sample, including indices
        """
        if self._tensors is None:
            self._convert_to_tensors()

        # Get the data for this index
        data = {key: tensor[idx] for key, tensor in self._tensors.items()}
        data["indices"] = _torch.tensor(
            idx, device=self.device, dtype=_torch.long, requires_grad=False
        )

        return data

    def _convert_to_tensors(self) -> None:
        """Convert the data to tensors and store them."""
        from emle.train._utils import pad_to_max

        self._tensors = {}
        for key, value in self._data.items():
            if not value:  # Skip empty lists
                continue
            try:
                if isinstance(value[0], (int, float)):
                    tensor = _torch.tensor(value, device=self.device)
                    if tensor.is_floating_point():
                        tensor = tensor.to(self.dtype)
                else:
                    tensor = pad_to_max(value)
                    if tensor.is_floating_point():
                        tensor = tensor.to(device=self.device, dtype=self.dtype)
                    else:
                        tensor = tensor.to(device=self.device)
                self._tensors[key] = tensor
            except Exception as e:
                raise ValueError(f"Failed to convert {key} to tensor format: {str(e)}")

    def append(self, data: Dict) -> None:
        """
        Append data to the reference data.
        """
        for key, value in data.items():
            self._data.setdefault(key, []).append(value)
        self._tensors = None  # Reset tensors to force reconversion

    def _extend_data(self, new_data: Dict) -> None:
        """
        Extend the current reference data with new data.

        Parameters
        ----------
        new_data: dict
            The data to merge with the existing reference data.
        """
        for key, value in new_data.items():
            self._data.setdefault(key, []).extend(value)
        self._tensors = None  # Reset tensors to force reconversion

    def get_data(self) -> Dict:
        """
        Get the current reference data.

        Returns
        -------
        dict
            The current reference data.
        """
        return dict(self._data)

    def clear_data(self) -> None:
        """
        Clear the reference data.
        """
        self._data.clear()
        self._tensors = None

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
        self._data.setdefault(key, []).append(data)
        self._tensors = None  # Reset tensors to force reconversion
        return dict(self._data)

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
            self._data.setdefault(key, []).extend(
                [value] if not isinstance(value, list) else value
            )
        self._tensors = None  # Reset tensors to force reconversion
        return dict(self._data)

    def to_tensors(self) -> Dict[str, _torch.Tensor]:
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
        if self._tensors is None:
            self._convert_to_tensors()
        return self._tensors
