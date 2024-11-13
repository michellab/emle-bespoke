import os as _os
from typing import Union

import h5py as _h5py
import numpy as _np
import torch as _torch
from loguru import logger as _logger

from ._base import BaseCalculator


class HortonCalculator(BaseCalculator):
    """
    Horton calculator.

    Parameters
    ----------
    name_prefix : str
        The prefix for the input and output files.

    Attributes
    ----------
    _NAME_PREFIX : str
        The default prefix for the input and output files.
    """

    _NAME_PREFIX = "horton"
    _HORTON_WPART_BIN = "horton-wpart.py"
    _HORTON_KEYS = (
        "cartesian_multipoles",
        "core_charges",
        "valence_charges",
        "valence_widths",
    )

    def __init__(self, name_prefix: Union[str, None] = None):
        self._name_prefix = name_prefix or self._NAME_PREFIX

        _logger.debug("Initialized HortonCalculator")
        _logger.debug(f"Name prefix: {self._name_prefix}")

    def get_horton_partitioning(
        self,
        input_file,
        directory=".",
        output_file=None,
        scheme="mbis",
        lmax=3,
        output_log=None,
    ):
        """
        Run the horton-wpart.py script and read the output file.

        Parameters
        ----------
        input_file : str
            The input file.
        output_file : str, optional
            The output file in HDF5 format.
        scheme : str, optional
            The scheme to use. Default is "mbis".
        lmax : int, optional
            The maximum angular momentum. Default is 3.
        output_log : str, optional
            The output log file.

        Returns
        -------
        dict
            The data from the output file.
        """
        output_file = self._run_horton_wpart(
            input_file=input_file,
            directory=directory,
            output_file=output_file,
            scheme=scheme,
            lmax=lmax,
            output_log=output_log,
        )

        return self._read_horton_output(_os.path.join(directory, output_file))

    def _read_horton_output(self, output_file):
        """
        Read the .h5 output file from horton-wpart.py.

        Parameters
        ----------
        output_file : str
            The output file in HDF5 format.

        Returns
        -------
        dict
            The data from the output file.
        """
        with _h5py.File(output_file, "r") as f:
            data = {key: f[key][:] for key in self._HORTON_KEYS}
            q = data["core_charges"] + data["valence_charges"]
            q_shift = _np.sum(_np.round(q) - q) / len(q)

        return {
            "s": _torch.tensor(data["valence_widths"]),
            "q_core": _torch.tensor(data["core_charges"]),
            "q_val": _torch.tensor(data["valence_charges"] + q_shift),
            "mu": _torch.tensor(data["cartesian_multipoles"][:, 1:4]),
        }

    def _run_horton_wpart(
        self,
        input_file,
        directory=".",
        output_file=None,
        scheme="mbis",
        lmax=3,
        output_log=None,
    ):
        """
        Run the horton-wpart.py script.

        Parameters
        ----------
        input_file : str
            The input file.
        output_file : str, optional
            The output file in HDF5 format.
        scheme : str, optional
            The scheme to use. Default is "mbis".
        lmax : int, optional
            The maximum angular momentum. Default is 3.
        output_log : str, optional
            The output log file.

        Returns
        -------
        str
            The output file name.
        """
        output_log = output_log or f"{self._name_prefix}.log"
        output_file = output_file or f"{self._name_prefix}.h5"

        self._run_process(
            bin_exec=self._HORTON_WPART_BIN,
            arguments=[input_file, output_file, scheme, "--lmax", str(lmax)],
            output_file_name=output_log,
            directory=directory,
        )
        return output_file
