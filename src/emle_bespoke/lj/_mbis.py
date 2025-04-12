"""Implementation of AI-based Lennard-Jones parameter derivation."""

from typing import Dict, Optional, Tuple, Union

import torch as _torch

from .._constants import ANGSTROM_TO_BOHR as _ANGSTROM_TO_BOHR
from .._constants import HARTREE_TO_KJ_MOL as _HARTREE_TO_KJ_MOL


class AILennardJones:
    """
    Class for deriving Lennard-Jones parameters from ab initio data using MBIS partitioning.

    This class implements methods to compute isotropic polarizabilities and derive
    Lennard-Jones parameters from ab initio data using the MBIS (Minimal Basis Iterative Stockholder)
    partitioning scheme.

    Attributes
    ----------
    _FREE_ATOM_MULTIPLICITY : Dict[int, int]
        Multiplicity of free atoms for common elements
    C6_COEFFICIENTS : Dict[int, float]
        C6 dispersion coefficients for common elements in atomic units (Ha.Bohr^6)
    RCUBED_FREE : Dict[int, float]
        Free atom volumes in Bohr^3 calculated at B3LYP/cc-pVTZ
    """

    # Multiplicity for free atoms
    _FREE_ATOM_MULTIPLICITY: Dict[int, int] = {
        1: 2,  # H
        3: 2,  # Li
        6: 3,  # C
        7: 4,  # N
        8: 3,  # O
        9: 2,  # F
        10: 0,  # Ne
        14: 3,  # Si
        15: 4,  # P
        16: 3,  # S
        17: 2,  # Cl
        18: 0,  # Ar
        35: 2,  # Br
        36: 0,  # Kr
        53: 2,  # I
    }

    # Values in atomic units, taken from:
    # Chu, X., & Dalgarno, A. (2004).
    # Linear response time-dependent density functional theory for van der Waals coefficients.
    # The Journal of Chemical Physics, 121(9), 4083--4088. http://doi.org/10.1063/1.1779576
    # Bfree: Ha.Bohr**6 (https://pubs.acs.org/doi/10.1021/acs.jctc.6b00027) #TODO: convert to Angstrom^6?
    C6_COEFFICIENTS: Dict[int, float] = {
        0: 0.0,  # Dummy atom
        1: 6.5,  # H
        3: 1393.0,  # Li
        6: 46.6,  # C
        7: 24.2,  # N
        8: 15.6,  # O
        9: 9.5,  # F
        10: 6.38,  # Ne
        14: 305.0,  # Si
        15: 185.0,  # P
        16: 134.0,  # S
        17: 94.6,  # Cl
        18: 64.3,  # Ar
        35: 162.0,  # Br
        36: 130.0,  # Kr
        53: 385.0,  # I
    }

    # Calculated free atom volumes in Angstrom^3 at B3LYP/cc-pVTZ #TODO: convert to Bohr^3?
    RCUBED_FREE: Dict[int, float] = {
        0: 0.0,  # Dummy atom
        1: 1.172520801342987,  # H
        6: 5.044203900156338,  # C
        7: 3.7853134153705934,  # N
        8: 3.135943377417264,  # O
        16: 11.015796687279208,  # S
    }

    def __init__(self) -> None:
        """Initialize the AILennardJones class."""
        pass

    @staticmethod
    def _compute_traces(A_thole: _torch.Tensor) -> _torch.Tensor:
        """
        Compute the trace of the inverse of each 3x3 block in each polarizability tensor.

        Parameters
        ----------
        A_thole : torch.Tensor(NMOL, 3N, 3N)
            Block matrix containing 3x3 polarizability tensors for each molecule.

        Returns
        -------
        torch.Tensor(NMOL, N)
            Trace of the inverse of each 3x3 block per molecule.
        """
        if A_thole.dim() not in [2, 3]:
            raise RuntimeError("Input tensor must be 2D or 3D")

        n_mol, dim, _ = A_thole.shape if A_thole.dim() == 3 else (1, *A_thole.shape)
        n_atoms = dim // 3
        traces = []

        for mol_idx in range(n_mol):
            mol_traces = _torch.stack(
                [
                    _torch.trace(
                        _torch.inverse(
                            A_thole[mol_idx, 3 * i : 3 * i + 3, 3 * i : 3 * i + 3]
                        )
                    )
                    for i in range(n_atoms)
                ]
            )
            traces.append(mol_traces)

        return _torch.stack(traces, dim=0) / 3.0

    def compute_isotropic_polarizabilities(
        self, A_thole: _torch.Tensor
    ) -> _torch.Tensor:
        """
        Compute isotropic polarizabilities from one or a batch of block polarizability matrices.

        Parameters
        ----------
        A_thole : torch.Tensor(3N, 3N) or torch.Tensor(NMOL, 3N, 3N)
            Full polarizability tensor in block form.

        Returns
        -------
        torch.Tensor(N) or torch.Tensor(NMOL, N)
            Isotropic polarizabilities per atom.
        """
        if A_thole.dim() not in [2, 3]:
            raise RuntimeError("Input tensor must be 2D or 3D")

        if A_thole.dim() == 2:
            A_thole = A_thole.unsqueeze(0)

        return self._compute_traces(A_thole)

    def get_lj_parameters(
        self, atomic_numbers: _torch.Tensor, rcubed: _torch.Tensor, alpha: _torch.Tensor
    ) -> Tuple[_torch.Tensor, _torch.Tensor]:
        """
        Compute Lennard-Jones sigma and epsilon parameters.

        Parameters
        ----------
        atomic_numbers : torch.Tensor(N) or torch.Tensor(NMOL, N)
            Atomic numbers of atoms in the molecule.
        rcubed : torch.Tensor(N) or torch.Tensor(NMOL, N)
            Cube of the vdW radii of atoms in the molecule in Angstrom^3.
        alpha : torch.Tensor(N) or torch.Tensor(NMOL, N)
            Isotropic polarizabilities per atom in Angstrom^3.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing:
            - sigma: LJ sigma values in Angstrom
            - epsilon: LJ epsilon values in kJ/mol

        Raises
        ------
        ValueError
            If any atomic number is missing from the parameter tables.
        RuntimeError
            If input tensor dimensions are invalid.
        """
        # Ensure tensors are batched
        if atomic_numbers.dim() == 1:
            atomic_numbers = atomic_numbers.unsqueeze(0)
        if rcubed.dim() == 1:
            rcubed = rcubed.unsqueeze(0)
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(0)

        # Mask out dummy atoms
        mask = atomic_numbers > 0
        alpha = alpha * mask
        rcubed = rcubed * mask

        # Create lookup tensors for free atom volumes
        try:
            vol_isolated = _torch.zeros(
                max(self.RCUBED_FREE.keys()) + 1, dtype=_torch.float
            )
            for Z, vol in self.RCUBED_FREE.items():
                vol_isolated[Z] = vol
            vol_isolated = vol_isolated[atomic_numbers]
        except KeyError as e:
            raise ValueError(f"Missing RCUBED_FREE entry for atomic number {e.args[0]}")

        # Volume scaling factor
        scaling = rcubed / vol_isolated

        # Create lookup tensor for C6 coefficients
        try:
            c6 = _torch.zeros(max(self.C6_COEFFICIENTS.keys()) + 1, dtype=_torch.float)
            for Z, c6_coeff in self.C6_COEFFICIENTS.items():
                c6[Z] = c6_coeff
            c6 = c6[atomic_numbers]  # TODO: convert to Angstrom^6?
            c6 = c6
        except KeyError as e:
            raise ValueError(
                f"Missing C6_COEFFICIENTS entry for atomic number {e.args[0]}"
            )

        # Scale C6 coefficients
        c6_scaled = c6 * scaling**2

        # Compute vdW radius from polarizability (Fedorov-Tkatchenko relation)
        radius = 2.54 * alpha ** (1.0 / 7.0)
        rmin = 2 * radius

        # Compute Lennard-Jones parameters
        sigma = rmin / (2 ** (1.0 / 6.0)) / _ANGSTROM_TO_BOHR
        epsilon = c6_scaled / (2 * rmin**6.0) * _HARTREE_TO_KJ_MOL

        return sigma, epsilon
