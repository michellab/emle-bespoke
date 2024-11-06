import os as _os
from typing import Union

import torch as _torch

from .._constants import ANGSTROM_TO_BOHR
from ._base import BaseCalculator


class ORCACalculator(BaseCalculator):
    """
    ORCA calculator.

    Parameters
    ----------
    name_prefix : str
        The prefix for the input and output files.
    orca_home : str
        The path to the ORCA installation directory.

    Attributes
    ----------
    _ORCA_BIN : str
        The ORCA binary.
    _ORCA_VPOT_BIN : str
        The ORCA vpot binary.
    _NAME_PREFIX : str
        The default prefix for the input and output files.
    """

    _ORCA_BIN = "orca"
    _ORCA_VPOT_BIN = "orca_vpot"
    _ORCA_2MKL_BIN = "orca_2mkl"
    _NAME_PREFIX = "orca"

    def __init__(
        self, name_prefix: Union[str, None] = None, orca_home: Union[str, None] = None
    ):
        """
        Initialize the ORCA calculator.
        """
        self._name_prefix = name_prefix or self._NAME_PREFIX
        self._orca_home = orca_home or _os.environ.get("ORCA_HOME")
        if not self._orca_home:
            raise ValueError("ORCA_HOME is not set.")
        
    @property
    def name_prefix(self):
        return self._name_prefix
    
    @name_prefix.setter
    def name_prefix(self, value):
        self._name_prefix = value


    # -------------------------------------------------------------------------------------------- #
    #                                                                                              #
    #                                       I/O STATIC METHODS                                     #
    #                                                                                              #
    # -------------------------------------------------------------------------------------------- #
    @staticmethod
    def read_vpot(output_file_name: str, directory: str) -> _torch.Tensor:
        """
        Read the vpot from the ORCA output file.
        """
        vpot_values = []
        with open(_os.path.join(directory, output_file_name), "r") as f:
            lines = f.readlines()[1:]
            for line in lines:
                vpot_values.append(float(line.split()[3]))

        return _torch.tensor(vpot_values)
    
    @staticmethod
    def read_single_point_energy(output_file_name: str, directory: str) -> float:
        """
        Read the single point energy from the ORCA output file.
        """
        file_path = _os.path.join(directory, output_file_name)

        try:
            with open(file_path, "r") as f:
                for line in f:
                    if "FINAL SINGLE POINT ENERGY" in line:
                        if "Wavefunction not fully converged" in line:
                            return float("nan")
                        return float(line.split()[-1])
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{file_path}' not found.")

        raise ValueError("Single point energy not found in the output file.")

    
    @staticmethod
    def read_polarizability(output_file_name: str, directory: str) -> _torch.Tensor:
        """
        Read the polarizabilities from the ORCA output file.
        """
        file_path = _os.path.join(directory, output_file_name)
        try:
            polarizabilities = []
            with open(file_path, "r") as f:
                for line in f:
                    if "THE POLARIZABILITY TENSOR" in line:
                        print("FOUND *****")
                        for _ in range(3):
                            next(f)

                        polarizabilities = [
                            [float(value) for value in next(f).split()[:]]
                            for _ in range(3)
                        ]
                        break
            if not polarizabilities:
                raise ValueError("Polarizability tensor not found in the output file.")
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{file_path}' not found.")

        return _torch.tensor(polarizabilities)

    @staticmethod
    def write_input_file(
        elements: list[str],
        positions: _torch.Tensor,
        orca_simple_input: str,
        orca_blocks: str,
        charge: int,
        multiplicity: int,
        input_file_name: str,
        directory: str,
    ) -> str:
        """
        Write the ORCA input file.

        Parameters
        ----------
        elements: list[str]
            The list of elements.
        positions: torch.Tensor
            The positions of the atoms.
        orca_simple_input: str
            The simple input for ORCA.
        orca_blocks: str
            The blocks for ORCA.
        charge: int
            The charge of the molecule.
        multiplicity: int
            The multiplicity of the molecule.
        input_file_name: str
            The name of the input file.
        directory: str
            The directory where to write the file.

        Returns
        -------
        str
            The input file
        """
        _os.makedirs(directory, exist_ok=True)

        input_file = (
            f"{orca_simple_input}\n{orca_blocks}\n* xyz {charge} {multiplicity}\n"
        )
        for element, position in zip(elements, positions):
            input_file += f"{element} {position[0].item()} {position[1].item()} {position[2].item()}\n"
        input_file += "*"

        with open(_os.path.join(directory, input_file_name), "w") as f:
            f.write(input_file)

        return input_file

    @staticmethod
    def write_external_potentials(
        external_potentials: _torch.Tensor,
        file_name: str = "pointcharges.pc",
        directory: str = ".",
    ) -> None:
        """
        Write the external potentials to a file.

        Parameters
        ----------
        external_potentials: torch.Tensor
            The external potentials in the format (charge, x, y, z).
        file_name: str
            The name of the file to write.
        directory: str
            The directory where to write
        """
        with open(_os.path.join(directory, file_name), "w") as f:
            f.write(f"{external_potentials.size(0)}\n")
            for charge, x, y, z in external_potentials:
                f.write(f"{charge.item()} {x.item()} {y.item()} {z.item()}\n")


    @staticmethod
    def write_mesh(mesh: _torch.Tensor, file_name: str, directory: str):
        """
        Write the mesh in Bohr to a file.

        Parameters
        ----------
        mesh: torch.Tensor
            The mesh in Angstrom.
        file_name: str
            The name of the file to write.
        directory: str
            The directory where to write
        """
        mesh_in_bohr = mesh * ANGSTROM_TO_BOHR
        with open(_os.path.join(directory, file_name), "w") as f:
            f.write(f"{mesh.size(0)}\n")
            for x, y, z in mesh_in_bohr:
                f.write(f"{x.item()} {y.item()} {z.item()}\n")

    # -------------------------------------------------------------------------------------------- #
    #                                                                                              #
    #                                      ORCA CALCULATIONS                                       # 
    #                                                                                              #
    # -------------------------------------------------------------------------------------------- #
    def get_potential_energy(
        self,
        elements: list[str],
        positions: _torch.Tensor,
        orca_simple_input: str = "! b3lyp cc-pvtz TightSCF NoFrozenCore KeepDens",
        orca_blocks: str = "%pal nprocs 1 end",
        orca_external_potentials: _torch.Tensor = None,
        charge: int = 0,
        multiplicity: int = 1,
        input_file_name: Union[str, None] = None,
        output_file_name: Union[str, None] = None,
        directory: str = ".",
    ) -> float:
        """
        Run a single point energy calculation with ORCA.

        Parameters
        ----------
        elements: list[str]
            The list of elements.
        positions: torch.Tensor
            The positions of the atoms.
        orca_simple_input: str
            The simple input for ORCA.
        orca_blocks: str
            The blocks for ORCA.
        orca_external_potentials: torch.Tensor 
            The external potentials in the format (charge, x, y, z).
        charge: int
            The charge of the molecule.
        multiplicity: int
            The multiplicity of the molecule.
        input_file_name: str
            The name of the input file.
        output_file_name: str
            The name of the output file.
        directory: str
            The directory where to write the files.

        Returns
        -------
        float
            The single point energy in Hartree.
        """
        input_file_name = input_file_name or f"{self._name_prefix}.inp"
        output_file_name = output_file_name or f"{self._name_prefix}.out"

        if orca_external_potentials is not None:
            orca_blocks += '\n%pointcharges "pointcharges.pc"\n'

        self.write_input_file(
            elements=elements,
            positions=positions,
            orca_simple_input=orca_simple_input,
            orca_blocks=orca_blocks,
            charge=charge,
            multiplicity=multiplicity,
            input_file_name=input_file_name,
            directory=directory,
        )

        if orca_external_potentials is not None:
            self.write_external_potentials(
                external_potentials=orca_external_potentials,
                file_name="pointcharges.pc",
                directory=directory,
            )

        self._run_process(_os.path.join(self._orca_home, self._ORCA_BIN), [input_file_name], output_file_name, directory)

        sp_energy = self.read_single_point_energy(output_file_name, directory)

        return sp_energy

    def get_vpot(
        self, mesh: _torch.Tensor, directory: str, output_file_name: str = "vpot.out"
    ) -> _torch.Tensor:
        """
        Get the vpot from ORCA. 

        Parameters
        ----------
        mesh: torch.Tensor (N, 3)
            The mesh in Angstrom.
        directory: str
            The directory where to write the files.
        output_file_name: str
            The name of the output file.

        Returns
        -------
        torch.Tensor (N,)
            The vpot values in Hartree.
        """ 
        self.write_mesh(mesh, f"{self._name_prefix}.vpot.xyz", directory)
        self._run_process(
            _os.path.join(self._orca_home, self._ORCA_VPOT_BIN),
            [
                f"{self._name_prefix}.gbw",
                f"{self._name_prefix}.scfp",
                f"{self._name_prefix}.vpot.xyz",
                f"{self._name_prefix}.vpot.out",
            ],
            output_file_name,
            directory,
        )

        return self.read_vpot(f"{self._name_prefix}.vpot.out", directory)

    def get_mkl(self, directory: str, output_file_name: str = "orca_2mkl.out") -> None:
        """
        Get the mkl from ORCA.

        Parameters
        ----------
        directory: str
            The directory where to write the files.
        output_file_name: str
            The name of the output file.
        """
        self._run_process(
            _os.path.join(self._orca_home, self._ORCA_2MKL_BIN),
            [f"{self._name_prefix}", "-molden"],
            output_file_name,
            directory,
        )

    def get_polarizability(self, directory: str, output_file_name: Union[str, None] = None) -> _torch.tensor:
        """
        Read the polarizabilities from the ORCA output file.

        Parameters
        """
        output_file_name = output_file_name or f"{self._name_prefix}.out"
        return self.read_polarizability(output_file_name, directory)

if __name__ == "__main__":
    import torch

    orca = ORCACalculator()

    # Define elements
    elements = ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"]

    # Define positions in Angstrom
    pos = torch.tensor(
        [
            [17.666, 16.280, 18.146],
            [17.596, 17.503, 18.812],
            [17.287, 18.682, 18.077],
            [17.030, 18.547, 16.770],
            [17.141, 17.310, 16.095],
            [17.495, 16.194, 16.787],
            [18.070, 15.324, 18.495],
            [18.076, 17.553, 19.770],
            [17.265, 19.688, 18.530],
            [16.648, 19.361, 16.185],
            [17.095, 17.231, 15.022],
            [17.594, 15.289, 16.273],
        ]
    )

    # Vacuum energy
    en = orca.get_potential_energy(elements=elements, positions=pos, directory="vacuum")

    assert torch.isclose(torch.tensor(en), torch.tensor(-232.159529305631), atol=1e-7)
