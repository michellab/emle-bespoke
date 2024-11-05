import os
import subprocess

import torch

from .._constants import ANGSTROM_TO_BOHR


class ORCACalculator:
    _ORCA_BIN = "orca"
    _ORCA_VPOT_BIN = "orca_vpot"
    _ORCA_VPOT_OUT = "orca.vpot.out"

    def __init__(self, orca_home: str = None):
        """
        Initialize the ORCA calculator.
        """
        self.orca_home = orca_home or os.environ.get("ORCA_HOME")
        if not self.orca_home:
            raise ValueError("ORCA_HOME is not set.")

    def write_input_file(
        self,
        elements: list[str],
        positions: torch.Tensor,
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
        os.makedirs(directory, exist_ok=True)

        input_file = (
            f"{orca_simple_input}\n{orca_blocks}\n* xyz {charge} {multiplicity}\n"
        )
        for element, position in zip(elements, positions):
            input_file += f"{element} {position[0].item()} {position[1].item()} {position[2].item()}\n"
        input_file += "*"

        with open(os.path.join(directory, input_file_name), "w") as f:
            f.write(input_file)

        return input_file

    def write_external_potentials(
        self,
        external_potentials: torch.Tensor,
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
        with open(os.path.join(directory, file_name), "w") as f:
            f.write(f"{external_potentials.size(0)}\n")
            for charge, x, y, z in external_potentials:
                f.write(f"{charge.item()} {x.item()} {y.item()} {z.item()}\n")

    def get_potential_energy(
        self,
        elements: list[str],
        positions: torch.Tensor,
        orca_simple_input: str = "! b3lyp cc-pvtz TightSCF NoFrozenCore KeepDens",
        orca_blocks: str = "%pal nprocs 1 end",
        orca_external_potentials: torch.Tensor = None,
        charge: int = 0,
        multiplicity: int = 1,
        input_file_name: str = "orca.inp",
        output_file_name: str = "orca.out",
        directory: str = ".",
    ) -> float:
        """
        Get the potential energy of a molecule with ORCA.
        """
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

        self.run_orca(self._ORCA_BIN, [input_file_name], output_file_name, directory)

        return self.read_single_point_energy(output_file_name, directory)

    def write_mesh(self, mesh: torch.Tensor, file_name: str, directory: str):
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
        with open(os.path.join(directory, file_name), "w") as f:
            f.write(f"{mesh.size(0)}\n")
            for x, y, z in mesh_in_bohr:
                f.write(f"{x.item()} {y.item()} {z.item()}\n")

    def get_vpot(
        self, mesh: torch.Tensor, directory: str, output_file_name: str = "vpot.out"
    ) -> torch.Tensor:
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
        self.write_mesh(mesh, "orca.vpot.xyz", directory)
        self.run_orca(
            self._ORCA_VPOT_BIN,
            ["orca.gbw", "orca.scfp", "orca.vpot.xyz", "orca.vpot.out"],
            output_file_name,
            directory,
        )
        return self.read_vpot(self._ORCA_VPOT_OUT, directory)
    
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
        self.run_orca(
            "orca_2mkl",
            ["b3lyp", "-molden"],
            output_file_name,
            directory,
        )

    def read_vpot(self, output_file_name: str, directory: str) -> torch.Tensor:
        """
        Read the vpot from the ORCA output file.
        """
        return torch.tensor(np.loadtxt(os.path.join(directory, output_file_name), skiprows=1)[:, 3])

    def read_single_point_energy(self, output_file_name: str, directory: str) -> float:
        """
        Read the single point energy from the ORCA output file.
        """
        with open(os.path.join(directory, output_file_name), "r") as f:
            lines = f.readlines()

        for line in lines:
            if "FINAL SINGLE POINT ENERGY" in line:
                if "Wavefunction not fully converged" in line:
                    return float("nan")
                return float(line.split()[-1])

        raise ValueError("Single point energy not found in the output file.")

    def run_orca(
        self, bin: str, arguments: list[str], output_file_name: str, directory: str
    ) -> subprocess.CompletedProcess:
        """
        Run ORCA with the given input file.
        """
        commands = [os.path.join(self.orca_home, bin)] + arguments

        assert os.path.isfile(commands[0]), f"{commands[0]} binary does not exist."

        with open(os.path.join(directory, output_file_name), "w") as output_file:
            process = subprocess.run(
                commands,
                cwd=directory,
                text=True,
                stdout=output_file,
                stderr=subprocess.STDOUT,
            )

        if process.returncode != 0:
            raise ValueError(f"{bin} failed with the following error: {process.stderr}")

        return process


if __name__ == "__main__":
    import torch

    orca = ORCACalculator(orca_home="/home/joaomorado/orca")

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
