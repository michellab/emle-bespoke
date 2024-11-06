"""Base class for calculators."""
import subprocess as _subprocess
import shutil as _shutil
from pathlib import Path as _Path

class BaseCalculator:
    """Base calculator class."""
    def _run_process(
        self, bin_exec: str, arguments: list[str], output_file_name: str, directory: str
    ) -> _subprocess.CalledProcessError:
        """
        Run a binary with the given arguments and capture the output in a specified file.

        Parameters
        ----------
        bin_exec : str
            The binary executable to run, either in the PATH or as a full path.
        arguments : list[str]
            The arguments to pass to the binary.
        output_file_name : str
            The name of the file where output should be saved.
        directory : str
            The directory in which to execute the command and write the output file.

        Returns
        -------
        CompletedProcess
            The completed process object containing execution details.

        Raises
        ------
        FileNotFoundError
            If the binary executable does not exist.
        subprocess.CalledProcessError
            If the process execution fails (non-zero return code).
        """
        full_path = _shutil.which(bin_exec)
        if full_path is None:
            raise FileNotFoundError(f"{bin_exec} not found.")

        bin_path = _Path(full_path)
        output_path = _Path(directory) / output_file_name
   
        if not bin_path.is_file():
            raise FileNotFoundError(f"{bin_path} binary does not exist.")

        commands = [str(bin_path)] + arguments

        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                process = _subprocess.run(
                    commands,
                    cwd=directory,
                    text=True,
                    stdout=output_file,
                    stderr=_subprocess.STDOUT,
                    check=True 
                )
        except _subprocess.CalledProcessError as e:
            raise _subprocess.CalledProcessError(
                e.returncode, e.cmd, output=e.output, stderr=e.stderr
            ) from e

        return process

    
    def get_potential_energy(self, *args, **kwargs):
        raise NotImplementedError("get_potential_energy method must be implemented.")
    
    def get_vpot(self, *args, **kwargs):
        raise NotImplementedError("get_vpot method must be implemented.")
    
    def get_mkl(self, *args, **kwargs):
        raise NotImplementedError("get_mkl method must be implemented.")
    
    def get_horton_partitioning(self, *args, **kwargs):
        raise NotImplementedError("get_horton_partitioning method must be implemented.")
    
    def get_polarizabilities(self, *args, **kwargs):
        raise NotImplementedError("get_polarizabilities method must be implemented.")