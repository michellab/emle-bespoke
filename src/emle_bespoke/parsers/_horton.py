import subprocess

def run_horton_wpart(input_file, output_file, scheme='mbis', lmax=3, output_log='horton.out'):
    """
    Runs horton-wpart.py with the specified arguments and captures the output.

    Parameters
    ----------
    input_file : str
        The wavefunction input file, e.g., 'b3lyp.molden.input'.
    output_file : str
        The output HDF5 file, e.g., '${i}.h5'.
    scheme : str
        The partitioning scheme to be used, default is 'mbis'.
    lmax : int
        The maximum angular momentum to consider in multipole expansions, default is 3.
    output_log : str
        The filename where the command output is saved, default is 'horton.out'.

    Returns
    -------
    str
        The captured standard output from the command.
    """
    # Define the command
    command = [
        'horton-wpart.py',
        input_file,
        output_file,
        scheme,
        '--lmax', str(lmax)
    ]

    # Run the command and capture the output
    with open(output_log, 'w') as log_file:
        result = subprocess.run(command, stdout=log_file, stderr=subprocess.PIPE, text=True)

    # Check for errors
    if result.returncode != 0:
        raise RuntimeError(f"Error running horton-wpart.py: {result.stderr}")

    # Return the captured output in the log file for confirmation
    with open(output_log, 'r') as log_file:
        output = log_file.read()

    return output
