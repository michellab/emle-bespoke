"""Module for geometry-related functions."""

import numpy as _np


def calculate_moment_of_inertia_tensor(positions, masses):
    """
    Calculate the moment of inertia tensor for a molecule.

    Parameters
    ----------
    positions : np.ndarray, shape=(n_atoms, 3)
        Atomic positions.
    masses : np.ndarray, shape=(n_atoms,)
        Atomic masses.

    Returns
    -------
    np.ndarray, shape=(3, 3)
        Moment of inertia tensor.
    """
    inertia_tensor = _np.zeros((3, 3))

    for i, mass in enumerate(masses):
        x, y, z = positions[i]
        inertia_tensor[0, 0] += mass * (y**2 + z**2)  # Ixx
        inertia_tensor[1, 1] += mass * (x**2 + z**2)  # Iyy
        inertia_tensor[2, 2] += mass * (x**2 + y**2)  # Izz
        inertia_tensor[0, 1] -= mass * x * y  # Ixy
        inertia_tensor[0, 2] -= mass * x * z  # Ixz
        inertia_tensor[1, 2] -= mass * y * z  # Iyz

    # Symmetry: Ixy = Iyx, Ixz = Izx, Iyz = Izy
    inertia_tensor[1, 0] = inertia_tensor[0, 1]
    inertia_tensor[2, 0] = inertia_tensor[0, 2]
    inertia_tensor[2, 1] = inertia_tensor[1, 2]

    return inertia_tensor


def principal_axes(positions, masses):
    """
    Compute the principal moments and axes of a molecule.

    Parameters
    ----------
    positions : np.ndarray, shape=(n_atoms, 3)
        Atomic positions.
    masses : np.ndarray, shape=(n_atoms,)
        Atomic masses.

    Returns
    -------
    eigenvalues, eigenvectors
        Principal moments and axes.
    """
    inertia_tensor = calculate_moment_of_inertia_tensor(positions, masses)
    eigenvalues, eigenvectors = _np.linalg.eigh(inertia_tensor)
    return eigenvalues, eigenvectors


def rodrigues_rotation(v, k, theta):
    """
    Rotate a vector v around an axis k by an angle theta using Rodrigues' rotation formula.

    :param v: The vector to rotate (3x1 array).
    :param k: The unit vector representing the axis of rotation (3x1 array).
    :param theta: The rotation angle in radians.

    :return: The rotated vector (3x1 array).
    """
    # Ensure k is a unit vector
    k = k / _np.linalg.norm(k)

    # Rodrigues' formula for rotation
    v_rot = (
        v * _np.cos(theta)
        + _np.cross(k, v) * _np.sin(theta)
        + k * _np.dot(k, v) * (1 - _np.cos(theta))
    )

    return v_rot


if __name__ == "__main__":
    # Benzene molecule
    positions = _np.array(
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
    masses = _np.array([12.011] * 6 + [1.008] * 6)

    # Calculate Principal Axes
    moments, axes = principal_axes(positions, masses)

    # Output Results
    print("Principal Moments of Inertia (Eigenvalues):")
    print(moments)

    print("Principal Axes (Eigenvectors):")
    print(axes)

    # Put the molecule in the origin
    positions -= positions.mean(axis=0)

    # Align to principal axis
    rotated_positions = positions @ axes

    # 3d plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        rotated_positions[:, 0], rotated_positions[:, 1], rotated_positions[:, 2]
    )

    plt.savefig("benzene.png")

    print(rotated_positions)
