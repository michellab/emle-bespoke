import h5py


with h5py.File('horton.out', 'r') as file:
    print(file['radial_moments'][:, 3][0])











