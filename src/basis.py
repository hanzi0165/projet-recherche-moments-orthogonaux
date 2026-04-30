import numpy as np

def normalized_coords(length):
    idx = np.arange(length)
    return (2 * idx + 1) / length - 1

def chebyshev_basis_1d(n_max, coords):
    L = len(coords)
    T = np.zeros((n_max + 1, L), dtype=np.float64)
    T[0] = 1.0
    if n_max >= 1:
        T[1] = coords
    for n in range(1, n_max):
        T[n + 1] = 2 * coords * T[n] - T[n - 1]
    return T