import numpy as np
from basis import normalized_coords, chebyshev_basis_1d

def chebyshev_moments_3d(vol, n_max):
    H, W, D = vol.shape

    x = normalized_coords(W)
    y = normalized_coords(H)
    z = normalized_coords(D)

    Tx = chebyshev_basis_1d(n_max, x)
    Ty = chebyshev_basis_1d(n_max, y)
    Tz = chebyshev_basis_1d(n_max, z)

    M = np.einsum('ijk,qi,pj,rk->qpr', vol, Ty, Tx, Tz)
    return M, Tx, Ty, Tz