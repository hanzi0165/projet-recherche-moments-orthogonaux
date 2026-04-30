import numpy as np
from basis import normalized_coords, chebyshev_basis_1d

def chebyshev_moments_2d(img, n_max):
    H, W = img.shape
    x = normalized_coords(W)
    y = normalized_coords(H)

    Tx = chebyshev_basis_1d(n_max, x)   # (n+1, W)
    Ty = chebyshev_basis_1d(n_max, y)   # (n+1, H)

    # Solve min ||img - Ty.T @ M @ Tx||_F in a numerically stable way.
    By = Ty.T  # (H, n+1)
    Bx = Tx.T  # (W, n+1)

    # Step 1: solve By @ A ~= img  -> A ~= M @ Bx.T
    A, *_ = np.linalg.lstsq(By, img, rcond=None)
    # Step 2: solve Bx @ Z ~= A.T with Z = M.T
    Z, *_ = np.linalg.lstsq(Bx, A.T, rcond=None)
    M = Z.T
    return M, Tx, Ty