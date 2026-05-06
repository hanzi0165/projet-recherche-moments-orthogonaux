import numpy as np
from basis import normalized_coords, chebyshev_basis_1d

def chebyshev_moments_2d(img, n_max):
    H, W = img.shape
    x = normalized_coords(W)
    y = normalized_coords(H)

    Tx = chebyshev_basis_1d(n_max, x)   # (n+1, W)
    Ty = chebyshev_basis_1d(n_max, y)   # (n+1, H)

    # Résoudre min ||img - Ty.T @ M @ Tx||_F de manière numériquement stable.
    By = Ty.T  # (H, n+1)
    Bx = Tx.T  # (W, n+1)

    # Étape 1 : résoudre By @ A ~= img  -> A ~= M @ Bx.T
    A, *_ = np.linalg.lstsq(By, img, rcond=None)
    # Étape 2 : résoudre Bx @ Z ~= A.T avec Z = M.T
    Z, *_ = np.linalg.lstsq(Bx, A.T, rcond=None)
    M = Z.T
    return M, Tx, Ty