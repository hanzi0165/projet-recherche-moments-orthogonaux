import numpy as np


def normalized_coords_01(length: int):
    # Map pixel centers to [-1, 1]
    idx = np.arange(length, dtype=np.float64)
    coords = 2.0 * (idx + 0.5) / length - 1.0
    delta = 2.0 / length
    return coords, delta


def fractional_chebyshev_basis_1d(n_max: int, coords: np.ndarray, alpha: float):
    """
    Simplified fractional-like Chebyshev basis.
    Uses sign(x)*|x|^alpha so it also works on [-1,1].
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    x = np.sign(coords) * (np.abs(coords) ** alpha)

    T = np.zeros((n_max + 1, len(x)), dtype=np.float64)
    T[0] = 1.0
    if n_max >= 1:
        T[1] = x

    for n in range(1, n_max):
        T[n + 1] = 2.0 * x * T[n] - T[n - 1]

    return T


def frcm_3d(vol, n_max, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0):
    """
    Stable 3D fractional-like moments:
    - fractional-style basis (sign(x)*|x|^alpha)
    - QR orthogonalization
    - einsum projection (like moments_3d)
    
    vol shape: (H, W, D)
    Returns: M (moments tensor), Tx, Ty, Tz (orthonormal basis matrices)
    """
    H, W, D = vol.shape

    x, dx = normalized_coords_01(W)
    y, dy = normalized_coords_01(H)
    z, dz = normalized_coords_01(D)

    Tx_raw = fractional_chebyshev_basis_1d(n_max, x, alpha_x)   # (n+1, W)
    Ty_raw = fractional_chebyshev_basis_1d(n_max, y, alpha_y)   # (n+1, H)
    Tz_raw = fractional_chebyshev_basis_1d(n_max, z, alpha_z)   # (n+1, D)

    # Orthogonalize basis for numerical stability
    Qx, _ = np.linalg.qr(Tx_raw.T)   # (W, n+1)
    Qy, _ = np.linalg.qr(Ty_raw.T)   # (H, n+1)
    Qz, _ = np.linalg.qr(Tz_raw.T)   # (D, n+1)

    Tx = Qx.T[:n_max+1, :]           # (n+1, W)
    Ty = Qy.T[:n_max+1, :]           # (n+1, H)
    Tz = Qz.T[:n_max+1, :]           # (n+1, D)

    # Projection using einsum (like moments_3d)
    M = np.einsum('ijk,qi,pj,rk->qpr', vol, Ty, Tx, Tz)

    return M, Tx, Ty, Tz
