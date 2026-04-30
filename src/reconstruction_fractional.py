import numpy as np


def reconstruct_fractional_2d(M, Tx, Ty, dx=None, dy=None):
    """
    Reconstruct 2D image from fitted fractional-like Chebyshev moments.
    M: moments matrix
    Tx, Ty: orthonormal basis matrices
    dx, dy: (unused, kept for API compatibility)
    """
    rec = Ty.T @ M @ Tx
    return rec


def reconstruct_fractional_3d(M, Tx, Ty, Tz, dx=None, dy=None, dz=None):
    """
    Reconstruct 3D volume from fitted fractional-like Chebyshev moments.
    M: moments tensor (n+1, n+1, n+1)
    Tx, Ty, Tz: orthonormal basis matrices
    dx, dy, dz: (unused, kept for API compatibility)
    """
    rec = np.einsum('qpr,qi,pj,rk->ijk', M, Ty, Tx, Tz)
    return rec