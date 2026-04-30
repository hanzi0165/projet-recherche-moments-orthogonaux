import numpy as np

def reconstruct_2d(M, Tx, Ty):
    return Ty.T @ M @ Tx

def reconstruct_3d(M, Tx, Ty, Tz):
    return np.einsum('qpr,qi,pj,rk->ijk', M, Ty, Tx, Tz)