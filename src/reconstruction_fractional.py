import numpy as np

def reconstruct_fractional_2d(C, Tx, Ty, dx, dy):
    """
    C  : (n_max+1, n_max+1)
    Tx : (n_max+1, W)
    Ty : (n_max+1, H)
    dx : pas spatial en x (= 1/W)
    dy : pas spatial en y (= 1/H)
    """
    # Recalcul des normes (mêmes que dans frcm_2d)
    d2x = np.sum(Tx * Tx, axis=1) * dx   # (n_max+1,)
    d2y = np.sum(Ty * Ty, axis=1) * dy   # (n_max+1,)

    # Facteur de normalisation pour chaque (n,m)
    norm = np.sqrt(np.outer(d2y, d2x))   # (n_max+1, n_max+1)

    # C_raw = C avant normalisation
    C_raw = C / norm                      # annule la division faite dans frcm_2d

    # Reconstruction
    rec = Ty.T @ C_raw @ Tx
    return rec