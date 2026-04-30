import numpy as np

def mse(img, rec):
    return np.mean((img - rec) ** 2)

def psnr(img, rec, max_val=1.0):
    m = mse(img, rec)
    if m == 0:
        return float("inf")
    return 10 * np.log10((max_val ** 2) / m)