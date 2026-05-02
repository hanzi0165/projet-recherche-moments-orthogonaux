"""
frcmi_2d.py
Ajoute l'invariance (translation + échelle + rotation) par-dessus frcm_2d.
Dépend de fractional_2d.py (inchangé).
"""

import numpy as np
from scipy.ndimage import rotate as nd_rotate
from fractional_2d import frgm_2d, chebyshev_power_coeffs, fractional_chebyshev_basis_1d, numerical_basis_norms


# ── 1. Barycentre (formule 11) ────────────────────────────────────────────────

def compute_centroid(img, alpha_x=1.0, alpha_y=1.0):
    """
    X̂ = FrGM_10 / FrGM_00   (en coordonnées normalisées [0,1])
    Ŷ = FrGM_01 / FrGM_00
    """
    G, _, _ = frgm_2d(img, p_max=1, q_max=1, alpha_x=alpha_x, alpha_y=alpha_y)
    mass = G[0, 0]
    if mass < 1e-12:
        H, W = img.shape
        return 0.5, 0.5          # centre par défaut
    return G[0, 1] / mass, G[1, 0] / mass   # (cx_norm, cy_norm)


# ── 2. Intégrales centrées (formules 14-16) ───────────────────────────────────

def _signed_pow(x, expo):
    """sign(x) * |x|^expo  — stable pour x négatif."""
    return np.sign(x) * (np.abs(x) ** expo)


def central_interval_integrals(length, p_max, alpha, centroid_norm):
    """
    IT_p^alpha(i) = 1/(alpha*p+1) * [(u_{i+1}-ĉ)^{alpha*p+1} - (u_i-ĉ)^{alpha*p+1}]
    centroid_norm : barycentre en [0,1]
    """
    dx = 1.0 / length
    lower = np.arange(length, dtype=np.float64) * dx
    upper = lower + dx
    lo = lower - centroid_norm
    hi = upper - centroid_norm

    IT = np.zeros((p_max + 1, length), dtype=np.float64)
    for p in range(p_max + 1):
        expo = alpha * p + 1.0
        IT[p] = (_signed_pow(hi, expo) - _signed_pow(lo, expo)) / expo
    return IT


# ── 3. Moments géométriques centrés (formules 12-13) ─────────────────────────

def central_frgm_2d(img, n_max, alpha_x=1.0, alpha_y=1.0):
    """η_{pq} = Σ f(i,j) IT_x^p(i) IT_y^q(j)  avec IT centrés sur le barycentre."""
    H, W = img.shape
    cx_norm, cy_norm = compute_centroid(img, alpha_x, alpha_y)
    ITx = central_interval_integrals(W, n_max, alpha_x, cx_norm)
    ITy = central_interval_integrals(H, n_max, alpha_y, cy_norm)
    G_central = np.einsum("ji,qj,pi->qp", img.astype(np.float64), ITy, ITx)
    return G_central


# ── 4. Angle canonique (formule 21) ──────────────────────────────────────────

def estimate_rotation_angle(G_central):
    """θ = 0.5 · arctan(2·η_11 / (η_20 - η_02))"""
    eta_20 = G_central[0, 2]
    eta_02 = G_central[2, 0]
    eta_11 = G_central[1, 1]
    return 0.5 * np.arctan2(2.0 * eta_11, eta_20 - eta_02)


# ── 5. Normalisation échelle (formule 19, 21) ─────────────────────────────────

def scale_normalize(G, n_max, alpha_x, alpha_y):
    """FrGMI_{pq} = η_{pq} / λ^γ,  λ = η_00,  γ = 1 + (αx·p + αy·q)/2"""
    lam = G[0, 0]
    if abs(lam) < 1e-12:
        return G.copy()
    G_inv = np.zeros_like(G)
    for q in range(n_max + 1):
        for p in range(n_max + 1):
            gamma = 1.0 + (alpha_x * p + alpha_y * q) / 2.0
            G_inv[q, p] = G[q, p] / (lam ** gamma)
    return G_inv


# ── 6. Pipeline complet FrCMI (formule 25) ───────────────────────────────────

def frcmi_2d(img, n_max, alpha_x=1.0, alpha_y=1.0):
    """
    Retourne le vecteur FrCMI aplati, invariant à translation/échelle/rotation.

    Étapes :
      1. Moments centrés  → invariance translation
      2. Rotation canonique → invariance rotation
      3. Normalisation λ^γ  → invariance échelle
      4. Combinaison B_{n,l} → FrCMI
    """
    H, W = img.shape
    img = img.astype(np.float64)

    # 1) Moments centrés pour estimer l'angle (besoin d'ordre >= 2)
    G_central = central_frgm_2d(img, max(n_max, 2), alpha_x, alpha_y)

    # 2) Rotation vers pose canonique
    theta_deg = np.degrees(estimate_rotation_angle(G_central))
    img_rot = nd_rotate(img, -theta_deg, reshape=False, mode="nearest")

    # 3) Moments centrés de l'image tournée + normalisation échelle
    G_rot = central_frgm_2d(img_rot, n_max, alpha_x, alpha_y)
    G_inv = scale_normalize(G_rot, n_max, alpha_x, alpha_y)

    # 4) FrCMI = B @ G_inv @ B.T  puis normalisation orthogonale
    B = chebyshev_power_coeffs(n_max)
    C_raw = B @ G_inv @ B.T

    Tx, _, dx = fractional_chebyshev_basis_1d(W, n_max, alpha_x)
    Ty, _, dy = fractional_chebyshev_basis_1d(H, n_max, alpha_y)
    d2x = numerical_basis_norms(Tx, dx)
    d2y = numerical_basis_norms(Ty, dy)
    norm = np.sqrt(np.outer(d2y, d2x))
    norm = np.where(norm < 1e-12, 1.0, norm)

    return (C_raw / norm).flatten()