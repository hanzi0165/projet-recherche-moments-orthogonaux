"""
reconstruction_3d.py

Pipeline complet pour les fichiers McGill .im.gz :
  1. Lecture + décompression gzip
  2. Moments de Chebyshev 3D
  3. Reconstruction inverse
  4. Métriques MSE / PSNR
  5. Visualisation coupes + courbes

Analogie avec le pipeline 2D :
  2D : image (H, W)         → moments M[n,m]     → reconstruction
  3D : volume (D, H, W)     → moments M[n,m,p]   → reconstruction
"""

import gzip
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LECTURE DU FICHIER .im.gz
# ═══════════════════════════════════════════════════════════════════════════════

def load_im_gz(path: str) -> np.ndarray:
    """
    Lit un fichier .im.gz du McGill Benchmark.
    Format : header 1024 bytes + données 128³ bytes (uint8).
    Retourne un array float64 de shape (128, 128, 128), valeurs ∈ {0, 1}.
    """
    with gzip.open(path, 'rb') as f:
        f.read(1024)   # sauter le header
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return (data.reshape(128, 128, 128) > 0).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BASE DE CHEBYSHEV 1D
# ═══════════════════════════════════════════════════════════════════════════════

def normalized_coords(length: int) -> np.ndarray:
    """Coordonnées normalisées dans [-1, 1]. Même formule que le pipeline 2D."""
    idx = np.arange(length, dtype=np.float64)
    return (2 * idx + 1) / length - 1


def chebyshev_basis_1d(n_max: int, coords: np.ndarray) -> np.ndarray:
    """
    Polynômes T_0 … T_{n_max} évalués sur coords.
    Récurrence : T_{n+1} = 2x·T_n - T_{n-1}
    Retourne T de shape (n_max+1, len(coords)).
    """
    L = len(coords)
    T = np.zeros((n_max + 1, L), dtype=np.float64)
    T[0] = 1.0
    if n_max >= 1:
        T[1] = coords
    for n in range(1, n_max):
        T[n + 1] = 2 * coords * T[n] - T[n - 1]
    return T


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MOMENTS DE CHEBYSHEV 3D
# ═══════════════════════════════════════════════════════════════════════════════

def chebyshev_moments_3d(vol: np.ndarray, n_max: int):
    """
    Calcule les moments M_{nmp} pour n,m,p ∈ [0, n_max].

    Analogie avec le 2D :
      2D : M_{nm}  = Σ_{i,j}   f(i,j)   · T_n(x_i) · T_m(y_j)
      3D : M_{nmp} = Σ_{i,j,k} f(i,j,k) · T_n(x_i) · T_m(y_j) · T_p(z_k)

    Retourne M de shape (n_max+1, n_max+1, n_max+1).
    """
    D, H, W = vol.shape

    Tx = chebyshev_basis_1d(n_max, normalized_coords(W))  # (n+1, W)
    Ty = chebyshev_basis_1d(n_max, normalized_coords(H))  # (n+1, H)
    Tz = chebyshev_basis_1d(n_max, normalized_coords(D))  # (n+1, D)

    # Calcul par étapes pour éviter un tenseur 6D en mémoire
    # Étape 1 : sommer sur x → shape (n+1, D, H)
    tmp = np.tensordot(Tx, vol, axes=([1], [2]))    # (n+1, D, H)
    # Étape 2 : sommer sur y → shape (n+1, n+1, D)
    tmp = np.tensordot(Ty, tmp, axes=([1], [2]))    # (n+1, n+1, D)
    # Étape 3 : sommer sur z → shape (n+1, n+1, n+1)
    M = np.tensordot(Tz, tmp, axes=([1], [2]))      # (n+1, n+1, n+1)

    # Normalisation par le nombre total de voxels
    M /= (D * H * W)

    return M, Tx, Ty, Tz


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RECONSTRUCTION INVERSE
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_3d(M: np.ndarray,
                   Tx: np.ndarray,
                   Ty: np.ndarray,
                   Tz: np.ndarray) -> np.ndarray:
    """
    Reconstruction inverse :
    f̂(k,j,i) = Σ_{n,m,p} M_{nmp} · Tz[n,k] · Ty[m,j] · Tx[p,i]

    Même logique que reconstruct_2d, étendue à 3 axes.
    Retourne f̂ de shape (D, H, W).
    """
    # Étape 1 : sommer sur p (axe x) → (n+1, n+1, W)
    tmp = np.tensordot(M, Tx, axes=([2], [0]))      # (n+1, n+1, W)
    # Étape 2 : sommer sur m (axe y) → (H, n+1, W) puis transpose
    tmp = np.tensordot(Ty, tmp, axes=([0], [1]))    # (H, n+1, W)
    tmp = tmp.transpose(1, 0, 2)                    # (n+1, H, W)
    # Étape 3 : sommer sur n (axe z) → (D, H, W)
    rec = np.tensordot(Tz, tmp, axes=([0], [0]))    # (D, H, W)

    return rec


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MÉTRIQUES
# ═══════════════════════════════════════════════════════════════════════════════

def mse_3d(f, f_hat):
    """Erreur quadratique moyenne entre volume original et reconstruit."""
    return float(np.mean((f - f_hat) ** 2))


def psnr_3d(f, f_hat, L=1.0):
    """PSNR en dB (L=1.0 car image binaire)."""
    m = mse_3d(f, f_hat)
    if m < 1e-15:
        return float('inf')
    return float(10 * np.log10(L ** 2 / m))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_slices(vol_orig, vol_rec, n_max, mse_val, psnr_val, out_path=None):
    """
    3 coupes orthogonales (XY / XZ / YZ) au centre du volume,
    original en haut, reconstruction en bas.
    """
    D, H, W = vol_orig.shape
    mid = D // 2

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        f"Reconstruction 3D — Moments de Chebyshev  N={n_max}\n"
        f"MSE = {mse_val:.6f}  |  PSNR = {psnr_val:.2f} dB",
        fontsize=12
    )

    slices = [
        (vol_orig[mid, :, :], vol_rec[mid, :, :], f"Coupe XY (z={mid})"),
        (vol_orig[:, mid, :], vol_rec[:, mid, :], f"Coupe XZ (y={mid})"),
        (vol_orig[:, :, mid], vol_rec[:, :, mid], f"Coupe YZ (x={mid})"),
    ]

    for col, (orig_s, rec_s, title) in enumerate(slices):
        axes[0, col].imshow(orig_s, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"Original — {title}", fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(rec_s, cmap="gray")
        axes[1, col].set_title(f"Reconstruit — {title}", fontsize=9)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Original",    fontsize=10)
    axes[1, 0].set_ylabel("Reconstruit", fontsize=10)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"  Coupes sauvegardées : {out_path}")
    plt.close(fig)


def plot_curves(n_values, mse_values, psnr_values, out_path=None):
    """Courbes MSE (log) et PSNR en fonction de N."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].semilogy(n_values, mse_values, marker='o', color='steelblue')
    axes[0].set_xlabel("N (ordre maximal)")
    axes[0].set_ylabel("MSE (échelle logarithmique)")
    axes[0].set_title("MSE vs N — Reconstruction 3D")
    axes[0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.5f}")
    )
    axes[0].set_ylim([min(mse_values) * 0.8, max(mse_values) * 1.3])
    axes[0].grid(True, which="both", linestyle="--", alpha=0.6)

    axes[1].plot(n_values, psnr_values, marker='o', color='darkorange')
    axes[1].set_xlabel("N (ordre maximal)")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("PSNR vs N — Reconstruction 3D")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"  Courbes sauvegardées : {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PROGRAMME PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Paramètres — modifier ici ────────────────────────────────────────────
    IM_PATH  = r"C:\Users\Flora\Downloads\cupsIm\cupsIm\b1.im.gz"   # ← ton fichier .im.gz
    N_VALUES = [3, 5, 8,12, 15, 20]  # ← les ordres N à tester
    # Conseil : commencer par [3, 5, 8] pour tester la vitesse
    # N=20 sur 128³ peut prendre quelques minutes

    # ── Répertoire de sortie ─────────────────────────────────────────────────
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()
    out_dir = script_dir.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── Chargement ───────────────────────────────────────────────────────────
    vol = load_im_gz(IM_PATH)
    stem = Path(IM_PATH).stem
    print(f"Fichier       : {Path(IM_PATH).name}")
    print(f"Shape         : {vol.shape}")
    print(f"Voxels actifs : {int(vol.sum())} / {vol.size}  ({100*vol.mean():.1f}%)")
    print()

    # ── Boucle sur les ordres N ──────────────────────────────────────────────
    mse_values  = []
    psnr_values = []
    last_rec    = None

    for n_max in N_VALUES:
        print(f"N = {n_max}  →  dimension vecteur = (N+1)³ = {(n_max+1)**3} ...")
        M, Tx, Ty, Tz = chebyshev_moments_3d(vol, n_max)
        rec = reconstruct_3d(M, Tx, Ty, Tz)

        m = mse_3d(vol, rec)
        p = psnr_3d(vol, rec)
        mse_values.append(m)
        psnr_values.append(p)
        last_rec = rec

        print(f"  MSE  = {m:.6f}")
        print(f"  PSNR = {p:.2f} dB")
        print()

    # ── Visualisation coupes ─────────────────────────────────────────────────
    slices_path = out_dir / f"3d_slices_{stem}_N{N_VALUES[-1]}_{ts}.png"
    plot_slices(vol, last_rec,
                n_max=N_VALUES[-1],
                mse_val=mse_values[-1],
                psnr_val=psnr_values[-1],
                out_path=slices_path)

    # ── Courbes MSE / PSNR ───────────────────────────────────────────────────
    curves_path = out_dir / f"3d_curves_{stem}_{ts}.png"
    plot_curves(N_VALUES, mse_values, psnr_values, out_path=curves_path)

    # ── Tableau récapitulatif ─────────────────────────────────────────────────
    print("=" * 52)
    print(f"{'N':>4}  {'(N+1)³':>8}  {'MSE':>12}  {'PSNR':>8}")
    print("-" * 52)
    for n, m, p in zip(N_VALUES, mse_values, psnr_values):
        print(f"{n:>4}  {(n+1)**3:>8}  {m:>12.6f}  {p:>8.2f} dB")
    print("=" * 52)