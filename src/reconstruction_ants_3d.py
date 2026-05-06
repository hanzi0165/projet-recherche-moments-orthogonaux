"""
reconstruction_3d_frcm_final.py

Version finale corrigée :
  - Lecture correcte des fichiers McGill .im.gz (valeurs uint8 brutes)
  - Volume réduit à 32³ pour calcul rapide
  - Rendu voxel limité pour éviter le freeze matplotlib
  - Récurrence stable pour les bases fractionnaires
  - Métriques MSE/PSNR/DICE correctes
"""

import gzip
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from datetime import datetime
from scipy.ndimage import zoom as nd_zoom


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def load_im_gz_raw(path: str) -> np.ndarray:
    """
    Lit un fichier .im.gz McGill sans binarisation.
    Retourne les valeurs uint8 brutes normalisées dans [0,1].
    Si toutes les valeurs sont dans {0,1}, c'est déjà binaire.
    """
    with gzip.open(path, 'rb') as f:
        f.read(1024)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    vol = data.reshape(128, 128, 128).astype(np.float64)
    vmax = vol.max()
    if vmax > 0:
        vol /= vmax
    print(f"  Valeurs uniques : {np.unique(vol)[:5]} ... max={vmax}")
    print(f"  → {'Données continues (0-255 normalisées)' if vmax > 1 else 'Données binaires {0,1}'}")
    return vol


def downsample_volume(vol: np.ndarray, target: int) -> np.ndarray:
    """Réduit le volume à target³ par interpolation."""
    factor = target / vol.shape[0]
    reduced = nd_zoom(vol, factor, order=1)   # order=1 pour préserver les valeurs continues
    return np.clip(reduced, 0, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BASE DE CHEBYSHEV FRACTIONNAIRE — RÉCURRENCE STABLE
# ═══════════════════════════════════════════════════════════════════════════════

def fractional_chebyshev_basis_stable(length: int, n_max: int,
                                        alpha: float):
    """
    Récurrence directe — numériquement stable :
      T̃_0 = 1
      T̃_1 = x^alpha
      T̃_{n+1} = 2·x^alpha·T̃_n - T̃_{n-1}
    """
    dx = 1.0 / length
    x  = np.arange(length, dtype=np.float64) / length + dx / 2.0
    x  = np.clip(x, 1e-12, None)
    xa = x ** alpha

    T = np.zeros((n_max + 1, length), dtype=np.float64)
    T[0] = 1.0
    if n_max >= 1:
        T[1] = xa
    for n in range(1, n_max):
        T[n + 1] = 2.0 * xa * T[n] - T[n - 1]
    return T, dx


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FrCM 3D STABLE
# ═══════════════════════════════════════════════════════════════════════════════

def frcm_3d_stable(vol: np.ndarray, n_max: int,
                    alpha_x: float = 1.2,
                    alpha_y: float = 1.0,
                    alpha_z: float = 1.2):
    """FrCM 3D par récurrence stable — formules (22)-(24) de l'article."""
    D, H, W = vol.shape

    Tx, dx = fractional_chebyshev_basis_stable(W, n_max, alpha_x)
    Ty, dy = fractional_chebyshev_basis_stable(H, n_max, alpha_y)
    Tz, dz = fractional_chebyshev_basis_stable(D, n_max, alpha_z)

    d2x = np.sum(Tx * Tx, axis=1) * dx
    d2y = np.sum(Ty * Ty, axis=1) * dy
    d2z = np.sum(Tz * Tz, axis=1) * dz

    tmp = np.tensordot(Tx, vol, axes=([1], [2]))
    tmp = np.tensordot(Ty, tmp, axes=([1], [2]))
    C   = np.tensordot(Tz, tmp, axes=([1], [2]))

    norm = np.sqrt(
        d2z[:, None, None] * d2y[None, :, None] * d2x[None, None, :]
    )
    norm = np.where(norm < 1e-12, 1.0, norm)
    C   /= norm

    return C, Tx, Ty, Tz


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_frcm_3d(C, Tx, Ty, Tz):
    """Reconstruction inverse — formule (4) de l'article."""
    tmp = np.tensordot(C,  Tx, axes=([2], [0]))
    tmp = np.tensordot(Ty, tmp, axes=([0], [1]))
    tmp = tmp.transpose(1, 0, 2)
    rec = np.tensordot(Tz, tmp, axes=([0], [0]))
    return rec


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MÉTRIQUES
# ═══════════════════════════════════════════════════════════════════════════════

def mse_3d(f, f_hat):
    return float(np.mean((f - f_hat) ** 2))

def psnr_3d(f, f_hat, L=1.0):
    m = mse_3d(f, f_hat)
    return float('inf') if m < 1e-15 else float(10 * np.log10(L**2 / m))

def dice_score(orig_bin, rec_bin):
    TP = int(np.sum((orig_bin == 1) & (rec_bin == 1)))
    FP = int(np.sum((orig_bin == 0) & (rec_bin == 1)))
    FN = int(np.sum((orig_bin == 1) & (rec_bin == 0)))
    return float(2 * TP / (2 * TP + FP + FN + 1e-12)), TP, FP, FN


# ═══════════════════════════════════════════════════════════════════════════════
# 6. VISUALISATION — COUPES ORTHOGONALES (rapide, sans voxels())
# ═══════════════════════════════════════════════════════════════════════════════

def plot_slices_comparison(vol, rec, n_max, alpha_x, alpha_y, alpha_z,
                            mse_val, psnr_val, out_path=None):
    """
    6 coupes orthogonales : original (haut) vs reconstruction (bas).
    Plus rapide et plus informatif que le rendu voxel pour des volumes 3D.
    """
    D, H, W = vol.shape
    mid = D // 2

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        f"Reconstruction FrCM 3D — N={n_max}  α=({alpha_x},{alpha_y},{alpha_z})\n"
        f"MSE={mse_val:.6f}  PSNR={psnr_val:.2f} dB",
        fontsize=11
    )

    slices = [
        (vol[mid, :, :],  rec[mid, :, :],  f"Coupe XY (z={mid})"),
        (vol[:, mid, :],  rec[:, mid, :],  f"Coupe XZ (y={mid})"),
        (vol[:, :, mid],  rec[:, :, mid],  f"Coupe YZ (x={mid})"),
    ]

    for col, (vo, re, title) in enumerate(slices):
        # Original
        axes[0, col].imshow(vo, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"Original — {title}", fontsize=9)
        axes[0, col].axis("off")
        # Reconstruit
        axes[1, col].imshow(re, cmap="gray", vmin=re.min(), vmax=re.max())
        axes[1, col].set_title(f"Reconstruit — {title}", fontsize=9)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Original",    fontsize=10)
    axes[1, 0].set_ylabel("Reconstruit", fontsize=10)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"  Coupes : {out_path}")
    plt.close(fig)


def plot_voxel_safe(orig_bin, rec_bin, n_max, alpha_x, alpha_y, alpha_z,
                     dice_val, mse_val, psnr_val, out_path=None,
                     max_voxels=5000):
    """
    Rendu voxel 3 panneaux — version sécurisée.
    Limite le nombre de voxels affichés pour éviter le freeze.
    """
    GRAY = [0.65, 0.65, 0.65, 0.85]
    BLUE = [0.20, 0.40, 0.80, 0.85]
    RED  = [0.85, 0.15, 0.15, 0.85]

    def make_rgba(shape, rgba):
        arr = np.zeros(shape + (4,), dtype=float)
        arr[...] = rgba
        return arr

    def _style_ax(ax):
        ax.set_xlabel("x", fontsize=7)
        ax.set_ylabel("y", fontsize=7)
        ax.set_zlabel("z", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_box_aspect([1, 1, 1])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    # Vérifier si le rendu est faisable
    n_orig = int(orig_bin.sum())
    n_rec  = int(rec_bin.sum())
    total  = n_orig + n_rec

    if total > max_voxels:
        print(f"  [INFO] Trop de voxels ({total}) pour le rendu 3D voxel.")
        print(f"         Utiliser les coupes orthogonales à la place.")
        return

    TP = (orig_bin == 1) & (rec_bin == 1)
    FP = (orig_bin == 0) & (rec_bin == 1)
    FN = (orig_bin == 1) & (rec_bin == 0)

    fig = plt.figure(figsize=(16, 5))
    fig.patch.set_facecolor('white')

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.voxels(orig_bin.astype(bool),
               facecolors=make_rgba(orig_bin.shape, GRAY), edgecolor='none')
    ax1.set_title("Original 3D", fontsize=10, pad=8)
    _style_ax(ax1)

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.voxels(rec_bin.astype(bool),
               facecolors=make_rgba(rec_bin.shape, BLUE), edgecolor='none')
    ax2.set_title(
        f"Reconstruction FrCM\nN={n_max}  α=({alpha_x},{alpha_y},{alpha_z})\n"
        f"DICE={dice_val:.4f}\nMSE={mse_val:.5f}  PSNR={psnr_val:.2f} dB",
        fontsize=8, pad=8
    )
    _style_ax(ax2)

    ax3 = fig.add_subplot(133, projection='3d')
    if TP.any():
        ax3.voxels(TP, facecolors=make_rgba(orig_bin.shape, GRAY), edgecolor='none')
    if FP.any():
        ax3.voxels(FP, facecolors=make_rgba(orig_bin.shape, BLUE), edgecolor='none')
    if FN.any():
        ax3.voxels(FN, facecolors=make_rgba(orig_bin.shape, RED),  edgecolor='none')
    ax3.legend(handles=[
        Patch(facecolor=GRAY[:3], label='TP'),
        Patch(facecolor=BLUE[:3], label='FP'),
        Patch(facecolor=RED[:3],  label='FN'),
    ], loc='upper left', fontsize=7)
    ax3.set_title("Error map", fontsize=10, pad=8)
    _style_ax(ax3)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  Voxel 3D : {out_path}")
    plt.close(fig)


def plot_curves_multi_alpha(n_values, results_by_alpha, out_path=None):
    """Courbes MSE/PSNR comparatives pour plusieurs configurations alpha."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "FrCM 3D — Comparaison configurations alpha\n"
        "(récurrence stable, conforme Gao 2024)",
        fontsize=11
    )

    markers = ['o', 's', '^', 'D']
    for i, (label, (mse_vals, psnr_vals)) in enumerate(results_by_alpha.items()):
        m = markers[i % len(markers)]
        axes[0].semilogy(n_values, mse_vals,  marker=m, label=label)
        axes[1].plot(n_values,    psnr_vals,  marker=m, label=label)

    axes[0].set_xlabel("N")
    axes[0].set_ylabel("MSE (log)")
    axes[0].set_title("MSE vs N")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("N")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("PSNR vs N")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"  Courbes : {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PROGRAMME PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Paramètres ───────────────────────────────────────────────────────────
    IM_PATH     = r"C:\Users\Flora\Downloads\antsIm\antsIm\8.im.gz"  # ← modifier
    VOLUME_SIZE = 32      # 32³ : rapide, N jusqu'à 40
    N_VALUES    = [10, 20, 30, 40]
    THRESHOLD_K = 2.0     # seuil binarisation pour DICE : mean + k*std

    ALPHA_CONFIGS = {
        "α=(1.2,1.0,1.2) [article optimal]": (1.2, 1.0, 1.2),
        "α=(1.0,1.0,1.0) [classique]":       (1.0, 1.0, 1.0),
        "α=(0.8,1.4,1.0) [article config3]":  (0.8, 1.4, 1.0),
        "α=(1.4,1.0,0.8) [article config2]":  (1.4, 1.0, 0.8),
    }

    # ── Sorties ───────────────────────────────────────────────────────────────
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()
    out_dir = script_dir.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = Path(IM_PATH).stem

    # ── Chargement ────────────────────────────────────────────────────────────
    print(f"Chargement : {Path(IM_PATH).name}")
    vol_128 = load_im_gz_raw(IM_PATH)
    print(f"  Shape original : {vol_128.shape}  ({int((vol_128>0).sum())} voxels non-nuls)")

    vol = downsample_volume(vol_128, VOLUME_SIZE)
    print(f"  Shape réduit {VOLUME_SIZE}³ : {vol.shape}")
    print(f"  Min={vol.min():.4f}  Max={vol.max():.4f}  Mean={vol.mean():.4f}")
    print()

    # ── Balayage alpha × N ────────────────────────────────────────────────────
    results_by_alpha = {}
    best_psnr = -9999
    best_info = {}

    for label, (ax, ay, az) in ALPHA_CONFIGS.items():
        print(f"{'─'*60}")
        print(f"Config : {label}")

        mse_list  = []
        psnr_list = []
        last_rec  = None

        for n_max in N_VALUES:
            print(f"  N={n_max:2d}  (N+1)³={(n_max+1)**3:6d} ...", end=" ", flush=True)
            C, Tx, Ty, Tz = frcm_3d_stable(vol, n_max,
                                              alpha_x=ax, alpha_y=ay, alpha_z=az)
            rec = reconstruct_frcm_3d(C, Tx, Ty, Tz)

            m = mse_3d(vol, rec)
            p = psnr_3d(vol, rec)
            mse_list.append(m)
            psnr_list.append(p)
            last_rec = rec
            print(f"MSE={m:.6f}  PSNR={p:.2f} dB")

        # Binarisation pour DICE
        thr     = last_rec.mean() + THRESHOLD_K * last_rec.std()
        rec_bin = (last_rec > thr).astype(np.float64)
        # Binarisation de l'original pour comparaison équitable
        orig_bin = (vol > 0.5).astype(np.float64)
        d, TP, FP, FN = dice_score(orig_bin, rec_bin)
        print(f"  Seuil={thr:.4f}  DICE={d:.4f}  TP={TP}  FP={FP}  FN={FN}")

        results_by_alpha[label] = (mse_list, psnr_list)

        # Meilleure config = PSNR le plus élevé au dernier N
        if psnr_list[-1] > best_psnr:
            best_psnr = psnr_list[-1]
            best_info = {
                "label": label, "ax": ax, "ay": ay, "az": az,
                "n_max": N_VALUES[-1],
                "mse": mse_list[-1], "psnr": psnr_list[-1],
                "rec_raw": last_rec, "rec_bin": rec_bin,
                "orig_bin": orig_bin, "dice": d,
            }
        print()

    # ── Résumé ────────────────────────────────────────────────────────────────
    print(f"{'═'*60}")
    print(f"Meilleure config (PSNR max) : {best_info['label']}")
    print(f"N={best_info['n_max']}  MSE={best_info['mse']:.6f}  PSNR={best_info['psnr']:.2f} dB  DICE={best_info['dice']:.4f}")
    print(f"{'═'*60}")
    print()

    # ── Coupes orthogonales ───────────────────────────────────────────────────
    slices_path = out_dir / f"frcm3d_slices_{stem}_N{best_info['n_max']}_{ts}.png"
    plot_slices_comparison(
        vol, best_info['rec_raw'],
        n_max=best_info['n_max'],
        alpha_x=best_info['ax'], alpha_y=best_info['ay'], alpha_z=best_info['az'],
        mse_val=best_info['mse'], psnr_val=best_info['psnr'],
        out_path=slices_path
    )

    # ── Rendu voxel (seulement si assez peu de voxels) ───────────────────────
    voxel_path = out_dir / f"frcm3d_voxel_{stem}_N{best_info['n_max']}_{ts}.png"
    plot_voxel_safe(
        best_info['orig_bin'], best_info['rec_bin'],
        n_max=best_info['n_max'],
        alpha_x=best_info['ax'], alpha_y=best_info['ay'], alpha_z=best_info['az'],
        dice_val=best_info['dice'],
        mse_val=best_info['mse'], psnr_val=best_info['psnr'],
        out_path=voxel_path,
        max_voxels=8000   # limite de sécurité
    )

    # ── Courbes comparatives ──────────────────────────────────────────────────
    curves_path = out_dir / f"frcm3d_curves_{stem}_{ts}.png"
    plot_curves_multi_alpha(N_VALUES, results_by_alpha, out_path=curves_path)

    # ── Tableau récapitulatif ─────────────────────────────────────────────────
    print("=" * 65)
    print(f"{'Config':40s}  {'N':>4}  {'MSE':>12}  {'PSNR':>8}")
    print("-" * 65)
    for label, (mse_vals, psnr_vals) in results_by_alpha.items():
        short = label.split("[")[0].strip()
        for n, m, p in zip(N_VALUES, mse_vals, psnr_vals):
            print(f"  {short:<38}  {n:>4}  {m:>12.6f}  {p:>8.2f} dB")
        print()
    print("=" * 65)
    print(f"Fichiers dans : {out_dir}")
