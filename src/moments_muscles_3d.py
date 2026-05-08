"""
moments_muscles_3d.py

Calcule les moments de Chebyshev fractionnaires 3D (FrCM)
pour chaque muscle segmenté dans l'IRM d'épaule.

Données :
  epaule_002_0000.nii.gz  → volume MRI (384×384×180)
  epaule_002_nii.gz       → segmentation (5 muscles, labels 1-5)

Pipeline :
  1. Lecture des fichiers .nii.gz
  2. Pour chaque muscle : extraction du volume binaire
  3. Calcul des moments FrCM 3D
  4. Comparaison des matrices de moments entre muscles
  5. Visualisation (heatmaps + coupes)
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
from scipy.ndimage import zoom as nd_zoom


# ═══════════════════════════════════════════════════════════════════════════════
# 0. PARAMÈTRES
# ═══════════════════════════════════════════════════════════════════════════════

MRI_PATH = r"C:\Users\Flora\Downloads\shoulder_TG\epaule_002_0000.nii.gz"   
SEG_PATH = r"C:\Users\Flora\Downloads\shoulder_TG\epaule_002.nii.gz"         

# Noms des muscles (labels 1-5, selon Slicer3D)
MUSCLE_NAMES = {
    1: "Muscle_1 (rose)",
    2: "Muscle_2 (vert)",
    3: "Muscle_3 (bleu)",
    4: "Muscle_4 (marron)",
    5: "Muscle_5 (jaune)",
}

N_MAX       = 10     # ordre des moments
ALPHA       = 1.0    # paramètre fractionnaire
TARGET_SIZE = 32     # réduire à 32³ pour accélérer le calcul


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def load_nii(path: str):
    """Charge un fichier .nii.gz et retourne les données numpy."""
    nii = nib.load(path)
    return nii.get_fdata()


def extract_muscle(vol_mri, seg_mask, label: int, target_size: int):
    """
    Extrait le volume d'un muscle spécifique.
    - Crée un masque binaire pour ce muscle
    - Coupe le bounding box autour du muscle
    - Réduit à target_size³
    Retourne (muscle_bin, muscle_gray) :
      muscle_bin  : volume binaire {0,1}
      muscle_gray : intensités MRI masquées
    """
    # Masque binaire du muscle
    binary = (seg_mask == label).astype(np.float64)

    # Bounding box
    coords = np.argwhere(binary > 0)
    if len(coords) == 0:
        return None, None

    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1

    muscle_bin  = binary[z0:z1, y0:y1, x0:x1]
    muscle_gray = vol_mri[z0:z1, y0:y1, x0:x1] * muscle_bin

    # Normalisation des intensités MRI
    vmax = muscle_gray.max()
    if vmax > 0:
        muscle_gray /= vmax

    # Réduction
    def resize(vol, t):
        factors = [t / s for s in vol.shape]
        return np.clip(nd_zoom(vol, factors, order=1), 0, 1)

    muscle_bin_r  = (resize(muscle_bin,  target_size) > 0.5).astype(np.float64)
    muscle_gray_r = resize(muscle_gray, target_size)

    return muscle_bin_r, muscle_gray_r


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BASE DE CHEBYSHEV FRACTIONNAIRE (récurrence stable)
# ═══════════════════════════════════════════════════════════════════════════════

def fractional_chebyshev_basis(length: int, n_max: int, alpha: float):
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
# 3. CALCUL DES MOMENTS FrCM 3D
# ═══════════════════════════════════════════════════════════════════════════════

def frcm_3d(vol: np.ndarray, n_max: int, alpha: float = 1.0):
    """
    Calcule la matrice de moments FrCM 3D.
    Retourne C shape (n_max+1, n_max+1, n_max+1).
    """
    D, H, W = vol.shape

    Tx, dx = fractional_chebyshev_basis(W, n_max, alpha)
    Ty, dy = fractional_chebyshev_basis(H, n_max, alpha)
    Tz, dz = fractional_chebyshev_basis(D, n_max, alpha)

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

    return C


# ═══════════════════════════════════════════════════════════════════════════════
# 4. COMPARAISON DES MATRICES DE MOMENTS
# ═══════════════════════════════════════════════════════════════════════════════

def compare_moments(moments_dict: dict):
    names = list(moments_dict.keys())
    n     = len(names)
    dist  = np.zeros((n, n), dtype=np.float64)

    # 先归一化每个矩矩阵到[0,1]
    normalized = {}
    for name, C in moments_dict.items():
        cmin = C.min()
        cmax = C.max()
        if cmax - cmin > 1e-12:
            normalized[name] = (C - cmin) / (cmax - cmin)
        else:
            normalized[name] = C.copy()

    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            Mi = normalized[ni].flatten()
            Mj = normalized[nj].flatten()
            dist[i, j] = float(np.mean((Mi - Mj) ** 2))

    return dist, names

# ═══════════════════════════════════════════════════════════════════════════════
# 5. VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_moment_slices(moments_dict: dict, out_path=None):
    """
    Pour chaque muscle, affiche une coupe centrale de la matrice de moments.
    La coupe centrale C[n_max//2, :, :] montre la distribution 2D des moments.
    """
    names  = list(moments_dict.keys())
    n      = len(names)
    mid    = N_MAX // 2

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.suptitle(
        f"Matrices de moments FrCM 3D — Coupe centrale (n={mid})\n"
        f"N={N_MAX}  α={ALPHA}",
        fontsize=11
    )

    for i, name in enumerate(names):
        C    = moments_dict[name]
        sl   = C[mid, :, :]
        im   = axes[i].imshow(np.abs(sl), cmap='viridis', aspect='auto')
        axes[i].set_title(name, fontsize=8)
        axes[i].set_xlabel("p")
        axes[i].set_ylabel("q")
        plt.colorbar(im, ax=axes[i], fraction=0.046)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"  Coupes moments : {out_path}")
    plt.close(fig)


def plot_distance_matrix(dist: np.ndarray, names: list, out_path=None):
    """
    Heatmap de la matrice de distances MSE entre les muscles.
    Diagonal = 0 (muscle comparé à lui-même).
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(dist, cmap='YlOrRd')
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title(
        f"Distance MSE entre matrices de moments\nN={N_MAX}  α={ALPHA}",
        fontsize=11
    )

    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{dist[i,j]:.2e}",
                    ha='center', va='center', fontsize=7,
                    color='white' if dist[i,j] > dist.max()/2 else 'black')

    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"  Matrice distances : {out_path}")
    plt.close(fig)


def plot_muscle_slices(muscles: dict, out_path=None):
    """
    Affiche une coupe centrale de chaque muscle (volume binaire).
    """
    names = list(muscles.keys())
    n     = len(names)
    mid   = TARGET_SIZE // 2

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.suptitle("Coupes centrales des muscles (volume binaire 32³)", fontsize=11)

    for i, name in enumerate(names):
        vol_bin = muscles[name]['bin']
        axes[i].imshow(vol_bin[mid, :, :], cmap='gray')
        axes[i].set_title(name, fontsize=8)
        axes[i].axis('off')

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"  Coupes muscles : {out_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PROGRAMME PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Sorties
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()
    out_dir = script_dir.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── 1. Chargement ─────────────────────────────────────────────────────────
    print("Chargement des fichiers NIfTI...")
    vol_mri  = load_nii(MRI_PATH)
    seg_mask = load_nii(SEG_PATH).astype(np.int32)
    labels   = np.unique(seg_mask)
    labels   = labels[labels > 0]
    print(f"  MRI shape    : {vol_mri.shape}")
    print(f"  Seg shape    : {seg_mask.shape}")
    print(f"  Nb muscles   : {len(labels)}  labels={labels}")
    print()

    # ── 2. Extraction + calcul moments ────────────────────────────────────────
    muscles       = {}   # {nom: {bin, gray}}
    moments_dict  = {}   # {nom: C (matrice moments)}

    for label in labels:
        name = MUSCLE_NAMES.get(label, f"Muscle_{label}")
        print(f"Traitement : {name} (label={label})")

        bin_vol, gray_vol = extract_muscle(
            vol_mri, seg_mask, label, TARGET_SIZE
        )

        if bin_vol is None:
            print(f"  [SKIP] Muscle vide")
            continue

        n_vox = int(bin_vol.sum())
        print(f"  Voxels actifs : {n_vox} / {TARGET_SIZE**3}")

        muscles[name] = {'bin': bin_vol, 'gray': gray_vol}

        # Calcul des moments sur le volume binaire
        print(f"  Calcul FrCM (N={N_MAX}, α={ALPHA})...")
        C = frcm_3d(bin_vol, N_MAX, alpha=ALPHA)
        moments_dict[name] = C

        print(f"  Moments shape : {C.shape}")
        print(f"  |C| max={np.abs(C).max():.4f}  mean={np.abs(C).mean():.6f}")
        print()

    # ── 3. Comparaison des matrices ───────────────────────────────────────────
    print("Comparaison des matrices de moments...")
    dist_matrix, names = compare_moments(moments_dict)

    print("\nMatrice de distances MSE :")
    print(f"{'':25s}", end="")
    for n in names:
        print(f"{n[:12]:>14s}", end="")
    print()
    for i, ni in enumerate(names):
        print(f"{ni[:25]:25s}", end="")
        for j in range(len(names)):
            print(f"{dist_matrix[i,j]:14.4e}", end="")
        print()

    # ── 4. Figures ────────────────────────────────────────────────────────────
    print("\nGénération des figures...")

    plot_muscle_slices(
        muscles,
        out_path=out_dir / f"muscles_slices_{ts}.png"
    )

    plot_moment_slices(
        moments_dict,
        out_path=out_dir / f"moments_slices_{ts}.png"
    )

    plot_distance_matrix(
        dist_matrix, names,
        out_path=out_dir / f"moments_distances_{ts}.png"
    )

    # ── 5. Résumé ─────────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  Résumé — Moments FrCM 3D par muscle")
    print("-" * 55)
    print(f"  N_max  = {N_MAX}  →  (N+1)³ = {(N_MAX+1)**3} moments")
    print(f"  Alpha  = {ALPHA}")
    print(f"  Taille = {TARGET_SIZE}³ voxels")
    print("-" * 55)
    for name, C in moments_dict.items():
        print(f"  {name[:30]:30s}  |C|max={np.abs(C).max():.4f}")
    print("=" * 55)
    print(f"\nFichiers dans : {out_dir}")