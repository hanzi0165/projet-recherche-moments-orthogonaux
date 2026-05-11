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

    # Normalisation min-max pour chaque matrice de moments
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
    Pour chaque muscle : 3 coupes (z_min, z_mid, z_max) de la matrice de moments.
    3 lignes × n_muscles colonnes.
    """
    names = list(moments_dict.keys())
    n     = len(names)
    
    # 3 coupes : n=0 (bas), n=N_MAX//2 (milieu), n=N_MAX (haut)
    slices_idx = [0, N_MAX // 2, N_MAX]
    slice_labels = ["n=0 (bas)", f"n={N_MAX//2} (milieu)", f"n={N_MAX} (haut)"]

    fig, axes = plt.subplots(3, n, figsize=(4 * n, 10))
    fig.suptitle(
        f"Matrices de moments FrCM 3D — 3 coupes\n"
        f"N={N_MAX}  α={ALPHA}",
        fontsize=11
    )

    for row, (idx, slabel) in enumerate(zip(slices_idx, slice_labels)):
        for col, name in enumerate(names):
            C  = moments_dict[name]
            sl = C[idx, :, :]
            im = axes[row, col].imshow(np.abs(sl), cmap='viridis', aspect='auto')
            if row == 0:
                axes[row, col].set_title(name, fontsize=8)
            axes[row, col].set_xlabel("p", fontsize=7)
            axes[row, col].set_ylabel(f"q\n{slabel}", fontsize=7)
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)

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

    
    # ── Figure 3D voxel pour chaque muscle ───────────────────────────────────
    GRAY = [0.65, 0.65, 0.65, 0.85]
    COLORS = [
        [0.85, 0.45, 0.45, 0.85],  # rose
        [0.30, 0.65, 0.30, 0.85],  # vert
        [0.25, 0.45, 0.85, 0.85],  # bleu
        [0.60, 0.40, 0.20, 0.85],  # marron
        [0.85, 0.75, 0.20, 0.85],  # jaune
    ]
    names_list = list(muscles.keys())
    n_muscles  = len(names_list)

    fig3d = plt.figure(figsize=(4 * n_muscles, 5))
    fig3d.suptitle("Volumes 3D des muscles segmentés (32³)", fontsize=11)

    for i, name in enumerate(names_list):
        ax = fig3d.add_subplot(1, n_muscles, i + 1, projection='3d')
        vol_bin = muscles[name]['bin']
        color   = COLORS[i % len(COLORS)]
        arr = np.zeros(vol_bin.shape + (4,), dtype=float)
        arr[...] = color
        ax.voxels(vol_bin.astype(bool), facecolors=arr, edgecolor='none')
        ax.set_title(name, fontsize=8)
        ax.set_box_aspect([1, 1, 1])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.tick_params(labelsize=5)

    plt.tight_layout()
    voxel3d_path = out_dir / f"muscles_voxel3d_{ts}.png"
    fig3d.savefig(voxel3d_path, dpi=150, bbox_inches='tight')
    plt.close(fig3d)
    print(f"  Muscles 3D : {voxel3d_path}")

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