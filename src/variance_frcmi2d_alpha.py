from PIL import Image
import numpy as np
from scipy.ndimage import shift, rotate, zoom
from pathlib import Path
import os
import csv

from frcmi_2d import frcmi_2d


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des images
# ─────────────────────────────────────────────────────────────────────────────

def load_binary_image(path):
    """Charge une image en niveaux de gris et la binarise au seuil 127."""
    img_pil = Image.open(path).convert("L")
    arr = np.array(img_pil, dtype=np.float64)
    return (arr > 127).astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Métriques de comparaison
# ─────────────────────────────────────────────────────────────────────────────

def relative_error(v1, v2, eps=1e-12):
    """Erreur relative entre deux vecteurs de moments."""
    return np.linalg.norm(v1 - v2) / (np.linalg.norm(v1) + eps)


def cosine_similarity(v1, v2, eps=1e-12):
    """Similarité cosinus entre deux vecteurs de moments."""
    return np.dot(v1, v2) / ((np.linalg.norm(v1) * np.linalg.norm(v2)) + eps)


# ─────────────────────────────────────────────────────────────────────────────
# Transformations géométriques
# ─────────────────────────────────────────────────────────────────────────────

def center_crop_or_pad(img, target_shape):
    """Recadre ou complète l'image pour atteindre la taille cible."""
    H, W = target_shape
    out = np.zeros((H, W), dtype=img.dtype)
    h, w = img.shape
    y0 = max((H - h) // 2, 0)
    x0 = max((W - w) // 2, 0)
    ys = max((h - H) // 2, 0)
    xs = max((w - W) // 2, 0)
    copy_h = min(H, h)
    copy_w = min(W, w)
    out[y0:y0+copy_h, x0:x0+copy_w] = img[ys:ys+copy_h, xs:xs+copy_w]
    return out


def scale_image(img, factor):
    """Redimensionne l'image par un facteur donné, puis recentre."""
    scaled = zoom(img, factor, order=0)
    return center_crop_or_pad(scaled, img.shape)


def build_tests(img):
    """
    Construit le dictionnaire des transformations de test.
    Les clés commencent par 'translation_', 'rotation_' ou 'scale_'
    pour permettre le regroupement par catégorie dans l'évaluation.
    """
    return {
        "translation_x10": shift(img, shift=(0, 10),  mode="nearest"),
        "translation_y10": shift(img, shift=(10, 0),  mode="nearest"),
        "rotation_10":     rotate(img, angle=10, reshape=False, mode="nearest"),
        "rotation_20":     rotate(img, angle=20, reshape=False, mode="nearest"),
        "scale_0.8":       scale_image(img, 0.8),
        "scale_1.2":       scale_image(img, 1.2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Évaluation pour un alpha et une image donnés
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_alpha(img, tests, n_max, alpha, class_name):
    """
    Calcule les métriques d'invariance pour un alpha donné sur une image.

    Pour chaque transformation :
      - on calcule le vecteur FrCMI de l'image originale (v0)
        et de l'image transformée (vt)
      - on mesure l'erreur relative et la similarité cosinus

    Les résultats sont ensuite moyennés par catégorie
    (translation / rotation / scale) pour identifier
    quelle transformation est la plus sensible au choix d'alpha.

    Retourne :
      rows    — lignes détaillées pour le CSV (une par transformation)
      summary — dict avec moyennes globales et par catégorie
    """
    v0 = frcmi_2d(img, n_max, alpha_x=alpha, alpha_y=alpha)

    rows = []
    # Accumulateurs par catégorie de transformation
    category_errors = {"translation": [], "rotation": [], "scale": []}
    category_cs     = {"translation": [], "rotation": [], "scale": []}

    for name, img_t in tests.items():
        vt = frcmi_2d(img_t, n_max, alpha_x=alpha, alpha_y=alpha)
        re = float(relative_error(v0, vt))
        cs = float(cosine_similarity(v0, vt))

        # Déterminer la catégorie à partir du préfixe de la clé
        if name.startswith("translation"):
            cat = "translation"
        elif name.startswith("rotation"):
            cat = "rotation"
        else:
            cat = "scale"

        category_errors[cat].append(re)
        category_cs[cat].append(cs)

        rows.append({
            "class":             class_name,
            "alpha":             alpha,
            "transform":         name,
            "category":          cat,
            "relative_error":    re,
            "cosine_similarity": cs,
        })

    # Moyennes globales sur toutes les transformations
    all_re = [r["relative_error"]    for r in rows]
    all_cs = [r["cosine_similarity"] for r in rows]

    summary = {
        "class":                  class_name,
        "alpha":                  alpha,
        "mean_relative_error":    float(np.mean(all_re)),
        "mean_cosine_similarity": float(np.mean(all_cs)),
        # Moyennes par catégorie — permettent de voir si alpha
        # améliore davantage la rotation, la translation ou l'échelle
        "mean_RE_translation":    float(np.mean(category_errors["translation"])),
        "mean_RE_rotation":       float(np.mean(category_errors["rotation"])),
        "mean_RE_scale":          float(np.mean(category_errors["scale"])),
        "mean_CS_translation":    float(np.mean(category_cs["translation"])),
        "mean_CS_rotation":       float(np.mean(category_cs["rotation"])),
        "mean_CS_scale":          float(np.mean(category_cs["scale"])),
    }

    return rows, summary


# ─────────────────────────────────────────────────────────────────────────────
# Écriture CSV
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(path, fieldnames, rows):
    """Écrit une liste de dictionnaires dans un fichier CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Recherche du chemin des données
# ─────────────────────────────────────────────────────────────────────────────

def resolve_dataset_root():
    """Cherche le dossier 'original' du dataset MPEG-7 dans plusieurs emplacements."""
    env_dir = os.environ.get("MPEG7_ORIGINAL_DIR")
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()
    project_root = script_dir.parent
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend([
        project_root / "MPEG7dataset" / "original",
        project_root / "data" / "MPEG7dataset" / "original",
        Path.home() / "Downloads" / "MPEG7dataset" / "original",
    ])
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Dossier MPEG-7 introuvable.\n"
        "Définissez MPEG7_ORIGINAL_DIR ou placez les données dans :\n"
        + "\n".join(f"  - {c}" for c in candidates)
    )


def find_existing_image(root, class_name, sample_no):
    """Cherche un fichier .gif correspondant à la classe et au numéro d'échantillon."""
    for fmt in [
        f"{class_name}-{sample_no:02d}.gif",
        f"{class_name}-{sample_no}.gif",
        f"{class_name.capitalize()}-{sample_no:02d}.gif",
        f"{class_name.capitalize()}-{sample_no}.gif",
    ]:
        p = root / fmt
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Image introuvable pour '{class_name}' échantillon {sample_no} dans {root}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Programme principal
# ─────────────────────────────────────────────────────────────────────────────

# Ordre maximal des moments de Chebyshev fractionnaires
N_MAX = 20

# Valeurs d'alpha à tester (de 0.6 à 2.0 par pas de 0.1)
ALPHA_VALUES = [float(a) for a in np.round(np.arange(0.6, 2.05, 0.1), 2)]

# Classes testées : plusieurs images pour obtenir un alpha plus robuste.
# Tester sur une seule image risque de biaiser le résultat vers
# les caractéristiques géométriques propres à cette forme.
# Chaque tuple : (nom_de_classe, numéro_d_échantillon)
CLASS_SAMPLES = [
    ("heart",     1),
    ("apple",     1),
    ("bird",      1),
    ("elephant",  1),
    ("butterfly", 1),
]

dataset_root = resolve_dataset_root()

all_detail_rows  = []   # toutes les lignes détaillées
all_summary_rows = []   # une ligne par (classe, alpha)

print("=== Balayage alpha — Invariance FrCMI 2D ===")
print(f"n_max        = {N_MAX}")
print(f"alpha values = {ALPHA_VALUES}")
print(f"classes      = {[c for c, _ in CLASS_SAMPLES]}")
print()

for class_name, sample_no in CLASS_SAMPLES:
    try:
        path = find_existing_image(dataset_root, class_name, sample_no)
        img  = load_binary_image(path)
    except FileNotFoundError as e:
        print(f"  [IGNORÉ] {e}")
        continue

    # Construire les images transformées une seule fois par classe
    tests = build_tests(img)
    print(f"--- Classe : {class_name} ---")

    for alpha in ALPHA_VALUES:
        rows, summary = evaluate_alpha(img, tests, N_MAX, alpha, class_name)
        all_detail_rows.extend(rows)
        all_summary_rows.append(summary)

        # Affichage compact : erreur globale + détail par catégorie
        print(
            f"  alpha={alpha:.2f}  "
            f"RE_global={summary['mean_relative_error']:.4f}  "
            f"RE_trans={summary['mean_RE_translation']:.4f}  "
            f"RE_rot={summary['mean_RE_rotation']:.4f}  "
            f"RE_scale={summary['mean_RE_scale']:.4f}"
        )
    print()

# ── Agrégation : moyenne de chaque alpha sur toutes les classes ───────────────
# On regroupe les résumés par valeur d'alpha et on fait la moyenne,
# ce qui donne une estimation plus robuste du meilleur alpha.

alpha_groups = {}
for row in all_summary_rows:
    a = row["alpha"]
    alpha_groups.setdefault(a, []).append(row)

aggregated_rows = []
for alpha, group in sorted(alpha_groups.items()):
    aggregated_rows.append({
        "alpha":               alpha,
        "mean_RE_global":      float(np.mean([r["mean_relative_error"]    for r in group])),
        "mean_CS_global":      float(np.mean([r["mean_cosine_similarity"] for r in group])),
        "mean_RE_translation": float(np.mean([r["mean_RE_translation"]    for r in group])),
        "mean_RE_rotation":    float(np.mean([r["mean_RE_rotation"]       for r in group])),
        "mean_RE_scale":       float(np.mean([r["mean_RE_scale"]          for r in group])),
    })

# Meilleur alpha : erreur globale minimale, à égalité cosinus maximal
best = min(aggregated_rows,
           key=lambda r: (r["mean_RE_global"], -r["mean_CS_global"]))

# ── Écriture des trois fichiers CSV ──────────────────────────────────────────

try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path.cwd()

output_dir = script_dir.parent / "outputs"

# CSV 1 : détail complet (une ligne par transformation par alpha par classe)
detail_csv = output_dir / "invariance_frcmi_alpha_detail.csv"
write_csv(
    detail_csv,
    ["class", "alpha", "transform", "category",
     "relative_error", "cosine_similarity"],
    all_detail_rows,
)

# CSV 2 : résumé par classe et par alpha (avec détail par catégorie)
summary_csv = output_dir / "invariance_frcmi_alpha_summary.csv"
write_csv(
    summary_csv,
    ["class", "alpha",
     "mean_relative_error", "mean_cosine_similarity",
     "mean_RE_translation", "mean_RE_rotation", "mean_RE_scale",
     "mean_CS_translation", "mean_CS_rotation", "mean_CS_scale"],
    all_summary_rows,
)

# CSV 3 : agrégé sur toutes les classes (une ligne par alpha)
# C'est ce fichier qui sert à choisir le meilleur alpha global.
aggregated_csv = output_dir / "invariance_frcmi_alpha_aggregated.csv"
write_csv(
    aggregated_csv,
    ["alpha", "mean_RE_global", "mean_CS_global",
     "mean_RE_translation", "mean_RE_rotation", "mean_RE_scale"],
    aggregated_rows,
)

# ── Résumé final affiché dans le terminal ────────────────────────────────────

print("=" * 60)
print("Meilleur alpha (moyenné sur toutes les classes) :")
print(f"  alpha              = {best['alpha']:.2f}")
print(f"  mean RE global     = {best['mean_RE_global']:.6f}")
print(f"  mean CS global     = {best['mean_CS_global']:.6f}")
print(f"  mean RE translation= {best['mean_RE_translation']:.6f}")
print(f"  mean RE rotation   = {best['mean_RE_rotation']:.6f}")
print(f"  mean RE scale      = {best['mean_RE_scale']:.6f}")
print()
print("Fichiers générés :")
print(f"  Détail     : {detail_csv}")
print(f"  Par classe : {summary_csv}")
print(f"  Agrégé     : {aggregated_csv}")