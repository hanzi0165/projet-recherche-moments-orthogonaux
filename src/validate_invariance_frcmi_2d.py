from PIL import Image
import numpy as np
from scipy.ndimage import shift, rotate, zoom
from pathlib import Path
import os
import csv
    
from frcmi_2d import frcmi_2d           


def load_binary_image(path):
    img_pil = Image.open(path).convert("L")
    arr = np.array(img_pil, dtype=np.float64)
    return (arr > 127).astype(np.float64)



def relative_error(v1, v2, eps=1e-12):
    return np.linalg.norm(v1 - v2) / (np.linalg.norm(v1) + eps)


def cosine_similarity(v1, v2, eps=1e-12):
    return np.dot(v1, v2) / ((np.linalg.norm(v1) * np.linalg.norm(v2)) + eps)


def center_crop_or_pad(img, target_shape):
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
    scaled = zoom(img, factor, order=0)
    return center_crop_or_pad(scaled, img.shape)


def resolve_dataset_root() -> Path:
    env_dir = os.environ.get("MPEG7_ORIGINAL_DIR")

    if "__file__" in globals():
        script_dir = Path(__file__).resolve().parent
    else:
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

    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidate_text = "\n".join(f"- {str(c)}" for c in candidates)
    raise FileNotFoundError(
        "Impossible de trouver le dossier MPEG-7 'original'.\n"
        "Définissez MPEG7_ORIGINAL_DIR ou placez les données dans l'un des chemins suivants:\n"
        f"{candidate_text}"
    )


def find_existing_image(root: Path, class_name: str, sample_no: int) -> Path:
    candidates = [
        root / f"{class_name}-{sample_no:02d}.gif",
        root / f"{class_name}-{sample_no}.gif",
        root / f"{class_name.capitalize()}-{sample_no:02d}.gif",
        root / f"{class_name.capitalize()}-{sample_no}.gif",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidate_text = "\n".join(f"- {str(c)}" for c in candidates)
    raise FileNotFoundError(
        f"Aucun fichier trouvé pour {class_name} (échantillon {sample_no}).\n"
        f"Chemins testés:\n{candidate_text}"
    )


def evaluate_alpha(img, tests, n_max, alpha):
    """
    Évalue un alpha donné sur tous les tests de transformation.

    Retourne:
      - rows: lignes détaillées pour CSV (une ligne par transformation)
      - mean_re: moyenne des erreurs relatives
      - mean_cs: moyenne des similarités cosinus
    """
    v0 = frcmi_2d(img, n_max, alpha_x=alpha, alpha_y=alpha)

    rows = []
    rel_errors = []
    cos_sims = []

    for name, img_t in tests.items():
        vt = frcmi_2d(img_t, n_max, alpha_x=alpha, alpha_y=alpha)
        re = float(relative_error(v0, vt))
        cs = float(cosine_similarity(v0, vt))

        rel_errors.append(re)
        cos_sims.append(cs)
        rows.append({
            "alpha": alpha,
            "transform": name,
            "relative_error": re,
            "cosine_similarity": cs,
        })

    mean_re = float(np.mean(rel_errors)) if rel_errors else float("nan")
    mean_cs = float(np.mean(cos_sims)) if cos_sims else float("nan")
    return rows, mean_re, mean_cs


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =========================
# Paramètres
# =========================
dataset_root = resolve_dataset_root()
path = find_existing_image(dataset_root, class_name="heart", sample_no=1)
img = load_binary_image(path)

n_max = 20

# Balayage alpha (tu peux modifier cette liste selon ton besoin)
alpha_values = [float(a) for a in np.round(np.arange(0.6, 2.05, 0.1), 2)]

tests = {
    "translation_x10": shift(img, shift=(0, 10), mode="nearest"),
    "translation_y10": shift(img, shift=(10, 0), mode="nearest"),
    "rotation_10": rotate(img, angle=10, reshape=False, mode="nearest"),
    "rotation_20": rotate(img, angle=20, reshape=False, mode="nearest"),
    "scale_0.8": scale_image(img, 0.8),
    "scale_1.2": scale_image(img, 1.2),
}

all_detail_rows = []
summary_rows = []

print("=== Fractional 2D Invariance / stabilité test (alpha sweep) ===")
print(f"n_max = {n_max}")
print(f"alpha values = {list(alpha_values)}")
print()

for alpha in alpha_values:
    rows, mean_re, mean_cs = evaluate_alpha(img, tests, n_max, alpha)
    all_detail_rows.extend(rows)
    summary_rows.append({
        "alpha": alpha,
        "mean_relative_error": mean_re,
        "mean_cosine_similarity": mean_cs,
    })

    print(f"alpha = {alpha:.2f} -> mean RE = {mean_re:.6f}, mean CS = {mean_cs:.6f}")

summary_sorted = sorted(
    summary_rows,
    key=lambda r: (r["mean_relative_error"], -r["mean_cosine_similarity"]),
)
best = summary_sorted[0]

script_dir = Path(__file__).resolve().parent
output_dir = script_dir.parent / "outputs"
detail_csv = output_dir / "invariance_frcmi_alpha_detail.csv"
summary_csv = output_dir / "invariance_frcmi_alpha_summary.csv"

write_csv(
    detail_csv,
    ["alpha", "transform", "relative_error", "cosine_similarity"],
    all_detail_rows,
)
write_csv(
    summary_csv,
    ["alpha", "mean_relative_error", "mean_cosine_similarity"],
    summary_rows,
)

print()
print("Best alpha (min mean_relative_error, tie-break by max mean_cosine_similarity):")
print(
    f"  alpha = {best['alpha']:.2f}, "
    f"mean RE = {best['mean_relative_error']:.6f}, "
    f"mean CS = {best['mean_cosine_similarity']:.6f}"
)
print()
print(f"Detailed CSV: {detail_csv}")
print(f"Summary CSV:  {summary_csv}")
