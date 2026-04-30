from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift, rotate, zoom
from pathlib import Path
import os

from moments_2d import chebyshev_moments_2d


def load_binary_image(path):
    img_pil = Image.open(path).convert("L")
    arr = np.array(img_pil, dtype=np.float64)
    return (arr > 127).astype(np.float64)


def flatten_moments(img, n_max):
    M, _, _ = chebyshev_moments_2d(img, n_max)
    return M.flatten()


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


dataset_root = resolve_dataset_root()
path = find_existing_image(dataset_root, class_name="heart", sample_no=1)
img = load_binary_image(path)

n_max = 20
v0 = flatten_moments(img, n_max)

tests = {
    "translation_x10": shift(img, shift=(0, 10), mode="nearest"),
    "translation_y10": shift(img, shift=(10, 0), mode="nearest"),
    "rotation_10": rotate(img, angle=10, reshape=False, mode="nearest"),
    "rotation_20": rotate(img, angle=20, reshape=False, mode="nearest"),
    "scale_0.8": scale_image(img, 0.8),
    "scale_1.2": scale_image(img, 1.2),
}

print("=== Invariance / stabilité test ===")
for name, img_t in tests.items():
    vt = flatten_moments(img_t, n_max)
    re = relative_error(v0, vt)
    cs = cosine_similarity(v0, vt)

    print(name)
    print("  relative error:", re)
    print("  cosine similarity:", cs)
    print()