import numpy as np
from scipy.ndimage import shift, rotate, zoom
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
    print("[WARN] SimpleITK 未安装，使用合成测试数据")

from fractional_3d import frcm_3d, normalized_coords_01


def load_3d_image_sitk(path):
    """使用 SimpleITK 加载 3D 医学图像"""
    if not SITK_AVAILABLE:
        raise RuntimeError("SimpleITK 未安装")
    img_itk = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img_itk)
    # 归一化到 [0, 1]
    arr = arr.astype(np.float64)
    arr_min, arr_max = np.min(arr), np.max(arr)
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    return arr


def create_synthetic_3d_volume():
    """创建合成 3D 测试体积（球体）"""
    size = 64
    vol = np.zeros((size, size, size), dtype=np.float64)
    center = size // 2
    radius = 15
    
    for i in range(size):
        for j in range(size):
            for k in range(size):
                dist = np.sqrt((i - center)**2 + (j - center)**2 + (k - center)**2)
                if dist <= radius:
                    vol[i, j, k] = 1.0
    return vol


def flatten_moments(vol, n_max, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0):
    M, _, _, _ = frcm_3d(vol, n_max, alpha_x=alpha_x, alpha_y=alpha_y, alpha_z=alpha_z)
    return M.flatten()


def relative_error(v1, v2, eps=1e-12):
    return np.linalg.norm(v1 - v2) / (np.linalg.norm(v1) + eps)


def cosine_similarity(v1, v2, eps=1e-12):
    return np.dot(v1, v2) / ((np.linalg.norm(v1) * np.linalg.norm(v2)) + eps)


def center_crop_or_pad_3d(vol, target_shape):
    """3D 版本的中心裁剪或填充"""
    H, W, D = target_shape
    out = np.zeros((H, W, D), dtype=vol.dtype)

    h, w, d = vol.shape
    y0 = max((H - h) // 2, 0)
    x0 = max((W - w) // 2, 0)
    z0 = max((D - d) // 2, 0)

    ys = max((h - H) // 2, 0)
    xs = max((w - W) // 2, 0)
    zs = max((d - D) // 2, 0)

    copy_h = min(H, h)
    copy_w = min(W, w)
    copy_d = min(D, d)

    out[y0:y0+copy_h, x0:x0+copy_w, z0:z0+copy_d] = \
        vol[ys:ys+copy_h, xs:xs+copy_w, zs:zs+copy_d]
    return out


def scale_3d_volume(vol, factor):
    """对 3D 体积进行缩放"""
    scaled = zoom(vol, factor, order=0)
    return center_crop_or_pad_3d(scaled, vol.shape)


# ========== 参数配置 ==========
n_max = 15  # 3D 通常用较小的 n_max
alpha_x = 1.0  # 可改
alpha_y = 1.0  # 可改
alpha_z = 1.0  # 可改

# ========== 加载或创建测试数据 ==========
print("=" * 70)
print("Fractional 3D Chebyshev Moments - 稳定性测试")
print("=" * 70)
print()

# 尝试加载真实 3D 医学图像
real_data_path = r"C:\Users\Flora\Downloads\camus_32_dlss21_ho4\camus_32\patient0162\patient0162_2CH_ED.nii.gz"

if SITK_AVAILABLE and Path(real_data_path).exists():
    try:
        vol = load_3d_image_sitk(real_data_path)
        print(f"[OK] 已从 {real_data_path} 加载真实医学图像")
        print(f"     体积形状: {vol.shape}")
        # 如果是 2D 数据，转换为 3D
        if len(vol.shape) == 2:
            print("     (2D 图像，沿第三维重复以创建 3D)")
            vol = np.stack([vol] * 32, axis=0)
    except Exception as e:
        print(f"[WARN] 加载失败: {e}")
        print("     使用合成数据")
        vol = create_synthetic_3d_volume()
else:
    print("[INFO] 使用合成 3D 测试体积（球体）")
    vol = create_synthetic_3d_volume()

print(f"测试体积形状: {vol.shape}")
print(f"参数: n_max={n_max}, α_x={alpha_x}, α_y={alpha_y}, α_z={alpha_z}")
print()

v0 = flatten_moments(vol, n_max, alpha_x=alpha_x, alpha_y=alpha_y, alpha_z=alpha_z)

# ========== Invariant 测试 ==========
tests = {
    "translation_z5": shift(vol, shift=(5, 0, 0), mode="nearest"),
    "translation_x5": shift(vol, shift=(0, 5, 0), mode="nearest"),
    "translation_y5": shift(vol, shift=(0, 0, 5), mode="nearest"),
    "rotation_z10": rotate(vol, angle=10, axes=(0, 1), reshape=False, mode="nearest"),
    "rotation_x10": rotate(vol, angle=10, axes=(1, 2), reshape=False, mode="nearest"),
    "scale_0.9": scale_3d_volume(vol, 0.9),
    "scale_1.1": scale_3d_volume(vol, 1.1),
}

print("=" * 80)
print(f"稳定性测试结果 (n_max={n_max})")
print("=" * 80)
print(f"{'变换类型':<25} | {'相对误差':>12} | {'余弦相似度':>12}")
print("-" * 80)

for name, vol_t in tests.items():
    vt = flatten_moments(vol_t, n_max, alpha_x=alpha_x, alpha_y=alpha_y, alpha_z=alpha_z)
    re = relative_error(v0, vt)
    cs = cosine_similarity(v0, vt)
    print(f"{name:<25} | {re:>12.6f} | {cs:>12.6f}")

print()

print("[INFO] 测试完成")
