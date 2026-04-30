import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

path = r"C:\Users\Flora\Downloads\camus_32_dlss21_ho4\camus_32\patient0162\patient0162_2CH_ED_gt.nii.gz"

img = sitk.ReadImage(path)
arr = sitk.GetArrayFromImage(img)

print("Read success!")
print("path:", path)
print("shape:", arr.shape)
print("dtype:", arr.dtype)
print("min:", np.min(arr))
print("max:", np.max(arr))
print("unique:", np.unique(arr))
print("spacing:", img.GetSpacing())

# Handle common singleton-slice layout from NIfTI: (1, H, W) -> (H, W)
if arr.ndim == 3 and arr.shape[0] == 1:
    arr_show = arr[0]
else:
    arr_show = arr

if arr_show.ndim != 2:
    raise ValueError(f"Expected 2D image for display, got shape {arr_show.shape}")

plt.figure(figsize=(5, 5))
plt.imshow(arr_show, cmap="gray")
plt.title("patient0162_2CH_ED_gt")
plt.axis("off")
plt.tight_layout()
plt.show()
