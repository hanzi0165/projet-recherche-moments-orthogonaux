import numpy as np
import gzip
import matplotlib.pyplot as plt

def load_im_gz(path):
    """
    Lit un fichier .im.gz du McGill Benchmark.
    Header : 1024 bytes, puis données 128³ bytes.
    """
    with gzip.open(path, 'rb') as f:
        f.read(1024)                          # sauter le header
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return (data.reshape(128, 128, 128) > 0).astype(np.float64)

# Test rapide
path = r"C:\Users\Flora\Downloads\cupsIm\cupsIm\b1.im.gz"
vol = load_im_gz(path)

print("Shape :", vol.shape)          # (128, 128, 128)
print("Voxels actifs :", int(vol.sum()))
print("Valeurs uniques :", np.unique(vol))



fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(vol[64, :, :], cmap='gray')
axes[0].set_title("Coupe XY (z=64)")
axes[1].imshow(vol[:, 64, :], cmap='gray')
axes[1].set_title("Coupe XZ (y=64)")
axes[2].imshow(vol[:, :, 64], cmap='gray')
axes[2].set_title("Coupe YZ (x=64)")
for ax in axes: ax.axis('off')
plt.tight_layout()
plt.show()