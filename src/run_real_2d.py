import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

from moments_2d import chebyshev_moments_2d
from reconstruction import reconstruct_2d
from metrics import mse, psnr

# Chemin vers l'image réelle du dataset CAMUS
path = r"C:\Users\Flora\Downloads\camus_32_dlss21_ho4\camus_32\patient0162\patient0162_2CH_ED.nii.gz"

# 1. Lecture de l'image originale
img_itk = sitk.ReadImage(path)
arr = sitk.GetArrayFromImage(img_itk)

# 2. Conversion en float et normalisation dans [0,1]
img = arr.astype(np.float64) / 255.0

# 3. Liste des ordres maximaux N à tester
n_values = [10, 20, 30, 35, 40, 50]

# 4. Affichage des informations de base
print("Image shape:", img.shape)
print()

# Listes pour stocker les résultats
results = []
mse_values = []
psnr_values = []

# 5. Boucle sur les différentes valeurs de N
for n_max in n_values:
    # Calcul des moments de Chebyshev 2D
    M, Tx, Ty = chebyshev_moments_2d(img, n_max)

    # Reconstruction de l'image à partir des moments
    rec = reconstruct_2d(M, Tx, Ty)

    # Calcul des indicateurs de qualité
    m = mse(img, rec)
    p = psnr(img, rec)

    # Sauvegarde des résultats
    results.append((n_max, rec, m, p))
    mse_values.append(m)
    psnr_values.append(p)

    # Affichage des résultats dans le terminal
    print(f"N = {n_max}")
    print("  Moment shape:", M.shape)
    print("  MSE:", m)
    print("  PSNR:", p)
    print()

# 6. Affichage de l'image originale et des reconstructions
num_plots = 1 + len(n_values)
plt.figure(figsize=(4 * num_plots, 4))

# Image originale
plt.subplot(1, num_plots, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

# Images reconstruites pour chaque N
for idx, (n_max, rec, m, p) in enumerate(results, start=2):
    plt.subplot(1, num_plots, idx)
    plt.imshow(rec, cmap="gray")
    plt.title(f"N={n_max}\nMSE={m:.2e}\nPSNR={p:.2f}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# 7. Courbe MSE en fonction de N
plt.figure(figsize=(6, 4))
plt.semilogy(n_values, mse_values, marker='o')
plt.xlabel("N (ordre maximal)")
plt.ylabel("MSE (échelle logarithmique)")
plt.title("MSE en fonction de l'ordre de reconstruction")
plt.grid(True, which="both")
plt.tight_layout()
plt.show()

# 8. Courbe PSNR en fonction de N
plt.figure(figsize=(6, 4))
plt.plot(n_values, psnr_values, marker='o')
plt.xlabel("N (ordre maximal)")
plt.ylabel("PSNR")
plt.title("PSNR en fonction de l'ordre de reconstruction")
plt.grid(True)
plt.tight_layout()
plt.show()