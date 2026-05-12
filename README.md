# PIR image 3D

Projet experimental en Python autour des moments d'image, avec un focus sur :

- les moments de Chebyshev 2D/3D et les moments fractionnaires (FrGM / FrCM)
- les invariants fractionnaires 2D (FrCMI) et la validation de stabilite face a la translation, la rotation et le changement d'echelle
- le calcul de descripteurs de moments et la reconstruction sur des volumes medicaux 3D (par ex. `.im.gz`, `.nii.gz`)

Le projet est actuellement organise comme un ensemble de scripts de recherche : le code principal est dans `src/` et les resultats sont ecrits dans `outputs/`.

## Structure du projet

- `src/` : algorithmes principaux et scripts experimentaux
- `outputs/` : resultats generes (CSV/NPZ/figures)

## Prerequis

Python 3.10+ est recommande.

Dependances principales :

- `numpy`
- `scipy`
- `matplotlib`
- `Pillow`
- `nibabel`

Installation (exemple Windows PowerShell) :

```powershell
py -m pip install numpy scipy matplotlib Pillow nibabel
```

Si vous utilisez l'environnement virtuel du projet (`.venv`), activez-le avant l'installation.

## Demarrage rapide

Depuis la racine du projet (exemple) :

```powershell
python src/variance_frcmi2d_alpha.py
```

Ce script lance un balayage de `alpha` pour FrCMI 2D et genere dans `outputs/` :

- `invariance_frcmi_alpha_detail.csv`
- `invariance_frcmi_alpha_summary.csv`
- `invariance_frcmi_alpha_aggregated.csv`

## Scripts utiles

### 1) Validation d'invariance 2D (formes MPEG-7)

- `src/validate_invariance_2d.py` : test de stabilite des moments classiques
- `src/validate_invariance_frcmi_2d.py` : test de stabilite FrCMI
- `src/variance_frcmi2d_alpha.py` : balayage multi-classes de `alpha` avec aggregation

Ordre de recherche des donnees :

1. variable d'environnement `MPEG7_ORIGINAL_DIR`
2. `MPEG7dataset/original`
3. `data/MPEG7dataset/original`
4. `~/Downloads/MPEG7dataset/original`

Il est recommande de definir la variable d'environnement, par exemple :

```powershell
$env:MPEG7_ORIGINAL_DIR = "D:\datasets\MPEG7dataset\original"
python src/variance_frcmi2d_alpha.py
```

### 2) Reconstruction et visualisation 3D

- `src/reconstruction_ants_3d.py`

Remarque : la variable `IM_PATH` dans le script pointe actuellement vers un chemin local absolu ; adaptez-la a votre machine avant execution.

### 3) Descripteurs de moments 3D sur volumes musculaires segmentes

- `src/moments_muscles_3d.py`

Remarque : `MRI_PATH` et `SEG_PATH` sont actuellement definis avec des chemins locaux absolus ; modifiez-les vers vos fichiers `.nii.gz` avant execution.

## Resultats

Le dossier `outputs/` contient generalement :

- des CSV d'evaluation d'invariance (par ex. resultats de balayage `alpha`)
- des fichiers de caracteristiques (`.npz`)
- des figures de visualisation (coupes de reconstruction, voxels, courbes)

## Notes

- Le projet ne propose pas encore d'entree CLI unifiee ; les experiences se lancent principalement script par script.
- Plusieurs scripts contiennent des chemins de donnees locaux ; verifiez et adaptez ces constantes avant la premiere execution.
- Si besoin, je peux ensuite ajouter un `requirements.txt` et une entree CLI unifiee (par ex. `python -m src.xxx --args`).
