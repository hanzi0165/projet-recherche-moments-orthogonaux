import numpy as np


def cell_centers_01(length: int):
    """
    Coordonnées des centres de pixels dans [0,1].
    x_i = i/N + Δx/2, avec Δx = 1/N
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    dx = 1.0 / length
    i = np.arange(length, dtype=np.float64)
    x = i / length + dx / 2.0
    return x, dx


def cell_bounds_01(length: int):
    """
    Bornes des cellules:
    [i/N, (i+1)/N]
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    dx = 1.0 / length
    lower = np.arange(length, dtype=np.float64) / length
    upper = lower + dx
    return lower, upper, dx


def chebyshev_power_coeffs(max_order: int):
    """
    Coefficients B_{n,l} des polynômes de Chebyshev classiques
    dans la base monomiale:
        T_n(x) = sum_{l=0}^n B[n,l] x^l

    Récurrence:
        T_0(x)=1
        T_1(x)=x
        T_{n+1}(x)=2xT_n(x)-T_{n-1}(x)
    """
    if max_order < 0:
        raise ValueError("max_order must be >= 0")

    B = np.zeros((max_order + 1, max_order + 1), dtype=np.float64)
    B[0, 0] = 1.0

    if max_order >= 1:
        B[1, 1] = 1.0

    for n in range(1, max_order):
        # 2x T_n
        two_x_Tn = np.zeros(max_order + 1, dtype=np.float64)
        two_x_Tn[1:] = 2.0 * B[n, :-1]

        # T_{n+1} = 2x T_n - T_{n-1}
        B[n + 1] = two_x_Tn - B[n - 1]

    return B


def fractional_interval_integrals(length: int, p_max: int, alpha: float):
    """
    Calcule I_p^alpha(x_i) sur chaque cellule:
        I_p^alpha(i) = ∫_{i/N}^{(i+1)/N} x^{alpha*p} dx
                     = [ upper^{alpha*p+1} - lower^{alpha*p+1} ] / (alpha*p+1)

    Retour:
        I shape = (p_max+1, length)
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if p_max < 0:
        raise ValueError("p_max must be >= 0")

    lower, upper, _ = cell_bounds_01(length)

    I = np.zeros((p_max + 1, length), dtype=np.float64)
    for p in range(p_max + 1):
        expo = alpha * p + 1.0
        I[p, :] = (upper ** expo - lower ** expo) / expo

    return I


def frgm_2d(image: np.ndarray, p_max: int, q_max: int, alpha_x: float = 1.0, alpha_y: float = 1.0):
    """
    Fractional geometric moments 2D:
        FrGM_{pq}^{alpha_x,alpha_y}
        = sum_{i,j} f(i,j) I_x^alpha_x(p,i) I_y^alpha_y(q,j)

    image shape = (H, W)
    Retour:
        G shape = (q_max+1, p_max+1)
        Ix shape = (p_max+1, W)
        Iy shape = (q_max+1, H)
    """
    if image.ndim != 2:
        raise ValueError("image must be 2D")

    H, W = image.shape
    img = image.astype(np.float64)

    Ix = fractional_interval_integrals(W, p_max, alpha_x)  # (p+1, W)
    Iy = fractional_interval_integrals(H, q_max, alpha_y)  # (q+1, H)

    # G[q,p] = sum_j sum_i img[j,i] * Iy[q,j] * Ix[p,i]
    G = np.einsum("ji,qj,pi->qp", img, Iy, Ix)

    return G, Ix, Iy


def fractional_chebyshev_basis_1d(length: int, n_max: int, alpha: float):
    """
    Base fractionnaire de Chebyshev approchée à partir de:
        T_n^alpha(x) = sum_{l=0}^n B_{n,l} x^{alpha*l}

    évaluée aux centres des pixels x_i dans [0,1].
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    x, dx = cell_centers_01(length)
    B = chebyshev_power_coeffs(n_max)

    T = np.zeros((n_max + 1, length), dtype=np.float64)
    for n in range(n_max + 1):
        for l in range(n + 1):
            T[n, :] += B[n, l] * (x ** (alpha * l))

    return T, x, dx


def numerical_basis_norms(T: np.ndarray, dx: float):
    """
    Approximation numérique de d_n^2:
        d_n^2 ≈ sum_i T_n(x_i)^2 * dx
    """
    return np.sum(T * T, axis=1) * dx


def frcm_2d(image, n_max, alpha_x=1.0, alpha_y=1.0, normalize=True):
    if image.ndim != 2:
        raise ValueError("image must be 2D")
    if alpha_x <= 0 or alpha_y <= 0:
        raise ValueError("alpha must be > 0")

    H, W = image.shape
    img = image.astype(np.float64)

    # 1) FrGM
    G, _, _ = frgm_2d(img, p_max=n_max, q_max=n_max,
                       alpha_x=alpha_x, alpha_y=alpha_y)

    # 2) Coefficients B
    Bx = chebyshev_power_coeffs(n_max)
    By = chebyshev_power_coeffs(n_max)

    # 3) Combinaison linéaire vectorisée
    C_raw = By @ G @ Bx.T  # shape (n_max+1, n_max+1)

    # 4) Normalisation
    if normalize:
        Tx, _, dx = fractional_chebyshev_basis_1d(W, n_max, alpha_x)
        Ty, _, dy = fractional_chebyshev_basis_1d(H, n_max, alpha_y)
        d2x = numerical_basis_norms(Tx, dx)
        d2y = numerical_basis_norms(Ty, dy)
        norm = np.sqrt(np.outer(d2y, d2x))  # (m, n)
        C = C_raw / norm
    else:
        C = C_raw

    Tx, _, _ = fractional_chebyshev_basis_1d(W, n_max, alpha_x)
    Ty, _, _ = fractional_chebyshev_basis_1d(H, n_max, alpha_y)

    return C, Tx, Ty