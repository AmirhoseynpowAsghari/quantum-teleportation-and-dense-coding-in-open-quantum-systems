# =============================================================================
# greens_functions.py
# =============================================================================

import numpy as np
from scipy.integrate import simpson
from config import R_MIN, R_MAX, N_R, THETA_DEG


def build_rgrid():
    r_vals     = np.linspace(R_MIN, R_MAX, N_R)
    theta_vals = np.radians(THETA_DEG)
    return r_vals, theta_vals


def compute_greens_functions(bdg, verbose=True):
    """
    Compute G_uu(r,theta) and F_ud(r,theta) by 2-D k-space integration.

    Returns
    -------
    r_vals, theta_vals, G_uu_polar, F_ud_polar
    """
    u_ks = bdg["u_ks"]
    v_ks = bdg["v_ks"]
    KX   = bdg["KX"]
    KY   = bdg["KY"]
    kx   = bdg["kx"]
    ky   = bdg["ky"]

    r_vals, theta_vals = build_rgrid()
    Nth, Nr = len(theta_vals), len(r_vals)

    G_uu_polar = np.zeros((Nth, Nr), dtype=complex)
    F_ud_polar = np.zeros((Nth, Nr), dtype=complex)

    for i, theta in enumerate(theta_vals):
        if verbose:
            print(f"  [greens] theta={np.degrees(theta):.0f} deg  "
                  f"({i+1}/{Nth})")
        rx = np.cos(theta)
        ry = np.sin(theta)

        for j, r in enumerate(r_vals):
            phase = np.exp(1j * (KX * r * rx + KY * r * ry))
            sumG  = np.zeros_like(KX, dtype=complex)
            sumF  = np.zeros_like(KX, dtype=complex)

            for s in range(2):
                u = u_ks[:, :, s]
                v = v_ks[:, :, s]
                sumG += -1j * u * u * phase
                sumF +=  1j * np.conj(v) * np.conj(u) * np.conj(phase)

            G_uu_polar[i, j] = (
                simpson(y=simpson(y=sumG, x=kx, axis=0), x=ky, axis=0)
                / (2.0 * np.pi)**2
            )
            F_ud_polar[i, j] = (
                simpson(y=simpson(y=sumF, x=kx, axis=0), x=ky, axis=0)
                / (2.0 * np.pi)**2
            )

    return r_vals, theta_vals, G_uu_polar, F_ud_polar


def evolve_greens_functions(G_uu_polar, F_ud_polar, t_grid,
                             gamma=0.05, GammaA=0.01,
                             kind="amp", verbose=True):
    """
    Apply non-Markovian decay envelope to Green's functions.

    Returns
    -------
    G_t, F_t  : complex arrays, shape (N_theta, N_R, N_t)
    """
    Nth, Nr = G_uu_polar.shape
    Nt      = len(t_grid)

    G_t = np.zeros((Nth, Nr, Nt), dtype=complex)
    F_t = np.zeros((Nth, Nr, Nt), dtype=complex)

    for k, t in enumerate(t_grid):
        if verbose and k % max(1, Nt // 8) == 0:
            print(f"  [greens_evolve] t={t:.2f}  ({k+1}/{Nt})")

        if kind == "amp":
            d2  = 2.0 * gamma * GammaA - GammaA**2
            if d2 > 0.0:
                d = np.sqrt(d2)
                P = np.exp(-GammaA * t) * (
                    np.cos(0.5*d*t) + (GammaA/d)*np.sin(0.5*d*t)
                )**2
            else:
                P = np.exp(-GammaA * t) * (1.0 + 0.5*GammaA*t)**2
            P = float(np.clip(P, 0.0, 1.0))
            G_t[:, :, k] = G_uu_polar * P
            F_t[:, :, k] = F_ud_polar * np.sqrt(P)

        elif kind == "dephasing":
            exponent = gamma * (t + (np.exp(-GammaA*t) - 1.0) / GammaA)
            G_t[:, :, k] = G_uu_polar * np.exp(-0.5 * exponent)
            F_t[:, :, k] = F_ud_polar * np.exp(-exponent)

        else:
            raise ValueError(f"Unknown kind '{kind}'")

    return G_t, F_t