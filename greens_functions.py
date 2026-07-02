# =============================================================================
# greens_functions.py
# Compute the normal (G) and anomalous (F) Green's functions on a (θ, r) grid.
# =============================================================================

import numpy as np
from scipy.integrate import simpson

from config import R_MIN, R_MAX, N_R, THETA_DEG


def build_rgrid():
    """Return radial and angular grids."""
    r_vals     = np.linspace(R_MIN, R_MAX, N_R)
    theta_vals = np.radians(THETA_DEG)
    return r_vals, theta_vals


def compute_greens_functions(bdg, verbose=True):
    """
    Compute G_{uu}(r,θ) and F_{ud}(r,θ) by 2-D k-space integration.

    Parameters
    ----------
    bdg : dict
        Output of ``self_consistency.run_selfconsistency()``.
    verbose : bool

    Returns
    -------
    r_vals      : 1-D array, length N_R
    theta_vals  : 1-D array (radians), length Nθ
    G_uu_polar  : complex array, shape (Nθ, N_R)
    F_ud_polar  : complex array, shape (Nθ, N_R)
    """
    u_ks = bdg["u_ks"]
    v_ks = bdg["v_ks"]
    KX   = bdg["KX"]
    KY   = bdg["KY"]
    kx   = bdg["kx"]
    ky   = bdg["ky"]

    r_vals, theta_vals = build_rgrid()
    Nθ, Nr = len(theta_vals), len(r_vals)

    G_uu_polar = np.zeros((Nθ, Nr), dtype=complex)
    F_ud_polar = np.zeros((Nθ, Nr), dtype=complex)

    for i, theta in enumerate(theta_vals):
        if verbose:
            print(f"[greens] θ = {np.degrees(theta):.0f}°  ({i+1}/{Nθ})")
        rx_dir = np.cos(theta)
        ry_dir = np.sin(theta)

        for j, r in enumerate(r_vals):
            phase = np.exp(1j * (KX * r * rx_dir + KY * r * ry_dir))

            sum_G = np.zeros_like(KX, dtype=complex)
            sum_F = np.zeros_like(KX, dtype=complex)

            for s in range(2):
                u = u_ks[:, :, s]
                v = v_ks[:, :, s]
                sum_G += -1j * u * u * phase
                sum_F +=  1j * np.conj(v) * np.conj(u) * np.conj(phase)

            # 2-D Simpson integration over (kx, ky)
            G_uu_polar[i, j] = (
                simpson(y=simpson(y=sum_G, x=kx, axis=0), x=ky, axis=0)
                / (2.0 * np.pi)**2
            )
            F_ud_polar[i, j] = (
                simpson(y=simpson(y=sum_F, x=kx, axis=0), x=ky, axis=0)
                / (2.0 * np.pi)**2
            )

    return r_vals, theta_vals, G_uu_polar, F_ud_polar


if __name__ == "__main__":
    from self_consistency import run_selfconsistency
    bdg = run_selfconsistency(seed=0)
    r_vals, theta_vals, G, F = compute_greens_functions(bdg)
    print(f"G_uu shape: {G.shape}   |G|_max={np.abs(G).max():.4f}")
    print(f"F_ud shape: {F.shape}   |F|_max={np.abs(F).max():.4f}")