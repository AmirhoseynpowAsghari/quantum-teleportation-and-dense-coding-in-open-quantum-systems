# =============================================================================
# greens_functions.py
# Compute the normal (G) and anomalous (F) Green's functions on a (theta, r)
# grid and optionally evolve them in time under non-Markovian noise.
# =============================================================================

import numpy as np
from scipy.integrate import simpson

from config import R_MIN, R_MAX, N_R, THETA_DEG


def build_rgrid():
    """
    Build radial and angular grids.

    Returns
    -------
    r_vals     : 1-D float array, length N_R
    theta_vals : 1-D float array (radians), length N_theta
    """
    r_vals     = np.linspace(R_MIN, R_MAX, N_R)
    theta_vals = np.radians(THETA_DEG)
    return r_vals, theta_vals


def compute_greens_functions(bdg, verbose=True):
    """
    Compute G_{uu}(r, theta) and F_{ud}(r, theta) by 2-D k-space integration.

    Parameters
    ----------
    bdg     : dict   output of self_consistency.run_selfconsistency()
    verbose : bool

    Returns
    -------
    r_vals     : 1-D float array, length N_R
    theta_vals : 1-D float array (radians), length N_theta
    G_uu_polar : complex array, shape (N_theta, N_R)
    F_ud_polar : complex array, shape (N_theta, N_R)
    """
    u_ks = bdg["u_ks"]
    v_ks = bdg["v_ks"]
    KX   = bdg["KX"]
    KY   = bdg["KY"]
    kx   = bdg["kx"]
    ky   = bdg["ky"]

    r_vals, theta_vals = build_rgrid()
    N_theta = len(theta_vals)
    Nr      = len(r_vals)

    G_uu_polar = np.zeros((N_theta, Nr), dtype=complex)
    F_ud_polar = np.zeros((N_theta, Nr), dtype=complex)

    for i, theta in enumerate(theta_vals):
        if verbose:
            print(f"[greens] theta = {np.degrees(theta):.0f} deg  ({i+1}/{N_theta})")

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
                / (2.0 * np.pi) ** 2
            )
            F_ud_polar[i, j] = (
                simpson(y=simpson(y=sum_F, x=kx, axis=0), x=ky, axis=0)
                / (2.0 * np.pi) ** 2
            )

    return r_vals, theta_vals, G_uu_polar, F_ud_polar


def evolve_greens_functions(G_uu_polar, F_ud_polar, t_grid,
                             gamma=0.05, Gamma=0.01,
                             kind="amp", verbose=True):
    """
    Evolve Green's functions in time under a non-Markovian decay envelope.

    For amplitude damping:
        G(r, t) = G(r, 0) * P(t)
        F(r, t) = F(r, 0) * sqrt(P(t))

    For dephasing:
        G(r, t) = G(r, 0) * exp(-gamma/2 * [t + (exp(-Gamma*t)-1)/Gamma])
        F(r, t) = F(r, 0) * exp(-gamma   * [t + (exp(-Gamma*t)-1)/Gamma])

    Parameters
    ----------
    G_uu_polar : complex array, shape (N_theta, N_R)
    F_ud_polar : complex array, shape (N_theta, N_R)
    t_grid     : 1-D float array, length N_t
    gamma      : float   coupling strength
    Gamma      : float   reservoir memory rate
    kind       : str     'amp' or 'dephasing'
    verbose    : bool

    Returns
    -------
    G_t : complex array, shape (N_theta, N_R, N_t)
    F_t : complex array, shape (N_theta, N_R, N_t)
    """
    N_theta, Nr = G_uu_polar.shape
    Nt          = len(t_grid)

    G_t = np.zeros((N_theta, Nr, Nt), dtype=complex)
    F_t = np.zeros((N_theta, Nr, Nt), dtype=complex)

    for k, t in enumerate(t_grid):
        if verbose and k % max(1, Nt // 10) == 0:
            print(f"[greens_evolve] t = {t:.2f}  ({k+1}/{Nt})")

        if kind == "amp":
            # Lorentzian structured reservoir
            d2  = 2.0 * gamma * Gamma - Gamma ** 2
            if d2 > 0.0:
                d   = np.sqrt(d2)
                P   = np.exp(-Gamma * t) * (
                    np.cos(0.5 * d * t) + (Gamma / d) * np.sin(0.5 * d * t)
                ) ** 2
            else:
                P = np.exp(-Gamma * t) * (1.0 + 0.5 * Gamma * t) ** 2
            P = float(np.clip(P, 0.0, 1.0))

            G_t[:, :, k] = G_uu_polar * P
            F_t[:, :, k] = F_ud_polar * np.sqrt(P)

        elif kind == "dephasing":
            # Ornstein-Uhlenbeck colored noise
            exponent = gamma * (t + (np.exp(-Gamma * t) - 1.0) / Gamma)
            decay_G  = np.exp(-0.5 * exponent)
            decay_F  = np.exp(-exponent)

            G_t[:, :, k] = G_uu_polar * decay_G
            F_t[:, :, k] = F_ud_polar * decay_F

        else:
            raise ValueError(f"Unknown kind '{kind}'. Use 'amp' or 'dephasing'.")

    return G_t, F_t


if __name__ == "__main__":
    from self_consistency import run_selfconsistency
    from config import T_MIN, T_MAX, N_T

    bdg = run_selfconsistency(seed=0)
    r_vals, theta_vals, G, F = compute_greens_functions(bdg, verbose=True)

    t_grid = np.linspace(T_MIN, T_MAX, N_T)
    G_t, F_t = evolve_greens_functions(G, F, t_grid,
                                        gamma=4.5, Gamma=0.01,
                                        kind="amp", verbose=True)

    print(f"G_t shape : {G_t.shape}")
    print(f"F_t shape : {F_t.shape}")
    print(f"|G|_max at t=0   : {np.abs(G_t[:,:,0]).max():.6f}")
    print(f"|G|_max at t=end : {np.abs(G_t[:,:,-1]).max():.6f}")