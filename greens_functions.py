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


def compute_greens_functions(bdg, verbose=True, convention="equal_time"):
    """Compute spatial normal and conjugate-anomalous correlators.

    Parameters
    ----------
    bdg : dict
        Self-consistent BdG solution.
    verbose : bool
        Print progress by angle.
    convention : {"equal_time", "legacy_unsigned"}
        ``equal_time`` (default) implements Ketterson's T=0 lesser/equal-time
        branch, with t2=t1+0+ and r=r1-r2:

            G_< = +i sum_k |v_k|^2 exp(-i k.r) / hbar,
            F*_< = +i sum_k u_k* v_k exp(-i k.r) / hbar.

        In the book's BdG convention, u_k* v_k = Delta_k*/(2 E_k).
        This is the occupation/anomalous input required by the Oh--Kim
        equal-time two-spin reduction in the V_SO=0 original spin basis.

        ``legacy_unsigned`` preserves the archived implementation exactly:
        the t1>t2 particle component proportional to |u_k|^2 for G and an
        unsigned real-u,v anomalous product.  It is provided only for
        reproduction and numerical comparison, not as the new default.

    Returns
    -------
    r_vals, theta_vals, G_uu_polar, F_ud_polar
        The returned anomalous array is F* in the equal-time convention.
        Overall Green-function phases cancel in normalized |g| and |f|.
    """
    if convention not in {"equal_time", "legacy_unsigned"}:
        raise ValueError(f"Unknown Green's-function convention '{convention}'")

    u_ks = bdg["u_ks"]
    v_ks = bdg["v_ks"]
    KX   = bdg["KX"]
    KY   = bdg["KY"]
    kx   = bdg["kx"]
    ky   = bdg["ky"]

    if convention == "equal_time":
        Delta_ks = bdg["Delta_ks"]
        E_ks = bdg["E_ks"]
        # Ketterson Eq. (55.13): u_k* v_k = Delta_k*/(2 E_k).
        anomalous_coherence = np.conj(Delta_ks) / (2.0 * E_ks)

    r_vals, theta_vals = build_rgrid()
    Nth, Nr = len(theta_vals), len(r_vals)

    G_uu_polar = np.zeros((Nth, Nr), dtype=complex)
    F_ud_polar = np.zeros((Nth, Nr), dtype=complex)

    for i, theta in enumerate(theta_vals):
        if verbose:
            print(f"  [greens:{convention}] theta={np.degrees(theta):.0f} deg  "
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
                if convention == "equal_time":
                    # Occupation branch used by the equal-time density matrix.
                    sumG += 1j * np.abs(v)**2 * np.conj(phase)
                    # Conjugate anomalous correlator F* = <psi^dagger psi^dagger>.
                    # Retain the gap sign/phase before Brillouin-zone integration.
                    sumF += 1j * anomalous_coherence[:, :, s] * np.conj(phase)
                else:
                    # Exact archived implementation (do not optimize or repair).
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
    """Apply non-Markovian decay envelope to Green's functions.

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
