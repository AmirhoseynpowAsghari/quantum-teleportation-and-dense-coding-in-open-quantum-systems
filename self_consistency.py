# =============================================================================
# self_consistency.py
# BdG self-consistency loop to determine Δ₀, Δ_S, and μ.
# =============================================================================

import numpy as np
from scipy.optimize import bisect

from config import (t_hop, U, Delta_t, V_SO, n_elec, Nk,
                    SC_MAX_ITER, SC_TOL, SC_DAMP)
from kspace_setup import build_kgrid


def _number_equation(mu_trial, epsilon_ks, Delta_ks):
    """F(μ) = n_computed(μ) − n_target  (root = chemical potential)."""
    E = np.sqrt(Delta_ks**2 + (epsilon_ks - mu_trial)**2)
    E = np.clip(E, 1e-10, None)
    n_computed = 1.0 - (1.0 / (2.0 * Nk**2)) * np.sum(
        (epsilon_ks - mu_trial) / E
    )
    return n_computed - n_elec


def run_selfconsistency(seed=None):
    """
    Run the BdG self-consistency loop.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducible initial guesses.

    Returns
    -------
    dict with keys:
        Delta0, DeltaS, mu   – converged gap amplitudes & chemical potential
        Delta_ks             – (Nk,Nk,2) array of k-resolved gaps
        E_ks                 – (Nk,Nk,2) quasi-particle energies
        u_ks, v_ks           – (Nk,Nk,2) BdG coherence factors
        kx, ky, KX, KY       – grid arrays
        epsilon_ks, sqrt_sin – dispersion helpers
        converged            – bool
        n_iter               – number of iterations taken
    """
    if seed is not None:
        np.random.seed(seed)

    kx, ky, KX, KY, epsilon_ks, sqrt_sin = build_kgrid()

    # --- initial guesses ---------------------------------------------------
    Delta0 = 0.8 + 0.1 * np.random.rand()
    DeltaS = 0.8 + 0.1 * np.random.rand()
    mu     = float(np.mean(epsilon_ks))

    converged = False
    n_iter    = 0

    for iteration in range(SC_MAX_ITER):
        n_iter = iteration + 1

        # k-resolved gap
        Delta_ks = Delta0 - (DeltaS / (4.0 * t_hop)) * epsilon_ks

        # --- update μ via bisection ----------------------------------------
        eps_min  = float(np.min(epsilon_ks))
        eps_max  = float(np.max(epsilon_ks))
        mu_low   = eps_min - 2.0 * t_hop
        mu_high  = eps_max + 2.0 * t_hop

        try:
            mu = bisect(_number_equation, mu_low, mu_high,
                        args=(epsilon_ks, Delta_ks), xtol=1e-6)
        except ValueError:
            mu_low  -= 20.0 * t_hop
            mu_high += 20.0 * t_hop
            mu = bisect(_number_equation, mu_low, mu_high,
                        args=(epsilon_ks, Delta_ks), xtol=1e-6)

        # --- quasi-particle energies ----------------------------------------
        E_ks = np.sqrt(Delta_ks**2 + (epsilon_ks - mu)**2)
        E_ks = np.clip(E_ks, 1e-10, None)

        # --- gap equations --------------------------------------------------
        s_k = 0.5 * (np.cos(KX) + np.cos(KY))

        sum_Delta0 = 0.0
        sum_DeltaS = 0.0
        for s_idx in range(2):
            s_prime = 1 if s_idx == 0 else -1
            term = (U
                    + 8.0 * Delta_t * s_k
                    + 4.0 * (Delta_t / t_hop) * V_SO * s_prime * sqrt_sin)
            sum_Delta0 += np.sum(term * Delta_ks[:, :, s_idx] / E_ks[:, :, s_idx])
            sum_DeltaS += np.sum(Delta_ks[:, :, s_idx] / E_ks[:, :, s_idx])

        sum_Delta0 /= 2.0
        sum_DeltaS /= 2.0

        Delta0_new = -sum_Delta0 / (4.0 * Nk**2)
        DeltaS_new = -8.0 * Delta_t * sum_DeltaS / (4.0 * Nk**2)

        # --- damped update --------------------------------------------------
        Delta0_next = (1.0 - SC_DAMP) * Delta0 + SC_DAMP * Delta0_new
        DeltaS_next = (1.0 - SC_DAMP) * DeltaS + SC_DAMP * DeltaS_new

        change0 = abs(Delta0_next - Delta0)
        changeS = abs(DeltaS_next - DeltaS)

        Delta0, DeltaS = Delta0_next, DeltaS_next

        if change0 < SC_TOL and changeS < SC_TOL:
            converged = True
            print(f"[self_consistency] Converged after {n_iter} iterations. "
                  f"Δ₀={Delta0:.6f}  Δ_S={DeltaS:.6f}  μ={mu:.6f}")
            break

    if not converged:
        print("[self_consistency] WARNING: did not converge within "
              f"{SC_MAX_ITER} iterations.")

    # --- final BdG factors -------------------------------------------------
    Delta_ks = Delta0 - (DeltaS / (4.0 * t_hop)) * epsilon_ks
    E_ks     = np.sqrt(Delta_ks**2 + (epsilon_ks - mu)**2)
    E_ks     = np.clip(E_ks, 1e-10, None)

    u_ks = np.sqrt(0.5 * (1.0 + (epsilon_ks - mu) / E_ks))
    v_ks = np.sqrt(0.5 * (1.0 - (epsilon_ks - mu) / E_ks))

    # sanity check
    assert np.allclose(u_ks**2 + v_ks**2, 1.0, atol=1e-4), \
        "BdG normalisation |u|²+|v|²=1 failed!"

    return {
        "Delta0": Delta0, "DeltaS": DeltaS, "mu": mu,
        "Delta_ks": Delta_ks, "E_ks": E_ks,
        "u_ks": u_ks, "v_ks": v_ks,
        "kx": kx, "ky": ky, "KX": KX, "KY": KY,
        "epsilon_ks": epsilon_ks, "sqrt_sin": sqrt_sin,
        "converged": converged, "n_iter": n_iter,
    }


if __name__ == "__main__":
    res = run_selfconsistency(seed=42)
    print(f"Δ₀={res['Delta0']:.6f}  Δ_S={res['DeltaS']:.6f}  μ={res['mu']:.6f}")