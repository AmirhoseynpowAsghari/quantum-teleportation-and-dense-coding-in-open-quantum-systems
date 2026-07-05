# =============================================================================
# self_consistency.py
# BdG self-consistency via fsolve.
# Residuals derived directly from the iterative update rules so that
# the two methods give identical results at T = 0.
# =============================================================================

import numpy as np
from scipy.optimize import fsolve

from config import (t_hop, U, Delta_t, V_SO, n_elec, Nk,
                    KBT, DELTA0_INIT, DELTAS_INIT,
                    FSOLVE_XTOL, FSOLVE_MAXFEV_FACTOR, FSOLVE_N_RESTARTS)
from kspace_setup import build_kgrid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _beta(kBT):
    return np.inf if kBT == 0.0 else 1.0 / kBT


def _tanh_factor(beta, E_ks):
    """tanh(beta*E/2);  returns ones at T=0."""
    if np.isinf(beta):
        return np.ones_like(E_ks)
    return np.tanh(np.clip(0.5 * beta * E_ks, 0.0, 500.0))


# ---------------------------------------------------------------------------
# Residual (matches iterative update rules exactly)
# ---------------------------------------------------------------------------

def residual(x, epsilon_ks, s_k_factor, sqrt_sin,
             Nk_loc, t_loc, U_loc, Dt_loc, Vso_loc, n_loc, beta):
    """
    F(x) = 0   for  x = (Delta0, DeltaS, mu).

    Derivation (T=0):
      iterative:  Delta0_new = -(1/2) SUM kernel*Delta/E / (4*N^2)
                             = -SUM kernel*Delta/E / (8*N^2)
      residual:   r1 = Delta0 + SUM kernel*Delta*th/E / (8*N^2)

      iterative:  DeltaS_new = -8*Dt * (1/2) SUM Delta/E / (4*N^2)
                             = -Dt * SUM Delta/E / N^2
      residual:   r2 = DeltaS + Dt * SUM Delta*th/E / N^2

      number eq:  n  = 1 - SUM (eps-mu)*th/E / (2*N^2)
      residual:   r3 = SUM (eps-mu)*th/E / (2*N^2) - (1-n)
    """
    D0, DS, mu = float(x[0]), float(x[1]), float(x[2])

    Dks = D0 - (DS / (4.0 * t_loc)) * epsilon_ks
    Eks = np.sqrt(Dks**2 + (epsilon_ks - mu)**2)
    Eks = np.clip(Eks, 1e-12, None)

    th = _tanh_factor(beta, Eks)
    th_over_E = th / Eks

    s_prime = np.array([1.0, -1.0])
    kernel = (
        U_loc
        + 8.0 * Dt_loc * s_k_factor[..., None]
        + 4.0 * (Dt_loc / t_loc) * Vso_loc
              * s_prime[None, None, :] * sqrt_sin[..., None]
    )

    N2 = float(Nk_loc**2)

    r1 = D0  + np.sum(kernel * Dks * th_over_E) / (8.0 * N2)
    r2 = DS  + Dt_loc * np.sum(Dks * th_over_E) / N2
    r3 = np.sum(((epsilon_ks - mu) / Eks) * th) / (2.0 * N2) - (1.0 - n_loc)

    return [r1, r2, r3]


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def run_selfconsistency(seed=None, U_override=None,
                        kBT_override=None, verbose=True):
    """
    Solve the BdG equations using fsolve.

    Returns
    -------
    dict  (same keys as the old iterative version)
    """
    if seed is not None:
        np.random.seed(seed)

    U_loc   = U         if U_override   is None else float(U_override)
    kBT_loc = KBT       if kBT_override is None else float(kBT_override)
    beta    = _beta(kBT_loc)

    kx, ky, KX, KY, eps_ks, ssin = build_kgrid()
    sk = 0.5 * (np.cos(KX) + np.cos(KY))

    args = (eps_ks, sk, ssin, Nk, t_hop, U_loc, Delta_t, V_SO, n_elec, beta)
    maxfev = FSOLVE_MAXFEV_FACTOR * 4

    mu_init = float(np.mean(eps_ks))
    x0 = np.array([DELTA0_INIT, DELTAS_INIT, mu_init])

    if verbose:
        tmp = "T=0" if np.isinf(beta) else f"kBT={kBT_loc:.4g}"
        print(f"[self_consistency] fsolve | U={U_loc:.4g} Dt={Delta_t:.4g} "
              f"VSO={V_SO:.4g} n={n_elec:.4f} {tmp} Nk={Nk}")
        print(f"  initial guess: D0={x0[0]:.4f} DS={x0[1]:.4f} mu={x0[2]:.4f}")

    sol, info, ier, msg = fsolve(
        residual, x0, args=args, full_output=True,
        xtol=FSOLVE_XTOL, maxfev=maxfev
    )
    nfev = info["nfev"]

    def _triv(s):
        return abs(s[0]) < 1e-5 and abs(s[1]) < 1e-5

    attempt = 0
    while (ier != 1 or _triv(sol)) and attempt < FSOLVE_N_RESTARTS:
        attempt += 1
        reason = "trivial" if _triv(sol) else "no-conv"
        xr = np.array([
            0.3 + 1.5 * np.random.rand(),
            0.2 + 1.0 * np.random.rand(),
            np.random.uniform(eps_ks.min(), eps_ks.max()),
        ])
        if verbose:
            print(f"  restart {attempt} ({reason}): x0={np.round(xr, 3)}")
        sol, info, ier, msg = fsolve(
            residual, xr, args=args, full_output=True,
            xtol=FSOLVE_XTOL, maxfev=maxfev
        )
        nfev += info["nfev"]

    converged = (ier == 1) and (not _triv(sol))
    D0, DS, mu_val = float(sol[0]), float(sol[1]), float(sol[2])
    res_check = residual(sol, *args)

    if verbose:
        status = "CONVERGED" if converged else "WARNING: NOT CONVERGED"
        print(f"  {status} ({nfev} evals)")
        print(f"  D0={D0:.8f}  DS={DS:.8f}  mu={mu_val:.8f}")
        print(f"  |r|=[{abs(res_check[0]):.2e}, "
              f"{abs(res_check[1]):.2e}, {abs(res_check[2]):.2e}]")

    Dks = D0 - (DS / (4.0 * t_hop)) * eps_ks
    Eks = np.sqrt(Dks**2 + (eps_ks - mu_val)**2)
    Eks = np.clip(Eks, 1e-12, None)

    uks = np.sqrt(np.clip(0.5 * (1.0 + (eps_ks - mu_val) / Eks), 0.0, 1.0))
    vks = np.sqrt(np.clip(0.5 * (1.0 - (eps_ks - mu_val) / Eks), 0.0, 1.0))

    norm_err = float(np.max(np.abs(uks**2 + vks**2 - 1.0)))
    assert norm_err < 1e-4, f"BdG norm failed: {norm_err:.2e}"

    return {
        "Delta0": D0, "DeltaS": DS, "mu": mu_val,
        "Delta_ks": Dks, "E_ks": Eks,
        "u_ks": uks, "v_ks": vks,
        "kx": kx, "ky": ky, "KX": KX, "KY": KY,
        "epsilon_ks": eps_ks, "sqrt_sin": ssin,
        "converged": converged, "n_iter": nfev,
        "beta": beta, "kBT": kBT_loc,
        "residuals": res_check,
    }


if __name__ == "__main__":
    res = run_selfconsistency(seed=42, kBT_override=0.0)
    print(f"\nD0={res['Delta0']:.6f}  DS={res['DeltaS']:.6f}  "
          f"mu={res['mu']:.6f}  ok={res['converged']}")