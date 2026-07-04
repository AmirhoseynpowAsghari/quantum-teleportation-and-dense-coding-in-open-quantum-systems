# =============================================================================
# self_consistency.py
# BdG self-consistency solved as a simultaneous root-finding problem
# with scipy.optimize.fsolve.
#
# Equations derived directly from the iterative method to guarantee
# identical results:
#
# The iterative method gives:
#   Delta0_new = -(1/2) * SUM_{k,s} kernel * Delta_ks/E_ks  / (4*Nk^2)
#   DeltaS_new = -8*Dt  * SUM_{k,s}          Delta_ks/E_ks  / (4*Nk^2)
#   n          =  1     - (1/(2*Nk^2)) * SUM_{k,s} (eps-mu)/E_ks
#
# So the residuals F(x)=0 are:
#   r1 = Delta0 + SUM_{k,s} kernel * Delta_ks/E_ks  / (8*Nk^2)
#   r2 = DeltaS + 8*Dt * SUM_{k,s} Delta_ks/E_ks   / (4*Nk^2)
#   r3 = n - 1  + (1/(2*Nk^2)) * SUM_{k,s} (eps-mu)/E_ks
#
# At finite T: replace 1/E_ks  -->  tanh(beta*E_ks/2) / E_ks
# =============================================================================

import numpy as np
from scipy.optimize import fsolve

from config import (t_hop, U, Delta_t, V_SO, n_elec, Nk,
                    KBT, DELTA0_INIT, DELTAS_INIT,
                    FSOLVE_XTOL, FSOLVE_MAXFEV_FACTOR, FSOLVE_N_RESTARTS)
from kspace_setup import build_kgrid


# ---------------------------------------------------------------------------
# Helper: thermal factor
# ---------------------------------------------------------------------------

def _tanh_factor(beta, E_ks):
    """
    Compute tanh(beta * E / 2) safely.

    At T = 0 (beta = inf):  returns ones  (BCS T=0 limit).
    At T > 0:               returns tanh(beta * E / 2).
    """
    if np.isinf(beta):
        return np.ones_like(E_ks)
    arg = np.clip(0.5 * beta * E_ks, 0.0, 500.0)
    return np.tanh(arg)


# ---------------------------------------------------------------------------
# Residual function — derived directly from the iterative update rules
# ---------------------------------------------------------------------------

def residual(x, epsilon_ks, s_k_factor, sqrt_sin,
             Nk_local, t_local, U_local, Delta_t_local,
             V_SO_local, n_local, beta):
    """
    Residual vector [r1, r2, r3] consistent with the iterative method.

    Derivation
    ----------
    Iterative method update rules (T=0):

        sum_Delta0 = (1/2) * SUM_{k,s} kernel * Delta_ks / E_ks
        Delta0_new = -sum_Delta0 / (4*Nk^2)
                   = -SUM_{k,s} kernel * Delta_ks / E_ks / (8*Nk^2)

        Self-consistency: Delta0 = Delta0_new
        =>  r1 = Delta0 + SUM kernel * Delta_ks / E_ks / (8*Nk^2) = 0

        sum_DeltaS = (1/2) * SUM_{k,s} Delta_ks / E_ks
        DeltaS_new = -8*Dt * sum_DeltaS / (4*Nk^2)
                   = -8*Dt * SUM_{k,s} Delta_ks / E_ks / (8*Nk^2)
                   = -Dt   * SUM_{k,s} Delta_ks / E_ks / (Nk^2)

        Self-consistency: DeltaS = DeltaS_new
        =>  r2 = DeltaS + Dt * SUM Delta_ks / E_ks / Nk^2 = 0

        Number equation:
        n = 1 - (1/(2*Nk^2)) * SUM_{k,s} (eps-mu)/E_ks
        =>  r3 = (1/(2*Nk^2)) * SUM (eps-mu)/E_ks - (1-n) = 0

    At finite T: replace 1/E_ks --> tanh(beta*E/2) / E_ks everywhere.

    Parameters
    ----------
    x          : array (3,)  trial (Delta0, DeltaS, mu)
    epsilon_ks : (Nk, Nk, 2)
    s_k_factor : (Nk, Nk)
    sqrt_sin   : (Nk, Nk)
    Nk_local   : int
    t_local, U_local, Delta_t_local, V_SO_local, n_local : float
    beta       : float   1/(k_B T);  np.inf for T=0

    Returns
    -------
    [r1, r2, r3] : list of floats  (zero at the solution)
    """
    Delta0, DeltaS, mu = float(x[0]), float(x[1]), float(x[2])

    # ------------------------------------------------------------------
    # k-resolved gap
    # ------------------------------------------------------------------
    Delta_ks = Delta0 - (DeltaS / (4.0 * t_local)) * epsilon_ks

    # ------------------------------------------------------------------
    # Quasiparticle energies
    # ------------------------------------------------------------------
    E_ks = np.sqrt(Delta_ks**2 + (epsilon_ks - mu)**2)
    E_ks = np.clip(E_ks, 1e-12, None)

    # ------------------------------------------------------------------
    # Thermal factor:  tanh(beta*E/2) = 1 at T=0
    # ------------------------------------------------------------------
    th = _tanh_factor(beta, E_ks)

    # ------------------------------------------------------------------
    # Kernel for Delta0 equation — matches the iterative method exactly
    # shape: (Nk, Nk, 2) with s' = [+1, -1]
    # ------------------------------------------------------------------
    s_prime = np.array([1.0, -1.0])                        # (2,)
    kernel = (
        U_local
        + 8.0 * Delta_t_local * s_k_factor[..., None]     # (Nk,Nk,1)
        + 4.0 * (Delta_t_local / t_local) * V_SO_local
              * s_prime[None, None, :] * sqrt_sin[..., None]
    )                                                       # (Nk, Nk, 2)

    # ------------------------------------------------------------------
    # Pair amplitude factor: tanh(beta*E/2) / E_ks
    # At T=0: this is just 1/E_ks
    # ------------------------------------------------------------------
    th_over_E = th / E_ks                                  # (Nk, Nk, 2)

    N2 = float(Nk_local ** 2)

    # ------------------------------------------------------------------
    # r1: Delta0 equation
    #
    # From iteration:
    #   Delta0 = -(1/2) * SUM kernel * Delta_ks/E_ks / (4*N2)
    #          = -SUM kernel * Delta_ks/E_ks / (8*N2)
    #
    # Residual:
    #   r1 = Delta0 + SUM kernel * Delta_ks * th/E_ks / (8*N2)
    # ------------------------------------------------------------------
    r1 = Delta0 + np.sum(kernel * Delta_ks * th_over_E) / (8.0 * N2)

    # ------------------------------------------------------------------
    # r2: DeltaS equation
    #
    # From iteration:
    #   sum_DeltaS = (1/2) * SUM Delta_ks/E_ks
    #   DeltaS_new = -8*Dt * sum_DeltaS / (4*N2)
    #              = -8*Dt * SUM Delta_ks/E_ks / (8*N2)
    #              = -Dt   * SUM Delta_ks/E_ks / N2
    #
    # Residual:
    #   r2 = DeltaS + Delta_t * SUM Delta_ks * th/E_ks / N2
    # ------------------------------------------------------------------
    r2 = DeltaS + Delta_t_local * np.sum(Delta_ks * th_over_E) / N2

    # ------------------------------------------------------------------
    # r3: number equation
    #
    # From iteration (bisect finds mu such that):
    #   n = 1 - (1/(2*N2)) * SUM (eps-mu)/E_ks
    #
    # Residual:
    #   r3 = (1/(2*N2)) * SUM (eps-mu)*th/E_ks - (1 - n)
    # ------------------------------------------------------------------
    r3 = (np.sum(((epsilon_ks - mu) / E_ks) * th) / (2.0 * N2)
          - (1.0 - n_local))

    return [r1, r2, r3]


# ---------------------------------------------------------------------------
# Validation: cross-check residual against the iterative formula
# ---------------------------------------------------------------------------

def _validate_residual(Delta0, DeltaS, mu,
                        epsilon_ks, KX, KY, sqrt_sin,
                        t_local, U_local, Delta_t_local,
                        V_SO_local, n_local, Nk_local):
    """
    Re-derive Delta0_new and DeltaS_new using the exact iterative formula
    and return |Delta0 - Delta0_new| and |DeltaS - DeltaS_new|.

    Used internally to verify that fsolve found the same answer as
    the iteration method.
    """
    Delta_ks = Delta0 - (DeltaS / (4.0 * t_local)) * epsilon_ks
    E_ks = np.sqrt(Delta_ks**2 + (epsilon_ks - mu)**2)
    E_ks = np.clip(E_ks, 1e-12, None)

    s_k = 0.5 * (np.cos(KX) + np.cos(KY))
    s_prime_vals = [1.0, -1.0]

    sum_D0 = 0.0
    sum_DS = 0.0
    for s_idx, s_prime in enumerate(s_prime_vals):
        term = (U_local
                + 8.0 * Delta_t_local * s_k
                + 4.0 * (Delta_t_local / t_local) * V_SO_local
                      * s_prime * sqrt_sin)
        sum_D0 += np.sum(term * Delta_ks[:, :, s_idx] / E_ks[:, :, s_idx])
        sum_DS += np.sum(Delta_ks[:, :, s_idx] / E_ks[:, :, s_idx])

    sum_D0 /= 2.0
    sum_DS /= 2.0

    N2 = float(Nk_local ** 2)
    Delta0_iter = -sum_D0 / (4.0 * N2)
    DeltaS_iter = -8.0 * Delta_t_local * sum_DS / (4.0 * N2)

    err0 = abs(Delta0 - Delta0_iter)
    errS = abs(DeltaS - DeltaS_iter)
    return Delta0_iter, DeltaS_iter, err0, errS


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def run_selfconsistency(seed=None, U_override=None,
                        kBT_override=None, verbose=True):
    """
    Solve the BdG self-consistency equations using fsolve.

    The residual function is derived directly from the iterative
    update rules, guaranteeing identical results at convergence.

    Parameters
    ----------
    seed         : int or None
    U_override   : float or None   alternative U value (e.g. for U-scans)
    kBT_override : float or None   alternative k_B T (0.0 = zero temperature)
    verbose      : bool

    Returns
    -------
    dict with keys (same as iterative version):
        Delta0, DeltaS, mu, Delta_ks, E_ks, u_ks, v_ks,
        kx, ky, KX, KY, epsilon_ks, sqrt_sin,
        converged, n_iter, beta, kBT, residuals
    """
    if seed is not None:
        np.random.seed(seed)

    U_local   = U         if U_override   is None else float(U_override)
    kBT_local = KBT       if kBT_override is None else float(kBT_override)
    beta      = np.inf    if kBT_local == 0.0    else 1.0 / kBT_local

    # ------------------------------------------------------------------
    # Grids
    # ------------------------------------------------------------------
    kx, ky, KX, KY, epsilon_ks, sqrt_sin = build_kgrid()
    s_k_factor = 0.5 * (np.cos(KX) + np.cos(KY))

    # ------------------------------------------------------------------
    # Print header
    # ------------------------------------------------------------------
    if verbose:
        temp_str = ("T = 0  (exact zero-temperature BCS)"
                    if np.isinf(beta)
                    else f"k_B T = {kBT_local:.4g}")
        print("=" * 65)
        print("[self_consistency] fsolve BdG solver")
        print(f"  U       = {U_local:.6g}")
        print(f"  Delta_t = {Delta_t:.6g}")
        print(f"  V_SO    = {V_SO:.6g}")
        print(f"  n       = {n_elec:.6g}")
        print(f"  Temp    : {temp_str}")
        print(f"  Nk      = {Nk}")
        print("=" * 65)

    # ------------------------------------------------------------------
    # Pack arguments
    # ------------------------------------------------------------------
    args = (epsilon_ks, s_k_factor, sqrt_sin,
            Nk, t_hop, U_local, Delta_t, V_SO, n_elec, beta)

    maxfev = FSOLVE_MAXFEV_FACTOR * (3 + 1)

    # ------------------------------------------------------------------
    # Attempt 1: configured initial guess
    # ------------------------------------------------------------------
    mu_init = float(np.mean(epsilon_ks))
    x0 = np.array([DELTA0_INIT, DELTAS_INIT, mu_init])

    if verbose:
        print(f"[self_consistency] Initial guess: "
              f"Delta0={x0[0]:.4f}  DeltaS={x0[1]:.4f}  mu={x0[2]:.6f}")

    solution, infodict, ier, mesg = fsolve(
        residual, x0, args=args,
        full_output=True,
        xtol=FSOLVE_XTOL,
        maxfev=maxfev,
    )
    nfev_total = infodict["nfev"]

    # ------------------------------------------------------------------
    # Guard: avoid trivial (zero gap) solution
    # ------------------------------------------------------------------
    def _is_trivial(sol):
        return abs(sol[0]) < 1e-5 and abs(sol[1]) < 1e-5

    # ------------------------------------------------------------------
    # Random restarts if not converged OR landed on trivial root
    # ------------------------------------------------------------------
    attempt = 0
    while (ier != 1 or _is_trivial(solution)) and attempt < FSOLVE_N_RESTARTS:
        attempt += 1
        reason = "trivial root" if _is_trivial(solution) else "no convergence"
        x0_retry = np.array([
            0.3 + 1.5 * np.random.rand(),
            0.2 + 1.0 * np.random.rand(),
            np.random.uniform(epsilon_ks.min(), epsilon_ks.max()),
        ])
        if verbose:
            print(f"[self_consistency] Restart {attempt}/{FSOLVE_N_RESTARTS} "
                  f"({reason}) x0={np.round(x0_retry, 4)}")
        solution, infodict, ier, mesg = fsolve(
            residual, x0_retry, args=args,
            full_output=True,
            xtol=FSOLVE_XTOL,
            maxfev=maxfev,
        )
        nfev_total += infodict["nfev"]

    converged  = (ier == 1) and (not _is_trivial(solution))
    Delta0, DeltaS, mu = float(solution[0]), float(solution[1]), float(solution[2])
    res_check  = residual(solution, *args)

    # ------------------------------------------------------------------
    # Cross-validate against the iterative formula (T=0 only)
    # ------------------------------------------------------------------
    if np.isinf(beta) and converged and verbose:
        D0_iter, DS_iter, err0, errS = _validate_residual(
            Delta0, DeltaS, mu,
            epsilon_ks, KX, KY, sqrt_sin,
            t_hop, U_local, Delta_t, V_SO, n_elec, Nk
        )
        print(f"[self_consistency] Cross-check vs iterative formula:")
        print(f"  Delta0_iter  = {D0_iter:.8f}   |diff| = {err0:.2e}")
        print(f"  DeltaS_iter  = {DS_iter:.8f}   |diff| = {errS:.2e}")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    if converged:
        if verbose:
            print(f"[self_consistency] CONVERGED "
                  f"(total evaluations: {nfev_total})")
            print(f"  Delta0  = {Delta0:.8f}")
            print(f"  DeltaS  = {DeltaS:.8f}")
            print(f"  mu      = {mu:.8f}")
            print(f"  |r1|    = {abs(res_check[0]):.2e}")
            print(f"  |r2|    = {abs(res_check[1]):.2e}")
            print(f"  |r3|    = {abs(res_check[2]):.2e}")
    else:
        print("[self_consistency] WARNING: did NOT converge "
              "to a non-trivial solution.")
        print(f"  Last: Delta0={Delta0:.6f}  "
              f"DeltaS={DeltaS:.6f}  mu={mu:.6f}")
        print(f"  Residuals: r1={res_check[0]:.2e}  "
              f"r2={res_check[1]:.2e}  r3={res_check[2]:.2e}")

    # ------------------------------------------------------------------
    # Final BdG coherence factors
    # ------------------------------------------------------------------
    Delta_ks = Delta0 - (DeltaS / (4.0 * t_hop)) * epsilon_ks
    E_ks     = np.sqrt(Delta_ks**2 + (epsilon_ks - mu)**2)
    E_ks     = np.clip(E_ks, 1e-12, None)

    u_ks = np.sqrt(np.clip(0.5 * (1.0 + (epsilon_ks - mu) / E_ks), 0.0, 1.0))
    v_ks = np.sqrt(np.clip(0.5 * (1.0 - (epsilon_ks - mu) / E_ks), 0.0, 1.0))

    norm_err = float(np.max(np.abs(u_ks**2 + v_ks**2 - 1.0)))
    assert norm_err < 1e-4, \
        f"BdG normalisation failed: max|u²+v²-1| = {norm_err:.2e}"

    if verbose:
        print(f"[self_consistency] BdG norm OK "
              f"(max|u²+v²-1| = {norm_err:.2e})")
        print("=" * 65)

    return {
        "Delta0"    : Delta0,
        "DeltaS"    : DeltaS,
        "mu"        : mu,
        "Delta_ks"  : Delta_ks,
        "E_ks"      : E_ks,
        "u_ks"      : u_ks,
        "v_ks"      : v_ks,
        "kx"        : kx,
        "ky"        : ky,
        "KX"        : KX,
        "KY"        : KY,
        "epsilon_ks": epsilon_ks,
        "sqrt_sin"  : sqrt_sin,
        "converged" : converged,
        "n_iter"    : nfev_total,
        "beta"      : beta,
        "kBT"       : kBT_local,
        "residuals" : res_check,
    }


# ---------------------------------------------------------------------------
# Self-test: compare fsolve vs iterative formula directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("SELF-TEST: fsolve vs iterative formula at T=0")
    print("=" * 65)

    res = run_selfconsistency(seed=42, kBT_override=0.0)

    print(f"\nFinal solution:")
    print(f"  Delta0 = {res['Delta0']:.8f}")
    print(f"  DeltaS = {res['DeltaS']:.8f}")
    print(f"  mu     = {res['mu']:.8f}")
    print(f"  converged = {res['converged']}")