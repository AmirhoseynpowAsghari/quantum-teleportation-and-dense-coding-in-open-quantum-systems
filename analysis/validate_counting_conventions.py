#!/usr/bin/env python3
"""Phase-1 counting-convention validation against arXiv:1801.08988.

Resolves the factor-of-two discrepancy between the paper's gap equations
(Eqs. 32-34, 38) and self_consistency.residual by testing four independent
properties:

V1  MASTER-EQUATION TEST
    Solve both conventions, then evaluate the paper's master gap equation
    Eq. (30)  Delta_bar(k,s) = -(1/2N) sum_{k's'} B_ss'(k,k') Delta_bar(k's') g(E)
    pointwise (no ansatz projection).  The correct convention must satisfy
    it to machine precision; a wrong counting leaves an O(Delta) defect.

V2  IDENTITY TEST
    legacy(U, Dt) must be *exactly* paper(U/2, Dt/2) -- pinning the
    discrepancy as a uniform factor 1/2 in the two gap equations.

V3  V_SO = 0 DEGENERACY TEST (user hypothesis)
    (a) helicity-branch sums equal spin sums to machine precision,
    (b) n capacity: one branch caps at n=1.5, two branches reach n=2
        (n=1.875 is impossible with a single branch),
    (c) alternative summation conventions (half-BZ reading, extra
        degeneracy factor 2) collapse onto the legacy solution; only the
        paper's full-BZ convention satisfies Eq. (30).

V4  PAPER-FIGURE ANCHOR (Fig. 4 regime: attractive Hubbard, Dt=0)
    The paper's Tc(n) at U<0, Dt=0 is textbook BCS.  Reproduce it:
    paper convention == textbook to ~1e-10; legacy == textbook at U/2.

Run:  .venv/bin/python analysis/validate_counting_conventions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, fsolve, least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "reports"


# ---------------------------------------------------------------------------
# Grid (identical convention to kspace_setup.build_kgrid)
# ---------------------------------------------------------------------------
def make_grid(Nk: int, t: float = 1.0, Vso: float = 0.0):
    kx = np.linspace(-np.pi, np.pi, Nk)
    KX, KY = np.meshgrid(kx, kx)
    ssin = np.sqrt(np.sin(KX) ** 2 + np.sin(KY) ** 2)
    sk = 0.5 * (np.cos(KX) + np.cos(KY))
    eps = np.empty((Nk, Nk, 2))
    eps[..., 0] = -2 * t * (np.cos(KX) + np.cos(KY)) - 2 * Vso * ssin  # s=+1
    eps[..., 1] = -2 * t * (np.cos(KX) + np.cos(KY)) + 2 * Vso * ssin  # s=-1
    return sk, ssin, eps


def _th(beta, E):
    if np.isinf(beta):
        return np.ones_like(E)
    return np.tanh(np.clip(0.5 * beta * E, 0.0, 500.0))


SP = np.array([1.0, -1.0])


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------
def residual_paper(x, sk, ssin, eps, Nk, t, U, Dt, Vso, n, beta):
    """Paper-faithful T=0/finite-T gap+number equations.

    Eqs. (32)-(34) with the ansatz Eq. (37) (two-parameter form) and Eq. (38):
      Delta0   = -(1/4N) sum_{k,s} [U + 8 Dt s_k + 4 (Dt/t) Vso s lam] Db/E
      DeltaS   = -(2 Dt/N) sum_{k,s} Db * th/E
      n        = 1 - (1/2N) sum_{k,s} (eps-mu)/E * th
    """
    D0, DS, mu = float(x[0]), float(x[1]), float(x[2])
    Dks = D0 - (DS / (4.0 * t)) * eps
    E = np.sqrt(Dks ** 2 + (eps - mu) ** 2)
    E = np.clip(E, 1e-12, None)
    th = _th(beta, E)
    thE = th / E
    kernel = (
        U
        + 8.0 * Dt * sk[..., None]
        + 4.0 * (Dt / t) * Vso * SP[None, None, :] * ssin[..., None]
    )
    N2 = float(Nk * Nk)
    r1 = D0 + np.sum(kernel * Dks * thE) / (4.0 * N2)
    r2 = DS + 2.0 * Dt * np.sum(Dks * thE) / N2
    r3 = np.sum((eps - mu) * thE) / (2.0 * N2) - (1.0 - n)
    return np.array([r1, r2, r3])


def residual_legacy(x, sk, ssin, eps, Nk, t, U, Dt, Vso, n, beta):
    """Exact copy of the pre-fix self_consistency.residual (factor 1/2 in
    r1 and r2; number equation unchanged)."""
    D0, DS, mu = float(x[0]), float(x[1]), float(x[2])
    Dks = D0 - (DS / (4.0 * t)) * eps
    E = np.sqrt(Dks ** 2 + (eps - mu) ** 2)
    E = np.clip(E, 1e-12, None)
    th = _th(beta, E)
    thE = th / E
    kernel = (
        U
        + 8.0 * Dt * sk[..., None]
        + 4.0 * (Dt / t) * Vso * SP[None, None, :] * ssin[..., None]
    )
    N2 = float(Nk * Nk)
    r1 = D0 + np.sum(kernel * Dks * thE) / (8.0 * N2)
    r2 = DS + Dt * np.sum(Dks * thE) / N2
    r3 = np.sum((eps - mu) * thE) / (2.0 * N2) - (1.0 - n)
    return np.array([r1, r2, r3])


def residual_scaled(x, sk, ssin, eps, Nk, t, U, Dt, Vso, n, beta, scale):
    """Paper equations with the momentum *sums* in the two gap equations
    multiplied by `scale` (inhomogeneous D0/DS terms untouched).

    scale=0.5 -> 'half-BZ reading of the paper sums' (integrand even in k)
                 which is algebraically identical to the legacy residual.
    scale=2   -> 'full BZ plus an extra degeneracy factor 2'.
    """
    D0, DS, mu = float(x[0]), float(x[1]), float(x[2])
    Dks = D0 - (DS / (4.0 * t)) * eps
    E = np.clip(np.sqrt(Dks ** 2 + (eps - mu) ** 2), 1e-12, None)
    thE = _th(beta, E) / E
    kernel = (
        U
        + 8.0 * Dt * sk[..., None]
        + 4.0 * (Dt / t) * Vso * SP[None, None, :] * ssin[..., None]
    )
    N2 = float(Nk * Nk)
    r1 = D0 + scale * np.sum(kernel * Dks * thE) / (4.0 * N2)
    r2 = DS + scale * 2.0 * Dt * np.sum(Dks * thE) / N2
    r3 = np.sum((eps - mu) * thE) / (2.0 * N2) - (1.0 - n)
    return np.array([r1, r2, r3])


# ---------------------------------------------------------------------------
# Solver (residual-gated: fsolve then LM, best of several starts)
# ---------------------------------------------------------------------------
def solve(resfn, args, starts, xtol=1.49e-10):
    best, best_r = None, np.inf
    for x0 in starts:
        for method in ("fsolve", "lm"):
            try:
                if method == "fsolve":
                    sol, info, ier, _ = fsolve(
                        resfn, x0, args=args, full_output=True, xtol=xtol,
                        maxfev=8000,
                    )
                    cand = np.asarray(sol, float)
                else:
                    from scipy.optimize import least_squares as ls
                    sol = ls(resfn, x0, args=args, method="lm",
                            ftol=1e-14, xtol=1e-14, gtol=1e-14)
                    cand = np.asarray(sol.x, float)
            except Exception:
                continue
            r = float(np.max(np.abs(resfn(cand, *args))))
            if r < best_r:
                best, best_r = cand, r
            if best_r < 1e-11:
                return best, best_r
    return best, best_r


def multi_starts(eps, nseed=4, seed=0):
    rng = np.random.default_rng(seed)
    mu0 = float(np.mean(eps))
    base = [
        np.array([0.7, 0.48, mu0]),
        np.array([0.35, 0.2, mu0]),
        np.array([1.2, 0.8, mu0]),
        np.array([0.05, 0.02, mu0]),
    ]
    for _ in range(nseed):
        base.append(np.array([
            0.1 + 1.6 * rng.random(),
            0.05 + 1.0 * rng.random(),
            float(rng.uniform(eps.min(), eps.max())),
        ]))
    return base


# ---------------------------------------------------------------------------
# V1: master Eq. (30) pointwise residual (ansatz not assumed)
# ---------------------------------------------------------------------------
def eq30_max_residual(x, sk, ssin, eps, Nk, t, U, Dt, Vso, beta):
    D0, DS, mu = float(x[0]), float(x[1]), float(x[2])
    Dbar = D0 - (DS / (4.0 * t)) * eps
    E = np.sqrt(Dbar ** 2 + (eps - mu) ** 2)
    E = np.clip(E, 1e-12, None)
    g = _th(beta, E) / (2.0 * E)
    N2 = float(Nk * Nk)
    # sum_{k's'} B_ss'(k,k') Db(k's') g(k's') = C_s(k)*P1 + P2
    P1 = float(np.sum(Dbar * g))
    Dpart = (
        8.0 * Dt * sk[..., None]
        + 4.0 * (Dt / t) * Vso * SP[None, None, :] * ssin[..., None]
    )
    P2 = float(np.sum(Dpart * Dbar * g))
    C = (
        U
        + 8.0 * Dt * sk[..., None]
        + 4.0 * (Dt / t) * Vso * SP[None, None, :] * ssin[..., None]
    )
    rhs = -(1.0 / (2.0 * N2)) * (C * P1 + P2)
    diff = Dbar - rhs
    scale = max(abs(D0), 1e-12)
    return float(np.max(np.abs(diff))), float(np.max(np.abs(diff)) / scale)


# ---------------------------------------------------------------------------
# V4: Tc machinery (Delta -> 0 linearized)
# ---------------------------------------------------------------------------
def mu_from_number(n, Nk, t, Vso, beta, U=0.0, Dt=0.0):
    """mu from Eq. (38) in the normal state (Delta = 0)."""
    _, _, eps = make_grid(Nk, t, Vso)

    def num_eq(mu):
        E = np.clip(np.abs(eps - mu), 1e-12, None)
        return float(np.sum((eps - mu) / E * _th(beta, E)) / (2.0 * Nk * Nk)
                     - (1.0 - n))

    return brentq(num_eq, eps.min() - 1.0, eps.max() + 1.0, xtol=1e-13)


def tc_from_condition(cond, n, Nk, t, Vso, Uabs):
    """Solve cond(T) = 0 for Tc.  cond has the form A*sum - 1 with the
    normal-state mu(n, T)."""
    def F(T):
        beta = 1.0 / T
        mu = mu_from_number(n, Nk, t, Vso, beta)
        _, _, eps = make_grid(Nk, t, Vso)
        E = np.clip(np.abs(eps - mu), 1e-12, None)
        th = _th(beta, E)
        return cond(mu, eps, th, E, Nk, Uabs)

    lo, hi = 1e-4, 3.0
    if F(lo) * F(hi) > 0:
        return None
    return brentq(F, lo, hi, xtol=1e-12)


def tc_paper(n, Nk, t, Vso, Uabs):
    # linearized r1: 1 = (Uabs/4N) sum_{k,s} th/E
    return tc_from_condition(
        lambda mu, eps, th, E, Nk, U: (U / (4.0 * Nk * Nk)) * np.sum(th / E) - 1.0,
        n, Nk, t, Vso, Uabs)


def tc_legacy(n, Nk, t, Vso, Uabs):
    # linearized legacy r1: 1 = (Uabs/8N) sum_{k,s} th/E
    return tc_from_condition(
        lambda mu, eps, th, E, Nk, U: (U / (8.0 * Nk * Nk)) * np.sum(th / E) - 1.0,
        n, Nk, t, Vso, Uabs)


def tc_textbook(n, Nk, t, Vso, Uabs):
    # Textbook attractive-BCS, implemented independently of the paper's
    # helicity sums: spin-degenerate, full BZ, one pair per k:
    #   1 = (Uabs/N) sum_k tanh(beta|xi|/2) / (2|xi|)
    # evaluated on a single band (s=+1) only.
    def cond(mu, eps, th, E, Nk, U):
        Es = E[..., 0]
        ths = th[..., 0]
        return (U / (2.0 * Nk * Nk)) * np.sum(ths / Es) - 1.0
    return tc_from_condition(cond, n, Nk, t, Vso, Uabs)


# ---------------------------------------------------------------------------
def main():
    report = {}
    print("=" * 78)
    print("PHASE 1: counting-convention validation vs arXiv:1801.08988")
    print("=" * 78)

    # ---------------- V1: master Eq. (30) ----------------
    Nk, t, U, Dt, Vso, n = 31, 1.0, 75.0, 4.5, 0.5, 1.875
    beta = np.inf
    sk, ssin, eps = make_grid(Nk, t, Vso)
    args_p = (sk, ssin, eps, Nk, t, U, Dt, Vso, n, beta)
    starts = multi_starts(eps)

    x_paper, r_paper = solve(residual_paper, args_p, starts,)
    x_leg, r_leg = solve(residual_legacy, args_p, starts,)
    e30_p = eq30_max_residual(x_paper, sk, ssin, eps, Nk, t, U, Dt, Vso, beta)
    e30_l = eq30_max_residual(x_leg, sk, ssin, eps, Nk, t, U, Dt, Vso, beta)

    print("\nV1  Master Eq. (30) pointwise test "
          f"(Nk={Nk}, U={U}, Dt={Dt}, Vso={Vso}, n={n}, T=0)")
    print(f"  paper convention : solve|r|={r_paper:.2e}  "
          f"Eq30 max|res|={e30_p[0]:.3e}  rel={e30_p[1]:.3e}")
    print(f"  legacy convention: solve|r|={r_leg:.2e}  "
          f"Eq30 max|res|={e30_l[0]:.3e}  rel={e30_l[1]:.3e}")
    print(f"  paper solution   : D0={x_paper[0]: .8f} DS={x_paper[1]: .8f} "
          f"mu={x_paper[2]: .8f}")
    print(f"  legacy solution  : D0={x_leg[0]: .8f} DS={x_leg[1]: .8f} "
          f"mu={x_leg[2]: .8f}")
    report["V1"] = {
        "paper": {"x": x_paper.tolist(), "solve_res": r_paper,
                  "eq30_abs": e30_p[0], "eq30_rel": e30_p[1]},
        "legacy": {"x": x_leg.tolist(), "solve_res": r_leg,
                   "eq30_abs": e30_l[0], "eq30_rel": e30_l[1]},
    }

    # ---------------- V2: identity legacy(U,Dt) == paper(U/2,Dt/2) ------------
    args_leg2 = (sk, ssin, eps, Nk, t, U / 2, Dt / 2, Vso, n, beta)
    x_leg2, r_leg2 = solve(residual_paper, args_leg2, starts,)
    d_id = float(np.max(np.abs(x_leg - x_leg2)))

    # mirror: paper(U,Dt) == legacy(2U,2Dt)
    args_leg3 = (sk, ssin, eps, Nk, t, 2 * U, 2 * Dt, Vso, n, beta)
    x_leg3, r_leg3 = solve(residual_legacy, args_leg3, starts,)
    d_id2 = float(np.max(np.abs(x_paper - x_leg3)))

    print("\nV2  Identity test")
    print(f"  max|legacy(75,4.5) - paper(37.5,2.25)| = {d_id:.3e}")
    print(f"  max|paper(75,4.5) - legacy(150,9.0)|    = {d_id2:.3e}")
    report["V2"] = {"legacy_of_U_Dt_vs_paper_half": d_id,
                    "paper_vs_legacy_double": d_id2}

    # ---------------- V3: V_SO = 0 hypothesis test ----------------
    print("\nV3  V_SO = 0 degeneracy / alternative-convention test")
    V0 = 0.0
    sk0, ss0, eps0 = make_grid(201, t, V0)
    # (a) helicity sums == spin sums
    rng = np.random.default_rng(7)
    mu_t = float(rng.uniform(-3, 3))
    T_t = 0.13
    b_t = 1.0 / T_t
    f_hel = np.sum(_th(b_t, np.abs(eps0 - mu_t)))
    eps_spin = -2 * t * (np.cos(np.linspace(-np.pi, np.pi, 201)[None, :])
                         + np.cos(np.linspace(-np.pi, np.pi, 201)[:, None]))
    f_spin = 2.0 * np.sum(_th(b_t, np.abs(eps_spin - mu_t)))
    deg_diff = float(abs(f_hel - f_spin) / max(abs(f_spin), 1.0))
    # (b) n capacity with one vs two branches (mu -> above band top)
    N2 = 201.0 ** 2
    mu_hi = float(eps0.max() + 5.0)
    E_hi = np.clip(np.abs(eps0 - mu_hi), 1e-12, None)
    n_two = 1.0 - np.sum((eps0 - mu_hi) / E_hi) / (2.0 * N2)
    n_one = 1.0 - np.sum((eps0[..., 0] - mu_hi) / E_hi[..., 0]) / (2.0 * N2)
    print(f"  (a) |sum_s f(eps_ks) - sum_sigma f(eps_k)|/norm = {deg_diff:.3e}")
    print(f"  (b) n(mue->top): two branches={n_two:.6f}  "
          f"one branch={n_one:.6f}  (n=1.875 needs two)")
    # (c) alternative conventions: grid MUST be rebuilt at Vso=0
    skz, ssinz, epsz = make_grid(Nk, t, 0.0)
    args_c = (skz, ssinz, epsz, Nk, t, U, Dt, 0.0, n, beta)
    starts_z = multi_starts(epsz, seed=11)
    x_half, r_half = solve(residual_scaled, args_c + (0.5,), starts_z)
    x_deg, r_deg = solve(residual_scaled, args_c + (2.0,), starts_z)
    x_leg0, r_leg0 = solve(residual_legacy, args_c, starts_z)
    # half-BZ reading is algebraically the legacy residual -> same roots
    d_half = float(np.max(np.abs(x_half - x_leg0)))
    # degeneracy x2 == paper equations at (2U, 2 Dt)
    args_2U = (skz, ssinz, epsz, Nk, t, 2 * U, 2 * Dt, 0.0, n, beta)
    x_pap2, r_pap2 = solve(residual_paper, args_2U, starts_z)
    d_deg = float(np.max(np.abs(x_deg - x_pap2)))
    # degeneracy x2 must itself violate Eq. (30)
    e30_deg = eq30_max_residual(x_deg, skz, ssinz, epsz, Nk, t, U, Dt, 0.0, beta)
    # paper at Vso=0 satisfies Eq. (30)
    x_pap0, r_pap0 = solve(residual_paper, args_c, starts_z)
    e30_p0 = eq30_max_residual(x_pap0, skz, ssinz, epsz, Nk, t, U, Dt, 0.0, beta)
    print(f"  (c) half-BZ reading   vs legacy       : max|dx| = {d_half:.3e} "
          f"(solve|r|={max(r_half, r_leg0):.1e})")
    print(f"      degeneracy x2     vs paper(2U,2Dt): max|dx| = {d_deg:.3e} "
          f"(solve|r|={max(r_deg, r_pap2):.1e})")
    print(f"      degeneracy x2 Eq30 max|res|       = {e30_deg[0]:.3e} "
          f"(must be O(Delta))")
    print(f"      paper(Vso=0) solve|r|={r_pap0:.1e}  "
          f"Eq30 max|res| = {e30_p0[0]:.3e}")
    report["V3"] = {"helicity_vs_spin_sum_rel": deg_diff,
                    "n_two_branches": n_two, "n_one_branch": n_one,
                    "halfBZ_vs_legacy": d_half,
                    "deg2_vs_paper_double": d_deg,
                    "deg2_eq30": e30_deg[0],
                    "paper_vso0_solve_res": r_pap0,
                    "paper_vso0_eq30": e30_p0[0],
                    "paper_vso0_x": x_pap0.tolist()}

    # ---------------- V4: Fig.4 regime Tc anchor ----------------
    print("\nV4  Attractive-Hubbard Tc anchor (paper Fig. 4 regime: "
          "Dt=0, Vso=0, U=-4t, n=0.5)")
    NkT, nT, Uabs = 401, 0.5, 4.0
    Tc_p = tc_paper(nT, NkT, t, 0.0, Uabs)
    Tc_l = tc_legacy(nT, NkT, t, 0.0, Uabs)
    Tc_tb = tc_textbook(nT, NkT, t, 0.0, Uabs)
    Tc_tb2 = tc_textbook(nT, NkT, t, 0.0, Uabs / 2)
    print(f"  Tc paper     (U=-4) = {Tc_p:.10f}")
    print(f"  Tc textbook  (U=-4) = {Tc_tb:.10f}   "
          f"|diff| = {abs(Tc_p - Tc_tb):.3e}")
    print(f"  Tc legacy    (U=-4) = {Tc_l:.10f}")
    print(f"  Tc textbook  (U=-2) = {Tc_tb2:.10f}   "
          f"|diff| = {abs(Tc_l - Tc_tb2):.3e}")
    print(f"  Tc legacy vs textbook(U=-4): |diff| = "
          f"{abs(Tc_l - Tc_tb):.6f}  (must be large)")
    report["V4"] = {"tc_paper_U4": Tc_p, "tc_textbook_U4": Tc_tb,
                    "tc_legacy_U4": Tc_l, "tc_textbook_U2": Tc_tb2}

    # ---------------- post-fix cross-check (if fixed already) ---------------
    try:
        import self_consistency as sc
        if hasattr(sc, "residual"):
            # self_consistency.residual arg order: (eps, sk, ssin, ...)
            r_fix = sc.residual(x_paper, eps, sk, ssin, Nk, t, U, Dt, Vso,
                                n, beta)
            d_fix = float(np.max(np.abs(
                np.asarray(r_fix) - residual_paper(x_paper, *args_p))))
            print(f"\nFIX-CHECK  max|self_consistency.residual - "
                  f"residual_paper| at paper solution = {d_fix:.3e}")
            report["fix_check"] = d_fix
    except Exception as exc:  # pragma: no cover
        print(f"\nFIX-CHECK skipped: {exc}")

    # ---------------- verdicts ----------------
    ok = (
        report["V1"]["paper"]["eq30_rel"] < 1e-9
        and report["V1"]["legacy"]["eq30_rel"] > 1e-3
        and report["V2"]["legacy_of_U_Dt_vs_paper_half"] < 1e-8
        and report["V2"]["paper_vs_legacy_double"] < 1e-8
        and report["V3"]["helicity_vs_spin_sum_rel"] < 1e-12
        and abs(report["V3"]["n_two_branches"] - 2.0) < 1e-9
        and abs(report["V3"]["n_one_branch"] - 1.5) < 1e-9
        and report["V3"]["halfBZ_vs_legacy"] < 1e-8
        and report["V3"]["deg2_vs_paper_double"] < 1e-8
        and report["V3"]["deg2_eq30"] > 1e-3
        and report["V3"]["paper_vso0_eq30"] < 1e-9
        and abs(Tc_p - Tc_tb) < 1e-8
        and abs(Tc_l - Tc_tb2) < 1e-8
        and abs(Tc_l - Tc_tb) > 1e-3
    )
    report["all_checks_passed"] = bool(ok)
    print("\n" + "=" * 78)
    print(f"VERDICT: {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    print("=" * 78)

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "counting_validation.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
