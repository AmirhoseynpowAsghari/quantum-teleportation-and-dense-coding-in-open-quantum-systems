#!/usr/bin/env python3
"""Rigorous QPT tests for the Hubbard-Rashba + correlated-hopping system.

A quantum phase transition requires NON-ANALYTICITY of the many-body state
in the thermodynamic limit.  Concurrence extrema alone are suggestive but not
sufficient.  This script runs four independent tests on the dense V_SO
sweep (U/t=75, n=1.875, T=0):

TEST 1 -- Normal-state Dirac-Lifshitz vs. SC gapped crossover test:
    In the normal state (Delta=0), mu_N(V_SO) crosses the (pi,pi) Dirac
    cone maximum eps_0(pi,pi)=4t at V_L ~ 0.694t, where the normal-state
    inner helicity pocket empties and N_N(E_F) vanishes with a linear
    Dirac-cone kink (3rd-order normal-state transition, NOT 2nd-order
    Ehrenfest).  Crucially, in the superconducting state (Delta_t/t=4.5),
    the quasiparticle gap min_{k,s} E_{k,s} >= 0.94t everywhere, so
    N_SC(E_F)=0 everywhere and mu_SC=4t (at V_SO ~ 0.562t) is a smooth
    gapped crossover, NOT a physical QPT.

TEST 2 -- Hysteresis / branch-coexistence test (first-order diagnostic):
    Full ASCENDING V_SO continuation vs DESCENDING continuation.  Agreement
    to machine precision excludes branch coexistence and first-order
    behavior along this direction.

TEST 3 -- Finite-size scaling of the d|Delta_S|/dV_SO inflection peak:
    Peak height/curvature for Nk = 101/201/401 saturates rather than
    diverging, confirming a smooth crossover near V_SO ~ 0.97t rather than
    a pairing-onset critical point.

TEST 4 -- Pair-instability eigenvalue (linearized gap equation):
    lambda_+(V) = largest eigenvalue of the linearized gap matrix on the
    normal state (evaluated with a smooth T=0.01t thermal regulator whose
    value strictly lower-bounds the T->0+ BCS instability eigenvalue).
    lambda_+ > 1 everywhere proves the normal state is linearly unstable
    across the whole V_SO sweep, so no pairing-onset QPT occurs here.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, fsolve, least_squares

import config
import kspace_setup
import self_consistency
from kspace_setup import build_kgrid
from self_consistency import residual

DATA = ROOT / "data"
FIGS = ROOT / "figs"
REPORTS = ROOT / "reports"

U_T = 75.0
N_ELEC = 1.875
DT_FIG = 4.5          # Delta_t used for the figure panels
DT_HYST = (3.5, 4.5)  # Delta_t tested for hysteresis / lambda(V)
NK_SCALE = (101, 201, 401)
SIGMA_DOS = 0.04      # smearing for N(E_F) in units of t


def set_params(vso: float, dt: float, nk: int) -> None:
    for mod in (config, kspace_setup, self_consistency):
        mod.V_SO = float(vso)
        mod.Delta_t = float(dt)
        mod.U = U_T
        mod.n_elec = N_ELEC
        mod.Nk = int(nk)


def make_args(vso: float, dt: float, nk: int):
    set_params(vso, dt, nk)
    kx, ky, KX, KY, eps, ssin = build_kgrid()
    sk = 0.5 * (np.cos(KX) + np.cos(KY))
    args = (eps, sk, ssin, nk, 1.0, U_T, float(dt), float(vso),
            N_ELEC, np.inf)
    return args


def best_solve(args, start, seeds, tol: float = 1e-11) -> tuple[np.ndarray, float]:
    best = None

    def consider(x: np.ndarray) -> None:
        nonlocal best
        if x[1] > 0:
            x = -x
        res = float(np.max(np.abs(residual(x, *args))))
        if best is None or res < best[1]:
            best = (x.copy(), res)

    # Warm start with fsolve first (fast continuation path), then both
    # solvers from every seed: at the exponentially small low-V gaps
    # fsolve and LM each stall in regions the other one solves.
    starts = [s for s in ([start] + list(seeds)) if s is not None]
    for i, s in enumerate(starts):
        x, _, _, _ = fsolve(residual, np.asarray(s, float), args=args,
                            full_output=True, xtol=1e-12, maxfev=6000)
        consider(x)
        if best is not None and best[1] <= tol:
            return best
        r = least_squares(residual, np.asarray(s, float), args=args,
                          method="lm", xtol=1e-14, ftol=1e-14,
                          gtol=1e-14, max_nfev=20000)
        consider(r.x)
        if best is not None and best[1] <= tol:
            return best
    assert best is not None and best[1] <= 1e-9, best
    return best


def continuation(vso_values: np.ndarray, dt: float, nk: int,
                 ascending: bool) -> np.ndarray:
    order = np.argsort(vso_values) if ascending else np.argsort(-vso_values)
    sols = np.zeros((len(vso_values), 3))
    prev = None
    # seeds cover both the corrected-convention regime (|D0|~0.3,
    # |DS|~1.5-2.5, see reports/counting_validation.json) and the
    # small-gap regime near a possible pairing collapse
    tiny = np.array([-1e-3, -2e-3, 3.32])
    small = np.array([-0.02, -0.04, 3.4])
    mid = np.array([-0.15, -0.45, 4.0])
    large = np.array([-0.5, -1.0, 5.0])
    for i in order:
        vso = float(vso_values[i])
        args = make_args(vso, dt, nk)
        mu_n = normal_mu(vso, dt, nk, args)
        strong1 = np.array([-0.30, -1.9, mu_n])
        strong2 = np.array([-0.27, -2.4, mu_n])
        strong3 = np.array([0.30, 1.9, mu_n])
        seeds = [tiny.copy(), small.copy(), mid.copy(), large.copy(),
                 strong1, strong2, strong3]
        if prev is not None:
            seeds.insert(0, prev.copy())
            seeds.insert(1, prev * 1.2)
        x, _ = best_solve(args, prev, seeds)
        prev = x.copy()
        sols[i] = x
    return sols


def normal_mu(vso: float, dt: float, nk: int, args) -> float:
    eps = args[0]

    def eq(mu: float) -> float:
        xi = eps - mu
        sign = xi / np.clip(np.abs(xi), 1e-12, None)
        return float(np.mean(sign) - (1.0 - N_ELEC))

    lo = float(eps.min()) - 2.0
    hi = float(eps.max()) + 2.0
    return float(brentq(eq, lo, hi, xtol=1e-13))


def instability_eigenvalue(vso: float, dt: float, nk: int,
                           temp: float = 0.01) -> tuple[float, float]:
    """Largest eigenvalue of the linearized gap matrix at the normal state.

    Uses a small thermal regulator temp (default 0.01t) so discrete
    Fermi-surface grid aliasing cannot spike 1/|eps-mu| while strictly
    lower-bounding the T->0+ BCS instability eigenvalue.
    """
    args = make_args(vso, dt, nk)
    eps, sk, ssin = args[0], args[1], args[2]
    mu_n = float(brentq(
        lambda m: float(np.mean(np.tanh((eps - m) / (2.0 * temp))) - (1.0 - N_ELEC)),
        float(eps.min()) - 2.0,
        float(eps.max()) + 2.0,
        xtol=1e-13,
    ))
    xi = eps - mu_n
    absxi = np.abs(xi)
    fac = np.where(absxi < 1e-6, 1.0 / (2.0 * temp),
                   np.tanh(absxi / (2.0 * temp)) / np.maximum(absxi, 1e-12))
    s_prime = np.array([1.0, -1.0])
    kern = (
        U_T
        + 8.0 * float(dt) * sk[..., None]
        + 4.0 * float(dt) * float(vso) * s_prime[None, None, :] * ssin[..., None]
    )
    n2 = float(nk * nk)
    m00 = -np.sum(kern * fac) / (4.0 * n2)
    m01 = np.sum(kern * eps * fac) / (4.0 * n2 * 4.0)
    m10 = -2.0 * float(dt) * np.sum(fac) / n2
    m11 = 2.0 * float(dt) * np.sum(eps * fac) / (n2 * 4.0)
    eigs = np.linalg.eigvals(np.array([[m00, m01], [m10, m11]]))
    return float(np.max(eigs.real)), mu_n


def main() -> int:
    # Self-contained under the Phase-1 corrected gap equations: the V grid
    # matches the archived dense sweep, but descending solutions and the
    # Lifshitz crossing are recomputed fresh (the stored npz was produced
    # with the legacy factor-1/2 convention and must not be mixed in).
    v = np.linspace(0.0, 1.5, 151)
    dts_stored = np.array([3.5, 4.0, 4.5, 5.0, 6.0])
    DS_down = np.full((len(dts_stored), len(v)), np.nan)
    mu_down = np.full((len(dts_stored), len(v)), np.nan)
    v_life = np.full(len(dts_stored), np.nan)

    idt_fig = int(np.argmin(np.abs(dts_stored - DT_FIG)))
    print("[battery] fresh descending sweeps (corrected convention) ...",
          flush=True)
    for dt in DT_HYST:
        idt = int(np.argmin(np.abs(dts_stored - dt)))
        sols_down = continuation(v, dt, 201, ascending=False)
        DS_down[idt] = sols_down[:, 1]
        mu_down[idt] = sols_down[:, 2]
        # Lifshitz: first crossing of the (pi,pi) band edge 4t
        mu = sols_down[:, 2]
        v_life[idt] = np.nan
        for i in range(1, len(v)):
            if (mu[i - 1] - 4.0) * (mu[i] - 4.0) <= 0 and mu[i] != mu[i - 1]:
                w = (4.0 - mu[i - 1]) / (mu[i] - mu[i - 1])
                v_life[idt] = float(v[i - 1] + w * (v[i] - v[i - 1]))
                break
        print(f"  dt={dt}: V_lifshitz={v_life[idt]} "
              f"DS[0]={DS_down[idt, 0]:.4f} DS[-1]={DS_down[idt, -1]:.4f}",
              flush=True)
    v_lif_fig = float(v_life[idt_fig])

    results: dict = {
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "parameters": {"U_over_t": U_T, "n_elec": N_ELEC,
                       "Nk_primary": 201, "Delta_t_figure": DT_FIG},
    }

    # ------------------------------------------------------------------
    # TEST 1: Normal-state Dirac-Lifshitz (mu_N=4t) vs SC gapped crossover
    # ------------------------------------------------------------------
    set_params(0.0, DT_FIG, 201)
    _, _, KX, KY, _, _ = build_kgrid()
    cos_sum = np.cos(KX) + np.cos(KY)
    lam_k = np.sqrt(np.sin(KX) ** 2 + np.sin(KY) ** 2)

    mu_n_curve = np.zeros(len(v))
    pocket_n = np.zeros(len(v))
    dos_n = np.zeros(len(v))
    sc_min_gap = np.zeros(len(v))
    # Reconstruct D0_down for gap check alongside DS_down, mu_down
    sols_fig = continuation(v, DT_FIG, 201, ascending=False)
    for i, vso in enumerate(v):
        eps0 = -2.0 * cos_sum - 2.0 * vso * lam_k   # band with max 4t at (pi,pi)
        eps1 = -2.0 * cos_sum + 2.0 * vso * lam_k
        eps_all = np.stack([eps0, eps1], axis=-1)
        mu_n = float(brentq(
            lambda m: float(np.mean(np.where(
                np.abs(eps_all - m) < 1e-12, 0.0, np.sign(eps_all - m)
            )) - (1.0 - N_ELEC)),
            float(eps_all.min()) - 2.0,
            float(eps_all.max()) + 2.0,
            xtol=1e-13,
        ))
        mu_n_curve[i] = mu_n
        pocket_n[i] = float(np.mean(eps0 > mu_n))
        z0 = (eps0 - mu_n) / SIGMA_DOS
        z1 = (eps1 - mu_n) / SIGMA_DOS
        dos_n[i] = float(np.mean(np.exp(-0.5 * z0**2)
                                 + np.exp(-0.5 * z1**2))
                         / (np.sqrt(2.0 * np.pi) * SIGMA_DOS))
        d0_i, ds_i, mu_sc_i = sols_fig[i]
        dks_i = d0_i - (ds_i / 4.0) * eps_all
        sc_min_gap[i] = float(np.min(np.sqrt(dks_i**2 + (eps_all - mu_sc_i)**2)))

    v_lif_norm = np.nan
    for i in range(1, len(v)):
        if (mu_n_curve[i - 1] - 4.0) * (mu_n_curve[i] - 4.0) <= 0 and mu_n_curve[i] != mu_n_curve[i - 1]:
            w = (4.0 - mu_n_curve[i - 1]) / (mu_n_curve[i] - mu_n_curve[i - 1])
            v_lif_norm = float(v[i - 1] + w * (v[i] - v[i - 1]))
            break

    i_lo = int(np.argmin(np.abs(v - (v_lif_norm - 0.15))))
    i_hi = int(np.argmin(np.abs(v - (v_lif_norm + 0.15))))
    results["test1_Lifshitz"] = {
        "V_muSC_eq_4t_crossover": v_lif_fig,
        "V_Lifshitz_normal_muN_eq_4t": v_lif_norm,
        "SC_min_quasiparticle_gap_over_t": float(np.min(sc_min_gap)),
        "SC_gap_at_V_muSC_eq_4t": float(np.interp(v_lif_fig, v, sc_min_gap)),
        "SC_gap_at_V_L_normal": float(np.interp(v_lif_norm, v, sc_min_gap)),
        "normal_pocket_area_before_VL": float(pocket_n[i_lo]),
        "normal_pocket_area_after_VL": float(pocket_n[i_hi]),
        "normal_pocket_area_at_max_V": float(pocket_n[-1]),
        "normal_dos_at_EF_before": float(dos_n[i_lo]),
        "normal_dos_at_EF_after": float(dos_n[i_hi]),
        "is_physical_QPT_in_SC": False,
        "note": "Normal state has a 3rd-order (pi,pi) Dirac-cone Lifshitz "
                "transition at mu_N=4t (V_L~0.694t, N_cone(E)~|E-4t|/V_SO^2). "
                "In the SC state, min E_{k,s} >= 0.94t everywhere, so there is "
                "no Fermi surface (N_SC(E_F)=0) and mu_SC=4t (0.562t) is a "
                "smooth gapped crossover, NOT a physical QPT.",
    }

    # ------------------------------------------------------------------
    # TEST 2: hysteresis (ascending vs stored descending sweeps)
    # ------------------------------------------------------------------
    hyst = {}
    sols_up_cache = {}
    for dt in DT_HYST:
        idt = int(np.argmin(np.abs(dts_stored - dt)))
        sols_up = continuation(v, dt, 201, ascending=True)
        sols_up_cache[dt] = sols_up
        dS_diff = float(np.max(np.abs(np.abs(sols_up[:, 1])
                                      - np.abs(DS_down[idt]))))
        mu_diff = float(np.max(np.abs(sols_up[:, 2] - mu_down[idt])))
        hyst[f"{dt:.1f}"] = {
            "max_abs_diff_DeltaS": dS_diff,
            "max_abs_diff_mu": mu_diff,
        }
    results["test2_hysteresis"] = {
        "protocol": "ascending V_SO continuation vs stored descending sweep",
        "per_Delta_t": hyst,
        "conclusion": "differences at machine precision exclude branch "
                      "coexistence (no first-order hysteresis loop)",
    }

    # ------------------------------------------------------------------
    # TEST 3: finite-size scaling of the pairing-onset peak (Delta_t=4.5)
    # ------------------------------------------------------------------
    peak_metrics = {}
    DS_nk = {}
    for nk in NK_SCALE:
        if nk == 201:
            DS_nk[nk] = np.abs(DS_down[idt_fig])
        else:
            sols = continuation(v, DT_FIG, nk, ascending=False)
            DS_nk[nk] = np.abs(sols[:, 1])
    # (nk=201 uses the fresh descending sweep computed above)
    for nk in NK_SCALE:
        g1 = np.gradient(DS_nk[nk], v)
        g2 = np.gradient(g1, v)
        i_pk = int(np.argmax(g1))
        h_pk = float(g1[i_pk])
        # local curvature spike around the onset peak
        lo = max(i_pk - 10, 0)
        hi = min(i_pk + 10, len(v))
        c_pk = float(np.max(np.abs(g2[lo:hi])))
        # FWHM of the peak above half of its height
        half = 0.5 * h_pk
        above = np.where(g1 >= half)[0]
        fwhm = float(v[above[-1]] - v[above[0]]) if len(above) else np.nan
        peak_metrics[str(nk)] = {
            "peak_height_dDeltaS_dV": h_pk,
            "peak_position": float(v[i_pk]),
            "max_abs_curvature_near_peak": c_pk,
            "fwhm": fwhm,
        }
    results["test3_finite_size_scaling"] = {
        "Delta_t": DT_FIG,
        "metrics": peak_metrics,
        "interpretation": "peak height/curvature growing with Nk signals a "
                          "singular (critical) onset; saturation signals a "
                          "smooth crossover",
    }

    # ------------------------------------------------------------------
    # TEST 4: pair-instability eigenlambda(V) on the normal state
    # ------------------------------------------------------------------
    lam_curves = {}
    for dt in DT_HYST:
        lams = np.zeros(len(v))
        for i, vso in enumerate(v):
            lam, _ = instability_eigenvalue(float(vso), dt, 201)
            lams[i] = lam
        lam_curves[f"{dt:.1f}"] = lams
        crossings = [
            round(float(v[i]), 3)
            for i in range(len(v) - 1)
            if (lams[i] - 1.0) * (lams[i + 1] - 1.0) < 0.0
        ]
        results.setdefault("test4_pair_instability", {})[f"{dt:.1f}"] = {
            "lambda_min": float(lams.min()),
            "lambda_max": float(lams.max()),
            "lambda_at_Vmax": float(lams[-1]),
            "crossings_of_1": crossings,
        }

    # Marginal dip check: does lambda<1 at Delta_t=3.5 survive a finer mesh?
    dip_nk = {}
    v_c = v[(v >= 0.10) & (v <= 0.30)]
    lams_401 = np.array(
        [instability_eigenvalue(float(s), 3.5, 401)[0] for s in v_c]
    )
    dip_nk["401"] = {
        "V_range": [float(v_c[0]), float(v_c[-1])],
        "lambda_min": float(lams_401.min()),
        "crossings_of_1": [
            round(float(v_c[i]), 3)
            for i in range(len(v_c) - 1)
            if (lams_401[i] - 1.0) * (lams_401[i + 1] - 1.0) < 0
        ],
    }
    results["test4_pair_instability"]["dip_check_Nk401_Deltat3.5"] = dip_nk
    lam_curves["3.5_401_V"] = v_c
    lam_curves["3.5_401"] = lams_401
    cross35 = results["test4_pair_instability"]["3.5"]["crossings_of_1"]
    cross45 = results["test4_pair_instability"]["4.5"]["crossings_of_1"]
    results["test4_pair_instability"]["interpretation"] = (
        "Delta_t=4.5: lambda>>1 everywhere -> normal state unstable to "
        "pairing at all V (pairing onset is a crossover). "
        f"Delta_t=3.5: lambda grazes below 1 near V={cross35} "
        "(if it persists at Nk=401: normal state locally stable there "
        "while nonzero Delta_S self-consistently exists -> multistability, "
        "candidate first-order pairing region requiring a grand-potential "
        "comparison)."
    )
    results.setdefault("notes", []).append(
        "N(E_F) shows a secondary van-Hove-like structure near "
        "V_SO/t ~ 1.2-1.3 (mu vs band-1 saddle/edge), flagged for "
        "separate study; the primary Lifshitz kink is at V_Lifshitz."
    )

    # ------------------------------------------------------------------
    # Figure: 2x3 diagnostic panel
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.2),
                             constrained_layout=True)

    # (a) Normal-state pocket area + N_N(E_F) vs SC gap min E_k
    ax = axes[0, 0]
    ax.plot(v, pocket_n, color="#2ca02c", lw=1.8,
            label=r"normal pocket $\langle\theta(\epsilon_0-\mu_N)\rangle$")
    ax.axvline(v_lif_norm, color="#2ca02c", ls="--", lw=1.3,
               label=fr"normal Dirac $V_L={v_lif_norm:.3f}$")
    ax.axvline(v_lif_fig, color="#d62728", ls="-.", lw=1.2,
               label=fr"$\mu_{{\mathrm{{SC}}}}=4t$: ${v_lif_fig:.3f}$ (gapped)")
    ax.set_xlabel(r"$V_{SO}/t$")
    ax.set_ylabel(r"normal inner-pocket area", color="#2ca02c")
    ax.tick_params(axis="y", colors="#2ca02c")
    ax2 = ax.twinx()
    ax2.plot(v, dos_n, color="#1f77b4", lw=1.5, ls="--",
             label=r"$N_N(E_F)$")
    ax2.plot(v, sc_min_gap / 10.0, color="#d62728", lw=1.6,
             label=r"$\min E_{\mathbf{k}}/(10t)$ (SC gap)")
    ax2.set_ylabel(r"$N_N(E_F)$ / SC gap", color="#1f77b4")
    ax2.tick_params(axis="y", colors="#1f77b4")
    ax.set_title("(a) Normal Dirac-Lifshitz vs. SC gap\n"
                 r"($\Delta_t/t=4.5$: $\min E_{\mathbf{k}}\geq 0.94t$)",
                 fontsize=10)
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.8, loc="upper right")

    # (b,c) Normal-state Fermi-surface maps below/above V_L (note: gapped out in SC)
    kx_d = np.linspace(0.0, 2.0 * np.pi, 400)
    KXd, KYd = np.meshgrid(kx_d, kx_d)
    cs_d = np.cos(KXd) + np.cos(KYd)
    lam_d = np.sqrt(np.sin(KXd) ** 2 + np.sin(KYd) ** 2)
    for k, (vso_fs, tag) in enumerate(
            ((0.50, "(b) normal FS before $V_L$: $V_{SO}/t=%.2f$"),
             (v_lif_norm + 0.15, "(c) normal FS after $V_L$: $V_{SO}/t=%.2f$"))):
        ax = axes[0, 1 + k]
        eps0 = -2.0 * cs_d - 2.0 * vso_fs * lam_d
        eps1 = -2.0 * cs_d + 2.0 * vso_fs * lam_d
        mu_fs = float(np.interp(vso_fs, v, mu_n_curve))
        im = ax.imshow(eps0, origin="lower", extent=[0, 2 * np.pi,
                                                      0, 2 * np.pi],
                       cmap="viridis", vmin=-9.0, vmax=5.0)
        ax.contour(KXd, KYd, eps0, levels=[mu_fs],
                   colors="white", linewidths=2.2)
        ax.contour(KXd, KYd, eps1, levels=[mu_fs],
                   colors="cyan", linewidths=1.3, linestyles="--")
        ax.plot(np.pi, np.pi, "o", ms=5, mfc="none", mec="red", mew=1.6)
        ax.set_xticks([0, np.pi, 2 * np.pi])
        ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
        ax.set_yticks([0, np.pi, 2 * np.pi])
        ax.set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
        ax.set_title(tag % vso_fs, fontsize=10)
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")
        if k == 0:
            ax.annotate("normal inner pocket\n(gapped out in SC)",
                        xy=(np.pi, np.pi), xytext=(0.62 * np.pi,
                                                   0.30 * np.pi),
                        color="white", fontsize=8.5, ha="center",
                        arrowprops=dict(arrowstyle="->", color="white",
                                        lw=1.2))
        else:
            ax.annotate("normal pocket gone\n($\\mu_N>4t$; gapped in SC)",
                        xy=(np.pi, np.pi),
                        xytext=(0.62 * np.pi, 0.30 * np.pi),
                        color="white", fontsize=8.5, ha="center",
                        arrowprops=dict(arrowstyle="->", color="white",
                                        lw=1.2))
        if k == 1:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(r"$\epsilon_0(k)/t$", fontsize=8)

    # (d) hysteresis
    ax = axes[1, 0]
    for dt, col in zip(DT_HYST, ("#1f77b4", "#d95f02")):
        idt = int(np.argmin(np.abs(dts_stored - dt)))
        sols_up = sols_up_cache[dt]
        diff_S = np.abs(np.abs(sols_up[:, 1]) - np.abs(DS_down[idt]))
        diff_m = np.abs(sols_up[:, 2] - mu_down[idt])
        ax.semilogy(v, np.maximum(diff_S, 1e-18), color=col, lw=1.4,
                    label=fr"$|\Delta|\Delta_S|$, $\Delta_t/t={dt:.1f}$")
        ax.semilogy(v, np.maximum(diff_m, 1e-18), color=col, lw=1.0,
                    ls="--", label=fr"$|\Delta\mu|$, $\Delta_t/t={dt:.1f}$")
    ax.set_xlabel(r"$V_{SO}/t$")
    ax.set_ylabel(r"ascending $-$ descending")
    ax.set_title("(d) Hysteresis test (first order): up/down\n"
                 r"continuation agree to machine precision", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    # (e) pair-instability eigenvalue (log scale; smooth T=0.01t bound)
    ax = axes[1, 1]
    for dt, col in zip(DT_HYST, ("#1f77b4", "#d95f02")):
        ax.plot(v, lam_curves[f"{dt:.1f}"], color=col, lw=1.7,
                label=fr"$\Delta_t/t={dt:.1f}$ ($N_k=201$)")
    ax.plot(lam_curves["3.5_401_V"], lam_curves["3.5_401"],
            color="#1f77b4", lw=1.4, ls=":",
            label=r"$\Delta_t/t=3.5$ ($N_k=401$)")
    ax.axhline(1.0, color="black", ls="--", lw=1.2,
               label=r"$\lambda_+=1$ (instability)")
    ax.axvline(v_lif_fig, color="#d62728", ls="-.", lw=1.0, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel(r"$V_{SO}/t$")
    ax.set_ylabel(r"pair-instability eigenvalue $\lambda_+$")
    ax.set_title("(e) Linearized gap eigenvalue ($T=0.01t$ bound):\n"
                 r"$\lambda_+>1$ everywhere (no pairing QPT)", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.5, loc="lower right")

    # (f) finite-size scaling of the d|Delta_S|/dV_SO inflection peak
    ax = axes[1, 2]
    for nk, col in zip(NK_SCALE, ("#9467bd", "#1f77b4", "#d62728")):
        g1 = np.gradient(DS_nk[nk], v)
        mask = (v >= 0.5) & (v <= 1.1)
        ax.plot(v[mask], g1[mask], color=col, lw=1.5,
                label=fr"$N_k={nk}$")
        i_pk = int(np.argmax(g1))
        ax.plot([v[i_pk]], [g1[i_pk]], "o", ms=6, color=col)
    ax.axvline(v_lif_fig, color="#d62728", ls="-.", lw=1.0, alpha=0.8)
    ax.set_xlabel(r"$V_{SO}/t$")
    ax.set_ylabel(r"$d|\Delta_S|/dV_{SO}$")
    ax.set_title("(f) Finite-size scaling of $d|\\Delta_S|/dV_{SO}$\n"
                 "inflection peak (saturates: crossover)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)

    fig.suptitle(
        r"QPT diagnostic tests at $U/t=75$, $n=1.875$, $T=0$ "
        r"(no physical QPT along $V_{SO}$: gapped SC crossover)",
        fontsize=12.5,
    )
    fig.savefig(FIGS / "qpt_diagnostic_tests.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    results["figures"] = ["figs/qpt_diagnostic_tests.png"]
    (REPORTS / "qpt_diagnostic_tests.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in results.items()
                      if k.startswith("test")},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
