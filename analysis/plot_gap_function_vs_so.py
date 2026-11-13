#!/usr/bin/env python3
"""Gap function Delta_k vs V_SO for several (U, Delta_t) -- three views.

The self-consistent gap is k-dependent but only through two harmonics:

    Delta_{k,s} = Delta_0 - (Delta_S / 4t) * epsilon_{k,s},
    epsilon_{k,s} = -2t (cos kx + cos ky) -/+ 2 V_SO sqrt(sin^2 kx + sin^2 ky).

This script visualizes it three ways:

  A. Harmonic amplitudes  Delta_0(V_SO), Delta_S(V_SO)  (approach A).
  B. Spectral gap  Delta_spec = min_{k,s} E_{k,s}  together with the
     amplitude band  [min_k |Delta_k|, max_k |Delta_k|]  (approach B).
  C. |Delta_{k,s}| along the Gamma-X-M-Gamma path as heatmaps vs V_SO
     for the representative point U/t=75, Delta_t/t=4.5 (approach C).

Solves reuse the dense-sweep branch continuation (tiered fsolve/LM,
residual-gated, descending from V_SO=1.5) at Nk=201 over V_SO/t in
[0, 1.5] (61 points, dV=0.025), for U/t in {40, 75, 115} x
Delta_t/t in {3.5, 4.5, 6.0}.  Points that fall onto the trivial
Delta=0 root are re-solved from nontrivial seeds.  All spectral
quantities are evaluated on an independent Nk=401 grid.

Outputs
-------
figs/gap_function_vs_so.png        (2x3 panels a-f)
data/gap_function_vs_so.npz        full arrays
reports/gap_function_vs_so.json    summary + diagnostics
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from scipy.optimize import fsolve, least_squares

from self_consistency import residual

DATA = ROOT / "data"
FIGS = ROOT / "figs"
REPORTS = ROOT / "reports"

# --------------------------------------------------------------------------- #
# Sweep configuration
# --------------------------------------------------------------------------- #
U_VALUES = np.array([40.0, 75.0, 115.0])
DT_VALUES = np.array([3.5, 4.5, 6.0])
VSO_VALUES = np.round(np.linspace(0.0, 1.5, 61), 4)   # dV = 0.025
N_ELEC = 1.875
NK_SOLVE = 201
NK_EVAL = 401
T_HOP = float(config.t_hop)
BAND_EDGE_4T = 4.0 * T_HOP          # epsilon_(pi,pi)/t = +4t
SMALL_GAP = 0.1 * T_HOP             # gray-shading threshold (as elsewhere)
REP_U, REP_DT = 75.0, 4.5           # representative combo for heatmaps

DT_COLORS = {3.5: "#1f77b4", 4.5: "#d62728", 6.0: "#2ca02c"}
U_STYLES = {40.0: "-", 75.0: "--", 115.0: ":"}


# --------------------------------------------------------------------------- #
# k-space helpers (parameterized -- no config mutation)
# --------------------------------------------------------------------------- #
def build_eps(nk: int, vso: float) -> tuple[np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray]:
    """Return KX, KY, epsilon_{k,s}, sqrt_sin on an nk x nk grid."""
    kx = np.linspace(-np.pi, np.pi, nk)
    KX, KY = np.meshgrid(kx, kx)
    ssin = np.sqrt(np.sin(KX) ** 2 + np.sin(KY) ** 2)
    band = -2.0 * T_HOP * (np.cos(KX) + np.cos(KY))
    eps = np.empty((nk, nk, 2))
    eps[:, :, 0] = band - 2.0 * vso * ssin     # helicity "+1" (index 0)
    eps[:, :, 1] = band + 2.0 * vso * ssin     # helicity "-1" (index 1)
    return KX, KY, eps, ssin


def gap_diagnostics(sol: np.ndarray, vso: float) -> dict:
    """Approach A+B scalars for one solution at coupling vso (Nk=401 eval)."""
    D0, DS, mu = float(sol[0]), float(sol[1]), float(sol[2])
    _, _, eps, _ = build_eps(NK_EVAL, vso)
    Dks = D0 - (DS / (4.0 * T_HOP)) * eps
    Eks = np.sqrt(Dks ** 2 + (eps - mu) ** 2)
    absD = np.abs(Dks)
    return {
        "Delta0": D0,
        "DeltaS": DS,
        "mu": mu,
        "gap_spec": float(Eks.min()),
        "max_absD": float(absD.max()),
        "min_absD": float(absD.min()),
        # Closed forms at the TRIM (Rashba lambda_k vanishes there):
        "D_Gamma": D0 + DS,          # eps(Gamma) = -4t
        "D_M": D0 - DS,              # eps(M)     = +4t
        "D_X": D0,                   # eps(X)     =  0  (both helicities)
        # Helicity splitting of the gap at k=(pi/2,pi/2):
        "split_pi2": float(abs(np.sqrt(2.0) * DS * vso / T_HOP)),
    }


def solve_point(vso: float, dt: float, u: float, n: float,
                warm: np.ndarray) -> tuple[np.ndarray, float, int]:
    """One residual-gated solve: fsolve -> seed tier -> LM -> random LM.

    Acceptance is decided only by the re-evaluated residual (never by
    fsolve's ier).  Returns (sol, residual, n_extra_restarts).
    """
    KX, KY, eps, ssin = build_eps(NK_SOLVE, vso)
    sk = 0.5 * (np.cos(KX) + np.cos(KY))
    args = (eps, sk, ssin, NK_SOLVE, T_HOP, float(u), float(dt),
            float(vso), float(n), np.inf)

    def _flip(s: np.ndarray) -> np.ndarray:
        return -s if s[1] > 0.0 else s

    def _res(s: np.ndarray) -> float:
        return float(np.max(np.abs(residual(s, *args))))

    n_extra = 0

    def _fsolve(guess: np.ndarray) -> tuple[np.ndarray, float]:
        nonlocal n_extra
        n_extra += 1
        s = _flip(fsolve(residual, guess, args=args, full_output=False,
                         xtol=1e-12, maxfev=3000))
        return s, _res(s)

    # Tier 0: warm start from the continuation chain.
    sol, res = _fsolve(warm)

    if res > 1e-9:
        # Robust nontrivial probe first: if a healthy SC root exists,
        # this classic attractor finds it (and we return immediately).
        robust = np.array([-0.15, -0.45, sol[2]])
        cand, cand_res = _fsolve(robust)
        if cand_res < res:
            sol, res = cand, cand_res
        if res <= 1e-9:
            return sol, res, n_extra, False

        # Small-gap seeds: catch tiny-but-exact roots (|Delta| ~ 1e-4).
        for a in (0.02, 0.001):
            cand, cand_res = _fsolve(np.array([-a, -2.0 * a, sol[2]]))
            if cand_res < res:
                sol, res = cand, cand_res
            if res <= 1e-9:
                return sol, res, n_extra, False

        # Early mesh-floor acceptance: pairing collapsed below mesh
        # resolution (T=0 number equation is a step function on the grid;
        # residual floor ~1/(2 Nk^2)).  The robust probe above already
        # excluded a hidden nontrivial root.
        if res <= 2.5e-4 and abs(sol[1]) <= 5e-3:
            return sol, res, n_extra, True

        # LM polish from the best candidates.
        for guess in (sol, robust, np.array([-0.02, -0.04, sol[2]])):
            n_extra += 1
            lm = least_squares(residual, np.asarray(guess, float), args=args,
                               method="lm", xtol=1e-14, ftol=1e-14,
                               gtol=1e-14, max_nfev=4000)
            cand = _flip(lm.x.copy())
            cand_res = _res(cand)
            if cand_res < res:
                sol, res = cand, cand_res
            if res <= 1e-11:
                return sol, res, n_extra, False

        if res <= 2.5e-4 and abs(sol[1]) <= 5e-3:
            return sol, res, n_extra, True

        # Extended tier: random starts (stalled nontrivial roots).
        rng = np.random.default_rng(12345 + int(round(1000 * vso)))
        for _ in range(8):
            n_extra += 1
            guess = np.array([
                rng.uniform(-0.5, 0.5),
                -rng.uniform(0.02, 2.5),
                rng.uniform(float(eps.min()), float(eps.max())),
            ])
            lm = least_squares(residual, guess, args=args, method="lm",
                               xtol=1e-14, ftol=1e-14, gtol=1e-14,
                               max_nfev=12000)
            cand = _flip(lm.x.copy())
            cand_res = _res(cand)
            if cand_res < res:
                sol, res = cand, cand_res
            if res <= 1e-11:
                break
        mesh_floor = bool(res > 1e-9 and res <= 2.5e-4
                          and abs(sol[1]) <= 5e-3)
        return sol, res, n_extra, mesh_floor

    return sol, res, n_extra, False


def solve_branch(vso_values: np.ndarray, dt: float, u: float,
                 n: float) -> tuple[np.ndarray, np.ndarray, int, int,
                                    np.ndarray]:
    """Descending continuation with warm starts, trivial-root retry."""
    sol_prev = np.array([-0.2, -1.8, 5.5], dtype=float)
    sols = np.zeros((len(vso_values), 3))
    resids = np.zeros(len(vso_values))
    floors = np.zeros(len(vso_values), dtype=bool)
    n_extra_total = 0
    n_retry = 0
    for iv, vso in enumerate(reversed(vso_values)):
        sol, res, n_extra, floor = solve_point(float(vso), dt, u, n, sol_prev)
        n_extra_total += n_extra
        if n_extra or floor:
            print(f"    V={vso:.4f}: res={res:.2e} DS={sol[1]:+.5f} "
                  f"extra={n_extra} floor={floor}", flush=True)
        if res > 1e-9 and not floor:
            raise RuntimeError(
                f"Failed to converge U={u}, Delta_t={dt}, V_SO={vso}: "
                f"residual={res:.3e}"
            )
        if abs(sol[1]) <= 1e-6 and not floor:
            # Landed on the trivial Delta=0 root: retry from nontrivial seeds.
            n_retry += 1
            alt, res_alt, n_extra2, floor2 = solve_point(
                float(vso), dt, u, n, np.array([-0.05, -0.15, sol[2]])
            )
            n_extra_total += n_extra2
            if abs(alt[1]) > 1e-6 and res_alt <= max(res, 1e-9):
                sol, res, floor = alt, res_alt, floor2
        sol_prev = sol.copy()
        idx = len(vso_values) - 1 - iv
        sols[idx] = sol
        resids[idx] = res
        floors[idx] = floor
    return sols, resids, n_extra_total, n_retry, floors


# --------------------------------------------------------------------------- #
# Approach C: gap along the Gamma-X-M-Gamma path (closed form, exact)
# --------------------------------------------------------------------------- #
def path_gap_curve(D0: float, DS: float, vso: float,
                   n_seg: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """|Delta_{k,s}| along Gamma->X->M->Gamma for both helicities."""
    t1 = np.linspace(0.0, np.pi, n_seg, endpoint=False)   # Gamma -> X
    t2 = np.linspace(0.0, np.pi, n_seg, endpoint=False)   # X -> M
    t3 = np.linspace(np.pi, 0.0, n_seg, endpoint=False)   # M -> Gamma
    kx = np.concatenate([t1, np.full(n_seg, np.pi), t3])
    ky = np.concatenate([np.zeros(n_seg), t2, t3])
    coord = np.concatenate([
        np.linspace(0.0, 1.0, n_seg, endpoint=False),
        np.linspace(1.0, 2.0, n_seg, endpoint=False),
        np.linspace(2.0, 3.0, n_seg, endpoint=False),
    ])
    ssin = np.sqrt(np.sin(kx) ** 2 + np.sin(ky) ** 2)
    band = -2.0 * T_HOP * (np.cos(kx) + np.cos(ky))
    out = np.empty((len(kx), 2))
    for s in range(2):
        sign = -1.0 if s == 0 else +1.0
        eps = band + sign * 2.0 * vso * ssin
        out[:, s] = np.abs(D0 - (DS / (4.0 * T_HOP)) * eps)
    return coord, out[:, 0], out[:, 1]


def main() -> int:
    n_u, n_dt, nv = len(U_VALUES), len(DT_VALUES), len(VSO_VALUES)
    shape = (n_u, n_dt, nv)

    Delta0 = np.zeros(shape)
    DeltaS = np.zeros(shape)
    mu_arr = np.zeros(shape)
    gap_spec = np.zeros(shape)
    max_absD = np.zeros(shape)
    min_absD = np.zeros(shape)
    D_Gamma = np.zeros(shape)
    D_M = np.zeros(shape)
    split_pi2 = np.zeros(shape)
    resid_arr = np.zeros(shape)
    normal_flags = np.zeros(shape, dtype=bool)
    floor_flags = np.zeros(shape, dtype=bool)
    n_fb_total = 0
    n_retry_total = 0
    n_floor_total = 0

    # ------------------------------------------------------------------ solve
    for iu, u in enumerate(U_VALUES):
        for idt, dt in enumerate(DT_VALUES):
            print(f"\n=== U/t={u:.0f}, Delta_t/t={dt:.1f} ===")
            sols, resids, n_fb, n_retry, floors = solve_branch(
                VSO_VALUES, dt, u, N_ELEC
            )
            n_fb_total += n_fb
            n_retry_total += n_retry
            n_floor_total += int(floors.sum())
            floor_flags[iu, idt, :] = floors
            for iv in range(nv):
                dg = gap_diagnostics(sols[iv], VSO_VALUES[iv])
                Delta0[iu, idt, iv] = dg["Delta0"]
                DeltaS[iu, idt, iv] = dg["DeltaS"]
                mu_arr[iu, idt, iv] = dg["mu"]
                gap_spec[iu, idt, iv] = dg["gap_spec"]
                max_absD[iu, idt, iv] = dg["max_absD"]
                min_absD[iu, idt, iv] = dg["min_absD"]
                D_Gamma[iu, idt, iv] = dg["D_Gamma"]
                D_M[iu, idt, iv] = dg["D_M"]
                split_pi2[iu, idt, iv] = dg["split_pi2"]
                resid_arr[iu, idt, iv] = resids[iv]
                normal_flags[iu, idt, iv] = abs(sols[iv, 1]) <= 1e-6
            print(f"  max|resid|={resids.max():.2e}  fallbacks={n_fb}  "
                  f"trivial-retries={n_retry}  mesh-floor={int(floors.sum())}  "
                  f"|Delta_S| in [{np.abs(DeltaS[iu, idt]).min():.3f}, "
                  f"{np.abs(DeltaS[iu, idt]).max():.3f}]  "
                  f"gap_spec min={gap_spec[iu, idt].min():.4f}")

    # ------------------------------------------------------------------ plot
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.5))

    handles, labels = [], []
    for iu, u in enumerate(U_VALUES):
        for idt, dt in enumerate(DT_VALUES):
            (h,) = axes[0, 0].plot(
                VSO_VALUES, Delta0[iu, idt],
                color=DT_COLORS[float(dt)], ls=U_STYLES[float(u)],
                lw=1.7, label=fr"$U/t={u:.0f}$, $\Delta_t/t={dt:.1f}$",
            )
            handles.append(h)
            labels.append(h.get_label())

    # (a) approach A: Delta_0(V)
    ax = axes[0, 0]
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_xlabel(r"$V_{SO}/t$")
    ax.set_ylabel(r"$\Delta_0/t$")
    ax.set_title(r"(a) Approach A: harmonic amplitude $\Delta_0(V_{SO})$")
    ax.grid(alpha=0.3)

    # (b) approach A: Delta_S(V)
    ax = axes[0, 1]
    for iu, u in enumerate(U_VALUES):
        for idt, dt in enumerate(DT_VALUES):
            ax.plot(VSO_VALUES, DeltaS[iu, idt],
                    color=DT_COLORS[float(dt)], ls=U_STYLES[float(u)], lw=1.7)
            un = np.abs(DeltaS[iu, idt]) <= 1e-3
            if np.any(un):
                ax.plot(VSO_VALUES[un], DeltaS[iu, idt][un], "o",
                        mfc="none", mec=DT_COLORS[float(dt)], ms=6, mew=1.4,
                        zorder=5)
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_xlabel(r"$V_{SO}/t$")
    ax.set_ylabel(r"$\Delta_S/t$")
    ax.set_title(r"(b) Approach A: harmonic amplitude $\Delta_S(V_{SO})$")
    ax.grid(alpha=0.3)

    # (c) approach B: spectral gap (log scale, gray band below 0.1t)
    ax = axes[0, 2]
    ax.axhspan(0.0, SMALL_GAP, color="0.85", zorder=0)
    for iu, u in enumerate(U_VALUES):
        for idt, dt in enumerate(DT_VALUES):
            ax.plot(VSO_VALUES, np.maximum(gap_spec[iu, idt], 1e-4),
                    color=DT_COLORS[float(dt)], ls=U_STYLES[float(u)], lw=1.7)
    ax.set_yscale("log")
    ax.set_xlabel(r"$V_{SO}/t$")
    ax.set_ylabel(r"$\Delta_{\rm spec}=\min_{\mathbf{k},s}E_{\mathbf{k},s}\;/t$")
    ax.set_title(r"(c) Approach B: spectral gap (gray: $<0.1t$)")
    ax.grid(alpha=0.3, which="both")

    # (d)/(e) approach C: heatmaps for the representative combo
    iu_rep = int(np.where(U_VALUES == REP_U)[0][0])
    idt_rep = int(np.where(DT_VALUES == REP_DT)[0][0])
    coord = None
    curves_p, curves_m = [], []
    for iv, vso in enumerate(VSO_VALUES):
        coord, d_plus, d_minus = path_gap_curve(
            Delta0[iu_rep, idt_rep, iv], DeltaS[iu_rep, idt_rep, iv], float(vso)
        )
        curves_p.append(d_plus)
        curves_m.append(d_minus)
    Zp = np.array(curves_p).T     # (path, V)
    Zm = np.array(curves_m).T

    for ax, Z, tag in ((axes[1, 0], Zp, r"$s=+1$"),
                       (axes[1, 1], Zm, r"$s=-1$")):
        im = ax.imshow(
            Z, origin="lower", aspect="auto", cmap="viridis",
            extent=[VSO_VALUES[0], VSO_VALUES[-1], 0.0, 3.0],
        )
        ax.set_xticks([0.0, 0.5, 1.0, 1.5])
        ax.set_yticks([0.0, 1.0, 2.0, 3.0])
        ax.set_yticklabels([r"$\Gamma$", "X", "M", r"$\Gamma$"])
        ax.set_xlabel(r"$V_{SO}/t$")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r"$|\Delta_{\mathbf{k},s}|/t$")
    axes[1, 0].set_title(
        r"(d) Approach C: $|\Delta_{\mathbf{k}}|$ along $\Gamma$-X-M-$\Gamma$ "
        r"($s=+1$)")
    axes[1, 1].set_title(
        r"(e) Approach C: $|\Delta_{\mathbf{k}}|$ along $\Gamma$-X-M-$\Gamma$ "
        r"($s=-1$)")

    # (f) representative combo: amplitudes vs spectral gap, small-gap shading
    ax = axes[1, 2]
    below = gap_spec[iu_rep, idt_rep] < SMALL_GAP
    # contiguous below-threshold runs -> gray vertical spans
    iv = 0
    while iv < nv:
        if below[iv]:
            j = iv
            while j + 1 < nv and below[j + 1]:
                j += 1
            ax.axvspan(VSO_VALUES[iv], VSO_VALUES[min(j + 1, nv - 1)],
                       color="0.88", zorder=0)
            iv = j + 1
        else:
            iv += 1
    ax.plot(VSO_VALUES, max_absD[iu_rep, idt_rep], color="#1f77b4", lw=1.8,
            label=r"$\max_{\mathbf{k},s}|\Delta_{\mathbf{k}}|$")
    ax.plot(VSO_VALUES, min_absD[iu_rep, idt_rep], color="#1f77b4", lw=1.2,
            ls="--", label=r"$\min_{\mathbf{k},s}|\Delta_{\mathbf{k}}|$")
    ax.plot(VSO_VALUES, D_Gamma[iu_rep, idt_rep], color="#d62728", lw=1.2,
            ls=":", label=r"$\Delta_{\Gamma}=\Delta_0+\Delta_S$")
    ax.plot(VSO_VALUES, D_M[iu_rep, idt_rep], color="#9467bd", lw=1.2,
            ls=":", label=r"$\Delta_{M}=\Delta_0-\Delta_S$")
    ax.plot(VSO_VALUES, gap_spec[iu_rep, idt_rep], color="black", lw=2.0,
            label=r"$\Delta_{\rm spec}=\min E_{\mathbf{k}}$")
    ax.plot(VSO_VALUES, split_pi2[iu_rep, idt_rep], color="#2ca02c", lw=1.2,
            ls="-.", label=r"$|\Delta_{+}-\Delta_{-}|$ at $(\pi/2,\pi/2)$")
    ax.set_xlabel(r"$V_{SO}/t$")
    ax.set_ylabel(r"$\Delta/t$")
    ax.set_title(
        fr"(f) Amplitudes vs spectral gap ($U/t={REP_U:.0f}$, "
        fr"$\Delta_t/t={REP_DT:.1f}$; gray: $\Delta_{{\rm spec}}<0.1t$)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        r"Gap function $\Delta_{\mathbf{k}}=\Delta_0-(\Delta_S/4t)"
        r"\epsilon_{\mathbf{k}}$ vs $V_{SO}$ for $U/t\in\{40,75,115\}$, "
        r"$\Delta_t/t\in\{3.5,4.5,6.0\}$ (color=$\Delta_t$, style=$U$)",
        fontsize=12,
    )
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.015), frameon=False)
    fig.savefig(FIGS / "gap_function_vs_so.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- archive
    np.savez_compressed(
        DATA / "gap_function_vs_so.npz",
        U_values=U_VALUES, Delta_t_over_t=DT_VALUES, V_SO_over_t=VSO_VALUES,
        Delta0=Delta0, DeltaS=DeltaS, mu=mu_arr,
        gap_spec=gap_spec, max_absD=max_absD, min_absD=min_absD,
        D_Gamma=D_Gamma, D_M=D_M, split_pi2=split_pi2,
        max_abs_residual=resid_arr, normal_flags=normal_flags,
        mesh_floor_flags=floor_flags,
        path_coord=coord,
    )

    def _sign_changes(arr: np.ndarray) -> list[float]:
        s = np.sign(arr)
        idx = np.where(np.diff(s) != 0)[0]
        out = []
        for i in idx:
            v0, v1 = VSO_VALUES[i], VSO_VALUES[i + 1]
            y0, y1 = arr[i], arr[i + 1]
            out.append(round(float(v0 + (v1 - v0) * (-y0) / (y1 - y0)), 4))
        return out

    def _unpaired_ranges(mask: np.ndarray) -> list[list[float]]:
        out, i = [], 0
        while i < len(mask):
            if mask[i]:
                j = i
                while j + 1 < len(mask) and mask[j + 1]:
                    j += 1
                out.append([float(VSO_VALUES[i]), float(VSO_VALUES[j])])
                i = j + 1
            else:
                i += 1
        return out

    combos = []
    for iu, u in enumerate(U_VALUES):
        for idt, dt in enumerate(DT_VALUES):
            g = gap_spec[iu, idt]
            un_mask = np.abs(DeltaS[iu, idt]) <= 1e-3
            combos.append({
                "U_over_t": float(u), "Delta_t_over_t": float(dt),
                "Delta0_range": [float(Delta0[iu, idt].min()),
                                 float(Delta0[iu, idt].max())],
                "DeltaS_range": [float(DeltaS[iu, idt].min()),
                                 float(DeltaS[iu, idt].max())],
                "Delta0_sign_change_V": _sign_changes(Delta0[iu, idt]),
                "DeltaS_sign_change_V": _sign_changes(DeltaS[iu, idt]),
                "gap_spec_min": float(g.min()),
                "V_at_gap_spec_min": float(VSO_VALUES[int(np.argmin(g))]),
                "V_range_gap_below_0p1t": (
                    None if not np.any(g < SMALL_GAP)
                    else [float(VSO_VALUES[g < SMALL_GAP].min()),
                          float(VSO_VALUES[g < SMALL_GAP].max())]
                ),
                "max_absD_at_Vmax": float(max_absD[iu, idt, -1]),
                "helicity_split_at_Vmax": float(split_pi2[iu, idt, -1]),
                "unpaired_points_absDS_le_1e-3": int(un_mask.sum()),
                "unpaired_V_ranges": _unpaired_ranges(un_mask),
                "mesh_floor_points": int(floor_flags[iu, idt].sum()),
                "max_abs_residual": float(resid_arr[iu, idt].max()),
            })

    summary = {
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "parameters": {
            "U_over_t": U_VALUES.tolist(),
            "Delta_t_over_t": DT_VALUES.tolist(),
            "V_SO_over_t_min": float(VSO_VALUES[0]),
            "V_SO_over_t_max": float(VSO_VALUES[-1]),
            "n_V_SO_points": nv,
            "dV_SO_over_t": float(VSO_VALUES[1] - VSO_VALUES[0]),
            "n_elec": N_ELEC, "temperature": 0.0,
            "Nk_solve": NK_SOLVE, "Nk_eval": NK_EVAL,
            "gap_form": "Delta_{k,s} = Delta_0 - (Delta_S/4t) eps_{k,s}",
            "small_gap_threshold": SMALL_GAP,
            "representative_combo": {"U_over_t": REP_U,
                                     "Delta_t_over_t": REP_DT},
        },
        "diagnostics": {
            "max_solver_residual": float(resid_arr.max()),
            "n_fallback_solves": int(n_fb_total),
            "n_trivial_retries": int(n_retry_total),
            "n_mesh_floor_points": int(n_floor_total),
            "n_unpaired_points_absDS_le_1e-3": int(
                np.sum(np.abs(DeltaS) <= 1e-3)),
            "n_normal_state_points": int(normal_flags.sum()),
        },
        "combos": combos,
        "interpretation_notes": [
            "At fixed Delta_t pairing weakens with increasing U in this "
            "parameter set (|Delta_S|(V=1.5): U40 > U75 > U115).",
            "Delta_0 sign changes (s+-type relative-sign reorganization of "
            "the harmonics) occur only for U/t=40 (Delta_t=3.5, 4.5) and "
            "U/t=75, Delta_t=6.0 -- see Delta0_sign_change_V per combo.",
            "For U/t=115, Delta_t=3.5 (and 4.5) pairing collapses to mesh "
            "resolution at low V_SO: exact roots become |Delta|~1e-4 "
            "(unpaired points) and below V~0.9 (3.5) the T=0 number "
            "equation hits its finite-mesh residual floor "
            "(<=2.5e-4, flagged mesh_floor_points); Delta_spec there "
            "measures min_k|eps-mu| of the normal state (mesh-limited).",
            "min_k|Delta_k| = 0 where the extended-s harmonic crosses zero "
            "(nodal/s+- lines); plot values < 1e-4 are clipped in the "
            "log-scale panel (c).",
        ],
        "generated_figures": ["figs/gap_function_vs_so.png"],
    }
    (REPORTS / "gap_function_vs_so.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["diagnostics"], indent=2))
    for c in combos:
        print(f"U={c['U_over_t']:.0f} Dt={c['Delta_t_over_t']:.1f}: "
              f"D0 in [{c['Delta0_range'][0]:+.3f},{c['Delta0_range'][1]:+.3f}] "
              f"DS in [{c['DeltaS_range'][0]:+.3f},{c['DeltaS_range'][1]:+.3f}] "
              f"gap_min={c['gap_spec_min']:.4f}@V={c['V_at_gap_spec_min']:.3f} "
              f"D0 zeros at {c['Delta0_sign_change_V']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
