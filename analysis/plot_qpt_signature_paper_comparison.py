#!/usr/bin/env python3
"""Compare our dense-sweep dC/dV_SO signatures with the QPT criteria of
Ebrahimian, Khosrojerdi & Afzali, arXiv:2011.14038.

That paper (BCS superconductor with Shiba-Rusinov impurities) identifies QPT
points from the concurrence C(v) of Cooper-pair spins as:

  1. turning point: minimum of dC/dv with d2C/dv2 = 0 (and d3C/dv3 != 0),
     where the curve changes from concave down to concave up;
  2. local maximum point: dC/dv = 0 with d2C/dv2 at a minimum;
  3. the points displace when the couplings (exchange J, pair interaction
     Delta) change, but NOT with impurity concentration or pair distance;
  4. cross-checked against the derivatives of the system energy.

This script evaluates the same criteria on our dense V_SO/t in [0,1.5]
(151-point, Nk=201) sweep of the exact conditional Wick concurrence and of
the legacy particle-branch X-state concurrence:

  - turning points V* = argmin dC/dV_SO with d2C/dV2 sign change,
  - interior zeros of dC/dV (local maxima of C),
  - non-analytic sudden-death/birth jumps of the legacy C,
  - displacement of V* with Delta_t vs invariance in r,
  - alignment of V* with the independent order-parameter transitions
    (Lifshitz mu = 4t crossing; pairing onset argmax d|Delta_S|/dV).
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

DATA = ROOT / "data"
FIGS = ROOT / "figs"
REPORTS = ROOT / "reports"

DT_COLORS = ["#1f77b4", "#2ca02c", "#d95f02", "#9467bd", "#d62728"]


def main() -> int:
    d = np.load(DATA / "concurrence_dense_vso_deriv.npz")
    v = d["V_SO_over_t"]
    dts = d["Delta_t_over_t"]
    rwick = d["r_wick"]
    rleg = d["legacy_r"]
    dC = d["dC_wick_dV"]          # (dt, v, basis, r)
    dCl = d["dC_legacy_dV"]       # (dt, v, theta, r)
    Cleg = d["C_legacy"]
    v_small = d["V_smallgap_region"]
    rep = json.load(open(REPORTS / "concurrence_dense_vso_deriv.json"))
    v_life = np.array([t["V_lifshitz_mu_eq_4t"]
                       for t in rep["transition_candidates"]], dtype=float)
    v_onset = np.array([t["V_pair_onset"]
                        for t in rep["transition_candidates"]], dtype=float)

    ndt = len(dts)
    j025 = int(np.argmin(np.abs(rwick - 0.25)))
    j50 = int(np.argmin(np.abs(rleg - 0.50)))

    # --- Criterion 1: turning points V* = argmin dC/dV, d2C/dV2 sign change
    vstar = np.zeros((ndt, 2, len(rwick)))   # (dt, basis, r)
    d2_ok = np.zeros((ndt, 2, len(rwick)), dtype=bool)
    for idt in range(ndt):
        for b in range(2):
            for j in range(len(rwick)):
                dc = dC[idt, :, b, j]
                i = int(np.argmin(dc))
                vstar[idt, b, j] = v[i]
                d2 = np.gradient(dc, v)
                lo, hi = d2[max(i - 3, 0)], d2[min(i + 3, len(v) - 1)]
                d2_ok[idt, b, j] = bool(lo < 0.0 < hi)

    # --- Criterion 2: interior zeros of dC/dV (local maxima of C)
    local_max = []
    for idt, dt in enumerate(dts):
        for b, basis in enumerate(("spin", "band")):
            for j, rv in enumerate(rwick):
                dc = dC[idt, :, b, j].copy()
                dc[v < v_small[idt]] = np.nan
                for i in range(len(v) - 1):
                    if np.isnan(dc[i]) or np.isnan(dc[i + 1]):
                        continue
                    if dc[i] > 0.0 > dc[i + 1]:
                        local_max.append({"Delta_t": float(dt),
                                          "basis": basis, "r": float(rv),
                                          "V": float(v[i])})

    # --- Criterion 1/2 on legacy C: non-analytic ESD/EB jumps
    esd = rep["legacy_ESD_EB_events"]

    # -------------------------------------------------------------------------
    # Figure: mirrors their Fig. 4 (derivative criteria) + displacement plot
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(19.0, 11.5))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.0, 1.0, 0.85],
                          hspace=0.42, wspace=0.25)

    # Row A: exact dC/dV with turning points (their Fig. 4a criterion)
    for idt, dt in enumerate(dts):
        ax = fig.add_subplot(gs[0, idt])
        if v_small[idt] > 0:
            ax.axvspan(0.0, v_small[idt], color="0.88", zorder=0)
        ax.plot(v, dC[idt, :, 0, j025], color="#1f77b4", lw=1.7,
                label="spin")
        ax.plot(v, dC[idt, :, 1, j025], color="#d95f02", lw=1.7, ls="--",
                label="band")
        for b, col in ((0, "#1f77b4"), (1, "#d95f02")):
            i = int(np.argmin(dC[idt, :, b, j025]))
            ax.axvline(v[i], color=col, ls=":", lw=1.2, alpha=0.9)
            ax.plot([v[i]], [dC[idt, i, b, j025]], "v", color=col, ms=7)
        ax.axhline(0.0, color="black", lw=0.6, ls=":")
        ax.grid(alpha=0.3)
        ax.set_title(fr"$\Delta_t/t={dt:.1f}$", fontsize=11)
        ax.set_xlabel(r"$V_{SO}/t$")
        if idt == 0:
            ax.set_ylabel(r"$dC/dV_{SO}$  (exact, $r=0.25$)")
            ax.legend(fontsize=7.5, loc="upper left")
        vstar_txt = (fr"$V^{{*}}_{{spin}}={vstar[idt,0,j025]:.2f}$"
                     if idt == 0 else
                     f"{vstar[idt, 0, j025]:.2f}/{vstar[idt, 1, j025]:.2f}")
        ax.text(0.97, 0.93, vstar_txt, transform=ax.transAxes, ha="right",
                fontsize=8.5, color="#333333")

    # Row B: legacy dC/dV with non-analytic death/birth jumps
    for idt, dt in enumerate(dts):
        ax = fig.add_subplot(gs[1, idt])
        if v_small[idt] > 0:
            ax.axvspan(0.0, v_small[idt], color="0.88", zorder=0)
        ax.plot(v, dCl[idt, :, 0, j50], color="#9467bd", lw=1.7,
                label=r"legacy $r=0.50$")
        ax.axhline(0.0, color="black", lw=0.6, ls=":")
        marked = False
        for ev in esd:
            if ev["Delta_t_over_t"] == float(dt) and \
                    ev["theta_deg"] == 0.0 and ev["r"] == float(rleg[j50]):
                col = "#d62728" if ev["event"] == "death" else "#2ca02c"
                ax.axvline(ev["V_SO"], color=col, ls=":", lw=1.3, alpha=0.9)
                marked = True
        ax.grid(alpha=0.3)
        ax.set_xlabel(r"$V_{SO}/t$")
        if idt == 0:
            ax.set_ylabel(r"$dC/dV_{SO}$  (legacy, $r=0.50$)")
            ax.legend(fontsize=7.5, loc="upper left")
        if not marked:
            ax.text(0.97, 0.9, "no death/birth\nfor this $r$",
                    transform=ax.transAxes, ha="right", fontsize=8,
                    color="#555555")

    # Row C: displacement of V* with Delta_t, invariance in r (their claim 3)
    ax = fig.add_subplot(gs[2, :])
    for idt, dt in enumerate(dts):
        ax.plot(np.full(len(rwick), dt) - 0.04, vstar[idt, 0, :], "o",
                ms=6, color=DT_COLORS[idt], alpha=0.85)
        ax.plot(np.full(len(rwick), dt) + 0.04, vstar[idt, 1, :], "s",
                ms=6, mfc="none", color=DT_COLORS[idt], alpha=0.95)
        ax.plot(dt, vstar[idt, 0, j025], "o", ms=9, color=DT_COLORS[idt],
                mec="black", mew=0.8)
    # Independent order parameters
    ok = ~np.isnan(v_life)
    ax.plot(dts[ok], v_life[ok], "--", color="#2ca02c", lw=1.6,
            label=r"Lifshitz crossing $\mu=4t$")
    ax.plot(dts, v_onset, "-.", color="#d62728", lw=1.6,
            label=r"pairing onset $\max_V\ d|\Delta_S|/dV$")
    ax.plot([], [], "o", color="0.4", label=r"turning point $V^*$ (spin)")
    ax.plot([], [], "s", mfc="none", color="0.4",
            label=r"turning point $V^*$ (band)")
    ax.set_xlabel(r"$\Delta_t/t$")
    ax.set_ylabel(r"$V^{*}/t$")
    ax.set_title(
        r"Displacement of the entanglement turning point $V^{*}$ with the "
        r"pair interaction $\Delta_t$ (scatter across $r=0.15\ldots0.40$; "
        r"markers offset for clarity)",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, ncol=3, loc="upper right")
    ax.set_xlim(3.3, 6.2)

    fig.suptitle(
        r"Same QPT diagnostics as arXiv:2011.14038 applied to our "
        r"$V_{SO}$ sweep: extrema/zeros of $dC/dV_{SO}$ (exact and "
        r"non-analytic legacy points), displaced by $\Delta_t$ but "
        r"essentially fixed vs. $r$",
        fontsize=12.5,
    )
    fig.savefig(FIGS / "qpt_signature_paper_comparison.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # JSON summary
    # -------------------------------------------------------------------------
    summary = {
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "paper": "arXiv:2011.14038 (Ebrahimian, Khosrojerdi, Afzali)",
        "paper_criteria": [
            "turning point: minimum of dC/dv with d2C/dv2 = 0",
            "local maximum point: dC/dv = 0 with d2C/dv2 minimum",
            "QPT positions displace with couplings J, Delta; invariant vs "
            "impurity concentration and pair distance",
            "cross-check via energy derivatives",
        ],
        "turning_points_Vstar": [
            {
                "Delta_t_over_t": float(dt),
                "Vstar_spin_r025": float(vstar[idt, 0, j025]),
                "Vstar_band_r025": float(vstar[idt, 1, j025]),
                "Vstar_spin_spread_over_r": [
                    float(vstar[idt, 0].min()), float(vstar[idt, 0].max())
                ],
                "Vstar_band_spread_over_r": [
                    float(vstar[idt, 1].min()), float(vstar[idt, 1].max())
                ],
                "d2C_sign_change_at_Vstar_all_r_basis": bool(d2_ok[idt].all()),
                "V_lifshitz_mu_eq_4t": (
                    None if np.isnan(v_life[idt]) else float(v_life[idt])
                ),
                "V_pair_onset": float(v_onset[idt]),
            }
            for idt, dt in enumerate(dts)
        ],
        "interior_local_maxima_of_C_outside_small_gap": local_max,
        "legacy_nonanalytic_events": esd,
        "figure": "figs/qpt_signature_paper_comparison.png",
    }
    (REPORTS / "qpt_paper_comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["turning_points_Vstar"], indent=2))
    print(f"interior local maxima (exact): {len(local_max)}")
    print(f"legacy non-analytic events: {len(esd)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
