"""Effect of U, V_SO and Delta_t on concurrence, singlet fidelity and Holevo.

Two figures (signed helicity construction, original-spin basis, theta=45 deg,
N_k=100, n=1.875):

  figs/parameter_effects_vs_r.png   initial state (t=0): C(r), F_s(r), chi(r)
  figs/parameter_effects_vs_t.png   dynamics at r=0.3 under the low
                                    non-Markovian amplitude-damping channel

Columns sweep one parameter while the other two sit at the defaults
U/t=75, V_SO/t=0.5, Delta_t/t=4.5.  Also writes reports/parameter_effects.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import config
import kspace_setup
import self_consistency
from self_consistency import run_selfconsistency

from regenerate_vso_figures import (  # noqa: E402
    States, LOW, T_FULL, observables, r_vec, evolve_stack,
)

NK = 100
THETA = 45.0
R_GRID = np.linspace(0.0, 6.0, 121)
R_DYN = 0.3                      # inside the entangled region for all sweeps
DEFAULTS = dict(U=75.0, V=0.5, DT=4.5)

SWEEPS = [
    ("U",   [40.0, 75.0, 115.0], dict(V=0.5, DT=4.5)),
    ("V",   [0.0, 0.25, 0.5, 0.75], dict(U=75.0, DT=4.5)),
    ("DT",  [3.5, 4.5, 6.0],   dict(U=75.0, V=0.5)),
]


def solve(v: float, U: float, dt: float) -> dict:
    for mod in (config, kspace_setup, self_consistency):
        mod.V_SO = float(v)
        mod.Nk = int(NK)
        mod.Delta_t = float(dt)
    bdg = run_selfconsistency(seed=42, U_override=float(U),
                              kBT_override=0.0, verbose=False)
    res = float(np.max(np.abs(bdg["residuals"])))
    if not bdg["converged"] or res > 1e-8:
        raise RuntimeError(f"U={U}, V={v}, Dt={dt}: residual={res:.3e}")
    return bdg


def measure(st: States) -> dict:
    """Initial radial profiles + dynamics at R_DYN (theta = THETA)."""
    rhos = np.stack([st.rho(r_vec(r, THETA)) for r in R_GRID])
    o = observables(rhos)
    rho_d = np.stack([st.rho(r_vec(R_DYN, THETA))])
    dyn = observables(evolve_stack(rho_d, T_FULL, **LOW))
    return {
        "C_r": np.asarray(o["C"]),
        "Fs_r": np.asarray(o["Fs"]),
        "chi_r": np.asarray(o["chi"]),
        "C_t": np.asarray(dyn["C"])[0],
        "Fs_t": np.asarray(dyn["Fs"])[0],
        "chi_t": np.asarray(dyn["chi"])[0],
    }


def main() -> None:
    data: dict = {}
    cache: dict = {}
    for name, values, fixed in SWEEPS:
        for val in values:
            U = fixed.get("U", DEFAULTS["U"]) if name != "U" else val
            v = fixed.get("V", DEFAULTS["V"]) if name != "V" else val
            dt = fixed.get("DT", DEFAULTS["DT"]) if name != "DT" else val
            key = (U, v, dt)
            if key not in cache:
                print(f"[solve] U={U} V_SO={v} Dt={dt} ...", flush=True)
                bdg = solve(v, U, dt)
                st = States(bdg)
                cache[key] = {
                    "meas": measure(st),
                    "Delta0": float(bdg["Delta0"]),
                    "DeltaS": float(bdg["DeltaS"]),
                    "mu": float(bdg["mu"]),
                }
            data[f"{name}={val}"] = {"params": {"U": U, "V_SO": v, "Dt": dt},
                                     **cache[key]}

    # ---------------- figure A: initial profiles vs r ----------------
    rows = [("C_r", r"$C(r,\,t=0)$", None),
            ("Fs_r", r"$F_s(r,\,t=0)$", 0.25),
            ("chi_r", r"$\chi_P(r,\,t=0)$ (bit)", None)]
    titles = [
        fr"$U/t$ sweep ($V_{{SO}}/t=0.5$, $\Delta_t/t=4.5$)",
        fr"$V_{{SO}}/t$ sweep ($U/t=75$, $\Delta_t/t=4.5$)",
        fr"$\Delta_t/t$ sweep ($U/t=75$, $V_{{SO}}/t=0.5$)",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11), constrained_layout=True)
    for col, (name, values, fixed) in enumerate(SWEEPS):
        colors = plt.cm.viridis(np.linspace(0.12, 0.92, len(values)))
        for row, (key, ylab, href) in enumerate(rows):
            ax = axes[row, col]
            for ci, val in enumerate(values):
                m = data[f"{name}={val}"]["meas"]
                lbl = {"U": fr"$U/t={val:g}$",
                       "V": fr"$V_{{SO}}/t={val:g}$",
                       "DT": fr"$\Delta_t/t={val:g}$"}[name]
                ax.plot(R_GRID, m[key], color=colors[ci], lw=1.8, label=lbl)
            if href is not None:
                ax.axhline(href, color="0.5", ls=":", lw=1)
            ax.set_xlim(R_GRID[0], R_GRID[-1])
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(titles[col], fontsize=11)
                ax.legend(fontsize=9, loc="upper right")
            if col == 0:
                ax.set_ylabel(ylab, fontsize=11)
            if row == 2:
                ax.set_xlabel(r"$r$ (lattice units)", fontsize=11)
    fig.suptitle(r"Effect of $U$, $V_{SO}$, $\Delta_t$ on the initial"
                 r" conditional-state resources"
                 r" ($\theta=45^\circ$, spin basis, $N_k=100$)",
                 fontsize=13)
    out_a = ROOT / "figs" / "parameter_effects_vs_r.png"
    fig.savefig(out_a, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_a}", flush=True)

    # ---------------- figure B: dynamics at r = R_DYN ----------------
    rows_t = [("C_t", r"$C(r=0.3,\,t)$", None),
              ("Fs_t", r"$F_s(r=0.3,\,t)$", 0.25),
              ("chi_t", r"$\chi_P(r=0.3,\,t)$ (bit)", 1.0)]
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11), constrained_layout=True)
    for col, (name, values, fixed) in enumerate(SWEEPS):
        colors = plt.cm.viridis(np.linspace(0.12, 0.92, len(values)))
        for row, (key, ylab, href) in enumerate(rows_t):
            ax = axes[row, col]
            for ci, val in enumerate(values):
                m = data[f"{name}={val}"]["meas"]
                lbl = {"U": fr"$U/t={val:g}$",
                       "V": fr"$V_{{SO}}/t={val:g}$",
                       "DT": fr"$\Delta_t/t={val:g}$"}[name]
                ax.plot(T_FULL, m[key], color=colors[ci], lw=1.6, label=lbl)
            if href is not None:
                ax.axhline(href, color="0.5", ls=":", lw=1)
            ax.set_xlim(T_FULL[0], T_FULL[-1])
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(titles[col], fontsize=11)
                ax.legend(fontsize=9, loc="upper right")
            if col == 0:
                ax.set_ylabel(ylab, fontsize=11)
            if row == 2:
                ax.set_xlabel(r"$t$ ($1/t$ units)", fontsize=11)
    fig.suptitle(r"Parameter effects under non-Markovian amplitude damping"
                 r" ($\gamma=1.5$, $\Gamma=0.1$, $r=0.3$, $\theta=45^\circ$)",
                 fontsize=13)
    out_b = ROOT / "figs" / "parameter_effects_vs_t.png"
    fig.savefig(out_b, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_b}", flush=True)

    # ---------------- figure C: sensitivity (y = observable, x = knob) --------
    # Effects are 1-5%: plotting the observable against the swept parameter
    # with tight y-limits makes the monotone dependence visible.
    R_LINES = {"C_r": [0.15, 0.3, 0.4],
               "Fs_r": [0.25, 0.5, 1.0],
               "chi_r": [0.25, 0.5, 0.75]}
    T_LINES = [0.0, 12.0, 30.0]
    REF_IDX = {"U": 1, "V": 2, "DT": 1}       # U=75, V=0.5, Dt=4.5

    def knob_x(name, values):
        return np.arange(len(values), dtype=float)

    def spread(vals, ref_i):
        v = np.asarray(vals, dtype=float)
        if abs(v[ref_i]) < 1e-12:
            return None                     # reference is zero: no % to show
        return 100.0 * (v[-1] - v[ref_i]) / v[ref_i]

    row_lab_r = [r"$C(r,\,t=0)$", r"$F_s(r,\,t=0)$",
                 r"$\chi_P(r,\,t=0)$ (bit)"]
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11), constrained_layout=True)
    keys_r = ["C_r", "Fs_r", "chi_r"]
    for col, (name, values, fixed) in enumerate(SWEEPS):
        ref = REF_IDX[name]
        for row, ykey in enumerate(keys_r):
            ax = axes[row, col]
            for lv in R_LINES[ykey]:
                ys = []
                for val in values:
                    m = data[f"{name}={val}"]["meas"]
                    ir = int(np.argmin(np.abs(R_GRID - lv)))
                    ys.append(float(m[ykey][ir]))
                sp = spread(ys, ref)
                ax.plot(np.arange(len(values)), ys, marker="o", lw=1.8,
                        ms=6, label=fr"$r={lv:g}$" + ("" if sp is None else f" ({sp:+.1f}%)"))
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel(row_lab_r[row], fontsize=11)
            xt = {"U": [fr"$U/t={v:g}$" for v in values],
                  "V": [fr"$V_{{SO}}/t={v:g}$" for v in values],
                  "DT": [fr"$\Delta_t/t={v:g}$" for v in values]}[name]
            ax.set_xticks(np.arange(len(values)))
            if row == 2:
                ax.set_xticklabels(xt, fontsize=9)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if row == 0:
                ax.set_title(titles[col], fontsize=11)
            ax.legend(fontsize=8.5, loc="best")
            ax.set_xlim(-0.3, len(values) - 0.7)
    fig.suptitle(r"Parameter sensitivity of the initial resources: observable"
                 r" vs parameter, labelled by separation"
                 r" ($\theta=45^\circ$, $t=0$, spin basis; % = change from"
                 r" the reference value)", fontsize=12)
    out_c = ROOT / "figs" / "parameter_effects_sensitivity.png"
    fig.savefig(out_c, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_c}", flush=True)

    # -- sensitivity of the dynamics (lines = times) --
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11), constrained_layout=True)
    keys_t = ["C_t", "Fs_t", "chi_t"]
    row_lab_t = [r"$C(r=0.3,\,t)$", r"$F_s(r=0.3,\,t)$",
                 r"$\chi_P(r=0.3,\,t)$ (bit)"]
    for col, (name, values, fixed) in enumerate(SWEEPS):
        ref = REF_IDX[name]
        for row, ykey in enumerate(keys_t):
            ax = axes[row, col]
            for lv in T_LINES:
                ys = []
                for val in values:
                    m = data[f"{name}={val}"]["meas"]
                    it = int(np.argmin(np.abs(T_FULL - lv)))
                    ys.append(float(m[ykey][it]))
                sp = spread(ys, ref)
                ax.plot(np.arange(len(values)), ys, marker="o", lw=1.8,
                        ms=6, label=fr"$t={lv:g}$" + ("" if sp is None else f" ({sp:+.1f}%)"))
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel(row_lab_t[row], fontsize=11)
            xt = {"U": [fr"$U/t={v:g}$" for v in values],
                  "V": [fr"$V_{{SO}}/t={v:g}$" for v in values],
                  "DT": [fr"$\Delta_t/t={v:g}$" for v in values]}[name]
            ax.set_xticks(np.arange(len(values)))
            if row == 2:
                ax.set_xticklabels(xt, fontsize=9)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if row == 0:
                ax.set_title(titles[col], fontsize=11)
            ax.legend(fontsize=8.5, loc="best")
            ax.set_xlim(-0.3, len(values) - 0.7)
    fig.suptitle(r"Parameter sensitivity of the dynamics at $r=0.3$:"
                 r" observable vs parameter at three times"
                 r" ($\gamma=1.5$, $\Gamma=0.1$; % = change from the"
                 r" reference value)", fontsize=12)
    out_d = ROOT / "figs" / "parameter_effects_sensitivity_t.png"
    fig.savefig(out_d, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_d}", flush=True)

    # -- relative change (% of the reference parameter value) --
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11), constrained_layout=True)
    for col, (name, values, fixed) in enumerate(SWEEPS):
        ref = REF_IDX[name]
        for row, ykey in enumerate(keys_r):
            ax = axes[row, col]
            for lv in R_LINES[ykey]:
                ys = []
                for val in values:
                    m = data[f"{name}={val}"]["meas"]
                    ir = int(np.argmin(np.abs(R_GRID - lv)))
                    ys.append(float(m[ykey][ir]))
                rel_y = 100.0 * np.asarray(ys) / ys[ref]
                ax.plot(np.arange(len(values)), rel_y, marker="o", lw=1.8,
                        ms=6, label=fr"$r={lv:g}$")
            ax.axhline(100.0, color="0.5", ls=":", lw=1)
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel(row_lab_r[row] + "  (rel. %)", fontsize=11)
            xt = {"U": [fr"$U/t={v:g}$" for v in values],
                  "V": [fr"$V_{{SO}}/t={v:g}$" for v in values],
                  "DT": [fr"$\Delta_t/t={v:g}$" for v in values]}[name]
            ax.set_xticks(np.arange(len(values)))
            if row == 2:
                ax.set_xticklabels(xt, fontsize=9)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if row == 0:
                ax.set_title(titles[col], fontsize=11)
            ax.legend(fontsize=8.5, loc="best")
            ax.set_xlim(-0.3, len(values) - 0.7)
            ax.margins(y=0.25)
    fig.suptitle(r"Relative parameter effect on the initial resources"
                 r" (observable / value at reference, %;"
                 r" ref: $U/t=75$, $V_{SO}/t=0.5$, $\Delta_t/t=4.5$,"
                 r" $\theta=45^\circ$)", fontsize=12)
    out_e = ROOT / "figs" / "parameter_effects_relative.png"
    fig.savefig(out_e, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_e}", flush=True)

    # ---------------- report ----------------
    report = {}
    for k, entry in data.items():
        m = entry["meas"]
        i05 = int(np.argmin(np.abs(R_GRID - 0.5)))
        i15 = int(np.argmin(np.abs(R_GRID - 1.5)))
        report[k] = {
            "params": entry["params"],
            "Delta0": entry["Delta0"], "DeltaS": entry["DeltaS"],
            "mu": entry["mu"],
            "initial": {
                "C_max": float(m["C_r"].max()),
                "Fs_r0p5": float(m["Fs_r"][i05]),
                "chi_r0p5": float(m["chi_r"][i05]),
                "chi_r1p5": float(m["chi_r"][i15]),
            },
            "dynamics_r0p3": {
                "C_t0": float(m["C_t"][0]),
                "Fs_t0": float(m["Fs_t"][0]),
                "chi_t0": float(m["chi_t"][0]),
                "chi_t80": float(m["C_t"][-1] * 0 + m["chi_t"][-1]),
            },
        }
    out_j = ROOT / "reports" / "parameter_effects.json"
    out_j.write_text(json.dumps(report, indent=1))
    print(f"saved {out_j}", flush=True)


if __name__ == "__main__":
    main()
