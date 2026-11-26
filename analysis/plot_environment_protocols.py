#!/usr/bin/env python3
"""Paper-style environment-parameter figures for teleportation, dense coding,
and concurrence.

Analog of Haddadi, Hadipour, Haseli, Rahman & Czerwinski, Mathematics 11,
1407 (2023), Figures 2, 3, 5 and 6 (their teleportation F_A, dense-coding
Holevo chi, and time/regime scans) for this repository's BdG two-qubit
resources at V_SO/t = 0.5 (helicity conditional-Wick states transformed to
the original-spin basis), with beta = 0 (no Stark shift).

Regime mapping (analytic, verified numerically against fidelity.py):
    paper (two-photon):        eps'' + lam eps' + g0 lam eps = 0
    repository (Lorentzian):   eta'' + Gamma eta' + (gamma Gamma/2) eta = 0
    => Gamma = lam, gamma = 2 g0, tau = g0 t with g0 = 1.
    Non-Markovian: lam = 0.1 g0  ->  (gamma, Gamma) = (2.0, 0.1)
    Markovian:     lam = 10 g0   ->  (gamma, Gamma) = (2.0, 10.0)

The spectral-width scan fixes gamma = 2 and varies Gamma over [0.01, 20]
(the upper bound keeps cosh(d t/2)^2 below float64 overflow for t <= 30);
the underdamped/overdamped crossover is Gamma = 2 gamma = 4.

Teleportation is reported as both the fixed singlet-protocol fidelity
(paper-comparable: their F_A tends to 1/3 after complete damping) and the
locally optimized fidelity (Horodecki (2F_max+1)/3).

Outputs
-------
  figs/environment_protocols_regimes.png   3 rows (F, chi, C) x 2 regimes
  figs/environment_parameter_scan.png      3 heatmaps vs (tau, Gamma)
  data/environment_protocols.npz
  reports/environment_protocols.json

Run from the repository root:
    .venv/bin/python analysis/plot_environment_protocols.py
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

import config
import greens_functions as gf_module
import kspace_setup
import self_consistency as sc_module
from density_matrix import build_density_matrices, verify_density_matrices
from evolution import evolve_rho
from fidelity import _evolve_rho_fid
from helicity_density_matrix import dense_coding_metrics
from self_consistency import run_selfconsistency
from teleportation import resource_metrics, singlet_state

DATA = ROOT / "data"
FIGS = ROOT / "figs"
REPORTS = ROOT / "reports"

# --- experiment design -------------------------------------------------------
U_VALUE = 75.0
VSO_VALUE = 0.5                        # nonzero spin-orbit coupling (helicity)
NK = 400
THETA_DEG = (45.0,)
R_POINTS = (0.0, 0.30, 0.50, 1.0)      # r=0 -> pure singlet at any V_SO
TAU = np.linspace(0.0, 30.0, 601)      # tau = gamma0 t, gamma0 = 1
REGIMES = {                            # Gamma = lam, gamma = 2 gamma0
    "non_markovian": dict(gamma=2.0, Gamma=0.1, lam="0.1 gamma0"),
    "markovian":     dict(gamma=2.0, Gamma=10.0, lam="10 gamma0"),
}
REGIME_ORDER = ("non_markovian", "markovian")
SCAN_GAMMA = 2.0
SCAN_GRID = np.logspace(-2.0, np.log10(20.0), 61)
METRIC_KEYS = ("fixed_average_fidelity", "optimized_average_fidelity",
               "holevo_pauli_bits", "concurrence")
CLASSICAL_F = 2.0 / 3.0
CLASSICAL_CHI = 1.0


# --- helpers copied from run_microscopic_teleportation_study.py --------------
def maxabs(value):
    return float(np.max(np.abs(value)))


class temporary_spatial_grid:
    def __init__(self, r_max, n_r, angles):
        self.old = (gf_module.R_MIN, gf_module.R_MAX, gf_module.N_R,
                    gf_module.THETA_DEG)
        self.new = (0.0, float(r_max), int(n_r), list(angles))

    def __enter__(self):
        (gf_module.R_MIN, gf_module.R_MAX, gf_module.N_R,
         gf_module.THETA_DEG) = self.new

    def __exit__(self, *exc):
        (gf_module.R_MIN, gf_module.R_MAX, gf_module.N_R,
         gf_module.THETA_DEG) = self.old
        return False


def solve_bdg(U, nk, vso=VSO_VALUE):
    old_sc_nk, old_k_nk = sc_module.Nk, kspace_setup.Nk
    old_vso = {m: float(getattr(m, "V_SO"))
               for m in (config, kspace_setup, sc_module)}
    sc_module.Nk = kspace_setup.Nk = int(nk)
    for m in (config, kspace_setup, sc_module):
        m.V_SO = float(vso)
    try:
        bdg = run_selfconsistency(seed=42, U_override=float(U),
                                  kBT_override=0.0, verbose=False)
    finally:
        sc_module.Nk, kspace_setup.Nk = old_sc_nk, old_k_nk
        for m, value in old_vso.items():
            m.V_SO = value
    residual = maxabs(bdg["residuals"])
    paired = max(abs(float(bdg["Delta0"])), abs(float(bdg["DeltaS"]))) > 1e-5
    if residual > 1e-8 or not paired or not bdg["converged"]:
        raise RuntimeError(f"U={U}, Nk={nk}: residual={residual}, "
                           f"converged={bdg['converged']}, paired={paired}")
    return bdg, residual


def greens_on_grid(bdg, r_max, n_r, angles=THETA_DEG):
    from greens_functions import compute_greens_functions
    with temporary_spatial_grid(r_max, n_r, angles):
        return compute_greens_functions(bdg, verbose=False,
                                        convention="equal_time")


def werner_matrix(p):
    singlet = np.outer(singlet_state(), singlet_state().conj())
    return (1.0 - p) * np.eye(4) / 4.0 + p * singlet


def metric_row(rho):
    rm = resource_metrics(rho)
    dc = dense_coding_metrics(rho)
    return [rm["fixed_average_fidelity"], rm["optimized_average_fidelity"],
            dc["holevo_pauli_bits"], rm["concurrence"]]


def evolve_metrics(rho0, t_grid, gamma, Gamma):
    out = np.zeros((len(t_grid), len(METRIC_KEYS)))
    for k, time in enumerate(t_grid):
        rho_t = evolve_rho(rho0, float(time), kind="amp",
                           bath="independent",
                           gammaA=gamma, GammaA=Gamma,
                           gammaB=gamma, GammaB=Gamma)
        out[k] = metric_row(rho_t)
        if abs(np.trace(rho_t).real - 1.0) > 1e-12:
            raise RuntimeError(f"trace drift at t={time}")
    return out


def threshold_summary(t, values, threshold):
    summary = {
        "initial": float(values[0]),
        "minimum": float(values.min()),
        "final": float(values[-1]),
        "first_loss_time": None,
        "n_losses": 0,
        "n_regains": 0,
    }
    if threshold is not None:
        above = values > threshold
        losses = [i for i in range(len(t) - 1)
                  if above[i] and not above[i + 1]]
        regains = [i for i in range(len(t) - 1)
                   if not above[i] and above[i + 1]]
        summary["first_loss_time"] = (float(t[losses[0]]) if losses else None)
        summary["n_losses"] = int(len(losses))
        summary["n_regains"] = int(len(regains))
    return summary


# --- figures -----------------------------------------------------------------
def plot_regimes(tau, series, output):
    """series[regime][ip] -> (Nt, 4) in METRIC_KEYS order."""
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(R_POINTS)))
    labels = [f"$r={r:.2f}$" for r in R_POINTS]
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 11.0), sharex=True)

    row_specs = (
        (0, 0, CLASSICAL_F, "Teleportation fidelity"),
        (1, 2, CLASSICAL_CHI, "Dense coding $\\chi$ [bits]"),
        (2, 3, None, "Concurrence"),
    )
    for col, regime in enumerate(REGIME_ORDER):
        for row, metric_idx, bound, title in row_specs:
            ax = axes[row, col]
            for ip, r in enumerate(R_POINTS):
                data = series[regime][ip]
                ax.plot(tau, data[:, metric_idx], color=colors[ip],
                        lw=1.8, label=labels[ip])
                if row == 0:      # optimized companion (dashed)
                    ax.plot(tau, data[:, 1], color=colors[ip],
                            lw=1.0, ls="--", alpha=0.75)
            if row == 0:
                ax.plot([], [], color="gray", lw=1.0, ls="--",
                        label="optimized")
            if bound is not None:
                ax.axhline(bound, color="black", lw=1.1, ls=":",
                           label=("classical $2/3$" if row == 0
                                  else "1-bit bound"))
            if col == 0 and row > 0:
                ax.set_title(title, fontsize=11, loc="left")
            ax.grid(alpha=0.25, lw=0.5)
            ax.set_ylim(-0.03, 2.08 if row == 1 else 1.03)
            if col == 1 and row == 0:
                ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
        pars = REGIMES[regime]
        axes[0, col].set_title(
            ("Non-Markovian: $\\lambda=0.1\\gamma_0$ "
             if regime == "non_markovian" else
             "Markovian: $\\lambda=10\\gamma_0$ ")
            + f"($\\gamma={pars['gamma']:g}$, $\\Gamma={pars['Gamma']:g}$)",
            fontsize=11)
        for row in range(3):
            axes[row, col].set_ylabel(
                ["$F$ (solid: fixed, dashed: optimized)",
                 "$\\chi$ [bits]", "$C$"][row])
        axes[2, col].set_xlabel(r"scaled time $\tau=\gamma_0 t$")
    fig.suptitle("Environment effects on teleportation, dense coding, and "
                 "concurrence (no Stark shift; BdG resources $U/t=75$)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[env-figs] saved -> {output}")


def plot_scan(tau, scan, output):
    """scan -> (N_Gamma, Nt, 4) for the r=0 resource."""
    fig, axes = plt.subplots(3, 1, figsize=(7.8, 11.0), sharex=True)
    specs = ((0, 0, "Teleportation $F$ (fixed protocol)", "viridis",
              CLASSICAL_F, (0.0, 1.0)),
             (1, 2, "Dense coding $\\chi$ [bits]", "magma",
              CLASSICAL_CHI, (0.0, 2.0)),
             (2, 3, "Concurrence $C$", "cividis", None, (0.0, 1.0)))
    for row, metric_idx, title, cmap, bound, vlim in specs:
        ax = axes[row]
        values = scan[:, :, metric_idx]     # (NGamma, Nt) = (len y, len x)
        mesh = ax.pcolormesh(tau, SCAN_GRID, values, shading="nearest",
                             cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        if bound is not None:
            ax.contour(tau, SCAN_GRID, values, levels=[bound],
                       colors="white", linewidths=1.3)
        ax.axhline(2.0 * SCAN_GAMMA, color="white", lw=1.0, ls="--")
        ax.text(tau[-1] * 0.985, 2.0 * SCAN_GAMMA * 1.15,
                r"$\Gamma=2\gamma$ crossover", color="white", fontsize=8.5,
                ha="right")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel(r"$\Gamma$ (spectral width)")
        fig.colorbar(mesh, ax=ax, pad=0.02)
    axes[-1].set_xlabel(r"scaled time $\tau=\gamma_0 t$")
    fig.suptitle("Reservoir spectral-width scan at $\\gamma=2$ "
                 "(maximally entangled $r=0$ resource); "
                 "white contours: classical bounds", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[env-figs] saved -> {output}")


# --- main --------------------------------------------------------------------
def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    print(f"Stage 1/4: BdG solve U={U_VALUE}, Nk={NK}", flush=True)
    bdg, residual = solve_bdg(U_VALUE, NK)
    print(f"  residual={residual:.2e}, "
          f"Delta0={float(bdg['Delta0']):.6f}, "
          f"DeltaS={float(bdg['DeltaS']):.6f}", flush=True)

    print("Stage 2/4: resource states (helicity spin basis, V_SO/t=0.5)",
          flush=True)
    from helicity_density_matrix import density_at_separation
    states, initial_metrics, rho_initial = [], [], []
    th = np.deg2rad(THETA_DEG[0])
    for radius in R_POINTS:
        r_vec = np.array([radius * np.cos(th), radius * np.sin(th)])
        rho = density_at_separation(bdg, r_vec, basis="spin")["rho"]
        states.append(np.asarray(rho))
        rho_initial.append(rho)
        initial_metrics.append(dict(zip(METRIC_KEYS, metric_row(rho))))
        print(f"  r={radius:.2f}: " + ", ".join(
            f"{k}={v:.4f}" for k, v in initial_metrics[-1].items()),
            flush=True)
    for rho in rho_initial:
        if not verify_density_matrices([[rho]], atol=2e-8):
            raise RuntimeError("density-matrix validation failed")

    print("Stage 3/4: regime dynamics + cross-implementation check", flush=True)
    cross_error = 0.0
    series = {regime: [] for regime in REGIME_ORDER}
    for regime in REGIME_ORDER:
        pars = REGIMES[regime]
        for ip, rho in enumerate(states):
            series[regime].append(
                evolve_metrics(rho, TAU, pars["gamma"], pars["Gamma"]))
            print(f"  {regime}, r={R_POINTS[ip]:.2f}", flush=True)
        # evolve_rho (noise_channels) must equal fidelity.py's eta path
        for ip in (0, 2):
            for time in (0.0, 5.0, 25.0):
                a = evolve_rho(states[ip], time, kind="amp",
                               bath="independent",
                               gammaA=pars["gamma"], GammaA=pars["Gamma"],
                               gammaB=pars["gamma"], GammaB=pars["Gamma"])
                b = _evolve_rho_fid(states[ip], time, kind="amp",
                                    bath="independent",
                                    nuA=0.2, GammaA=0.05, nuB=0.2,
                                    GammaB=0.05,
                                    gammaA=pars["gamma"],
                                    GammaAD_A=pars["Gamma"],
                                    gammaB=pars["gamma"],
                                    GammaAD_B=pars["Gamma"])
                cross_error = max(cross_error, maxabs(a - b))
    if cross_error > 2e-12:
        raise RuntimeError(f"channel implementations disagree: {cross_error}")
    print(f"  cross-implementation max|diff| = {cross_error:.2e}", flush=True)

    print("Stage 4/4: spectral-width scan, figures, outputs", flush=True)
    scan = np.zeros((len(SCAN_GRID), len(TAU), len(METRIC_KEYS)))
    for ig, width in enumerate(SCAN_GRID):
        scan[ig] = evolve_metrics(states[0], TAU, SCAN_GAMMA, float(width))
        if ig % 15 == 0:
            print(f"  scan Gamma index {ig+1}/{len(SCAN_GRID)}", flush=True)

    # --- validation ----------------------------------------------------------
    all_series = [s for regime in REGIME_ORDER for s in series[regime]]
    all_series.append(scan.reshape(-1, len(METRIC_KEYS)))
    stack = np.concatenate(all_series, axis=0)
    validation = {
        "cross_implementation_max_abs_error": float(cross_error),
        "fidelity_range": [float(stack[:, 0].min()), float(stack[:, 0].max())],
        "holevo_range": [float(stack[:, 2].min()), float(stack[:, 2].max())],
        "concurrence_range": [float(stack[:, 3].min()),
                              float(stack[:, 3].max())],
        "fixed_vs_optimized_max_gap":
            float(np.max(np.abs(stack[:, 0] - stack[:, 1]))),
    }
    if not (-1e-9 <= validation["fidelity_range"][0]
            and validation["fidelity_range"][1] <= 1 + 1e-9):
        raise RuntimeError(f"fidelity out of range: {validation}")
    if not (-1e-9 <= validation["holevo_range"][0]
            and validation["holevo_range"][1] <= 2 + 1e-9):
        raise RuntimeError(f"holevo out of range: {validation}")
    if not (-1e-9 <= validation["concurrence_range"][0]
            and validation["concurrence_range"][1] <= 1 + 1e-9):
        raise RuntimeError(f"concurrence out of range: {validation}")

    # --- figures -------------------------------------------------------------
    plot_regimes(TAU, series,
                 FIGS / "environment_protocols_regimes.png")
    plot_scan(TAU, scan, FIGS / "environment_parameter_scan.png")

    # --- data ----------------------------------------------------------------
    np.savez_compressed(
        DATA / "environment_protocols.npz",
        tau=TAU,
        r_points=np.array(R_POINTS),
        metric_keys=np.array(METRIC_KEYS),
        regime_names=np.array(REGIME_ORDER),
        gamma_values=np.array([REGIMES[r]["gamma"] for r in REGIME_ORDER]),
        Gamma_values=np.array([REGIMES[r]["Gamma"] for r in REGIME_ORDER]),
        metrics_regimes=np.stack(
            [np.stack(series[r]) for r in REGIME_ORDER]),   # (2,Nr,Nt,4)
        scan_gamma_grid=SCAN_GRID,
        scan_gamma=SCAN_GAMMA,
        metrics_scan=scan,                                  # (NG,Nt,4)
        rho_initial=np.array(rho_initial),
        initial_metrics=np.array([list(m.values())
                                  for m in initial_metrics]),
    )

    # --- report --------------------------------------------------------------
    dynamics = {}
    for regime in REGIME_ORDER:
        dynamics[regime] = {}
        for ip, radius in enumerate(R_POINTS):
            data = series[regime][ip]
            dynamics[regime][f"{radius:.2f}"] = {
                "fixed_fidelity": threshold_summary(
                    TAU, data[:, 0], CLASSICAL_F),
                "optimized_fidelity": threshold_summary(
                    TAU, data[:, 1], CLASSICAL_F),
                "holevo": threshold_summary(TAU, data[:, 2], CLASSICAL_CHI),
                "concurrence": threshold_summary(
                    TAU, data[:, 3], None),
            }
    scan_selected = {}
    for target in (0.01, 0.1, 1.0, 4.0, 10.0, 20.0):
        ig = int(np.argmin(np.abs(SCAN_GRID - target)))
        row = scan[ig]
        scan_selected[f"{SCAN_GRID[ig]:g}"] = {
            "underdamped": bool(2.0 * SCAN_GAMMA * SCAN_GRID[ig]
                                - SCAN_GRID[ig] ** 2 > 0),
            "fixed_fidelity_final": float(row[-1, 0]),
            "holevo_final": float(row[-1, 2]),
            "concurrence_final": float(row[-1, 3]),
            "holevo_minimum": float(row[:, 2].min()),
        }
    report = {
        "paper": {
            "title": "Quantum Advantages of Teleportation and Dense Coding "
                     "Protocols in an Open System",
            "authors": "Haddadi, Hadipour, Haseli, Rahman, Czerwinski",
            "journal": "Mathematics 11, 1407 (2023)",
            "doi": "10.3390/math11061407",
            "analog_figures": [2, 3, 5, 6],
            "stark_shift": "beta = 0 (excluded by request)",
        },
        "regime_mapping": {
            "paper_ode": "eps'' + lam eps' + g0 lam eps = 0 (beta=0)",
            "repo_ode": "eta'' + Gamma eta' + (gamma Gamma/2) eta = 0",
            "map": "Gamma = lam, gamma = 2 g0, tau = g0 t (g0 = 1)",
            "non_markovian": {"lam": "0.1 gamma0", "gamma": 2.0, "Gamma": 0.1},
            "markovian": {"lam": "10 gamma0", "gamma": 2.0, "Gamma": 10.0},
        },
        "model": {"U_over_t": U_VALUE, "V_SO_over_t": VSO_VALUE,
                  "state_construction": "helicity conditional Wick, spin basis",
                  "Nk": NK, "theta_deg": list(THETA_DEG),
                  "Delta_t_over_t": float(config.Delta_t),
                  "n_elec": float(config.n_elec)},
        "resources": {f"{r:.2f}": init
                      for r, init in zip(R_POINTS, initial_metrics)},
        "dynamics": dynamics,
        "spectral_width_scan": {"gamma": SCAN_GAMMA,
                                "Gamma_range": [float(SCAN_GRID[0]),
                                                float(SCAN_GRID[-1])],
                                "selected": scan_selected},
        "validation": validation,
        "git_commit": commit,
    }
    (REPORTS / "environment_protocols.json").write_text(
        json.dumps(report, indent=2))
    print(f"[env-figs] saved -> {REPORTS / 'environment_protocols.json'}",
          flush=True)
    print(json.dumps({"validation": validation,
                      "markovian_final": dynamics["markovian"]["0.00"],
                      "non_markovian_final":
                          dynamics["non_markovian"]["0.00"]}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
