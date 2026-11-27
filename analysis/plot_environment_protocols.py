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
from helicity_density_matrix import (
    SINGLET,
    concurrence as wootters_concurrence,
    dense_coding_metrics,
)
from noise_channels import get_kraus_pair
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
            dc["holevo_pauli_bits"], wootters_concurrence(rho)]


_PAULIS = (
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
)
_PAULI_PRODS = np.array(
    [[np.kron(a, b) for b in _PAULIS] for a in _PAULIS], dtype=complex
)  # (3, 3, 4, 4)
_Y2 = np.kron(_PAULIS[1], _PAULIS[1])


def evolve_metrics(rho0, t_grid, gamma, Gamma):
    """Vectorized exact Kraus evolution + protocol metrics for (Nt, 4)."""
    rho0 = np.asarray(rho0, dtype=complex)
    nt = len(t_grid)
    rhos = np.empty((nt, 4, 4), dtype=complex)
    v0 = rho0.reshape(16)
    for k, time in enumerate(t_grid):
        Ka, Kb = get_kraus_pair(float(time), "amp", gamma, Gamma, gamma, Gamma)
        S = np.zeros((16, 16), dtype=complex)
        for A in Ka:
            for B in Kb:
                K = np.kron(A, B)
                S += np.kron(K.conj(), K)
        rho_t = (v0 @ S.T).reshape(4, 4)
        rho_t = 0.5 * (rho_t + rho_t.conj().T)
        tr = float(np.trace(rho_t).real)
        if abs(tr - 1.0) > 1e-12:
            raise RuntimeError(f"trace drift at t={time}: {tr}")
        rhos[k] = rho_t / tr

    # Fixed average fidelity = (2 * F_singlet + 1) / 3 (verified against 6-state average)
    fs = np.real(np.einsum("a,nab,b->n", SINGLET.conj(), rhos, SINGLET))
    f_fixed = (2.0 * fs + 1.0) / 3.0

    # Optimized average fidelity via T_ij = Tr(rho (sigma_i x sigma_j))
    T_corr = np.real(np.einsum("nab,ijba->nij", rhos, _PAULI_PRODS))
    svals = np.linalg.svd(T_corr, compute_uv=False)
    fef = np.clip((1.0 + np.sum(svals, axis=-1)) / 4.0, 0.0, 1.0)
    f_opt = (2.0 * fef + 1.0) / 3.0

    # Pauli Holevo chi = 1 + S(rho_B) - S(rho_AB)
    evals_ab = np.clip(np.linalg.eigvalsh(rhos).real, 0.0, 1.0)
    nz_ab = evals_ab > 1e-15
    s_ab = -np.sum(np.where(nz_ab, evals_ab * np.log2(np.where(nz_ab, evals_ab, 1.0)), 0.0), axis=-1)
    rho_b = np.trace(rhos.reshape(nt, 2, 2, 2, 2), axis1=1, axis2=3)
    evals_b = np.clip(np.linalg.eigvalsh(rho_b).real, 0.0, 1.0)
    nz_b = evals_b > 1e-15
    s_b = -np.sum(np.where(nz_b, evals_b * np.log2(np.where(nz_b, evals_b, 1.0)), 0.0), axis=-1)
    chi = np.clip(1.0 + s_b - s_ab, 0.0, 2.0)

    # Exact Wootters concurrence
    flipped = _Y2 @ rhos.conj() @ _Y2
    eigs_r = np.linalg.eigvals(rhos @ flipped)
    roots = np.sqrt(np.clip(eigs_r.real, 0.0, None))
    roots.sort(axis=-1)
    conc = np.maximum(0.0, roots[:, -1] - np.sum(roots[:, :-1], axis=-1))

    return np.column_stack([f_fixed, f_opt, chi, conc])


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
def plot_regimes(tau, series_spin, series_band, output):
    """3 rows (F, chi, C) x 3 cols (Non-Markovian | Markovian | Band - Spin)."""
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(R_POINTS)))
    labels = [f"$r={r:.2f}$" for r in R_POINTS]
    fig, axes = plt.subplots(3, 3, figsize=(15.4, 11.2), sharex=True)

    row_specs = (
        (0, 0, CLASSICAL_F, "Teleportation fidelity"),
        (1, 2, CLASSICAL_CHI, "Dense coding $\\chi_{\\mathrm{P}}$ [bits]"),
        (2, 3, None, "Concurrence"),
    )
    for col, regime in enumerate(REGIME_ORDER):
        for row, metric_idx, bound, title in row_specs:
            ax = axes[row, col]
            for ip, _ in enumerate(R_POINTS):
                ds = series_spin[regime][ip]
                db = series_band[regime][ip]
                ax.plot(tau, ds[:, metric_idx], color=colors[ip],
                        lw=1.8, label=labels[ip])
                ax.plot(tau, db[:, metric_idx], color=colors[ip],
                        lw=1.5, ls="-.")
                if row == 0:      # optimized companion (thin dashed)
                    ax.plot(tau, ds[:, 1], color=colors[ip],
                            lw=1.0, ls="--", alpha=0.65)
            if row == 0:
                ax.plot([], [], color="black", lw=1.6, ls="-", label="spin (fixed)")
                ax.plot([], [], color="black", lw=1.5, ls="-.", label="band (fixed)")
                ax.plot([], [], color="gray", lw=1.0, ls="--", label="spin (opt.)")
            if bound is not None:
                ax.axhline(bound, color="black", lw=1.1, ls=":",
                           label=("classical $2/3$" if row == 0
                                  else "1-bit bound"))
            if col == 0 and row > 0:
                ax.set_title(title, fontsize=11, loc="left")
            ax.grid(alpha=0.25, lw=0.5)
            ax.set_ylim(-0.03, 2.08 if row == 1 else 1.03)
            if col == 1 and row == 0:
                ax.legend(loc="lower left", fontsize=7.5, ncol=2, framealpha=0.9)
        pars = REGIMES[regime]
        axes[0, col].set_title(
            ("Non-Markovian: $\\lambda=0.1\\gamma_0$ "
             if regime == "non_markovian" else
             "Markovian: $\\lambda=10\\gamma_0$ ")
            + f"($\\gamma={pars['gamma']:g}$, $\\Gamma={pars['Gamma']:g}$)",
            fontsize=11)
        for row in range(3):
            axes[row, col].set_ylabel(
                ["$F$ (solid: spin, -.: band, --: opt.)",
                 "$\\chi_{\\mathrm{P}}$ [bits] (solid: spin, -.: band)",
                 "$C$ (solid: spin, -.: band)"][row])
        axes[2, col].set_xlabel(r"scaled time $\tau=\gamma_0 t$")

    # Column 2: Band - Spin basis difference across both regimes
    diff_specs = (
        (0, 0, r"$\Delta F_{\mathrm{fixed}} = F^{\mathrm{band}} - F^{\mathrm{spin}}$",
         "Basis difference (band $-$ spin)"),
        (1, 2, r"$\Delta \chi_{\mathrm{P}} = \chi_{\mathrm{P}}^{\mathrm{band}} - \chi_{\mathrm{P}}^{\mathrm{spin}}$ [bits]",
         None),
        (2, 3, r"$\Delta C = C^{\mathrm{band}} - C^{\mathrm{spin}}$", None),
    )
    for row, metric_idx, ylabel, title in diff_specs:
        ax = axes[row, 2]
        for ip, _ in enumerate(R_POINTS):
            nm_diff = series_band["non_markovian"][ip][:, metric_idx] - series_spin["non_markovian"][ip][:, metric_idx]
            mk_diff = series_band["markovian"][ip][:, metric_idx] - series_spin["markovian"][ip][:, metric_idx]
            ax.plot(tau, nm_diff, color=colors[ip], lw=1.7, label=labels[ip])
            ax.plot(tau, mk_diff, color=colors[ip], lw=1.3, ls="--")
        ax.axhline(0.0, color="gray", lw=0.8, ls=":")
        if row == 0:
            ax.plot([], [], color="black", lw=1.6, ls="-", label="non-Markovian")
            ax.plot([], [], color="black", lw=1.3, ls="--", label="Markovian")
            ax.set_title(title, fontsize=11)
            ax.legend(loc="best", fontsize=7.5, ncol=2, framealpha=0.9)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, lw=0.5)
    axes[2, 2].set_xlabel(r"scaled time $\tau=\gamma_0 t$")

    fig.suptitle("Environment effects on teleportation, dense coding, and "
                 "concurrence: spin vs. helicity band basis ($U/t=75$, $V_{\\mathrm{SO}}/t=0.5$)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[env-figs] saved -> {output}")


def plot_scan(tau, scan_r0, scan_r030_spin, scan_r030_band, output):
    """3 rows (F, chi, C) x 3 cols: r=0 (spin=band) | r=0.30 band | r=0.30 (band - spin)."""
    fig, axes = plt.subplots(3, 3, figsize=(15.6, 11.2), sharex=True, sharey=True)
    specs = (
        (0, 0, "Teleportation $F$ (fixed)", "viridis", CLASSICAL_F, (0.0, 1.0), r"$\Delta F_{\mathrm{fixed}}$"),
        (1, 2, "Dense coding $\\chi_{\\mathrm{P}}$ [bits]", "magma", CLASSICAL_CHI, (0.0, 2.0), r"$\Delta \chi_{\mathrm{P}}$ [bits]"),
        (2, 3, "Concurrence $C$", "cividis", None, (0.0, 1.0), r"$\Delta C$"),
    )
    diff_arr = scan_r030_band - scan_r030_spin
    col_titles = (
        r"$r=0$ (spin $\equiv$ helicity band)",
        r"$r=0.30$ (helicity band basis)",
        r"$r=0.30$ difference (band $-$ spin)",
    )
    for row, metric_idx, label, cmap, bound, vlim, diff_lbl in specs:
        dvals = diff_arr[:, :, metric_idx]
        dlim = max(float(np.max(np.abs(dvals))), 1e-6)
        cols_data = (
            (scan_r0[:, :, metric_idx], cmap, vlim[0], vlim[1], bound, label),
            (scan_r030_band[:, :, metric_idx], cmap, vlim[0], vlim[1], bound, label),
            (dvals, "coolwarm", -dlim, dlim, None, diff_lbl),
        )
        for col, (values, cm, lo, hi, bnd, clbl) in enumerate(cols_data):
            ax = axes[row, col]
            mesh = ax.pcolormesh(tau, SCAN_GRID, values, shading="nearest",
                                 cmap=cm, vmin=lo, vmax=hi)
            if bnd is not None and float(values.max()) >= bnd and float(values.min()) <= bnd:
                ax.contour(tau, SCAN_GRID, values, levels=[bnd],
                           colors="white", linewidths=1.3)
            ax.axhline(2.0 * SCAN_GAMMA, color=("black" if col == 2 else "white"),
                       lw=1.0, ls="--")
            if col < 2 and row == 0:
                ax.text(tau[-1] * 0.985, 2.0 * SCAN_GAMMA * 1.15,
                        r"$\Gamma=2\gamma$", color="white", fontsize=8.5, ha="right")
            ax.set_yscale("log")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(fr"{label}" + "\n" + r"$\Gamma$ (spectral width)")
            fig.colorbar(mesh, ax=ax, pad=0.02, label=clbl)
    for col in range(3):
        axes[-1, col].set_xlabel(r"scaled time $\tau=\gamma_0 t$")
    fig.suptitle("Reservoir spectral-width scan at $\\gamma=2$ "
                 "(spin vs. helicity band at $r=0$ and $r=0.30$); "
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

    print("Stage 2/4: resource states (spin + helicity band, V_SO/t=0.5)",
          flush=True)
    from helicity_density_matrix import density_at_separation
    states_spin, states_band = [], []
    init_spin, init_band = [], []
    rho_init_spin, rho_init_band = [], []
    th = np.deg2rad(THETA_DEG[0])
    for radius in R_POINTS:
        r_vec = np.array([radius * np.cos(th), radius * np.sin(th)])
        rs = density_at_separation(bdg, r_vec, basis="spin")["rho"]
        rb = density_at_separation(bdg, r_vec, basis="band")["rho"]
        states_spin.append(np.asarray(rs))
        states_band.append(np.asarray(rb))
        rho_init_spin.append(rs)
        rho_init_band.append(rb)
        init_spin.append(dict(zip(METRIC_KEYS, metric_row(rs))))
        init_band.append(dict(zip(METRIC_KEYS, metric_row(rb))))
        print(f"  r={radius:.2f} [spin]: " + ", ".join(
            f"{k}={v:.4f}" for k, v in init_spin[-1].items()), flush=True)
        print(f"  r={radius:.2f} [band]: " + ", ".join(
            f"{k}={v:.4f}" for k, v in init_band[-1].items()), flush=True)
    for rho in rho_init_spin + rho_init_band:
        if not verify_density_matrices([[rho]], atol=2e-8):
            raise RuntimeError("density-matrix validation failed")

    print("Stage 3/4: regime dynamics + cross-implementation check", flush=True)
    cross_error = 0.0
    vec_vs_scalar_error = 0.0
    series_spin = {regime: [] for regime in REGIME_ORDER}
    series_band = {regime: [] for regime in REGIME_ORDER}
    for regime in REGIME_ORDER:
        pars = REGIMES[regime]
        for ip in range(len(R_POINTS)):
            series_spin[regime].append(
                evolve_metrics(states_spin[ip], TAU, pars["gamma"], pars["Gamma"]))
            series_band[regime].append(
                evolve_metrics(states_band[ip], TAU, pars["gamma"], pars["Gamma"]))
            print(f"  {regime}, r={R_POINTS[ip]:.2f}", flush=True)
        # evolve_rho (noise_channels) must equal fidelity.py's eta path & vectorized batch
        for ip in (0, 1, 2):
            for it_idx in (0, 100, 500):
                time = float(TAU[it_idx])
                a = evolve_rho(states_spin[ip], time, kind="amp",
                               bath="independent",
                               gammaA=pars["gamma"], GammaA=pars["Gamma"],
                               gammaB=pars["gamma"], GammaB=pars["Gamma"])
                b = _evolve_rho_fid(states_spin[ip], time, kind="amp",
                                    bath="independent",
                                    nuA=0.2, GammaA=0.05, nuB=0.2,
                                    GammaB=0.05,
                                    gammaA=pars["gamma"],
                                    GammaAD_A=pars["Gamma"],
                                    gammaB=pars["gamma"],
                                    GammaAD_B=pars["Gamma"])
                cross_error = max(cross_error, maxabs(a - b))
                vec_vs_scalar_error = max(
                    vec_vs_scalar_error,
                    maxabs(series_spin[regime][ip][it_idx] - np.array(metric_row(a))))
    if cross_error > 2e-12 or vec_vs_scalar_error > 2e-12:
        raise RuntimeError(f"channel implementations disagree: {cross_error}, {vec_vs_scalar_error}")
    print(f"  cross-implementation max|diff| = {cross_error:.2e}, "
          f"vec vs scalar = {vec_vs_scalar_error:.2e}", flush=True)

    print("Stage 4/4: spectral-width scan (r=0 and r=0.30 spin/band), figures, outputs", flush=True)
    scan_r0 = np.zeros((len(SCAN_GRID), len(TAU), len(METRIC_KEYS)))
    scan_r030_spin = np.zeros_like(scan_r0)
    scan_r030_band = np.zeros_like(scan_r0)
    for ig, width in enumerate(SCAN_GRID):
        w = float(width)
        scan_r0[ig] = evolve_metrics(states_spin[0], TAU, SCAN_GAMMA, w)
        scan_r030_spin[ig] = evolve_metrics(states_spin[1], TAU, SCAN_GAMMA, w)
        scan_r030_band[ig] = evolve_metrics(states_band[1], TAU, SCAN_GAMMA, w)
        if ig % 15 == 0:
            print(f"  scan Gamma index {ig+1}/{len(SCAN_GRID)}", flush=True)

    # --- validation ----------------------------------------------------------
    all_series = (
        [s for regime in REGIME_ORDER for s in series_spin[regime]]
        + [s for regime in REGIME_ORDER for s in series_band[regime]]
        + [scan_r0.reshape(-1, len(METRIC_KEYS)),
           scan_r030_spin.reshape(-1, len(METRIC_KEYS)),
           scan_r030_band.reshape(-1, len(METRIC_KEYS))]
    )
    stack = np.concatenate(all_series, axis=0)
    validation = {
        "cross_implementation_max_abs_error": float(cross_error),
        "vectorized_vs_scalar_max_abs_error": float(vec_vs_scalar_error),
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
    plot_regimes(TAU, series_spin, series_band,
                 FIGS / "environment_protocols_regimes.png")
    plot_scan(TAU, scan_r0, scan_r030_spin, scan_r030_band,
              FIGS / "environment_parameter_scan.png")

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
            [np.stack(series_spin[r]) for r in REGIME_ORDER]),
        metrics_regimes_band=np.stack(
            [np.stack(series_band[r]) for r in REGIME_ORDER]),
        scan_gamma_grid=SCAN_GRID,
        scan_gamma=SCAN_GAMMA,
        metrics_scan=scan_r0,
        metrics_scan_r030_spin=scan_r030_spin,
        metrics_scan_r030_band=scan_r030_band,
        rho_initial=np.array(rho_init_spin),
        rho_initial_band=np.array(rho_init_band),
        initial_metrics=np.array([list(m.values()) for m in init_spin]),
        initial_metrics_band=np.array([list(m.values()) for m in init_band]),
    )

    # --- report --------------------------------------------------------------
    def build_dyn(ser):
        out = {}
        for regime in REGIME_ORDER:
            out[regime] = {}
            for ip, radius in enumerate(R_POINTS):
                data = ser[regime][ip]
                out[regime][f"{radius:.2f}"] = {
                    "fixed_fidelity": threshold_summary(TAU, data[:, 0], CLASSICAL_F),
                    "optimized_fidelity": threshold_summary(TAU, data[:, 1], CLASSICAL_F),
                    "holevo": threshold_summary(TAU, data[:, 2], CLASSICAL_CHI),
                    "concurrence": threshold_summary(TAU, data[:, 3], None),
                }
        return out

    dynamics = build_dyn(series_spin)
    dynamics_band = build_dyn(series_band)
    basis_diffs = {}
    for regime in REGIME_ORDER:
        basis_diffs[regime] = {}
        for ip, radius in enumerate(R_POINTS):
            d = series_band[regime][ip] - series_spin[regime][ip]
            basis_diffs[regime][f"{radius:.2f}"] = {
                "max_abs_fixed_fidelity_diff": float(np.max(np.abs(d[:, 0]))),
                "max_abs_optimized_fidelity_diff": float(np.max(np.abs(d[:, 1]))),
                "max_abs_holevo_diff": float(np.max(np.abs(d[:, 2]))),
                "max_abs_concurrence_diff": float(np.max(np.abs(d[:, 3]))),
            }

    scan_selected = {}
    for target in (0.01, 0.1, 1.0, 4.0, 10.0, 20.0):
        ig = int(np.argmin(np.abs(SCAN_GRID - target)))
        row = scan_r0[ig]
        row_s = scan_r030_spin[ig]
        row_b = scan_r030_band[ig]
        scan_selected[f"{SCAN_GRID[ig]:g}"] = {
            "underdamped": bool(2.0 * SCAN_GAMMA * SCAN_GRID[ig]
                                - SCAN_GRID[ig] ** 2 > 0),
            "fixed_fidelity_final": float(row[-1, 0]),
            "holevo_final": float(row[-1, 2]),
            "concurrence_final": float(row[-1, 3]),
            "holevo_minimum": float(row[:, 2].min()),
            "r030_spin_final": [float(row_s[-1, 0]), float(row_s[-1, 2]), float(row_s[-1, 3])],
            "r030_band_final": [float(row_b[-1, 0]), float(row_b[-1, 2]), float(row_b[-1, 3])],
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
                  "state_construction": "helicity conditional Wick, spin + band bases",
                  "Nk": NK, "theta_deg": list(THETA_DEG),
                  "Delta_t_over_t": float(config.Delta_t),
                  "n_elec": float(config.n_elec)},
        "resources": {f"{r:.2f}": init
                      for r, init in zip(R_POINTS, init_spin)},
        "resources_band": {f"{r:.2f}": init
                           for r, init in zip(R_POINTS, init_band)},
        "dynamics": dynamics,
        "dynamics_band": dynamics_band,
        "band_minus_spin_max_abs": basis_diffs,
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
                      "basis_diffs_nm": basis_diffs["non_markovian"]}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
