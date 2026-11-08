#!/usr/bin/env python3
"""Microscopic, noise, and convergence study for BdG-resource teleportation.

The existing BdG solver, equal-time Green's functions, Oh--Kim density matrix,
Kraus channels, and explicit teleportation implementation are reused unchanged.
The Markovian comparison uses the exact overdamped continuation already present
in fidelity.py and the analytical exponential limit of the same Kraus channel.

Run from the repository root:
    .venv/bin/python analysis/run_microscopic_teleportation_study.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
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
from fidelity import _evolve_rho_fid, eta_amp_lorentz, kraus_amp_from_eta
from greens_functions import compute_greens_functions
from noise_channels import p_amplitude_damping, p_dephasing
from observables import concurrence_xstate
from self_consistency import run_selfconsistency
from teleportation import CLASSICAL_BENCHMARK, resource_metrics, singlet_state

DATA = ROOT / "data"
FIGS = ROOT / "figs"
REPORTS = ROOT / "reports"
DATA.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

U_VALUES = np.arange(40.0, 115.0 + 0.1, 5.0)
U_FULL = (40.0, 75.0, 115.0)
NK_VALUES = (100, 200, 400)
ANGLES_FULL = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
SITES = ((1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0))
NOISE = {
    "low": dict(gammaA=1.5, GammaA=0.1, gammaB=1.5, GammaB=0.1),
    "high": dict(gammaA=4.5, GammaA=0.01, gammaB=4.5, GammaB=0.01),
}
METRICS = ("concurrence", "singlet_fraction", "fixed_average_fidelity",
           "fully_entangled_fraction", "optimized_average_fidelity")
P_THRESHOLD = 1.0 / 3.0


def maxabs(value):
    return float(np.max(np.abs(value)))


@contextmanager
def temporary_spatial_grid(r_max, n_r, angles):
    """Use the existing GF implementation on a temporary analysis grid."""
    old = (gf_module.R_MIN, gf_module.R_MAX, gf_module.N_R,
           gf_module.THETA_DEG)
    gf_module.R_MIN = 0.0
    gf_module.R_MAX = float(r_max)
    gf_module.N_R = int(n_r)
    gf_module.THETA_DEG = list(angles)
    try:
        yield
    finally:
        (gf_module.R_MIN, gf_module.R_MAX, gf_module.N_R,
         gf_module.THETA_DEG) = old


def solve_bdg(U, nk, allow_trivial=False):
    """Solve at a requested momentum resolution without editing the model.

    A small residual alone is insufficient: the solver can return the trivial
    zero-gap fixed point. Such a state is accepted only when explicitly
    requested as an unpaired control and is always labeled in the outputs.
    """
    old_sc_nk = sc_module.Nk
    old_k_nk = kspace_setup.Nk
    sc_module.Nk = int(nk)
    kspace_setup.Nk = int(nk)
    try:
        bdg = run_selfconsistency(seed=42, U_override=float(U),
                                  kBT_override=0.0, verbose=False)
    finally:
        sc_module.Nk = old_sc_nk
        kspace_setup.Nk = old_k_nk
    residual = maxabs(bdg["residuals"])
    paired = max(abs(float(bdg["Delta0"])), abs(float(bdg["DeltaS"]))) > 1e-5
    if residual > 1e-8 or (not paired and not allow_trivial) or (paired and not bdg["converged"]):
        raise RuntimeError(f"U={U}, Nk={nk}: residual={residual}, "
                           f"converged={bdg['converged']}, paired={paired}")
    return bdg, residual


def greens_on_grid(bdg, r_max, n_r, angles=(45.0,)):
    with temporary_spatial_grid(r_max, n_r, angles):
        return compute_greens_functions(bdg, verbose=False,
                                        convention="equal_time")


def normalized_correlators(G, F):
    G0 = np.abs(G[:, [0]])
    x = (np.abs(G) / G0) ** 2       # g^2
    y = (np.abs(F) / G0) ** 2       # f^2
    p = (x + y) / (2.0 + y - x)
    return x, y, p


def werner_matrix(p):
    singlet = np.outer(singlet_state(), singlet_state().conj())
    return (1.0 - p) * np.eye(4) / 4.0 + p * singlet


def critical_radius(r, p):
    """First continuous p=1/3 down-crossing, linearly interpolated."""
    above = p > P_THRESHOLD
    for i in range(len(r) - 1):
        if above[i] and not above[i + 1]:
            fraction = (P_THRESHOLD - p[i]) / (p[i + 1] - p[i])
            return float(r[i] + fraction * (r[i + 1] - r[i])), i
    return None, None


def threshold_events(t, values):
    """Threshold losses and subsequent upward re-crossings."""
    values = np.asarray(values)
    above = values > CLASSICAL_BENCHMARK
    losses, regains = [], []
    for i in range(len(t) - 1):
        if above[i] == above[i + 1]:
            continue
        denominator = values[i + 1] - values[i]
        fraction = 0.0 if denominator == 0 else (
            (CLASSICAL_BENCHMARK - values[i]) / denominator)
        crossing = float(t[i] + fraction * (t[i + 1] - t[i]))
        if above[i] and not above[i + 1]:
            losses.append(crossing)
        else:
            regains.append(crossing)
    first_loss = losses[0] if losses else None
    return {
        "initial_above": bool(above[0]),
        "ever_above": bool(np.any(above)),
        "loss_times": losses,
        "regain_times_after_a_loss": regains,
        "first_loss_time": first_loss,
        "first_regain_time": regains[0] if regains else None,
        "maximum_after_first_loss": (
            float(np.max(values[np.asarray(t) >= first_loss]))
            if first_loss is not None else None),
        "maximum": float(np.max(values)),
        "final": float(values[-1]),
    }


def state_metrics(rho):
    result = resource_metrics(rho, validate=False)
    return np.array([result[name] for name in METRICS], dtype=float)


def evolve_metrics(rho0, t_grid, noise, kind="amp", bath="independent"):
    output = np.zeros((len(t_grid), len(METRICS)))
    for i, time in enumerate(t_grid):
        rho_t = evolve_rho(rho0, float(time), kind=kind, bath=bath, **noise)
        output[i] = state_metrics(rho_t)
    return output


def evolve_metrics_overdamped(rho0, t_grid, gamma, Gamma):
    """Use the existing exact hyperbolic continuation in fidelity.py."""
    output = np.zeros((len(t_grid), len(METRICS)))
    for i, time in enumerate(t_grid):
        rho_t = _evolve_rho_fid(
            rho0, float(time), kind="amp", bath="independent",
            nuA=0.2, GammaA=Gamma, nuB=0.2, GammaB=Gamma,
            gammaA=gamma, GammaAD_A=Gamma,
            gammaB=gamma, GammaAD_B=Gamma)
        output[i] = state_metrics(rho_t)
    return output


def evolve_metrics_exponential(rho0, t_grid, gamma):
    """Analytical Markovian limit eta(t)=exp(-gamma*t), same Kraus map."""
    output = np.zeros((len(t_grid), len(METRICS)))
    for i, time in enumerate(t_grid):
        eta = float(np.exp(-gamma * time))
        Ka = kraus_amp_from_eta(eta)
        Kb = kraus_amp_from_eta(eta)
        rho_t = np.zeros((4, 4), complex)
        for A in Ka:
            for B in Kb:
                K = np.kron(A, B)
                rho_t += K @ rho0 @ K.conj().T
        output[i] = state_metrics(rho_t)
    return output


def plot_initial_study(r_line, U, p_line, x_line, y_line, rcrit,
                       site_p, site_r, bdg_records):
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(U)))
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True)
    for i, interaction in enumerate(U):
        if interaction in U_FULL:
            axes[0, 0].plot(r_line, (1 + p_line[i]) / 2, color=colors[i],
                            lw=2, label=fr"$U/t={interaction:.0f}$")
    axes[0, 0].axhline(CLASSICAL_BENCHMARK, color="black", ls="--",
                       label=r"classical $2/3$")
    axes[0, 0].set(xlabel=r"$r/a$", ylabel=r"Initial $\overline{\mathcal{F}}_{tel}$",
                   xlim=(0, 1.5), ylim=(0.48, 1.01))
    axes[0, 0].legend()

    image = axes[0, 1].pcolormesh(r_line, U, (1 + p_line) / 2,
                                  shading="auto", vmin=0.5, vmax=1.0,
                                  cmap="viridis")
    axes[0, 1].plot(rcrit, U, "w--", lw=1.5, label=r"$p=1/3$")
    axes[0, 1].set(xlabel=r"$r/a$", ylabel=r"$U/t$", xlim=(0, 1.0))
    axes[0, 1].legend(loc="upper right")
    fig.colorbar(image, ax=axes[0, 1], label=r"$(1+p)/2$")

    axes[0, 2].plot(U, rcrit, "o-")
    axes[0, 2].set(xlabel=r"$U/t$", ylabel=r"First critical $r_c/a$")
    axes[0, 2].set_title("Continuous-distance initial threshold")

    for i, interaction in enumerate(U):
        if interaction in U_FULL:
            axes[1, 0].plot(r_line, x_line[i], color=colors[i], lw=2,
                            label=fr"$g^2$, $U/t={interaction:.0f}$")
            axes[1, 0].plot(r_line, y_line[i], color=colors[i], ls="--", lw=1.5,
                            label=fr"$f^2$, $U/t={interaction:.0f}$")
    axes[1, 0].set(xlabel=r"$r/a$", ylabel="Squared normalized correlator",
                   xlim=(0, 1.0), yscale="log", ylim=(1e-6, 1.2))
    axes[1, 0].legend(fontsize=8, ncol=2)

    for i, interaction in enumerate(U):
        if interaction in U_FULL:
            axes[1, 1].plot(site_r, site_p[i], "o-", color=colors[i],
                            label=fr"$U/t={interaction:.0f}$")
    axes[1, 1].axhline(P_THRESHOLD, color="black", ls="--", label=r"$p=1/3$")
    axes[1, 1].set(xlabel=r"Exact lattice-site separation $r/a$",
                   ylabel=r"Werner $p$", yscale="log")
    axes[1, 1].legend()

    axes[1, 2].plot(U, np.abs(bdg_records["Delta0"]), "o-", label=r"$|\Delta_0|/t$")
    axes[1, 2].plot(U, np.abs(bdg_records["DeltaS"]), "s-", label=r"$|\Delta_S|/t$")
    axes[1, 2].set(xlabel=r"$U/t$", ylabel="Self-consistent gap magnitude")
    axes[1, 2].legend()
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.suptitle(r"Corrected equal-time resource at $\theta=45^\circ$; "
                 r"$V_{SO}=0$, $n=1.875$, $N_k=100$")
    fig.savefig(FIGS / "microscopic_initial_U_r_map.png", dpi=210,
                bbox_inches="tight")
    plt.close(fig)


def plot_gf_decomposition(r, x, y, p, rcrit):
    p_normal = x / (2.0 - x)
    p_anomalous = y / (2.0 + y)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), constrained_layout=True)
    axes[0].plot(r, p, label=r"full $p=(g^2+f^2)/(2+f^2-g^2)$")
    axes[0].plot(r, p_normal, "--", label=r"$f=0$ diagnostic: $g^2/(2-g^2)$")
    axes[0].plot(r, p_anomalous, ":", label=r"$g=0$ diagnostic: $f^2/(2+f^2)$")
    axes[0].axhline(P_THRESHOLD, color="black", ls="--")
    axes[0].axvline(rcrit, color="gray", ls=":")
    axes[0].set(xlabel=r"$r/a$", ylabel=r"$p$", xlim=(0, 1.0), ylim=(0, 1.02))
    axes[0].legend(fontsize=8)
    axes[1].plot(r, 2 * x + y)
    axes[1].axhline(1.0, color="black", ls="--", label=r"$2g^2+f^2=1$")
    axes[1].axvline(rcrit, color="gray", ls=":")
    axes[1].set(xlabel=r"$r/a$", ylabel=r"$2g^2+f^2$", xlim=(0, 1.0))
    axes[1].legend()
    axes[2].plot(r, p - p_normal, label=r"$p-p(f=0)$")
    axes[2].set(xlabel=r"$r/a$", ylabel="Direct anomalous contribution",
                xlim=(0, 1.0))
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(r"Normal versus anomalous correlators at $U/t=75$, $\theta=45^\circ$")
    fig.savefig(FIGS / "microscopic_gf_decomposition.png", dpi=210,
                bbox_inches="tight")
    plt.close(fig)


def plot_convergence(r_conv, p_conv, rcrit_conv):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    styles = {100: ":", 200: "--", 400: "-"}
    for iu, interaction in enumerate(U_FULL):
        ax = axes[iu]
        for ik, nk in enumerate(NK_VALUES):
            label = fr"$N_k={nk}$"
            if interaction == 115 and nk == 400:
                label += " (zero-gap control)"
            ax.plot(r_conv, p_conv[iu, ik], styles[nk], lw=1.8,
                    label=label)
        ax.axhline(P_THRESHOLD, color="black", ls="--")
        ax.set(xlabel=r"$r/a$", ylabel=r"$p$", xlim=(0.25, 0.65),
               ylim=(0.15, 0.7), title=fr"$U/t={interaction:.0f}$")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Momentum-grid convergence near the initial teleportation threshold")
    fig.savefig(FIGS / "microscopic_teleportation_convergence.png", dpi=210,
                bbox_inches="tight")
    plt.close(fig)


def plot_dynamics(t, dynamics, point_r, t_map, r_map, map_metrics,
                  t_u, u_dynamics):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True)
    selected = [0, 1, 2, 4, 5]
    colors = plt.get_cmap("tab10").colors
    for col, noise in enumerate(("low", "high")):
        ax = axes[0, col]
        for ip in selected:
            ax.plot(t, dynamics[noise][ip, :, 2], color=colors[ip],
                    label=fr"$r={point_r[ip]:.2f}$ fixed")
            ax.plot(t, dynamics[noise][ip, :, 4], "--", color=colors[ip],
                    label=fr"$r={point_r[ip]:.2f}$ optimized")
        ax.axhline(CLASSICAL_BENCHMARK, color="black", ls=":")
        ax.set(xlabel=r"$t\,t/\hbar$", ylabel="Average teleportation fidelity",
               xlim=(0, 80), ylim=(0.30, 1.01), title=f"{noise} amplitude damping")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)

    for iu, interaction in enumerate(U_FULL):
        label = fr"$U/t={interaction:.0f}$"
        if interaction == 115:
            label += " (zero gap)"
        axes[0, 2].plot(t_u, u_dynamics[iu, :, 2], color=colors[iu],
                        label=label + " fixed")
        axes[0, 2].plot(t_u, u_dynamics[iu, :, 4], "--", color=colors[iu],
                        label=label + " optimized")
    axes[0, 2].axhline(CLASSICAL_BENCHMARK, color="black", ls=":")
    axes[0, 2].set(xlabel=r"$t\,t/\hbar$", xlim=(0, 80), ylim=(0.30, 0.76),
                   title=r"Interaction dependence at $r=0.40$, high noise")
    axes[0, 2].grid(alpha=0.25)
    axes[0, 2].legend(fontsize=7, ncol=2)

    for col, (metric_index, title) in enumerate(
            ((2, "Fixed"), (4, "Optimized")), start=0):
        ax = axes[1, col]
        advantage = map_metrics[:, :, metric_index] > CLASSICAL_BENCHMARK
        image = ax.pcolormesh(t_map, r_map, map_metrics[:, :, metric_index],
                              shading="auto", vmin=0.33, vmax=1.0, cmap="viridis")
        ax.contour(t_map, r_map, advantage.astype(float), levels=[0.5],
                   colors=["red"], linewidths=[1.5])
        ax.set(xlabel=r"$t\,t/\hbar$", ylabel=r"$r/a$",
               title=fr"{title} fidelity; red contour $=2/3$")
        fig.colorbar(image, ax=ax, label="Average fidelity")
    axes[1, 2].plot(t_map, [p_amplitude_damping(time, 4.5, 0.01)
                            for time in t_map],
                    label=r"survival $P(t)$")
    axes[1, 2].set(xlabel=r"$t\,t/\hbar$", ylabel=r"$P(t)$",
                   title="High-regime environmental envelope")
    axes[1, 2].grid(alpha=0.25)
    axes[1, 2].legend()
    fig.suptitle(r"Amplitude-damping revivals at $U/t=75$, $\theta=45^\circ$, $N_k=400$")
    fig.savefig(FIGS / "microscopic_teleportation_revival_maps.png", dpi=210,
                bbox_inches="tight")
    plt.close(fig)


def plot_markovian(t_nm, nm, t_m, markovian, exponential):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    labels = ("r=0", "r=0.40")
    for ip, label in enumerate(labels):
        axes[0].plot(t_nm, nm[ip, :, 2], label=label + " fixed")
        axes[0].plot(t_nm, nm[ip, :, 4], "--", label=label + " optimized")
    axes[0].axhline(CLASSICAL_BENCHMARK, color="black", ls=":")
    axes[0].set(xlabel=r"$t\,t/\hbar$", ylabel="Average fidelity",
                xlim=(0, 80), title=r"Non-Markovian: $\gamma=4.5$, $\Gamma=0.01$")
    axes[0].legend(fontsize=8)
    for ip, label in enumerate(labels):
        axes[1].plot(t_m, markovian[ip, :, 4], label=label + r", $\Gamma/\gamma=100$")
        axes[1].plot(t_m, exponential[ip, :, 4], "--",
                     label=label + ", exponential limit")
    axes[1].axhline(CLASSICAL_BENCHMARK, color="black", ls=":")
    axes[1].set(xlabel=r"$t\,t/\hbar$", xlim=(0, 1.5), ylim=(0.32, 1.01),
                title=r"Markovian comparison: $\gamma=4.5$")
    axes[1].legend(fontsize=8)
    for ip, label in enumerate(labels):
        axes[2].plot(t_m, markovian[ip, :, 2], label=label + " fixed")
        axes[2].plot(t_m, exponential[ip, :, 2], "--", label=label + " fixed, exponential")
    axes[2].axhline(CLASSICAL_BENCHMARK, color="black", ls=":")
    axes[2].set(xlabel=r"$t\,t/\hbar$", xlim=(0, 1.5), ylim=(0.32, 1.01),
                title="No revival in the memoryless limit")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(FIGS / "microscopic_markovian_comparison.png", dpi=210,
                bbox_inches="tight")
    plt.close(fig)


def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    print("Stage 1/6: interaction-resolved initial-state maps", flush=True)
    bdg100 = {}
    solver_records = []
    for U in U_VALUES:
        bdg, residual = solve_bdg(U, 100)
        bdg100[U] = bdg
        solver_records.append({
            "U_over_t": float(U), "Nk": 100, "max_abs_residual": residual,
            "Delta0_over_t": float(bdg["Delta0"]),
            "DeltaS_over_t": float(bdg["DeltaS"]),
            "mu_over_t": float(bdg["mu"]),
            "max_abs_Delta_ks_over_t": float(np.max(np.abs(bdg["Delta_ks"]))),
            "mean_abs_Delta_ks_over_t": float(np.mean(np.abs(bdg["Delta_ks"]))),
            "paired_solution": bool(max(abs(float(bdg["Delta0"])),
                                        abs(float(bdg["DeltaS"]))) > 1e-5),
            "solver_converged_flag": bool(bdg["converged"]),
        })
        print(f"  U={U:.0f}, residual={residual:.2e}", flush=True)

    r_line = theta_line = None
    x_line = np.zeros((len(U_VALUES), 301))
    y_line = np.zeros_like(x_line)
    p_line = np.zeros_like(x_line)
    G_line = np.zeros_like(x_line, dtype=complex)
    F_line = np.zeros_like(x_line, dtype=complex)
    for iu, U in enumerate(U_VALUES):
        r_line, theta_line, G, F = greens_on_grid(bdg100[U], 1.5, 301)
        G_line[iu], F_line[iu] = G[0], F[0]
        x_line[iu], y_line[iu], p_line[iu] = normalized_correlators(G, F)
    rcrit = np.array([critical_radius(r_line, p)[0] for p in p_line])

    # Full angular/radial maps for the three previously converged interactions.
    full = {name: np.zeros((len(U_FULL), len(ANGLES_FULL), 200))
            for name in ("x", "y", "p")}
    full_G = np.zeros((len(U_FULL), len(ANGLES_FULL), 200), complex)
    full_F = np.zeros_like(full_G)
    werner_identity_max_error = 0.0
    for iu, U in enumerate(U_FULL):
        r_full, theta_full, G, F = greens_on_grid(
            bdg100[U], 10.0, 200, ANGLES_FULL)
        full_G[iu], full_F[iu] = G, F
        full["x"][iu], full["y"][iu], full["p"][iu] = normalized_correlators(G, F)
        rho = np.asarray(build_density_matrices(G, F))
        if not verify_density_matrices(rho, atol=2e-8):
            raise RuntimeError(f"Density validation failed at U={U}")
        werner = np.zeros_like(rho)
        for it in range(len(theta_full)):
            for ir in range(len(r_full)):
                werner[it, ir] = werner_matrix(full["p"][iu, it, ir])
        error = maxabs(rho - werner)
        werner_identity_max_error = max(werner_identity_max_error, error)
        if error > 2e-12:
            raise RuntimeError(f"Werner identity failed at U={U}: {error}")

    # Exact nonzero lattice-site separations, not nearest radial-grid points.
    site_r = np.array([np.hypot(x, y) for x, y in SITES])
    site_theta = np.array([np.degrees(np.arctan2(y, x)) for x, y in SITES])
    site_x = np.zeros((len(U_VALUES), len(SITES)))
    site_y = np.zeros_like(site_x)
    site_p = np.zeros_like(site_x)
    for iu, U in enumerate(U_VALUES):
        for isite, (radius, angle) in enumerate(zip(site_r, site_theta)):
            _, _, G, F = greens_on_grid(bdg100[U], radius, 2, (angle,))
            x, y, p = normalized_correlators(G, F)
            site_x[iu, isite] = x[0, 1]
            site_y[iu, isite] = y[0, 1]
            site_p[iu, isite] = p[0, 1]

    # Validate the analytic initial Werner formulas against the full protocol.
    initial_protocol_errors = {"fixed": 0.0, "optimized": 0.0,
                               "concurrence": 0.0,
                               "werner_matrix": werner_identity_max_error}
    validation_states = []
    for iu in range(len(U_VALUES)):
        validation_states.append((p_line[iu, 100], G_line[iu, 100],
                                  F_line[iu, 100], abs(G_line[iu, 0])))
    for iu in (0, 7, 15):
        for isite in range(len(SITES)):
            validation_states.append((site_p[iu, isite], None, None, None))
    for p, _, _, _ in validation_states:
        rho = werner_matrix(p)
        metrics = resource_metrics(rho, validate=True)
        initial_protocol_errors["fixed"] = max(
            initial_protocol_errors["fixed"],
            abs(metrics["fixed_average_fidelity"] - (1 + p) / 2))
        initial_protocol_errors["optimized"] = max(
            initial_protocol_errors["optimized"],
            abs(metrics["optimized_average_fidelity"] - (1 + p) / 2))
        initial_protocol_errors["concurrence"] = max(
            initial_protocol_errors["concurrence"],
            abs(metrics["concurrence"] - max(0.0, (3 * p - 1) / 2)))

    print("Stage 2/6: momentum-grid convergence", flush=True)
    r_conv = np.linspace(0.0, 0.8, 321)
    p_conv = np.zeros((len(U_FULL), len(NK_VALUES), len(r_conv)))
    x_conv = np.zeros_like(p_conv)
    y_conv = np.zeros_like(p_conv)
    convergence_records = []
    bdg400 = {}
    for iu, U in enumerate(U_FULL):
        for ik, nk in enumerate(NK_VALUES):
            bdg, residual = solve_bdg(U, nk, allow_trivial=True)
            r_tmp, _, G, F = greens_on_grid(bdg, 0.8, 321)
            if not np.array_equal(r_tmp, r_conv):
                raise RuntimeError("Unexpected convergence grid")
            x, y, p = normalized_correlators(G, F)
            p_conv[iu, ik], x_conv[iu, ik], y_conv[iu, ik] = p[0], x[0], y[0]
            rc, _ = critical_radius(r_conv, p[0])
            G0_discrete = 0.5 * np.sum(bdg["v_ks"] ** 2) / nk ** 2
            record = {
                "U_over_t": float(U), "Nk": nk, "max_abs_residual": residual,
                "critical_r_over_a": rc,
                "G0_simpson_abs": float(abs(G[0, 0])),
                "G0_occupancy_target_n_over_2": config.n_elec / 2,
                "G0_simpson_relative_error": float(
                    abs(abs(G[0, 0]) - config.n_elec / 2) / (config.n_elec / 2)),
                "G0_discrete": float(G0_discrete),
                "G0_discrete_abs_error": float(abs(G0_discrete - config.n_elec / 2)),
                "Delta0_over_t": float(bdg["Delta0"]),
                "DeltaS_over_t": float(bdg["DeltaS"]),
                "paired_solution": bool(max(abs(float(bdg["Delta0"])),
                                            abs(float(bdg["DeltaS"]))) > 1e-5),
                "solver_converged_flag": bool(bdg["converged"]),
            }
            convergence_records.append(record)
            print(f"  U={U:.0f}, Nk={nk}, rc={rc:.8f}, "
                  f"paired={record['paired_solution']}", flush=True)
            if nk == 400:
                bdg400[U] = bdg

    # Radial interpolation check at fixed Nk=200 and U=75.
    radial_records = []
    bdg200_u75, _ = solve_bdg(75, 200)
    for nr in (161, 321, 641):
        r_rad, _, G, F = greens_on_grid(bdg200_u75, 0.8, nr)
        _, _, p_rad = normalized_correlators(G, F)
        rc, _ = critical_radius(r_rad, p_rad[0])
        radial_records.append({"Nk": 200, "Nr": nr, "dr": float(r_rad[1]-r_rad[0]),
                               "critical_r_over_a": rc})

    print("Stage 3/6: production resources and amplitude-damping dynamics", flush=True)
    point_r = np.array([0.0, 0.30, 0.40, 0.45, 0.50, 1.0])
    point_rho = []
    point_micro = []
    for radius in point_r:
        if radius == 0:
            rho = werner_matrix(1.0)
            x0, y0, p0 = 1.0, None, 1.0
        else:
            _, _, G, F = greens_on_grid(bdg400[75.0], float(radius), 2, (45.0,))
            x, y, p = normalized_correlators(G, F)
            rho = np.asarray(build_density_matrices(G, F))[0, 1]
            x0, y0, p0 = float(x[0, 1]), float(y[0, 1]), float(p[0, 1])
        point_rho.append(rho)
        point_micro.append({"r_over_a": float(radius), "g2": x0, "f2": y0,
                            "p_initial": p0,
                            "fixed_initial": (1 + p0) / 2,
                            "optimized_initial": (1 + p0) / 2})
    point_rho = np.asarray(point_rho)
    t = np.linspace(0.0, 80.0, 1601)
    dynamics = {noise: np.zeros((len(point_r), len(t), len(METRICS)))
                for noise in NOISE}
    for noise, parameters in NOISE.items():
        for ip, rho in enumerate(point_rho):
            dynamics[noise][ip] = evolve_metrics(rho, t, parameters)
            print(f"  {noise}, r={point_r[ip]:.2f}", flush=True)

    # Dephasing controls at r=0 and r=0.40, independent and common baths.
    dephasing = {bath: np.zeros((2, len(t), len(METRICS)))
                 for bath in ("independent", "common")}
    for bath in dephasing:
        for ip, point_index in enumerate((0, 2)):
            dephasing[bath][ip] = evolve_metrics(
                point_rho[point_index], t, NOISE["low"],
                kind="dephasing", bath=bath)

    # Interaction dependence of the high-noise revival at r=0.40.
    t_u = np.linspace(0.0, 80.0, 1001)
    u_dynamics = np.zeros((len(U_FULL), len(t_u), len(METRICS)))
    for iu, U in enumerate(U_FULL):
        _, _, G, F = greens_on_grid(bdg400[U], 0.40, 2, (45.0,))
        rho = np.asarray(build_density_matrices(G, F))[0, 1]
        u_dynamics[iu] = evolve_metrics(rho, t_u, NOISE["high"])

    print("Stage 4/6: radial-time advantage map", flush=True)
    r_map = np.linspace(0.0, 0.8, 81)
    t_map = np.linspace(0.0, 63.0, 421)
    _, _, G_map, F_map = greens_on_grid(bdg400[75.0], 0.8, 81, (45.0,))
    rho_map = np.asarray(build_density_matrices(G_map, F_map))[0]
    map_metrics = np.zeros((len(r_map), len(t_map), len(METRICS)))
    for ir, rho in enumerate(rho_map):
        map_metrics[ir] = evolve_metrics(rho, t_map, NOISE["high"])
        if ir % 20 == 0:
            print(f"  radial-time map row {ir+1}/81", flush=True)

    print("Stage 5/6: exact overdamped/Markovian comparison", flush=True)
    # First verify the two existing implementations coincide in the underdamped regime.
    underdamped_consistency = 0.0
    for ip in (0, 2):
        for time in (0.0, 0.37, 3.1, 20.9, 47.3):
            rho_main = evolve_rho(point_rho[ip], time, kind="amp",
                                  bath="independent", **NOISE["high"])
            rho_fid = _evolve_rho_fid(
                point_rho[ip], time, kind="amp", bath="independent",
                nuA=0.2, GammaA=0.01, nuB=0.2, GammaB=0.01,
                gammaA=4.5, GammaAD_A=0.01, gammaB=4.5, GammaAD_B=0.01)
            underdamped_consistency = max(underdamped_consistency,
                                          maxabs(rho_main - rho_fid))
    if underdamped_consistency > 2e-12:
        raise RuntimeError("Underdamped implementations disagree")

    # Gamma/gamma=100 is already close to the memoryless limit. The shorter
    # interval avoids overflow in fidelity.py's direct cosh/sinh evaluation;
    # gamma*t=6.75 at the endpoint, so the advantage has long disappeared.
    t_m = np.linspace(0.0, 1.5, 1201)
    gamma_mark = 4.5
    Gamma_mark = 100 * gamma_mark
    markovian = np.zeros((2, len(t_m), len(METRICS)))
    exponential = np.zeros_like(markovian)
    eta_lorentz = np.array([eta_amp_lorentz(time, gamma_mark, Gamma_mark)
                            for time in t_m])
    eta_exponential = np.exp(-gamma_mark * t_m)
    for ip, point_index in enumerate((0, 2)):
        markovian[ip] = evolve_metrics_overdamped(
            point_rho[point_index], t_m, gamma_mark, Gamma_mark)
        exponential[ip] = evolve_metrics_exponential(
            point_rho[point_index], t_m, gamma_mark)

    print("Stage 6/6: figures, arrays, and numerical summary", flush=True)
    i75 = int(np.where(U_VALUES == 75)[0][0])

    # Numerical diagnostics and concise machine-readable conclusions.
    optimized_deficit = 0.0
    for array in (dynamics["low"], dynamics["high"], map_metrics,
                  markovian, exponential, u_dynamics):
        optimized_deficit = max(optimized_deficit,
                                float(np.max(array[..., 2] - array[..., 4])))
    if optimized_deficit > 2e-8:
        raise RuntimeError(f"Optimized below fixed by {optimized_deficit}")

    events = {}
    for noise in NOISE:
        events[noise] = {}
        for ip, radius in enumerate(point_r):
            events[noise][str(radius)] = {
                protocol: threshold_events(t, dynamics[noise][ip, :, index])
                for protocol, index in (("fixed", 2), ("optimized", 4))}
    dephasing_events = {}
    for bath, array in dephasing.items():
        dephasing_events[bath] = {
            str(point_r[idx]): threshold_events(t, array[ip, :, 4])
            for ip, idx in enumerate((0, 2))}
    markovian_events = {
        str(point_r[idx]): threshold_events(t_m, markovian[ip, :, 4])
        for ip, idx in enumerate((0, 2))}
    exponential_events = {
        str(point_r[idx]): threshold_events(t_m, exponential[ip, :, 4])
        for ip, idx in enumerate((0, 2))}
    u_events = {
        str(U): threshold_events(t_u, u_dynamics[iu, :, 4])
        for iu, U in enumerate(U_FULL)}

    # Direct normal/anomalous decomposition along the U=75 radial line.
    p_normal = x_line[i75] / (2.0 - x_line[i75])
    p_anomalous = y_line[i75] / (2.0 + y_line[i75])
    rc_index = int(np.argmin(np.abs(r_line - rcrit[i75])))
    outside_first = r_line >= 1.0
    full_outside = {
        str(U): {
            "max_p_at_r_greater_or_equal_1_all_angles": float(
                np.max(full["p"][iu][:, r_full >= 1.0])),
            "any_advantage_at_r_greater_or_equal_1": bool(
                np.any(full["p"][iu][:, r_full >= 1.0] > P_THRESHOLD)),
            "max_finite_r_p_all_angles": float(
                np.max(full["p"][iu][:, 1:])),
        }
        for iu, U in enumerate(U_FULL)}

    summary = {
        "repository_commit": commit,
        "scope": {
            "U_over_t_values": U_VALUES.tolist(),
            "fixed_parameters": {
                "Delta_t_over_t": config.Delta_t, "V_SO_over_t": config.V_SO,
                "n_elec": config.n_elec, "temperature": 0.0},
            "greens_convention": "equal_time v^2 normal and signed Delta*/(2E) anomalous, single-spin helicity factor 1/2",
            "initial_U_r_map_Nk": 100,
            "key_dynamics_Nk": 400,
            "angle_for_U_r_and_dynamics_deg": 45.0,
            "classical_average_fidelity": CLASSICAL_BENCHMARK,
            "Werner_threshold": P_THRESHOLD,
            "distance_unit": "lattice constant a",
            "time_unit": "hbar/t",
        },
        "mathematical_identities": {
            "initial_state": "Werner with p=(g^2+f^2)/(2+f^2-g^2)",
            "initial_fixed_and_optimized_fidelity": "(1+p)/2",
            "initial_concurrence": "max(0,(3p-1)/2)",
            "threshold_in_g_f": "2*g^2+f^2=1",
            "initial_protocol_validation_max_abs_errors": initial_protocol_errors,
            "local_noise_cannot_create_entanglement": (
                "No above-2/3 revival is expected from an initially separable finite-r Werner resource under the implemented local channels; this was also checked numerically on the selected grid."),
        },
        "solver_records": solver_records,
        "initial_thresholds": {
            str(U): {
                "critical_r_over_a_Nk100": float(rcrit[iu]),
                "max_finite_p_theta45": float(np.max(p_line[iu, 1:])),
                "max_p_at_r_greater_or_equal_1_theta45": float(np.max(p_line[iu, r_line >= 1])),
                "any_above_threshold_beyond_first_crossing_up_to_r1.5": bool(
                    np.any(p_line[iu, int(np.searchsorted(r_line, rcrit[iu])) + 1:] > P_THRESHOLD)),
            } for iu, U in enumerate(U_VALUES)},
        "exact_lattice_sites": {
            "coordinates": [list(site) for site in SITES],
            "r_over_a": site_r.tolist(),
            "theta_deg": site_theta.tolist(),
            "maximum_p_over_U_and_sites": float(np.max(site_p)),
            "maximum_average_fidelity_over_U_and_sites": float((1 + np.max(site_p)) / 2),
            "any_above_classical": bool(np.any(site_p > P_THRESHOLD)),
            "p_values": {str(U): site_p[iu].tolist() for iu, U in enumerate(U_VALUES)},
        },
        "full_angular_checks": full_outside,
        "normal_anomalous_decomposition_U75": {
            "max_abs_p_minus_normal_only": float(np.max(np.abs(p_line[i75] - p_normal))),
            "max_anomalous_only_p": float(np.max(p_anomalous)),
            "nearest_grid_to_critical": {
                "r_over_a": float(r_line[rc_index]),
                "g2": float(x_line[i75, rc_index]),
                "f2": float(y_line[i75, rc_index]),
                "p": float(p_line[i75, rc_index]),
                "normal_only_p": float(p_normal[rc_index]),
            },
            "interpretation_boundary": (
                "The f=0 diagnostic isolates the direct anomalous term in the Werner formula. Pairing also affects g through the self-consistent v^2 weights, so this is not a complete no-superconductivity counterfactual."),
        },
        "convergence": convergence_records,
        "radial_interpolation_convergence": radial_records,
        "dynamic_points": point_micro,
        "threshold_events_amplitude": events,
        "threshold_events_dephasing_optimized": dephasing_events,
        "threshold_events_markovian_optimized_Gamma_over_gamma_100": markovian_events,
        "threshold_events_exact_exponential_optimized": exponential_events,
        "threshold_events_high_U_at_r0.40_optimized": u_events,
        "high_U_dynamics_solution_types": {
            str(U): ("paired" if max(abs(float(bdg400[U]["Delta0"])),
                                     abs(float(bdg400[U]["DeltaS"]))) > 1e-5
                     else "zero-gap control")
            for U in U_FULL},
        "underdamped_main_vs_exact_continuation_max_matrix_error": underdamped_consistency,
        "markovian_limit": {
            "gamma": gamma_mark, "Gamma": Gamma_mark,
            "Gamma_over_gamma": 100.0,
            "max_abs_eta_lorentz_minus_exp": float(np.max(np.abs(eta_lorentz - eta_exponential))),
            "max_optimized_curve_abs_difference": float(np.max(np.abs(
                markovian[..., 4] - exponential[..., 4]))),
            "max_fixed_curve_abs_difference": float(np.max(np.abs(
                markovian[..., 2] - exponential[..., 2]))),
            "implementation_note": (
                "noise_channels.py uses a critically damped expression throughout d^2<=0. The overdamped comparison therefore uses fidelity.py's exact hyperbolic continuation, after checking agreement with evolution.py in the underdamped regime."),
        },
        "dephasing_envelope_monotonic": bool(np.all(np.diff([
            p_dephasing(time, 1.5, 0.1) for time in t]) <= 1e-14)),
        "optimized_minus_fixed_max_violation": optimized_deficit,
        "backflow_statement": (
            "Threshold re-crossings are reported as fidelity revivals only. No operational information-backflow witness or divisibility analysis is claimed."),
        "files": {
            "initial_data": "data/microscopic_teleportation_initial.npz",
            "dynamics_data": "data/microscopic_teleportation_dynamics.npz",
            "convergence_data": "data/microscopic_teleportation_convergence.npz",
            "figures": [
                "figs/microscopic_initial_U_r_map.png",
                "figs/microscopic_gf_decomposition.png",
                "figs/microscopic_teleportation_convergence.png",
                "figs/microscopic_teleportation_revival_maps.png",
                "figs/microscopic_markovian_comparison.png",
            ],
        },
    }

    np.savez_compressed(
        DATA / "microscopic_teleportation_initial.npz",
        U_values=U_VALUES, r_line=r_line, theta_line_deg=theta_line,
        x_line=x_line, y_line=y_line, p_line=p_line,
        G_line=G_line, F_line=F_line, critical_r=rcrit,
        U_full=np.array(U_FULL), r_full=r_full, theta_full=theta_full,
        x_full=full["x"], y_full=full["y"], p_full=full["p"],
        G_full=full_G, F_full=full_F,
        sites=np.array(SITES), site_r=site_r, site_theta_deg=site_theta,
        site_x=site_x, site_y=site_y, site_p=site_p,
        solver_records=np.array([json.dumps(r) for r in solver_records]),
        repository_commit=np.bytes_(commit))
    np.savez_compressed(
        DATA / "microscopic_teleportation_dynamics.npz",
        t=t, point_r=point_r, point_rho=point_rho,
        point_micro=np.array([json.dumps(r) for r in point_micro]),
        noise_names=np.array(list(NOISE)),
        low_metrics=dynamics["low"], high_metrics=dynamics["high"],
        metric_names=np.array(METRICS),
        dephasing_independent=dephasing["independent"],
        dephasing_common=dephasing["common"],
        t_u=t_u, U_full=np.array(U_FULL), U_dynamics=u_dynamics,
        r_map=r_map, t_map=t_map, map_metrics=map_metrics,
        t_markovian=t_m, markovian_metrics=markovian,
        exponential_metrics=exponential,
        eta_lorentz=eta_lorentz, eta_exponential=eta_exponential,
        repository_commit=np.bytes_(commit))
    np.savez_compressed(
        DATA / "microscopic_teleportation_convergence.npz",
        U_full=np.array(U_FULL), Nk_values=np.array(NK_VALUES), r_conv=r_conv,
        p_conv=p_conv, x_conv=x_conv, y_conv=y_conv,
        convergence_records=np.array([json.dumps(r) for r in convergence_records]),
        radial_records=np.array([json.dumps(r) for r in radial_records]),
        repository_commit=np.bytes_(commit))

    (REPORTS / "microscopic_teleportation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    # Serialize all numerical evidence before rendering figures. A plotting
    # failure therefore cannot discard the completed simulation outputs.
    plot_initial_study(r_line, U_VALUES, p_line, x_line, y_line, rcrit,
                       site_p, site_r, {
                           "Delta0": np.array([r["Delta0_over_t"] for r in solver_records]),
                           "DeltaS": np.array([r["DeltaS_over_t"] for r in solver_records]),
                       })
    plot_gf_decomposition(r_line, x_line[i75], y_line[i75], p_line[i75], rcrit[i75])
    plot_convergence(r_conv, p_conv,
                     np.array([[r["critical_r_over_a"] for r in convergence_records
                                if r["U_over_t"] == U] for U in U_FULL]))
    plot_dynamics(t, dynamics, point_r, t_map, r_map, map_metrics,
                  t_u, u_dynamics)
    plot_markovian(t, dynamics["high"][[0, 2]], t_m, markovian, exponential)

    print(json.dumps({
        "status": "completed",
        "critical_r_Nk100_U75": float(rcrit[i75]),
        "critical_r_convergence": [r for r in convergence_records if r["U_over_t"] == 75],
        "max_exact_lattice_site_p": float(np.max(site_p)),
        "high_noise_r0.40_events": events["high"]["0.4"],
        "markovian_limit_max_eta_error": summary["markovian_limit"]["max_abs_eta_lorentz_minus_exp"],
        "optimized_minus_fixed_violation": optimized_deficit,
        "summary": "reports/microscopic_teleportation_summary.json",
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
