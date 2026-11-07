#!/usr/bin/env python3
"""Run explicit and optimized teleportation on existing BdG-derived resources.

The BdG, Green's-function, and Kraus implementations are reused unchanged.
Q is an effective input spin.  Addressable resource modes, ideal Bell
measurement, and ideal classical feedforward are assumptions of the protocol.

Run from repository root:
    .venv/bin/python analysis/run_teleportation_analysis.py
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
from density_matrix import build_density_matrices, verify_density_matrices
from evolution import evolve_rho
from greens_functions import compute_greens_functions
from self_consistency import run_selfconsistency
from teleportation import CLASSICAL_BENCHMARK, resource_metrics

DATA = ROOT / "data"
FIGS = ROOT / "figs"
REPORTS = ROOT / "reports"
FIGS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

NOISE = {
    "low": dict(gammaA=1.5, GammaA=0.1, gammaB=1.5, GammaB=0.1),
    "high": dict(gammaA=4.5, GammaA=0.01, gammaB=4.5, GammaB=0.01),
}
R_TARGETS = (0.0, 1.5, 3.0)
ANGLE_TARGETS = (0.0, 45.0, 90.0)


def nearest_index(values, target):
    return int(np.argmin(np.abs(np.asarray(values) - target)))


def plot_metric_group(t, arrays, points, title, path):
    """Plot C, Fs, fixed/optimized fidelity, and their difference."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    for ip, point in enumerate(points):
        color = colors[ip % len(colors)]
        r_value, angle = point["r"], point["theta_deg"]
        label = fr"$r={r_value:.3f},\ \theta={angle:.0f}^\circ$"
        axes[0, 0].plot(t, arrays["concurrence"][ip], color=color, label=label)
        axes[0, 1].plot(t, arrays["singlet_fraction"][ip], color=color, label=label)
        axes[1, 0].plot(t, arrays["fixed_average_fidelity"][ip], color=color,
                        label=label + " fixed")
        axes[1, 0].plot(t, arrays["optimized_average_fidelity"][ip], "--",
                        color=color, label=label + " optimized")
        axes[1, 1].plot(t,
                        arrays["optimized_average_fidelity"][ip]
                        - arrays["fixed_average_fidelity"][ip],
                        color=color, label=label)

    axes[0, 0].set_ylabel(r"Concurrence $C(t)$")
    axes[0, 1].set_ylabel(r"Singlet fraction $F_s(t)$")
    axes[1, 0].set_ylabel(r"Average teleportation fidelity")
    axes[1, 1].set_ylabel(r"$\overline{\mathcal{F}}_{opt}-\overline{\mathcal{F}}_{fixed}$")
    axes[1, 0].axhline(CLASSICAL_BENCHMARK, color="black", ls=":", lw=1.5,
                       label=r"classical $2/3$")
    axes[1, 1].axhline(0.0, color="black", lw=0.8)
    axes[1, 0].set_ylim(0.0, 1.03)
    for ax in axes.flat:
        ax.set_xlim(t[0], t[-1])
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel(r"$t$")
    fig.suptitle(title)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    print("Solving the existing U/t=75 BdG model ...")
    bdg = run_selfconsistency(seed=42, U_override=75, kBT_override=0.0,
                              verbose=False)
    residual = float(np.max(np.abs(bdg["residuals"])))
    if not bdg["converged"] or residual > 1e-8:
        raise RuntimeError(f"Unreliable BdG solution: residual={residual}")

    r, theta, G, F = compute_greens_functions(
        bdg, verbose=False, convention="equal_time")
    rho0 = np.asarray(build_density_matrices(G, F))
    if not verify_density_matrices(rho0, atol=2e-8):
        raise RuntimeError("Initial BdG density matrices failed validation")

    # Three distances at 45 degrees, plus 0 and 90 degrees at the middle
    # distance.  The middle point is shared by the two plot groups.
    point_specs = []
    for target_r in R_TARGETS:
        point_specs.append((nearest_index(r, target_r),
                            nearest_index(np.degrees(theta), 45.0)))
    middle_ir = point_specs[1][0]
    for target_angle in (0.0, 90.0):
        point_specs.append((middle_ir,
                            nearest_index(np.degrees(theta), target_angle)))
    points = [{"ir": ir, "itheta": it, "r": float(r[ir]),
               "theta_deg": float(np.degrees(theta[it]))}
              for ir, it in point_specs]

    t = np.linspace(config.T_MIN, config.T_MAX, config.N_T)
    noise_names = tuple(NOISE)
    metric_names = ("concurrence", "singlet_fraction",
                    "fixed_average_fidelity", "fully_entangled_fraction",
                    "optimized_average_fidelity")
    arrays = {
        noise: {name: np.zeros((len(points), len(t))) for name in metric_names}
        for noise in noise_names
    }
    diagnostics = {
        "minimum_output_eigenvalue": np.inf,
        "maximum_hermiticity_error": 0.0,
        "maximum_trace_error": 0.0,
        "maximum_probability_sum_error": 0.0,
        "maximum_optimized_minus_fixed_violation": 0.0,
    }

    for noise, noise_params in NOISE.items():
        for ip, point in enumerate(points):
            rho_initial = rho0[point["itheta"], point["ir"]]
            print(f"{noise}: r={point['r']:.4f}, theta={point['theta_deg']:.0f}")
            for it, time in enumerate(t):
                rho_t = evolve_rho(rho_initial, float(time), kind="amp",
                                   bath="independent", **noise_params)
                metrics = resource_metrics(rho_t, validate=False)
                for name in metric_names:
                    arrays[noise][name][ip, it] = metrics[name]
                diag = metrics["diagnostics"]
                diagnostics["minimum_output_eigenvalue"] = min(
                    diagnostics["minimum_output_eigenvalue"],
                    diag["minimum_output_eigenvalue"])
                diagnostics["maximum_hermiticity_error"] = max(
                    diagnostics["maximum_hermiticity_error"],
                    diag["maximum_hermiticity_error"])
                diagnostics["maximum_trace_error"] = max(
                    diagnostics["maximum_trace_error"],
                    diag["maximum_trace_error"])
                diagnostics["maximum_probability_sum_error"] = max(
                    diagnostics["maximum_probability_sum_error"],
                    diag["maximum_probability_sum_error"])
                diagnostics["maximum_optimized_minus_fixed_violation"] = max(
                    diagnostics["maximum_optimized_minus_fixed_violation"],
                    float(metrics["fixed_average_fidelity"]
                          - metrics["optimized_average_fidelity"]))

    if diagnostics["minimum_output_eigenvalue"] < -1e-9:
        raise RuntimeError("Negative teleportation-output eigenvalue detected")
    for key in ("maximum_hermiticity_error", "maximum_trace_error",
                "maximum_probability_sum_error"):
        if diagnostics[key] > 1e-8:
            raise RuntimeError(f"Teleportation diagnostics failed: {key}")
    if diagnostics["maximum_optimized_minus_fixed_violation"] > 1e-8:
        raise RuntimeError("Optimized fidelity is below fixed-protocol fidelity")

    plot_metric_group(
        t, arrays["low"], points[:3],
        "Teleportation at $U/t=75$, $\\theta=45^\\circ$, low coupling",
        FIGS / "teleportation_theta45_low.png")
    plot_metric_group(
        t, arrays["high"], points[:3],
        "Teleportation at $U/t=75$, $\\theta=45^\\circ$, high coupling",
        FIGS / "teleportation_theta45_high.png")
    angle_points = [points[3], points[1], points[4]]
    angle_arrays = {name: values[[3, 1, 4]] for name, values in arrays["low"].items()}
    plot_metric_group(
        t, angle_arrays, angle_points,
        fr"Teleportation at $U/t=75$, $r={points[1]['r']:.4f}$, low coupling",
        FIGS / "teleportation_angles_low.png")

    save = {
        "t_grid": t, "noise_names": np.array(noise_names),
        "point_r": np.array([p["r"] for p in points]),
        "point_theta_deg": np.array([p["theta_deg"] for p in points]),
        "point_r_indices": np.array([p["ir"] for p in points]),
        "point_theta_indices": np.array([p["itheta"] for p in points]),
        "U_over_t": 75.0, "V_SO": config.V_SO, "n_elec": config.n_elec,
        "Nk": config.Nk, "Delta_t": config.Delta_t,
        "greens_convention": np.bytes_("equal_time_single_spin"),
        "classical_benchmark": CLASSICAL_BENCHMARK,
        "repository_commit": np.bytes_(commit),
    }
    for noise in noise_names:
        for name, values in arrays[noise].items():
            save[f"{noise}_{name}"] = values
    np.savez_compressed(DATA / "teleportation_metrics.npz", **save)

    summary = {
        "repository_commit": commit,
        "model": {
            "U_over_t": 75.0, "V_SO_over_t": config.V_SO,
            "Delta_t_over_t": config.Delta_t, "n_elec": config.n_elec,
            "Nk": config.Nk, "max_abs_bdg_residual": residual,
            "greens_convention": "equal_time single-spin v^2 and signed F*",
        },
        "noise": NOISE,
        "time_grid": {"min": float(t[0]), "max": float(t[-1]), "points": len(t)},
        "classical_benchmark": CLASSICAL_BENCHMARK,
        "optimization": (
            "Fully entangled fraction evaluated exactly by SVD of the 3x3 "
            "Pauli correlation matrix; no numerical optimizer used."),
        "assumptions": [
            "Q is an effective input spin qubit, not automatically a Bogoliubov quasiparticle.",
            "A and B are independently addressable spin modes after extraction.",
            "Q is initially independent of the resource.",
            "Bell measurement and classical feedforward are ideal.",
            "No microscopic quasiparticle--Cooper-pair interaction is simulated.",
        ],
        "diagnostics": diagnostics,
        "points": [],
        "data_file": "data/teleportation_metrics.npz",
        "figures": [
            "figs/teleportation_theta45_low.png",
            "figs/teleportation_theta45_high.png",
            "figs/teleportation_angles_low.png",
        ],
    }
    for ip, point in enumerate(points):
        entry = dict(point)
        for noise in noise_names:
            entry[noise] = {}
            for name in metric_names:
                values = arrays[noise][name][ip]
                entry[noise][name] = {
                    "initial": float(values[0]), "final": float(values[-1]),
                    "min": float(values.min()), "max": float(values.max()),
                }
            fixed = arrays[noise]["fixed_average_fidelity"][ip]
            optimized = arrays[noise]["optimized_average_fidelity"][ip]
            entry[noise]["fixed_ever_above_classical"] = bool(
                np.any(fixed > CLASSICAL_BENCHMARK))
            entry[noise]["optimized_ever_above_classical"] = bool(
                np.any(optimized > CLASSICAL_BENCHMARK))
            entry[noise]["max_optimized_minus_fixed"] = float(
                np.max(optimized - fixed))
        summary["points"].append(entry)

    (REPORTS / "teleportation_analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    (REPORTS / "teleportation_analysis_stdout.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "completed", "diagnostics": diagnostics,
                      "data": save["data_file"] if "data_file" in save else
                      "data/teleportation_metrics.npz",
                      "figures": summary["figures"]}, indent=2))


if __name__ == "__main__":
    main()
