#!/usr/bin/env python3
"""Compare Ketterson equal-time correlators with the archived legacy code.

The archived arrays and manuscript are not overwritten.  Both conventions use
identical converged BdG solutions, density-matrix algebra, and Kraus dynamics.

Run from the repository root:
    .venv/bin/python analysis/compare_equal_time_greens.py
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
from analysis.reproduce_and_validate import observable_arrays
from density_matrix import build_density_matrices, verify_density_matrices
from greens_functions import compute_greens_functions
from self_consistency import run_selfconsistency

DATA = ROOT / "data"
FIGS = ROOT / "figs"
REPORTS = ROOT / "reports"
FIGS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

U_VALUES = (40, 75, 115)
CONVENTIONS = ("legacy_unsigned", "equal_time")
NOISE = {
    "low": dict(gamma=1.5, Gamma=0.1),
    "high": dict(gamma=4.5, Gamma=0.01),
}
OBS = ("C", "Fs", "chi")


def maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def normalized_gf(G, F):
    G0 = np.abs(G[:, [0]])
    return np.abs(G) / G0, np.abs(F) / G0


def werner_p(g, f):
    return (g * g + f * f) / (2.0 + f * f - g * g)


def plot_greens_u75(r, gf, p_init):
    i45 = 3
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.3), constrained_layout=True)
    labels = {"legacy_unsigned": "legacy $u^2$, unsigned $uv$",
              "equal_time": "equal-time $v^2$, signed $F^*$"}
    styles = {"legacy_unsigned": "--", "equal_time": "-"}
    for conv in CONVENTIONS:
        G, F, g, f = gf[conv]
        axes[0, 0].plot(r, np.abs(G[i45]), styles[conv], label=labels[conv])
        axes[0, 1].plot(r, g[i45] ** 2, styles[conv], label=labels[conv])
        axes[1, 0].plot(r, f[i45] ** 2, styles[conv], label=labels[conv])
        axes[1, 1].plot(r, p_init[conv][i45], styles[conv], label=labels[conv])
    axes[0, 0].set_ylabel(r"$|G(r,45^\circ)|$")
    axes[0, 1].set_ylabel(r"$g^2(r,45^\circ)$")
    axes[1, 0].set_ylabel(r"$f^2(r,45^\circ)$")
    axes[1, 1].set_ylabel(r"Werner $p(r,45^\circ)$")
    for ax in axes.flat:
        ax.set_xlabel(r"$r$ (lattice constant)")
        ax.set_xlim(0, 6)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    axes[1, 1].set_ylim(-0.02, 1.02)
    fig.suptitle("Green's-function convention comparison at $U/t=75$, $V_{SO}=0$")
    fig.savefig(FIGS / "equal_time_greens_comparison_U75.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def plot_initial_u_sweep(r, initial):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    colors = {40: "tab:blue", 75: "tab:orange", 115: "tab:green"}
    for U in U_VALUES:
        for conv in CONVENTIONS:
            style = "--" if conv == "legacy_unsigned" else "-"
            label = fr"$U={U}$ " + ("legacy" if conv == "legacy_unsigned" else "equal-time")
            axes[0, 0].plot(r, initial[conv]["Fs"][U][3], style,
                            color=colors[U], label=label)
            axes[0, 1].plot(r, initial[conv]["chi"][U][3], style,
                            color=colors[U], label=label)
        axes[1, 0].plot(r, initial["equal_time"]["Fs"][U][3]
                        - initial["legacy_unsigned"]["Fs"][U][3],
                        color=colors[U], label=fr"$U={U}$")
        axes[1, 1].plot(r, initial["equal_time"]["C"][U][3]
                        - initial["legacy_unsigned"]["C"][U][3],
                        color=colors[U], label=fr"$U={U}$")
    axes[0, 0].set_ylabel(r"Initial $F_s(r,45^\circ)$")
    axes[0, 1].set_ylabel(r"Initial $\chi_P(r,45^\circ)$ [bits]")
    axes[1, 0].set_ylabel(r"$\Delta F_s=F_s^{equal}-F_s^{legacy}$")
    axes[1, 1].set_ylabel(r"$\Delta C=C^{equal}-C^{legacy}$")
    for ax in axes.flat:
        ax.set_xlabel(r"$r$ (lattice constant)")
        ax.set_xlim(0, 6)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 1].legend(fontsize=8, ncol=2)
    axes[1, 0].legend()
    axes[1, 1].legend()
    for ax in axes[1]:
        ax.axhline(0, color="black", lw=0.8)
    fig.suptitle("Initial-state changes for the converged interaction strengths")
    fig.savefig(FIGS / "equal_time_initial_U_comparison.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def plot_dynamics_u75(t_grids, time_selected, r_selected):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), constrained_layout=True)
    titles = {"C": "Concurrence", "Fs": "Singlet fraction", "chi": r"$\chi_P$ [bits]"}
    for row, noise in enumerate(("low", "high")):
        t = t_grids[noise]
        for col, obs in enumerate(OBS):
            ax = axes[row, col]
            for ir, r_value in enumerate(r_selected):
                for conv in CONVENTIONS:
                    style = "--" if conv == "legacy_unsigned" else "-"
                    label = (fr"$r={r_value:.3f}$ " +
                             ("legacy" if conv == "legacy_unsigned" else "equal-time"))
                    ax.plot(t, time_selected[noise][obs][conv][75][ir], style,
                            label=label)
            ax.set_title(titles[obs] + fr", {noise} coupling")
            ax.set_xlabel(r"$t$")
            ax.set_xlim(t[0], t[-1])
            ax.grid(alpha=0.25)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Time-dependent changes at $U/t=75$, $\theta=45^\circ$")
    fig.savefig(FIGS / "equal_time_dynamics_U75.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    t_grids = {
        "low": np.linspace(config.T_MIN, config.T_MAX, config.N_T),
    }
    with np.load(DATA / "hubbard_amp_independent_gA4.50_GA0.010_gB4.50_GB0.010_fidelity.npz") as d:
        t_grids["high"] = d["t_grid"].copy()

    results = {}
    r = theta = None
    for U in U_VALUES:
        print(f"Solving U/t={U} ...")
        bdg = run_selfconsistency(seed=42, U_override=U, kBT_override=0.0,
                                  verbose=False)
        residual = float(np.max(np.abs(bdg["residuals"])))
        if not bdg["converged"] or residual > 1e-8:
            raise RuntimeError(f"U={U}: unreliable solution, residual={residual}")
        results[U] = {"bdg": bdg, "residual": residual}
        for conv in CONVENTIONS:
            print(f"  {conv}: Green's functions and observables")
            r, theta, G, F = compute_greens_functions(bdg, verbose=False,
                                                      convention=conv)
            rho = np.asarray(build_density_matrices(G, F))
            if not verify_density_matrices(rho, atol=2e-8):
                raise RuntimeError(f"U={U}, {conv}: density-matrix check failed")
            g, f = normalized_gf(G, F)
            results[U][conv] = {
                "G": G, "F": F, "g": g, "f": f, "rho": rho,
                "p": werner_p(g, f),
                "eigen_min": float(np.min(np.linalg.eigvalsh(rho))),
            }
            for noise, pars in NOISE.items():
                results[U][conv][noise] = observable_arrays(
                    rho, t_grids[noise], **pars)

    ir_fixed = int(np.argmin(np.abs(r - 1.5)))
    ir_1608 = int(np.argmin(np.abs(r - 1.608)))
    ir_050 = int(np.argmin(np.abs(r - 0.5)))
    r_selected = np.array([r[0], r[ir_fixed]])

    initial = {conv: {obs: {} for obs in OBS} for conv in CONVENTIONS}
    p_init = {}
    gf_u75 = {}
    for conv in CONVENTIONS:
        gf_u75[conv] = tuple(results[75][conv][key] for key in ("G", "F", "g", "f"))
        p_init[conv] = results[75][conv]["p"]
        for obs in OBS:
            for U in U_VALUES:
                initial[conv][obs][U] = results[U][conv]["low"][obs][:, :, 0]

    time_selected = {noise: {obs: {conv: {} for conv in CONVENTIONS}
                             for obs in OBS} for noise in NOISE}
    for noise in NOISE:
        for obs in OBS:
            for conv in CONVENTIONS:
                for U in U_VALUES:
                    # theta=45 degrees; radial slices r=0 and r approximately 1.5075.
                    time_selected[noise][obs][conv][U] = np.stack([
                        results[U][conv][noise][obs][3, 0, :],
                        results[U][conv][noise][obs][3, ir_fixed, :],
                    ])

    summary = {
        "repository_commit_at_analysis": commit,
        "default_convention": "Ketterson equal-time G_< and F*_<; v^2 and Delta*/(2E), with degenerate-helicity factor 1/2",
        "legacy_convention": "Archived u^2 normal particle branch and unsigned real-u,v anomalous product",
        "fixed_parameters": {"V_SO_over_t": config.V_SO,
                             "Delta_t_over_t": config.Delta_t,
                             "n_elec": config.n_elec, "Nk": config.Nk},
        "U_values": list(U_VALUES),
        "r_fixed": float(r[ir_fixed]),
        "r_1608": float(r[ir_1608]),
        "r_050": float(r[ir_050]),
        "solutions": {},
        "maximum_changes": {},
        "selected_initial_theta45": {},
        "r0_amplitude_damping_identity_max_abs_error": {},
    }
    with np.load(DATA / "hubbard_amp_independent_gA1.50_GA0.100_gB1.50_GB0.100_greens.npz") as archived:
        summary["archived_legacy_reproduction_max_abs_error"] = {
            "G_twice_single_spin_vs_archived": maxdiff(
                2.0 * results[75]["legacy_unsigned"]["G"], archived["G_t"][:, :, 0]),
            "F_twice_single_spin_vs_archived": maxdiff(
                2.0 * results[75]["legacy_unsigned"]["F"], archived["F_t"][:, :, 0]),
            "G_single_spin_vs_half_archived": maxdiff(
                results[75]["legacy_unsigned"]["G"], 0.5 * archived["G_t"][:, :, 0]),
            "F_single_spin_vs_half_archived": maxdiff(
                results[75]["legacy_unsigned"]["F"], 0.5 * archived["F_t"][:, :, 0]),
        }
    for U in U_VALUES:
        summary["solutions"][str(U)] = {
            "Delta0_over_t": float(results[U]["bdg"]["Delta0"]),
            "DeltaS_over_t": float(results[U]["bdg"]["DeltaS"]),
            "mu_over_t": float(results[U]["bdg"]["mu"]),
            "max_abs_residual": results[U]["residual"],
        }
        legacy = results[U]["legacy_unsigned"]
        equal = results[U]["equal_time"]
        summary["maximum_changes"][str(U)] = {
            "G": maxdiff(equal["G"], legacy["G"]),
            "F": maxdiff(equal["F"], legacy["F"]),
            "g": maxdiff(equal["g"], legacy["g"]),
            "f": maxdiff(equal["f"], legacy["f"]),
            "initial_Werner_p": maxdiff(equal["p"], legacy["p"]),
            "initial_C": maxdiff(initial["equal_time"]["C"][U],
                                   initial["legacy_unsigned"]["C"][U]),
            "initial_Fs": maxdiff(initial["equal_time"]["Fs"][U],
                                  initial["legacy_unsigned"]["Fs"][U]),
            "initial_chi": maxdiff(initial["equal_time"]["chi"][U],
                                   initial["legacy_unsigned"]["chi"][U]),
            "legacy_G0_theta0_abs": float(abs(legacy["G"][0, 0])),
            "equal_time_G0_theta0_abs": float(abs(equal["G"][0, 0])),
            "legacy_unoccupied_sum_rule_target_(2_minus_n)/2": (2.0 - config.n_elec) / 2.0,
            "legacy_unoccupied_sum_rule_relative_quadrature_error": float(
                abs(abs(legacy["G"][0, 0]) - (2.0 - config.n_elec) / 2.0)
                / ((2.0 - config.n_elec) / 2.0)),
            "legacy_discrete_G0": float(
                0.5 * np.sum(results[U]["bdg"]["u_ks"]**2) / config.Nk**2),
            "equal_time_occupancy_sum_rule_target_n/2": config.n_elec / 2.0,
            "equal_time_occupancy_sum_rule_relative_quadrature_error": float(
                abs(abs(equal["G"][0, 0]) - config.n_elec / 2.0) / (config.n_elec / 2.0)),
            "equal_time_discrete_G0": float(
                0.5 * np.sum(results[U]["bdg"]["v_ks"]**2) / config.Nk**2),
            "equal_time_discrete_sum_rule_abs_error": float(abs(
                0.5 * np.sum(results[U]["bdg"]["v_ks"]**2) / config.Nk**2
                - config.n_elec / 2.0)),
            "legacy_density_min_eigenvalue": legacy["eigen_min"],
            "equal_time_density_min_eigenvalue": equal["eigen_min"],
        }
        for noise in NOISE:
            for obs in OBS:
                summary["maximum_changes"][str(U)][f"{noise}_{obs}_all_r_theta_t"] = maxdiff(
                    equal[noise][obs], legacy[noise][obs])
            for obs in ("Fs", "C"):
                key = f"U{U}_{noise}_{obs}_minus_survival_p"
                summary["r0_amplitude_damping_identity_max_abs_error"][key] = maxdiff(
                    equal[noise][obs][:, 0, :], equal[noise]["p"])
        for label, ir in [("r0", 0), ("r050", ir_050),
                          ("r_fixed", ir_fixed), ("r1608", ir_1608)]:
            summary["selected_initial_theta45"].setdefault(str(U), {})[label] = {
                "r": float(r[ir]),
                "legacy_Fs": float(initial["legacy_unsigned"]["Fs"][U][3, ir]),
                "equal_time_Fs": float(initial["equal_time"]["Fs"][U][3, ir]),
                "legacy_C": float(initial["legacy_unsigned"]["C"][U][3, ir]),
                "equal_time_C": float(initial["equal_time"]["C"][U][3, ir]),
                "legacy_chi": float(initial["legacy_unsigned"]["chi"][U][3, ir]),
                "equal_time_chi": float(initial["equal_time"]["chi"][U][3, ir]),
            }

    plot_greens_u75(r, gf_u75, p_init)
    plot_initial_u_sweep(r, initial)
    plot_dynamics_u75(t_grids, time_selected, r_selected)

    # Compact evidence file: complete initial radial profiles and selected time slices.
    save = {
        "U_values": np.array(U_VALUES, float), "r_vals": r, "theta_vals": theta,
        "t_low": t_grids["low"], "t_high": t_grids["high"],
        "r_selected": r_selected, "r_fixed_index": ir_fixed,
        "conventions": np.array(CONVENTIONS),
    }
    for ci, conv in enumerate(CONVENTIONS):
        save[f"G_U75_{conv}"] = results[75][conv]["G"]
        save[f"F_U75_{conv}"] = results[75][conv]["F"]
        save[f"g_U75_{conv}"] = results[75][conv]["g"]
        save[f"f_U75_{conv}"] = results[75][conv]["f"]
        for obs in OBS:
            save[f"initial_{obs}_{conv}"] = np.stack([
                initial[conv][obs][U] for U in U_VALUES])
            for noise in NOISE:
                save[f"{noise}_{obs}_{conv}_selected"] = np.stack([
                    time_selected[noise][obs][conv][U] for U in U_VALUES])
    np.savez_compressed(DATA / "equal_time_vs_legacy_comparison.npz", **save)
    with open(REPORTS / "equal_time_vs_legacy_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    with open(REPORTS / "equal_time_vs_legacy_stdout.txt", "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
