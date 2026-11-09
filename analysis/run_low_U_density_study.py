#!/usr/bin/env python3
"""Density and teleportation study at U/t=15 and 25, Delta_t/t=4.5.

Reuses the validated lattice/BdG and equal-time Green-function conventions from
analysis/run_density_teleportation_study.py. Outputs are isolated from the
archived U/t=75 density study.

Run from the repository root:
    .venv/bin/python analysis/run_low_U_density_study.py
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
from analysis.run_density_teleportation_study import (
    ANGLES, DENSITIES, FINE_ANGLES, NK_PRODUCTION, NK_VALUES, P_THRESHOLD,
    RESIDUAL_TOL, SITES, ZERO_GAP_TOL, bdg_from_solution, critical_radius,
    exact_site_greens, fast_greens_equal_time, json_records, maxabs,
    normal_stationary_state, normalized_correlators, observables,
    paired_solver_state, protocol_validation, solution_record,
    threshold_recrossings, validate_fast_greens,
)

DATA = ROOT / "data"
FIGS = ROOT / "figs"
REPORTS = ROOT / "reports"
U_VALUES = np.array([15.0, 25.0])


def canonical_pair_gauge(bdg, n, nk):
    """Fix the irrelevant global gap sign by choosing DeltaS<=0."""
    if bdg["DeltaS"] > 0.0:
        return bdg_from_solution(n, nk, -bdg["Delta0"], -bdg["DeltaS"], bdg["mu"])
    return bdg


def solve(U, n, nk, seed=42):
    old_u = config.U
    config.U = float(U)
    try:
        candidate = paired_solver_state(n, nk, seed=seed)
        candidate = canonical_pair_gauge(candidate, n, nk)
        candidate_residual = maxabs(candidate["residuals"])
        candidate_paired = max(abs(candidate["Delta0"]),
                               abs(candidate["DeltaS"])) > ZERO_GAP_TOL
        normal, normal_info = normal_stationary_state(n, nk)
        if candidate_paired and candidate_residual < RESIDUAL_TOL:
            selected = candidate
            branch = "nonzero-gap stationary BdG branch"
        else:
            selected = normal
            branch = "zero-gap normal stationary branch"
    finally:
        config.U = old_u
    extra = dict(normal_info)
    extra.update({
        "U_over_t": float(U),
        "normal_reference_mu_over_t": float(normal["mu"]),
        "solver_candidate_max_abs_residual": candidate_residual,
        "solver_candidate_paired": bool(candidate_paired),
        "solver_candidate_converged_flag": bool(candidate.get("converged", False)),
        "solver_function_evaluations": int(candidate.get("n_iter", 0)),
        "gap_gauge": "DeltaS<=0; global gap sign is physically irrelevant",
    })
    record = solution_record(n, nk, selected, branch, seed=seed, extra=extra)
    return selected, normal, record


def plot_solutions(records):
    production = [r for r in records if r["Nk"] == NK_PRODUCTION]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for iu, U in enumerate(U_VALUES):
        rows = [r for r in production if r["U_over_t"] == U]
        color = plt.get_cmap("tab10").colors[iu]
        axes[0].plot(DENSITIES, [r["Delta0_over_t"] for r in rows], "o-",
                     color=color, label=fr"$U/t={U:g}$, $\Delta_0$")
        axes[0].plot(DENSITIES, [r["DeltaS_over_t"] for r in rows], "s--",
                     color=color, label=fr"$U/t={U:g}$, $\Delta_S$")
        axes[1].plot(DENSITIES, [r["mu_over_t"] for r in rows], "o-",
                     color=color, label=fr"$U/t={U:g}$")
        axes[2].plot(DENSITIES, [r["max_abs_Delta_ks_over_t"] for r in rows], "o-",
                     color=color, label=fr"$U/t={U:g}$")
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set(xlabel="Electron filling $n$", ylabel="Signed gap parameter (gauge $\\Delta_S\\leq0$)")
    axes[1].set(xlabel="Electron filling $n$", ylabel=r"$\mu/t$")
    axes[2].set(xlabel="Electron filling $n$", ylabel=r"$\max_{\bf k}|\Delta_{\bf k}|/t$")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(r"Lower-$U$ self-consistent branches at $\Delta_t/t=4.5$, $N_k=400$")
    fig.savefig(FIGS / "low_U_density_bdg_solutions.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_resources(r, g2, f2, p, rc):
    C, Ftel = observables(p)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True)
    for iu, U in enumerate(U_VALUES):
        for i, n in enumerate(DENSITIES):
            color = plt.get_cmap("viridis")(i/(len(DENSITIES)-1))
            ls = "-" if U == 15 else "--"
            label = fr"$U={U:g}$, $n={n:g}$"
            axes[0, 0].plot(r, g2[iu, i, 1], color=color, ls=ls, label=label)
            axes[0, 1].plot(r, f2[iu, i, 1], color=color, ls=ls, label=label)
            axes[0, 2].plot(r, C[iu, i, 1], color=color, ls=ls, label=label)
            axes[1, 0].plot(r, Ftel[iu, i, 1], color=color, ls=ls, label=label)
        axes[1, 1].plot(DENSITIES, rc[iu, :, 0], "o-", color=f"C{iu}",
                        label=fr"$U/t={U:g}$, $0^\circ$")
        axes[1, 1].plot(DENSITIES, rc[iu, :, 1], "s--", color=f"C{iu}",
                        label=fr"$U/t={U:g}$, $45^\circ$")
    axes[1, 0].axhline(2/3, color="black", ls=":", label="classical $2/3$")
    axes[0, 0].set(xlabel="$r/a$", ylabel="$g^2$", xlim=(0, 2), yscale="log", ylim=(1e-6, 1.2))
    axes[0, 1].set(xlabel="$r/a$", ylabel="$f^2$", xlim=(0, 2))
    axes[0, 2].set(xlabel="$r/a$", ylabel="$C(r,45^\circ)$", xlim=(0, 1.2))
    axes[1, 0].set(xlabel="$r/a$", ylabel=r"$\overline{\mathcal{F}}_{tel}$", xlim=(0, 1.2), ylim=(0.48, 1.01))
    axes[1, 1].set(xlabel="Electron filling $n$", ylabel=r"Continuous $r_c/a$")
    # Use the empty sixth panel for the consolidated U/n legend.
    handles, labels = axes[0, 2].get_legend_handles_labels()
    axes[1, 2].axis("off")
    axes[1, 2].legend(handles, labels, loc="center", ncol=1, fontsize=8)
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat[:5]:
        ax.grid(alpha=0.25)
    fig.suptitle(r"Continuous-distance resources at $\theta=45^\circ$; solid $U/t=15$, dashed $U/t=25$")
    fig.savefig(FIGS / "low_U_density_resources.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_sites(site_g2, site_f2, site_p):
    _, site_Ftel = observables(site_p)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for iu, U in enumerate(U_VALUES):
        color = f"C{iu}"
        axes[0].plot(DENSITIES, site_p[iu, :, 0], "o-", color=color,
                     label=fr"$U/t={U:g}$, $(1,0)$")
        axes[0].plot(DENSITIES, site_p[iu, :, 1], "s--", color=color,
                     label=fr"$U/t={U:g}$, $(1,1)$")
        axes[1].plot(DENSITIES, site_Ftel[iu, :, 0], "o-", color=color,
                     label=fr"$U/t={U:g}$")
        axes[2].plot(DENSITIES, site_g2[iu, :, 0], "o-", color=color,
                     label=fr"$U/t={U:g}$, $g^2$")
        axes[2].plot(DENSITIES, site_f2[iu, :, 0], "s--", color=color,
                     label=fr"$U/t={U:g}$, $f^2$")
    axes[0].axhline(P_THRESHOLD, color="black", ls=":", label="$p=1/3$")
    axes[1].axhline(2/3, color="black", ls=":", label="classical $2/3$")
    axes[0].set(xlabel="Electron filling $n$", ylabel="Exact-site Werner $p$")
    axes[1].set(xlabel="Electron filling $n$", ylabel=r"$\overline{\mathcal{F}}_{tel}(r=a)$", ylim=(0.49, 0.70))
    axes[2].set(xlabel="Electron filling $n$", ylabel="Nearest-site squared correlator")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Exact integer lattice vectors; no continuous-distance interpolation")
    fig.savefig(FIGS / "low_U_density_lattice_sites.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(records):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for iu, U in enumerate(U_VALUES):
        for i, n in enumerate(DENSITIES):
            rows = [r for r in records if r["U_over_t"] == U and r["n_elec"] == n]
            color = plt.get_cmap("viridis")(i/(len(DENSITIES)-1))
            ls = "-" if U == 15 else "--"
            nk = [r["Nk"] for r in rows]
            axes[0].plot(nk, [r["critical_r_over_a_theta45"] for r in rows],
                         color=color, ls=ls, marker="o", label=fr"$U={U:g}$, $n={n:g}$")
            axes[1].plot(nk, [r["nearest_site_p"] for r in rows],
                         color=color, ls=ls, marker="o")
            axes[2].plot(nk, [r["max_abs_Delta_ks_over_t"] for r in rows],
                         color=color, ls=ls, marker="o")
    axes[0].set(xlabel="$N_k$", ylabel=r"$r_c/a$ at $45^\circ$", xscale="log")
    axes[1].set(xlabel="$N_k$", ylabel="$p(r=a)$", xscale="log")
    axes[2].set(xlabel="$N_k$", ylabel=r"$\max_{\bf k}|\Delta_{\bf k}|/t$", xscale="log")
    axes[0].legend(ncol=2, fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("Momentum-mesh convergence; solid $U/t=15$, dashed $U/t=25$")
    fig.savefig(FIGS / "low_U_density_convergence.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    print("Stage 1/6: lower-U solutions on four momentum meshes", flush=True)
    states, normals, records = {}, {}, []
    for U in U_VALUES:
        for nk in NK_VALUES:
            for n in DENSITIES:
                state, normal, record = solve(U, n, nk)
                states[(U, n, nk)], normals[(U, n, nk)] = state, normal
                records.append(record)
                print(f"  U={U:g}, Nk={nk}, n={n:.3f}: {record['branch']}, "
                      f"D0={state['Delta0']:+.7f}, DS={state['DeltaS']:+.7f}, "
                      f"res={record['max_abs_residual']:.2e}", flush=True)

    print("Stage 2/6: six-seed checks and GF implementation validation", flush=True)
    multistart = []
    for U in U_VALUES:
        for n in DENSITIES:
            roots = []
            for seed in (7, 19, 42, 73, 101, 173):
                state, _, record = solve(U, n, 100, seed=seed)
                roots.append({
                    "seed": seed, "Delta0_over_t": record["Delta0_over_t"],
                    "DeltaS_over_t": record["DeltaS_over_t"], "mu_over_t": record["mu_over_t"],
                    "max_abs_residual": record["max_abs_residual"], "paired": record["paired"],
                    "branch": record["branch"],
                })
            paired = [r for r in roots if r["paired"] and r["max_abs_residual"] < RESIDUAL_TOL]
            multistart.append({
                "U_over_t": float(U), "n_elec": float(n), "Nk": 100,
                "paired_root_count": len(paired), "seed_results": roots,
                "canonical_Delta0_span": (max(r["Delta0_over_t"] for r in paired)-
                                            min(r["Delta0_over_t"] for r in paired)) if paired else None,
                "canonical_DeltaS_span": (max(r["DeltaS_over_t"] for r in paired)-
                                            min(r["DeltaS_over_t"] for r in paired)) if paired else None,
                "mu_span": (max(r["mu_over_t"] for r in paired)-
                            min(r["mu_over_t"] for r in paired)) if paired else None,
            })
            print(f"  U={U:g}, n={n:.3f}: {len(paired)} paired roots", flush=True)
    gf_error = validate_fast_greens(states[(15.0, 1.0, 100)],
                                    states[(25.0, 0.5, 100)])

    print("Stage 3/6: production radial/angular resources", flush=True)
    r_full = np.linspace(0.0, 10.0, 401)
    r_fine = np.linspace(0.0, 2.0, 801)
    full_shape = (len(U_VALUES), len(DENSITIES), len(ANGLES), len(r_full))
    fine_shape = (len(U_VALUES), len(DENSITIES), len(FINE_ANGLES), len(r_fine))
    G_full, F_full = np.zeros(full_shape, complex), np.zeros(full_shape, complex)
    g2_full, f2_full, p_full = (np.zeros(full_shape) for _ in range(3))
    g2_fine, f2_fine, p_fine = (np.zeros(fine_shape) for _ in range(3))
    normal_p_fine = np.zeros(fine_shape)
    rc_full = np.zeros((len(U_VALUES), len(DENSITIES), len(ANGLES)))
    rc_fine = np.zeros((len(U_VALUES), len(DENSITIES), len(FINE_ANGLES)))
    recrossings = []
    for iu, U in enumerate(U_VALUES):
        for i, n in enumerate(DENSITIES):
            state = states[(U, n, NK_PRODUCTION)]
            G, F, _, _ = fast_greens_equal_time(state, r_full, ANGLES)
            G_full[iu, i], F_full[iu, i] = G, F
            g2_full[iu, i], f2_full[iu, i], p_full[iu, i] = normalized_correlators(G, F)
            G, F, _, _ = fast_greens_equal_time(state, r_fine, FINE_ANGLES)
            g2_fine[iu, i], f2_fine[iu, i], p_fine[iu, i] = normalized_correlators(G, F)
            Gn, Fn, _, _ = fast_greens_equal_time(normals[(U, n, NK_PRODUCTION)],
                                                  r_fine, FINE_ANGLES)
            _, _, normal_p_fine[iu, i] = normalized_correlators(Gn, Fn)
            for ia, angle in enumerate(ANGLES):
                rc_full[iu, i, ia] = critical_radius(r_full, p_full[iu, i, ia])
                losses, regains = threshold_recrossings(r_full, p_full[iu, i, ia])
                recrossings.append({
                    "U_over_t": float(U), "n_elec": float(n), "theta_deg": float(angle),
                    "loss_r_over_a": losses, "regain_r_over_a": regains,
                    "max_p_after_first_loss": float(np.max(
                        p_full[iu, i, ia, r_full > losses[0]])) if losses else None,
                })
            for ia in range(len(FINE_ANGLES)):
                rc_fine[iu, i, ia] = critical_radius(r_fine, p_fine[iu, i, ia])
            print(f"  U={U:g}, n={n:.3f}: rc0={rc_fine[iu,i,0]:.8f}, "
                  f"rc45={rc_fine[iu,i,1]:.8f}", flush=True)

    print("Stage 4/6: exact lattice sites and convergence", flush=True)
    nsite = len(SITES)
    site_shape = (len(U_VALUES), len(DENSITIES), nsite)
    site_g2, site_f2, site_p = (np.zeros(site_shape) for _ in range(3))
    normal_site_p = np.zeros(site_shape)
    for iu, U in enumerate(U_VALUES):
        for i, n in enumerate(DENSITIES):
            _, _, G, F, _, _ = exact_site_greens(states[(U, n, NK_PRODUCTION)])
            g2, f2, p = normalized_correlators(G, F)
            site_g2[iu, i], site_f2[iu, i], site_p[iu, i] = g2[0, 1:], f2[0, 1:], p[0, 1:]
            _, _, Gn, Fn, _, _ = exact_site_greens(normals[(U, n, NK_PRODUCTION)])
            _, _, pn = normalized_correlators(Gn, Fn)
            normal_site_p[iu, i] = pn[0, 1:]
            print(f"  U={U:g}, n={n:.3f}: p(a)={p[0,1]:.9f}, "
                  f"p(sqrt2a)={p[0,2]:.9f}", flush=True)

    r_conv = np.linspace(0.0, 1.5, 601)
    p_conv = np.zeros((len(U_VALUES), len(DENSITIES), len(NK_VALUES), len(r_conv)))
    convergence = []
    for iu, U in enumerate(U_VALUES):
        for i, n in enumerate(DENSITIES):
            for j, nk in enumerate(NK_VALUES):
                state = states[(U, n, nk)]
                G, F, _, _ = fast_greens_equal_time(state, r_conv, [45.0])
                g2, f2, p = normalized_correlators(G, F)
                p_conv[iu, i, j] = p[0]
                rc = critical_radius(r_conv, p[0])
                _, _, Gs, Fs, _, _ = exact_site_greens(state)
                gs2, fs2, ps = normalized_correlators(Gs, Fs)
                row = next(r for r in records if r["U_over_t"] == U and
                           r["n_elec"] == n and r["Nk"] == nk)
                convergence.append({
                    "U_over_t": float(U), "n_elec": float(n), "Nk": nk,
                    "branch": row["branch"], "paired": row["paired"],
                    "Delta0_over_t": row["Delta0_over_t"], "DeltaS_over_t": row["DeltaS_over_t"],
                    "mu_over_t": row["mu_over_t"],
                    "max_abs_Delta_ks_over_t": row["max_abs_Delta_ks_over_t"],
                    "max_abs_residual": row["max_abs_residual"],
                    "exact_filling_on_mesh": row["exact_filling_on_mesh"],
                    "critical_r_over_a_theta45": rc,
                    "nearest_site_g2": float(gs2[0, 1]), "nearest_site_f2": float(fs2[0, 1]),
                    "nearest_site_p": float(ps[0, 1]), "diagonal_site_p": float(ps[0, 2]),
                    "maximum_tested_site_p": float(np.max(ps[0, 1:])),
                    "G0_simpson_abs": float(abs(G[0, 0])),
                    "G0_simpson_relative_error": float(abs(abs(G[0, 0])-n/2)/(n/2)),
                })

    print("Stage 5/6: physical protocol validation", flush=True)
    p_samples = list(p_fine[:, :, :, ::100].ravel()) + list(site_p.ravel())
    protocol = protocol_validation(p_samples)
    if max(protocol[k] for k in ("fixed", "optimized", "concurrence")) > 2e-12:
        raise RuntimeError(f"Protocol identity failure: {protocol}")
    if protocol["minimum_eigenvalue"] < -1e-12:
        raise RuntimeError("Negative resource eigenvalue")

    print("Stage 6/6: arrays, summary, and figures", flush=True)
    production = [r for r in records if r["Nk"] == NK_PRODUCTION]
    spatial_results = []
    for iu, U in enumerate(U_VALUES):
        for i, n in enumerate(DENSITIES):
            rec_rows = [r for r in recrossings if r["U_over_t"] == U and r["n_elec"] == n]
            spatial_results.append({
                "U_over_t": float(U), "n_elec": float(n),
                "critical_r_over_a_theta0": float(rc_fine[iu, i, 0]),
                "critical_r_over_a_theta45": float(rc_fine[iu, i, 1]),
                "critical_r_over_a_min_over_angles": float(np.min(rc_full[iu, i])),
                "critical_r_over_a_max_over_angles": float(np.max(rc_full[iu, i])),
                "max_angular_p_variation": float(np.max(np.ptp(p_full[iu, i], axis=0))),
                "any_concurrence_revival_up_to_r10": bool(any(r["regain_r_over_a"] for r in rec_rows)),
                "max_p_after_first_loss_all_angles": float(max(
                    (r["max_p_after_first_loss"] or 0.0) for r in rec_rows)),
                "max_abs_paired_minus_normal_reference_p": float(maxabs(
                    p_fine[iu, i]-normal_p_fine[iu, i])),
                "nearest_site_p_normal_reference": float(normal_site_p[iu, i, 0]),
            })
    max_index = np.unravel_index(np.argmax(site_p), site_p.shape)
    summary = {
        "repository_commit": commit,
        "scope": {
            "U_over_t_values": U_VALUES.tolist(), "density_values": DENSITIES.tolist(),
            "fixed_parameters": {"Delta_t_over_t": config.Delta_t/config.t_hop,
                                 "V_SO_over_t": config.V_SO/config.t_hop,
                                 "temperature": 0.0, "t_hop": config.t_hop,
                                 "lattice_constant": 1.0},
            "production_Nk": NK_PRODUCTION, "convergence_Nk": list(NK_VALUES),
            "angles_deg": ANGLES.tolist(), "fine_angles_deg": FINE_ANGLES.tolist(),
            "exact_site_coordinates": SITES.tolist(),
            "continuous_distance_caveat": "Noninteger r/a is a Fourier interpolation, not a lattice site.",
            "gap_gauge": "DeltaS<=0; changing both gap signs leaves physical resource magnitudes invariant.",
        },
        "solutions_production": production, "all_solution_records": records,
        "multistart_checks": multistart, "spatial_results": spatial_results,
        "convergence_records": convergence, "recrossings_up_to_r10": recrossings,
        "exact_lattice_site_overview": {
            "any_above_classical": bool(np.any(site_p > P_THRESHOLD)),
            "maximum_p": float(np.max(site_p)),
            "maximum_average_fidelity": float((1+np.max(site_p))/2),
            "maximum_at_U_over_t": float(U_VALUES[max_index[0]]),
            "maximum_at_n": float(DENSITIES[max_index[1]]),
            "maximum_at_coordinate": SITES[max_index[2]].tolist(),
            "tested_site_count_per_U_and_density": nsite,
        },
        "numerical_methods": {
            "batched_gf_validation_max_abs_error": gf_error,
            "protocol_validation": protocol, "protocol_state_count": len(p_samples),
            "stability_boundary": (
                "Residual, seed, and mesh checks establish stationary numerical branches, not global free-energy minima. "
                "The tiny U=25,n=0.75 Nk=100 paired root is reported as mesh-dependent if absent on finer meshes."),
        },
        "files": {
            "data": "data/low_U_density_teleportation.npz",
            "figures": ["figs/low_U_density_bdg_solutions.png",
                        "figs/low_U_density_resources.png",
                        "figs/low_U_density_lattice_sites.png",
                        "figs/low_U_density_convergence.png"],
        },
    }
    np.savez_compressed(
        DATA / "low_U_density_teleportation.npz",
        U_values=U_VALUES, densities=DENSITIES, Nk_values=np.array(NK_VALUES),
        r_full=r_full, theta_full_deg=ANGLES, r_fine=r_fine, theta_fine_deg=FINE_ANGLES,
        G_full=G_full, F_full=F_full, g2_full=g2_full, f2_full=f2_full, p_full=p_full,
        g2_fine=g2_fine, f2_fine=f2_fine, p_fine=p_fine, normal_p_fine=normal_p_fine,
        rc_full=rc_full, rc_fine=rc_fine, sites=SITES,
        site_g2=site_g2, site_f2=site_f2, site_p=site_p, normal_site_p=normal_site_p,
        r_conv=r_conv, p_conv=p_conv,
        production_records=json_records(production), convergence_records=json_records(convergence),
        multistart_records=json_records(multistart), repository_commit=np.bytes_(commit))
    (REPORTS / "low_U_density_summary.json").write_text(json.dumps(summary, indent=2)+"\n")

    plot_solutions(records)
    plot_resources(r_fine, g2_fine, f2_fine, p_fine, rc_fine)
    plot_sites(site_g2, site_f2, site_p)
    plot_convergence(convergence)
    print(json.dumps({
        "status": "completed",
        "paired_production": [[U, n] for U in U_VALUES for n in DENSITIES
                              if states[(U, n, NK_PRODUCTION)]["DeltaS"] != 0.0],
        "exact_site_overview": summary["exact_lattice_site_overview"],
        "summary": "reports/low_U_density_summary.json",
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
