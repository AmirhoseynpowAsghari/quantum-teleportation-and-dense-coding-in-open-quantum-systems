#!/usr/bin/env python3
"""Refine the first high-noise revival boundary and exact-site controls.

Run after analysis/run_microscopic_teleportation_study.py. All physics routines
are reused through that study's wrappers and the original repository modules.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.run_microscopic_teleportation_study import (
    CLASSICAL_BENCHMARK, METRICS, NOISE, SITES, DATA, FIGS, REPORTS,
    evolve_metrics, greens_on_grid, normalized_correlators, solve_bdg,
    threshold_events)
from density_matrix import build_density_matrices


def main():
    bdg, residual = solve_bdg(75, 400)
    radii = np.arange(0.36, 0.40001, 0.0025)
    t = np.linspace(0.0, 25.0, 1001)
    curves = np.zeros((len(radii), len(t), len(METRICS)))
    initial_p = np.zeros(len(radii))
    for ir, radius in enumerate(radii):
        _, _, G, F = greens_on_grid(bdg, float(radius), 2, (45.0,))
        x, y, p = normalized_correlators(G, F)
        initial_p[ir] = p[0, 1]
        rho = np.asarray(build_density_matrices(G, F))[0, 1]
        curves[ir] = evolve_metrics(rho, t, NOISE["high"])
        print(f"r={radius:.4f}", flush=True)

    events = []
    for ir, radius in enumerate(radii):
        event = threshold_events(t, curves[ir, :, 4])
        event["r_over_a"] = float(radius)
        event["initial_p"] = float(initial_p[ir])
        events.append(event)
    reviving = [event for event in events if event["regain_times_after_a_loss"]]
    nonreviving = [event for event in events if not event["regain_times_after_a_loss"]]
    max_reviving = max(event["r_over_a"] for event in reviving)
    min_nonreviving = min(event["r_over_a"] for event in nonreviving)

    # Exact lattice sites at the production momentum resolution.
    site_results = []
    for x_site, y_site in SITES:
        radius = float(np.hypot(x_site, y_site))
        angle = float(np.degrees(np.arctan2(y_site, x_site)))
        _, _, G, F = greens_on_grid(bdg, radius, 2, (angle,))
        x, y, p = normalized_correlators(G, F)
        site_results.append({
            "site": [x_site, y_site], "r_over_a": radius, "theta_deg": angle,
            "g2": float(x[0, 1]), "f2": float(y[0, 1]),
            "p": float(p[0, 1]),
            "fixed_and_optimized_initial_fidelity": float((1 + p[0, 1]) / 2),
        })

    r_rev = np.array([event["r_over_a"] for event in reviving])
    first_regain = np.array([event["first_regain_time"] for event in reviving])
    post_loss_peak = np.array([event["maximum_after_first_loss"] for event in events])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    axes[0].plot(radii, post_loss_peak, "o-", ms=3,
                 label="Maximum after first threshold loss")
    axes[0].axhline(CLASSICAL_BENCHMARK, color="black", ls="--",
                    label=r"classical $2/3$")
    axes[0].axvspan(max_reviving, min_nonreviving, color="gray", alpha=0.2,
                    label="Boundary bracket")
    axes[0].set(xlabel=r"$r/a$", ylabel="Optimized average fidelity",
                ylim=(0.64, 0.74))
    axes[0].legend(fontsize=8)
    axes[1].plot(r_rev, first_regain, "o-")
    axes[1].axvspan(max_reviving, min_nonreviving, color="gray", alpha=0.2)
    axes[1].set(xlabel=r"$r/a$", ylabel="First regain time ($\hbar/t$ units)")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(r"First high-noise revival boundary at $U/t=75$, $N_k=400$")
    fig.savefig(FIGS / "microscopic_revival_boundary.png", dpi=210,
                bbox_inches="tight")
    plt.close(fig)

    summary = {
        "U_over_t": 75.0, "Nk": 400, "max_abs_bdg_residual": residual,
        "theta_deg": 45.0, "noise": NOISE["high"],
        "time_grid": {"min": 0.0, "max": 25.0, "points": len(t)},
        "radial_step_over_a": 0.0025,
        "maximum_sampled_reviving_r_over_a": max_reviving,
        "minimum_sampled_nonreviving_r_over_a": min_nonreviving,
        "boundary_bracket_over_a": [max_reviving, min_nonreviving],
        "events": events,
        "exact_lattice_sites_Nk400": site_results,
        "maximum_exact_lattice_site_p": max(r["p"] for r in site_results),
        "any_exact_lattice_site_above_classical": any(
            r["p"] > 1/3 for r in site_results),
        "interpretation": (
            "The bracket concerns the first high-noise above-2/3 regain after an initial loss. It is not a thermodynamic critical point and is grid-bounded."),
    }
    np.savez_compressed(
        DATA / "microscopic_revival_boundary.npz",
        radii=radii, t=t, curves=curves, initial_p=initial_p,
        metric_names=np.array(METRICS), events=np.array([json.dumps(e) for e in events]),
        exact_site_records=np.array([json.dumps(r) for r in site_results]))
    (REPORTS / "microscopic_revival_boundary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "status": "completed", "boundary_bracket": [max_reviving, min_nonreviving],
        "max_exact_site_p_Nk400": summary["maximum_exact_lattice_site_p"],
        "report": "reports/microscopic_revival_boundary.json"}, indent=2))


if __name__ == "__main__":
    main()
