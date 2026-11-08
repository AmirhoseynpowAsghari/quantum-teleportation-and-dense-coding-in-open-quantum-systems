#!/usr/bin/env python3
"""Validate saved microscopic-study artifacts and sampled physical states."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from evolution import evolve_rho
from fidelity import _evolve_rho_fid
from teleportation import resource_metrics

DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def main():
    checks = []

    def record(name, condition, detail=None):
        checks.append({"check": name, "passed": bool(condition),
                       "detail": detail})

    initial = np.load(DATA / "microscopic_teleportation_initial.npz")
    dynamics = np.load(DATA / "microscopic_teleportation_dynamics.npz")
    convergence = np.load(DATA / "microscopic_teleportation_convergence.npz")
    boundary = np.load(DATA / "microscopic_revival_boundary.npz")
    summary = json.loads((REPORTS / "microscopic_teleportation_summary.json").read_text())
    boundary_summary = json.loads((REPORTS / "microscopic_revival_boundary.json").read_text())

    for label in ("x_line", "y_line", "p_line", "x_full", "y_full", "p_full",
                  "site_p", "site_x", "site_y"):
        record(f"initial {label} finite", np.all(np.isfinite(initial[label])))
    for label in ("x_line", "y_line", "p_line", "x_full", "y_full", "p_full",
                  "site_p", "site_x", "site_y"):
        record(f"initial {label} in [0,1]",
               np.min(initial[label]) >= -1e-12 and np.max(initial[label]) <= 1+1e-12,
               {"min": float(np.min(initial[label])),
                "max": float(np.max(initial[label]))})
    for stem in ("line", "full"):
        x, y, p = initial[f"x_{stem}"], initial[f"y_{stem}"], initial[f"p_{stem}"]
        reconstructed = (x+y)/(2+y-x)
        record(f"Werner p identity {stem}", np.max(np.abs(p-reconstructed)) < 2e-15,
               float(np.max(np.abs(p-reconstructed))))
    record("Exact lattice sites below Werner threshold",
           np.max(initial["site_p"]) < 1/3,
           float(np.max(initial["site_p"])))

    array_names = ("low_metrics", "high_metrics", "dephasing_independent",
                   "dephasing_common", "U_dynamics", "map_metrics",
                   "markovian_metrics", "exponential_metrics")
    for name in array_names:
        array = dynamics[name]
        record(f"{name} finite", np.all(np.isfinite(array)))
        record(f"{name} metric range", np.min(array) >= -2e-10 and np.max(array) <= 1+2e-10,
               {"min": float(np.min(array)), "max": float(np.max(array))})
        record(f"{name} optimized not below fixed",
               np.max(array[..., 2]-array[..., 4]) < 2e-8,
               float(np.max(array[..., 2]-array[..., 4])))
        above = array[..., 4] > 2/3 + 1e-10
        record(f"{name} above-threshold implies nonzero concurrence",
               not np.any(above & (array[..., 0] <= 1e-10)))

    for name in ("low_metrics", "high_metrics", "U_dynamics", "map_metrics"):
        initial_metrics = dynamics[name][..., 0, :] if name != "map_metrics" else dynamics[name][:, 0, :]
        fs = initial_metrics[..., 1]
        expected_f = (1+2*fs)/3
        expected_c = np.maximum(0, 2*fs-1)
        record(f"{name} initial Werner fixed fidelity",
               np.max(np.abs(initial_metrics[..., 2]-expected_f)) < 3e-12,
               float(np.max(np.abs(initial_metrics[..., 2]-expected_f))))
        record(f"{name} initial Werner optimized fidelity",
               np.max(np.abs(initial_metrics[..., 4]-expected_f)) < 3e-12,
               float(np.max(np.abs(initial_metrics[..., 4]-expected_f))))
        record(f"{name} initial Werner concurrence",
               np.max(np.abs(initial_metrics[..., 0]-expected_c)) < 3e-12,
               float(np.max(np.abs(initial_metrics[..., 0]-expected_c))))

    common = dynamics["dephasing_common"]
    record("Common dephasing preserves tested Werner resources",
           np.max(np.abs(common-common[:, [0], :])) == 0.0,
           float(np.max(np.abs(common-common[:, [0], :]))))
    record("Independent dephasing has no optimized regain",
           all(not item["regain_times_after_a_loss"]
               for bath in ("independent",)
               for item in summary["threshold_events_dephasing_optimized"][bath].values()))
    record("Markovian overdamped control has no optimized regain",
           all(not item["regain_times_after_a_loss"]
               for item in summary["threshold_events_markovian_optimized_Gamma_over_gamma_100"].values()))
    record("Exact exponential control has no optimized regain",
           all(not item["regain_times_after_a_loss"]
               for item in summary["threshold_events_exact_exponential_optimized"].values()))
    record("High-noise r=0.30 has demonstrated regain",
           bool(summary["threshold_events_amplitude"]["high"]["0.3"]["optimized"]["regain_times_after_a_loss"]))
    record("High-noise r=0.40 has no demonstrated regain",
           not summary["threshold_events_amplitude"]["high"]["0.4"]["optimized"]["regain_times_after_a_loss"])
    record("High-noise initially separable r=0.50 never above",
           not summary["threshold_events_amplitude"]["high"]["0.5"]["optimized"]["ever_above"])
    record("High-noise initially separable r=1 never above",
           not summary["threshold_events_amplitude"]["high"]["1.0"]["optimized"]["ever_above"])

    conv_records = [json.loads(x) for x in convergence["convergence_records"]]
    for U in (40.0, 75.0):
        paired = [r for r in conv_records if r["U_over_t"] == U and r["paired_solution"]]
        span = max(r["critical_r_over_a"] for r in paired)-min(r["critical_r_over_a"] for r in paired)
        record(f"Paired U={U} critical radius Nk convergence <0.001", span < 0.001, span)
    record("Zero-gap U=115 Nk=400 explicitly labeled",
           any(r["U_over_t"] == 115 and r["Nk"] == 400 and not r["paired_solution"]
               for r in conv_records))
    analytic_errors = summary["mathematical_identities"]["initial_protocol_validation_max_abs_errors"]
    record("Analytic protocol errors below 1e-12", max(analytic_errors.values()) < 1e-12,
           analytic_errors)
    record("Underdamped implementations agree",
           summary["underdamped_main_vs_exact_continuation_max_matrix_error"] < 1e-12)
    record("Dephasing envelope monotonic", summary["dephasing_envelope_monotonic"])
    record("Revival boundary bracket matches refinement",
           boundary_summary["maximum_sampled_reviving_r_over_a"] == 0.3875 and
           boundary_summary["minimum_sampled_nonreviving_r_over_a"] == 0.39,
           boundary_summary["boundary_bracket_over_a"])
    record("Refined exact sites below threshold at Nk=400",
           not boundary_summary["any_exact_lattice_site_above_classical"],
           boundary_summary["maximum_exact_lattice_site_p"])
    record("Boundary arrays finite", np.all(np.isfinite(boundary["curves"])))

    # Physical-state checks at representative evolved times, including full
    # conditional Bell-output validation rather than only array-range checks.
    point_rho = dynamics["point_rho"]
    t = dynamics["t"]
    min_eigen = np.inf
    max_trace_error = 0.0
    max_hermiticity_error = 0.0
    max_probability_error = 0.0
    sample_count = 0
    for ip in (0, 2, 5):
        rho0 = point_rho[ip]
        for regime, parameters in (("low", {"gammaA": 1.5, "GammaA": 0.1,
                                             "gammaB": 1.5, "GammaB": 0.1}),
                                   ("high", {"gammaA": 4.5, "GammaA": 0.01,
                                             "gammaB": 4.5, "GammaB": 0.01})):
            for target in (0.0, 3.1, 20.9, 63.0):
                time = float(t[np.argmin(np.abs(t-target))])
                rho = evolve_rho(rho0, time, kind="amp", bath="independent", **parameters)
                metrics = resource_metrics(rho, validate=True)
                diag = metrics["diagnostics"]
                min_eigen = min(min_eigen, diag["minimum_output_eigenvalue"])
                max_trace_error = max(max_trace_error, diag["maximum_trace_error"])
                max_hermiticity_error = max(max_hermiticity_error,
                                            diag["maximum_hermiticity_error"])
                max_probability_error = max(max_probability_error,
                                            diag["maximum_probability_sum_error"])
                sample_count += 1
        for time in (0.1, 1.5):
            rho = _evolve_rho_fid(
                rho0, time, kind="amp", bath="independent",
                nuA=0.2, GammaA=450.0, nuB=0.2, GammaB=450.0,
                gammaA=4.5, GammaAD_A=450.0, gammaB=4.5, GammaAD_B=450.0)
            metrics = resource_metrics(rho, validate=True)
            diag = metrics["diagnostics"]
            min_eigen = min(min_eigen, diag["minimum_output_eigenvalue"])
            max_trace_error = max(max_trace_error, diag["maximum_trace_error"])
            max_hermiticity_error = max(max_hermiticity_error,
                                        diag["maximum_hermiticity_error"])
            max_probability_error = max(max_probability_error,
                                        diag["maximum_probability_sum_error"])
            sample_count += 1
    record("Sampled conditional outputs positive", min_eigen >= -1e-10, min_eigen)
    record("Sampled conditional output traces", max_trace_error < 1e-10, max_trace_error)
    record("Sampled conditional outputs Hermitian", max_hermiticity_error < 1e-10,
           max_hermiticity_error)
    record("Sampled Bell probabilities normalized", max_probability_error < 1e-10,
           max_probability_error)

    report = {
        "status": "passed" if all(c["passed"] for c in checks) else "failed",
        "check_count": len(checks), "passed_count": sum(c["passed"] for c in checks),
        "physical_state_samples": sample_count, "checks": checks,
    }
    (REPORTS / "microscopic_study_validation.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("status", "check_count", "passed_count", "physical_state_samples")},
                     indent=2))
    if report["status"] != "passed":
        for check in checks:
            if not check["passed"]:
                print("FAILED:", check)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
