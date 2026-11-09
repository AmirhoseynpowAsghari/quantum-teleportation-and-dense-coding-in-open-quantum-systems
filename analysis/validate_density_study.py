#!/usr/bin/env python3
"""Independent checks of the saved density/teleportation study artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGS = ROOT / "figs"


def main():
    checks = []

    def record(name, condition, detail=None):
        checks.append({"check": name, "passed": bool(condition), "detail": detail})

    spatial = np.load(DATA / "density_teleportation_spatial.npz")
    convergence = np.load(DATA / "density_teleportation_convergence.npz")
    summary = json.loads((REPORTS / "density_teleportation_summary.json").read_text())
    densities = spatial["densities"]
    nk_values = convergence["Nk_values"]

    for name in ("G_full", "F_full", "F0_full", "FS_full", "G_fine", "F_fine",
                 "site_G", "site_F"):
        record(f"{name} finite", np.all(np.isfinite(spatial[name])))
    for name in ("g2_full", "f2_full", "p_full", "g2_fine", "f2_fine", "p_fine",
                 "site_g2", "site_f2", "site_p", "site_C", "site_Ftel"):
        array = spatial[name]
        record(f"{name} finite", np.all(np.isfinite(array)))
        record(f"{name} in physical range", np.min(array) >= -2e-12 and
               np.max(array) <= 1.0 + 2e-12,
               {"min": float(np.min(array)), "max": float(np.max(array))})

    for stem in ("full", "fine"):
        g2, f2, p = spatial[f"g2_{stem}"], spatial[f"f2_{stem}"], spatial[f"p_{stem}"]
        expected = (g2+f2)/(2+f2-g2)
        error = float(np.max(np.abs(p-expected)))
        record(f"Werner formula {stem}", error < 3e-14, error)
        C, F = spatial[f"C_{stem}"], spatial[f"Ftel_{stem}"]
        record(f"Concurrence formula {stem}",
               np.max(np.abs(C-np.maximum(0, (3*p-1)/2))) < 3e-14)
        record(f"Teleportation formula {stem}", np.max(np.abs(F-(1+p)/2)) < 3e-14)
        record(f"Origin singlet {stem}", np.max(np.abs(p[:, :, 0]-1)) < 3e-14 and
               np.max(np.abs(C[:, :, 0]-1)) < 3e-14 and
               np.max(np.abs(F[:, :, 0]-1)) < 3e-14)

    expected_p = (spatial["site_g2"]+spatial["site_f2"])/(
        2+spatial["site_f2"]-spatial["site_g2"])
    record("Exact-site Werner formula", np.max(np.abs(expected_p-spatial["site_p"])) < 3e-14)
    record("Exact-site concurrence formula",
           np.max(np.abs(spatial["site_C"]-np.maximum(0, (3*spatial["site_p"]-1)/2))) < 3e-14)
    record("Exact-site teleportation formula",
           np.max(np.abs(spatial["site_Ftel"]-(1+spatial["site_p"])/2)) < 3e-14)
    record("No tested nonzero lattice-site advantage",
           not np.any(spatial["site_p"] > 1/3) and np.max(spatial["site_C"]) == 0.0,
           {"maximum_p": float(np.max(spatial["site_p"])),
            "maximum_Ftel": float(np.max(spatial["site_Ftel"]))})
    record("Strongest tested site is n=0.5 nearest neighbor",
           np.unravel_index(np.argmax(spatial["site_p"]), spatial["site_p"].shape) == (0, 0))

    zero_gap = densities <= 1.5
    record("Anomalous functions vanish on selected zero-gap branches",
           np.max(np.abs(spatial["F_full"][zero_gap])) == 0.0 and
           np.max(np.abs(spatial["site_F"][zero_gap])) == 0.0)
    record("Normal-state exact-site occupancy bound",
           all(np.max(spatial["site_g2"][i]) <= (min(n, 2-n)/n)**2 + 2e-12
               for i, n in enumerate(densities) if n <= 1.5),
           {str(n): float(np.max(spatial["site_g2"][i]))
            for i, n in enumerate(densities) if n <= 1.5})

    # Square-lattice symmetry: 0<->90, 15<->75, 30<->60 degrees.
    for a, b in ((0, 6), (1, 5), (2, 4)):
        record(f"Square-lattice angular symmetry {a}/{b}",
               np.max(np.abs(spatial["p_full"][:, a]-spatial["p_full"][:, b])) < 2e-12 and
               np.max(np.abs(spatial["G_full"][:, a]-spatial["G_full"][:, b])) < 2e-12)
    record("Fixed-solution anomalous decomposition",
           np.max(np.abs(spatial["F_full"]-spatial["F0_full"]-spatial["FS_full"])) < 2e-12)

    production = summary["solutions_production"]
    record("Production residual tolerance",
           all(r["max_abs_residual"] < 1e-8 for r in production),
           max(r["max_abs_residual"] for r in production))
    record("Production discrete filling matches target",
           all(abs(r["discrete_filling_from_v2"]-r["n_elec"]) < 2e-12 for r in production))
    record("Paired branches only at n=1.75 and 1.875",
           [r["n_elec"] for r in production if r["paired"]] == [1.75, 1.875])
    record("Positive quasiparticle energies on paired branches",
           all(r["min_quasiparticle_E_over_t"] > 1e-6 for r in production if r["paired"]),
           {r["n_elec"]: r["min_quasiparticle_E_over_t"] for r in production if r["paired"]})
    record("Multistart finds no lower-density paired root",
           all(r["paired_root_count"] == 0 for r in summary["multistart_checks"] if r["n_elec"] <= 1.5))
    record("Multistart paired roots reproducible",
           all(r["paired_root_count"] == 6 and
               max(r["paired_root_max_Delta0_span"], r["paired_root_max_DeltaS_span"],
                   r["paired_root_max_mu_span"]) < 1e-10
               for r in summary["multistart_checks"] if r["n_elec"] >= 1.75))
    record("Global stability limitation explicit",
           all(not r["global_thermodynamic_stability_established"] for r in production))

    conv = summary["convergence_records"]
    record("Convergence includes all densities and meshes",
           len(conv) == len(densities)*len(nk_values) and
           set(r["Nk"] for r in conv) == set(nk_values.tolist()))
    rc400_800_spans = {}
    nearest400_800_changes = {}
    for n in densities:
        rows = [r for r in conv if r["n_elec"] == n and r["Nk"] in (400, 800)]
        rc400_800_spans[str(n)] = abs(rows[0]["critical_r_over_a_theta45"]-
                                      rows[1]["critical_r_over_a_theta45"])
        nearest400_800_changes[str(n)] = abs(rows[0]["nearest_site_p"]-rows[1]["nearest_site_p"])
    record("Critical radius 400-to-800 change below 0.0014a",
           max(rc400_800_spans.values()) < 0.0014, rc400_800_spans)
    record("Nearest-site p 400-to-800 change below 0.0008",
           max(nearest400_800_changes.values()) < 0.0008, nearest400_800_changes)
    record("All Nk=800 zero-gap fillings exact",
           all(r["exact_filling_on_mesh"] for r in conv if r["Nk"] == 800))
    record("Finite-mesh Nk=100 filling caveats retained",
           sum(not r["exact_filling_on_mesh"] for r in conv
               if r["Nk"] == 100 and r["branch"] == "zero-gap normal stationary branch") == 2)
    record("Production G0 occupancy quadrature error below 0.51 percent",
           max(r["G0_simpson_relative_error"] for r in production) < 0.0051,
           max(r["G0_simpson_relative_error"] for r in production))

    # Independently locate every first down-crossing on the saved fine grid.
    rc_errors = []
    for i in range(len(densities)):
        for j in range(2):
            p = spatial["p_fine"][i, j]
            r = spatial["r_fine"]
            indices = np.where((p[:-1] > 1/3) & (p[1:] <= 1/3))[0]
            k = indices[0]
            rc = r[k] + (1/3-p[k])/(p[k+1]-p[k])*(r[k+1]-r[k])
            rc_errors.append(abs(rc-spatial["rc_fine"][i, j]))
    record("Independent critical-radius interpolation", max(rc_errors) < 2e-14, max(rc_errors))
    record("No continuous concurrence regain after first loss",
           not any(r["regain_r_over_a"] for r in summary["recrossings_up_to_r10"]))
    record("All post-loss maxima remain below threshold",
           all(r["max_p_after_first_loss"] is None or r["max_p_after_first_loss"] < 1/3
               for r in summary["recrossings_up_to_r10"]))

    methods = summary["numerical_methods"]
    record("Batched GF agrees with repository implementation",
           methods["batched_gf_validation_max_abs_error"] < 2e-13,
           methods["batched_gf_validation_max_abs_error"])
    record("Density-matrix Werner identity",
           methods["density_matrix_werner_max_abs_error"] < 2e-12)
    protocol = methods["protocol_validation"]
    record("Explicit fixed and optimized protocol identities",
           max(protocol["fixed"], protocol["optimized"], protocol["concurrence"]) < 2e-12,
           protocol)
    record("Sampled physical states positive and normalized",
           protocol["minimum_eigenvalue"] > -1e-12 and
           protocol["maximum_trace_error"] < 2e-12)

    for filename in summary["files"]["figures"]:
        path = ROOT / filename
        record(f"Figure exists: {filename}", path.is_file() and path.stat().st_size > 10000)
    for filename in summary["files"].values():
        if isinstance(filename, str) and filename.startswith("data/"):
            record(f"Data exists: {filename}", (ROOT/filename).is_file())

    report = {
        "status": "passed" if all(c["passed"] for c in checks) else "failed",
        "check_count": len(checks), "passed_count": sum(c["passed"] for c in checks),
        "checks": checks,
    }
    (REPORTS / "density_study_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "check_count", "passed_count")}, indent=2))
    if report["status"] != "passed":
        for check in checks:
            if not check["passed"]:
                print("FAILED:", check)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
