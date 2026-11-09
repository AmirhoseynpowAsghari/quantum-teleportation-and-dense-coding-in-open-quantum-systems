#!/usr/bin/env python3
"""Independent validation of the U/t=15 and 25 density-study artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    checks = []

    def record(name, condition, detail=None):
        checks.append({"check": name, "passed": bool(condition), "detail": detail})

    data = np.load(ROOT / "data/low_U_density_teleportation.npz")
    summary = json.loads((ROOT / "reports/low_U_density_summary.json").read_text())
    U_values, densities = data["U_values"], data["densities"]

    for name in ("G_full", "F_full"):
        record(f"{name} finite", np.all(np.isfinite(data[name])))
    for name in ("g2_full", "f2_full", "p_full", "g2_fine", "f2_fine", "p_fine",
                 "normal_p_fine", "site_g2", "site_f2", "site_p", "normal_site_p", "p_conv"):
        array = data[name]
        record(f"{name} finite", np.all(np.isfinite(array)))
        record(f"{name} in physical range", np.min(array) >= -2e-12 and np.max(array) <= 1+2e-12,
               {"min": float(np.min(array)), "max": float(np.max(array))})

    for stem in ("full", "fine"):
        g2, f2, p = data[f"g2_{stem}"], data[f"f2_{stem}"], data[f"p_{stem}"]
        error = float(np.max(np.abs(p-(g2+f2)/(2+f2-g2))))
        record(f"Werner identity {stem}", error < 3e-14, error)
        record(f"Origin singlet {stem}", np.max(np.abs(p[:, :, :, 0]-1)) < 3e-14)
    error = float(np.max(np.abs(data["site_p"]-
        (data["site_g2"]+data["site_f2"])/(2+data["site_f2"]-data["site_g2"]))))
    record("Exact-site Werner identity", error < 3e-14, error)
    record("No tested nonzero-site advantage",
           not np.any(data["site_p"] > 1/3),
           {"maximum_p": float(np.max(data["site_p"])),
            "maximum_Ftel": float((1+np.max(data["site_p"]))/2)})
    record("No exact-site advantage on any convergence mesh",
           all(r["maximum_tested_site_p"] <= 1/3 for r in summary["convergence_records"]))
    record("Strongest lower-U site is U=25,n=0.5,(1,0)",
           np.unravel_index(np.argmax(data["site_p"]), data["site_p"].shape) == (1, 0, 0))

    for a, b in ((0, 6), (1, 5), (2, 4)):
        record(f"Square-lattice symmetry {a}/{b}",
               np.max(np.abs(data["p_full"][:, :, a]-data["p_full"][:, :, b])) < 2e-12 and
               np.max(np.abs(data["G_full"][:, :, a]-data["G_full"][:, :, b])) < 2e-12)

    production = summary["solutions_production"]
    record("Production residual tolerance",
           all(r["max_abs_residual"] < 1e-8 for r in production),
           max(r["max_abs_residual"] for r in production))
    record("Production filling matches target",
           all(abs(r["discrete_filling_from_v2"]-r["n_elec"]) < 2e-10 for r in production))
    record("U=15 paired at every production density",
           all(r["paired"] for r in production if r["U_over_t"] == 15.0))
    record("U=25 production zero gap only at n=0.5 and 0.75",
           [r["n_elec"] for r in production if r["U_over_t"] == 25.0 and not r["paired"]] == [0.5, 0.75])
    record("U=25 zero-gap anomalous functions vanish",
           np.max(np.abs(data["F_full"][1, :2])) == 0.0 and
           np.max(np.abs(data["site_f2"][1, :2])) == 0.0)
    record("U=15 anomalous functions nonzero at every density",
           all(np.max(np.abs(data["F_full"][0, i])) > 1e-6 for i in range(len(densities))))
    record("Gap gauge DeltaS nonpositive on paired branches",
           all(r["DeltaS_over_t"] <= 0.0 for r in production if r["paired"]))
    record("Global stability limitation explicit",
           all(not r["global_thermodynamic_stability_established"] for r in production))

    multi = summary["multistart_checks"]
    record("U=15 six-seed paired roots at every density",
           all(r["paired_root_count"] == 6 for r in multi if r["U_over_t"] == 15.0))
    record("U=25 six-seed result by density",
           all(r["paired_root_count"] == (0 if r["n_elec"] == 0.5 else 6)
               for r in multi if r["U_over_t"] == 25.0))
    record("Multistart canonical solutions reproducible",
           all(r["paired_root_count"] == 0 or
               max(r["canonical_Delta0_span"], r["canonical_DeltaS_span"], r["mu_span"]) < 1e-9
               for r in multi))
    mesh_dependent = [r for r in summary["convergence_records"]
                      if r["U_over_t"] == 25.0 and r["n_elec"] == 0.75]
    record("U=25,n=0.75 tiny coarse-mesh root explicitly mesh dependent",
           mesh_dependent[0]["paired"] and all(not r["paired"] for r in mesh_dependent[1:]))

    convergence = summary["convergence_records"]
    record("Convergence has every U,n,Nk combination", len(convergence) == 2*len(densities)*4)
    rc_changes, p_changes = {}, {}
    for U in U_values:
        for n in densities:
            rows = [r for r in convergence if r["U_over_t"] == U and r["n_elec"] == n and r["Nk"] in (400, 800)]
            key = f"U={U:g},n={n:g}"
            rc_changes[key] = abs(rows[0]["critical_r_over_a_theta45"]-rows[1]["critical_r_over_a_theta45"])
            p_changes[key] = abs(rows[0]["nearest_site_p"]-rows[1]["nearest_site_p"])
    record("Critical-radius 400-to-800 change below 0.0034a",
           max(rc_changes.values()) < 0.0034, rc_changes)
    record("Nearest-site p 400-to-800 change below 0.002",
           max(p_changes.values()) < 0.002, p_changes)
    record("U=15,n=0.5 gap mesh sensitivity retained",
           next(r for r in convergence if r["U_over_t"] == 15 and r["n_elec"] == 0.5 and r["Nk"] == 800)["max_abs_Delta_ks_over_t"] < 0.62)

    # Independent first-down-crossing interpolation.
    errors = []
    for iu in range(len(U_values)):
        for i in range(len(densities)):
            for ia in range(2):
                p, r = data["p_fine"][iu, i, ia], data["r_fine"]
                k = np.where((p[:-1] > 1/3) & (p[1:] <= 1/3))[0][0]
                rc = r[k]+(1/3-p[k])/(p[k+1]-p[k])*(r[k+1]-r[k])
                errors.append(abs(rc-data["rc_fine"][iu, i, ia]))
    record("Independent critical-radius interpolation", max(errors) < 2e-14, max(errors))
    record("No continuous concurrence regain after first loss",
           not any(r["regain_r_over_a"] for r in summary["recrossings_up_to_r10"]))
    record("All post-loss maxima below threshold",
           all(r["max_p_after_first_loss"] is None or r["max_p_after_first_loss"] < 1/3
               for r in summary["recrossings_up_to_r10"]))

    methods = summary["numerical_methods"]
    record("Batched GF agrees with source implementation",
           methods["batched_gf_validation_max_abs_error"] < 2e-13,
           methods["batched_gf_validation_max_abs_error"])
    protocol = methods["protocol_validation"]
    record("Explicit teleportation protocol identities",
           max(protocol["fixed"], protocol["optimized"], protocol["concurrence"]) < 2e-12,
           protocol)
    record("Sampled states positive and normalized",
           protocol["minimum_eigenvalue"] > -1e-12 and protocol["maximum_trace_error"] < 2e-12)
    for filename in summary["files"]["figures"]:
        path = ROOT / filename
        record(f"Figure exists: {filename}", path.is_file() and path.stat().st_size > 10000)
    record("Data artifact exists", (ROOT / summary["files"]["data"]).is_file())

    report = {"status": "passed" if all(c["passed"] for c in checks) else "failed",
              "check_count": len(checks), "passed_count": sum(c["passed"] for c in checks),
              "checks": checks}
    (ROOT / "reports/low_U_density_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "check_count", "passed_count")}, indent=2))
    if report["status"] != "passed":
        for check in checks:
            if not check["passed"]:
                print("FAILED:", check)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
