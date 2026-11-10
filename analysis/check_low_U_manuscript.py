#!/usr/bin/env python3
"""Check the lower-U manuscript tables directly against the archived results.

This checks numerical transcription and threshold statements; it does not replace
LaTeX compilation or the full lower-U artifact validation suite.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manuscript.tex"
DATA = ROOT / "data" / "low_U_density_teleportation.npz"
SUMMARY = ROOT / "reports" / "low_U_density_summary.json"
REPORT = ROOT / "reports" / "low_U_manuscript_consistency.json"


def table_rows(text: str, label: str) -> list[list[float]]:
    block = text.split(r"\label{" + label + "}", 1)[1].split(r"\end{table*}", 1)[0]
    rows = []
    for line in block.splitlines():
        if "&" not in line or not line.lstrip().startswith(("15 ", "25 ")):
            continue
        cells = line.split("\\\\", 1)[0].split("&")
        rows.append([float(cell.strip()) for cell in cells])
    return rows


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    with np.load(DATA) as data:
        interactions = data["U_values"]
        densities = data["densities"]
        rc400 = data["rc_fine"][:, :, 1]
        f2_nearest = data["site_f2"][:, :, 0]
        p_sites = data["site_p"]
        p_nearest = p_sites[:, :, 0]
        site_vectors = data["sites"].tolist()

    production = summary["solutions_production"]
    convergence = summary["convergence_records"]
    solution_rows = table_rows(text, "tab:low_U_solutions")
    resource_rows = table_rows(text, "tab:low_U_resources")
    errors: list[str] = []
    if len(solution_rows) != 14:
        errors.append(f"Expected 14 solution rows, found {len(solution_rows)}")
    if len(resource_rows) != 14:
        errors.append(f"Expected 14 resource rows, found {len(resource_rows)}")

    solution_errors = []
    resource_errors = []
    rc_changes = []
    nearest_p_changes = []
    for iu, interaction in enumerate(interactions):
        for idensity, density in enumerate(densities):
            row_index = iu * len(densities) + idensity
            record = next(
                item for item in production
                if item["U_over_t"] == interaction and item["n_elec"] == density
            )
            fine = next(
                item for item in convergence
                if item["U_over_t"] == interaction
                and item["n_elec"] == density and item["Nk"] == 800
            )
            expected_solution = [
                interaction, density, record["Delta0_over_t"],
                record["DeltaS_over_t"], record["mu_over_t"],
                record["max_abs_Delta_ks_over_t"],
            ]
            expected_resource = [
                interaction, density, rc400[iu, idensity],
                fine["critical_r_over_a_theta45"], f2_nearest[iu, idensity],
                p_nearest[iu, idensity], (1.0 + p_nearest[iu, idensity]) / 2.0,
            ]
            if row_index < len(solution_rows):
                difference = np.abs(np.asarray(solution_rows[row_index]) - expected_solution)
                solution_errors.append(float(np.max(difference)))
                if np.any(difference > 5.01e-7):
                    errors.append(f"Solution transcription mismatch at row {row_index + 1}")
            if row_index < len(resource_rows):
                difference = np.abs(np.asarray(resource_rows[row_index]) - expected_resource)
                resource_errors.append(float(np.max(difference)))
                if np.any(difference > 5.01e-7):
                    errors.append(f"Resource transcription mismatch at row {row_index + 1}")
            rc_changes.append(abs(rc400[iu, idensity] - fine["critical_r_over_a_theta45"]))
            nearest_p_changes.append(abs(p_nearest[iu, idensity] - fine["nearest_site_p"]))

    max_site_p = float(np.max(p_sites))
    max_site_fidelity = (1.0 + max_site_p) / 2.0
    concurrence_sites = np.maximum(0.0, (3.0 * p_sites - 1.0) / 2.0)
    convergence_site_maxima = [float(item["maximum_tested_site_p"]) for item in convergence]
    if max_site_p >= 1.0 / 3.0 or max_site_fidelity >= 2.0 / 3.0:
        errors.append("A production exact-site state crosses the entanglement threshold")
    if np.any(concurrence_sites != 0.0):
        errors.append("A production exact-site state has nonzero concurrence")
    if max(convergence_site_maxima) >= 1.0 / 3.0:
        errors.append("A convergence-mesh exact-site state crosses the entanglement threshold")
    if max(nearest_p_changes) >= 0.002:
        errors.append("The stated 400-to-800 nearest-site p bound is incorrect")

    result = {
        "source": SOURCE.name,
        "data": str(DATA.relative_to(ROOT)),
        "summary": str(SUMMARY.relative_to(ROOT)),
        "check_type": "table transcription and threshold consistency; not a LaTeX compile",
        "solution_table_rows": len(solution_rows),
        "resource_table_rows": len(resource_rows),
        "maximum_solution_table_rounding_error": max(solution_errors, default=None),
        "maximum_resource_table_rounding_error": max(resource_errors, default=None),
        "production_nonzero_site_state_count": int(p_sites.size),
        "tested_site_vectors": site_vectors,
        "maximum_production_site_p": max_site_p,
        "maximum_production_site_average_teleportation_fidelity": max_site_fidelity,
        "maximum_production_site_concurrence": float(np.max(concurrence_sites)),
        "maximum_convergence_mesh_site_p": max(convergence_site_maxima),
        "maximum_400_to_800_critical_radius_change_over_a": max(rc_changes),
        "maximum_400_to_800_nearest_site_p_change": max(nearest_p_changes),
        "errors": errors,
        "checks_passed": not errors,
    }
    REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
