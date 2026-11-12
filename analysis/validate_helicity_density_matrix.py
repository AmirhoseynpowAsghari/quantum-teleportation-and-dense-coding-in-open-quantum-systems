#!/usr/bin/env python3
"""Validate and demonstrate generalized spin/helicity two-qubit density matrices.

This writes only new helicity-specific artifacts; archived V_SO=0 arrays are not
modified.  The nonzero-SOC case is a methodological demonstration, not a
production convergence study.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
import kspace_setup
import self_consistency
from helicity_density_matrix import (
    SINGLET,
    concurrence,
    dense_coding_metrics,
    density_at_separation,
    momentum_correlators,
    physical_diagnostics,
    resource_metrics,
)
from self_consistency import run_selfconsistency
from teleportation import average_teleportation_fidelity
from holevo import compute_holevo as repository_compute_holevo


def oh_kim_reference(N_A, N_AB, F_AB):
    """Independent scalar-singlet special case of the generalized construction."""
    n0 = N_A[0, 0]
    g2 = float(abs(N_AB[0, 0] / n0) ** 2)
    f2 = float(abs(F_AB[0, 1] / n0) ** 2)
    norm = 4.0 - 2.0 * g2 + 2.0 * f2
    return np.array([
        [1.0 - g2, 0.0, 0.0, 0.0],
        [0.0, 1.0 + f2, -(g2 + f2), 0.0],
        [0.0, -(g2 + f2), 1.0 + f2, 0.0],
        [0.0, 0.0, 0.0, 1.0 - g2],
    ], dtype=complex) / norm


def solve(vso: float, nk: int = 101):
    config.V_SO = float(vso)
    kspace_setup.V_SO = float(vso)
    self_consistency.V_SO = float(vso)
    config.Nk = int(nk)
    kspace_setup.Nk = int(nk)
    self_consistency.Nk = int(nk)
    bdg = run_selfconsistency(seed=42, U_override=75.0, kBT_override=0.0,
                              verbose=False)
    residual = float(np.max(np.abs(bdg["residuals"])))
    if not bdg["converged"] or residual > 1e-8:
        raise RuntimeError(f"Unreliable V_SO={vso} solution: residual={residual}")
    return bdg, residual


def main():
    # Algebraic checks independent of the microscopic solver.
    bell_errors = []
    bell_vectors = [
        np.array([1, 0, 0, 1], complex) / np.sqrt(2),
        np.array([1, 0, 0, -1], complex) / np.sqrt(2),
        np.array([0, 1, 1, 0], complex) / np.sqrt(2),
        SINGLET,
    ]
    for vector in bell_vectors:
        bell = np.outer(vector, vector.conj())
        bell_errors.append(abs(concurrence(bell) - 1.0))
        bell_errors.append(abs(dense_coding_metrics(bell)["holevo_pauli_bits"] - 2.0))
    product = np.zeros((4, 4), dtype=complex)
    product[0, 0] = 1.0
    product_holevo_error = abs(dense_coding_metrics(product)["holevo_pauli_bits"] - 1.0)
    werner_errors = []
    for p in np.linspace(0.0, 1.0, 21):
        rho = (1.0 - p) * np.eye(4) / 4.0 + p * np.outer(SINGLET, SINGLET.conj())
        metrics = resource_metrics(rho)
        werner_errors.append(abs(metrics["concurrence"] - max(0.0, (3.0 * p - 1.0) / 2.0)))
        werner_errors.append(abs(metrics["fully_entangled_fraction"] - (1.0 + 3.0 * p) / 4.0))
        werner_errors.append(abs(metrics["optimized_average_fidelity"] - (1.0 + p) / 2.0))
        lambda_s = (1.0 + 3.0 * p) / 4.0
        lambda_t = (1.0 - p) / 4.0
        entropy = 0.0
        for eigenvalue, multiplicity in ((lambda_s, 1), (lambda_t, 3)):
            if eigenvalue > 0.0:
                entropy -= multiplicity * eigenvalue * np.log2(eigenvalue)
        expected_chi = 2.0 - entropy
        werner_errors.append(abs(
            dense_coding_metrics(rho)["holevo_pauli_bits"] - expected_chi))

    vso_values = np.array([0.0, 0.5])
    r_vectors = np.array([[0, 0], [1, 0], [1, 1], [2, 0], [0, 2], [3, 1]], dtype=int)
    records = []
    arrays = {key: [] for key in (
        "rho_spin", "rho_band", "weight_spin", "weight_band",
        "N_spin", "N_band", "F_spin", "F_band",
        "concurrence_spin", "concurrence_band", "fef_spin", "fef_band",
        "fixed_fidelity_spin", "fixed_fidelity_band",
        "optimized_fidelity_spin", "optimized_fidelity_band",
        "holevo_spin", "holevo_band", "entropy_AB_spin", "entropy_AB_band",
        "entropy_B_spin", "entropy_B_band",
    )}
    solutions = {}
    max_oh_kim_error = 0.0
    max_r0_singlet_error = 0.0
    max_trace_error = 0.0
    max_hermiticity_error = 0.0
    minimum_eigenvalue = np.inf
    max_inversion_normal_error = 0.0
    max_inversion_pair_error = 0.0
    fixed_protocol_error = 0.0
    max_pauli_average_error = 0.0
    max_holevo_crosscheck_error = 0.0
    holevo_min = np.inf
    holevo_max = -np.inf

    for iu, vso in enumerate(vso_values):
        bdg, residual = solve(float(vso))
        solutions[str(float(vso))] = {
            "U_over_t": 75.0, "Delta_t_over_t": 4.5, "n_elec": 1.875,
            "Nk": 101, "temperature": 0.0, "max_abs_residual": residual,
            "Delta0_over_t": float(bdg["Delta0"]),
            "DeltaS_over_t": float(bdg["DeltaS"]),
            "mu_over_t": float(bdg["mu"]),
        }
        N_band_k, F_band_k = momentum_correlators(bdg, basis="band")
        N_reverse = N_band_k[::-1, ::-1]
        F_reverse = F_band_k[::-1, ::-1]
        off_trim = bdg["sqrt_sin"] > 1e-14
        max_inversion_normal_error = max(
            max_inversion_normal_error, float(np.max(np.abs(N_band_k - N_reverse))))
        max_inversion_pair_error = max(
            max_inversion_pair_error,
            float(np.max(np.abs((F_band_k + F_reverse)[off_trim]))))

        per_basis = {basis: {key: [] for key in arrays if key.endswith("_" + basis)}
                     for basis in ("spin", "band")}
        for r in r_vectors:
            for basis in ("spin", "band"):
                result = density_at_separation(bdg, r, basis=basis)
                rho = result["rho"]
                diagnostics = physical_diagnostics(rho)
                metrics = resource_metrics(rho)
                dense = dense_coding_metrics(rho)
                repository_chi, _, _ = repository_compute_holevo(rho)
                max_holevo_crosscheck_error = max(
                    max_holevo_crosscheck_error,
                    abs(repository_chi - dense["holevo_pauli_bits"]))
                max_pauli_average_error = max(
                    max_pauli_average_error, dense["pauli_average_identity_error"])
                holevo_min = min(holevo_min, dense["holevo_pauli_bits"])
                holevo_max = max(holevo_max, dense["holevo_pauli_bits"])
                max_trace_error = max(max_trace_error, diagnostics["trace_error"])
                max_hermiticity_error = max(max_hermiticity_error,
                                            diagnostics["hermiticity_error"])
                minimum_eigenvalue = min(minimum_eigenvalue,
                                         diagnostics["minimum_eigenvalue"])
                r0_error = float(np.max(np.abs(rho - np.outer(SINGLET, SINGLET.conj()))))
                if np.all(r == 0):
                    max_r0_singlet_error = max(max_r0_singlet_error, r0_error)
                if vso == 0.0 and basis == "spin":
                    reference = oh_kim_reference(result["N_A"], result["N_AB"],
                                                 result["F_AB"])
                    max_oh_kim_error = max(max_oh_kim_error,
                                           float(np.max(np.abs(rho - reference))))
                values = {
                    f"rho_{basis}": rho,
                    f"weight_{basis}": result["weight"],
                    f"N_{basis}": result["N_AB"],
                    f"F_{basis}": result["F_AB"],
                    f"concurrence_{basis}": metrics["concurrence"],
                    f"fef_{basis}": metrics["fully_entangled_fraction"],
                    f"fixed_fidelity_{basis}": metrics["fixed_reference_average_fidelity"],
                    f"optimized_fidelity_{basis}": metrics["optimized_average_fidelity"],
                    f"holevo_{basis}": dense["holevo_pauli_bits"],
                    f"entropy_AB_{basis}": dense["entropy_AB_bits"],
                    f"entropy_B_{basis}": dense["entropy_B_bits"],
                }
                for key, value in values.items():
                    per_basis[basis][key].append(value)
                records.append({
                    "V_SO_over_t": float(vso), "basis": basis,
                    "r_vector": r.tolist(), "weight": result["weight"],
                    **diagnostics, **metrics, **dense,
                    "same_site_singlet_error": r0_error if np.all(r == 0) else None,
                })
                if vso == 0.5 and basis == "spin" and np.array_equal(r, [1, 0]):
                    explicit = average_teleportation_fidelity(rho)
                    fixed_protocol_error = abs(
                        explicit["average_fidelity"] - metrics["fixed_reference_average_fidelity"])
        for basis in ("spin", "band"):
            for key, values in per_basis[basis].items():
                arrays[key].append(values)

    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    checks = {
        "maximum_bell_concurrence_error": max(bell_errors),
        "maximum_werner_metric_error": max(werner_errors),
        "maximum_zero_SOC_Oh_Kim_matrix_error": max_oh_kim_error,
        "maximum_same_site_singlet_error": max_r0_singlet_error,
        "maximum_trace_error": max_trace_error,
        "maximum_hermiticity_error": max_hermiticity_error,
        "minimum_density_eigenvalue": float(minimum_eigenvalue),
        "maximum_normal_inversion_error": max_inversion_normal_error,
        "maximum_anomalous_inversion_error": max_inversion_pair_error,
        "fixed_protocol_identity_error": fixed_protocol_error,
        "product_state_holevo_error": product_holevo_error,
        "maximum_pauli_average_identity_error": max_pauli_average_error,
        "maximum_holevo_repository_crosscheck_error": max_holevo_crosscheck_error,
        "minimum_holevo_pauli_bits": float(holevo_min),
        "maximum_holevo_pauli_bits": float(holevo_max),
    }
    passed = (
        checks["maximum_bell_concurrence_error"] < 1e-10
        and checks["maximum_werner_metric_error"] < 1e-10
        and checks["maximum_zero_SOC_Oh_Kim_matrix_error"] < 1e-9
        and checks["maximum_same_site_singlet_error"] < 1e-9
        and checks["maximum_trace_error"] < 1e-10
        and checks["maximum_hermiticity_error"] < 1e-10
        and checks["minimum_density_eigenvalue"] > -1e-10
        and checks["maximum_normal_inversion_error"] < 1e-12
        and checks["maximum_anomalous_inversion_error"] < 1e-12
        and checks["fixed_protocol_identity_error"] < 1e-10
        and checks["product_state_holevo_error"] < 1e-12
        and checks["maximum_pauli_average_identity_error"] < 1e-12
        and checks["maximum_holevo_repository_crosscheck_error"] < 1e-10
        and checks["minimum_holevo_pauli_bits"] >= 0.0
        and checks["maximum_holevo_pauli_bits"] <= 2.0
    )
    report = {
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "Generalized conditional two-qubit matrices; methodological demo, not a production convergence study",
        "source": "Hutchinson--Hirsch--Marsiglio arXiv:1801.08988 Eqs. (16)--(29)",
        "parameters": solutions,
        "state_count": len(records),
        "checks": checks,
        "checks_passed": bool(passed),
        "records": records,
        "interpretation": {
            "spin": "Original-spin conditional state; ordinary spin-qubit resource",
            "band": "Helicity-band-lattice pseudospin state; operations are band-pseudospin operations, not automatically physical spin operations",
            "fixed_vs_optimized": "Fixed singlet-reference and locally optimized teleportation are distinct for general non-Werner matrices",
            "dense_coding": "Equal-probability Pauli-ensemble Holevo quantity in the selected spin or band-pseudospin basis; not a general optimized capacity or accessible-information result",
        },
    }
    REPORTS = ROOT / "reports"
    DATA = ROOT / "data"
    REPORTS.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    (REPORTS / "helicity_density_matrix_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        DATA / "helicity_density_matrix_demo.npz",
        U_over_t=np.array([75.0, 75.0]), V_SO_over_t=vso_values,
        r_vectors=r_vectors, **arrays,
    )
    print(json.dumps({"checks": checks, "checks_passed": passed,
                      "state_count": len(records),
                      "data": "data/helicity_density_matrix_demo.npz"}, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
