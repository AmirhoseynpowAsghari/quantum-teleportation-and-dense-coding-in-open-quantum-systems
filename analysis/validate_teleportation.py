#!/usr/bin/env python3
"""Validation tests for explicit and optimized teleportation.

Run from repository root:
    .venv/bin/python analysis/validate_teleportation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from observables import concurrence_xstate, singlet_fraction
from teleportation import (
    CLASSICAL_BENCHMARK,
    average_teleportation_fidelity,
    fully_entangled_fraction,
    input_spin_state,
    optimized_teleportation_fidelity,
    singlet_state,
    six_input_states,
    teleport_resource,
)

REPORT = ROOT / "reports" / "teleportation_validation.json"


def projector(state):
    return np.outer(state, state.conj())


def assert_close(actual, expected, label, records, atol=2e-10):
    error = float(abs(actual - expected))
    records.append({"test": label, "actual": float(actual),
                    "expected": float(expected), "absolute_error": error,
                    "tolerance": atol, "passed": bool(error <= atol)})
    if error > atol:
        raise AssertionError(f"{label}: {actual} != {expected} (error {error})")


def check_diagnostics(result, label, records):
    diag = result["diagnostics"]
    checks = {
        "minimum_output_eigenvalue": diag["minimum_output_eigenvalue"],
        "maximum_hermiticity_error": diag["maximum_hermiticity_error"],
        "maximum_trace_error": diag["maximum_trace_error"],
        "maximum_probability_sum_error": diag["maximum_probability_sum_error"],
    }
    passed = (checks["minimum_output_eigenvalue"] >= -1e-10
              and checks["maximum_hermiticity_error"] <= 1e-10
              and checks["maximum_trace_error"] <= 1e-10
              and checks["maximum_probability_sum_error"] <= 1e-10)
    records.append({"test": f"{label}: Hermiticity, trace, positivity, probabilities",
                    "values": checks, "passed": bool(passed)})
    if not passed:
        raise AssertionError(f"{label}: output diagnostics failed: {checks}")


def main():
    records = []
    singlet = projector(singlet_state())

    # Every Bell branch must return the arbitrary input, not merely its average.
    chi = input_spin_state(0.37 + 0.11j, -0.83 + 0.07j)
    outcomes = teleport_resource(singlet, chi, validate=True)
    assert len(outcomes) == 4
    for item in outcomes:
        assert_close(item.probability, 0.25,
                     f"singlet {item.name}: probability", records)
        error = float(np.max(np.abs(item.output - projector(chi))))
        passed = error < 2e-10
        records.append({"test": f"singlet {item.name}: corrected output equals input",
                        "max_abs_matrix_error": error, "passed": bool(passed)})
        if not passed:
            raise AssertionError(f"Bell correction failed in {item.name}")

    # Ideal singlet for all six inputs.
    result = average_teleportation_fidelity(singlet, validate=True)
    assert_close(result["average_fidelity"], 1.0,
                 "ideal singlet: six-state explicit fidelity", records)
    check_diagnostics(result, "ideal singlet", records)
    assert_close(fully_entangled_fraction(singlet), 1.0,
                 "ideal singlet: fully entangled fraction", records)
    assert_close(optimized_teleportation_fidelity(singlet), 1.0,
                 "ideal singlet: optimized fidelity", records)
    assert_close(concurrence_xstate(singlet), 1.0,
                 "ideal singlet: concurrence", records)

    # Maximally mixed resource: B remains maximally mixed in every branch.
    mixed = np.eye(4, dtype=complex) / 4.0
    mixed_outcomes = teleport_resource(mixed, six_input_states()["+"], validate=True)
    for item in mixed_outcomes:
        error = float(np.max(np.abs(item.output - np.eye(2) / 2.0)))
        records.append({"test": f"mixed {item.name}: output is I/2",
                        "max_abs_matrix_error": error,
                        "passed": bool(error < 2e-10)})
        if error >= 2e-10:
            raise AssertionError("Mixed-resource output is not I/2")
    result = average_teleportation_fidelity(mixed, validate=True)
    assert_close(result["average_fidelity"], 0.5,
                 "maximally mixed: six-state explicit fidelity", records)
    check_diagnostics(result, "maximally mixed", records)
    assert_close(fully_entangled_fraction(mixed), 0.25,
                 "maximally mixed: fully entangled fraction", records)
    assert_close(optimized_teleportation_fidelity(mixed), 0.5,
                 "maximally mixed: optimized fidelity", records)
    assert_close(concurrence_xstate(mixed), 0.0,
                 "maximally mixed: concurrence", records)

    # Werner checks use F=(1+w)/2, equivalently (2 f_FEF+1)/3 with
    # f_FEF=(1+3w)/4.  The fixed singlet protocol is optimal for these states.
    for w in (0.0, 0.2, 1.0/3.0, 0.5, 0.75, 1.0):
        rho = w * singlet + (1.0 - w) * mixed
        expected = (1.0 + w) / 2.0
        expected_fef = (1.0 + 3.0*w) / 4.0
        result = average_teleportation_fidelity(rho, validate=True)
        assert_close(result["average_fidelity"], expected,
                     f"Werner w={w:.8f}: explicit fidelity", records)
        check_diagnostics(result, f"Werner w={w:.8f}", records)
        assert_close(fully_entangled_fraction(rho), expected_fef,
                     f"Werner w={w:.8f}: fully entangled fraction", records)
        assert_close(optimized_teleportation_fidelity(rho), expected,
                     f"Werner w={w:.8f}: optimized fidelity", records)
        assert_close(singlet_fraction(rho), expected_fef,
                     f"Werner w={w:.8f}: singlet fraction", records)
        assert_close(concurrence_xstate(rho), max(0.0, (3.0*w-1.0)/2.0),
                     f"Werner w={w:.8f}: concurrence", records)

    # Pure product control: the singlet-reference protocol is poor, while the
    # local-unitary optimized benchmark reaches the classical threshold.
    product = np.zeros((4, 4), dtype=complex)
    product[0, 0] = 1.0
    result = average_teleportation_fidelity(product, validate=True)
    assert_close(result["average_fidelity"], 1.0/3.0,
                 "pure product |00>: fixed-protocol fidelity", records)
    check_diagnostics(result, "pure product |00>", records)
    assert_close(fully_entangled_fraction(product), 0.5,
                 "pure product |00>: fully entangled fraction", records)
    assert_close(optimized_teleportation_fidelity(product), 2.0/3.0,
                 "pure product |00>: optimized fidelity", records)

    # A non-Werner X state: the local-unitary optimum must not be worse than
    # the fixed singlet-reference protocol.
    psi_plus = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
    non_werner = (0.31 * singlet + 0.19 * projector(psi_plus)
                  + 0.50 * mixed)
    result = average_teleportation_fidelity(non_werner, validate=True)
    fixed = result["average_fidelity"]
    optimized = optimized_teleportation_fidelity(non_werner)
    records.append({"test": "non-Werner: optimized fidelity >= fixed fidelity",
                    "fixed": fixed, "optimized": optimized,
                    "passed": bool(optimized + 1e-10 >= fixed)})
    if optimized + 1e-10 < fixed:
        raise AssertionError("FEF benchmark is below explicit fixed protocol")
    check_diagnostics(result, "non-Werner", records)

    assert_close(CLASSICAL_BENCHMARK, 2.0/3.0,
                 "classical benchmark", records)
    summary = {
        "status": "passed",
        "test_count": len(records),
        "tests": records,
        "optimization_method": (
            "Exact singular-value formula for the fully entangled fraction; "
            "no numerical optimizer was used."),
        "werner_formula": "F_average=(1+w)/2; singlet_fraction=FEF=(1+3w)/4",
        "bell_correction_order": ["Phi+", "Phi-", "Psi+", "Psi-"],
        "pauli_correction_order": ["iY", "X", "Z", "I"],
    }
    REPORT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"],
                      "test_count": len(records),
                      "report": str(REPORT.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
