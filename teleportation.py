# =============================================================================
# teleportation.py
# =============================================================================
"""Explicit two-qubit-resource quantum teleportation.

Physical scope
--------------
The input Q is an effective spin qubit, not necessarily a Bogoliubov
quasiparticle.  The resource rho_AB is assumed to describe two independently
addressable spin modes.  Bell measurement and classical feedforward are ideal
mathematical operations; this module does not establish their microscopic
experimental implementation.

Protocol
--------
For an input state |chi>_Q and resource rho_AB, form rho_QAB and project Q,A
onto each Bell state.  Condition on the outcome, apply the corresponding
Pauli correction to B, and calculate <chi|rho_B,out|chi>.  The six-state
average is the standard uniform pure-input average for a qubit.

The fully entangled fraction is calculated exactly (not by numerical
optimization) from the singular values of the 3x3 Pauli correlation matrix:
    f_FEF = (1 + sum_i singular_value_i(T)) / 4.
The corresponding optimal average teleportation fidelity is
    F_opt = (2 f_FEF + 1) / 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from observables import concurrence_xstate, singlet_fraction

CLASSICAL_BENCHMARK = 2.0 / 3.0


# ---------------------------------------------------------------------------
# Basic states and operators
# ---------------------------------------------------------------------------


def pauli_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Y, Z


def input_spin_state(alpha: complex, beta: complex) -> np.ndarray:
    """Return normalized |chi> = alpha|up> + beta|down>."""
    state = np.asarray([alpha, beta], dtype=complex)
    norm = np.linalg.norm(state)
    if norm < 1e-15:
        raise ValueError("The input spin state must be nonzero.")
    return state / norm


def six_input_states() -> Dict[str, np.ndarray]:
    """Six cardinal pure states used for the uniform input average."""
    return {
        "0": input_spin_state(1, 0),
        "1": input_spin_state(0, 1),
        "+": input_spin_state(1, 1),
        "-": input_spin_state(1, -1),
        "+i": input_spin_state(1, 1j),
        "-i": input_spin_state(1, -1j),
    }


def singlet_state() -> np.ndarray:
    return np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)


def bell_states_qa() -> Dict[str, np.ndarray]:
    """Bell vectors in the order |Q A>, basis 00,01,10,11."""
    return {
        "Phi+": np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
        "Phi-": np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),
        "Psi+": np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),
        "Psi-": np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),
    }


def bell_projectors_qa() -> Dict[str, np.ndarray]:
    return {name: np.outer(state, state.conj())
            for name, state in bell_states_qa().items()}


def feedforward_corrections() -> Dict[str, np.ndarray]:
    """Corrections for a singlet-reference resource.

    With the Bell ordering in bell_states_qa(), ideal singlet-resource outputs
    are, up to global phase:
      Phi+ -> iY|chi>, Phi- -> X|chi>, Psi+ -> Z|chi>, Psi- -> |chi>.
    Therefore these unitaries restore |chi> in every branch.
    """
    I, X, Y, Z = pauli_matrices()
    return {"Phi+": 1j * Y, "Phi-": X, "Psi+": Z, "Psi-": I}


# ---------------------------------------------------------------------------
# Density-matrix helpers
# ---------------------------------------------------------------------------


def partial_trace(rho: np.ndarray, dims: Iterable[int], keep: Iterable[int]) -> np.ndarray:
    """Partial trace over ordered tensor-product subsystems.

    Parameters
    ----------
    rho : square density matrix
    dims : subsystem dimensions in the original ordering
    keep : indices of subsystems to retain
    """
    dims = tuple(int(d) for d in dims)
    keep = tuple(int(i) for i in keep)
    if np.prod(dims) != rho.shape[0] or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho shape is incompatible with dims.")
    if not keep:
        raise ValueError("At least one subsystem must be retained.")
    if any(i < 0 or i >= len(dims) for i in keep):
        raise ValueError("Invalid retained subsystem index.")

    tensor = np.asarray(rho, dtype=complex).reshape(dims + dims)
    trace_over = sorted((i for i in range(len(dims)) if i not in keep),
                        reverse=True)
    for index in trace_over:
        current_n = tensor.ndim // 2
        tensor = np.trace(tensor, axis1=index, axis2=index + current_n)
    retained_dims = [dims[i] for i in keep]
    dim = int(np.prod(retained_dims))
    return tensor.reshape(dim, dim)


def construct_qab_state(input_state: np.ndarray, rho_ab: np.ndarray) -> np.ndarray:
    """Construct rho_QAB = |chi><chi| tensor rho_AB as an 8x8 matrix."""
    state = np.asarray(input_state, dtype=complex).reshape(2)
    state = input_spin_state(state[0], state[1])
    rho_ab = np.asarray(rho_ab, dtype=complex)
    if rho_ab.shape != (4, 4):
        raise ValueError("rho_AB must be a two-qubit 4x4 density matrix.")
    rho_q = np.outer(state, state.conj())
    return np.kron(rho_q, rho_ab)


@dataclass
class TeleportationOutcome:
    name: str
    probability: float
    output_before_correction: np.ndarray
    output: np.ndarray
    fidelity: float
    correction: np.ndarray


# ---------------------------------------------------------------------------
# Explicit Bell-measurement protocol
# ---------------------------------------------------------------------------


def teleport_resource(rho_ab: np.ndarray, input_state: np.ndarray,
                      validate: bool = False) -> Tuple[TeleportationOutcome, ...]:
    """Teleport one pure input state through a two-qubit resource.

    Returns all four conditional branches.  No outcome postselection is used in
    the average fidelity; each branch is weighted by its Born probability.
    """
    state = np.asarray(input_state, dtype=complex).reshape(2)
    state = input_spin_state(state[0], state[1])
    rho_ab = np.asarray(rho_ab, dtype=complex)
    rho_qab = construct_qab_state(state, rho_ab)

    projectors = bell_projectors_qa()
    corrections = feedforward_corrections()
    outcomes = []
    for name, projector_qa in projectors.items():
        projector = np.kron(projector_qa, np.eye(2, dtype=complex))
        projected = projector @ rho_qab @ projector.conj().T
        probability = float(np.trace(projected).real)
        if probability < -1e-12:
            raise ValueError(f"Negative Bell-outcome probability for {name}.")
        probability = max(probability, 0.0)
        if probability <= 1e-15:
            # A zero-probability branch has no conditional physical state and
            # contributes zero to the outcome-averaged fidelity.
            continue
        before = partial_trace(projected, (2, 2, 2), keep=(2,)) / probability
        before = 0.5 * (before + before.conj().T)
        before /= np.trace(before).real
        correction = corrections[name]
        output = correction @ before @ correction.conj().T
        output = 0.5 * (output + output.conj().T)
        output /= np.trace(output).real
        fidelity = float(np.real(state.conj() @ output @ state))
        outcomes.append(TeleportationOutcome(
            name=name, probability=probability,
            output_before_correction=before, output=output,
            fidelity=fidelity, correction=correction,
        ))

    if not outcomes:
        raise ValueError("All Bell-outcome probabilities are numerically zero.")
    probability_sum = sum(item.probability for item in outcomes)
    if not np.isclose(probability_sum, 1.0, atol=1e-9, rtol=0):
        raise ValueError(f"Bell probabilities sum to {probability_sum}, not one.")
    if validate:
        for item in outcomes:
            _validate_output(item)
    return tuple(outcomes)


def _validate_output(item: TeleportationOutcome) -> None:
    for output in (item.output_before_correction, item.output):
        if not np.allclose(output, output.conj().T, atol=1e-10, rtol=0):
            raise ValueError(f"Non-Hermitian output in branch {item.name}.")
        if not np.isclose(np.trace(output).real, 1.0, atol=1e-10, rtol=0):
            raise ValueError(f"Non-unit output trace in branch {item.name}.")
        if np.linalg.eigvalsh(output).min() < -1e-10:
            raise ValueError(f"Nonpositive output in branch {item.name}.")
    if item.fidelity < -1e-10 or item.fidelity > 1.0 + 1e-10:
        raise ValueError(f"Out-of-range fidelity in branch {item.name}.")


def input_averaged_teleportation(rho_ab: np.ndarray, input_state: np.ndarray,
                                 validate: bool = False) -> dict:
    outcomes = teleport_resource(rho_ab, input_state, validate=validate)
    average = float(sum(item.probability * item.fidelity for item in outcomes))
    eigen_min = min(float(np.linalg.eigvalsh(item.output).min())
                    for item in outcomes)
    hermiticity_error = max(float(np.max(np.abs(item.output-item.output.conj().T)))
                            for item in outcomes)
    trace_error = max(abs(float(np.trace(item.output).real)-1.0)
                      for item in outcomes)
    return {
        "average_fidelity": average,
        "outcomes": outcomes,
        "probability_sum": float(sum(item.probability for item in outcomes)),
        "minimum_output_eigenvalue": eigen_min,
        "maximum_hermiticity_error": hermiticity_error,
        "maximum_trace_error": trace_error,
    }


def average_teleportation_fidelity(rho_ab: np.ndarray,
                                   validate: bool = False) -> dict:
    """Average the explicit protocol over the six cardinal input states."""
    per_input = {}
    diagnostics = {
        "minimum_output_eigenvalue": np.inf,
        "maximum_hermiticity_error": 0.0,
        "maximum_trace_error": 0.0,
        "maximum_probability_sum_error": 0.0,
    }
    for name, state in six_input_states().items():
        result = input_averaged_teleportation(rho_ab, state, validate=validate)
        per_input[name] = result["average_fidelity"]
        diagnostics["minimum_output_eigenvalue"] = min(
            diagnostics["minimum_output_eigenvalue"],
            result["minimum_output_eigenvalue"])
        diagnostics["maximum_hermiticity_error"] = max(
            diagnostics["maximum_hermiticity_error"],
            result["maximum_hermiticity_error"])
        diagnostics["maximum_trace_error"] = max(
            diagnostics["maximum_trace_error"], result["maximum_trace_error"])
        diagnostics["maximum_probability_sum_error"] = max(
            diagnostics["maximum_probability_sum_error"],
            abs(result["probability_sum"] - 1.0))
    return {
        "average_fidelity": float(np.mean(list(per_input.values()))),
        "per_input_fidelity": per_input,
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# Optimized local-unitary benchmark
# ---------------------------------------------------------------------------


def fully_entangled_fraction(rho_ab: np.ndarray) -> float:
    """Exact maximum singlet overlap over local single-qubit unitaries.

    If T_ij = Tr[rho (sigma_i tensor sigma_j)], then the maximum over local
    Bloch rotations is (1 + ||T||_1)/4, where ||T||_1 is the sum of singular
    values.  Bell-state local Bloch vectors vanish, so no additional terms
    enter the optimization.
    """
    rho_ab = np.asarray(rho_ab, dtype=complex)
    if rho_ab.shape != (4, 4):
        raise ValueError("rho_AB must be 4x4.")
    _, X, Y, Z = pauli_matrices()
    paulis = (X, Y, Z)
    correlation = np.zeros((3, 3), dtype=float)
    for i, left in enumerate(paulis):
        for j, right in enumerate(paulis):
            value = np.trace(rho_ab @ np.kron(left, right))
            correlation[i, j] = float(np.real(value))
    singular_values = np.linalg.svd(correlation, compute_uv=False)
    result = (1.0 + float(np.sum(singular_values))) / 4.0
    if result < -1e-10 or result > 1.0 + 1e-10:
        raise ValueError(f"Fully entangled fraction outside [0,1]: {result}")
    return float(np.clip(result, 0.0, 1.0))


def optimized_teleportation_fidelity(rho_ab: np.ndarray) -> float:
    return (2.0 * fully_entangled_fraction(rho_ab) + 1.0) / 3.0


def resource_metrics(rho_ab: np.ndarray, validate: bool = False) -> dict:
    """Calculate all quantities used in the project's teleportation plots.

    Concurrence uses the repository's X-state reduction, which is valid for
    the BdG-derived states and their implemented independent amplitude damping.
    """
    explicit = average_teleportation_fidelity(rho_ab, validate=validate)
    fef = fully_entangled_fraction(rho_ab)
    return {
        "concurrence": concurrence_xstate(rho_ab),
        "singlet_fraction": singlet_fraction(rho_ab),
        "fixed_average_fidelity": explicit["average_fidelity"],
        "fully_entangled_fraction": fef,
        "optimized_average_fidelity": (2.0 * fef + 1.0) / 3.0,
        "per_input_fidelity": explicit["per_input_fidelity"],
        "diagnostics": explicit["diagnostics"],
    }
