"""Conditional two-qubit density matrices for the Rashba helicity BdG model.

The V_SO=0 Oh--Kim matrix assumes a spin-diagonal normal correlator and a
spin-singlet anomalous correlator.  At nonzero Rashba coupling those assumptions
need not hold.  This module instead retains 2x2 internal correlator matrices and
uses Wick's theorem before conditioning on one electron at each of two sites.

Two internal bases are available:

``spin``
    Original spin-up/spin-down modes at the two positions.  This is the basis
    for ordinary spin-qubit entanglement and teleportation.
``band``
    Fourier-transformed helicity-band modes d_{i,s}=sum_k exp(i k.R_i)c_{k,s}.
    These are valid mathematical fermionic modes, but are nonlocal combinations
    of the original spin/site modes.  Observables in this basis describe
    helicity-pseudospin qubits, not automatically physical spin qubits.

Conventions follow Hutchinson, Hirsch, and Marsiglio, arXiv:1801.08988,
Eqs. (16)--(29).  The repository's bdg["Delta_ks"] is the even scalar
bar-Delta_ks.  The full gap has the additional factor s exp(-i theta_k).
At T=0, b_ks=bar-Delta_ks/(2E_ks), and the overall sign of the anomalous
correlator is omitted because only F*F enters the conditional density matrix.

No archived V_SO=0 result is modified by this module.
"""
from __future__ import annotations

import numpy as np

from teleportation import fully_entangled_fraction, optimized_teleportation_fidelity

PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SINGLET = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2.0)


def helicity_phase(bdg: dict) -> np.ndarray:
    """Return exp(i theta_k), with an arbitrary benign gauge at lambda_k=0."""
    KX, KY, lam = bdg["KX"], bdg["KY"], bdg["sqrt_sin"]
    phase = np.ones_like(KX, dtype=complex)
    nonzero = lam > 1e-14
    phase[nonzero] = (np.sin(KY[nonzero]) - 1j * np.sin(KX[nonzero])) / lam[nonzero]
    return phase


def helicity_spinors(bdg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return U_k and U_-k for c_{k,sigma}=sum_s U_{sigma,s}c_{k,s}."""
    phase = helicity_phase(bdg)
    shape = phase.shape
    U = np.zeros(shape + (2, 2), dtype=complex)
    U[..., 0, 0] = 1.0 / np.sqrt(2.0)
    U[..., 0, 1] = 1.0 / np.sqrt(2.0)
    U[..., 1, 0] = phase / np.sqrt(2.0)
    U[..., 1, 1] = -phase / np.sqrt(2.0)
    # exp(i theta_-k)=-exp(i theta_k), hence U_-k=Z U_k.
    Z = np.diag([1.0, -1.0])
    U_minus = np.einsum("ab,...bc->...ac", Z, U)
    return U, U_minus


def momentum_correlators(bdg: dict, basis: str = "spin") -> tuple[np.ndarray, np.ndarray]:
    """Return N_k and F_k for zero-COM-momentum intraband pairing.

    N_k[sigma,sigma'] = <c_{k,sigma}^dagger c_{k,sigma'}>.
    F_k[sigma,sigma'] = <c_{k,sigma} c_{-k,sigma'}> (up to one common sign).

    The normal spin matrix uses U* N_band U^T because N is defined with the
    creation operator on the left.  The anomalous matrix uses U_k F_band U_-k^T.
    """
    if basis not in {"spin", "band"}:
        raise ValueError("basis must be 'spin' or 'band'")
    shape = bdg["Delta_ks"].shape[:-1]
    N_band = np.zeros(shape + (2, 2), dtype=complex)
    F_band = np.zeros(shape + (2, 2), dtype=complex)
    occupation = np.abs(bdg["v_ks"]) ** 2
    pair = bdg["Delta_ks"] / (2.0 * bdg["E_ks"])
    helicity_sign = np.array([1.0, -1.0])
    phase = helicity_phase(bdg)
    for s in range(2):
        N_band[..., s, s] = occupation[..., s]
        # Full phase of s exp(-i theta_k) bar-Delta_ks.  Dropping it would
        # violate F_-k=-F_k for same-helicity time-reversed pairing.
        F_band[..., s, s] = (helicity_sign[s] * np.conj(phase)
                             * pair[..., s])
    lam = bdg["sqrt_sin"]
    nondegenerate_soc = bool(np.max(np.abs(
        bdg["epsilon_ks"][..., 0] - bdg["epsilon_ks"][..., 1])) > 1e-12)
    if basis == "band" or nondegenerate_soc:
        # At a TRIM, c_{k,s} and c_{-k,s} coincide (up to a gauge), so Pauli
        # exclusion requires zero same-band pair amplitude.  At exact zero-SOC
        # degeneracy the original-spin singlet is instead gauge-invariant.
        F_band[lam <= 1e-14] = 0.0
    if basis == "band":
        return N_band, F_band
    U, U_minus = helicity_spinors(bdg)
    N_spin = np.einsum("...ab,...bc,...dc->...ad", U.conj(), N_band, U)
    F_spin = np.einsum("...ab,...bc,...dc->...ad", U, F_band, U_minus)
    return N_spin, F_spin


def position_correlators(bdg: dict, r: np.ndarray, basis: str = "spin") -> dict:
    """Fourier-transform N_k and F_k to two positions separated by r=R_B-R_A."""
    if basis not in {"spin", "band"}:
        raise ValueError("basis must be 'spin' or 'band'")
    r = np.asarray(r, dtype=float).reshape(2)
    N_k, F_k = momentum_correlators(bdg, basis=basis)
    phase = np.exp(1j * (bdg["KX"] * r[0] + bdg["KY"] * r[1]))
    weight = 1.0 / (bdg["KX"].size)
    N_AB = weight * np.einsum("ij,ijab->ab", phase, N_k)
    N_A = weight * np.sum(N_k, axis=(0, 1))
    F_AB = weight * np.einsum("ij,ijab->ab", phase.conj(), F_k)
    if np.array_equal(r, np.zeros(2)):
        # c_{A,alpha}c_{A,beta}=-c_{A,beta}c_{A,alpha} is exact.  Remove
        # finite-quadrature symmetric residue, not density-matrix eigenvalues.
        F_AB = 0.5 * (F_AB - F_AB.T)
    # At the same ordered pair of positions, anticommutation gives directly
    # <c_B,beta c_A,alpha>=-<c_A,alpha c_B,beta>.
    F_BA = -F_AB.T
    return {
        "N_A": N_A,
        "N_B": N_A.copy(),
        "N_AB": N_AB,
        "N_BA": N_AB.conj().T,
        "F_AB": F_AB,
        "F_BA": F_BA,
    }


def conditional_two_qubit_density(correlators: dict, *, normalize: bool = True,
                                  hermitian_atol: float = 1e-10) -> tuple[np.ndarray, float]:
    """Construct the conditional state from matrix-valued Wick contractions.

    In the basis |00>,|01>,|10>,|11>, the unnormalized element is

      <b_beta'^dagger a_alpha'^dagger a_alpha b_beta>
        = N_B[beta',beta] N_A[alpha',alpha]
          - N_BA[beta',alpha] N_AB[alpha',beta]
          + F_AB[alpha',beta']* F_AB[alpha,beta].

    Equivalently the last creation-side factor is -F_BA[beta',alpha']*.
    Using F_AB directly avoids an operator-order sign error.

    The returned weight is the trace of this two-detection sector before
    normalization.  No X-state projection or eigenvalue clipping is applied.
    """
    N_A = np.asarray(correlators["N_A"], dtype=complex)
    N_B = np.asarray(correlators["N_B"], dtype=complex)
    N_AB = np.asarray(correlators["N_AB"], dtype=complex)
    N_BA = np.asarray(correlators["N_BA"], dtype=complex)
    F_AB = np.asarray(correlators["F_AB"], dtype=complex)
    F_BA = np.asarray(correlators["F_BA"], dtype=complex)
    if any(matrix.shape != (2, 2) for matrix in (N_A, N_B, N_AB, N_BA, F_AB, F_BA)):
        raise ValueError("Every normal and anomalous correlator must be 2x2")
    rho = np.zeros((4, 4), dtype=complex)
    for alpha in range(2):
        for beta in range(2):
            for alpha_p in range(2):
                for beta_p in range(2):
                    rho[2 * alpha + beta, 2 * alpha_p + beta_p] = (
                        N_B[beta_p, beta] * N_A[alpha_p, alpha]
                        - N_BA[beta_p, alpha] * N_AB[alpha_p, beta]
                        + np.conj(F_AB[alpha_p, beta_p]) * F_AB[alpha, beta]
                    )
    hermiticity_error = float(np.max(np.abs(rho - rho.conj().T)))
    if hermiticity_error > hermitian_atol:
        raise ValueError(f"Two-detection matrix is not Hermitian: {hermiticity_error:.3e}")
    rho = 0.5 * (rho + rho.conj().T)
    weight = float(np.trace(rho).real)
    if not np.isfinite(weight) or weight <= 0.0:
        raise ValueError(f"The two-detection weight must be positive, got {weight}")
    if normalize:
        rho = rho / weight
    return rho, weight


def density_at_separation(bdg: dict, r: np.ndarray, basis: str = "spin") -> dict:
    """Return a normalized conditional matrix, its weight, and correlators."""
    correlators = position_correlators(bdg, r, basis=basis)
    rho, weight = conditional_two_qubit_density(correlators)
    return {"rho": rho, "weight": weight, **correlators}


def concurrence(rho: np.ndarray) -> float:
    """Wootters concurrence for a general two-qubit density matrix."""
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (4, 4):
        raise ValueError("rho must be a two-qubit 4x4 matrix")
    spin_flipped = np.kron(PAULI_Y, PAULI_Y) @ rho.conj() @ np.kron(PAULI_Y, PAULI_Y)
    eigenvalues = np.linalg.eigvals(rho @ spin_flipped)
    if np.max(np.abs(eigenvalues.imag)) > 1e-8:
        raise ValueError("Unexpected complex eigenvalues in concurrence calculation")
    roots = np.sqrt(np.clip(eigenvalues.real, 0.0, None))
    roots = np.sort(roots)[::-1]
    return float(max(0.0, roots[0] - np.sum(roots[1:])))


def resource_metrics(rho: np.ndarray) -> dict:
    """Separate fixed-reference and locally optimized pseudospin protocols."""
    rho = np.asarray(rho, dtype=complex)
    singlet_fraction = float(np.real(SINGLET.conj() @ rho @ SINGLET))
    fef = fully_entangled_fraction(rho)
    return {
        "concurrence": concurrence(rho),
        "singlet_fraction": singlet_fraction,
        "fixed_reference_average_fidelity": (2.0 * singlet_fraction + 1.0) / 3.0,
        "fully_entangled_fraction": fef,
        "optimized_average_fidelity": optimized_teleportation_fidelity(rho),
    }


def entropy_bits(rho: np.ndarray) -> float:
    """Von Neumann entropy in bits, rejecting more than roundoff negativity."""
    eigenvalues = np.linalg.eigvalsh(np.asarray(rho, dtype=complex))
    if np.min(eigenvalues) < -1e-10:
        raise ValueError("Cannot take entropy of a nonpositive density matrix")
    eigenvalues = np.clip(eigenvalues.real, 0.0, 1.0)
    nonzero = eigenvalues > 1e-15
    return float(-np.sum(eigenvalues[nonzero] * np.log2(eigenvalues[nonzero])))


def reduced_state_b(rho: np.ndarray) -> np.ndarray:
    """Trace out subsystem A from a two-qubit state ordered as |A B>."""
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (4, 4):
        raise ValueError("rho must be a two-qubit 4x4 matrix")
    return np.trace(rho.reshape(2, 2, 2, 2), axis1=0, axis2=2)


def pauli_ensemble_average(rho: np.ndarray) -> np.ndarray:
    """Average over I,X,Y,Z applied to A with equal probability."""
    rho = np.asarray(rho, dtype=complex)
    identity = np.eye(2, dtype=complex)
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    averaged = np.zeros((4, 4), dtype=complex)
    for pauli in (identity, pauli_x, PAULI_Y, np.diag([1.0, -1.0])):
        unitary = np.kron(pauli, identity)
        averaged += unitary @ rho @ unitary.conj().T
    return averaged / 4.0


def dense_coding_metrics(rho: np.ndarray) -> dict:
    """Implemented Pauli-ensemble Holevo quantity, not a general capacity.

    The ideal protocol uses local Pauli encodings on A, noiseless transmission
    of A to B, and asymptotic joint decoding.  In the helicity-band basis the
    encodings are band-pseudospin operations, not original-spin rotations.
    """
    rho = np.asarray(rho, dtype=complex)
    rho_b = reduced_state_b(rho)
    averaged = pauli_ensemble_average(rho)
    identity = np.kron(np.eye(2) / 2.0, rho_b)
    average_error = float(np.max(np.abs(averaged - identity)))
    entropy_ab = entropy_bits(rho)
    entropy_b = entropy_bits(rho_b)
    entropy_average = entropy_bits(averaged)
    chi = 1.0 + entropy_b - entropy_ab
    if chi < -1e-10 or chi > 2.0 + 1e-10:
        raise ValueError(f"Pauli-ensemble Holevo quantity outside [0,2]: {chi}")
    return {
        "holevo_pauli_bits": float(np.clip(chi, 0.0, 2.0)),
        "entropy_AB_bits": entropy_ab,
        "entropy_B_bits": entropy_b,
        "pauli_averaged_entropy_bits": entropy_average,
        "pauli_average_identity_error": average_error,
        "above_one_bit_benchmark": bool(chi > 1.0),
    }


def physical_diagnostics(rho: np.ndarray) -> dict:
    """Return normalization and positivity diagnostics without clipping them."""
    rho = np.asarray(rho, dtype=complex)
    return {
        "trace_error": float(abs(np.trace(rho) - 1.0)),
        "hermiticity_error": float(np.max(np.abs(rho - rho.conj().T))),
        "minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(rho))),
    }
