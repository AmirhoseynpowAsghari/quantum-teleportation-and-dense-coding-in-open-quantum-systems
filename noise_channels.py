# =============================================================================
# noise_channels.py
# Non-Markovian Kraus operators for dephasing and amplitude-damping channels.
#
# Each channel is parameterised by:
#   γ   – system-bath coupling strength
#   Γ   – reservoir correlation (memory) rate
# =============================================================================

import numpy as np


# ---------------------------------------------------------------------------
# Decay functions
# ---------------------------------------------------------------------------

def p_dephasing(t, gamma, Gamma):
    """
    Non-Markovian dephasing factor for an Ornstein-Uhlenbeck (colored) noise bath.

    p(t) = exp{ -γ/2 · [t + (e^{-Γt} - 1)/Γ] }

    Parameters
    ----------
    t     : float  ≥ 0
    gamma : float  coupling strength
    Gamma : float  reservoir memory rate (Γ → ∞ gives white noise / Markovian)

    Returns
    -------
    float in [0, 1]
    """
    exponent = -(gamma / 2.0) * (t + (np.exp(-Gamma * t) - 1.0) / Gamma)
    return float(np.clip(np.exp(exponent), 0.0, 1.0))


def p_amplitude_damping(t, gamma, Gamma):
    """
    Non-Markovian amplitude-damping decay for a Lorentzian structured reservoir.

    Weak-coupling (d² > 0, oscillatory) regime:
        P(t) = e^{-Γt} [cos(d·t/2) + (Γ/d)·sin(d·t/2)]²
    Overdamped (d² ≤ 0) regime:
        P(t) = e^{-Γt} [1 + Γt/2]²

    where  d² = 2γΓ − Γ².

    Parameters
    ----------
    t     : float  ≥ 0
    gamma : float  coupling strength
    Gamma : float  reservoir memory rate

    Returns
    -------
    float P(t) in [0, 1]
    """
    d2 = 2.0 * gamma * Gamma - Gamma**2
    if d2 > 0.0:
        d   = np.sqrt(d2)
        val = np.exp(-Gamma * t) * (
            np.cos(0.5 * d * t) + (Gamma / d) * np.sin(0.5 * d * t)
        )**2
    else:                          # overdamped (smooth d → 0 limit)
        val = np.exp(-Gamma * t) * (1.0 + 0.5 * Gamma * t)**2
    return float(np.clip(val, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Single-qubit Kraus sets
# ---------------------------------------------------------------------------

def kraus_dephasing_1q(p):
    """
    Kraus operators for single-qubit dephasing with coherence retention p ∈ [0,1].

    K₀ = diag(1, p),   K₁ = diag(0, √(1−p²))
    """
    p   = float(np.clip(p, 0.0, 1.0))
    K0  = np.array([[1.0, 0.0], [0.0, p           ]], dtype=complex)
    K1  = np.array([[0.0, 0.0], [0.0, np.sqrt(max(0.0, 1.0 - p**2))]], dtype=complex)
    return [K0, K1]


def kraus_amplitude_damping_1q(p):
    """
    Kraus operators for single-qubit amplitude damping with survival probability p.

    K₀ = [[1, 0], [0, √p]],   K₁ = [[0, √(1−p)], [0, 0]]
    """
    p   = float(np.clip(p, 0.0, 1.0))
    K0  = np.array([[1.0, 0.0          ], [0.0, np.sqrt(p)           ]], dtype=complex)
    K1  = np.array([[0.0, np.sqrt(1.0 - p)], [0.0, 0.0               ]], dtype=complex)
    return [K0, K1]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get_kraus_pair(t, kind, gammaA, GammaA, gammaB, GammaB):
    """
    Return Kraus operator lists (Ka, Kb) for qubit A and qubit B at time t.

    Parameters
    ----------
    t              : float  ≥ 0
    kind           : str    'dephasing' or 'amp'
    gammaA/GammaA  : floats bath parameters for qubit A
    gammaB/GammaB  : floats bath parameters for qubit B

    Returns
    -------
    Ka : list of (2×2) complex arrays
    Kb : list of (2×2) complex arrays
    """
    if kind == "dephasing":
        pA = p_dephasing(t, gammaA, GammaA)
        pB = p_dephasing(t, gammaB, GammaB)
        Ka = kraus_dephasing_1q(pA)
        Kb = kraus_dephasing_1q(pB)
    elif kind == "amp":
        pA = p_amplitude_damping(t, gammaA, GammaA)
        pB = p_amplitude_damping(t, gammaB, GammaB)
        Ka = kraus_amplitude_damping_1q(pA)
        Kb = kraus_amplitude_damping_1q(pB)
    else:
        raise ValueError(f"Unknown noise kind '{kind}'. Use 'dephasing' or 'amp'.")
    return Ka, Kb


if __name__ == "__main__":
    # Quick unit tests
    for kind in ("dephasing", "amp"):
        Ka, Kb = get_kraus_pair(5.0, kind,
                                gammaA=0.05, GammaA=0.01,
                                gammaB=0.05, GammaB=0.01)
        # Verify completeness  Σ Kᵢ†Kᵢ = I  for qubit A
        I_check = sum(K.conj().T @ K for K in Ka)
        assert np.allclose(I_check, np.eye(2), atol=1e-10), \
            f"Completeness FAILED for kind={kind}"
        print(f"[noise_channels] {kind}: completeness OK")