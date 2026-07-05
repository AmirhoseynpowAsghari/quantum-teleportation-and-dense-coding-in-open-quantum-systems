# =============================================================================
# noise_channels.py
# =============================================================================

import numpy as np


def p_dephasing(t, gamma, Gamma):
    exp = -(gamma/2.0)*(t + (np.exp(-Gamma*t) - 1.0)/Gamma)
    return float(np.clip(np.exp(exp), 0.0, 1.0))


def p_amplitude_damping(t, gamma, Gamma):
    d2 = 2.0*gamma*Gamma - Gamma**2
    if d2 > 0.0:
        d   = np.sqrt(d2)
        val = np.exp(-Gamma*t)*(np.cos(0.5*d*t)+(Gamma/d)*np.sin(0.5*d*t))**2
    else:
        val = np.exp(-Gamma*t)*(1.0 + 0.5*Gamma*t)**2
    return float(np.clip(val, 0.0, 1.0))


def kraus_dephasing_1q(p):
    p  = float(np.clip(p, 0.0, 1.0))
    K0 = np.array([[1.0, 0.0], [0.0, p]], dtype=complex)
    K1 = np.array([[0.0, 0.0], [0.0, np.sqrt(max(0.0, 1.0-p**2))]], dtype=complex)
    return [K0, K1]


def kraus_amplitude_damping_1q(p):
    p  = float(np.clip(p, 0.0, 1.0))
    K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(p)]], dtype=complex)
    K1 = np.array([[0.0, np.sqrt(1.0-p)], [0.0, 0.0]], dtype=complex)
    return [K0, K1]


def get_kraus_pair(t, kind, gammaA, GammaA, gammaB, GammaB):
    if kind == "dephasing":
        Ka = kraus_dephasing_1q(p_dephasing(t, gammaA, GammaA))
        Kb = kraus_dephasing_1q(p_dephasing(t, gammaB, GammaB))
    elif kind == "amp":
        Ka = kraus_amplitude_damping_1q(p_amplitude_damping(t, gammaA, GammaA))
        Kb = kraus_amplitude_damping_1q(p_amplitude_damping(t, gammaB, GammaB))
    else:
        raise ValueError(f"Unknown kind '{kind}'")
    return Ka, Kb