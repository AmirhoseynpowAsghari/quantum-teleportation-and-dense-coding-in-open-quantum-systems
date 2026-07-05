# =============================================================================
# observables.py
# =============================================================================

import numpy as np


def concurrence_xstate(rho):
    r11 = max(rho[0,0].real, 0.0)
    r22 = max(rho[1,1].real, 0.0)
    r33 = max(rho[2,2].real, 0.0)
    r44 = max(rho[3,3].real, 0.0)
    C1  = 2.0*max(0.0, abs(rho[1,2]) - np.sqrt(r11*r44))
    C2  = 2.0*max(0.0, abs(rho[0,3]) - np.sqrt(r22*r33))
    return float(max(C1, C2))


def singlet_fraction(rho):
    psi = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2.0)
    return float(np.clip(np.real(psi.conj() @ rho @ psi), 0.0, 1.0))


def purity(rho):
    return float(np.real(np.trace(rho @ rho)))


def von_neumann_entropy(rho):
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 0.0]
    return float(-np.sum(evals * np.log(evals)))


def compute_observables_grid(rho_rt, r_vals, t_grid, theta_idx=0):
    """
    Compute C, Fs, purity, entropy for one theta slice.

    Returns
    -------
    dict with keys 'C', 'Fs', 'pur', 'S'  each shape (Nr, Nt)
    """
    Nr, Nt = len(r_vals), len(t_grid)
    C   = np.zeros((Nr, Nt))
    Fs  = np.zeros((Nr, Nt))
    pur = np.zeros((Nr, Nt))
    S   = np.zeros((Nr, Nt))

    for j in range(Nr):
        for k in range(Nt):
            rho        = rho_rt[theta_idx][j][k]
            C  [j, k]  = concurrence_xstate(rho)
            Fs [j, k]  = singlet_fraction(rho)
            pur[j, k]  = purity(rho)
            S  [j, k]  = von_neumann_entropy(rho)

    return {"C": C, "Fs": Fs, "pur": pur, "S": S}