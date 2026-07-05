# =============================================================================
# evolution.py
# =============================================================================

import numpy as np
from noise_channels import get_kraus_pair


def evolve_rho(rho0, t, *, kind="amp", bath="independent",
               gammaA=0.05, GammaA=0.01, gammaB=0.05, GammaB=0.01):
    rho0 = np.asarray(rho0, dtype=complex)

    if bath == "independent":
        Ka, Kb = get_kraus_pair(t, kind, gammaA, GammaA, gammaB, GammaB)
        rho_t  = np.zeros_like(rho0, dtype=complex)
        for A in Ka:
            for B in Kb:
                K      = np.kron(A, B)
                rho_t += K @ rho0 @ K.conj().T

    elif bath == "common":
        if kind != "dephasing":
            raise ValueError("Common bath only for dephasing.")
        s2    = gammaA * (t + (np.exp(-GammaA*t) - 1.0)/GammaA)
        Jz    = np.array([+1.0, 0.0, 0.0, -1.0])
        delta = Jz[:, None] - Jz[None, :]
        rho_t = np.exp(-0.5*s2*delta**2) * rho0

    else:
        raise ValueError(f"Unknown bath '{bath}'")

    rho_t  = 0.5*(rho_t + rho_t.conj().T)
    tr     = rho_t.trace().real
    rho_t /= max(tr, 1e-15)
    return rho_t


def evolve_grid(rho_matrices, t_grid, *, kind="amp", bath="independent",
                gammaA=0.05, GammaA=0.01, gammaB=0.05, GammaB=0.01,
                verbose=True):
    Nth = len(rho_matrices)
    Nr  = len(rho_matrices[0])
    Nt  = len(t_grid)
    kw  = dict(kind=kind, bath=bath,
               gammaA=gammaA, GammaA=GammaA,
               gammaB=gammaB, GammaB=GammaB)

    rho_rt = [[[None]*Nt for _ in range(Nr)] for _ in range(Nth)]

    for i in range(Nth):
        if verbose:
            print(f"  [evolution] theta {i+1}/{Nth}")
        for j in range(Nr):
            for k, t_k in enumerate(t_grid):
                rho_rt[i][j][k] = evolve_rho(rho_matrices[i][j], t_k, **kw)

    return rho_rt