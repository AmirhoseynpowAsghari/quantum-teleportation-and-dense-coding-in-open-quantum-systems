# =============================================================================
# evolution.py
# Evolve 4×4 density matrices under non-Markovian noise channels.
#
# Supported channels
# ------------------
# kind='dephasing', bath='independent'  – two-qubit independent OU dephasing
# kind='amp',       bath='independent'  – two-qubit independent amplitude damping
# kind='dephasing', bath='common'       – collective dephasing (shared Gaussian bath)
# =============================================================================

import numpy as np
from noise_channels import get_kraus_pair


# ---------------------------------------------------------------------------
# Single density-matrix evolution
# ---------------------------------------------------------------------------

def evolve_rho(rho0, t, *,
               kind="dephasing", bath="independent",
               gammaA=0.05, GammaA=0.01,
               gammaB=0.05, GammaB=0.01):
    """
    Evolve one 4×4 density matrix rho0 to time t.

    Parameters
    ----------
    rho0   : (4,4) complex array   initial density matrix
    t      : float ≥ 0             evolution time
    kind   : str                   'dephasing' or 'amp'
    bath   : str                   'independent' or 'common'
    gammaA, GammaA : float         bath parameters for qubit A
    gammaB, GammaB : float         bath parameters for qubit B

    Returns
    -------
    rho_t : (4,4) complex array   evolved density matrix (Tr=1, Hermitian)
    """
    rho0 = np.asarray(rho0, dtype=complex)

    if bath == "independent":
        Ka, Kb = get_kraus_pair(t, kind, gammaA, GammaA, gammaB, GammaB)

        rho_t = np.zeros_like(rho0, dtype=complex)
        for A in Ka:
            for B in Kb:
                K      = np.kron(A, B)           # 4×4 two-qubit Kraus operator
                rho_t += K @ rho0 @ K.conj().T

    elif bath == "common":
        if kind != "dephasing":
            raise ValueError("Common bath is implemented only for 'dephasing'.")

        # Collective dephasing: ρ_{mn} → ρ_{mn} · exp(-½ σ² (Jz_m − Jz_n)²)
        # Basis order: |↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩  →  Jz eigenvalues +1,0,0,−1
        s2 = gammaA * (t + (np.exp(-GammaA * t) - 1.0) / GammaA)
        Jz = np.array([+1.0, 0.0, 0.0, -1.0])
        delta = Jz[:, None] - Jz[None, :]       # (4,4)
        M     = np.exp(-0.5 * s2 * delta**2)    # modulation matrix
        rho_t = M * rho0                         # element-wise product

    else:
        raise ValueError(f"Unknown bath '{bath}'. Use 'independent' or 'common'.")

    # Enforce Hermiticity and unit trace (absorb small numerical drift)
    rho_t  = 0.5 * (rho_t + rho_t.conj().T)
    tr     = rho_t.trace().real
    rho_t /= tr
    return rho_t


# ---------------------------------------------------------------------------
# Full grid evolution
# ---------------------------------------------------------------------------

def evolve_grid(rho_matrices, t_grid, *,
                kind="dephasing", bath="independent",
                gammaA=0.05, GammaA=0.01,
                gammaB=0.05, GammaB=0.01,
                verbose=True):
    """
    Evolve all (θ, r) density matrices over a time grid.

    Parameters
    ----------
    rho_matrices : list of lists [Nθ][Nr] of (4,4) arrays
    t_grid       : 1-D array of time values
    (other kwargs forwarded to evolve_rho)

    Returns
    -------
    rho_rt : list [Nθ][Nr][Nt] of (4,4) complex arrays
    """
    Nθ  = len(rho_matrices)
    Nr  = len(rho_matrices[0]) if Nθ else 0
    Nt  = len(t_grid)

    # Pre-allocate with None for clarity
    rho_rt = [[[None] * Nt for _ in range(Nr)] for _ in range(Nθ)]

    kw = dict(kind=kind, bath=bath,
              gammaA=gammaA, GammaA=GammaA,
              gammaB=gammaB, GammaB=GammaB)

    for i in range(Nθ):
        if verbose:
            print(f"[evolution] θ index {i+1}/{Nθ}")
        for j in range(Nr):
            rho0 = rho_matrices[i][j]
            for k, t_k in enumerate(t_grid):
                rho_rt[i][j][k] = evolve_rho(rho0, t_k, **kw)

    return rho_rt


if __name__ == "__main__":
    # Smoke test with a Bell state
    rho_bell = np.array([
        [0.5,  0.0,  0.0,  0.5],
        [0.0,  0.0,  0.0,  0.0],
        [0.0,  0.0,  0.0,  0.0],
        [0.5,  0.0,  0.0,  0.5],
    ], dtype=complex)

    for kind in ("dephasing", "amp"):
        for bath in (["independent"] + (["common"] if kind == "dephasing" else [])):
            rho_t = evolve_rho(rho_bell, 10.0,
                               kind=kind, bath=bath,
                               gammaA=0.05, GammaA=0.01,
                               gammaB=0.05, GammaB=0.01)
            print(f"[smoke] kind={kind} bath={bath} "
                  f"Tr={rho_t.trace().real:.6f} "
                  f"evals={np.linalg.eigvalsh(rho_t).round(4)}")