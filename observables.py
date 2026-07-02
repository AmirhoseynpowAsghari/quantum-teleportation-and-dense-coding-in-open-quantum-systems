# =============================================================================
# observables.py
# Quantum-information observables for 4×4 two-qubit density matrices.
# =============================================================================

import numpy as np


def concurrence_xstate(rho):
    """
    Concurrence for an X-state density matrix.

    An X-state has non-zero elements only at (0,0),(1,1),(2,2),(3,3),(0,3),(3,0),
    (1,2),(2,1) in the {|00⟩,|01⟩,|10⟩,|11⟩} basis.

    C = max(0, C₁, C₂)
    where
        C₁ = 2|ρ₂₃| − 2√(ρ₁₁ ρ₄₄)
        C₂ = 2|ρ₁₄| − 2√(ρ₂₂ ρ₃₃)

    Parameters
    ----------
    rho : (4,4) complex array

    Returns
    -------
    float in [0, 1]
    """
    ρ11 = max(rho[0, 0].real, 0.0)
    ρ22 = max(rho[1, 1].real, 0.0)
    ρ33 = max(rho[2, 2].real, 0.0)
    ρ44 = max(rho[3, 3].real, 0.0)
    ρ14 = rho[0, 3]
    ρ23 = rho[1, 2]

    C1 = 2.0 * max(0.0, abs(ρ23) - np.sqrt(ρ11 * ρ44))
    C2 = 2.0 * max(0.0, abs(ρ14) - np.sqrt(ρ22 * ρ33))
    return float(max(C1, C2))


def singlet_fraction(rho):
    """
    Overlap of ρ with the singlet state |Ψ⁻⟩ = (|01⟩ − |10⟩)/√2.

    F_s = ⟨Ψ⁻|ρ|Ψ⁻⟩

    Parameters
    ----------
    rho : (4,4) complex array

    Returns
    -------
    float in [0, 1]
    """
    psi = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2.0)
    return float(np.real(psi.conj() @ rho @ psi))


def purity(rho):
    """Tr[ρ²] ∈ [1/d, 1] where d=4 for a two-qubit system."""
    return float(np.real(np.trace(rho @ rho)))


def von_neumann_entropy(rho):
    """
    S(ρ) = −Tr[ρ ln ρ]  (nats, natural log).

    Returns
    -------
    float ≥ 0
    """
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 0.0]                # discard numerical negatives
    return float(-np.sum(evals * np.log(evals)))


def compute_observables_grid(rho_rt, r_vals, t_grid, theta_idx=0):
    """
    Compute C(r,t), F_s(r,t), purity(r,t), S(r,t) for one angle slice.

    Parameters
    ----------
    rho_rt    : list [Nθ][Nr][Nt]
    r_vals    : 1-D array
    t_grid    : 1-D array
    theta_idx : int  which θ slice to use

    Returns
    -------
    dict with keys 'C', 'Fs', 'pur', 'S',
        each a (Nr × Nt) float array.
    """
    Nr = len(r_vals)
    Nt = len(t_grid)

    C   = np.zeros((Nr, Nt))
    Fs  = np.zeros((Nr, Nt))
    pur = np.zeros((Nr, Nt))
    S   = np.zeros((Nr, Nt))

    for j in range(Nr):
        for k in range(Nt):
            rho = rho_rt[theta_idx][j][k]
            C  [j, k] = concurrence_xstate(rho)
            Fs [j, k] = singlet_fraction(rho)
            pur[j, k] = purity(rho)
            S  [j, k] = von_neumann_entropy(rho)

    return {"C": C, "Fs": Fs, "pur": pur, "S": S}


if __name__ == "__main__":
    rho_bell = np.array([
        [0.5, 0, 0, 0.5],
        [0,   0, 0, 0  ],
        [0,   0, 0, 0  ],
        [0.5, 0, 0, 0.5],
    ], dtype=complex)

    print(f"Concurrence  = {concurrence_xstate(rho_bell):.4f}")   # expect 1
    print(f"Singlet frac = {singlet_fraction(rho_bell):.4f}")      # expect 0
    print(f"Purity       = {purity(rho_bell):.4f}")               # expect 1
    print(f"VN entropy   = {von_neumann_entropy(rho_bell):.4f}")   # expect 0