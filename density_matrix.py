# =============================================================================
# density_matrix.py
# =============================================================================

import numpy as np


def build_density_matrices(G_uu_polar, F_ud_polar):
    """
    Build 4x4 X-state density matrices for every (theta, r).

    Returns
    -------
    rho_matrices : list[N_theta][N_r] of (4,4) complex arrays
    """
    Nth, Nr = G_uu_polar.shape
    G0      = np.abs(G_uu_polar[:, 0])    # |G(r=0)| per theta

    rho_matrices = []
    for i in range(Nth):
        row = []
        for j in range(Nr):
            g = np.abs(G_uu_polar[i, j]) / G0[i]
            f = np.abs(F_ud_polar[i, j]) / G0[i]
            N = 4.0 - 2.0*g**2 + 2.0*f**2

            rho = np.array([
                [1.0 - g**2,                0,                   0,          0],
                [0,               1.0 + f**2,  -(g**2 + f**2),              0],
                [0,          -(g**2 + f**2),       1.0 + f**2,              0],
                [0,                        0,                   0, 1.0 - g**2],
            ], dtype=complex) / N
            row.append(rho)
        rho_matrices.append(row)

    return rho_matrices


def verify_density_matrices(rho_matrices, atol=1e-6):
    """Check trace=1, Hermitian, PSD for every matrix. Returns True if all OK."""
    all_ok = True
    for i, row in enumerate(rho_matrices):
        for j, rho in enumerate(row):
            tr = rho.trace().real
            if abs(tr - 1.0) > atol:
                print(f"  [density_matrix] trace error (i={i},j={j}): {tr:.6f}")
                all_ok = False
            if not np.allclose(rho, rho.conj().T, atol=atol):
                print(f"  [density_matrix] not Hermitian (i={i},j={j})")
                all_ok = False
            if np.any(np.linalg.eigvalsh(rho) < -atol):
                print(f"  [density_matrix] negative eigenvalue (i={i},j={j})")
                all_ok = False
    return all_ok