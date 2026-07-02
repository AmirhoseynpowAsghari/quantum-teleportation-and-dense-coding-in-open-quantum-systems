# =============================================================================
# density_matrix.py
# Build 4×4 two-particle density matrices ρ(θ, r) from the Green's functions.
# =============================================================================

import numpy as np


def build_density_matrices(G_uu_polar, F_ud_polar):
    """
    Construct the 4×4 X-state density matrix for every (θ_i, r_j).

    The matrix is normalised so that Tr[ρ] = 1.

    Parameters
    ----------
    G_uu_polar : complex array, shape (Nθ, Nr)
    F_ud_polar : complex array, shape (Nθ, Nr)

    Returns
    -------
    rho_matrices : list of lists  [Nθ][Nr], each element is a (4,4) complex ndarray
    """
    Nθ, Nr = G_uu_polar.shape

    # Reference value: |G(r=0)| for each angle
    G0 = np.abs(G_uu_polar[:, 0])  # shape (Nθ,)

    rho_matrices = []
    for i in range(Nθ):
        rho_theta = []
        for j in range(Nr):
            g_r = np.abs(G_uu_polar[i, j]) / G0[i]
            f_r = np.abs(F_ud_polar[i, j]) / G0[i]

            N = 4.0 - 2.0*g_r**2 + 2.0*f_r**2   # normalisation

            rho = np.array([
                [1.0 - g_r**2,              0.0,                        0.0,              0.0],
                [0.0,                  1.0 + f_r**2,   -(g_r**2 + f_r**2),               0.0],
                [0.0,          -(g_r**2 + f_r**2),      1.0 + f_r**2,                    0.0],
                [0.0,                       0.0,                        0.0,    1.0 - g_r**2]
            ], dtype=complex) / N

            rho_theta.append(rho)
        rho_matrices.append(rho_theta)

    return rho_matrices


def verify_density_matrices(rho_matrices, atol=1e-6):
    """
    Quick sanity checks: trace = 1, Hermitian, positive semi-definite.

    Returns True if all tests pass.
    """
    all_ok = True
    for i, row in enumerate(rho_matrices):
        for j, rho in enumerate(row):
            tr = rho.trace().real
            if abs(tr - 1.0) > atol:
                print(f"[density_matrix] Trace error at (i={i},j={j}): Tr={tr:.6f}")
                all_ok = False
            if not np.allclose(rho, rho.conj().T, atol=atol):
                print(f"[density_matrix] Not Hermitian at (i={i},j={j})")
                all_ok = False
            evals = np.linalg.eigvalsh(rho)
            if np.any(evals < -atol):
                print(f"[density_matrix] Negative eigenvalue at (i={i},j={j}): {evals.min():.2e}")
                all_ok = False
    return all_ok


if __name__ == "__main__":
    import numpy as np
    # Minimal smoke test with random Green's functions
    rng = np.random.default_rng(0)
    Nθ, Nr = 3, 5
    G_test = rng.random((Nθ, Nr)) + 1j*rng.random((Nθ, Nr))
    F_test = rng.random((Nθ, Nr)) * 0.1 + 1j*rng.random((Nθ, Nr)) * 0.1
    rho_m  = build_density_matrices(G_test, F_test)
    ok = verify_density_matrices(rho_m)
    print(f"Smoke-test passed: {ok}")