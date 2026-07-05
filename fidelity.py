# =============================================================================
# fidelity.py
# =============================================================================

import numpy as np
from utils import data_path


# ---------------------------------------------------------------------------
# Uhlmann fidelity
# ---------------------------------------------------------------------------

def fidelity_uhlmann(rho, sigma):
    def _clean(M):
        M  = 0.5*(M + M.conj().T)
        tr = M.trace().real
        return M / tr if tr > 0 else M

    rho   = _clean(np.asarray(rho,   dtype=complex))
    sigma = _clean(np.asarray(sigma, dtype=complex))

    evals, evecs = np.linalg.eigh(rho)
    evals = np.clip(evals, 0.0, None)
    sqrt_rho = (evecs * np.sqrt(evals)) @ evecs.conj().T

    X   = sqrt_rho @ sigma @ sqrt_rho
    X   = 0.5*(X + X.conj().T)
    lam = np.clip(np.linalg.eigvalsh(X), 0.0, None)
    return float(np.sum(np.sqrt(lam))**2)


# ---------------------------------------------------------------------------
# Noise models (OU dephasing + Lorentzian amplitude damping)
# ---------------------------------------------------------------------------

def eta_dephasing_OU(t, nu, Gamma):
    exp = -(nu**2/(2.0*Gamma**2))*(Gamma*t + np.exp(-Gamma*t) - 1.0)
    return float(np.clip(np.exp(exp), 0.0, 1.0))


def kraus_dephasing_from_eta(eta):
    eta = float(np.clip(eta, 0.0, 1.0))
    K0  = np.sqrt(0.5*(1.0+eta)) * np.eye(2, dtype=complex)
    K1  = np.sqrt(0.5*(1.0-eta)) * np.diag([1.0, -1.0]).astype(complex)
    return [K0, K1]


def eta_amp_lorentz(t, gamma, Gamma):
    disc = Gamma**2 - 2.0*gamma*Gamma
    if disc >= 0.0:
        d   = np.sqrt(disc) if disc > 0 else 1e-15
        val = np.exp(-Gamma*t)*(np.cosh(0.5*d*t)+(Gamma/d)*np.sinh(0.5*d*t))**2
    else:
        d   = np.sqrt(-disc)
        val = np.exp(-Gamma*t)*(np.cos(0.5*d*t)+(Gamma/d)*np.sin(0.5*d*t))**2
    return float(np.clip(val, 0.0, 1.0))


def kraus_amp_from_eta(eta):
    eta = float(np.clip(eta, 0.0, 1.0))
    K0  = np.array([[1.0, 0.0],[0.0, np.sqrt(eta)]], dtype=complex)
    K1  = np.array([[0.0, np.sqrt(1.0-eta)],[0.0, 0.0]], dtype=complex)
    return [K0, K1]


# ---------------------------------------------------------------------------
# Two-qubit evolution (for fidelity module — separate from evolution.py)
# ---------------------------------------------------------------------------

def _evolve_rho_fid(rho0, t, *, kind, bath,
                    nuA, GammaA, nuB, GammaB,
                    gammaA, GammaAD_A, gammaB, GammaAD_B):
    rho0 = np.asarray(rho0, dtype=complex)

    if bath == "independent":
        if kind == "dephasing":
            KA = kraus_dephasing_from_eta(eta_dephasing_OU(t, nuA, GammaA))
            KB = kraus_dephasing_from_eta(eta_dephasing_OU(t, nuB, GammaB))
        elif kind == "amp":
            KA = kraus_amp_from_eta(eta_amp_lorentz(t, gammaA, GammaAD_A))
            KB = kraus_amp_from_eta(eta_amp_lorentz(t, gammaB, GammaAD_B))
        else:
            raise ValueError(f"Unknown kind '{kind}'")

        rho_t = np.zeros_like(rho0, dtype=complex)
        for A in KA:
            for B in KB:
                K      = np.kron(A, B)
                rho_t += K @ rho0 @ K.conj().T

    elif bath == "common":
        if kind != "dephasing":
            raise ValueError("Common bath only for dephasing.")
        s2    = (nuA**2/GammaA**2)*(GammaA*t + np.exp(-GammaA*t) - 1.0)
        Jz    = np.array([+2.0, 0.0, 0.0, -2.0])
        delta = Jz[:,None] - Jz[None,:]
        rho_t = np.exp(-0.5*s2*delta**2) * rho0
    else:
        raise ValueError(f"Unknown bath '{bath}'")

    rho_t  = 0.5*(rho_t + rho_t.conj().T)
    tr     = rho_t.trace().real
    rho_t /= max(tr, 1e-15)
    return rho_t


# ---------------------------------------------------------------------------
# Fidelity grid
# ---------------------------------------------------------------------------

def compute_fidelity_grid(rho_matrices, t_grid, theta_idx=0,
                           reference_type="initial", reference_idx=None,
                           kind="amp", bath="independent",
                           nuA=0.2, GammaA=0.05, nuB=0.2, GammaB=0.05,
                           gammaA=0.5, GammaAD_A=0.1,
                           gammaB=0.5, GammaAD_B=0.1,
                           verbose=True):
    """
    Compute F(r, t) for a fixed theta slice.

    reference_type
    --------------
    'initial' : per-r reference rho_j(t=0)  [F(r,0) = 1 exactly]
    'pure'    : |Psi^-><Psi^-|
    'fixed'   : rho at (r_idx, t_idx) evolved to t_ref
    """
    Nr = len(rho_matrices[theta_idx])
    Nt = len(t_grid)
    F_rt = np.zeros((Nr, Nt), dtype=float)

    noise_kw = dict(
        kind=kind, bath=bath,
        nuA=nuA, GammaA=GammaA, nuB=nuB, GammaB=GammaB,
        gammaA=gammaA, GammaAD_A=GammaAD_A,
        gammaB=gammaB, GammaAD_B=GammaAD_B,
    )

    # --- prepare global reference -----------------------------------------
    if reference_type == "pure":
        psi = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2.0)
        rho_ref_global = np.outer(psi, psi.conj())

    elif reference_type == "fixed":
        if reference_idx is None:
            raise ValueError("'fixed' needs reference_idx=(r_idx, t_idx)")
        r_idx, t_idx = reference_idx
        t_ref = t_grid[min(t_idx, Nt-1)]
        rho_ref_global = _evolve_rho_fid(
            rho_matrices[theta_idx][r_idx], t_ref, **noise_kw
        )

    elif reference_type == "initial":
        rho_ref_global = None

    else:
        raise ValueError(f"Unknown reference_type '{reference_type}'")

    # --- main loop --------------------------------------------------------
    for j in range(Nr):
        if verbose and j % max(1, Nr//5) == 0:
            print(f"  [fidelity] r-index {j+1}/{Nr}  "
                  f"(ref={reference_type})")

        rho0_j  = rho_matrices[theta_idx][j]
        rho_ref = rho0_j if reference_type == "initial" else rho_ref_global

        for k, t in enumerate(t_grid):
            rho_t      = _evolve_rho_fid(rho0_j, t, **noise_kw)
            F_rt[j, k] = fidelity_uhlmann(rho_t, rho_ref)

        if (reference_type == "initial"
                and not np.isclose(F_rt[j, 0], 1.0, atol=1e-5)):
            print(f"  [fidelity] WARNING: F(r_idx={j}, t=0) = "
                  f"{F_rt[j,0]:.6g} (expected 1)")

    return F_rt, rho_ref_global


# ---------------------------------------------------------------------------
# Decay analysis
# ---------------------------------------------------------------------------

def analyze_fidelity_decay(r_vals, t_grid, F_rt,
                            thresholds=(0.9, 0.8, 0.7, 0.5)):
    decay_times = {}
    for th in thresholds:
        arr = np.full(len(r_vals), np.nan)
        for j in range(len(r_vals)):
            row = F_rt[j, :]
            idx = np.argmax(row < th)
            if row[idx] < th:
                arr[j] = t_grid[idx]
        decay_times[th] = arr

    F_init = F_rt[:, 0]
    F_fin  = F_rt[:, -1]

    print("=" * 50)
    print("Fidelity Decay Analysis")
    print("=" * 50)
    print(f"  F at t=0    : [{np.nanmin(F_init):.4f}, {np.nanmax(F_init):.4f}]")
    print(f"  F at t=tmax : [{np.nanmin(F_fin):.4f},  {np.nanmax(F_fin):.4f}]")
    print(f"  Global      : [{np.nanmin(F_rt):.4f},   {np.nanmax(F_rt):.4f}]")
    for th in thresholds:
        v = decay_times[th][~np.isnan(decay_times[th])]
        if v.size:
            print(f"  t* (F<{th}): mean={np.mean(v):.2f}  std={np.std(v):.2f}")
        else:
            print(f"  t* (F<{th}): never reached")

    return {
        "decay_times"   : decay_times,
        "F_initial"     : F_init,
        "F_final"       : F_fin,
        "fidelity_range": (float(np.nanmin(F_rt)), float(np.nanmax(F_rt))),
    }


# ---------------------------------------------------------------------------
# Compare initial vs singlet
# ---------------------------------------------------------------------------

def compare_fidelity_references(rho_matrices, r_vals, t_grid,
                                 theta_idx=0, verbose=True, **noise_params):
    print("[fidelity] Computing F vs INITIAL ...")
    F_init, _ = compute_fidelity_grid(
        rho_matrices, t_grid, theta_idx,
        reference_type="initial", verbose=verbose, **noise_params
    )
    print("[fidelity] Computing F vs PURE SINGLET ...")
    F_pure, _ = compute_fidelity_grid(
        rho_matrices, t_grid, theta_idx,
        reference_type="pure", verbose=verbose, **noise_params
    )
    return F_init, F_pure


# ---------------------------------------------------------------------------
# Save fidelity data
# ---------------------------------------------------------------------------

def save_fidelity_data(F_init, F_pure, r_vals, t_grid,
                        theta_deg, prefix="fidelity"):
    """Save fidelity arrays to data/<prefix>_fidelity.npz"""
    out = data_path(f"{prefix}_fidelity.npz")
    np.savez_compressed(
        out,
        F_init=F_init,
        F_pure=F_pure,
        r_vals=r_vals,
        t_grid=t_grid,
        theta_deg=float(theta_deg),
    )
    print(f"[fidelity] Saved -> {out}")
    return out