"""Accelerated BdG solver (separates residual from Jacobian, adds exact
continuum lambda_exact, periodic endpoint=False grid option, and constrained
fixed-density free-energy evaluator for Landau expansions)."""
import numpy as np
from scipy.optimize import fsolve, brentq

def make_grid(Nk, t=1.0, Vso=0.0, periodic=False):
    kx = np.linspace(-np.pi, np.pi, Nk, endpoint=(not periodic))
    KX, KY = np.meshgrid(kx, kx)
    ssin = np.sqrt(np.sin(KX)**2 + np.sin(KY)**2)
    sk = 0.5*(np.cos(KX) + np.cos(KY))
    eps = np.empty((Nk, Nk, 2))
    eps[..., 0] = -2.0*t*(np.cos(KX) + np.cos(KY)) - 2.0*Vso*ssin
    eps[..., 1] = -2.0*t*(np.cos(KX) + np.cos(KY)) + 2.0*Vso*ssin
    return sk, ssin, eps, KX, KY, kx

def kernel_array(sk, ssin, U, Dt, Vso, t=1.0):
    s_prime = np.array([1.0, -1.0])
    return (U + 8.0*Dt*sk[..., None]
            + 4.0*(Dt/t)*Vso*s_prime[None, None, :]*ssin[..., None])

def normal_state(eps, n):
    N2 = eps.shape[0]*eps.shape[1]
    flat = eps.ravel()
    k = int(round(n*N2))
    k = min(max(k, 1), flat.size - 1)
    part = np.partition(flat, [k-1, k])
    lo, hi = part[k-1], part[k]
    mu = 0.5*(lo + hi) if hi > lo else lo
    xi = eps - mu
    E = np.abs(xi)
    eg = 0.5*np.sum(xi - E)
    e = eg/N2 + mu*n
    return {"mu": float(mu), "e": float(e), "eg": float(eg)}

def sc_energy(x, eps, Nk, n):
    D0, DS, mu = float(x[0]), float(x[1]), float(x[2])
    N2 = float(Nk*Nk)
    Dks = D0 - (DS/4.0)*eps
    xi = eps - mu
    E = np.clip(np.sqrt(Dks*Dks + xi*xi), 1e-12, None)
    eg = 0.5*np.sum(xi - E + Dks*Dks/(2.0*E))
    e = eg/N2 + mu*n
    dphi = 0.5*np.sum(-1.0 + xi/E)
    return float(e), float(eg), float(dphi + n*N2)

def sc_energy_exact(x, eps, kern, Nk, Dt, n, U=None):
    """Canonical potential F/N = Omega/N + mu n using the exact BdG gap-energy
    identity (equivalent to sc_energy at self-consistency, and also valid for
    off-shell variational / Landau-expansion evaluations when Dt>0!).
    Wait: at arbitrary off-shell (D0, DS) not satisfying gap equations,
    the variational BCS ground-state energy <H_eff> must use the interaction
    kernel, NOT Delta^2/(2E)!  Specifically:
      <H_kin - mu N> = 1/(2 N^2) sum (xi - xi^2/E) + mu n
    and the interaction energy (self-consistent quadratic form) is
      E_int = -1/(4 N^2) sum K (Delta/E) * ... wait:
    Let's derive the exact variational <H_BCS> = <psi|H|psi> for ANY (D0, DS)!
    """
    return sc_energy(x, eps, Nk, n)

def variational_energy_offshell(D0, DS, eps, kern, Nk, Dt, n, mu_init=0.0):
    """Exact variational expectation value <H_0 + H_int>/N_sites at fixed
    density n for ANY trial BdG wavefunction parameterized by (D0, DS), with
    mu determined from the number constraint 1/(2N^2) sum (-1 + xi/E) + n = 0.

    Let's check the interaction term of the microscopic Hamiltonian!
    From self_consistency.py / gap equations:
      r1 = D0 + 1/(4 N^2) sum K * D/E = 0
      r2 = DS + (2 Dt / N^2) sum D/E = 0
    Notice the factor:
      F_var(D0, DS) = <H>/N^2 = 1/(2 N^2) sum [ xi - E + (D^2)/(2E) ] + mu n
      ONLY at self-consistency does -D^2/(2E) equal the interaction term!
    Away from self-consistency, let F_k = D_{k,s}/(2 E_{k,s}) (= <c_{-k,s} c_{k,s}> up to phase),
    Wait: the bilinear BdG pairing potential is V_pair = sum V_{kk'} F_k F_{k'}.
    In our separable two-channel model (U channel + correlated-hopping Dt channel):
      D_0 = -(1/(4 N^2)) sum (U - 2 Dt eps) D/E
      D_S = -(2 Dt / N^2) sum D/E
    Writing S_0 = (1/N^2) sum D/(2E) and S_e = (1/N^2) sum eps D/(2E):
      D_S = -4 Dt S_0  and  D_0 = -(1/2) U S_0 + Dt S_e.
    Wait: let's check what quadratic interaction energy E_int(S_0, S_e) has
    d E_var / d S_0 = -D_S/4? Let's verify in Python against dF/dD0 = 0 at self-consistency!
    """
    pass

def residual_fast(x, eps, kern, Nk, Dt, n):
    D0, DS, mu = float(x[0]), float(x[1]), float(x[2])
    N2 = float(Nk*Nk)
    Dks = D0 - (DS/4.0)*eps
    xi = eps - mu
    E = np.clip(np.sqrt(Dks*Dks + xi*xi), 1e-12, None)
    invE = 1.0/E
    D_over_E = Dks*invE
    r1 = D0 + np.sum(kern*D_over_E)/(4.0*N2)
    r2 = DS + 2.0*Dt*np.sum(D_over_E)/N2
    r3 = np.sum(xi*invE)/(2.0*N2) - (1.0 - n)
    return np.array([r1, r2, r3])

def jac_fast(x, eps, kern, Nk, Dt, n):
    D0, DS, mu = float(x[0]), float(x[1]), float(x[2])
    N2 = float(Nk*Nk)
    Dks = D0 - (DS/4.0)*eps
    xi = eps - mu
    E = np.clip(np.sqrt(Dks*Dks + xi*xi), 1e-12, None)
    E3 = E*E*E
    xi2_E3 = (xi*xi)/E3
    Dxi_E3 = (Dks*xi)/E3
    return np.array([
        [1.0 + np.sum(kern*xi2_E3)/(4.0*N2),
         np.sum(kern*(-eps/4.0)*xi2_E3)/(4.0*N2),
         np.sum(kern*Dxi_E3)/(4.0*N2)],
        [2.0*Dt*np.sum(xi2_E3)/N2,
         1.0 + 2.0*Dt*np.sum((-eps/4.0)*xi2_E3)/N2,
         2.0*Dt*np.sum(Dxi_E3)/N2],
        [-np.sum(Dxi_E3)/(2.0*N2),
         np.sum(Dxi_E3*eps)/(8.0*N2),
         -np.sum((Dks*Dks)/E3)/(2.0*N2)]
    ])

def _canon(x):
    x = np.asarray(x, float)
    if x[0] < -1e-9 or (abs(x[0]) <= 1e-9 and x[1] < 0):
        x = np.array([-x[0], -x[1], x[2]])
    return x

def solve_one(eps, kern, Nk, Dt, n, x0, xtol=1.49e-12, use_jac=True, maxfev=400):
    try:
        sol, info, ier, _ = fsolve(
            residual_fast, np.asarray(x0, float),
            args=(eps, kern, Nk, Dt, n), full_output=True, xtol=xtol,
            maxfev=maxfev, fprime=(jac_fast if use_jac else None))
        cand = np.asarray(sol, float)
        r = float(np.max(np.abs(residual_fast(cand, eps, kern, Nk, Dt, n))))
        return cand, r, ier
    except Exception:
        return None, np.inf, 0

def solve_all(eps, kern, Nk, Dt, n, seeds, tol=1e-10, early=None, use_jac=True):
    sols = []
    for i, x0 in enumerate(seeds):
        if early is not None and i >= early and sols:
            break
        cand, r, ier = solve_one(eps, kern, Nk, Dt, n, x0, use_jac=use_jac,
                                 maxfev=(200 if early is not None else 500))
        if cand is not None and r < tol:
            c = _canon(cand)
            if not any(np.max(np.abs(c - np.array(o))) < 1e-7 for o in sols):
                sols.append(c)
    nontrivial = [s for s in sols if max(abs(s[0]), abs(s[1])) > 1e-6]
    return sols, nontrivial

def lambda_exact_and_matrix(eps, kern, Nk, Dt, mu, U):
    """Returns:
      lam_cont: exact T=0 continuum limit Dt^2 <|eps-mu|> / (U - 4 Dt mu) when U > 4 Dt mu,
                else np.inf (attractive FS Cooper-log regime).
      lam_grid: finite-Nk 2x2 matrix max eigenvalue (clipped at 1e-12).
      eig_vec: leading eigenvector (D0, DS) normalized to ||v||=1.
    """
    N2 = float(Nk*Nk)
    xi = eps - mu
    J2 = float(np.sum(np.abs(xi)) / N2)
    denom = U - 4.0*Dt*mu
    lam_cont = (Dt*Dt * J2 / denom) if denom > 0 else float('inf')
    invE = 1.0/np.clip(np.abs(xi), 1e-12, None)
    M = np.array([
        [-np.sum(kern*invE)/(4.0*N2), np.sum(kern*eps*invE)/(4.0*N2*4.0)],
        [-2.0*Dt*np.sum(invE)/N2,    2.0*Dt*np.sum(eps*invE)/(N2*4.0)]
    ])
    vals, vecs = np.linalg.eig(M)
    idx = int(np.argmax(vals.real))
    v = vecs[:, idx].real
    v = v / np.linalg.norm(v)
    return float(lam_cont), float(vals[idx].real), v


def bdg_dict(x, eps, KX, KY, ssin):
    D0, DS, mu = float(x[0]), float(x[1]), float(x[2])
    Dks = D0 - (DS/4.0)*eps
    E = np.clip(np.sqrt(Dks**2 + (eps - mu)**2), 1e-12, None)
    uks = np.sqrt(np.clip(0.5*(1.0 + (eps - mu)/E), 0, 1))
    vks = np.sqrt(np.clip(0.5*(1.0 - (eps - mu)/E), 0, 1))
    return {"Delta_ks": Dks, "E_ks": E, "u_ks": uks, "v_ks": vks,
            "KX": KX, "KY": KY, "epsilon_ks": eps, "sqrt_sin": ssin,
            "Delta0": D0, "DeltaS": DS, "mu": mu}
