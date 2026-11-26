"""TASK 8 -- Comprehensive Electron-Density (n) QPT Study:
Equation (28) of arXiv:1801.08988 (e_GS = E_g/N + mu*n) vs. Two-Qubit Concurrence C(n).

Evaluates across the full band n in (0, 2) on converged periodic (endpoint=False)
Nk=401 and Nk=801 grids:
  1. Exact Eq. (28) canonical energy e_GS(n), grand energy E_g/N, thermodynamic
     identities (de_GS/dn = mu, d^2 e_GS/dn^2 = dmu/dn), and condensation energy dF.
  2. Dual pairing boundaries (re-entrant low-n nodal s_pm pocket n_{c,1} when
     U < U_c(0^+) = (2*Dt - 4)*Dt*|E_min|, and main high-n pairing onset n_{c,2})
     for U=75, 115 and V_SO=0.0, 0.5, 0.7.
  3. Normal-state logarithmic van Hove saddle points (n_vH-, n_vH+) and Dirac-cone
     Lifshitz transitions (Dirac0 at (0,pi), Dirac4 at (pi,pi)) vs. SC smearing.
  4. Vectorized exact conditional-Wick concurrence C(n), raw Wootters C_raw(n),
     Fermi-smearing decomposition C_normonly(n), dC/dn, d^2C/dn^2 across 6 separations
     and both 'spin' and 'band' bases, demonstrating both real continuum van Hove/Dirac
     signatures in C_N(n) and false ESD sudden-death kinks shifting with r.
  5. Nk convergence (101, 201, 401, 801) and T=0 vs T=1e-3t derivative audit.
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np
from scipy.optimize import brentq

import fast_solver as fs
import helicity_density_matrix as hdm

R_KEYS = ("0.25_0", "0.35_0", "0.5_0", "0.75_0", "1_0", "0.5_0.5")
R_VECS = {
    "0.25_0": np.array([0.25, 0.0]),
    "0.35_0": np.array([0.35, 0.0]),
    "0.5_0": np.array([0.50, 0.0]),
    "0.75_0": np.array([0.75, 0.0]),
    "1_0": np.array([1.00, 0.0]),
    "0.5_0.5": np.array([0.50, 0.50]),
}
DT = 4.5


def raw_wootters(rho: np.ndarray) -> tuple[float, float]:
    rho = np.asarray(rho, dtype=complex)
    yy = np.kron(hdm.PAULI_Y, hdm.PAULI_Y)
    sf = yy @ rho.conj() @ yy
    ev = np.linalg.eigvals(rho @ sf)
    roots = np.sort(np.sqrt(np.clip(ev.real, 0.0, None)))[::-1]
    raw = float(roots[0] - np.sum(roots[1:]))
    return max(0.0, raw), raw


def precompute_r_cache(KX: np.ndarray, KY: np.ndarray, ssin: np.ndarray,
                       r_keys=R_KEYS) -> dict:
    nonzero = ssin > 1e-14
    hel_phase = np.ones_like(KX, dtype=complex)
    hel_phase[nonzero] = (np.sin(KY[nonzero]) - 1j * np.sin(KX[nonzero])) / ssin[nonzero]
    phases = np.stack([np.exp(1j * (KX * R_VECS[k][0] + KY * R_VECS[k][1]))
                       for k in r_keys], axis=0)
    return {
        "r_keys": list(r_keys),
        "r_vecs": [R_VECS[k] for k in r_keys],
        "hel_phase": hel_phase,
        "nonzero": nonzero,
        "phases": phases,
    }


def eval_fast_metrics(bdg: dict, cache: dict, nondegenerate_soc: bool,
                      bases=("spin", "band")) -> dict:
    v2 = np.abs(bdg["v_ks"]) ** 2
    pair_raw = bdg["Delta_ks"] / (2.0 * bdg["E_ks"])
    N2 = float(bdg["KX"].size)
    phases = cache["phases"]
    hp = cache["hel_phase"]
    nz = cache["nonzero"]
    out = {}

    for basis in bases:
        pair = np.where(nz[..., None], pair_raw, 0.0) if (basis == "band" or nondegenerate_soc) else pair_raw
        corrs = []
        if basis == "band":
            n_avg = np.sum(v2, axis=(0, 1)) / N2
            N_A = np.diag(n_avg).astype(complex)
            N_AB_diag = np.einsum("rij,ijs->rs", phases, v2) / N2
            f_mult = pair * np.array([1.0, -1.0])[None, None, :] * hp.conj()[..., None]
            F_AB_diag = np.einsum("rij,ijs->rs", phases.conj(), f_mult) / N2
            for ir, r in enumerate(cache["r_vecs"]):
                N_AB = np.diag(N_AB_diag[ir]).astype(complex)
                F_AB = np.diag(F_AB_diag[ir]).astype(complex)
                if np.array_equal(r, np.zeros(2)):
                    F_AB = 0.5 * (F_AB - F_AB.T)
                corrs.append((N_A, N_AB, F_AB))
        else:
            v_sum = 0.5 * (v2[..., 0] + v2[..., 1])
            v_diff = 0.5 * (v2[..., 0] - v2[..., 1])
            n00A = np.sum(v_sum) / N2
            n01A = np.sum(v_diff * hp) / N2
            N_A = np.array([[n00A, n01A], [np.conj(n01A), n00A]], dtype=complex)
            p0, p1 = pair[..., 0], pair[..., 1]
            f_plus = 0.5 * (p0 + p1)
            f_minus = 0.5 * (p0 - p1)
            n00 = np.einsum("rij,ij->r", phases, v_sum) / N2
            n01 = np.einsum("rij,ij->r", phases, v_diff * hp) / N2
            n10 = np.einsum("rij,ij->r", phases, v_diff * hp.conj()) / N2
            f00 = np.einsum("rij,ij->r", phases.conj(), f_minus * hp.conj()) / N2
            f01 = -np.einsum("rij,ij->r", phases.conj(), f_plus) / N2
            f11 = -np.einsum("rij,ij->r", phases.conj(), f_minus * hp) / N2
            for ir, r in enumerate(cache["r_vecs"]):
                N_AB = np.array([[n00[ir], n01[ir]], [n10[ir], n00[ir]]], dtype=complex)
                F_AB = np.array([[f00[ir], f01[ir]], [-f01[ir], f11[ir]]], dtype=complex)
                if np.array_equal(r, np.zeros(2)):
                    F_AB = 0.5 * (F_AB - F_AB.T)
                corrs.append((N_A, N_AB, F_AB))

        bdict = {}
        for rk, (N_A, N_AB, F_AB) in zip(cache["r_keys"], corrs):
            corr = dict(N_A=N_A, N_B=N_A.copy(), N_AB=N_AB, N_BA=N_AB.conj().T,
                        F_AB=F_AB, F_BA=-F_AB.T)
            rho, w = hdm.conditional_two_qubit_density(corr)
            C, C_raw = raw_wootters(rho)
            Fs = float(np.real(hdm.SINGLET.conj() @ rho @ hdm.SINGLET))

            corr0 = dict(N_A=N_A, N_B=N_A.copy(), N_AB=N_AB, N_BA=N_AB.conj().T,
                         F_AB=np.zeros((2, 2), dtype=complex),
                         F_BA=np.zeros((2, 2), dtype=complex))
            rho0, _ = hdm.conditional_two_qubit_density(corr0)
            C0, C0_raw = raw_wootters(rho0)

            bdict[rk] = {
                "C": float(C),
                "C_raw": float(C_raw),
                "C_normonly": float(C0),
                "C_normonly_raw": float(C0_raw),
                "Fs": float(Fs),
                "weight": float(w),
            }
        out[basis] = bdict
    return out


def normal_bdg_exact_or_T(eps: np.ndarray, KX: np.ndarray, KY: np.ndarray,
                         ssin: np.ndarray, n: float, T: float = 0.0) -> tuple[dict, float, float, float]:
    N2 = float(KX.size)
    if T <= 0.0:
        nrm = fs.normal_state(eps, n)
        mu, e, eg = nrm["mu"], nrm["e"], nrm["eg"] / N2
        xi = eps - mu
        E = np.clip(np.abs(xi), 1e-12, None)
        u = np.where(xi > 0, 1.0, np.where(xi < 0, 0.0, np.sqrt(0.5)))
        v = np.where(xi < 0, 1.0, np.where(xi > 0, 0.0, np.sqrt(0.5)))
        bdg = {
            "Delta_ks": np.zeros_like(eps),
            "E_ks": E,
            "u_ks": u,
            "v_ks": v,
            "KX": KX,
            "KY": KY,
            "epsilon_ks": eps,
            "sqrt_sin": ssin,
            "Delta0": 0.0,
            "DeltaS": 0.0,
            "mu": float(mu),
        }
        return bdg, float(mu), float(e), float(eg)

    mu = float(brentq(
        lambda m: float(np.sum(1.0 / (np.exp(np.clip((eps - m) / T, -200.0, 200.0)) + 1.0)) / N2 - n),
        float(eps.min()) - 1.0,
        float(eps.max()) + 1.0,
        xtol=1e-13,
    ))
    x_raw = (eps - mu) / T
    x = np.clip(x_raw, -200.0, 200.0)
    f = 1.0 / (np.exp(x) + 1.0)
    # Exact log(1 + exp(-x_raw)) without clipping away -(eps-mu)/T for deep occupied states:
    omega = float(-T * np.sum(np.where(-x_raw > 50.0, -x_raw,
                                       np.log1p(np.exp(np.clip(-x_raw, -200.0, 50.0))))) / N2)
    e = float(omega + mu * n)
    bdg = {
        "Delta_ks": np.zeros_like(eps),
        "E_ks": np.clip(np.abs(eps - mu), 1e-12, None),
        "u_ks": np.sqrt(np.clip(1.0 - f, 0.0, 1.0)),
        "v_ks": np.sqrt(np.clip(f, 0.0, 1.0)),
        "KX": KX,
        "KY": KY,
        "epsilon_ks": eps,
        "sqrt_sin": ssin,
        "Delta0": 0.0,
        "DeltaS": 0.0,
        "mu": mu,
    }
    return bdg, mu, e, omega


def make_seeds(mu: float, prev=None) -> list[np.ndarray]:
    seeds = []
    if prev is not None:
        p = np.asarray(prev, float)
        seeds += [p, np.array([p[0], -p[1], p[2]], float)]
    seeds += [np.array(s, float) for s in [
        [1e-4, 4e-4, mu],
        [-0.02, 0.025, mu],
        [-0.06, 0.07, mu],
        [0.01, -0.05, mu],
        [0.05, -0.25, mu],
        [0.15, -0.75, mu],
        [0.25, -1.40, mu],
        [0.30, 1.60, mu],
    ]]
    return seeds


def run_density_sweep(name: str, ns: np.ndarray, U: float, V_SO: float, Nk: int,
                      r_keys=R_KEYS, bases=("spin", "band"), T_norm: float = 0.0) -> dict:
    t0 = time.time()
    sk, ssin, eps, KX, KY, _ = fs.make_grid(Nk, t=1.0, Vso=V_SO, periodic=True)
    kern = fs.kernel_array(sk, ssin, U, DT, V_SO)
    cache = precompute_r_cache(KX, KY, ssin, r_keys=r_keys)
    nondeg = bool(np.max(np.abs(eps[..., 0] - eps[..., 1])) > 1e-12)
    N2 = float(Nk * Nk)

    rows = []
    prev = None
    for n in ns:
        n = float(n)
        bdg_N, muN, eN, egN = normal_bdg_exact_or_T(eps, KX, KY, ssin, n, T=T_norm)
        lam_c, lam_g, _ = fs.lambda_exact_and_matrix(eps, kern, Nk, DT, muN, U)

        nt = []
        if T_norm <= 0.0 and lam_c > 0.992:
            _, nt = fs.solve_all(eps, kern, Nk, DT, n, make_seeds(muN, prev),
                                 tol=1e-10, early=2, use_jac=True)

        m_N = eval_fast_metrics(bdg_N, cache, nondeg, bases=bases)

        if nt:
            cands = [(fs.sc_energy(s, eps, Nk, n), s) for s in nt]
            cands.sort(key=lambda t: t[0][0])
            if prev is None:
                (e_sc, eg_sc, id_err), x = cands[0]
            else:
                (e_sc, eg_sc, id_err), x = min(
                    cands,
                    key=lambda t: max(abs(t[1][0] - prev[0]),
                                      abs(t[1][1] - prev[1]),
                                      6.0 * abs(t[1][2] - prev[2])),
                )
            prev = x
            D0, DS, muS = float(x[0]), float(x[1]), float(x[2])
            egS = float(eg_sc / N2)
            dF = float(e_sc - eN)
            bdg_SC = fs.bdg_dict(x, eps, KX, KY, ssin)
            m_SC = eval_fast_metrics(bdg_SC, cache, nondeg, bases=bases)
            E_min = float(np.min(bdg_SC["E_ks"]))
            D_FS = float(D0 - 0.25 * DS * muS)
            sc_flag = True
        else:
            prev = None
            D0 = DS = D_FS = 0.0
            muS = float(muN)
            e_sc = float(eN)
            egS = float(egN)
            dF = 0.0
            id_err = 0.0
            m_SC = m_N
            E_min = 0.0
            sc_flag = False

        rows.append({
            "n": n,
            "sc": sc_flag,
            "lam_cont": float(lam_c),
            "lam_grid": float(lam_g),
            "D0": D0,
            "DS": DS,
            "D_FS": D_FS,
            "E_min": E_min,
            "muN": float(muN),
            "muS": muS,
            "eN": float(eN),
            "egN": float(egN),
            "eS": float(e_sc),
            "egS": egS,
            "dF": dF,
            "dphi_id_err": float(id_err),
            "sc_metrics": m_SC,
            "norm_metrics": m_N,
        })

    # Compute Eq. (28) & concurrence derivatives along n
    n_arr = np.array([r["n"] for r in rows])
    eN_arr = np.array([r["eN"] for r in rows])
    eS_arr = np.array([r["eS"] for r in rows])
    muN_arr = np.array([r["muN"] for r in rows])
    muS_arr = np.array([r["muS"] for r in rows])

    deN = np.gradient(eN_arr, n_arr, edge_order=2)
    deS = np.gradient(eS_arr, n_arr, edge_order=2)
    d2eN = np.gradient(muN_arr, n_arr, edge_order=2)   # dmu_N/dn = d^2 e_N / dn^2
    d2eS = np.gradient(muS_arr, n_arr, edge_order=2)   # dmu_SC/dn = d^2 e_GS / dn^2
    d3eN = np.gradient(d2eN, n_arr, edge_order=2)
    d3eS = np.gradient(d2eS, n_arr, edge_order=2)

    # Interior Eq. (28) thermodynamic consistency |de/dn - mu|
    interior = slice(2, -2) if len(n_arr) > 6 else slice(None)
    eq28_id_max_S = float(np.max(np.abs(deS[interior] - muS_arr[interior])))
    eq28_id_max_N = float(np.max(np.abs(deN[interior] - muN_arr[interior])))

    derivs = {
        "deN_dn": deN.tolist(),
        "deS_dn": deS.tolist(),
        "d2eN_dn2": d2eN.tolist(),
        "d2eS_dn2": d2eS.tolist(),
        "d3eN_dn3": d3eN.tolist(),
        "d3eS_dn3": d3eS.tolist(),
        "eq28_id_max_err_S": eq28_id_max_S,
        "eq28_id_max_err_N": eq28_id_max_N,
        "concurrence": {},
    }

    for basis in bases:
        derivs["concurrence"][basis] = {}
        for rk in r_keys:
            C_S = np.array([r["sc_metrics"][basis][rk]["C"] for r in rows])
            C_N = np.array([r["norm_metrics"][basis][rk]["C"] for r in rows])
            C_0 = np.array([r["sc_metrics"][basis][rk]["C_normonly"] for r in rows])
            C_raw = np.array([r["sc_metrics"][basis][rk]["C_raw"] for r in rows])
            dCS = np.gradient(C_S, n_arr, edge_order=2)
            d2CS = np.gradient(dCS, n_arr, edge_order=2)
            dCN = np.gradient(C_N, n_arr, edge_order=2)
            d2CN = np.gradient(dCN, n_arr, edge_order=2)
            derivs["concurrence"][basis][rk] = {
                "dCS": dCS.tolist(),
                "d2CS": d2CS.tolist(),
                "dCN": dCN.tolist(),
                "d2CN": d2CN.tolist(),
                "dC_diff": (C_S - C_N).tolist(),
                "max_abs_dC_sc_vs_norm": float(np.max(np.abs(C_S - C_N))),
                "max_abs_anom_only": float(np.max(np.abs(C_S - C_0))),
                "max_abs_d2CS_interior": float(np.max(np.abs(d2CS[interior]))),
                "max_abs_d2CN_interior": float(np.max(np.abs(d2CN[interior]))),
                "C_raw_min": float(np.min(C_raw)),
                "C_raw_max": float(np.max(C_raw)),
            }

    elapsed = time.time() - t0
    print(f"[{name}] Nk={Nk}, U={U}, V_SO={V_SO}, T={T_norm}, pts={len(ns)}, "
          f"eq28_err={eq28_id_max_S:.2e}, elapsed={elapsed:.1f}s", flush=True)
    return {
        "name": name,
        "Nk": int(Nk),
        "U": float(U),
        "Dt": float(DT),
        "V_SO": float(V_SO),
        "T_norm": float(T_norm),
        "r_keys": list(r_keys),
        "bases": list(bases),
        "rows": rows,
        "derivs": derivs,
    }


def compute_exact_boundaries() -> dict:
    """High-precision Nk=801 critical densities (n_{c,1}, n_{c,2}, van Hove, Dirac)."""
    out = {}
    for V in (0.0, 0.4, 0.5, 0.7):
        sk, ssin, eps, KX, KY, _ = fs.make_grid(801, t=1.0, Vso=V, periodic=True)
        flat = np.sort(eps.ravel())
        N2 = float(801 * 801)
        E_sad = -2.0 * (1.0 + np.sqrt(1.0 + V * V))
        E_vh = 2.0 * (np.sqrt(1.0 + V * V) - 1.0)
        E_min = float(flat[0])
        Uc_zero = (2.0 * DT - 4.0) * DT * abs(E_min)
        v_rec = {
            "E_min": E_min,
            "E_sad": float(E_sad),
            "E_vh": float(E_vh),
            "Uc_n0_plus": float(Uc_zero),
            "n_sad_minus": float(np.sum(flat < E_sad) / N2),
            "n_vh_minus": float(np.sum(flat < -E_vh) / N2),
            "n_dirac0": float(np.sum(flat < 0.0) / N2),
            "n_vh_plus": float(np.sum(flat < E_vh) / N2),
            "n_dirac4": float(np.sum(flat < 4.0) / N2),
            "n_sad_plus": float(np.sum(flat < -E_sad) / N2),
            "n_c_by_U": {},
        }
        for U in (75.0, 90.0, 115.0):
            def lam_minus_1(n_val: float) -> float:
                nrm = fs.normal_state(eps, float(n_val))
                mu = nrm["mu"]
                denom = U - 4.0 * DT * mu
                if denom <= 0.0:
                    return 1e6
                J2 = float(np.sum(np.abs(eps - mu)) / N2)
                return DT * DT * J2 / denom - 1.0

            grid = np.linspace(0.002, 1.98, 250)
            vals = [lam_minus_1(float(g)) for g in grid]
            roots = []
            for i in range(len(grid) - 1):
                if vals[i] * vals[i + 1] < 0.0:
                    r = brentq(lam_minus_1, float(grid[i]), float(grid[i + 1]), xtol=1e-6)
                    roots.append(float(r))
            v_rec["n_c_by_U"][f"U{int(U)}"] = roots
        out[f"V{V:.1f}"] = v_rec
    return out


def main() -> int:
    t_all = time.time()
    out_dir = os.path.join(HERE, "data")
    os.makedirs(out_dir, exist_ok=True)

    print("=== Computing exact Nk=801 critical densities & van Hove markers ===", flush=True)
    boundaries = compute_exact_boundaries()
    for vk, rec in boundaries.items():
        print(f"  {vk}: Uc(0+)={rec['Uc_n0_plus']:.3f}, n_vh-={rec['n_vh_minus']:.5f}, "
              f"n_dirac4={rec['n_dirac4']:.5f}, n_c={rec['n_c_by_U']}", flush=True)

    # 1. Full-band Nk=401 periodic density sweeps across n in [0.005, 1.985]
    ns_full = np.unique(np.round(np.concatenate([
        np.array([0.005, 0.01, 0.015]),
        np.arange(0.02, 1.981, 0.01),
    ]), 4))

    full_sweeps = {
        "full_U75_V0.5_Nk401": run_density_sweep(
            "full_U75_V0.5_Nk401", ns_full, U=75.0, V_SO=0.5, Nk=401),
        "full_U115_V0.5_Nk401": run_density_sweep(
            "full_U115_V0.5_Nk401", ns_full, U=115.0, V_SO=0.5, Nk=401),
        "full_U75_V0.0_Nk401": run_density_sweep(
            "full_U75_V0.0_Nk401", ns_full, U=75.0, V_SO=0.0, Nk=401),
    }

    # 2. High-resolution Nk=801 periodic zoom sweeps
    zooms = {
        # Z1: Re-entrant low-density nodal SC pocket at U=75, V=0.5
        "zoom_low_U75_V0.5_Nk801": run_density_sweep(
            "zoom_low_U75_V0.5_Nk801",
            np.round(np.arange(0.004, 0.281, 0.006), 4),
            U=75.0, V_SO=0.5, Nk=801,
            r_keys=("0.25_0", "0.5_0", "1_0"), bases=("spin", "band"),
        ),
        # Z2a: Mid-band U=75, V=0.5 (n_vH-=0.9065 -> Dirac0=1.0 -> n_c2=1.0194 -> smeared n_vH+=1.0935)
        "zoom_mid_U75_V0.5_Nk801": run_density_sweep(
            "zoom_mid_U75_V0.5_Nk801",
            np.round(np.arange(0.84, 1.161, 0.005), 4),
            U=75.0, V_SO=0.5, Nk=801,
            r_keys=("0.25_0", "0.5_0", "0.5_0.5"), bases=("spin", "band"),
        ),
        # Z2b: Mid-band U=115, V=0.5 (pure Normal across BOTH n_vH-=0.9065 & n_vH+=1.0935)
        "zoom_mid_U115_V0.5_Nk801": run_density_sweep(
            "zoom_mid_U115_V0.5_Nk801",
            np.round(np.arange(0.84, 1.161, 0.005), 4),
            U=115.0, V_SO=0.5, Nk=801,
            r_keys=("0.25_0", "0.5_0", "0.5_0.5"), bases=("spin", "band"),
        ),
        # Z2c: Mid-band Normal T=1e-3t comparison at Nk=801
        "zoom_mid_Normal_T1e3_V0.5_Nk801": run_density_sweep(
            "zoom_mid_Normal_T1e3_V0.5_Nk801",
            np.round(np.arange(0.84, 1.161, 0.005), 4),
            U=115.0, V_SO=0.5, Nk=801,
            r_keys=("0.25_0", "0.5_0"), bases=("spin", "band"), T_norm=1e-3,
        ),
        # Z3: Isolated high-density SC transition n_c=1.4596 at U=115, V=0.5
        "zoom_high_U115_V0.5_Nk801": run_density_sweep(
            "zoom_high_U115_V0.5_Nk801",
            np.round(np.arange(1.38, 1.701, 0.005), 4),
            U=115.0, V_SO=0.5, Nk=801,
            r_keys=("0.25_0", "0.5_0"), bases=("spin", "band"),
        ),
        # Z4: V_SO=0.0 separated van Hove (n=1.000) & SC onset (n=1.1963)
        "zoom_mid_U75_V0.0_Nk801": run_density_sweep(
            "zoom_mid_U75_V0.0_Nk801",
            np.round(np.arange(0.92, 1.321, 0.005), 4),
            U=75.0, V_SO=0.0, Nk=801,
            r_keys=("0.25_0", "0.5_0"), bases=("spin", "band"),
        ),
        # Z5: (pi,pi) Dirac cone at n=1.9290: Normal (T=1e-3, U=300) vs SC (U=75)
        "zoom_dirac_Normal_V0.5_Nk801": run_density_sweep(
            "zoom_dirac_Normal_V0.5_Nk801",
            np.round(np.arange(1.88, 1.966, 0.003), 4),
            U=300.0, V_SO=0.5, Nk=801,
            r_keys=("0.25_0",), bases=("spin", "band"), T_norm=1e-3,
        ),
        "zoom_dirac_SC_U75_V0.5_Nk801": run_density_sweep(
            "zoom_dirac_SC_U75_V0.5_Nk801",
            np.round(np.arange(1.88, 1.966, 0.003), 4),
            U=75.0, V_SO=0.5, Nk=801,
            r_keys=("0.25_0",), bases=("spin", "band"),
        ),
    }

    # 3. Grid-size noise audit on d^2 C_N / dn^2 and d^2 e_N / dn^2 across Nk = 101, 201, 401, 801
    nk_grid_audit = {}
    ns_noise = np.round(np.arange(0.85, 1.061, 0.003), 4)
    for nk in (101, 201, 401, 801):
        nk_grid_audit[f"Nk{nk}"] = run_density_sweep(
            f"noise_U75_V0.5_Nk{nk}", ns_noise, U=75.0, V_SO=0.5, Nk=nk,
            r_keys=("0.25_0",), bases=("spin",),
        )

    payload = {
        "meta": {
            "Dt": DT,
            "grid": "periodic (endpoint=False)",
            "elapsed_sec": round(time.time() - t_all, 2),
        },
        "boundaries": boundaries,
        "full_sweeps": full_sweeps,
        "zooms": zooms,
        "nk_grid_audit": nk_grid_audit,
    }

    out_path = os.path.join(out_dir, "task8_density_qpt_eq28.json")
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"Wrote {out_path} ({os.path.getsize(out_path)/1e6:.2f} MB) in {time.time()-t_all:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
