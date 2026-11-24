# =============================================================================
# analysis/n_density_scan.py
# =============================================================================
"""Density (n) scan: where does superconductivity vanish, and does the
two-spin entanglement survive it?

For each density n we solve the BdG self-consistency at the protocol point
(U/t=75, Delta_t/t=4.5, V_SO/t=0.5, T=0, Nk=100) with the current helicity
conditional-Wick pipeline, then evaluate the t=0 concurrence and the
singlet-referenced fidelity of the two-detection state at theta=0.

If the solver reports the trivial root (|Delta| ~ 0, no nontrivial solution
found after its internal random restarts), we confirm the exact normal-state
root by solving the number equation at Delta=0 and evaluate the same
observables on that normal state, plus a linearized-gap (stability)
indicator at Delta=0.

Run:  .venv/bin/python analysis/n_density_scan.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.optimize import fsolve as _fsolve

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
import kspace_setup
import self_consistency as sc
from self_consistency import residual, run_selfconsistency
from helicity_density_matrix import (
    SINGLET,
    density_at_separation,
    concurrence,
)

VSO = 0.5
U_VAL = 75.0
DT_VAL = 4.5
NK = 100
N_LIST = [0.30, 0.60, 0.90, 1.00, 1.05, 1.10, 1.15, 1.20, 1.40, 1.60,
          1.75, 1.875, 1.95]
R_LIST = [0.0, 0.25, 0.50, 1.00]          # theta = 0


def patch(vso: float, n: float, nk: int = NK) -> None:
    for mod in (config, kspace_setup, sc):
        if hasattr(mod, "V_SO"):
            mod.V_SO = float(vso)
        if hasattr(mod, "Nk"):
            mod.Nk = int(nk)
        if hasattr(mod, "n_elec"):
            mod.n_elec = float(n)


def normal_root(n: float, eps: np.ndarray, sk: np.ndarray,
                ssin: np.ndarray) -> float:
    """Chemical potential from the number equation at Delta = 0."""
    beta = np.inf

    def eq(mu: np.ndarray) -> np.ndarray:
        return residual([0.0, 0.0, float(mu[0])], eps, sk, ssin, NK,
                        config.t_hop, U_VAL, DT_VAL, VSO, n, beta)[2]

    mu0 = float(np.mean(eps))
    mu = _fsolve(eq, [mu0], xtol=1.49e-10, full_output=False)
    return float(mu[0])


def normal_bdgs(eps: np.ndarray, mu: float, KX: np.ndarray,
                KY: np.ndarray, ssin: np.ndarray) -> dict:
    Dks = np.zeros_like(eps)
    E = np.clip(np.abs(eps - mu), 1e-12, None)
    u = np.sqrt(np.clip(0.5 * (1.0 + (eps - mu) / E), 0.0, 1.0))
    v = np.sqrt(np.clip(0.5 * (1.0 - (eps - mu) / E), 0.0, 1.0))
    return {"Delta_ks": Dks, "E_ks": E, "u_ks": u, "v_ks": v,
            "KX": KX, "KY": KY, "epsilon_ks": eps, "sqrt_sin": ssin}


def linearized_gap_indicator(eps: np.ndarray, sk: np.ndarray,
                             ssin: np.ndarray, mu: float, n: float) -> float:
    """Largest eigenvalue of the T=0 linearized gap map at Delta=0.

    >1 means the normal state is linearly unstable to pairing (grid-cut
    BCS log), <=1 means the normal state is stable.
    """
    kernel = (U_VAL
              + 8.0 * DT_VAL * sk[..., None]
              + 4.0 * (DT_VAL / config.t_hop) * VSO
              * np.array([1.0, -1.0])[None, None, :] * ssin[..., None])
    invE = 1.0 / np.clip(np.abs(eps - mu), 1e-9, None)
    N2 = float(NK * NK)
    t = config.t_hop
    M00 = -np.sum(kernel * invE) / (4.0 * N2)
    M01 = np.sum(kernel * eps * invE) / (4.0 * N2 * 4.0 * t)
    M10 = -2.0 * DT_VAL * np.sum(invE) / N2
    M11 = 2.0 * DT_VAL * np.sum(eps * invE) / (N2 * 4.0 * t)
    eig = np.linalg.eigvals(np.array([[M00, M01], [M10, M11]]))
    return float(np.max(eig.real))


def observables(bdg: dict) -> dict:
    out = {}
    for r in R_LIST:
        st = density_at_separation(bdg, [r, 0.0], basis="spin")
        rho = st["rho"]
        out[f"C_r{r:g}"] = concurrence(rho)
        out[f"Fs_r{r:g}"] = float(np.real(SINGLET.conj() @ rho @ SINGLET))
    return out


def main() -> None:
    patch(VSO, N_LIST[0])
    rows = []
    print(f"n-scan: U={U_VAL} Dt={DT_VAL} V_SO={VSO} Nk={NK} T=0, "
          f"theta=0, r in {R_LIST}", flush=True)
    hdr = (f"{'n':>6} {'SC':>5} {'D0':>10} {'DS':>10} {'Dmax':>10} "
           f"{'mu':>9} {'eig':>7} {'C(.25)':>8} {'C(.5)':>8} {'C(1)':>8} "
           f"{'Fs(.25)':>9}")
    print(hdr, flush=True)

    for n in N_LIST:
        patch(VSO, n)
        bdg = run_selfconsistency(seed=42, U_override=U_VAL,
                                  kBT_override=0.0, verbose=False)
        res = float(np.max(np.abs(bdg["residuals"])))
        Dmax = float(np.max(np.abs(bdg["Delta_ks"])))
        sc_ok = bool(bdg["converged"]) and res < 1e-8 and Dmax > 1e-6

        eps = bdg["epsilon_ks"]
        sk = 0.5 * (np.cos(bdg["KX"]) + np.cos(bdg["KY"]))
        ssin = bdg["sqrt_sin"]

        if sc_ok:
            use_bdg = bdg
            mu = float(bdg["mu"])
            note = "SC"
        else:
            mu = normal_root(n, eps, sk, ssin)
            use_bdg = normal_bdgs(eps, mu, bdg["KX"], bdg["KY"], ssin)
            note = "normal"

        obs = observables(use_bdg)
        eig = linearized_gap_indicator(eps, sk, ssin, mu, n)
        row = {
            "n": n, "status": note,
            "converged_flag": bool(bdg["converged"]),
            "residual": res,
            "Delta0": float(bdg["Delta0"]), "DeltaS": float(bdg["DeltaS"]),
            "Delta_max": Dmax if sc_ok else 0.0,
            "mu_sc": float(bdg["mu"]),
            "mu_normal": mu,
            "linearized_gap_eig": eig,
            **obs,
        }
        rows.append(row)
        print(f"{n:6.3f} {note:>5} {row['Delta0']:10.6f} "
              f"{row['DeltaS']:10.6f} {row['Delta_max']:10.6f} "
              f"{mu:9.5f} {eig:7.3f} {obs['C_r0.25']:8.5f} "
              f"{obs['C_r0.5']:8.5f} {obs['C_r1']:8.5f} "
              f"{obs['Fs_r0.25']:9.5f}", flush=True)

    report = {
        "params": {"U": U_VAL, "Delta_t": DT_VAL, "V_SO": VSO, "Nk": NK,
                   "T": 0, "theta_deg": 0, "r_list": R_LIST},
        "rows": rows,
    }
    path = ROOT / "reports" / "n_density_scan.json"
    path.write_text(json.dumps(report, indent=1))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
