"""Shared machinery: continuation with seed families, branch classification,
free-energy comparison at fixed density, and linearized-gap eigenvalue."""
from __future__ import annotations
import numpy as np
from solver import (make_grid, kernel_array, normal_state, sc_energy,
                    solve_all, lambda_max, residual_only)

U_T = 75.0
DT = 4.5

def setup(n, V, Nk=201, U=U_T, Dt=DT, t=1.0):
    sk, ssin, eps, KX, KY, kx = make_grid(Nk, t, V)
    kern = kernel_array(sk, ssin, U, Dt, V, t)
    nrm = normal_state(eps, n)
    lam, M, eigs = lambda_max(eps, kern, Nk, Dt, nrm["mu"])
    return dict(sk=sk, ssin=ssin, eps=eps, KX=KX, KY=KY, kx=kx,
                kern=kern, nrm=nrm, lam=lam, n=n, V=V, Nk=Nk, U=U, Dt=Dt)

def seeds_for(g, prev=None, strong=((0.30, 1.90), (0.27, 2.40), (0.30, -1.90), (0.27, -2.40))):
    mu = g["nrm"]["mu"]
    S = []
    if prev is not None:
        S.append(np.asarray(prev, float))
        S.append(np.asarray(prev, float) * 1.15)
    S += [np.array([1e-6, 1e-6, mu]), np.array([0.001, 0.003, mu]),
          np.array([0.01, 0.03, mu]), np.array([0.05, 0.10, mu]),
          np.array([0.15, 0.45, mu]), np.array([0.5, 1.0, mu])]
    S += [np.array([a, b, mu]) for a, b in strong]
    return S

def classify(g, nontriv, tol_energy=1e-9):
    """Return dict of branches with e_F, order params, stability info."""
    out = []
    for s in nontriv:
        e, eg, dphi = sc_energy(s, g["eps"], g["Nk"], g["n"])
        out.append(dict(x=s, e=e, eg=eg, dphi=dphi, D0=s[0], DS=s[1], mu=s[2]))
    out.sort(key=lambda d: d["e"])
    return out

def dF(g, branches):
    """F_SC - F_N per site (t=1)."""
    for b in branches:
        b["dF"] = b["e"] - g["nrm"]["e"]
    return branches

def run_sweep(n, Vs, Nk=201, U=U_T, Dt=DT, ascending=True, track=True, verbose=False):
    order = range(len(Vs)) if ascending else range(len(Vs) - 1, -1, -1)
    prev = None
    rows = []
    for i in order:
        V = float(Vs[i])
        g = setup(n, V, Nk, U, Dt)
        sols, ntr = solve_all(g["eps"], g["kern"], Nk, Dt, n, seeds_for(g, prev))
        br = dF(g, classify(g, ntr))
        if track and br:
            pick = min(br, key=lambda b: np.max(np.abs(b["x"] - prev))) if prev is not None else br[0]
            prev = pick["x"]
        rows.append(dict(V=V, i=i, n=n, lam=g["lam"], muN=g["nrm"]["mu"],
                         eN=g["nrm"]["e"], branches=br))
        if verbose:
            print(f"  V={V:.4f} lam={g['lam']:.4f} nb={len(br)} "
                  + " | ".join(f"D0={b['D0']:.4f},DS={b['DS']:.4f},dF={b['dF']:+.3e}" for b in br),
                  flush=True)
    rows.sort(key=lambda r: r["i"])
    return rows
