"""Sharp-QPT probes: pairing onset in Delta_t, and the Lifshitz transition
in a *normal* ground state (Delta_t = 0 -> no pairing -> e_GS = e_N).

Three scans (all at the corrected convention):

D. e_GS(Delta_t), V=0, U=75, n=1.875, fine grid: locate Delta_t_c where the
   gap opens from zero; a mean-field continuous (second-order) QPT shows
   Delta ~ sqrt(Delta_t - Delta_t_c) and a kink in d2e/dDelta_t^2.

L. e_GS(V_SO) at Delta_t=0, U=75, n=1.875, fine grid near V_L: the ground
   state is exactly normal (no pairing channel), so the Lifshitz
   nonanalyticity of e_N survives in e_GS -- a genuine ground-state QPT
   (step in d2e/dV^2 at mu_N = 4t).

M. e_GS(n) at Delta_t=0, U=75, V=0.7, fine grid: filling-driven Lifshitz,
   same logic (step in d2e/dn^2 at mu_N = 4t).

Outputs: reports/qpt_onset_lifshitz.json, figs/qpt_onset_lifshitz.png
Run:     .venv/bin/python analysis/qpt_onset_lifshitz.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ground_state_energy_qpt import (  # noqa: E402
    make_grid, normal_state, sc_energy, solve_sc)

T_SEEDS = [
    [0.30, 1.9], [0.30, -1.9], [0.27, 2.4], [0.27, -2.4],
    [0.7, 0.48], [0.5, 0.4], [1.0, 0.8], [0.1, 0.1], [0.02, 0.01],
    [-0.5, -1.0], [0.5, -1.0], [0.05, -0.2],
]


def seeds_for(mu_n, prev=None):
    out = []
    if prev is not None:
        out += [list(prev), [prev[0] * 1.2, prev[1] * 1.2, prev[2]],
                [prev[0], -prev[1], prev[2]]]
    out += [[s[0], s[1], mu_n] for s in T_SEEDS]
    return out


def solve_point(U, Dt, V, n, Nk, prev=None):
    sk, ssin, eps = make_grid(Nk, 1.0, V)
    norm = normal_state(eps, n)
    args = (eps, sk, ssin, Nk, 1.0, U, Dt, V, n, np.inf)
    sols = solve_sc(args, seeds_for(norm["mu"], prev))
    best, be = None, None
    for x in sols:
        e, _, _ = sc_energy(x, eps, sk, ssin, Nk, 1.0, U, Dt, V, n, np.inf)
        if be is None or e < be:
            best, be = list(map(float, x)), float(e)
    return norm, best, be


def scan(kind):
    rec = {"kind": kind, "pts": []}
    if kind == "D":           # pairing onset in Delta_t (both directions)
        grid = np.round(np.arange(0.5, 4.01, 0.05), 6)
        U, V, n, Nk = 75.0, 0.0, 1.875, 201
        prev = None
        for Dt in grid:       # up: from unpaired side
            norm, x, e = solve_point(U, Dt, V, n, Nk, prev)
            rec["pts"].append({"Dt": float(Dt), "mu_N": norm["mu"],
                               "e_N": norm["e"],
                               "e": e if e is not None else norm["e"],
                               "binding": (e - norm["e"]) if e else 0.0,
                               "x": x, "dir": "up"})
            prev = x
        prev = None
        for Dt in grid[::-1]:  # down: from paired side
            norm, x, e = solve_point(U, Dt, V, n, Nk, prev)
            rec.setdefault("down", []).append(
                {"Dt": float(Dt),
                 "e": e if e is not None else norm["e"],
                 "binding": (e - norm["e"]) if e else 0.0,
                 "x": x})
            prev = x
    elif kind == "L":         # Lifshitz in e_GS(V) with Delta_t = 0
        grid = np.round(np.arange(0.30, 1.101, 0.01), 6)
        U, Dt, n, Nk = 75.0, 0.0, 1.875, 401
        prev = None
        for V in grid:
            norm, x, e = solve_point(U, Dt, V, n, Nk, prev)
            rec["pts"].append({"V": float(V), "mu_N": norm["mu"],
                               "e_N": norm["e"],
                               "e": e if e is not None else norm["e"],
                               "binding": (e - norm["e"]) if e else 0.0,
                               "x": x})
            prev = x
    elif kind == "M":         # Lifshitz in e_GS(n) with Delta_t = 0
        grid = np.round(np.arange(1.70, 1.981, 0.01), 6)
        U, Dt, V, Nk = 75.0, 0.0, 0.7, 401
        prev = None
        for nn in grid:
            sk, ssin, eps = make_grid(Nk, 1.0, V)
            norm = normal_state(eps, nn)
            args = (eps, sk, ssin, Nk, 1.0, U, Dt, V, nn, np.inf)
            sols = solve_sc(args, seeds_for(norm["mu"], prev))
            e = norm["e"]
            prev = None
            if sols:
                best, be = None, None
                for x in sols:
                    ev, _, _ = sc_energy(x, eps, sk, ssin, Nk, 1.0, U, Dt,
                                         V, nn, np.inf)
                    if be is None or ev < be:
                        best, be = list(map(float, x)), float(ev)
                e, prev = be, best
            rec["pts"].append({"n": float(nn), "mu_N": norm["mu"],
                               "e_N": norm["e"], "e": e,
                               "binding": e - norm["e"], "x": prev})
    return rec


def main():
    os.makedirs("reports", exist_ok=True)
    os.makedirs("figs", exist_ok=True)
    out = {}
    for kind in ("D", "L", "M"):
        print(f"[scan] {kind} ...", flush=True)
        out[kind] = scan(kind)
        pts = out[kind]["pts"]
        par = "Dt" if kind == "D" else ("V" if kind == "L" else "n")
        gaps = [abs(p["x"][1]) if p["x"] else 0.0 for p in pts]
        print(f"  {par}: n_pts={len(pts)} gap range "
              f"[{min(gaps):.2e},{max(gaps):.2e}]", flush=True)
    with open("reports/qpt_onset_lifshitz.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote reports/qpt_onset_lifshitz.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
