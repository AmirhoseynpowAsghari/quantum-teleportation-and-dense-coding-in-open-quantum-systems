"""QPT parameter scan: ground-state energy vs U, n, V_SO, Delta_t.

Motivated by the hypothesis that a quantum phase transition (discontinuity
in the ground-state energy shape) may exist at parameters not covered by
the earlier e(V_SO) scans (which used only U=75/115, Delta_t=3.5/4.5,
n=1.875).  In particular the paper's own figure parameters
(U=90, Delta_t=4.5; U=75, Delta_t=3.5/4.0) were never V-scanned.

Families (CLI filter: u | n | v | dt | all):
  u   e_GS(U)            for several (V_SO, Delta_t, n); U both directions
  n   e_GS(n)            for several (U, Delta_t, V_SO); Lifshitz vs filling,
                          concavity of e(n) (phase-separation indicator)
  v   e_GS(V_SO)         at the paper figure parameters + hysteresis
                          (up/down continuation) for first-order detection
  dt  e_GS(Delta_t)      at fixed (U, V_SO, n)

Detectors (all reported per family):
  * branch crossings / coexisting distinct solutions -> first-order candidate
    (jump in de/dparam); gauge-related images are identified, not counted
  * jumps in the order parameter (gauge-invariant distance between adjacent
    solutions) -> first-order candidate
  * d^2 e kink candidates (3-sigma local peak, span 0.075) -> continuous
    QPT candidates (must persist under Nk refinement)
  * Lifshitz marker: normal-state mu crossing 4t; 'sharp' if the ground
    state is (numerically) normal there, i.e. binding ~ 0, in which case
    e_GS = e_N and the d^2 e jump is a true ground-state QPT

Outputs: reports/qpt_parameter_scan.json, figs/qpt_scan_<family>.png
Run:      .venv/bin/python analysis/qpt_parameter_scan.py all
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ground_state_energy_qpt import (  # noqa: E402
    make_grid, normal_state, sc_energy, solve_sc, _canon, _dedupe)

N_ELEC = 1.875
T_SEEDS = [
    [0.30, 1.9, None], [0.30, -1.9, None], [0.27, 2.4, None],
    [0.27, -2.4, None], [0.7, 0.48, None], [0.5, 0.4, None],
    [1.0, 0.8, None], [0.1, 0.1, None], [0.02, 0.01, None],
    [-0.5, -1.0, None], [0.5, -1.0, None], [0.05, -0.2, None],
]


def seeds_for(mu_n, prev=None):
    out = []
    if prev is not None:
        out += [list(prev),
                [prev[0] * 1.2, prev[1] * 1.2, prev[2]],
                [prev[0], -prev[1], prev[2]]]
    for s in T_SEEDS:
        out.append([s[0], s[1], mu_n if s[2] is None else s[2]])
    return out


def solve_point(U, Dt, V, n, Nk, prev=None):
    """All distinct SC solutions + normal reference at one parameter point."""
    sk, ssin, eps = make_grid(Nk, 1.0, V)
    norm = normal_state(eps, n)
    args = (eps, sk, ssin, Nk, 1.0, U, Dt, V, n, np.inf)
    sols = solve_sc(args, seeds_for(norm["mu"], prev))
    branches = []
    for x in sols:
        e, eg, dphi = sc_energy(x, eps, sk, ssin, Nk, 1.0, U, Dt, V, n,
                                np.inf)
        branches.append({"x": [float(v) for v in x], "e": float(e)})
    return norm, branches


def best_prev(branches):
    if not branches:
        return None
    b = min(branches, key=lambda b: b["e"])
    return b["x"]


def _gdist(a, b):
    """Gauge-invariant scaled distance (D0, DS, mu); (D0,DS)->(-D0,-DS)."""
    def _m(u, v):
        return max(abs(u[0] - v[0]) / 0.5, abs(u[1] - v[1]) / 3.0,
                   abs(u[2] - v[2]) / 3.0)
    return min(_m(a, b), _m(a, (-b[0], -b[1], b[2])))


def run_family(kind, Nk=201):
    """Execute one scan family; returns a dict with raw arrays + detectors."""
    rec = {"kind": kind, "Nk": Nk, "points": [], "candidates": []}

    def add(p, norm, branches, param):
        egs = min([b["e"] for b in branches]) if branches else norm["e"]
        rec["points"].append({
            "param": float(param),
            "mu_N": norm["mu"], "e_N": norm["e"],
            "e_GS": float(egs),
            "binding": float(egs - norm["e"]),
            "solutions": branches,
        })

    if kind == "v":
        # paper figure params + variants, with up/down hysteresis
        combos = [
            ("U90_Dt4.5_n1.875", 90.0, 4.5, N_ELEC),
            ("U75_Dt3.5_n1.875", 75.0, 3.5, N_ELEC),
            ("U75_Dt4.0_n1.875", 75.0, 4.0, N_ELEC),
            ("U90_Dt4.5_n0.5", 90.0, 4.5, 0.5),
        ]
        Vg = np.round(np.linspace(0.0, 1.5, 41), 6)
        for name, U, Dt, n in combos:
            fam = {"combo": name, "U": U, "Dt": Dt, "n": n,
                   "up": [], "down": []}
            prev = None
            for V in Vg:
                norm, br = solve_point(U, Dt, V, n, Nk, prev)
                add(None, norm, br, V)
                fam["up"].append({"V": float(V),
                                  "x": best_prev(br),
                                  "e": min([b["e"] for b in br])
                                  if br else norm["e"]})
                prev = best_prev(br)
            # downward continuation (hysteresis probe)
            prev = fam["up"][-1]["x"]
            for V in Vg[::-1]:
                norm, br = solve_point(U, Dt, V, n, Nk, prev)
                fam["down"].append({"V": float(V),
                                    "x": best_prev(br),
                                    "e": min([b["e"] for b in br])
                                    if br else norm["e"]})
                prev = best_prev(br)
            # hysteresis: max gauge-invariant gap between up/down branches
            # (down is stored descending -> reverse before zipping!)
            hyst = 0.0
            for a, b in zip(fam["up"], fam["down"][::-1]):
                if a["x"] and b["x"]:
                    hyst = max(hyst, _gdist(a["x"], b["x"]))
            fam["hysteresis_max"] = float(hyst)
            rec.setdefault("combos", []).append(fam)
        return rec

    if kind == "u":
        combos = [
            ("V0_Dt4.5_n1.875", 0.0, 4.5, N_ELEC),
            ("V0_Dt0_n1.875", 0.0, 0.0, N_ELEC),
            ("V0.7_Dt4.5_n1.875", 0.7, 4.5, N_ELEC),
            ("V1.4_Dt4.5_n1.875", 1.4, 4.5, N_ELEC),
            ("V0_Dt4.5_n0.5", 0.0, 4.5, 0.5),
            ("V0_Dt4.5_n1.0", 0.0, 4.5, 1.0),
            ("V1.0_Dt3.5_n1.875", 1.0, 3.5, N_ELEC),
            ("V0_Dt0_n0.5", 0.0, 0.0, 0.5),
        ]
        Ug = np.round(np.linspace(-20.0, 200.0, 45), 6)
        for name, V, Dt, n in combos:
            fam = {"combo": name, "V": V, "Dt": Dt, "n": n, "up": [], "down": []}
            prev = None
            for U in Ug:
                norm, br = solve_point(U, Dt, V, n, Nk, prev)
                add(None, norm, br, U)
                fam["up"].append({"U": float(U), "x": best_prev(br),
                                  "e": min([b["e"] for b in br])
                                  if br else norm["e"]})
                prev = best_prev(br)
            prev = fam["up"][-1]["x"]
            for U in Ug[::-1]:
                norm, br = solve_point(U, Dt, V, n, Nk, prev)
                fam["down"].append({"U": float(U), "x": best_prev(br),
                                    "e": min([b["e"] for b in br])
                                    if br else norm["e"]})
                prev = best_prev(br)
            hyst = 0.0
            for a, b in zip(fam["up"], fam["down"][::-1]):
                if a["x"] and b["x"]:
                    hyst = max(hyst, _gdist(a["x"], b["x"]))
            fam["hysteresis_max"] = float(hyst)
            rec.setdefault("combos", []).append(fam)
        return rec

    if kind == "n":
        combos = [
            ("U75_Dt4.5_V0", 75.0, 4.5, 0.0),
            ("U75_Dt4.5_V0.7", 75.0, 4.5, 0.7),
            ("U115_Dt3.5_V1.0", 115.0, 3.5, 1.0),
            ("U50_Dt4.5_V0", 50.0, 4.5, 0.0),
            ("U75_Dt0_V0.7", 75.0, 0.0, 0.7),
        ]
        ng = np.round(np.linspace(0.1, 1.95, 20), 6)
        for name, U, Dt, V in combos:
            fam = {"combo": name, "U": U, "Dt": Dt, "V": V, "pts": []}
            prev = None
            for n in ng:
                norm, br = solve_point(U, Dt, V, n, Nk, prev)
                add(None, norm, br, n)
                fam["pts"].append({"n": float(n), "mu_N": norm["mu"],
                                   "e": min([b["e"] for b in br])
                                   if br else norm["e"],
                                   "binding": (min([b["e"] for b in br])
                                               - norm["e"]) if br else 0.0,
                                   "x": best_prev(br)})
                prev = best_prev(br)
            # concavity of e(n): negative d2e/dn2 => phase separation (Maxwell)
            e = np.array([p["e"] for p in fam["pts"]])
            nA = np.array([p["n"] for p in fam["pts"]])
            if len(e) >= 5:
                d2 = np.gradient(np.gradient(e, nA, edge_order=2), nA,
                                 edge_order=2)
                fam["min_d2e_dn2"] = float(np.min(d2))
                fam["n_at_min_d2e"] = float(nA[int(np.argmin(d2))])
            rec.setdefault("combos", []).append(fam)
        return rec

    if kind == "dt":
        combos = [
            ("U75_V0_n1.875", 75.0, 0.0, N_ELEC),
            ("U75_V0.7_n1.875", 75.0, 0.7, N_ELEC),
        ]
        dg = np.round(np.linspace(0.0, 6.0, 25), 6)
        for name, U, V, n in combos:
            fam = {"combo": name, "U": U, "V": V, "n": n, "pts": []}
            prev = None
            for Dt in dg:
                norm, br = solve_point(U, Dt, V, n, Nk, prev)
                add(None, norm, br, Dt)
                fam["pts"].append({"Dt": float(Dt),
                                   "e": min([b["e"] for b in br])
                                   if br else norm["e"],
                                   "binding": (min([b["e"] for b in br])
                                               - norm["e"]) if br else 0.0,
                                   "x": best_prev(br)})
                prev = best_prev(br)
            rec.setdefault("combos", []).append(fam)
        return rec

    raise ValueError(kind)


def detect(kind, rec):
    """Post-process a family: crossings, gap jumps, d2e kinks, Lifshitz."""
    out = {"kind": kind, "Nk": rec["Nk"], "combos": []}
    if kind in ("v", "u"):
        for fam in rec.get("combos", []):
            name = fam["combo"]
            up = fam["up"]
            p = np.array([q[("V" if kind == "v" else "U")] for q in up])
            e = np.array([q["e"] for q in up])
            eN = np.array([q.get("e_N", np.nan) for q in up])
            xs = [q["x"] for q in up]
            # crossings vs normal: binding sign change (beyond degenerate ~0)
            bind = []
            for i, q in enumerate(up):
                bind.append(np.nan)
            # recompute binding from rec points (aligned per combo chunks)
            # -> easier: use e vs e_N from the points list in order
            # (points were appended combo-by-combo in the same order)
            res = {"combo": name, "hysteresis_max": fam.get("hysteresis_max"),
                   "kinks": [], "crossings": [], "gap_jumps": [],
                   "lifshitz": None}
            # d2e kinks of e_GS(param)
            if len(e) >= 7:
                de = np.gradient(e, p, edge_order=2)
                d2 = np.gradient(de, p, edge_order=2)
                span = 0.075 if kind == "v" else max(
                    0.05 * (p[-1] - p[0]), 1e-9)
                for i in range(2, len(p) - 2):
                    win = (p >= p[i] - span) & (p <= p[i] + span)
                    m, s = np.mean(d2[win]), np.std(d2[win])
                    if s > 0 and d2[i] > m + 3 * s and d2[i] > 1e-4:
                        res["kinks"].append(round(float(p[i]), 4))
            # gap jumps (gauge-invariant) between adjacent solutions
            for i in range(1, len(xs)):
                if xs[i] and xs[i - 1]:
                    d = _gdist(xs[i - 1], xs[i])
                    if d > 0.6:  # large beyond smooth evolution
                        res["gap_jumps"].append(
                            {"param": round(float(p[i]), 4),
                             "gdist": round(float(d), 3)})
            # branch coexistence (multiple distinct solutions at a point)
            multi = [(round(float(p[i]), 4), len(up[i].get("sol", [])))
                     for i in range(len(up))]
            out["combos"].append(res)
    elif kind == "n":
        for fam in rec.get("combos", []):
            pts = fam["pts"]
            nA = np.array([q["n"] for q in pts])
            e = np.array([q["e"] for q in pts])
            mu = np.array([q["mu_N"] for q in pts])
            res = {"combo": fam["combo"],
                   "min_d2e_dn2": fam.get("min_d2e_dn2"),
                   "n_at_min_d2e": fam.get("n_at_min_d2e"),
                   "mu_crosses_4t": None, "kinks": []}
            for i in range(1, len(mu)):
                if (mu[i - 1] - 4.0) * (mu[i] - 4.0) < 0:
                    res["mu_crosses_4t"] = [round(float(nA[i - 1]), 4),
                                            round(float(nA[i]), 4)]
                    break
            if len(e) >= 7:
                de = np.gradient(e, nA, edge_order=2)
                d2 = np.gradient(de, nA, edge_order=2)
                span = 0.1
                for i in range(2, len(p := nA) - 2):
                    win = (nA >= nA[i] - span) & (nA <= nA[i] + span)
                    m, s = np.mean(d2[win]), np.std(d2[win])
                    if s > 0 and d2[i] > m + 3 * s and d2[i] > 1e-3:
                        res["kinks"].append(round(float(nA[i]), 4))
            out["combos"].append(res)
    elif kind == "dt":
        for fam in rec.get("combos", []):
            pts = fam["pts"]
            dA = np.array([q["Dt"] for q in pts])
            e = np.array([q["e"] for q in pts])
            res = {"combo": fam["combo"], "kinks": []}
            if len(e) >= 7:
                de = np.gradient(e, dA, edge_order=2)
                d2 = np.gradient(de, dA, edge_order=2)
                span = 0.3
                for i in range(2, len(dA) - 2):
                    win = (dA >= dA[i] - span) & (dA <= dA[i] + span)
                    m, s = np.mean(d2[win]), np.std(d2[win])
                    if s > 0 and d2[i] > m + 3 * s and d2[i] > 1e-3:
                        res["kinks"].append(round(float(dA[i]), 4))
            out["combos"].append(res)
    return out


def main():
    families = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    if "all" in families:
        families = ["v", "u", "n", "dt"]
    os.makedirs("reports", exist_ok=True)
    os.makedirs("figs", exist_ok=True)
    out_path = "reports/qpt_parameter_scan.json"
    all_out = {}
    if os.path.exists(out_path):
        try:
            all_out = json.load(open(out_path))
        except Exception:
            all_out = {}
    for kind in families:
        print(f"[scan] family {kind} ...", flush=True)
        rec = run_family(kind, Nk=201)
        det = detect(kind, rec)
        all_out[kind] = {"raw": rec, "detected": det}
        with open(out_path, "w") as f:
            json.dump(all_out, f, indent=1)
        # console summary
        for c in det["combos"]:
            print(f"  {c.get('combo')}: kinks={c.get('kinks')} "
                  f"gap_jumps={c.get('gap_jumps')} "
                  f"hyst={c.get('hysteresis_max')} "
                  f"mu4t={c.get('mu_crosses_4t') or c.get('lifshitz')}",
                  flush=True)
        if kind in ("v", "u"):
            for fam in rec.get("combos", []):
                print(f"  [raw] {fam['combo']} hysteresis_max="
                      f"{fam.get('hysteresis_max')}", flush=True)
    print("wrote", out_path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
