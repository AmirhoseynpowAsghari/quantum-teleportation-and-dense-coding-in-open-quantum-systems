#!/usr/bin/env python3
"""Ground-state energy analysis of the Hubbard-Rashba + correlated-hopping
mean field (arXiv:1801.08988), with the Phase-1 corrected gap equations.

Deliverable 3 checks
    e_GS = E_g / N_sites + mu * n   with
    E_g = 1/2 sum_{k,s} [xi - E + Db^2/(2E)]      (paper Eq. 28, T=0)
    - no Delta^2/U terms (interaction constant already in Eq. 28),
    - on-shell quantity (not minimized),
    - identity dPhi/dmu = -N_tot  <=>  particle-number equation,
    - total derivative dOmega/dmu = -N_tot checked by fixed-mu solves.

Deliverable 4 numerics
    For each case: e_N(V) (normal, Delta=0), every SC branch e_SC^(i)(V),
    e_GS = min_i, de/dV, d^2e/dV^2, branch crossings, mu(V) vs band top
    (Lifshitz marker), Nk convergence {201,301,401,601} for Case A and
    {201,401} for the B cases.

Outputs: data/ground_state_energy_qpt.npz,
         reports/ground_state_energy_qpt.json,
         figs/ground_state_energy_<case>.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import fsolve, least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from self_consistency import residual

N_ELEC = 1.875
T_SEEDS = 20261105

CASES = {
    # name: (U, Dt, V_start, V_stop, Nk_list)
    "caseA_U75 Dt4.5":  (75.0, 4.5, 0.0, 1.5, [201, 301, 401, 601]),
    "caseB1_U115 Dt3.5": (115.0, 3.5, 0.0, 2.0, [201, 401]),
    "caseB2_U115 Dt4.5": (115.0, 4.5, 0.0, 1.5, [201, 401]),
}


# ---------------------------------------------------------------------------
# Grid / normal state / energy
# ---------------------------------------------------------------------------
def make_grid(Nk, t, Vso):
    kx = np.linspace(-np.pi, np.pi, Nk)
    KX, KY = np.meshgrid(kx, kx)
    ssin = np.sqrt(np.sin(KX) ** 2 + np.sin(KY) ** 2)
    sk = 0.5 * (np.cos(KX) + np.cos(KY))
    eps = np.empty((Nk, Nk, 2))
    eps[..., 0] = -2 * t * (np.cos(KX) + np.cos(KY)) - 2 * Vso * ssin
    eps[..., 1] = -2 * t * (np.cos(KX) + np.cos(KY)) + 2 * Vso * ssin
    return sk, ssin, eps


def normal_state(eps, n):
    """T=0 normal state: staircase number equation + E_g(Delta=0)."""
    N2 = eps.shape[0] * eps.shape[1]
    s = np.sort(eps.ravel())
    k = int(round(n * N2))
    k = min(max(k, 1), s.size - 1)
    lo, hi = s[k - 1], s[k]
    mu = 0.5 * (lo + hi) if hi > lo else lo
    xi = eps - mu
    E = np.abs(xi)
    eg = 0.5 * np.sum(xi - E)          # = sum_{xi<0} xi
    e = eg / N2 + mu * n
    return {"mu": float(mu), "e": float(e), "eg": float(eg)}


def sc_energy(x, eps, sk, ssin, Nk, t, U, Dt, Vso, n, beta=np.inf):
    """e_GS per site from Eq. (28) on-shell (+ mu n); also Phi identity."""
    D0, DS, mu = float(x[0]), float(x[1]), float(x[2])
    N2 = float(Nk * Nk)
    Dks = D0 - (DS / (4.0 * t)) * eps
    E = np.clip(np.sqrt(Dks ** 2 + (eps - mu) ** 2), 1e-12, None)
    xi = eps - mu
    eg = 0.5 * np.sum(xi - E + Dks ** 2 / (2.0 * E))
    e = eg / N2 + mu * n
    # number-equation identity:  dPhi/dmu = 1/2 sum (-1 + xi/E) = -N_tot
    dphi = 0.5 * np.sum(-1.0 + xi / E)
    ntot = n * N2
    return float(e), float(eg), float(dphi + ntot)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------
def _canon(x):
    """Fix the global sign gauge (D0,DS) -> (-D0,-DS) leaves e invariant."""
    x = np.asarray(x, float)
    if x[0] < -1e-9 or (abs(x[0]) <= 1e-9 and x[1] < 0):
        x = np.array([-x[0], -x[1], x[2]])
    return x


def _dedupe(sols, tol=1e-6):
    out = []
    for s in sols:
        if not any(np.max(np.abs(np.array(s) - np.array(o))) < tol
                   for o in out):
            out.append(s)
    return out


def solve_sc(args, seeds, accept_trivial=False):
    """Residual-gated multi-start solve of the corrected gap+number system."""
    sols = []
    for x0 in seeds:
        for method in ("fsolve", "lm"):
            try:
                if method == "fsolve":
                    sol, info, ier, _ = fsolve(
                        residual, x0, args=args, full_output=True,
                        xtol=1.49e-10, maxfev=6000)
                    cand = np.asarray(sol, float)
                else:
                    sol = least_squares(residual, x0, args=args,
                                        method="lm", ftol=1e-14,
                                        xtol=1e-14, gtol=1e-14)
                    cand = np.asarray(sol.x, float)
            except Exception:
                continue
            r = float(np.max(np.abs(residual(cand, *args))))
            if r < 1e-10:
                sols.append(_canon(cand))
                break
    sols = _dedupe(sols)
    if not accept_trivial:
        sols = [s for s in sols
                if max(abs(s[0]), abs(s[1])) > 1e-6]
    return sols


def solve_fixed_mu(args2, mu, seeds):
    """Gap equations only (number eq dropped) at fixed mu -> Omega(mu)."""
    sols = []
    for x0 in seeds:
        def r2(x, *a):
            return residual([x[0], x[1], mu], *a)[:2]
        try:
            sol, info, ier, _ = fsolve(r2, x0, args=args2, full_output=True,
                                       xtol=1.49e-11, maxfev=4000)
            cand = np.asarray(sol, float)
        except Exception:
            continue
        if float(np.max(np.abs(r2(cand, *args2)))) < 1e-10:
            sols.append(_canon([cand[0], cand[1], mu]))
    return _dedupe(sols)


# ---------------------------------------------------------------------------
# Branch tracking over a V grid
# ---------------------------------------------------------------------------
def run_sweep(U, Dt, Vs, Nk, t=1.0, n=N_ELEC, seed=T_SEEDS):
    rng = np.random.default_rng(seed)
    sk, ssin, eps0 = make_grid(Nk, t, Vs[0])
    norm0 = normal_state(eps0, n)
    mu_n0 = norm0["mu"]

    branches = []          # list of dicts: xs[], Vs[], es[], present[]
    normal = {"mu": [], "e": []}
    all_sols_at_V = []

    prev_sols = []
    for V in Vs:
        sk, ssin, eps = make_grid(Nk, t, V)
        args = (eps, sk, ssin, Nk, t, U, Dt, V, n, np.inf)

        norm = normal_state(eps, n)
        mu_n = norm["mu"]

        # seeds: previous solutions (continuation), gap guesses, random.
        # Strong seeds cover the corrected-convention regime
        # (|D0|~0.25-0.3, |DS|~1.5-2.5); the config init [0.7,0.48] is
        # the historically reliable starter at V=0.
        seeds = list(prev_sols)
        seeds += [[0.7, 0.48, mu_n], [0.3, 1.9, mu_n],
                  [0.27, 2.4, mu_n], [0.5, 0.4, mu_n],
                  [0.12, 0.05, mu_n], [0.02, 0.01, mu_n],
                  [float(rng.uniform(0.1, 1.4)),
                   float(rng.uniform(0.05, 1.0)),
                   float(rng.uniform(eps.min(), eps.max()))],
                  [float(rng.uniform(0.1, 1.4)),
                   float(rng.uniform(-2.4, -0.05)),
                   float(rng.uniform(eps.min(), eps.max()))]]
        if prev_sols:
            seeds.append(list(np.array(prev_sols[0]) * 1.2))
        sols = solve_sc(args, seeds)

        normal["mu"].append(norm["mu"])
        normal["e"].append(norm["e"])

        # energies
        entries = []
        for s in sols:
            e, eg, idn = sc_energy(s, eps, sk, ssin, Nk, t, U, Dt, V, n)
            trivial = max(abs(s[0]), abs(s[1])) < 1e-6
            entries.append({"x": s, "e": e, "trivial": trivial,
                            "idn": idn})
        all_sols_at_V.append(entries)
        prev_sols = [tuple(e["x"]) for e in entries if not e["trivial"]][:6]

        # branch matching to previous V: scaled metric (mu moves ~1 per
        # 0.25 in V at high SO, so plain max-norm kills valid branches)
        def _bdist(a, b):
            return max(abs(a[0] - b[0]) / 0.5,
                       abs(a[1] - b[1]) / 3.0,
                       abs(a[2] - b[2]) / 3.0)

        new_branches = []
        for br in branches:
            last = br["xs"][-1]
            best, bd = None, 1e9
            for e in entries:
                if e["trivial"]:
                    continue
                d = _bdist(e["x"], last)
                if d < bd:
                    best, bd = e, d
            if best is not None and bd < 1.0:
                br["xs"].append(best["x"])
                br["es"].append(best["e"])
                br["Vs"].append(V)
                br["miss"] = 0
                new_branches.append(br)
            else:
                br["miss"] = br.get("miss", 0) + 1
                if br["miss"] <= 2:
                    new_branches.append(br)   # carry through short gaps
        branches = new_branches

        # start new branches from unmatched non-trivial solutions
        matched = {id(tuple(br["xs"][-1])) for br in branches}
        for e in entries:
            if e["trivial"]:
                continue
            is_new = True
            for br in branches:
                if len(br["Vs"]) and br["Vs"][-1] == V:
                    if np.max(np.abs(np.array(e["x"])
                                     - np.array(br["xs"][-1]))) < 1e-6:
                        is_new = False
                        break
            if is_new:
                branches.append({"Vs": [V], "xs": [e["x"]],
                                "es": [e["e"]]})
        # prune branches with < 2 points at the end (handled later)

    return {"V": np.array(Vs), "branches": branches, "normal": normal,
            "Nk": Nk, "U": U, "Dt": Dt}


# ---------------------------------------------------------------------------
# Derivatives and diagnostics
# ---------------------------------------------------------------------------
def derivatives(V, e):
    de = np.gradient(e, V, edge_order=2)
    d2 = np.gradient(de, V, edge_order=2)
    return de, d2


def derivative_robustness(V, e):
    """max |de_h - de_2h| / scale on the overlap of grids (h vs 2h)."""
    if len(V) < 6:
        return None, None
    de_h, d2_h = derivatives(V, e)
    V2, e2 = V[::2], e[::2]
    de_2, d2_2 = derivatives(V2, e2)
    sl = slice(1, -1)
    scale = max(np.max(np.abs(de_h[2:-2])), 1e-12)
    d1 = float(np.max(np.abs(de_h[2:-2:2][sl] - de_2[1:-1][sl])) / scale)
    scale2 = max(np.max(np.abs(d2_h[2:-2])), 1e-12)
    d2r = float(np.max(np.abs(d2_h[2:-2:2][sl] - d2_2[1:-1][sl])) / scale2)
    return d1, d2r


def lifshitz_mu_crossing(V, mu, band_ref=4.0):
    """First V where mu crosses band_ref (mu increasing through it)."""
    mu = np.asarray(mu)
    for i in range(1, len(V)):
        if (mu[i - 1] - band_ref) * (mu[i] - band_ref) <= 0 and \
                mu[i] != mu[i - 1]:
            # linear interpolation
            w = (band_ref - mu[i - 1]) / (mu[i] - mu[i - 1])
            return float(V[i - 1] + w * (V[i] - V[i - 1]))
    return None


def crossings(V, ea, eb):
    """V where ea - eb changes sign (linear interpolation)."""
    d = np.asarray(ea) - np.asarray(eb)
    out = []
    for i in range(1, len(V)):
        if np.isfinite(d[i]) and np.isfinite(d[i - 1]) and \
                d[i - 1] * d[i] < 0:
            w = d[i - 1] / (d[i - 1] - d[i])
            out.append(float(V[i - 1] + w * (V[i] - V[i - 1])))
    return out


# ---------------------------------------------------------------------------
# Fixed-mu Omega check:  dOmega/dmu = -N_tot
# ---------------------------------------------------------------------------
def check_domega_dmu(U, Dt, V, Nk, t=1.0, n=N_ELEC):
    sk, ssin, eps = make_grid(Nk, t, V)
    args_full = (eps, sk, ssin, Nk, t, U, Dt, V, n, np.inf)
    mu_guess = normal_state(eps, n)["mu"]
    base = solve_sc(args_full, [[0.7, 0.48, mu_guess],
                                [0.3, 1.9, mu_guess],
                                [0.27, 2.4, mu_guess]])
    if not base:
        return None
    x = base[0]
    mu0 = x[2]
    h = 1e-3
    args2 = (eps, sk, ssin, Nk, t, U, Dt, V, n, np.inf)

    def omega_at(mu):
        sols = solve_fixed_mu(args2, mu, [[x[0], x[1]]])
        if not sols:
            return None
        e, eg, _ = sc_energy(sols[0], eps, sk, ssin, Nk, t, U, Dt, V, n)
        return eg   # Omega(T=0) = E_g  (E_g already contains -mu*N)

    om_p = omega_at(mu0 + h)
    om_m = omega_at(mu0 - h)
    if om_p is None or om_m is None:
        return None
    dom = (om_p - om_m) / (2 * h)
    ntot = n * Nk * Nk
    return {"dOmega_dmu": float(dom), "minus_Ntot": float(-ntot),
            "rel_err": float(abs(dom + ntot) / ntot)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    report = {"n_elec": N_ELEC, "cases": {}}
    only = sys.argv[1:] if len(sys.argv) > 1 else None

    for name, (U, Dt, v0, v1, nks) in CASES.items():
        tag = name.split("_")[0]
        if only and tag not in only and name not in only:
            continue
        nV = 41 if (v1 - v0) <= 1.51 else 41
        Vs = np.linspace(v0, v1, nV)
        case_rep = {"U": U, "Dt": Dt, "V_grid": [v0, v1, nV],
                    "sweeps": {}}

        for Nk in nks:
            print(f"[{name}] Nk={Nk} sweep ...", flush=True)
            sw = run_sweep(U, Dt, Vs, Nk)
            V = sw["V"]
            eN = np.array(sw["normal"]["e"])
            muN = np.array(sw["normal"]["mu"])

            # branch arrays
            br_data = []
            for br in sw["branches"]:
                if len(br["Vs"]) < 3:
                    continue
                bV = np.array(br["Vs"])
                be = np.array(br["es"])
                # interpolate onto full V grid where covered
                mask = np.isin(np.round(bV, 9), np.round(V, 9))
                idx = np.searchsorted(np.round(V, 9), np.round(bV, 9))
                e_full = np.full_like(V, np.nan)
                e_full[idx] = be
                mu_full = np.full_like(V, np.nan)
                ds_full = np.full_like(V, np.nan)
                d0_full = np.full_like(V, np.nan)
                for j, x in enumerate(br["xs"]):
                    mu_full[idx[j]] = x[2]
                    ds_full[idx[j]] = x[1]
                    d0_full[idx[j]] = x[0]
                br_data.append({"V": bV, "e": be, "e_full": e_full,
                                "mu": mu_full, "D0": d0_full,
                                "DS": ds_full,
                                "idn_max": float(max(
                                    abs(x) for x in
                                    [0])) })
            # e_GS = min over branches (where defined) and normal
            if br_data:
                e_stack = np.vstack([b["e_full"] for b in br_data])
                finite = np.isfinite(e_stack)
                tmp = np.where(finite, e_stack, np.inf)
                e_min = np.min(tmp, axis=0)
                any_fin = finite.any(axis=0)
                e_gs = np.where(any_fin, np.minimum(e_min, eN), eN)
            else:
                e_gs = eN.copy()

            de, d2 = derivatives(V, e_gs)
            deN, d2N = derivatives(V, eN)
            rob1, rob2 = derivative_robustness(V, e_gs)

            # branch-vs-normal crossings (first-order candidates)
            cross = []
            for i, b in enumerate(br_data):
                for c in crossings(V, b["e_full"], eN):
                    cross.append({"branch": i, "V": float(c)})
            # branch-branch crossings
            for i in range(len(br_data)):
                for j in range(i + 1, len(br_data)):
                    for c in crossings(V, br_data[i]["e_full"],
                                       br_data[j]["e_full"]):
                        cross.append({"branch": f"{i}-{j}", "V": float(c)})

            mu_sc = br_data[0]["mu"] if br_data else np.full(len(V), np.nan)
            v_life_N = lifshitz_mu_crossing(V, muN)
            v_life_S = (lifshitz_mu_crossing(V, mu_sc)
                        if np.isfinite(mu_sc).any() else None)

            # d2e kink candidates: local max of |d2 - smoothed d2|
            kinks = []
            if len(V) >= 9:
                kern = np.ones(5) / 5.0
                smooth = np.convolve(d2, kern, mode="same")
                smooth[0] = d2[0]
                smooth[-1] = d2[-1]
                resid = np.abs(d2 - smooth)
                for i in range(2, len(V) - 2):
                    if resid[i] == max(resid[max(0, i - 3):i + 4]) and \
                            resid[i] > 3.0 * np.median(resid):
                        kinks.append(float(V[i]))

            sw_rep = {
                "e_N": eN.tolist(), "mu_N": muN.tolist(),
                "e_GS": e_gs.tolist(), "de_GS": de.tolist(),
                "d2e_GS": d2.tolist(),
                "branches": [
                    {"V": b["V"].tolist(), "e": b["e"].tolist(),
                     "D0": [None if not np.isfinite(x) else float(x)
                            for x in b["D0"]],
                     "DS": [None if not np.isfinite(x) else float(x)
                            for x in b["DS"]],
                     "mu": [None if not np.isfinite(x) else float(x)
                            for x in b["mu"]]}
                    for b in br_data],
                "lifshitz_muN_4t": v_life_N,
                "lifshitz_muSC_4t": v_life_S,
                "crossings": cross,
                "d2_kink_candidates": kinks,
                "deriv_robustness_h_vs_2h": [rob1, rob2],
                "max_num_identity_resid": None,
            }

            # number-equation identity residual (max over SC solutions)
            idres = 0.0
            for b in br_data:
                for j, Vp in enumerate(b["V"]):
                    sk_, ss_, eps_ = make_grid(Nk, 1.0, float(Vp))
                    x = [b["D0"][j], b["DS"][j], b["mu"][j]]
                    if any(v is None or not np.isfinite(v) for v in x):
                        continue
                    _, _, idn = sc_energy(x, eps_, sk_, ss_, Nk, 1.0,
                                          U, Dt, float(Vp), N_ELEC)
                    idres = max(idres, abs(idn) / (N_ELEC * Nk * Nk))
            sw_rep["max_num_identity_resid"] = float(idres)

            case_rep["sweeps"][str(Nk)] = sw_rep
            print(f"  branches={len(br_data)} crossings={cross} "
                  f"Lifshitz(N)={v_life_N} kinks={kinks[:4]} "
                  f"robust(d1,d2)={rob1},{rob2} id={idres:.1e}",
                  flush=True)

        # dOmega/dmu check at one representative point (coarsest Nk)
        chk = check_domega_dmu(U, Dt, Vs[len(Vs) // 2],
                               int(list(case_rep["sweeps"])[0]))
        case_rep["dOmega_dmu_check"] = chk

        # ---- figure ----
        Nk_main = int(list(case_rep["sweeps"])[0])
        sw = case_rep["sweeps"][str(Nk_main)]
        V = np.linspace(v0, v1, sw["V_grid"][2])
        eN = np.array(sw["e_N"])
        e_gs = np.array(sw["e_GS"])
        de = np.array(sw["de_GS"])
        d2 = np.array(sw["d2e_GS"])

        fig, axes = plt.subplots(3, 1, figsize=(8.5, 10.5), sharex=True)
        ax = axes[0]
        ax.plot(V, e_gs - eN, "k-", lw=2.0, label=r"$e_{GS}-e_N$")
        for i, b in enumerate(sw["branches"]):
            bv = np.array(b["V"])
            be = np.array(b["e"])
            ax.plot(bv, be - np.interp(bv, V, eN),
                    "--", lw=1.2, label=f"SC branch {i}")
        ax.plot(V, np.zeros_like(V), "r:", lw=1.0, label="normal")
        if sw["lifshitz_muN_4t"] is not None:
            ax.axvline(sw["lifshitz_muN_4t"], color="gray", ls="-.",
                       label=r"Lifshitz $\mu_N=4t$")
        for c in sw["crossings"]:
            ax.axvline(c["V"], color="orange", alpha=0.6)
        ax.set_ylabel(r"$e-e_N$  [/t]")
        ax.set_title(f"{name}  Nk={Nk_main}  n={N_ELEC}")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

        axes[1].plot(V, de, "k-", lw=1.6, label=r"$de_{GS}/dV$")
        axes[1].plot(V, deN, "b--", lw=1.0, label=r"$de_N/dV$")
        axes[1].set_ylabel(r"$de/dV$")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        axes[2].plot(V, d2, "k-", lw=1.6, label=r"$d^2e_{GS}/dV^2$")
        axes[2].plot(V, d2N, "b--", lw=1.0, label=r"$d^2e_N/dV^2$")
        for k in sw["d2_kink_candidates"]:
            axes[2].axvline(k, color="orange", alpha=0.5)
        axes[2].set_ylabel(r"$d^2e/dV^2$")
        axes[2].set_xlabel(r"$V_{SO}/t$")
        axes[2].legend(fontsize=8)
        axes[2].grid(alpha=0.3)

        fig.tight_layout()
        out = ROOT / "figs" / f"ground_state_energy_{tag}.png"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  wrote {out}", flush=True)

        report["cases"][name] = case_rep

        # npz dump (main Nk)
        arrays = {"V": V, "e_N": eN, "e_GS": e_gs, "de_GS": de,
                  "d2e_GS": d2, "mu_N": np.array(sw["mu_N"])}
        for i, b in enumerate(sw["branches"]):
            arrays[f"e_br{i}"] = np.array(b["e"])
            arrays[f"V_br{i}"] = np.array(b["V"])
        data_out = ROOT / "data" / "ground_state_energy_qpt.npz"
        data_out.parent.mkdir(exist_ok=True)
        # accumulate across cases
        prev = dict(np.load(data_out)) if data_out.exists() else {}
        for k, v in arrays.items():
            prev[f"{tag}_{k}"] = v
        np.savez(data_out, **prev)

    rep_out = ROOT / "reports" / "ground_state_energy_qpt.json"
    rep_out.parent.mkdir(exist_ok=True)
    prev = json.loads(rep_out.read_text()) if rep_out.exists() else {}
    prev.update(report)
    rep_out.write_text(json.dumps(prev, indent=2))
    print(f"wrote {rep_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
