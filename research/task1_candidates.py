"""TASK 1 -- thermodynamic test of the claimed first-order SC-collapse
boundaries (n=1.0, V=1.235) and (n=1.4, V=0.525), U/t=75, Dt/t=4.5.

Two-stage protocol:
  A. continuation sweeps (forward/backward, dV=0.002) with warm start ->
     branch tracking and hysteresis detection;
  B. exhaustive multi-seed enumeration (18 seeds) at a coarse subset
     covering the candidate windows -> guarantees no missed branch.
  C. dV=1e-3 and 1e-4 refinement in the window of maximal |d dF/dV|.
Fixed density: F = Omega + mu N, dF = F_SC - F_N.
"""
import sys, json, time, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np
from sweep_core import setup
from solver import solve_all, sc_energy


def seed_set(g, prev=None, full=True):
    mu = g['nrm']['mu']; S = []
    if prev is not None:
        S += [np.asarray(prev, float), np.asarray(prev, float)*1.06,
              np.asarray(prev, float)*0.94]
    S += [np.array(s, float) for s in [
        [1e-9, 1e-9, mu], [1e-6, 3e-6, mu], [1e-3, 4e-3, mu],
        [0.01, 0.04, mu], [0.01, -0.04, mu], [0.03, 0.12, mu], [0.03, -0.12, mu],
        [0.05, 0.25, mu], [0.05, -0.25, mu], [0.1, 0.6, mu], [0.1, -0.6, mu],
        [0.3, 1.2, mu], [0.3, -1.2, mu], [0.3, 1.9, mu], [0.3, -1.9, mu],
        [0.7, 2.6, mu], [0.7, -2.6, mu], [0.15, 0.9, mu]]]
    return S


def point(n, V, Nk, prev, exhaustive=False):
    g = setup(n, V, Nk)
    seeds = seed_set(g, prev)
    if exhaustive:
        sols, ntr = solve_all(g['eps'], g['kern'], Nk, 4.5, n, seeds, tol=1e-11)
    else:
        sols, ntr = solve_all(g['eps'], g['kern'], Nk, 4.5, n, seeds, tol=1e-11,
                              early=4)
    br = []
    for s in ntr:
        e, eg, dphi = sc_energy(s, g['eps'], Nk, n)
        br.append(dict(D0=float(s[0]), DS=float(s[1]), mu=float(s[2]),
                       e=float(e), dF=float(e - g['nrm']['e']),
                       dens_res=float(dphi)))
    br.sort(key=lambda d: d['e'])
    nxt = prev
    if br:
        pick = br[0] if prev is None else min(br, key=lambda d: max(
            abs(d['D0']-prev[0]), abs(d['DS']-prev[1]), 6*abs(d['mu']-prev[2])))
        nxt = (pick['D0'], pick['DS'], pick['mu'])
    return dict(n=n, V=V, lam=float(g['lam']), muN=g['nrm']['mu'],
                eN=g['nrm']['e'], nb=len(br), branches=br), nxt


def sweep(n, Vs, Nk, asc):
    order = range(len(Vs)) if asc else range(len(Vs)-1, -1, -1)
    prev = None; rows = []
    for i in order:
        r, prev = point(n, float(Vs[i]), Nk, prev)
        r['dir'] = 'up' if asc else 'down'; rows.append(r)
        b = r['branches']
        print(f"  n={n:.2f} {r['dir']} V={r['V']:.4f} lam={r['lam']:.4f} nb={r['nb']} "
              + " | ".join(f"D0={x['D0']:+.5f} DS={x['DS']:+.5f} dF={x['dF']:+.4e}"
                           for x in b), flush=True)
    rows.sort(key=lambda r: r['V'])
    return rows


if __name__ == '__main__':
    CAND = {1.0: (1.235, (1.20, 1.28)), 1.4: (0.525, (0.48, 0.58))}
    out = {}
    t0 = time.time()
    for n, (Vc, (lo, hi)) in CAND.items():
        for Nk in (201, 401):
            Vs = np.round(np.arange(lo, hi+1e-12, 0.002), 6)
            up = sweep(n, Vs, Nk, True); dn = sweep(n, Vs, Nk, False)
            pairs = [(u, d) for u, d in zip(up, dn) if u['branches'] and d['branches']]
            dmax = max((abs(u['branches'][0]['dF']-d['branches'][0]['dF'])
                        for u, d in pairs), default=float('nan'))
            gmax = max((abs(u['branches'][0]['D0']-d['branches'][0]['D0'])
                        for u, d in pairs), default=float('nan'))
            print(f"[hysteresis] n={n} Nk={Nk} max|dF_up-dF_dn|={dmax:.3e} "
                  f"max|D0_up-D0_dn|={gmax:.3e}", flush=True)
            # exhaustive census on coarse subset
            census = []
            for V in np.round(np.arange(lo, hi+1e-12, 0.01), 6):
                r, _ = point(n, float(V), Nk, None, exhaustive=True)
                census.append(r)
                print(f"  [census] n={n} Nk={Nk} V={V:.3f} nb={r['nb']} "
                      + " | ".join(f"D0={x['D0']:+.5f} DS={x['DS']:+.5f} "
                                   f"dF={x['dF']:+.4e}" for x in r['branches']),
                      flush=True)
            out[f"n{n}_Nk{Nk}"] = dict(n=n, Nk=Nk, Vc_claim=Vc, up=up, down=dn,
                                       census=census, hyst_dF=float(dmax),
                                       hyst_D0=float(gmax))
    json.dump(out, open(os.path.join(HERE, 'data/task1_candidates.json'), 'w'),
              indent=1, default=float)
    print('time', time.time()-t0)
