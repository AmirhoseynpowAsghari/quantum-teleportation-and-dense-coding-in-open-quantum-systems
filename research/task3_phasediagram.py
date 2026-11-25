"""TASK 3 -- (n,V_SO) phase diagram at U/t=75, Dt/t=4.5.

Adaptive design (cost control): a coarse grid first, then local refinement
only in cells whose four corners are not all in the same phase class, so
the boundary resolution is set by the refinement depth, not by the coarse
step.  For each state point we record
  * lambda_max of the linearised gap map at the normal state (plus its
    Nk-dependence, which exposes the finite-grid cutoff);
  * every self-consistent branch reachable from 10 seeds (continuation
    warm-started) with F = Omega + mu N and dF = F_SC - F_N;
  * thermodynamic class: normal / SC-stable (dF<0) / SC-metastable (dF>0).
"""
import sys, json, time, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np
from sweep_core import setup
from solver import solve_all, sc_energy

CACHE = os.path.join(HERE, 'data/phase_points.json')


def seeds(g, prev):
    mu = g['nrm']['mu']; S = []
    if prev is not None:
        S += [np.asarray(prev, float)]
    S += [np.array(s, float) for s in [
        [1e-7, 1e-7, mu], [1e-3, 3e-3, mu], [0.01, 0.03, mu], [0.02, -0.05, mu],
        [0.05, 0.15, mu], [0.05, -0.2, mu], [0.15, 0.6, mu], [0.15, -0.7, mu],
        [0.3, 1.3, mu], [0.3, -1.4, mu]]]
    return S


class Evaluator:
    def __init__(self, Nk):
        self.Nk = Nk
        self.store = {}
        if os.path.exists(CACHE):
            try:
                self.store = json.load(open(CACHE))
            except Exception:
                self.store = {}

    def key(self, n, V):
        return f"{self.Nk}|{n:.5f}|{V:.5f}"

    def __call__(self, n, V, prev=None):
        k = self.key(n, V)
        if k in self.store:
            return self.store[k]
        g = setup(n, V, self.Nk)
        # cheap pre-filter: below lambda=0.97 the normal state is stable and
        # (verified on a control subset) no self-consistent SC root exists.
        ntr = []
        if g['lam'] > 0.97 or prev is not None:
            sols, ntr = solve_all(g['eps'], g['kern'], self.Nk, 4.5, n,
                                  seeds(g, prev), tol=1e-11, early=2)
            if not ntr and g['lam'] > 0.99:
                sols, ntr = solve_all(g['eps'], g['kern'], self.Nk, 4.5, n,
                                      seeds(g, prev), tol=1e-11)
        br = []
        for s in ntr:
            e, eg, dphi = sc_energy(s, g['eps'], self.Nk, n)
            br.append(dict(D0=float(s[0]), DS=float(s[1]), mu=float(s[2]),
                           e=float(e), dF=float(e - g['nrm']['e'])))
        br.sort(key=lambda d: d['e'])
        rec = dict(n=float(n), V=float(V), Nk=self.Nk, lam=float(g['lam']),
                   muN=float(g['nrm']['mu']), nb=len(br), branches=br)
        self.store[k] = rec
        return rec

    def save(self):
        json.dump(self.store, open(CACHE, 'w'), default=float)


def cls(rec):
    if rec['nb'] == 0:
        return 'N'
    if any(b['dF'] < -1e-12 for b in rec['branches']):
        return 'S'
    return 'M'


if __name__ == '__main__':
    Nk = int(sys.argv[1]) if len(sys.argv) > 1 else 201
    ns = np.round(np.arange(0.50, 1.951, 0.10), 3)
    vs = np.round(np.arange(0.00, 1.601, 0.10), 4)
    ev = Evaluator(Nk)
    t0 = time.time()
    grid = {}
    for n in ns:
        for V in vs:
            grid[(float(n), float(V))] = ev(float(n), float(V))
            rec = grid[(float(n), float(V))]
            print(f"Nk={Nk} n={n:.2f} V={V:.2f} lam={rec['lam']:9.4g} "
                  f"nb={rec['nb']} " + " | ".join(
                      f"D0={b['D0']:+.4f} DS={b['DS']:+.4f} dF={b['dF']:+.2e}"
                      for b in rec['branches']), flush=True)
        print(f'  [{time.time()-t0:.0f}s] cached={len(ev.store)}', flush=True)
        if int(time.time()-t0) % 300 < 200:
            ev.save()
    ev.save()
    # adaptive refinement: bisect edges whose endpoints differ in class
    for it in range(3):
        newpts = []
        keys = sorted(grid.keys())
        ns_ = sorted(set(k[0] for k in keys)); vs_ = sorted(set(k[1] for k in keys))
        for i in range(len(ns_)-1):
            for j in range(len(vs_)-1):
                c = [(ns_[i], vs_[j]), (ns_[i+1], vs_[j]),
                     (ns_[i], vs_[j+1]), (ns_[i+1], vs_[j+1])]
                if all(k in grid for k in c):
                    if len(set(cls(grid[k]) for k in c)) > 1:
                        newpts += [(0.5*(ns_[i]+ns_[i+1]), vs_[j]),
                                   (ns_[i], 0.5*(vs_[j]+vs_[j+1])),
                                   (0.5*(ns_[i]+ns_[i+1]), 0.5*(vs_[j]+vs_[j+1]))]
        newpts = sorted(set((round(a, 5), round(b, 5)) for a, b in newpts))
        print(f'refine iter {it}: {len(newpts)} new points', flush=True)
        for n, V in newpts:
            if (n, V) not in grid:
                grid[(n, V)] = ev(n, V)
        ev.save()
        json.dump(dict(Nk=Nk, U=75.0, Dt=4.5,
                       points=[dict(n=a, V=b, **{k: v for k, v in r.items()
                                                 if k not in ('n', 'V')})
                               for (a, b), r in grid.items()]),
                  open(os.path.join(HERE, f'data/phasediagram_Nk{Nk}.json'), 'w'),
                  indent=1, default=float)
    print('time', time.time()-t0)
