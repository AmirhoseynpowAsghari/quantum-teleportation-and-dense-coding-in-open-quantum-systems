"""TASK 3 (Fast Jacobian BdG solver) -- High-resolution adaptive (n, V_SO)
phase diagram at U/t=75, Dt/t=4.5 for Nk=201 (4-level adaptive refinement)
and Nk=401 (2-level adaptive refinement), loading existing phase_points.json
cache and reporting exact continuum lambda_cont alongside finite-grid SC class."""
import sys, json, time, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np
import fast_solver as fs

CACHE = os.path.join(HERE, 'data/phase_points.json')


def seeds(mu, prev=None):
    S = []
    if prev is not None:
        S += [np.asarray(prev, float), np.array([prev[0], -prev[1], prev[2]], float)]
    S += [np.array(s, float) for s in [
        [1e-7, 1e-7, mu], [1e-4, 4e-4, mu], [0.005, 0.02, mu], [0.02, -0.05, mu],
        [0.05, 0.2, mu], [0.05, -0.25, mu], [0.15, 0.65, mu], [0.15, -0.7, mu],
        [0.35, 1.5, mu], [0.35, -1.5, mu]]]
    return S


class Evaluator:
    def __init__(self, Nk, periodic=False):
        self.Nk = Nk
        self.periodic = periodic
        self.store = {}
        if not periodic and os.path.exists(CACHE):
            try:
                self.store = json.load(open(CACHE))
            except Exception:
                self.store = {}
        self.grids = {}

    def get_grid(self, V):
        Vk = round(float(V), 5)
        if Vk not in self.grids:
            sk, ssin, eps, _, _, _ = fs.make_grid(self.Nk, 1.0, Vk, periodic=self.periodic)
            kern = fs.kernel_array(sk, ssin, 75.0, 4.5, Vk)
            self.grids[Vk] = (eps, kern)
        return self.grids[Vk]

    def key(self, n, V):
        return f"{self.Nk}|{n:.5f}|{V:.5f}"

    def __call__(self, n, V, prev=None):
        k = self.key(n, V)
        if not self.periodic and k in self.store:
            rec = self.store[k]
            if 'lam_cont' in rec:
                return rec
            # enrich legacy record with exact lam_cont
            eps, kern = self.get_grid(V)
            nrm = fs.normal_state(eps, float(n))
            lam_c, lam_g, _ = fs.lambda_exact_and_matrix(eps, kern, self.Nk, 4.5, nrm['mu'], 75.0)
            rec['lam_cont'] = float(lam_c)
            rec['lam_grid'] = float(lam_g)
            return rec

        eps, kern = self.get_grid(V)
        nrm = fs.normal_state(eps, float(n))
        lam_c, lam_g, _ = fs.lambda_exact_and_matrix(eps, kern, self.Nk, 4.5, nrm['mu'], 75.0)
        ntr = []
        if lam_c > 0.96 or lam_g > 0.96 or prev is not None:
            _, ntr = fs.solve_all(eps, kern, self.Nk, 4.5, float(n),
                                  seeds(nrm['mu'], prev), tol=1e-11, early=2, use_jac=True)
        br = []
        for s in ntr:
            e, eg, dphi = fs.sc_energy(s, eps, self.Nk, float(n))
            br.append(dict(D0=float(s[0]), DS=float(s[1]), mu=float(s[2]),
                           e=float(e), dF=float(e - nrm['e'])))
        br.sort(key=lambda d: d['e'])
        rec = dict(n=float(n), V=float(V), Nk=self.Nk,
                   lam=float(lam_g), lam_cont=float(lam_c), lam_grid=float(lam_g),
                   muN=float(nrm['mu']), nb=len(br), branches=br)
        if not self.periodic:
            self.store[k] = rec
        return rec

    def save(self):
        if not self.periodic:
            json.dump(self.store, open(CACHE, 'w'), default=float)


def cls(rec):
    if rec['nb'] == 0:
        return 'N'
    if any(b['dF'] < -1e-12 for b in rec['branches']):
        return 'S'
    return 'M'


def run_adaptive(Nk, n_iters=3, periodic=False):
    ns = np.round(np.arange(0.50, 1.951, 0.10), 3)
    vs = np.round(np.arange(0.00, 1.601, 0.10), 4)
    ev = Evaluator(Nk, periodic=periodic)
    t0 = time.time()
    grid = {}
    for n in ns:
        prev = None
        for V in vs:
            rec = ev(float(n), float(V), prev=prev)
            grid[(float(n), float(V))] = rec
            if rec['nb'] > 0:
                b0 = rec['branches'][0]
                prev = [b0['D0'], b0['DS'], b0['mu']]
            else:
                prev = None
    print(f"[Nk={Nk} per={periodic}] coarse {len(grid)} pts in {time.time()-t0:.1f}s", flush=True)

    for it in range(n_iters):
        newpts = []
        keys = sorted(grid.keys())
        ns_ = sorted(set(k[0] for k in keys)); vs_ = sorted(set(k[1] for k in keys))
        for i in range(len(ns_)-1):
            for j in range(len(vs_)-1):
                c = [(ns_[i], vs_[j]), (ns_[i+1], vs_[j]),
                     (ns_[i], vs_[j+1]), (ns_[i+1], vs_[j+1])]
                if all(k in grid for k in c):
                    classes = set(cls(grid[k]) for k in c)
                    # also refine around lam_cont=1 if not yet resolved!
                    lams = [grid[k]['lam_cont'] for k in c]
                    cross_lam = (min(lams) <= 1.0 <= max(lams))
                    if len(classes) > 1 or cross_lam:
                        newpts += [(0.5*(ns_[i]+ns_[i+1]), vs_[j]),
                                   (ns_[i], 0.5*(vs_[j]+vs_[j+1])),
                                   (0.5*(ns_[i]+ns_[i+1]), 0.5*(vs_[j]+vs_[j+1]))]
        newpts = sorted(set((round(a, 5), round(b, 5)) for a, b in newpts if (round(a, 5), round(b, 5)) not in grid))
        print(f"[Nk={Nk}] refine iter {it}: {len(newpts)} new pts (elapsed {time.time()-t0:.1f}s)", flush=True)
        for n, V in newpts:
            grid[(n, V)] = ev(n, V)
        ev.save()
        tag = f"phasediagram_Nk{Nk}{'_per' if periodic else ''}.json"
        json.dump(dict(Nk=Nk, periodic=periodic, U=75.0, Dt=4.5,
                       points=[dict(n=a, V=b, **{k: v for k, v in r.items() if k not in ('n', 'V')})
                               for (a, b), r in sorted(grid.items())]),
                  open(os.path.join(HERE, 'data', tag), 'w'), indent=1, default=float)
    print(f"[Nk={Nk}] FINISHED {len(grid)} pts in {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    run_adaptive(201, n_iters=3, periodic=False)
    run_adaptive(401, n_iters=2, periodic=True)
