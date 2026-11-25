"""TASK 2 -- continuous QPT: refine n_c at V=0.5, test |Delta| vs |n-n_c|^beta
and |Delta|^2 vs lambda_max-1, at several Nk, plus the lambda_max=1 locus."""
import sys, json, time, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np
from sweep_core import setup
from solver import solve_all, sc_energy


def seeds(g, prev, tiny=True):
    mu = g['nrm']['mu']; S = []
    if prev is not None:
        S += [np.asarray(prev, float)]
    S += [np.array(s, float) for s in [
        [1e-8, 1e-8, mu], [1e-6, 2e-6, mu], [1e-4, 3e-4, mu],
        [1e-3, 4e-3, mu], [0.003, -0.01, mu], [0.01, 0.03, mu],
        [0.01, -0.04, mu], [0.03, 0.12, mu], [0.03, -0.12, mu],
        [0.1, 0.5, mu], [0.1, -0.6, mu]]]
    return S


def sc_gap(g, n, prev, tol=1e-11):
    """Return (order-parameter norms, dF) for every self-consistent branch."""
    sols, ntr = solve_all(g['eps'], g['kern'], g['Nk'], 4.5, n,
                          seeds(g, prev), tol=tol, early=4)
    out = []
    for s in ntr:
        e, eg, dphi = sc_energy(s, g['eps'], g['Nk'], n)
        # gauge-invariant amplitude: gap on the FS, Delta_k = D0 - DS eps/4
        mu = s[2]
        Dk = abs(s[0] - s[1]*mu/4.0)
        out.append(dict(D0=float(s[0]), DS=float(s[1]), mu=float(mu),
                        absD_fs=float(Dk), dF=float(e - g['nrm']['e']),
                        amp=float(np.hypot(s[0], s[1]))))
    out.sort(key=lambda d: d['dF'])
    return out


if __name__ == '__main__':
    res = {}
    t0 = time.time()
    for Nk in (201, 401, 801):
        for V in (0.45, 0.50, 0.55):
            ns = np.round(np.arange(0.94, 1.1151, 0.005), 4)
            rows = []; prev = None
            for n in ns:
                g = setup(n, V, Nk)
                br = sc_gap(g, float(n), prev)
                if br:
                    prev = (br[0]['D0'], br[0]['DS'], br[0]['mu'])
                rows.append(dict(n=float(n), lam=float(g['lam']),
                                 muN=float(g['nrm']['mu']), branches=br))
                print(f"Nk={Nk} V={V} n={n:.3f} lam={g['lam']:.5f} "
                      f"nb={len(br)} " + " | ".join(
                          f"|D_FS|={b['absD_fs']:.3e} dF={b['dF']:+.2e}"
                          for b in br), flush=True)
            res[f"{Nk}|{V}"] = dict(Nk=Nk, V=V, rows=rows)
            json.dump(res, open(os.path.join(HERE, 'data/scaling.json'), 'w'),
                      indent=1, default=float)
            print(f'--- Nk={Nk} V={V} {time.time()-t0:.0f}s', flush=True)
    print('time', time.time()-t0)
