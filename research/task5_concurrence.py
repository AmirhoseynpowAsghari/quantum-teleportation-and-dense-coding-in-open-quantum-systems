"""TASK 5 -- concurrence across every phase boundary.

Uses the repository's own helicity conditional-Wick construction
(helicity_density_matrix.py) so the concurrence is exactly the manuscript's
definition (general Wootters, spin basis, conditional on one electron per
site).

Decomposition: N_AB (normal exchange) and F_AB (anomalous) are recorded
separately so the anomalous contribution to the concurrence can be removed
("normal-background-only" density matrix F_AB=0) and compared.

Outputs research/data/concurrence_scan.json
"""
import sys, json, time, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np
from sweep_core import setup
from solver import solve_all, sc_energy, bdg_dict
import helicity_density_matrix as hdm

R_LIST = [np.array([0.25, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 1.0]),
          np.array([2.0, 0.0]), np.array([0.5, 0.5])]


def seeds(g, prev):
    mu = g['nrm']['mu']; S = []
    if prev is not None:
        S += [np.asarray(prev, float)]
    S += [np.array(s, float) for s in [
        [1e-7, 1e-7, mu], [1e-3, 3e-3, mu], [0.01, 0.03, mu], [0.02, -0.05, mu],
        [0.05, 0.15, mu], [0.05, -0.2, mu], [0.15, 0.6, mu], [0.15, -0.7, mu],
        [0.3, 1.3, mu], [0.3, -1.4, mu]]]
    return S


def concurrences(g, x):
    bdg = bdg_dict(np.asarray(x), g['eps'], g['KX'], g['KY'], g['ssin'])
    out = {}
    for r in R_LIST:
        tag = f"{r[0]:g}_{r[1]:g}"
        d = hdm.density_at_separation(bdg, r, basis='spin')
        corr = {k: d[k] for k in ('N_A', 'N_B', 'N_AB', 'N_BA', 'F_AB', 'F_BA')}
        rho, w = hdm.conditional_two_qubit_density(corr)
        C = hdm.concurrence(rho)
        corr0 = dict(corr); corr0['F_AB'] = np.zeros_like(corr['F_AB'])
        corr0['F_BA'] = np.zeros_like(corr['F_BA'])
        rho0, w0 = hdm.conditional_two_qubit_density(corr0)
        C0 = hdm.concurrence(rho0)
        out[tag] = dict(C=C, C_normonly=C0, weight=w,
                        Fnorm=float(np.linalg.norm(corr['F_AB'])),
                        Nnorm=float(np.linalg.norm(corr['N_AB'])))
    # normal state concurrence (same machinery, Delta=0)
    return out


def run(Nk, tag):
    NS = np.round(np.arange(0.80, 1.551, 0.02), 3)
    VS = np.round(np.arange(0.30, 1.101, 0.02), 4)
    rows = []
    for n in NS:
        prev = None
        for V in VS:
            g = setup(n, float(V), Nk)
            sols, ntr = solve_all(g['eps'], g['kern'], Nk, 4.5, n,
                                  seeds(g, prev), tol=1e-11, early=3)
            br = []
            for s in ntr:
                e, eg, dphi = sc_energy(s, g['eps'], Nk, n)
                br.append((float(e), s))
            br.sort(key=lambda t: t[0])
            if br:
                pick = min(br, key=lambda t: max(abs(t[1][0]-prev[0]),
                                                 abs(t[1][1]-prev[1]),
                                                 6*abs(t[1][2]-prev[2]))) if prev is not None else br[0]
                prev = (float(pick[1][0]), float(pick[1][1]), float(pick[1][2]))
                Cs = concurrences(g, pick[1])
                dF = float(pick[0] - g['nrm']['e'])
                D0, DS = float(pick[1][0]), float(pick[1][1])
            else:
                # normal state: construct a zero-gap BdG dictionary
                bdg = bdg_dict(np.array([1e-12, 0.0, g['nrm']['mu']]),
                               g['eps'], g['KX'], g['KY'], g['ssin'])
                Cs = {}
                for r in R_LIST:
                    tt = f"{r[0]:g}_{r[1]:g}"
                    d = hdm.density_at_separation(bdg, r, basis='spin')
                    corr = {k: d[k] for k in ('N_A', 'N_B', 'N_AB', 'N_BA',
                                              'F_AB', 'F_BA')}
                    rho, w = hdm.conditional_two_qubit_density(corr)
                    Cs[tt] = dict(C=hdm.concurrence(rho), C_normonly=hdm.concurrence(rho),
                                  weight=w, Fnorm=0.0,
                                  Nnorm=float(np.linalg.norm(corr['N_AB'])))
                dF = 0.0; D0 = DS = 0.0
            rows.append(dict(n=float(n), V=float(V), Nk=Nk, lam=float(g['lam']),
                             muN=float(g['nrm']['mu']), dF=dF, D0=D0, DS=DS,
                             SC=bool(D0 != 0.0 or DS != 0.0), C=Cs))
            print(f"{tag} n={n:.2f} V={V:.2f} lam={g['lam']:7.4f} "
                  f"SC={int(D0!=0 or DS!=0)} dF={dF:+.2e} "
                  f"C.25={Cs['0.25_0']['C']:.4f} "
                  f"C.25n={Cs['0.25_0']['C_normonly']:.4f} "
                  f"C1={Cs['1_0']['C']:.4f}", flush=True)
        json.dump(dict(Nk=Nk, rows=rows),
                  open(os.path.join(HERE, f'data/concurrence_{tag}.json'), 'w'),
                  indent=1, default=float)
    return rows


if __name__ == '__main__':
    t0 = time.time()
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 201,
        sys.argv[2] if len(sys.argv) > 2 else 'Nk201')
    print('time', time.time()-t0)
