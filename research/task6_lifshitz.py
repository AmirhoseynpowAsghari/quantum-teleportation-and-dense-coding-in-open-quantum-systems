"""TASK 6/7 -- Lifshitz diagnostics at n=1.875, U/t=75, Dt/t=4.5.

Distinguishes three different "Lifshitz" objects that the manuscript
conflates:
  (i)  NORMAL-state band-topology change: mu_N crossing the lower-helicity
       band maximum eps_(pi,pi) = 4t (inner pocket empties).
  (ii) SC-quasiparticle spectral reconstruction: mu_SC crossing 4t, which is
       where the manuscript's quoted 0.5487 (V) and 118.04 (U) come from.
  (iii) Bogoliubov Fermi-surface (node) topology: sign change of
       Delta_k = D0 - (DS/4t) eps_k.

Diagnostics: pocket area in the normal state, the SC quasiparticle gap
closure on the FS, d2e/dV2, and the raw BCS 1/|xi| sums (which are the
source of the spurious lambda_max growth).
"""
import sys, json, time, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np
from sweep_core import setup
from solver import solve_all, sc_energy, make_grid, normal_state


def seeds(g, prev):
    mu = g['nrm']['mu']; S = []
    if prev is not None:
        S += [np.asarray(prev, float)]
    S += [np.array(s, float) for s in [
        [1e-7, 1e-7, mu], [0.01, 0.03, mu], [0.01, -0.03, mu],
        [0.05, 0.15, mu], [0.05, -0.2, mu], [0.15, 0.6, mu],
        [0.15, -0.7, mu], [0.3, 1.3, mu], [0.3, -1.4, mu]]]
    return S


def pocket_area(eps, mu, which=0):
    """Fraction of the BZ occupied by the inner (lower-helicity) pocket,
    i.e. theta(4t - eps_s) restricted to eps_s < mu."""
    e = eps[..., which]
    return float(np.mean((e < mu) & (e > 4.0 - 1e-12)))


def dos_ef(eps, mu, sigma=0.04):
    e = eps.ravel()
    return float(np.mean(np.exp(-0.5*((e-mu)/sigma)**2)/np.sqrt(2*np.pi)/sigma))


if __name__ == '__main__':
    Nk = int(sys.argv[1]) if len(sys.argv) > 1 else 401
    out = {'V': [], 'U': []}
    t0 = time.time()
    # ---- V sweep at U=75
    prev = None
    for V in np.round(np.arange(0.30, 0.801, 0.01), 4):
        g = setup(1.875, float(V), Nk)
        sols, ntr = solve_all(g['eps'], g['kern'], Nk, 4.5, 1.875,
                              seeds(g, prev), tol=1e-11, early=3)
        br = []
        for s in ntr:
            e, eg, dphi = sc_energy(s, g['eps'], Nk, 1.875)
            Dk = s[0] - s[1]*g['eps']/4.0
            br.append(dict(D0=float(s[0]), DS=float(s[1]), mu=float(s[2]),
                           e=float(e), dF=float(e-g['nrm']['e']),
                           min_absD=float(np.min(np.abs(Dk)))))
        br.sort(key=lambda d: d['e'])
        if br:
            prev = (br[0]['D0'], br[0]['DS'], br[0]['mu'])
        out['V'].append(dict(V=float(V), muN=float(g['nrm']['mu']),
                             lam=float(g['lam']),
                             area_inner=pocket_area(g['eps'], g['nrm']['mu']),
                             dos_N=dos_ef(g['eps'], g['nrm']['mu']),
                             branches=br))
        b = br[0] if br else None
        print(f"V={V:.3f} muN={g['nrm']['mu']:.5f} area={out['V'][-1]['area_inner']:.5f} "
              f"dosN={out['V'][-1]['dos_N']:.4f} "
              + (f"mu_SC={b['mu']:.5f} min|D|={b['min_absD']:.3e} dF={b['dF']:+.3e}"
                 if b else 'no SC'), flush=True)
    json.dump(out, open(os.path.join(HERE, f'data/lifshitz_V_Nk{Nk}.json'), 'w'),
              indent=1, default=float)
    print('time', time.time()-t0)
