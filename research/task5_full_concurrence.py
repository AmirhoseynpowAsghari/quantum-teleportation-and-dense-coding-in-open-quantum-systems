"""TASK 5 -- Concurrence, derivatives, normal/anomalous decomposition, raw
Wootters eigenvalue gap, and multiple separations across all phase boundaries
and crossovers."""
import sys, json, time, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np
import fast_solver as fs
import helicity_density_matrix as hdm

R_VEC = {
    '0.25_0': np.array([0.25, 0.0]),
    '0.35_0': np.array([0.35, 0.0]),
    '0.5_0':  np.array([0.50, 0.0]),
    '0.75_0': np.array([0.75, 0.0]),
    '1_0':    np.array([1.00, 0.0]),
    '0.5_0.5': np.array([0.50, 0.50]),
    '1_1':    np.array([1.00, 1.00]),
}


def raw_wootters_and_C(rho):
    rho = np.asarray(rho, dtype=complex)
    sf = np.kron(hdm.PAULI_Y, hdm.PAULI_Y) @ rho.conj() @ np.kron(hdm.PAULI_Y, hdm.PAULI_Y)
    ev = np.linalg.eigvals(rho @ sf)
    roots = np.sort(np.sqrt(np.clip(ev.real, 0.0, None)))[::-1]
    raw = float(roots[0] - np.sum(roots[1:]))
    return max(0.0, raw), raw


def eval_bdg_metrics(bdg, r_keys=('0.25_0', '0.5_0', '1_0', '0.5_0.5'),
                     bases=('spin', 'band')):
    out = {}
    for basis in bases:
        bdict = {}
        for rk in r_keys:
            r = R_VEC[rk]
            d = hdm.density_at_separation(bdg, r, basis=basis)
            rho = d['rho']
            C, C_raw = raw_wootters_and_C(rho)
            Fs = float(np.real(hdm.SINGLET.conj() @ rho @ hdm.SINGLET))
            corr0 = {k: d[k] for k in ('N_A', 'N_B', 'N_AB', 'N_BA', 'F_AB', 'F_BA')}
            corr0['F_AB'] = np.zeros_like(corr0['F_AB'])
            corr0['F_BA'] = np.zeros_like(corr0['F_BA'])
            rho0, _ = hdm.conditional_two_qubit_density(corr0)
            C0, C0_raw = raw_wootters_and_C(rho0)
            bdict[rk] = dict(
                C=C, C_raw=C_raw, C_normonly=C0, C_normonly_raw=C0_raw,
                Fs=Fs, weight=float(d['weight']),
                Fnorm=float(np.linalg.norm(d['F_AB'])),
                Nnorm=float(np.linalg.norm(d['N_AB'])),
            )
        out[basis] = bdict
    return out


def make_seeds(mu, prev=None):
    S = []
    if prev is not None:
        S += [np.asarray(prev, float),
              np.array([prev[0], -prev[1], prev[2]], float)]
    S += [np.array(s, float) for s in [
        [1e-8, 1e-8, mu], [1e-4, 4e-4, mu], [0.01, 0.03, mu], [0.01, -0.04, mu],
        [0.05, 0.2, mu], [0.05, -0.25, mu], [0.15, 0.7, mu], [0.15, -0.7, mu],
        [0.3, 1.5, mu], [0.3, -1.5, mu]]]
    return S


def run_cut(name, pts, Nk=201, r_keys=('0.25_0', '0.35_0', '0.5_0', '0.75_0', '1_0', '0.5_0.5')):
    rows = []; prev = None
    for n, U, V, param_val in pts:
        sk, ssin, eps, KX, KY, kx = fs.make_grid(Nk, 1.0, float(V))
        kern = fs.kernel_array(sk, ssin, float(U), 4.5, float(V))
        nrm = fs.normal_state(eps, float(n))
        lam_c, lam_g, _ = fs.lambda_exact_and_matrix(eps, kern, Nk, 4.5, nrm['mu'], float(U))
        _, ntr = fs.solve_all(eps, kern, Nk, 4.5, float(n), make_seeds(nrm['mu'], prev),
                              tol=1e-11, early=3, use_jac=True)
        # normal state reference BdG
        bdg_N = fs.bdg_dict([0.0, 0.0, nrm['mu']], eps, KX, KY, ssin)
        m_N = eval_bdg_metrics(bdg_N, r_keys=r_keys, bases=('spin', 'band'))

        if ntr:
            br = [(fs.sc_energy(s, eps, Nk, float(n))[0], s) for s in ntr]
            br.sort(key=lambda t: t[0])
            pick = br[0] if prev is None else min(br, key=lambda t: max(
                abs(t[1][0]-prev[0]), abs(t[1][1]-prev[1]), 6*abs(t[1][2]-prev[2])))
            x = pick[1]; prev = x
            e_sc = pick[0]; dF = float(e_sc - nrm['e'])
            bdg_SC = fs.bdg_dict(x, eps, KX, KY, ssin)
            m_SC = eval_bdg_metrics(bdg_SC, r_keys=r_keys, bases=('spin', 'band'))
            D0, DS, mu = float(x[0]), float(x[1]), float(x[2])
            sc_flag = True
        else:
            prev = None; dF = 0.0; D0 = DS = 0.0; mu = float(nrm['mu'])
            m_SC = m_N; sc_flag = False

        rows.append(dict(param=float(param_val), n=float(n), U=float(U), V=float(V),
                         lam_cont=lam_c, lam_grid=lam_g, sc=sc_flag,
                         D0=D0, DS=DS, mu=mu, muN=float(nrm['mu']), dF=dF,
                         sc_metrics=m_SC, norm_metrics=m_N))
    # attach derivatives for primary spin & band observables
    p = np.array([r['param'] for r in rows])
    derivs = {}
    for basis in ('spin', 'band'):
        derivs[basis] = {}
        for rk in r_keys:
            C = np.array([r['sc_metrics'][basis][rk]['C'] for r in rows])
            Cn = np.array([r['sc_metrics'][basis][rk]['C_normonly'] for r in rows])
            Cp = np.array([r['norm_metrics'][basis][rk]['C'] for r in rows])
            dC = np.gradient(C, p, edge_order=2)
            d2C = np.gradient(dC, p, edge_order=2)
            derivs[basis][rk] = dict(dC=dC.tolist(), d2C=d2C.tolist(),
                                     max_abs_dC=float(np.max(np.abs(dC))),
                                     max_abs_d2C=float(np.max(np.abs(d2C))),
                                     max_anom_diff=float(np.max(np.abs(C - Cn))),
                                     max_sc_vs_norm=float(np.max(np.abs(C - Cp))))
    return dict(name=name, Nk=Nk, rows=rows, derivs=derivs)


if __name__ == '__main__':
    t0 = time.time()
    out = {}

    # 1. Cut along n at V_SO=0.50 (continuous QPT n_c ~ 1.019)
    ns1 = np.unique(np.round(np.concatenate([
        np.arange(0.85, 1.351, 0.01),
        np.arange(0.99, 1.061, 0.002)]), 5))
    out['cut_n_V0.5'] = run_cut('cut_n_V0.5', [(n, 75.0, 0.50, n) for n in ns1], Nk=201)
    print("cut_n_V0.5 done", round(time.time()-t0, 1), "s", flush=True)

    # 2. Cut along V_SO at n=1.0 (onset ~0.54, candidate 1.235)
    Vs2 = np.unique(np.round(np.concatenate([
        np.arange(0.20, 1.401, 0.02),
        np.arange(0.50, 0.621, 0.005),
        np.arange(1.20, 1.271, 0.005)]), 5))
    out['cut_V_n1.0'] = run_cut('cut_V_n1.0', [(1.0, 75.0, v, v) for v in Vs2], Nk=201)
    print("cut_V_n1.0 done", round(time.time()-t0, 1), "s", flush=True)

    # 3. Cut along V_SO at n=1.4 (candidate 0.525, nodal flip ~0.98)
    Vs3 = np.unique(np.round(np.concatenate([
        np.arange(0.00, 1.201, 0.02),
        np.arange(0.49, 0.561, 0.005),
        np.arange(0.94, 1.021, 0.005)]), 5))
    out['cut_V_n1.4'] = run_cut('cut_V_n1.4', [(1.4, 75.0, v, v) for v in Vs3], Nk=201)
    print("cut_V_n1.4 done", round(time.time()-t0, 1), "s", flush=True)

    # 4. Cleanest continuous pairing QPT: along U at n=1.0, V_SO=0.50 (mu_N fixed!)
    Us4 = np.unique(np.round(np.concatenate([
        np.arange(40.0, 110.1, 2.0),
        np.arange(68.0, 80.1, 0.5)]), 4))
    out['cut_U_n1.0_V0.5'] = run_cut('cut_U_n1.0_V0.5', [(1.0, u, 0.50, u) for u in Us4], Nk=201)
    print("cut_U_n1.0_V0.5 done", round(time.time()-t0, 1), "s", flush=True)

    # 5. Original protocol n=1.875 cuts (V_SO sweep & U sweeps at V=0.5 and V=0.7)
    Vs5 = np.round(np.arange(0.00, 1.501, 0.02), 4)
    out['cut_V_n1.875'] = run_cut('cut_V_n1.875', [(1.875, 75.0, v, v) for v in Vs5], Nk=201)
    Us5 = np.round(np.arange(20.0, 160.1, 2.5), 4)
    out['cut_U_n1.875_V0.5'] = run_cut('cut_U_n1.875_V0.5', [(1.875, u, 0.50, u) for u in Us5], Nk=201)
    out['cut_U_n1.875_V0.7'] = run_cut('cut_U_n1.875_V0.7', [(1.875, u, 0.70, u) for u in Us5], Nk=201)
    print("cut_n1.875 done", round(time.time()-t0, 1), "s", flush=True)

    # 6. Nk=401 convergence check on U and n cuts
    Us6 = np.round(np.arange(56.0, 92.1, 2.0), 4)
    out['cut_U_n1.0_V0.5_Nk401'] = run_cut('cut_U_n1.0_V0.5_Nk401',
                                           [(1.0, u, 0.50, u) for u in Us6], Nk=401,
                                           r_keys=('0.25_0', '0.5_0'))
    ns6 = np.round(np.arange(0.95, 1.151, 0.01), 4)
    out['cut_n_V0.5_Nk401'] = run_cut('cut_n_V0.5_Nk401',
                                      [(n, 75.0, 0.50, n) for n in ns6], Nk=401,
                                      r_keys=('0.25_0', '0.5_0'))

    json.dump(out, open(os.path.join(HERE, 'data/task5_concurrence.json'), 'w'),
              indent=1, default=float)
    print('DONE task5_full_concurrence in', round(time.time()-t0, 1), 's')
