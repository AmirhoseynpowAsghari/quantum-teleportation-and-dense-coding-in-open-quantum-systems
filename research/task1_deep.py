"""TASK 1 DEEP -- ultra-fine 0.001 & 0.0001 hysteresis sweeps around candidates,
full V in [0, 2.5] scans at n=1.0 and n=1.4 (Nk=201, 401), and compressibility /
phase-separation check (dmu/dn = d^2 F / dn^2) across n in [0.5, 1.95]."""
import sys, json, time, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np
import fast_solver as fs

def make_seeds(mu, prev=None):
    S = []
    if prev is not None:
        S += [np.asarray(prev, float),
              np.array([prev[0], -prev[1], prev[2]], float)]
    S += [np.array(s, float) for s in [
        [1e-8, 1e-8, mu], [1e-4, 4e-4, mu], [0.01, 0.03, mu], [0.01, -0.04, mu],
        [0.05, 0.2, mu], [0.05, -0.25, mu], [0.15, 0.7, mu], [0.15, -0.7, mu],
        [0.3, 1.5, mu], [0.3, -1.5, mu], [0.7, 2.8, mu], [0.7, -2.8, mu]]]
    return S

def solve_pt(n, V, Nk, prev=None, U=75.0, Dt=4.5, early=3):
    sk, ssin, eps, KX, KY, kx = fs.make_grid(Nk, 1.0, V)
    kern = fs.kernel_array(sk, ssin, U, Dt, V)
    nrm = fs.normal_state(eps, n)
    lam_c, lam_g, _ = fs.lambda_exact_and_matrix(eps, kern, Nk, Dt, nrm['mu'], U)
    _, ntr = fs.solve_all(eps, kern, Nk, Dt, n, make_seeds(nrm['mu'], prev),
                          tol=1e-11, early=early, use_jac=True)
    br = []
    for s in ntr:
        e, eg, dphi = fs.sc_energy(s, eps, Nk, n)
        eps_node = 4.0*s[0]/s[1] if abs(s[1]) > 1e-12 else float('nan')
        br.append(dict(D0=float(s[0]), DS=float(s[1]), mu=float(s[2]),
                       e=float(e), dF=float(e - nrm['e']),
                       dens_res=float(dphi), eps_node=float(eps_node)))
    br.sort(key=lambda d: d['e'])
    nxt = prev
    if br:
        pick = br[0] if prev is None else min(br, key=lambda d: max(
            abs(d['D0']-prev[0]), abs(d['DS']-prev[1]), 6*abs(d['mu']-prev[2])))
        nxt = (pick['D0'], pick['DS'], pick['mu'])
    return dict(n=float(n), V=float(V), Nk=int(Nk), lam_cont=lam_c, lam_grid=lam_g,
                muN=float(nrm['mu']), eN=float(nrm['e']), nb=len(br), branches=br), nxt

def hyst_sweep(n, Vs, Nk):
    prev = None; up = []
    for V in Vs:
        r, prev = solve_pt(n, float(V), Nk, prev)
        up.append(r)
    prev = (up[-1]['branches'][0]['D0'], up[-1]['branches'][0]['DS'], up[-1]['branches'][0]['mu']) if up[-1]['branches'] else None
    dn = []
    for V in Vs[::-1]:
        r, prev = solve_pt(n, float(V), Nk, prev)
        dn.append(r)
    dn = dn[::-1]
    pairs = [(u, d) for u, d in zip(up, dn) if u['branches'] and d['branches']]
    hdF = max((abs(u['branches'][0]['dF'] - d['branches'][0]['dF']) for u, d in pairs), default=0.0)
    hD0 = max((abs(u['branches'][0]['D0'] - d['branches'][0]['D0']) for u, d in pairs), default=0.0)
    hDS = max((abs(u['branches'][0]['DS'] - d['branches'][0]['DS']) for u, d in pairs), default=0.0)
    return dict(n=n, Nk=Nk, npts=len(Vs), V_min=float(Vs[0]), V_max=float(Vs[-1]),
                dV=float(Vs[1]-Vs[0]), hyst_dF=float(hdF), hyst_D0=float(hD0), hyst_DS=float(hDS),
                up=up, down=dn)

t0 = time.time()
out = {'adaptive': {}, 'wide_V': {}, 'compressibility': {}}

print("=== 1. Adaptive 0.001 & 0.0001 sweeps around candidate boundaries ===")
specs = [
    ('n1.0_dV0.001', 1.0, np.round(np.arange(1.220, 1.2501, 0.001), 6)),
    ('n1.0_dV0.0001', 1.0, np.round(np.arange(1.2330, 1.23705, 0.0001), 6)),
    ('n1.4_dV0.001', 1.4, np.round(np.arange(0.510, 0.5401, 0.001), 6)),
    ('n1.4_dV0.0001', 1.4, np.round(np.arange(0.5230, 0.52705, 0.0001), 6)),
]
for label, n, Vs in specs:
    for Nk in (201, 401):
        res = hyst_sweep(n, Vs, Nk)
        out['adaptive'][f"{label}_Nk{Nk}"] = res
        b0 = res['up'][0]['branches'][0]; b1 = res['up'][-1]['branches'][0]
        print(f"  {label}_Nk{Nk} ({len(Vs)} pts): hyst_dF={res['hyst_dF']:.2e} hyst_D0={res['hyst_D0']:.2e} "
              f"| dF: {b0['dF']:+.5e} -> {b1['dF']:+.5e} | max_nb={max(r['nb'] for r in res['up'])}", flush=True)

print("=== 2. Wide V in [0, 2.5] sweeps at n=1.0 and n=1.4 ===")
Vs_wide = np.round(np.arange(0.0, 2.501, 0.05), 4)
for n in (1.0, 1.4):
    for Nk in (201, 401):
        prev = None; rows = []
        for V in Vs_wide:
            r, prev = solve_pt(n, float(V), Nk, prev, early=4)
            rows.append(r)
        out['wide_V'][f"n{n}_Nk{Nk}"] = rows
        nsc = sum(1 for r in rows if r['nb'] > 0)
        max_nb = max(r['nb'] for r in rows)
        print(f"  wide_V n={n} Nk={Nk}: {nsc}/{len(rows)} SC, max_nb={max_nb}", flush=True)

print("=== 3. Compressibility & phase separation across n in [0.50, 1.95] ===")
ns = np.round(np.arange(0.50, 1.951, 0.025), 4)
for V in (0.0, 0.5, 0.7, 1.0, 1.235, 1.4):
    Nks = (201, 401) if V in (0.5, 1.0) else (201,)
    for Nk in Nks:
        prev = None; rows = []
        for n in ns:
            r, prev = solve_pt(float(n), float(V), Nk, prev, early=3)
            rows.append(r)
        e_gs = np.array([r['branches'][0]['e'] if r['branches'] else r['eN'] for r in rows])
        mu_gs = np.array([r['branches'][0]['mu'] if r['branches'] else r['muN'] for r in rows])
        de_dn = np.gradient(e_gs, ns, edge_order=2)
        dmu_dn = np.gradient(mu_gs, ns, edge_order=2)
        d2e_dn2 = np.gradient(de_dn, ns, edge_order=2)
        max_thermo_err = float(np.max(np.abs(de_dn[2:-2] - mu_gs[2:-2])))
        min_dmu = float(np.min(dmu_dn))
        min_d2e = float(np.min(d2e_dn2))
        out['compressibility'][f"V{V}_Nk{Nk}"] = dict(
            V=V, Nk=Nk, rows=rows,
            min_dmu_dn=min_dmu, min_d2e_dn2=min_d2e,
            max_dedn_minus_mu=max_thermo_err)
        print(f"  comp V={V:5.3f} Nk={Nk}: min(dmu/dn)={min_dmu:+.4f}, min(d2e/dn2)={min_d2e:+.4f}, "
              f"max|de/dn-mu|={max_thermo_err:.3e}", flush=True)

json.dump(out, open(os.path.join(HERE, 'data/task1_deep.json'), 'w'), indent=1, default=float)
print('DONE task1_deep in', round(time.time()-t0, 1), 's')
