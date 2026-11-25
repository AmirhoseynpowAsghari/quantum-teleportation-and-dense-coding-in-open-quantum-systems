"""TASKS 3 & 4 -- U/t slices, (U, V_SO) & (n, U) phase diagrams, broader
parameter-space first-order search (including Dt=3.5, 2.5), and Task 4
Landau free-energy expansion F_eff(psi) + phase-boundary topology."""
import sys, json, time, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np
from scipy.optimize import fsolve, brentq
import fast_solver as fs


def make_seeds(mu, prev=None):
    S = []
    if prev is not None:
        S += [np.asarray(prev, float),
              np.array([prev[0], -prev[1], prev[2]], float)]
    S += [np.array(s, float) for s in [
        [1e-8, 1e-8, mu], [1e-4, 4e-4, mu], [0.01, 0.04, mu], [0.01, -0.04, mu],
        [0.05, 0.2, mu], [0.05, -0.25, mu], [0.15, 0.7, mu], [0.15, -0.7, mu],
        [0.35, 1.6, mu], [0.35, -1.6, mu], [0.8, 3.0, mu], [0.8, -3.0, mu]]]
    return S


def compute_exact_lambda_maps():
    """High-resolution 81x81 maps of continuum lambda_+ and mu_N using periodic
    Nk=301 grid (free of endpoint=True zone-edge bias)."""
    Nk = 301
    ns = np.round(np.linspace(0.50, 1.95, 59), 5)
    Vs = np.round(np.linspace(0.00, 1.60, 65), 5)
    Us = np.round(np.linspace(20.0, 200.0, 73), 4)
    Dt = 4.5

    # Precompute mu_N(n, V) and J2(n, V) = <|eps - mu_N|>
    muN = np.zeros((len(ns), len(Vs)))
    J2 = np.zeros((len(ns), len(Vs)))
    area_inner = np.zeros((len(ns), len(Vs)))
    for j, V in enumerate(Vs):
        sk, ssin, eps, KX, KY, kx = fs.make_grid(Nk, 1.0, float(V), periodic=True)
        N2 = float(Nk*Nk)
        for i, n in enumerate(ns):
            nrm = fs.normal_state(eps, float(n))
            mu = nrm['mu']
            muN[i, j] = mu
            J2[i, j] = float(np.sum(np.abs(eps - mu)) / N2)
            # Inner pocket (band 0) unoccupied above mu (or occupied below mu, max at (pi,pi) is 4t)
            area_inner[i, j] = float(np.mean(eps[..., 0] > mu))

    # (n, V_SO) maps at representative U/t slices
    slices_nV = {}
    for U in (40.0, 55.0, 75.0, 100.0, 130.0, 160.0):
        denom = U - 4.0*Dt*muN
        lam = np.where(denom > 0, (Dt*Dt*J2)/np.where(denom > 0, denom, 1.0), np.inf)
        slices_nV[str(int(U))] = lam.tolist()

    # Exact U_c(n, V_SO) = 4*Dt*muN + Dt^2 * J2 where lambda_cont = 1
    Uc_nV = 4.0*Dt*muN + (Dt**2)*J2

    # (U, V_SO) maps at n=1.0, 1.4, 1.875
    slices_UV = {}
    for n_val in (1.0, 1.4, 1.875):
        i_n = int(np.argmin(np.abs(ns - n_val)))
        # compute exact at n_val
        mu_row = np.zeros(len(Vs)); J2_row = np.zeros(len(Vs))
        for j, V in enumerate(Vs):
            sk, ssin, eps, _, _, _ = fs.make_grid(Nk, 1.0, float(V), periodic=True)
            mu = fs.normal_state(eps, n_val)['mu']
            mu_row[j] = mu
            J2_row[j] = float(np.sum(np.abs(eps - mu)) / (Nk*Nk))
        lam_UV = np.zeros((len(Us), len(Vs)))
        for iu, U in enumerate(Us):
            denom = U - 4.0*Dt*mu_row
            lam_UV[iu] = np.where(denom > 0, (Dt*Dt*J2_row)/np.where(denom > 0, denom, 1.0), np.inf)
        slices_UV[f"n{n_val}"] = dict(
            muN=mu_row.tolist(), J2=J2_row.tolist(),
            Uc=(4.0*Dt*mu_row + Dt*Dt*J2_row).tolist(),
            lam=lam_UV.tolist())

    # (n, U) maps at V_SO = 0.0, 0.5, 0.7
    slices_nU = {}
    for V_val in (0.0, 0.5, 0.7):
        sk, ssin, eps, _, _, _ = fs.make_grid(Nk, 1.0, float(V_val), periodic=True)
        mu_col = np.zeros(len(ns)); J2_col = np.zeros(len(ns))
        for i, n in enumerate(ns):
            mu = fs.normal_state(eps, float(n))['mu']
            mu_col[i] = mu
            J2_col[i] = float(np.sum(np.abs(eps - mu)) / (Nk*Nk))
        lam_nU = np.zeros((len(ns), len(Us)))
        for iu, U in enumerate(Us):
            denom = U - 4.0*Dt*mu_col
            lam_nU[:, iu] = np.where(denom > 0, (Dt*Dt*J2_col)/np.where(denom > 0, denom, 1.0), np.inf)
        slices_nU[f"V{V_val}"] = dict(
            muN=mu_col.tolist(), J2=J2_col.tolist(),
            Uc=(4.0*Dt*mu_col + Dt*Dt*J2_col).tolist(),
            lam=lam_nU.tolist())

    return dict(Nk=Nk, periodic=True, Dt=Dt,
                ns=ns.tolist(), Vs=Vs.tolist(), Us=Us.tolist(),
                muN=muN.tolist(), J2=J2.tolist(), area_inner=area_inner.tolist(),
                Uc_nV=Uc_nV.tolist(), slices_nV=slices_nV,
                slices_UV=slices_UV, slices_nU=slices_nU)


def sc_slice_scan(pairs, fixed_kind, fixed_val, Nk=201, Dt=4.5):
    """Self-consistent multi-seed BdG verification over a 2D parameter slice
    to confirm whether nb>1, metastable SC (dF>0), or subcritical SC (lam<1)
    ever occurs."""
    rows = []
    for p1, p2 in pairs:
        if fixed_kind == 'n':
            U, V, n = float(p1), float(p2), float(fixed_val)
        elif fixed_kind == 'V':
            n, U, V = float(p1), float(p2), float(fixed_val)
        elif fixed_kind == 'U':
            n, V, U = float(p1), float(p2), float(fixed_val)
        sk, ssin, eps, _, _, _ = fs.make_grid(Nk, 1.0, V)
        kern = fs.kernel_array(sk, ssin, U, Dt, V)
        nrm = fs.normal_state(eps, n)
        lam_c, lam_g, _ = fs.lambda_exact_and_matrix(eps, kern, Nk, Dt, nrm['mu'], U)
        ntr = []
        if lam_c > 0.96 or lam_g > 0.96:
            _, ntr = fs.solve_all(eps, kern, Nk, Dt, n, make_seeds(nrm['mu']),
                                  tol=1e-11, early=3, use_jac=True)
        else:
            # also check strongly paired seeds even when lam < 0.96 to make
            # sure no isolated first-order strong-pairing branch exists!
            strong_seeds = [np.array(s, float) for s in [
                [0.2, 1.0, nrm['mu']], [0.2, -1.0, nrm['mu']],
                [0.5, 2.5, nrm['mu']], [0.5, -2.5, nrm['mu']],
                [1.0, 4.0, nrm['mu']], [1.0, -4.0, nrm['mu']]]]
            _, ntr = fs.solve_all(eps, kern, Nk, Dt, n, strong_seeds,
                                  tol=1e-11, early=None, use_jac=True)
        br = []
        for s in ntr:
            e, _, _ = fs.sc_energy(s, eps, Nk, n)
            br.append(dict(D0=float(s[0]), DS=float(s[1]), mu=float(s[2]),
                           dF=float(e - nrm['e'])))
        br.sort(key=lambda d: d['dF'])
        rows.append(dict(n=n, U=U, V=V, Dt=Dt, lam_cont=lam_c, lam_grid=lam_g,
                         muN=float(nrm['mu']), nb=len(br), branches=br))
    return rows


def landau_potential(n, V, U=75.0, Dt=4.5, Nk=201):
    """Exact 1D fixed-density Landau potential F_eff(psi) - F_N along the
    r2=r3=0 ridge (psi = Delta_S)."""
    sk, ssin, eps, _, _, _ = fs.make_grid(Nk, 1.0, V, periodic=True)
    kern = fs.kernel_array(sk, ssin, U, Dt, V)
    nrm = fs.normal_state(eps, n)
    lam_c, lam_g, _ = fs.lambda_exact_and_matrix(eps, kern, Nk, Dt, nrm['mu'], U)
    N2 = float(Nk*Nk)
    psis = np.array([0.002, 0.005, 0.01, 0.02, 0.04, 0.07, 0.10, 0.15,
                     0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00])
    guess = [0.0, nrm['mu']]
    pts = []
    for psi in psis:
        def eq23(u):
            D0, mu = u
            Dks = D0 - (psi/4.0)*eps
            xi = eps - mu
            E = np.clip(np.sqrt(Dks*Dks + xi*xi), 1e-12, None)
            r2 = psi + 2.0*Dt*np.sum(Dks/E)/N2
            r3 = np.sum(xi/E)/(2.0*N2) - (1.0 - n)
            return [r2, r3]
        sol, _, ier, _ = fsolve(eq23, guess, full_output=True, xtol=1e-12)
        guess = list(sol)
        D0, mu = sol
        Dks = D0 - (psi/4.0)*eps; xi = eps - mu
        E = np.clip(np.sqrt(Dks*Dks + xi*xi), 1e-12, None)
        F = 0.5*np.sum(xi - E)/N2 + mu*n - (D0*psi)/(4.0*Dt) + (U/(64.0*Dt*Dt))*psi*psi
        dF = float(F - nrm['e'])
        pts.append(dict(psi=float(psi), D0=float(D0), mu=float(mu),
                        dF=dF, dF_over_psi2=float(dF/(psi*psi))))
    # Polynomial fit dF/psi^2 = a + b psi^2 + c psi^4 over psi in [0.02, 0.65]
    sub = [p for p in pts if 0.02 <= p['psi'] <= 0.65]
    p2 = np.array([p['psi']**2 for p in sub])
    y = np.array([p['dF_over_psi2'] for p in sub])
    c, b, a = np.polyfit(p2, y, 2)
    return dict(n=n, V=V, U=U, Dt=Dt, Nk=Nk, lam_cont=lam_c,
                muN=float(nrm['mu']), a_poly=float(a), b_poly=float(b),
                c_poly=float(c), pts=pts)


if __name__ == '__main__':
    t0 = time.time()
    print("=== 1. Exact continuum lambda_+ & mu_N maps (Nk=301 periodic) ===", flush=True)
    maps = compute_exact_lambda_maps()
    print(f"  done maps in {time.time()-t0:.1f}s", flush=True)

    print("=== 2. SC verification on (U,V), (n,U), and Dt=3.5,2.5 slices ===", flush=True)
    Us_sc = [30, 45, 60, 75, 90, 115, 140, 170]
    Vs_sc = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    ns_sc = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.875]

    sc_uv_10 = sc_slice_scan([(u, v) for u in Us_sc for v in Vs_sc], 'n', 1.0)
    sc_uv_14 = sc_slice_scan([(u, v) for u in Us_sc for v in Vs_sc], 'n', 1.4)
    sc_nu_05 = sc_slice_scan([(n, u) for n in ns_sc for u in Us_sc], 'V', 0.5)
    sc_nu_07 = sc_slice_scan([(n, u) for n in ns_sc for u in Us_sc], 'V', 0.7)
    sc_dt35_n10 = sc_slice_scan([(1.0, v) for v in np.round(np.arange(0.0, 1.51, 0.1), 2)], 'U', 75.0, Dt=3.5)
    sc_dt35_n14 = sc_slice_scan([(1.4, v) for v in np.round(np.arange(0.0, 1.51, 0.1), 2)], 'U', 75.0, Dt=3.5)
    sc_dt25_n18 = sc_slice_scan([(1.875, v) for v in np.round(np.arange(0.0, 1.51, 0.1), 2)], 'U', 75.0, Dt=2.5)

    for name, rows in [('UV_n1.0', sc_uv_10), ('UV_n1.4', sc_uv_14),
                       ('nU_V0.5', sc_nu_05), ('nU_V0.7', sc_nu_07),
                       ('Dt3.5_n1.0', sc_dt35_n10), ('Dt3.5_n1.4', sc_dt35_n14),
                       ('Dt2.5_n1.875', sc_dt25_n18)]:
        max_nb = max(r['nb'] for r in rows)
        meta = sum(1 for r in rows for b in r['branches'] if b['dF'] > 1e-10)
        subcrit = sum(1 for r in rows if r['nb'] > 0 and r['lam_cont'] < 0.99 and r['lam_grid'] < 0.99)
        print(f"  {name:14s}: pts={len(rows):3d} sc={sum(1 for r in rows if r['nb']>0):3d} "
              f"max_nb={max_nb} meta_dF>0={meta} subcrit_SC={subcrit}", flush=True)

    print("=== 3. Task 4 Landau free-energy curves along continuous boundary ===", flush=True)
    landau_cases = [
        (0.80, 0.75, 75.0), (0.95, 0.50, 75.0), (1.00, 0.50, 75.0),
        (1.019, 0.50, 75.0), (1.05, 0.50, 75.0), (1.15, 0.50, 75.0),
        (1.00, 0.54, 75.0), (1.00, 0.70, 75.0), (1.00, 1.235, 75.0),
        (1.40, 0.00, 75.0), (1.40, 0.525, 75.0),
        (1.00, 0.50, 65.0), (1.00, 0.50, 73.9), (1.00, 0.50, 85.0),
    ]
    landau = []
    for n, V, U in landau_cases:
        L = landau_potential(n, V, U=U)
        landau.append(L)
        print(f"  Landau n={n:.3f} V={V:.3f} U={U:.1f} lam={L['lam_cont']:.4f}: "
              f"a={L['a_poly']:+.4e} b={L['b_poly']:+.4e} c={L['c_poly']:+.4e}", flush=True)

    out = dict(maps=maps,
               sc_slices=dict(UV_n10=sc_uv_10, UV_n14=sc_uv_14,
                              nU_V05=sc_nu_05, nU_V07=sc_nu_07,
                              Dt35_n10=sc_dt35_n10, Dt35_n14=sc_dt35_n14,
                              Dt25_n18=sc_dt25_n18),
               landau=landau)
    json.dump(out, open(os.path.join(HERE, 'data/task3_4_slices.json'), 'w'),
              indent=1, default=float)
    print('DONE task3_4_slices in', round(time.time()-t0, 1), 's')
