"""TASKS 6 & 7 -- Revalidation of n=1.875 manuscript claims (Lifshitz vs
mu_SC=4t crossover, U=68.55 at V=0.5, U=118.04 at V=0.7, Bogoliubov spectral
gap) and systematic Nk=(101,201,401,801) + endpoint=True/False + tolerance +
derivative convergence audit."""
import sys, json, time, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np
from scipy.optimize import brentq
import fast_solver as fs
import helicity_density_matrix as hdm


def solve_single(n, U, V, Nk, prev=None, periodic=False, xtol=1.49e-12):
    sk, ssin, eps, KX, KY, kx = fs.make_grid(Nk, 1.0, V, periodic=periodic)
    kern = fs.kernel_array(sk, ssin, U, 4.5, V)
    nrm = fs.normal_state(eps, n)
    seeds = []
    if prev is not None:
        seeds.append(np.asarray(prev, float))
    seeds += [np.array(s, float) for s in [
        [0.24, 1.76, nrm['mu']], [0.22, 2.0, nrm['mu']], [0.3, -2.0, nrm['mu']],
        [0.14, 1.98, nrm['mu']], [0.65, -3.1, nrm['mu']], [0.02, 0.3, nrm['mu']],
        [0.02, -0.3, nrm['mu']]]]
    best = None
    for s in seeds:
        cand, r, ier = fs.solve_one(eps, kern, Nk, 4.5, n, s, xtol=xtol, use_jac=True)
        if cand is not None and r < 1e-10 and max(abs(cand[0]), abs(cand[1])) > 1e-6:
            cand = fs._canon(cand)
            e, eg, dphi = fs.sc_energy(cand, eps, Nk, n)
            if best is None or e < best['e']:
                Dks = cand[0] - (cand[1]/4.0)*eps
                xi = eps - cand[2]
                Eks = np.sqrt(Dks*Dks + xi*xi)
                v2_0 = 0.5*(1.0 - xi[..., 0]/np.clip(Eks[..., 0], 1e-12, None))
                # near (pi, pi) corner where eps_0 has band maximum 4t:
                corner = eps[..., 0] > 3.9
                best = dict(
                    D0=float(cand[0]), DS=float(cand[1]), mu_SC=float(cand[2]),
                    e_SC=float(e), e_N=float(nrm['e']), mu_N=float(nrm['mu']),
                    dF=float(e - nrm['e']), res=float(r), dens_res=float(dphi),
                    min_E_band0=float(np.min(Eks[..., 0])),
                    min_E_band1=float(np.min(Eks[..., 1])),
                    E_pipi_band0=float(Eks[-1, -1, 0]),
                    occ_pipi_SC=float(v2_0[-1, -1]),
                    occ_corner_SC=float(np.mean(v2_0[corner])) if corner.any() else float('nan'),
                    area_N_eps0_gt_muN=float(np.mean(eps[..., 0] > nrm['mu'])),
                    area_mixed_eps0_gt_muSC=float(np.mean(eps[..., 0] > cand[2])),
                    e=float(e), x=cand,
                )
            if r < 1e-11:
                break
    return best, nrm


def crossing(xs, ys, target=4.0):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    for i in range(len(xs) - 1):
        y0, y1 = ys[i] - target, ys[i+1] - target
        if y0 == 0.0:
            return float(xs[i])
        if y0 * y1 < 0.0:
            w = -y0 / (y1 - y0)
            return float(xs[i] + w*(xs[i+1] - xs[i]))
    return None


if __name__ == '__main__':
    t0 = time.time()
    out = {'task6_V_sweep': {}, 'task6_U_sweeps': {}, 'task7_convergence': {}}

    print("=== 1. Task 6 V_SO sweep at n=1.875, U=75 (Nk=201, 401, 801) ===", flush=True)
    Vs = np.unique(np.round(np.concatenate([
        np.arange(0.00, 1.501, 0.02),
        np.arange(0.52, 0.741, 0.005)]), 5))
    for Nk in (201, 401):
        for periodic in (False, True):
            tag = f"Nk{Nk}_{'per' if periodic else 'repo'}"
            rows = []; prev = None
            for V in Vs:
                b, nrm = solve_single(1.875, 75.0, float(V), Nk, prev=prev, periodic=periodic)
                prev = b['x']
                r = {k: v for k, v in b.items() if k not in ('e', 'x')}
                r['V'] = float(V)
                rows.append(r)
            vA = np.array([r['V'] for r in rows])
            eSC = np.array([r['e_SC'] for r in rows])
            eN = np.array([r['e_N'] for r in rows])
            muN = np.array([r['mu_N'] for r in rows])
            muSC = np.array([r['mu_SC'] for r in rows])
            d2_SC = np.gradient(np.gradient(eSC, vA, edge_order=2), vA, edge_order=2)
            d2_N = np.gradient(np.gradient(eN, vA, edge_order=2), vA, edge_order=2)
            v_muN = crossing(vA, muN, 4.0)
            v_muSC = crossing(vA, muSC, 4.0)
            out['task6_V_sweep'][tag] = dict(
                Nk=Nk, periodic=periodic,
                V_Lifshitz_muN_4t=v_muN, V_crossing_muSC_4t=v_muSC,
                rows=rows, d2eSC_dV2=d2_SC.tolist(), d2eN_dV2=d2_N.tolist())
            print(f"  {tag:12s}: V(muN=4t)={v_muN:.5f}, V(muSC=4t)={v_muSC:.5f}", flush=True)

    # Focused Nk=801 check around both crossings
    for periodic in (False, True):
        tag = f"Nk801_{'per' if periodic else 'repo'}"
        Vs_801 = np.round(np.arange(0.52, 0.721, 0.01), 4)
        rows = []; prev = None
        for V in Vs_801:
            b, _ = solve_single(1.875, 75.0, float(V), 801, prev=prev, periodic=periodic)
            prev = b['x']
            r = {k: v for k, v in b.items() if k not in ('e', 'x')}
            r['V'] = float(V); rows.append(r)
        vA = [r['V'] for r in rows]
        v_muN = crossing(vA, [r['mu_N'] for r in rows], 4.0)
        v_muSC = crossing(vA, [r['mu_SC'] for r in rows], 4.0)
        out['task6_V_sweep'][tag] = dict(Nk=801, periodic=periodic,
                                         V_Lifshitz_muN_4t=v_muN,
                                         V_crossing_muSC_4t=v_muSC, rows=rows)
        print(f"  {tag:12s}: V(muN=4t)={v_muN:.5f}, V(muSC=4t)={v_muSC:.5f}", flush=True)

    print("=== 2. Task 6 U sweeps at n=1.875 (V=0.5 -> U=68.55; V=0.7 -> U=118.04) ===", flush=True)
    Us = np.round(np.arange(40.0, 145.1, 2.5), 3)
    for V in (0.5, 0.7):
        for Nk in (201, 401):
            for periodic in (False, True):
                if Nk == 401 and periodic:
                    pass
                tag = f"V{V}_Nk{Nk}_{'per' if periodic else 'repo'}"
                rows = []; prev = None
                for U in Us:
                    b, _ = solve_single(1.875, float(U), V, Nk, prev=prev, periodic=periodic)
                    prev = b['x']
                    r = {k: v for k, v in b.items() if k not in ('e', 'x')}
                    r['U'] = float(U); rows.append(r)
                u_muSC = crossing([r['U'] for r in rows], [r['mu_SC'] for r in rows], 4.0)
                out['task6_U_sweeps'][tag] = dict(V=V, Nk=Nk, periodic=periodic,
                                                  muN=rows[0]['mu_N'],
                                                  U_crossing_muSC_4t=u_muSC, rows=rows)
                print(f"  {tag:16s}: muN={rows[0]['mu_N']:.5f} (const!) U(muSC=4t)={u_muSC}", flush=True)

    print("=== 3. Task 7 Convergence benchmarks (Nk=101,201,401,801; repo vs per; tol) ===", flush=True)
    bench_pts = [
        ('n1.875_V0.5', 1.875, 75.0, 0.5),
        ('n1.875_V0.7', 1.875, 75.0, 0.7),
        ('n1.0_V1.235', 1.0, 75.0, 1.235),
        ('n1.4_V0.525', 1.4, 75.0, 0.525),
        ('n1.05_V0.5',  1.05, 75.0, 0.5),
    ]
    conv = {}
    for pname, n, U, V in bench_pts:
        conv[pname] = {}
        for periodic in (False, True):
            for Nk in (101, 201, 401, 801):
                sk, ssin, eps, KX, KY, _ = fs.make_grid(Nk, 1.0, V, periodic=periodic)
                b, _ = solve_single(n, U, V, Nk, periodic=periodic)
                bdg = fs.bdg_dict(b['x'], eps, KX, KY, ssin)
                d_s = hdm.density_at_separation(bdg, np.array([0.25, 0.0]), basis='spin')
                C_s = hdm.concurrence(d_s['rho'])
                key = f"{'per' if periodic else 'repo'}_{Nk}"
                conv[pname][key] = dict(
                    D0=b['D0'], DS=b['DS'], mu_SC=b['mu_SC'], mu_N=b['mu_N'],
                    e_SC=b['e_SC'], dF=b['dF'], res=b['res'], dens_res=b['dens_res'],
                    C_spin_r025=float(C_s))
        p = conv[pname]
        print(f"  {pname:12s}: repo_801 D0={p['repo_801']['D0']:.6f} vs per_801={p['per_801']['D0']:.6f} | "
              f"dF diff(401-801 repo)={abs(p['repo_401']['dF']-p['repo_801']['dF']):.2e} "
              f"(per)={abs(p['per_401']['dF']-p['per_801']['dF']):.2e}", flush=True)

    # Tolerance sensitivity at (n=1.05, V=0.5, Nk=401) and (n=1.4, V=0.525, Nk=401)
    tol_check = {}
    for pname, n, U, V in [('n1.05_V0.5', 1.05, 75.0, 0.5), ('n1.4_V0.525', 1.4, 75.0, 0.525)]:
        tol_check[pname] = {}
        for xtol in (1e-8, 1e-11, 1e-14):
            b, _ = solve_single(n, U, V, 401, xtol=xtol)
            tol_check[pname][f"{xtol:.0e}"] = dict(
                D0=b['D0'], DS=b['DS'], mu_SC=b['mu_SC'], dF=b['dF'],
                res=b['res'], dens_res=b['dens_res'])
    out['task7_convergence'] = dict(benchmarks=conv, tolerance=tol_check)

    json.dump(out, open(os.path.join(HERE, 'data/task6_7_audit.json'), 'w'),
              indent=1, default=float)
    print('DONE task6_7_audit in', round(time.time()-t0, 1), 's')
