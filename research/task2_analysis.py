"""TASK 2 -- Quantitative critical-exponent, goodness-of-fit, window
sensitivity, essential-singularity vs power-law model comparison, and
targeted Nk=801 (endpoint=False & endpoint=True) critical scaling."""
import sys, json, time, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np
from scipy.optimize import curve_fit
import fast_solver as fs


def targeted_high_res_scaling():
    """Run Nk=801 (periodic=True & False) across the critical window at
    V_SO=0.50 (n in [1.00, 1.15]) plus Nk=401 periodic=True for V=0.45, 0.50, 0.55."""
    out = {}
    configs = [
        ('per_Nk201_V0.50', 201, True,  0.50, np.arange(0.990, 1.151, 0.005)),
        ('per_Nk401_V0.50', 401, True,  0.50, np.arange(0.995, 1.151, 0.005)),
        ('per_Nk801_V0.50', 801, True,  0.50, np.arange(1.005, 1.151, 0.005)),
        ('repo_Nk801_V0.50', 801, False, 0.50, np.arange(1.005, 1.151, 0.005)),
        ('per_Nk401_V0.45', 401, True,  0.45, np.arange(1.035, 1.161, 0.005)),
        ('per_Nk401_V0.55', 401, True,  0.55, np.arange(0.955, 1.121, 0.005)),
    ]
    for tag, Nk, periodic, V, ns in configs:
        sk, ssin, eps, _, _, _ = fs.make_grid(Nk, 1.0, V, periodic=periodic)
        kern = fs.kernel_array(sk, ssin, 75.0, 4.5, V)
        rows = []; prev = None
        for n in ns:
            n = float(round(n, 5))
            nrm = fs.normal_state(eps, n)
            lam_c, lam_g, _ = fs.lambda_exact_and_matrix(eps, kern, Nk, 4.5, nrm['mu'], 75.0)
            seeds = []
            if prev is not None:
                seeds.append(prev)
            seeds += [np.array(s, float) for s in [
                [1e-6, 5e-6, nrm['mu']], [1e-4, 5e-4, nrm['mu']],
                [0.01, 0.04, nrm['mu']], [0.05, 0.2, nrm['mu']],
                [0.1, 0.6, nrm['mu']], [0.1, -0.6, nrm['mu']]]]
            best = None
            if lam_c > 0.99 or lam_g > 0.99:
                for s in seeds:
                    cand, r, ier = fs.solve_one(eps, kern, Nk, 4.5, n, s,
                                                xtol=1.49e-12, use_jac=True)
                    if cand is not None and r < 1e-10 and max(abs(cand[0]), abs(cand[1])) > 1e-8:
                        cand = fs._canon(cand)
                        e, _, dr = fs.sc_energy(cand, eps, Nk, n)
                        if best is None or e < best['e']:
                            best = dict(x=cand, e=e, r=r, dr=dr)
                        if r < 1e-11:
                            break
            if best is not None:
                x = best['x']; prev = x
                D0, DS, mu = float(x[0]), float(x[1]), float(x[2])
                D_FS = float(D0 - (DS/4.0)*nrm['mu'])
                dF = float(best['e'] - nrm['e'])
            else:
                prev = None; D0 = DS = D_FS = dF = 0.0; mu = float(nrm['mu'])
            rows.append(dict(n=n, lam_cont=lam_c, lam_grid=lam_g, muN=float(nrm['mu']),
                             D0=D0, DS=DS, D_FS=D_FS, mu=mu, dF=dF))
        out[tag] = dict(Nk=Nk, periodic=periodic, V=V, rows=rows)
        print(f"  {tag:18s}: {len(rows)} pts, SC={sum(1 for r in rows if abs(r['DS'])>0)}", flush=True)
    return out


def analyze_fits(series_dict):
    """Run power-law fits |Delta| = A (n - n_c)^beta and essential-singularity
    fits ln|Delta| = ln(Lambda) - C / (n - n_c) over sliding windows."""
    results = {}
    for tag, d in series_dict.items():
        rows = d['rows']
        sc = [r for r in rows if abs(r['DS']) > 1e-7]
        if len(sc) < 6:
            continue
        ns = np.array([r['n'] for r in sc])
        yDS = np.array([abs(r['DS']) for r in sc])
        yDFS = np.array([abs(r['D_FS']) for r in sc])
        lam = np.array([r['lam_cont'] for r in sc])

        # Exact continuum n_c from linear/cubic interpolation of lam_cont == 1
        all_n = np.array([r['n'] for r in rows])
        all_lam = np.array([r['lam_cont'] for r in rows])
        nc_lam = None
        for i in range(len(all_n) - 1):
            if (all_lam[i] - 1.0) * (all_lam[i+1] - 1.0) <= 0.0 and all_lam[i+1] != all_lam[i]:
                w = (1.0 - all_lam[i]) / (all_lam[i+1] - all_lam[i])
                nc_lam = float(all_n[i] + w*(all_n[i+1] - all_n[i]))
                break

        # Window sensitivity: drop near-onset or vary upper cutoff n_max
        win_fits = []
        for n_max in [ns.min() + w for w in (0.04, 0.06, 0.08, 0.10, 0.13)]:
            m = (ns <= n_max + 1e-9) & (yDS > 1e-5)
            if np.sum(m) < 5:
                continue
            x = ns[m]; y = yDS[m]; yf = yDFS[m]
            # 1) Free (A, nc, beta) power law on |Delta_S|
            try:
                p0 = [2.0, max(x[0] - 0.005, 0.8), 0.5]
                bnds = ([0, 0.5, 0.05], [100, x[0] - 1e-5, 5.0])
                popt, pcov = curve_fit(lambda t, A, nc, b: A*np.power(t - nc, b),
                                       x, y, p0=p0, bounds=bnds, maxfev=20000)
                perr = np.sqrt(np.diag(pcov))
                resid = y - popt[0]*np.power(x - popt[1], popt[2])
                rmse_pow = float(np.sqrt(np.mean(resid**2)))
                rss = float(np.sum(resid**2))
                aic_pow = float(len(x)*np.log(max(rss/len(x), 1e-30)) + 2*3)
                pow_free = dict(A=float(popt[0]), nc=float(popt[1]), nc_err=float(perr[1]),
                                beta=float(popt[2]), beta_err=float(perr[2]),
                                rmse=rmse_pow, aic=aic_pow)
            except Exception as ex:
                pow_free = dict(error=str(ex))

            # 2) Fixed nc=nc_lam power law
            pow_fixed_nc = {}
            if nc_lam is not None and np.all(x > nc_lam):
                try:
                    popt2, pcov2 = curve_fit(lambda t, A, b: A*np.power(t - nc_lam, b),
                                             x, y, p0=[2.0, 0.5], maxfev=10000)
                    perr2 = np.sqrt(np.diag(pcov2))
                    res2 = y - popt2[0]*np.power(x - nc_lam, popt2[1])
                    pow_fixed_nc = dict(nc=nc_lam, A=float(popt2[0]),
                                        beta=float(popt2[1]), beta_err=float(perr2[1]),
                                        rmse=float(np.sqrt(np.mean(res2**2))))
                except Exception as ex:
                    pow_fixed_nc = dict(error=str(ex))

            # 3) Essential-singularity form on |Delta_FS|: Lambda * exp(-C/(n - nc))
            ess = {}
            if nc_lam is not None and np.all(x > nc_lam):
                try:
                    lny = np.log(yf)
                    inv_dn = 1.0 / (x - nc_lam)
                    # linear regression on log(|Delta_FS|) vs 1/(n - nc_lam) plus quadratic check
                    slope, intercept = np.polyfit(inv_dn, lny, 1)
                    pred_y = np.exp(intercept + slope*inv_dn)
                    res_e = yf - pred_y
                    rss_e = float(np.sum(res_e**2))
                    aic_ess = float(len(x)*np.log(max(rss_e/len(x), 1e-30)) + 2*2)
                    # also log-space R^2
                    ss_tot = np.sum((lny - np.mean(lny))**2)
                    r2_log = float(1.0 - np.sum((lny - (intercept + slope*inv_dn))**2)/max(ss_tot, 1e-30))
                    ess = dict(nc=nc_lam, C_ess=float(-slope), ln_Lambda=float(intercept),
                               r2_log=r2_log, rmse_DFS=float(np.sqrt(np.mean(res_e**2))),
                               aic=aic_ess)
                except Exception as ex:
                    ess = dict(error=str(ex))

            # 4) |Delta_S|^2 vs (lam_cont - 1) linear check (does |Delta|^2 scale linearly?)
            ds2_lin = {}
            mlam = m & (lam > 1.0)
            if np.sum(mlam) >= 4:
                dl = lam[mlam] - 1.0
                y2 = yDS[mlam]**2
                sl, ic = np.polyfit(dl, y2, 1)
                r2_lin = float(1.0 - np.sum((y2 - (sl*dl + ic))**2)/max(np.sum((y2-np.mean(y2))**2), 1e-30))
                ds2_lin = dict(slope=float(sl), intercept=float(ic), r2=r2_lin)

            win_fits.append(dict(n_max=float(round(n_max, 4)), n_pts=int(np.sum(m)),
                                 pow_free=pow_free, pow_fixed_nc=pow_fixed_nc,
                                 essential_DFS=ess, DS2_vs_lam_minus_1=ds2_lin))

        results[tag] = dict(Nk=d['Nk'], periodic=d.get('periodic', False),
                            V=d['V'], nc_lam_cont=nc_lam, win_fits=win_fits)
    return results


if __name__ == '__main__':
    t0 = time.time()
    print("=== 1. Targeted high-res Nk=801 + periodic=True scaling ===", flush=True)
    tcache = os.path.join(HERE, 'data/task2_targeted.json')
    if os.path.exists(tcache):
        targeted = json.load(open(tcache))
    else:
        targeted = targeted_high_res_scaling()
        json.dump(targeted, open(tcache, 'w'), indent=1, default=float)

    # Load repo-grid scaling.json too if available
    all_series = dict(targeted)
    sc_path = os.path.join(HERE, 'data/scaling.json')
    if os.path.exists(sc_path):
        sc_raw = json.load(open(sc_path))
        for k, v in sc_raw.items():
            rows = []
            for r in v['rows']:
                br = r.get('branches', [])
                if br:
                    b0 = min(br, key=lambda b: b['dF'])
                    D0, DS, mu, dF = b0['D0'], b0['DS'], b0['mu'], b0['dF']
                    D_FS = b0.get('D_FS', D0 - (DS/4.0)*r['muN'])
                else:
                    D0 = DS = D_FS = dF = 0.0; mu = r['muN']
                rows.append(dict(n=r['n'], lam_cont=r['lam'], lam_grid=r['lam'],
                                 muN=r['muN'], D0=D0, DS=DS,
                                 D_FS=D_FS, mu=mu, dF=dF))
            all_series[f"repo_{k}"] = dict(Nk=v['Nk'], periodic=False, V=v['V'], rows=rows)

    print("=== 2. Window sensitivity & model comparison fits ===", flush=True)
    fits = analyze_fits(all_series)
    for tag, f in sorted(fits.items()):
        print(f"\n[{tag}] nc_lam_cont={f['nc_lam_cont']}", flush=True)
        for wf in f['win_fits']:
            pf = wf['pow_free']
            es = wf['essential_DFS']
            pf_str = (f"nc={pf['nc']:.5f}+-{pf['nc_err']:.5f} beta={pf['beta']:.3f}+-{pf['beta_err']:.3f} rmse={pf['rmse']:.2e}"
                      if 'beta' in pf else pf.get('error', ''))
            es_str = (f"R2log={es['r2_log']:.4f} C_ess={es['C_ess']:.4f} rmseDFS={es['rmse_DFS']:.2e}"
                      if 'r2_log' in es else '')
            print(f"  n_max={wf['n_max']:.3f} ({wf['n_pts']:2d}): POW[{pf_str}] | ESS[{es_str}]", flush=True)

    json.dump(dict(targeted=targeted, fits=fits),
              open(os.path.join(HERE, 'data/task2_analysis.json'), 'w'),
              indent=1, default=float)
    print('DONE task2_analysis in', round(time.time()-t0, 1), 's')
