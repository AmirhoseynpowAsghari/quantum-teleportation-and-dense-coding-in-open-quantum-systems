"""Export clean, reproducible CSV datasets for all Tasks 1-7 into research/data/csv/."""
import os, json, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
CSV_DIR = os.path.join(DATA, 'csv')
os.makedirs(CSV_DIR, exist_ok=True)


def write_csv(fname, fieldnames, rows):
    path = os.path.join(CSV_DIR, fname)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})
    print(f"  wrote {fname} ({len(rows)} rows)")


# 1. Task 1 adaptive sweeps & hysteresis
t1d = json.load(open(os.path.join(DATA, 'task1_deep.json')))
t1_rows = []
for name, sw in t1d['adaptive'].items():
    up = sw['up']; dn = sw['down']
    v_arr = np.array([r['V'] for r in up])
    df_arr = np.array([r['branches'][0]['dF'] if r['branches'] else 0.0 for r in up])
    d2f = np.gradient(np.gradient(df_arr, v_arr, edge_order=2), v_arr, edge_order=2)
    for i, (ru, rd) in enumerate(zip(up, dn)):
        bu = ru['branches'][0] if ru['branches'] else {}
        bd = rd['branches'][0] if rd['branches'] else {}
        t1_rows.append(dict(
            sweep=name, n=sw['n'], Nk=sw['Nk'], dV=sw['dV'],
            V=ru['V'], lam_cont=ru['lam_cont'], nb_up=ru['nb'], nb_dn=rd['nb'],
            D0_up=bu.get('D0', 0.0), DS_up=bu.get('DS', 0.0),
            mu_up=bu.get('mu', ru['muN']), dF_up=bu.get('dF', 0.0),
            D0_dn=bd.get('D0', 0.0), DS_dn=bd.get('DS', 0.0),
            mu_dn=bd.get('mu', rd['muN']), dF_dn=bd.get('dF', 0.0),
            diff_dF=abs(bu.get('dF', 0.0) - bd.get('dF', 0.0)),
            diff_D0=abs(bu.get('D0', 0.0) - bd.get('D0', 0.0)),
            dF_second_deriv=float(d2f[i])))
write_csv('task1_adaptive_hysteresis.csv', list(t1_rows[0].keys()), t1_rows)

# Compressibility
comp_rows = []
for name, c in t1d['compressibility'].items():
    rows_c = c['rows']
    ns_c = np.array([r['n'] for r in rows_c])
    mus_c = np.array([r['branches'][0]['mu'] if r['branches'] else r['muN'] for r in rows_c])
    es_c = np.array([r['branches'][0]['e'] if r['branches'] else r['eN'] for r in rows_c])
    dmu = np.gradient(mus_c, ns_c, edge_order=2)
    dedn = np.gradient(es_c, ns_c, edge_order=2)
    d2e = np.gradient(dedn, ns_c, edge_order=2)
    for i, r in enumerate(rows_c):
        b0 = r['branches'][0] if r['branches'] else {}
        comp_rows.append(dict(
            sweep=name, V=c['V'], Nk=c['Nk'], n=r['n'],
            D0=b0.get('D0', 0.0), DS=b0.get('DS', 0.0), mu=float(mus_c[i]),
            e_GS=float(es_c[i]), dF=b0.get('dF', 0.0),
            dmu_dn=float(dmu[i]), d2e_dn2=float(d2e[i]),
            de_dn_minus_mu=float(dedn[i] - mus_c[i])))
write_csv('task1_compressibility.csv', list(comp_rows[0].keys()), comp_rows)

# 2. Task 2 critical scaling (targeted + fits)
t2 = json.load(open(os.path.join(DATA, 'task2_analysis.json')))
t2_rows = []
for tag, d in t2['targeted'].items():
    for r in d['rows']:
        t2_rows.append(dict(
            series=tag, Nk=d['Nk'], periodic=d['periodic'], V=d['V'],
            n=r['n'], lam_cont=r['lam_cont'], lam_grid=r['lam_grid'],
            muN=r['muN'], D0=r['D0'], DS=r['DS'], D_FS=r['D_FS'],
            mu=r['mu'], dF=r['dF']))
write_csv('task2_critical_scaling.csv', list(t2_rows[0].keys()), t2_rows)

t2_fit_rows = []
for tag, f in t2['fits'].items():
    for wf in f['win_fits']:
        pf = wf['pow_free']; pf_fix = wf['pow_fixed_nc']; es = wf['essential_DFS']
        t2_fit_rows.append(dict(
            series=tag, Nk=f['Nk'], periodic=f['periodic'], V=f['V'],
            nc_lam_cont=f['nc_lam_cont'], n_max=wf['n_max'], n_pts=wf['n_pts'],
            pow_nc=pf.get('nc', ''), pow_beta=pf.get('beta', ''),
            pow_beta_err=pf.get('beta_err', ''), pow_rmse=pf.get('rmse', ''),
            pow_aic=pf.get('aic', ''),
            fixed_beta=pf_fix.get('beta', ''), fixed_rmse=pf_fix.get('rmse', ''),
            ess_r2_log=es.get('r2_log', ''), ess_C=es.get('C_ess', ''),
            ess_rmse_DFS=es.get('rmse_DFS', ''), ess_aic=es.get('aic', '')))
write_csv('task2_window_fits.csv', list(t2_fit_rows[0].keys()), t2_fit_rows)

# 3. Task 3 phase diagram points (Nk=201 and Nk=401_per) + continuum maps
pd_rows = []
for fn, tag in [('phasediagram_Nk201.json', 'Nk201_repo'), ('phasediagram_Nk401_per.json', 'Nk401_per')]:
    pd = json.load(open(os.path.join(DATA, fn)))
    for p in pd['points']:
        b0 = p['branches'][0] if p['branches'] else {}
        pd_rows.append(dict(
            grid=tag, Nk=pd['Nk'], n=p['n'], V=p['V'],
            lam_grid=p['lam'], lam_cont=p.get('lam_cont', p['lam']),
            muN=p['muN'], nb=p['nb'],
            D0=b0.get('D0', 0.0), DS=b0.get('DS', 0.0),
            mu_SC=b0.get('mu', p['muN']), dF=b0.get('dF', 0.0)))
write_csv('task3_phase_diagram_points.csv', list(pd_rows[0].keys()), pd_rows)

# Task 3&4 continuum maps & Landau curves
t34 = json.load(open(os.path.join(DATA, 'task3_4_slices.json')))
m = t34['maps']
map_rows = []
for i, n in enumerate(m['ns']):
    for j, V in enumerate(m['Vs']):
        map_rows.append(dict(
            n=n, V=V, muN=m['muN'][i][j], J2=m['J2'][i][j],
            area_inner=m['area_inner'][i][j], Uc=m['Uc_nV'][i][j],
            lam_U40=m['slices_nV']['40'][i][j],
            lam_U75=m['slices_nV']['75'][i][j],
            lam_U100=m['slices_nV']['100'][i][j],
            lam_U130=m['slices_nV']['130'][i][j]))
write_csv('task3_continuum_nV_maps.csv', list(map_rows[0].keys()), map_rows)

lan_rows = []
for L in t34['landau']:
    for pt in L['pts']:
        lan_rows.append(dict(
            n=L['n'], V=L['V'], U=L['U'], lam_cont=L['lam_cont'],
            muN=L['muN'], psi=pt['psi'], D0=pt['D0'], mu=pt['mu'],
            dF=pt['dF'], dF_over_psi2=pt['dF_over_psi2']))
write_csv('task4_landau_potential.csv', list(lan_rows[0].keys()), lan_rows)

# 4. Task 5 concurrence cuts
t5 = json.load(open(os.path.join(DATA, 'task5_concurrence.json')))
c_rows = []
for cname, c in t5.items():
    for i, r in enumerate(c['rows']):
        for basis in ('spin', 'band'):
            for rk in ('0.25_0', '0.5_0', '1_0'):
                if rk not in r['sc_metrics'][basis]:
                    continue
                sm = r['sc_metrics'][basis][rk]; nm = r['norm_metrics'][basis][rk]
                c_rows.append(dict(
                    cut=cname, Nk=c['Nk'], basis=basis, r_sep=rk,
                    param=r['param'], n=r['n'], U=r['U'], V=r['V'],
                    lam_cont=r['lam_cont'], sc=r['sc'], D0=r['D0'], DS=r['DS'],
                    dF=r['dF'], C=sm['C'], C_raw=sm['C_raw'],
                    C_normonly=sm['C_normonly'], C_pure_normal=nm['C'],
                    Fs=sm['Fs'], Fnorm=sm['Fnorm'], Nnorm=sm['Nnorm'],
                    dC=c['derivs'][basis][rk]['dC'][i],
                    d2C=c['derivs'][basis][rk]['d2C'][i]))
write_csv('task5_concurrence_cuts.csv', list(c_rows[0].keys()), c_rows)

# 5. Task 6 & 7 audit
t67 = json.load(open(os.path.join(DATA, 'task6_7_audit.json')))
t6v_rows = []
for tag, d in t67['task6_V_sweep'].items():
    for i, r in enumerate(d['rows']):
        t6v_rows.append(dict(
            grid=tag, Nk=d['Nk'], periodic=d['periodic'], V=r['V'],
            D0=r['D0'], DS=r['DS'], mu_SC=r['mu_SC'], mu_N=r['mu_N'],
            e_SC=r['e_SC'], e_N=r['e_N'], dF=r['dF'],
            min_E_band0=r['min_E_band0'], E_pipi_band0=r['E_pipi_band0'],
            occ_pipi_SC=r['occ_pipi_SC'],
            area_N_eps0_gt_muN=r['area_N_eps0_gt_muN'],
            area_mixed_eps0_gt_muSC=r['area_mixed_eps0_gt_muSC'],
            d2eSC_dV2=d['d2eSC_dV2'][i] if 'd2eSC_dV2' in d else '',
            d2eN_dV2=d['d2eN_dV2'][i] if 'd2eN_dV2' in d else ''))
write_csv('task6_n1875_Vsweep.csv', list(t6v_rows[0].keys()), t6v_rows)

t6u_rows = []
for tag, d in t67['task6_U_sweeps'].items():
    for r in d['rows']:
        t6u_rows.append(dict(
            grid=tag, V=d['V'], Nk=d['Nk'], periodic=d['periodic'], U=r['U'],
            D0=r['D0'], DS=r['DS'], mu_SC=r['mu_SC'], mu_N=r['mu_N'],
            e_SC=r['e_SC'], dF=r['dF'], E_pipi_band0=r['E_pipi_band0'],
            area_N_eps0_gt_muN=r['area_N_eps0_gt_muN']))
write_csv('task6_n1875_Usweep.csv', list(t6u_rows[0].keys()), t6u_rows)

t7_rows = []
for pname, bd in t67['task7_convergence']['benchmarks'].items():
    for gkey, vals in bd.items():
        t7_rows.append(dict(point=pname, grid_Nk=gkey, **vals))
write_csv('task7_convergence_benchmarks.csv', list(t7_rows[0].keys()), t7_rows)
print("ALL CSVs EXPORTED.")
