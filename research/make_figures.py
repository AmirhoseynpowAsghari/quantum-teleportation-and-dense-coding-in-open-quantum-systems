"""TASK 8 -- Generate all 12 publication-quality figure sets in PDF, SVG, and
300-dpi PNG inside research/figs/ from verified CSV/JSON datasets."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, 'data')
FIGS = os.path.join(HERE, 'figs')
sys.path.insert(0, HERE)
import fast_solver as fs

os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 9.5,
    'axes.labelsize': 10,
    'axes.titlesize': 10.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.0,
    'axes.linewidth': 0.9,
    'lines.linewidth': 1.6,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.06,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
})

# Colorblind-safe Okabe-Ito palette
C_BLUE = '#0072B2'
C_VERM = '#D55E00'
C_TEAL = '#009E73'
C_PURP = '#CC79A7'
C_AMBR = '#E69F00'
C_SKY  = '#56B4E9'
C_BLK  = '#222222'
C_GRAY = '#777777'


def save_all(fig, stem):
    for ext in ('png', 'pdf', 'svg'):
        kw = {'dpi': 300} if ext == 'png' else {}
        fig.savefig(os.path.join(FIGS, f"{stem}.{ext}"), **kw)
    plt.close(fig)
    print(f"  saved {stem}.{{png,pdf,svg}}")


# ---------------------------------------------------------------------------
# FIG 01: Task 1 Candidate First-Order Falsification (n=1.0 & n=1.4)
# ---------------------------------------------------------------------------
def make_fig01():
    t1d = json.load(open(os.path.join(DATA, 'task1_deep.json')))
    t1c = json.load(open(os.path.join(DATA, 'task1_candidates.json')))
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6), constrained_layout=True)

    # Row 0: n=1.0 around V=1.235 | Row 1: n=1.4 around V=0.525
    configs = [
        (0, 'n1.0_dV0.0001_Nk401', 'n1.0_Nk401', 1.235, r'$n = 1.0$, candidate $V_{\mathrm{SO}}/t = 1.235$'),
        (1, 'n1.4_dV0.0001_Nk401', 'n1.4_Nk401', 0.525, r'$n = 1.4$, candidate $V_{\mathrm{SO}}/t = 0.525$'),
    ]
    for row, dk, ck, Vc, label in configs:
        sw = t1d['adaptive'][dk]
        up = sw['up']; dn = sw['down']
        v = np.array([r['V'] for r in up])
        d0u = np.array([r['branches'][0]['D0'] for r in up])
        d0d = np.array([r['branches'][0]['D0'] for r in dn])
        dsu = np.array([r['branches'][0]['DS'] for r in up])
        dsd = np.array([r['branches'][0]['DS'] for r in dn])
        dfu = np.array([r['branches'][0]['dF'] for r in up])
        dfd = np.array([r['branches'][0]['dF'] for r in dn])

        # (a)/(d): Gaps up vs down + multi-seed census
        ax = axes[row, 0]
        ax.plot(v, d0u, color=C_BLUE, lw=2.0, label=r'$\Delta_0$ (up)')
        ax.plot(v, d0d, '--', color=C_SKY, lw=1.6, label=r'$\Delta_0$ (down)')
        ax.plot(v, np.abs(dsu), color=C_VERM, lw=2.0, label=r'$|\Delta_S|$ (up)')
        ax.plot(v, np.abs(dsd), '--', color=C_AMBR, lw=1.6, label=r'$|\Delta_S|$ (down)')
        # overlay 18-seed census
        cen = t1c[ck]['census']
        cv = [p['V'] for p in cen]
        cd0 = [p['branches'][0]['D0'] for p in cen]
        cds = [abs(p['branches'][0]['DS']) for p in cen]
        ax.scatter(cv, cd0, s=18, color=C_BLK, zorder=5, label='18-seed census ($N_b=1$)')
        ax.scatter(cv, cds, s=18, color=C_BLK, zorder=5)
        ax.axvline(Vc, color=C_GRAY, ls=':', lw=1.2)
        ax.set_xlabel(r'$V_{\mathrm{SO}}/t$')
        ax.set_ylabel(r'Gap components ($t$)')
        ax.set_title(f"({'ad'[row]}) {label}")
        ax.grid(alpha=0.25)
        ax.legend(framealpha=0.9, fontsize=7.5, ncol=2)

        # (b)/(e): Condensation energy dF = F_SC - F_N and hysteresis
        ax = axes[row, 1]
        ax.plot(v, dfu, color=C_TEAL, lw=2.2, label=r'$\Delta F = F_{\mathrm{SC}} - F_N$ (up)')
        ax.plot(v, dfd, '--', color=C_BLK, lw=1.3, label=r'$\Delta F$ (down)')
        ax.axvline(Vc, color=C_GRAY, ls=':', lw=1.2)
        ax2 = ax.twinx()
        hyst = np.maximum(np.abs(dfu - dfd), 1e-16)
        ax2.semilogy(v, hyst, ':', color=C_PURP, lw=1.4, label=r'$|\Delta F_\uparrow-\Delta F_\downarrow|$')
        ax2.set_ylim(1e-17, 1e-10)
        ax2.set_ylabel(r'Hysteresis $|\Delta F_\uparrow - \Delta F_\downarrow|$ ($t$)', color=C_PURP)
        ax2.tick_params(axis='y', colors=C_PURP)
        ax.set_xlabel(r'$V_{\mathrm{SO}}/t$')
        ax.set_ylabel(r'$\Delta F = F_{\mathrm{SC}} - F_N$ ($t$)')
        ax.set_title(f"({'be'[row]}) Smooth $\\Delta F$ & $\\leq 8.9\\times 10^{{-16}}$ hysteresis ($\\Delta V=10^{{-4}}$)")
        ax.grid(alpha=0.25)
        ax.legend(loc='lower left', framealpha=0.9, fontsize=7.5)

    # Panel (c): wide-V sweep [0, 2.5] showing smooth dF and d2F/dV2 at n=1.0 and n=1.4
    ax = axes[0, 2]
    for nk_key, lbl, col in [('n1.0_Nk401', r'$n=1.0$ ($V_c \approx 0.53t$)', C_BLUE),
                              ('n1.4_Nk401', r'$n=1.4$ (all $V_{\mathrm{SO}}$)', C_VERM)]:
        rows = t1d['wide_V'][nk_key]
        vw = np.array([r['V'] for r in rows])
        dfw = np.array([r['branches'][0]['dF'] if r['branches'] else 0.0 for r in rows])
        ax.plot(vw, dfw, '-o', ms=2.8, color=col, label=lbl)
    ax.axvline(1.235, color=C_BLUE, ls=':', lw=1.1, alpha=0.8)
    ax.axvline(0.525, color=C_VERM, ls=':', lw=1.1, alpha=0.8)
    ax.set_xlabel(r'$V_{\mathrm{SO}}/t$')
    ax.set_ylabel(r'$\Delta F = F_{\mathrm{SC}} - F_N$ ($t$)')
    ax.set_title(r'(c) Global $V_{\mathrm{SO}}/t \in [0, 2.5]$ ($N_k=401$)')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.8)

    # Panel (f): Compressibility dmu/dn > 0 & d2e/dn2 > 0 (no phase separation)
    ax = axes[1, 2]
    for ck, lbl, col in [('V0.0_Nk201', r'$V_{\mathrm{SO}}=0$', C_GRAY),
                         ('V0.5_Nk401', r'$V_{\mathrm{SO}}=0.50t$', C_BLUE),
                         ('V1.0_Nk401', r'$V_{\mathrm{SO}}=1.00t$', C_TEAL),
                         ('V1.4_Nk201', r'$V_{\mathrm{SO}}=1.40t$', C_VERM)]:
        c = t1d['compressibility'][ck]
        rows = c['rows']
        ns = np.array([r['n'] for r in rows])
        mus = np.array([r['branches'][0]['mu'] if r['branches'] else r['muN'] for r in rows])
        dmu = np.gradient(mus, ns, edge_order=2)
        ax.plot(ns, dmu, color=col, label=lbl)
    ax.axhline(0, color=C_BLK, lw=0.8, ls='--')
    ax.set_xlabel(r'Electron filling $n$')
    ax.set_ylabel(r'Inverse compressibility $\partial\mu/\partial n$ ($t$)')
    ax.set_title(r'(f) Thermodynamic stability ($\partial\mu/\partial n \geq 1.21t > 0$)')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.5)

    save_all(fig, 'fig01_task1_candidate_falsification')


# ---------------------------------------------------------------------------
# FIG 02: Forensic Audit of Original Repository Artifacts & Bugs
# ---------------------------------------------------------------------------
def make_fig02():
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)

    # (a) Repo qpt_parameter_scan.json raw up vs down[::-1]
    ax = axes[0]
    qp = json.load(open(os.path.join(ROOT, 'reports/qpt_parameter_scan.json')))
    c0 = qp['v']['raw']['combos'][0]  # U90_Dt4.5_n1.875
    v_up = np.array([p['V'] for p in c0['up']])
    ds_up = np.array([p['x'][1] for p in c0['up']])
    v_dn_rev = np.array([p['V'] for p in c0['down'][::-1]])
    ds_dn_rev = np.array([p['x'][1] for p in c0['down'][::-1]])
    ds_dn_unrev = np.array([p['x'][1] for p in c0['down']])
    ax.plot(v_up, ds_up, 'o-', ms=4.5, color=C_BLUE, label=r'Repo up ($0 \to 1.5$)')
    ax.plot(v_dn_rev, ds_dn_rev, 's--', ms=3.5, color=C_VERM, label=r'Repo down[::-1] ($0 \to 1.5$)')
    ax2 = ax.twinx()
    ax2.semilogy(v_up, np.maximum(np.abs(ds_up - ds_dn_unrev), 1e-16), ':', color=C_GRAY,
                 label=r'$|\Delta_{\mathrm{up}}[i]-\Delta_{\mathrm{down}}[i]|$ (bug)')
    ax2.semilogy(v_up, np.maximum(np.abs(ds_up - ds_dn_rev), 1e-16), '-.', color=C_TEAL,
                 label=r'$|\Delta_{\mathrm{up}}-\Delta_{\mathrm{down[::-1]}}|$ ($\leq 3.6\times 10^{-13}$)')
    ax2.set_ylim(1e-16, 10.0)
    ax2.set_ylabel(r'Sweep difference ($t$)', color=C_TEAL)
    ax2.tick_params(axis='y', colors=C_TEAL)
    ax.set_xlabel(r'$V_{\mathrm{SO}}/t$')
    ax.set_ylabel(r'$\Delta_S / t$')
    ax.set_title(r'(a) Repo qpt_parameter_scan index-reversal bug')
    ax.text(0.04, 0.34, f"Reported hysteresis: {c0['hysteresis_max']:.2f}\n"
                        f"True $|\\Delta_{{up}}-\\Delta_{{down[::-1]}}|$: $3.6\\times 10^{{-13}}$",
            transform=ax.transAxes, fontsize=7.4,
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFF8DC', ec=C_AMBR, alpha=0.92))
    ax.grid(alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, framealpha=0.9, fontsize=6.9, loc='upper left')

    # (b) Finite-grid tiny-gap artifacts at (n=1.0, V=0.52) and (n=1.6, U=140, V=0.5)
    ax = axes[1]
    Nks = [101, 201, 301, 401, 601, 801]
    # Compute live exact check on (n=1.0, V=0.52) for repo vs periodic
    gaps_repo = []; gaps_per = []
    for Nk in Nks:
        for per, arr in [(False, gaps_repo), (True, gaps_per)]:
            sk, ssin, eps, _, _, _ = fs.make_grid(Nk, 1.0, 0.52, periodic=per)
            kern = fs.kernel_array(sk, ssin, 75.0, 4.5, 0.52)
            nrm = fs.normal_state(eps, 1.0)
            _, ntr = fs.solve_all(eps, kern, Nk, 4.5, 1.0,
                                  [np.array([1e-4, 4e-4, nrm['mu']]),
                                   np.array([0.02, 0.08, nrm['mu']])],
                                  tol=1e-12, use_jac=True)
            arr.append(max(abs(s[1]) for s in ntr) if ntr else 0.0)
    ax.plot(Nks, gaps_repo, 'o-', color=C_VERM, label=r'Repo endpoint=True ($n=1, V=0.52$)')
    ax.plot(Nks, gaps_per, 's--', color=C_BLUE, label=r'Periodic endpoint=False ($n=1, V=0.52$)')
    ax.set_xlabel(r'BZ grid size $N_k$')
    ax.set_ylabel(r'Solved $|\Delta_S|/t$ at $(n=1.0, V_{\mathrm{SO}}=0.52t)$')
    ax.set_title(r'(b) Finite-grid SC artifact disappears ($N_k \geq 301$)')
    ax.annotate(r'Continuum $\lambda_+ = 0.9945 < 1$' '\n' r'(true phase = Normal)',
                xy=(401, 0.002), xytext=(320, 0.025),
                arrowprops=dict(arrowstyle='->', color=C_BLK), fontsize=7.8)
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.5)

    # (c) O(1/Nk) bias in endpoint=True vs endpoint=False (Task 7 benchmark n=1.875, V=0.5)
    ax = axes[2]
    t67 = json.load(open(os.path.join(DATA, 'task6_7_audit.json')))
    bm = t67['task7_convergence']['benchmarks']['n1.875_V0.5']
    nk_b = [101, 201, 401]
    err_repo = [abs(bm[f'repo_{k}']['dF'] - bm['per_801']['dF']) for k in nk_b]
    err_per = [abs(bm[f'per_{k}']['dF'] - bm['per_801']['dF']) for k in nk_b]
    ax.loglog(nk_b, err_repo, 'o-', color=C_VERM, label=r'Repo endpoint=True ($\propto N_k^{-1}$)')
    ax.loglog(nk_b, err_per, 's-', color=C_TEAL, label=r'Periodic endpoint=False ($\sim 10^{-8}$ at 401)')
    # reference O(1/Nk) slope
    ref = err_repo[0]*(101.0/np.array(nk_b))
    ax.loglog(nk_b, ref, ':', color=C_GRAY, label=r'$\mathcal{O}(N_k^{-1})$ reference')
    ax.set_xlabel(r'$N_k$')
    ax.set_ylabel(r'$|\Delta F(N_k) - \Delta F_{\mathrm{per}}(801)|$ ($t$)')
    ax.set_title(r'(c) Zone-edge double-counting error ($n=1.875, V=0.5$)')
    ax.grid(alpha=0.25, which='both'); ax.legend(framealpha=0.9, fontsize=7.5)

    save_all(fig, 'fig02_task1_repo_bug_audit')


# ---------------------------------------------------------------------------
# FIG 03: Task 2 Linearized Instability & Exact Continuum Boundary
# ---------------------------------------------------------------------------
def make_fig03():
    t34 = json.load(open(os.path.join(DATA, 'task3_4_slices.json')))
    m = t34['maps']
    ns = np.array(m['ns']); Vs = np.array(m['Vs'])
    lam75 = np.array(m['slices_nV']['75'])
    Uc = np.array(m['Uc_nV'])
    lamconv = json.load(open(os.path.join(DATA, 'lambda_convergence.json')))

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)

    # (a) lambda_+(n) at representative V_SO
    ax = axes[0]
    for V_tgt, col in [(0.0, C_GRAY), (0.3, C_SKY), (0.5, C_BLUE), (0.7, C_TEAL), (0.9, C_VERM)]:
        j = int(np.argmin(np.abs(Vs - V_tgt)))
        col_lam = np.where(lam75[:, j] < 5.0, lam75[:, j], np.nan)
        ax.plot(ns, col_lam, color=col, label=rf'$V_{{\mathrm{{SO}}}}={Vs[j]:.2f}t$')
    ax.axhline(1.0, color=C_BLK, ls='--', lw=1.1, label=r'$\lambda_+ = 1$ onset')
    ax.set_ylim(0.5, 3.0)
    ax.set_xlabel(r'Electron filling $n$')
    ax.set_ylabel(r'Exact continuum eigenvalue $\lambda_+$')
    ax.set_title(r'(a) Instability $\lambda_+(n, V_{\mathrm{SO}})$ at $U/t=75$')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.5)

    # (b) Grid convergence of lambda_max (endpoint=True vs periodic vs exact continuum)
    ax = axes[1]
    Nks = np.array(lamconv['Nks'])
    for key, lbl, col in [('0.50|1.00', r'$n=1.00, V=0.50$', C_BLUE),
                          ('0.50|1.02', r'$n=1.02, V=0.50$', C_VERM),
                          ('0.50|1.05', r'$n=1.05, V=0.50$', C_TEAL)]:
        vals = lamconv['out'][key]['vals']
        # compute exact continuum via 801 periodic
        n_v = lamconv['out'][key]['n']; V_v = lamconv['out'][key]['V']
        _, _, eps, _, _, _ = fs.make_grid(801, 1.0, V_v, periodic=True)
        nrm = fs.normal_state(eps, n_v)
        l_inf = (4.5**2 * float(np.sum(np.abs(eps - nrm['mu']))/(801*801))) / (75.0 - 4*4.5*nrm['mu'])
        ax.plot(1.0/Nks, vals, 'o-', color=col, label=lbl)
        ax.scatter([0.0], [l_inf], marker='*', s=95, color=col, zorder=6)
    ax.axhline(1.0, color=C_BLK, ls='--', lw=1.0)
    ax.set_xlabel(r'$1/N_k$ (repo endpoint=True)')
    ax.set_ylabel(r'$\lambda_{\max}(N_k)$ ($\star$ = continuum)')
    ax.set_title(r'(b) $\mathcal{O}(1/N_k)$ drift shifts $n_c: 1.010 \to 1.0194$')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.5)

    # (c) Exact U_c(n, V_SO) continuous threshold
    ax = axes[2]
    for V_tgt, col in [(0.0, C_GRAY), (0.3, C_SKY), (0.5, C_BLUE), (0.7, C_TEAL), (0.9, C_VERM)]:
        j = int(np.argmin(np.abs(Vs - V_tgt)))
        ax.plot(ns, Uc[:, j], color=col, label=rf'$V_{{\mathrm{{SO}}}}={Vs[j]:.2f}t$')
    ax.axhline(75.0, color=C_BLK, ls='--', lw=1.1, label=r'$U/t=75$')
    ax.set_ylim(20, 175)
    ax.set_xlabel(r'Electron filling $n$')
    ax.set_ylabel(r'Critical repulsion $U_c / t$')
    ax.set_title(r'(c) Exact threshold $U_c = 4\Delta_t \mu_N + \Delta_t^2 \langle|\varepsilon-\mu_N|\rangle$')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.5)

    save_all(fig, 'fig03_task2_lambda_instability')


# ---------------------------------------------------------------------------
# FIG 04: Task 2 Non-Mean-Field Critical Scaling & Window Sensitivity
# ---------------------------------------------------------------------------
def make_fig04():
    t2 = json.load(open(os.path.join(DATA, 'task2_analysis.json')))
    targeted = t2['targeted']; fits = t2['fits']
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)

    # (a) |Delta_FS| and |Delta_S| across Nk=201, 401, 801 (per) at V=0.50
    ax = axes[0]
    for tag, lbl, col, ls in [('per_Nk201_V0.50', r'Per $N_k=201$', C_SKY, ':'),
                              ('per_Nk401_V0.50', r'Per $N_k=401$', C_BLUE, '--'),
                              ('per_Nk801_V0.50', r'Per $N_k=801$', C_BLK, '-'),
                              ('repo_Nk801_V0.50', r'Repo $N_k=801$', C_VERM, '-.')]:
        rows = targeted[tag]['rows']
        ns = np.array([r['n'] for r in rows])
        ds = np.array([abs(r['DS']) for r in rows])
        ax.plot(ns, ds, ls=ls, color=col, label=lbl)
    ax.axvline(1.01944, color=C_TEAL, ls=':', lw=1.2, label=r'$n_c^{\mathrm{cont}}=1.0194$')
    ax.set_xlim(1.00, 1.15)
    ax.set_xlabel(r'Electron filling $n$')
    ax.set_ylabel(r'$|\Delta_S| / t$')
    ax.set_title(r'(a) Order-parameter onset ($V_{\mathrm{SO}}=0.50t$)')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.4)

    # (b) Essential-singularity Arrhenius scaling (lam - 1 - c2*DS^2) vs 1/ln(2/|Delta_FS|)
    ax = axes[1]
    for tag, lbl, col in [('per_Nk401_V0.45', r'$V=0.45t$', C_TEAL),
                          ('per_Nk801_V0.50', r'$V=0.50t$ ($N_k=801$)', C_BLUE),
                          ('per_Nk401_V0.55', r'$V=0.55t$', C_VERM)]:
        rows = [r for r in targeted[tag]['rows'] if r['lam_cont'] > 1.003 and abs(r['D_FS']) > 1e-8]
        dl = np.array([r['lam_cont'] - 1.0 for r in rows])
        dfs = np.array([abs(r['D_FS']) for r in rows])
        ds2 = np.array([r['DS']**2 for r in rows])
        x_invlog = 1.0/np.log(2.0/dfs)
        X = np.column_stack([x_invlog, ds2])
        coef, *_ = np.linalg.lstsq(X, dl, rcond=None)
        dl_corr = dl - coef[1]*ds2
        ax.plot(x_invlog, dl_corr, 'o-', ms=3.5, color=col,
                label=rf'{lbl} ($c_1={coef[0]:.3f}$)')
        xx = np.linspace(0, x_invlog.max()*1.05, 50)
        ax.plot(xx, coef[0]*xx, ':', color=col, alpha=0.85)
    ax.set_xlim(0.0, 0.31); ax.set_ylim(-0.002, 0.028)
    ax.set_xlabel(r'$1 / \ln(2t / |\Delta_{\mathrm{FS}}|)$')
    ax.set_ylabel(r'$(\lambda_+ - 1) - c_2 \Delta_S^2$')
    ax.set_title(r'(b) Exact $c_1/\ln(\Lambda/|\Delta_{\mathrm{FS}}|)$ law through $(0,0)$')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.3)

    # (c) Window sensitivity of effective exponent beta_eff
    ax = axes[2]
    for tag, lbl, col in [('per_Nk401_V0.45', r'$V=0.45t$', C_TEAL),
                          ('per_Nk801_V0.50', r'$V=0.50t$ ($N_k=801$)', C_BLUE),
                          ('per_Nk401_V0.55', r'$V=0.55t$', C_VERM)]:
        wfs = fits[tag]['win_fits']
        nc_l = fits[tag]['nc_lam_cont']
        dn_max = [w['n_max'] - nc_l for w in wfs if 'beta' in w['pow_free']]
        betas = [w['pow_free']['beta'] for w in wfs if 'beta' in w['pow_free']]
        berrs = [w['pow_free']['beta_err'] for w in wfs if 'beta' in w['pow_free']]
        ax.errorbar(dn_max, betas, yerr=berrs, fmt='o-', capsize=3, color=col, label=lbl)
    ax.axhline(0.5, color=C_BLK, ls='--', lw=1.2, label=r'Mean-field $\beta=0.5$')
    ax.set_ylim(0.3, 2.5)
    ax.set_xlabel(r'Fit window width $n_{\max} - n_c$')
    ax.set_ylabel(r'Effective exponent $\beta_{\mathrm{eff}}$')
    ax.set_title(r'(c) Window drift proves non-power-law threshold')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.5)

    save_all(fig, 'fig04_task2_critical_scaling')


# ---------------------------------------------------------------------------
# FIG 05: Task 3 High-Resolution (n, V_SO) Phase Diagram at U/t = 75
# ---------------------------------------------------------------------------
def make_fig05():
    t34 = json.load(open(os.path.join(DATA, 'task3_4_slices.json')))
    pd201 = json.load(open(os.path.join(DATA, 'phasediagram_Nk201.json')))
    pd401 = json.load(open(os.path.join(DATA, 'phasediagram_Nk401_per.json')))
    m = t34['maps']
    ns = np.array(m['ns']); Vs = np.array(m['Vs'])
    muN = np.array(m['muN']); lam75 = np.array(m['slices_nV']['75'])

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), constrained_layout=True)

    # (a) Phase diagram with adaptive points and analytic contours
    ax = axes[0]
    N_grid, V_grid = np.meshgrid(ns, Vs, indexing='ij')
    # Shade regions cleanly using lam75
    ax.contourf(N_grid, V_grid, lam75, levels=[0, 1.0, 100.0],
                colors=['#E8EEF5', '#FCECE3'], alpha=0.75)
    # 2nd-order boundary lambda_+ = 1
    cs_lam = ax.contour(N_grid, V_grid, lam75, levels=[1.0], colors=[C_BLUE], linewidths=2.4)
    # Normal-state Dirac-Lifshitz mu_N = 4t
    ax.contour(N_grid, V_grid, muN, levels=[4.0], colors=[C_TEAL], linewidths=1.9, linestyles='--')
    # Half-filling mu_N = 0
    ax.axvline(1.0, color=C_GRAY, ls='-.', lw=1.1, label=r'$\mu_N = 0$ (half-filling)')
    # Self-consistent adaptive Nk=401 points
    pts4 = pd401['points']
    n_n = [p['n'] for p in pts4 if p['nb'] == 0]
    v_n = [p['V'] for p in pts4 if p['nb'] == 0]
    n_s = [p['n'] for p in pts4 if p['nb'] > 0]
    v_s = [p['V'] for p in pts4 if p['nb'] > 0]
    ax.scatter(n_n, v_n, s=11, c=C_SKY, alpha=0.8, marker='o', label=r'Normal ($N_k=401$, $\Delta=0$)')
    ax.scatter(n_s, v_s, s=11, c=C_VERM, alpha=0.8, marker='o', label=r'Unique SC ($N_k=401$, $\Delta F<0$)')
    # Candidate/refuted markers
    ax.scatter([1.0, 1.4], [1.235, 0.525], s=90, marker='X', c=C_BLK, zorder=8,
               label='Refuted 1st-order candidates')
    ax.scatter([1.875, 1.875], [0.562, 0.694], s=80, marker='*', c=[C_AMBR, C_TEAL],
               edgecolors=C_BLK, linewidths=0.6, zorder=9,
               label=r'$n=1.875$: $\mu_{\mathrm{SC}}=4t$ vs $\mu_N=4t$')
    ax.text(0.68, 0.25, 'NORMAL\n' r'($\Delta=0$, $\lambda_+<1$)', fontsize=9.5,
            fontweight='bold', color=C_BLUE, ha='center')
    ax.text(1.42, 0.95, 'SUPERCONDUCTING\n' r'($N_b=1$, $\Delta F<0$)' '\nNo 1st-order boundary',
            fontsize=9.5, fontweight='bold', color=C_VERM, ha='center')
    # dummy handles for contours
    custom = [
        Line2D([0], [0], color=C_BLUE, lw=2.4, label=r'2nd-order onset ($\lambda_+=1$, ess. sing.)'),
        Line2D([0], [0], color=C_TEAL, lw=1.9, ls='--', label=r'Normal Dirac-Lifshitz ($\mu_N=4t$, crossover in SC)'),
    ]
    h, l = ax.get_legend_handles_labels()
    ax.legend(custom + h, [c.get_label() for c in custom] + l,
              loc='lower left', framealpha=0.92, fontsize=7.3)
    ax.set_xlim(0.50, 1.95); ax.set_ylim(0.0, 1.60)
    ax.set_xlabel(r'Electron filling $n$')
    ax.set_ylabel(r'Rashba coupling $V_{\mathrm{SO}}/t$')
    ax.set_title(r'(a) $(n, V_{\mathrm{SO}}/t)$ Phase Diagram ($U/t=75, \Delta_t/t=4.5$)')
    ax.grid(alpha=0.2)

    # (b) Condensation energy |Delta F| heatmap from Nk=201 adaptive points (504 pts)
    ax = axes[1]
    pts2 = pd201['points']
    x2 = np.array([p['n'] for p in pts2]); y2 = np.array([p['V'] for p in pts2])
    z2 = np.array([abs(p['branches'][0]['DS']) if p['nb'] > 0 else 0.0 for p in pts2])
    tc = ax.tricontourf(x2, y2, z2, levels=20, cmap='magma')
    cb = fig.colorbar(tc, ax=ax, pad=0.02)
    cb.set_label(r'Order-parameter amplitude $|\Delta_S|/t$')
    ax.contour(N_grid, V_grid, lam75, levels=[1.0], colors=['white'], linewidths=2.0)
    ax.contour(N_grid, V_grid, muN, levels=[4.0], colors=[C_SKY], linewidths=1.5, linestyles='--')
    # also overlay Delta_0 = 0 nodal sign-flip crossover (relative sign Delta_0 * sign(Delta_S) = 0)
    import matplotlib.tri as mtri
    sc2 = [p for p in pts2 if p['nb'] > 0]
    x_sc = np.array([p['n'] for p in sc2]); y_sc = np.array([p['V'] for p in sc2])
    d0_z = np.array([p['branches'][0]['D0'] * np.sign(p['branches'][0]['DS']) for p in sc2])
    ds_sc = np.array([abs(p['branches'][0]['DS']) for p in sc2])
    lam_sc = np.array([p['lam_cont'] for p in sc2])
    tri_sc = mtri.Triangulation(x_sc, y_sc)
    tri_sc.set_mask((np.min(ds_sc[tri_sc.triangles], axis=1) < 0.05) |
                    (np.min(lam_sc[tri_sc.triangles], axis=1) < 1.005))
    ax.tricontour(tri_sc, d0_z, levels=[0.0], colors=[C_TEAL], linewidths=1.9, linestyles='-.')
    ax.text(1.48, 1.04, r'$\Delta_0=0$ nodal crossover', color=C_TEAL, fontsize=8.0,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=C_TEAL, alpha=0.88))
    ax.set_xlim(0.50, 1.95); ax.set_ylim(0.0, 1.60)
    ax.set_xlabel(r'Electron filling $n$')
    ax.set_ylabel(r'Rashba coupling $V_{\mathrm{SO}}/t$')
    ax.set_title(r'(b) $|\Delta_S(n, V_{\mathrm{SO}})|$ ($504$ adaptive points, $N_k=201$)')
    ax.grid(alpha=0.2)

    save_all(fig, 'fig05_task3_phase_diagram_nV')


# ---------------------------------------------------------------------------
# FIG 06: Task 3 Parameter-Space Slices (U slices, (U, V_SO), (n, U))
# ---------------------------------------------------------------------------
def make_fig06():
    t34 = json.load(open(os.path.join(DATA, 'task3_4_slices.json')))
    m = t34['maps']
    ns = np.array(m['ns']); Vs = np.array(m['Vs']); Us = np.array(m['Us'])
    N_grid, V_grid = np.meshgrid(ns, Vs, indexing='ij')

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)

    # (a) lambda_+=1 across U/t = 65..160 in (n, V_SO) using exact Uc_nV contours
    ax = axes[0]
    Uc_nV = np.array(m['Uc_nV'])
    u_cols = [(65, C_TEAL), (70, C_SKY), (75, C_BLUE), (100, C_AMBR), (130, C_VERM), (160, C_PURP)]
    handles = []
    for Uv, col in u_cols:
        ax.contour(N_grid, V_grid, Uc_nV, levels=[float(Uv)], colors=[col], linewidths=1.9)
        handles.append(Line2D([0], [0], color=col, lw=1.9, label=rf'$U/t={Uv}$'))
    ax.text(0.55, 0.08, r'$U/t \leq 56$: entire $(n, V_{\mathrm{SO}})$ domain is SC',
            fontsize=7.3, color=C_BLK, bbox=dict(boxstyle='round,pad=0.2', fc='#F5F5F5', ec=C_GRAY, alpha=0.9))
    ax.set_xlim(0.50, 1.95); ax.set_ylim(0.0, 1.60)
    ax.set_xlabel(r'Electron filling $n$'); ax.set_ylabel(r'$V_{\mathrm{SO}}/t$')
    ax.set_title(r'(a) $\lambda_+=1$ phase boundary across $U/t$')
    ax.grid(alpha=0.25); ax.legend(handles=handles, framealpha=0.9, fontsize=7.5)

    # (b) (U, V_SO) plane at n=1.0 and n=1.4 + SC multi-seed points
    ax = axes[1]
    V_ug, U_ug = np.meshgrid(Vs, Us, indexing='xy')
    for n_key, col, lbl in [('n1.0', C_BLUE, r'$n=1.0$ ($\lambda_+=1$)'),
                            ('n1.4', C_VERM, r'$n=1.4$ ($\lambda_+=1$)'),
                            ('n1.875', C_TEAL, r'$n=1.875$ ($\lambda_+=1$)')]:
        Uc_row = m['slices_UV'][n_key]['Uc']
        ax.plot(Vs, Uc_row, color=col, lw=2.0, label=lbl)
    # overlay SC multi-seed points for n=1.0 (UV_n10)
    for r in t34['sc_slices']['UV_n10']:
        mk = 'o' if r['nb'] > 0 else 'x'
        c = C_BLUE if r['nb'] > 0 else C_GRAY
        ax.scatter([r['V']], [r['U']], s=18, marker=mk, color=c, alpha=0.65)
    ax.axhline(75.0, color=C_BLK, ls='--', lw=1.0, label=r'$U/t=75$')
    ax.set_xlim(0.0, 1.55); ax.set_ylim(25, 180)
    ax.set_xlabel(r'$V_{\mathrm{SO}}/t$'); ax.set_ylabel(r'On-site repulsion $U/t$')
    ax.set_title(r'(b) $(V_{\mathrm{SO}}, U)$ slice ($N_b^{\max}=1$, $0$ metastable)')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.4, loc='upper left')

    # (c) (n, U) plane at V_SO=0.0, 0.5, 0.7
    ax = axes[2]
    for V_key, col, lbl in [('V0.0', C_GRAY, r'$V_{\mathrm{SO}}=0.0$'),
                            ('V0.5', C_BLUE, r'$V_{\mathrm{SO}}=0.5t$'),
                            ('V0.7', C_VERM, r'$V_{\mathrm{SO}}=0.7t$')]:
        Uc_col = m['slices_nU'][V_key]['Uc']
        ax.plot(ns, Uc_col, color=col, lw=2.0, label=lbl)
    for r in t34['sc_slices']['nU_V05']:
        mk = 'o' if r['nb'] > 0 else 'x'
        c = C_BLUE if r['nb'] > 0 else C_GRAY
        ax.scatter([r['n']], [r['U']], s=18, marker=mk, color=c, alpha=0.65)
    ax.axhline(75.0, color=C_BLK, ls='--', lw=1.0, label=r'$U/t=75$')
    ax.set_xlim(0.50, 1.95); ax.set_ylim(25, 180)
    ax.set_xlabel(r'Electron filling $n$'); ax.set_ylabel(r'On-site repulsion $U/t$')
    ax.set_title(r'(c) $(n, U)$ slice (SC below $U_c$, Normal above)')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.4, loc='upper left')

    save_all(fig, 'fig06_task3_slices_UV_nU')


# ---------------------------------------------------------------------------
# FIG 07: Task 4 Landau Free-Energy Functional F_eff(psi) & Topology
# ---------------------------------------------------------------------------
def make_fig07():
    t34 = json.load(open(os.path.join(DATA, 'task3_4_slices.json')))
    lan = t34['landau']
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)

    # (a) dF/psi^2 vs psi^2 across n at V=0.50 (convex everywhere, no barrier!)
    ax = axes[0]
    sel = [(0.95, 0.50, 75.0, C_GRAY, r'$n=0.95$ ($\lambda=0.954$)'),
           (1.00, 0.50, 75.0, C_SKY, r'$n=1.00$ ($\lambda=0.986$)'),
           (1.019, 0.50, 75.0, C_BLUE, r'$n=1.019$ ($\lambda=0.999$)'),
           (1.05, 0.50, 75.0, C_TEAL, r'$n=1.05$ ($\lambda=1.024$)'),
           (1.15, 0.50, 75.0, C_VERM, r'$n=1.15$ ($\lambda=1.094$)')]
    for nv, Vv, Uv, col, lbl in sel:
        L = next(x for x in lan if abs(x['n']-nv)<1e-3 and abs(x['V']-Vv)<1e-3 and abs(x['U']-Uv)<0.2)
        psi = np.array([p['psi'] for p in L['pts'] if p['psi'] <= 0.65])
        y = np.array([p['dF_over_psi2'] for p in L['pts'] if p['psi'] <= 0.65])
        ax.plot(psi**2, y*1e3, 'o-', ms=3.0, color=col, label=lbl)
    ax.axhline(0, color=C_BLK, ls='--', lw=0.9)
    ax.set_xlabel(r'$\psi^2 = \Delta_S^2 / t^2$')
    ax.set_ylabel(r'$(F_{\mathrm{eff}}(\psi)-F_N)/\psi^2$ ($10^{-3}\,t^{-1}$)')
    ax.set_title(r'(a) Exact Landau ridge: strictly convex ($\kappa, b > 0$)')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.2)

    # (b) Downward bend in dF/psi^2 vs 1/ln(4/psi) demonstrating T=0 Cooper-log term
    ax = axes[1]
    for nv, Vv, Uv, col, lbl in sel[1:]:
        L = next(x for x in lan if abs(x['n']-nv)<1e-3 and abs(x['V']-Vv)<1e-3 and abs(x['U']-Uv)<0.2)
        psi = np.array([p['psi'] for p in L['pts'] if p['psi'] <= 0.40])
        y = np.array([p['dF_over_psi2'] for p in L['pts'] if p['psi'] <= 0.40])
        xlog = 1.0/np.log(4.0/psi)
        ax.plot(xlog, y*1e3, 'o-', ms=3.2, color=col, label=lbl)
    ax.axhline(0, color=C_BLK, ls='--', lw=0.9)
    ax.set_xlabel(r'$1 / \ln(4t / |\psi|)$')
    ax.set_ylabel(r'$(F_{\mathrm{eff}}(\psi)-F_N)/\psi^2$ ($10^{-3}\,t^{-1}$)')
    ax.set_title(r'(b) Linear $\kappa/\ln(\Lambda/|\psi|)$ nonanalyticity as $\psi\to 0$')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.2)

    # (c) Positive quartic/log stability parameters along the phase boundary (no tricritical/CEP)
    ax = axes[2]
    bdy_cases = [x for x in lan if abs(x['U']-75.0)<0.2]
    labels = []; kappas = []; b_polys = []
    for L in bdy_cases:
        pts = [p for p in L['pts'] if p['psi'] <= 0.65]
        psi = np.array([p['psi'] for p in pts])
        y = np.array([p['dF_over_psi2'] for p in pts])
        X = np.column_stack([np.ones_like(psi), 1.0/np.log(4.0/psi), psi**2])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        _, b_p, _ = np.polyfit(psi**2, y, 2)
        labels.append(f"({L['n']:.2f},{L['V']:.2f})")
        kappas.append(coef[1]*1e3); b_polys.append(b_p*1e3)
    idx = np.arange(len(labels))
    ax.bar(idx - 0.18, kappas, width=0.35, color=C_BLUE, label=r'Log coeff $\kappa$ ($\times 10^3$)')
    ax.bar(idx + 0.18, b_polys, width=0.35, color=C_VERM, label=r'Eff. quartic $b_{\mathrm{poly}}$ ($\times 10^3$)')
    ax.axhline(0, color=C_BLK, lw=0.8)
    ax.set_xticks(idx); ax.set_xticklabels(labels, rotation=38, ha='right', fontsize=7.0)
    ax.set_ylabel(r'Stability coefficient ($\times 10^3$)')
    ax.set_title(r'(c) $\kappa, b_{\mathrm{poly}} > 0$ globally $\Rightarrow$ no tricritical/CEP')
    ax.grid(alpha=0.25, axis='y'); ax.legend(framealpha=0.9, fontsize=7.5)

    save_all(fig, 'fig07_task4_landau_topology')


# ---------------------------------------------------------------------------
# FIG 08: Task 5 Two-Spin Concurrence & Derivatives Across Boundaries
# ---------------------------------------------------------------------------
def make_fig08():
    t5 = json.load(open(os.path.join(DATA, 'task5_concurrence.json')))
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6), constrained_layout=True)

    # Col 0: Clean U cut (n=1.0, V=0.5, Uc ~ 73.9)
    cU = t5['cut_U_n1.0_V0.5']['rows']
    u = np.array([r['param'] for r in cU])
    Cs_u = np.array([r['sc_metrics']['spin']['0.25_0']['C'] for r in cU])
    Csn_u = np.array([r['sc_metrics']['spin']['0.25_0']['C_normonly'] for r in cU])
    Csp_u = np.array([r['norm_metrics']['spin']['0.25_0']['C'] for r in cU])
    Cb_u = np.array([r['sc_metrics']['band']['0.25_0']['C'] for r in cU])
    dCs_u = np.gradient(Cs_u, u); d2Cs_u = np.gradient(dCs_u, u)

    ax = axes[0, 0]
    ax.plot(u, Cs_u, color=C_BLUE, label=r'SC full $C_{\mathrm{spin}}(r=0.25)$')
    ax.plot(u, Csn_u, '--', color=C_SKY, label=r'SC $F_{AB}=0$ ($C_{\mathrm{norm\text{-}only}}$)')
    ax.plot(u, Csp_u, ':', color=C_GRAY, label=r'Pure normal ($\Delta=0$)')
    ax.plot(u, Cb_u, '-.', color=C_VERM, label=r'SC full $C_{\mathrm{band}}(r=0.25)$')
    ax.axvline(73.9, color=C_BLK, ls=':', lw=1.1)
    ax.set_xlabel(r'$U/t$ ($n=1.0, V_{\mathrm{SO}}=0.5t$)')
    ax.set_ylabel(r'Concurrence $C(r=0.25)$')
    ax.set_title(r'(a) Clean QPT at $U_c\approx 73.9t$ (fixed $\mu_N$)')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.2)

    ax = axes[1, 0]
    ax.plot(u, dCs_u*1e3, color=C_BLUE, label=r'$dC_{\mathrm{spin}}/dU$ ($\times 10^3$)')
    ax2 = ax.twinx()
    ax2.plot(u, d2Cs_u*1e3, '--', color=C_VERM, label=r'$d^2C_{\mathrm{spin}}/dU^2$ ($\times 10^3$)')
    ax2.set_ylabel(r'$d^2C/dU^2$ ($\times 10^{-3}\,t^{-2}$)', color=C_VERM)
    ax2.tick_params(axis='y', colors=C_VERM)
    ax.axvline(73.9, color=C_BLK, ls=':', lw=1.1)
    ax.set_xlabel(r'$U/t$')
    ax.set_ylabel(r'$dC/dU$ ($\times 10^{-3}\,t^{-1}$)')
    ax.set_title(r'(d) $dC/dU$ onset & peak $|d^2C/dU^2|$ near $U_c$')
    ax.grid(alpha=0.25); ax.legend(loc='upper right', framealpha=0.9, fontsize=7.3)

    # Col 1: Density cut (V=0.5, nc ~ 1.019)
    cn = t5['cut_n_V0.5']['rows']
    # take regular 0.01 grid for clean derivative vs fine 0.002 inset/comparison
    cn_reg = [r for r in cn if round(r['param']*100)%1 == 0 and (r['param']<0.991 or r['param']>1.059 or round(r['param']*1000)%10==0)]
    n_r = np.array([r['param'] for r in cn_reg])
    Cs_n = np.array([r['sc_metrics']['spin']['0.25_0']['C'] for r in cn_reg])
    Csp_n = np.array([r['norm_metrics']['spin']['0.25_0']['C'] for r in cn_reg])
    Cb_n = np.array([r['sc_metrics']['band']['0.25_0']['C'] for r in cn_reg])
    dCs_n = np.gradient(Cs_n, n_r); dCsp_n = np.gradient(Csp_n, n_r)

    ax = axes[0, 1]
    ax.plot(n_r, Cs_n, color=C_BLUE, label=r'SC $C_{\mathrm{spin}}$')
    ax.plot(n_r, Csp_n, ':', color=C_GRAY, lw=1.8, label=r'Normal $C_{\mathrm{spin}}$ ($\Delta=0$)')
    ax.plot(n_r, Cb_n, '-.', color=C_VERM, label=r'SC $C_{\mathrm{band}}$')
    ax.axvline(1.012, color=C_BLK, ls=':', lw=1.1, label=r'$n_c$ ($N_k=201$)')
    ax.set_xlabel(r'Electron filling $n$ ($V_{\mathrm{SO}}=0.5t, U=75t$)')
    ax.set_ylabel(r'Concurrence $C(r=0.25)$')
    ax.set_title(r'(b) Density cut: $>95\%$ normal background')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.3)

    ax = axes[1, 1]
    ax.plot(n_r, dCs_n, color=C_BLUE, label=r'SC $dC_{\mathrm{spin}}/dn$ ($\Delta n=0.01$)')
    ax.plot(n_r, dCsp_n, ':', color=C_GRAY, label=r'Normal $dC_{\mathrm{spin}}/dn$')
    # also plot Nk=401 check
    cn4 = t5['cut_n_V0.5_Nk401']['rows']
    n4 = np.array([r['param'] for r in cn4])
    C4 = np.array([r['sc_metrics']['spin']['0.25_0']['C'] for r in cn4])
    ax.plot(n4, np.gradient(C4, n4), '--', color=C_TEAL, label=r'SC $dC/dn$ ($N_k=401$)')
    ax.axvline(1.019, color=C_BLK, ls=':', lw=1.1)
    ax.set_xlabel(r'Electron filling $n$')
    ax.set_ylabel(r'$dC_{\mathrm{spin}}/dn$')
    ax.set_title(r'(e) Slope $dC/dn$ smooth (FS noise at fine $\Delta n$)')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.2)

    # Col 2: V_SO cuts at n=1.0 (onset Vc~0.54, smooth at 1.235) and n=1.4 (smooth at 0.525)
    cV10 = t5['cut_V_n1.0']['rows']
    cV14 = t5['cut_V_n1.4']['rows']
    v10 = np.array([r['param'] for r in cV10]); C10 = np.array([r['sc_metrics']['spin']['0.25_0']['C'] for r in cV10])
    Cp10 = np.array([r['norm_metrics']['spin']['0.25_0']['C'] for r in cV10])
    v14 = np.array([r['param'] for r in cV14]); C14 = np.array([r['sc_metrics']['spin']['0.25_0']['C'] for r in cV14])
    Cp14 = np.array([r['norm_metrics']['spin']['0.25_0']['C'] for r in cV14])

    ax = axes[0, 2]
    ax.plot(v10, C10, color=C_BLUE, label=r'$n=1.0$ SC')
    ax.plot(v10, Cp10, ':', color=C_SKY, label=r'$n=1.0$ Normal')
    ax.plot(v14, C14, color=C_VERM, label=r'$n=1.4$ SC')
    ax.plot(v14, Cp14, ':', color=C_AMBR, label=r'$n=1.4$ Normal')
    ax.axvline(0.532, color=C_BLUE, ls=':', lw=1.0)
    ax.axvline(1.235, color=C_GRAY, ls='--', lw=1.0)
    ax.axvline(0.525, color=C_VERM, ls='--', lw=1.0)
    ax.set_xlabel(r'$V_{\mathrm{SO}}/t$')
    ax.set_ylabel(r'Concurrence $C_{\mathrm{spin}}(r=0.25)$')
    ax.set_title(r'(c) $V_{\mathrm{SO}}$ cuts: smooth at $0.525t$ & $1.235t$')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.2)

    ax = axes[1, 2]
    ax.plot(v10, np.gradient(C10, v10), color=C_BLUE, label=r'$n=1.0$ SC $dC/dV_{\mathrm{SO}}$')
    ax.plot(v10, np.gradient(Cp10, v10), ':', color=C_SKY, lw=1.4, label=r'$n=1.0$ Normal $dC_N/dV_{\mathrm{SO}}$')
    ax.plot(v14, np.gradient(C14, v14), color=C_VERM, label=r'$n=1.4$ SC $dC/dV_{\mathrm{SO}}$')
    ax.axvline(0.532, color=C_BLUE, ls=':', lw=1.1, label=r'$V_c(1.0)\approx 0.532t$')
    ax.axvline(1.235, color=C_GRAY, ls='--', lw=1.0, label=r'Refuted $1.235t$')
    ax.set_xlabel(r'$V_{\mathrm{SO}}/t$')
    ax.set_ylabel(r'$dC_{\mathrm{spin}}/dV_{\mathrm{SO}}$ ($t^{-1}$)')
    ax.set_title(r'(f) $dC/dV_{\mathrm{SO}}$ smooth at $0.525t$ & $1.235t$')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=6.9)

    save_all(fig, 'fig08_task5_concurrence_boundaries')


# ---------------------------------------------------------------------------
# FIG 09: Task 5 Decomposition, Distance Scaling & Wootters ESD Artifacts
# ---------------------------------------------------------------------------
def make_fig09():
    t5 = json.load(open(os.path.join(DATA, 'task5_concurrence.json')))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)

    # (a) Distance scaling across r_sep on cut_U_n1.0_V0.5
    ax = axes[0]
    rows = t5['cut_U_n1.0_V0.5']['rows']
    u = np.array([r['param'] for r in rows])
    for rk, lbl, col in [('0.25_0', r'$r=0.25a$', C_BLUE),
                         ('0.35_0', r'$r=0.35a$', C_TEAL),
                         ('0.5_0',  r'$r=0.50a$', C_VERM),
                         ('0.75_0', r'$r=0.75a$', C_AMBR),
                         ('1_0',    r'$r=1.00a$ (lattice)', C_BLK)]:
        C = [r['sc_metrics']['spin'][rk]['C'] for r in rows]
        ax.plot(u, C, color=col, label=lbl)
    ax.axvline(73.9, color=C_GRAY, ls=':', lw=1.1)
    ax.set_xlabel(r'$U/t$ ($n=1.0, V_{\mathrm{SO}}=0.5t$)')
    ax.set_ylabel(r'Spin concurrence $C(r)$')
    ax.set_title(r'(a) Short-range exchange dominance ($C(r\geq 1)\equiv 0$)')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.4)

    # (b) Wootters ESD cutoff artifact in cut_V_n1.4 (r=0.5) & cut_V_n1.0 (band r=0.5)
    ax = axes[1]
    rows14 = t5['cut_V_n1.4']['rows']
    v14 = np.array([r['param'] for r in rows14])
    C14_s = np.array([r['sc_metrics']['spin']['0.5_0']['C'] for r in rows14])
    Cr14_s = np.array([r['sc_metrics']['spin']['0.5_0']['C_raw'] for r in rows14])
    C14_b = np.array([r['sc_metrics']['band']['0.5_0']['C'] for r in rows14])
    Cr14_b = np.array([r['sc_metrics']['band']['0.5_0']['C_raw'] for r in rows14])
    ax.plot(v14, C14_s, color=C_BLUE, lw=2.0, label=r'Spin $C=\max(0,C_{\mathrm{raw}})$')
    ax.plot(v14, Cr14_s, '--', color=C_SKY, lw=1.5, label=r'Spin raw $\tilde{\lambda}_1-\sum_{i>1}\tilde{\lambda}_i$')
    ax.plot(v14, C14_b, color=C_VERM, lw=2.0, label=r'Band $C=\max(0,C_{\mathrm{raw}})$')
    ax.plot(v14, Cr14_b, '--', color=C_AMBR, lw=1.5, label=r'Band raw $C_{\mathrm{raw}}$')
    ax.axhline(0, color=C_BLK, lw=0.9)
    ax.annotate('ESD kink ($V\\approx 0.75, 0.81$)\nnot a QPT!', xy=(0.81, 0.0),
                xytext=(0.50, -0.06), arrowprops=dict(arrowstyle='->', color=C_BLK),
                fontsize=7.6)
    ax.set_xlabel(r'$V_{\mathrm{SO}}/t$ ($n=1.4$, always SC)')
    ax.set_ylabel(r'Concurrence at $r=0.50a$')
    ax.set_title(r'(b) False $d^2C/dV^2$ spikes from Wootters cutoff')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.2)

    # (c) Decomposition of SC change into Fermi-smearing vs anomalous F_AB
    ax = axes[2]
    for basis, col, lbl_b in [('spin', C_BLUE, 'Spin'), ('band', C_VERM, 'Band')]:
        d_smear = np.array([r['sc_metrics'][basis]['0.25_0']['C_normonly'] -
                            r['norm_metrics'][basis]['0.25_0']['C'] for r in rows])
        d_anom = np.array([r['sc_metrics'][basis]['0.25_0']['C'] -
                           r['sc_metrics'][basis]['0.25_0']['C_normonly'] for r in rows])
        ax.plot(u, d_smear, '-', color=col, label=rf'{lbl_b}: Fermi smearing ($C_{{F=0}}-C_N$)')
        ax.plot(u, d_anom, '--', color=col, label=rf'{lbl_b}: Anomalous $F_{{AB}}$ ($C-C_{{F=0}}$)')
    ax.axvline(73.9, color=C_BLK, ls=':', lw=1.1)
    ax.axhline(0, color=C_GRAY, lw=0.8)
    ax.set_xlabel(r'$U/t$ ($n=1.0, V_{\mathrm{SO}}=0.5t$)')
    ax.set_ylabel(r'$\Delta C(r=0.25)$ decomposition')
    ax.set_title(r'(c) Fermi-step smearing dominates over $F_{AB}$')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.1)

    save_all(fig, 'fig09_task5_decomposition_esd')


# ---------------------------------------------------------------------------
# FIG 10: Task 6 Revalidation at n = 1.875 (V_SO & U sweeps)
# ---------------------------------------------------------------------------
def make_fig10():
    t67 = json.load(open(os.path.join(DATA, 'task6_7_audit.json')))
    t5 = json.load(open(os.path.join(DATA, 'task5_concurrence.json')))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)

    # (a) mu_SC vs mu_N vs 4t across V_SO (comparing repo Nk=201 vs periodic Nk=401)
    ax = axes[0]
    d_r = t67['task6_V_sweep']['Nk201_repo']['rows']
    d_p = t67['task6_V_sweep']['Nk401_per']['rows']
    vr = [r['V'] for r in d_r]; vp = [r['V'] for r in d_p]
    ax.plot(vr, [r['mu_SC'] for r in d_r], '--', color=C_SKY, label=r'$\mu_{\mathrm{SC}}$ (repo $N_k=201$)')
    ax.plot(vp, [r['mu_SC'] for r in d_p], '-', color=C_BLUE, label=r'$\mu_{\mathrm{SC}}$ (per $N_k=401$)')
    ax.plot(vp, [r['mu_N'] for r in d_p], '-', color=C_TEAL, label=r'$\mu_N$ (normal, per $N_k=401$)')
    ax.axhline(4.0, color=C_BLK, ls=':', lw=1.2, label=r'$4t$ (band maximum / Dirac pt)')
    ax.axvline(0.5487, color=C_SKY, ls=':', lw=1.0)
    ax.axvline(0.5619, color=C_BLUE, ls=':', lw=1.0)
    ax.axvline(0.6936, color=C_TEAL, ls=':', lw=1.0)
    ax.set_xlim(0.2, 1.1); ax.set_ylim(3.2, 4.8)
    ax.set_xlabel(r'$V_{\mathrm{SO}}/t$ ($n=1.875, U=75t$)')
    ax.set_ylabel(r'Chemical potential ($t$)')
    ax.set_title(r'(a) $\mu_{\mathrm{SC}}=4t$ ($0.549/0.562$) vs $\mu_N=4t$ ($0.694$)')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.2)

    # (b) U sweeps at V=0.5 (U=68.55/66.81) & V=0.7 (U=118.04/111.71) + spectral gap
    ax = axes[1]
    for tag, col, lbl in [('V0.5_Nk201_repo', C_SKY, r'$V=0.5$ repo ($U=68.57$)'),
                          ('V0.5_Nk401_per',  C_BLUE, r'$V=0.5$ per ($U=66.81$)'),
                          ('V0.7_Nk201_repo', C_AMBR, r'$V=0.7$ repo ($U=118.08$)'),
                          ('V0.7_Nk401_per',  C_VERM, r'$V=0.7$ per ($U=111.71$)')]:
        rows = t67['task6_U_sweeps'][tag]['rows']
        ax.plot([r['U'] for r in rows], [r['mu_SC'] for r in rows], color=col, label=lbl)
    ax.axhline(4.0, color=C_BLK, ls=':', lw=1.2, label=r'$4t$ ($\mu_N$ constant vs $U$!)')
    ax.set_xlabel(r'On-site repulsion $U/t$ ($n=1.875$)')
    ax.set_ylabel(r'$\mu_{\mathrm{SC}} / t$')
    ax.set_title(r'(b) Companion $\mu_{\mathrm{SC}}=4t$ crossings in $U/t$')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.1)

    # (c) Spectral gap min_E & smoothness of SC d2e/dV2 & concurrence across V_SO
    ax = axes[2]
    d4p = t67['task6_V_sweep']['Nk401_per']
    v4p = np.array([r['V'] for r in d4p['rows']])
    ax.plot(v4p, [r['min_E_band0'] for r in d4p['rows']], color=C_BLUE, label=r'$\min_k E_{k,-}/t$ (full gap $\geq 0.94t$)')
    ax.plot(v4p, [r['E_pipi_band0'] for r in d4p['rows']], '--', color=C_TEAL, label=r'$E_{(\pi,\pi),-}/t$')
    cV = t5['cut_V_n1.875']['rows']
    ax.plot([r['param'] for r in cV], [r['sc_metrics']['spin']['0.25_0']['C'] for r in cV],
            color=C_VERM, label=r'$C_{\mathrm{spin}}(r=0.25)$ (smooth)')
    ax2 = ax.twinx()
    ax2.plot(v4p, d4p['d2eSC_dV2'], ':', color=C_PURP, label=r'$d^2e_{\mathrm{SC}}/dV^2$')
    ax2.set_ylabel(r'$d^2e_{\mathrm{SC}}/dV_{\mathrm{SO}}^2$ ($t^{-1}$)', color=C_PURP)
    ax2.tick_params(axis='y', colors=C_PURP)
    ax.axvline(0.5619, color=C_BLUE, ls=':', lw=1.0)
    ax.axvline(0.6936, color=C_TEAL, ls=':', lw=1.0)
    ax.set_xlabel(r'$V_{\mathrm{SO}}/t$ ($n=1.875, U=75t$)')
    ax.set_ylabel(r'Gap ($t$) / Concurrence $C$')
    ax.set_title(r'(c) Fully gapped spectrum & smooth observables')
    ax.grid(alpha=0.25); ax.legend(loc='center right', framealpha=0.9, fontsize=7.1)

    save_all(fig, 'fig10_task6_n1875_revalidation')


# ---------------------------------------------------------------------------
# FIG 11: Task 6 k-Space Topology & Dirac-Cone Normal vs Gapped SC
# ---------------------------------------------------------------------------
def make_fig11():
    t67 = json.load(open(os.path.join(DATA, 'task6_7_audit.json')))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)

    # (a) Normal-state inner pocket disappearance near (pi,pi) as V_SO crosses 0.694t
    ax = axes[0]
    sk, ssin, eps, KX, KY, kx = fs.make_grid(401, 1.0, 0.40, periodic=True)
    for Vv, col, lbl in [(0.40, C_BLUE, r'$V=0.40t$ (inner pocket)'),
                         (0.562, C_AMBR, r'$V=0.562t$ ($\mu_{\mathrm{SC}}=4t$)'),
                         (0.694, C_TEAL, r'$V=0.694t$ ($\mu_N=4t$ Dirac pt)')]:
        _, _, eps_v, KXv, KYv, _ = fs.make_grid(401, 1.0, Vv, periodic=True)
        muN = fs.normal_state(eps_v, 1.875)['mu']
        # outer pocket contour
        ax.contour(KXv/np.pi, KYv/np.pi, eps_v[..., 1], levels=[muN], colors=[col], linewidths=1.3, linestyles='--')
        if np.max(eps_v[..., 0]) > muN + 1e-4:
            ax.contour(KXv/np.pi, KYv/np.pi, eps_v[..., 0], levels=[muN], colors=[col], linewidths=2.0)
    # focus on top-right corner near (pi,pi)
    ax.set_xlim(0.55, 1.0); ax.set_ylim(0.55, 1.0)
    ax.scatter([1.0], [1.0], c=C_BLK, s=45, zorder=5)
    custom = [Line2D([0], [0], color=C_BLUE, lw=2.0, label=r'$V=0.40t$: inner+outer'),
              Line2D([0], [0], color=C_AMBR, lw=2.0, label=r'$V=0.562t$: inner shrinks'),
              Line2D([0], [0], color=C_TEAL, lw=1.5, ls='--', label=r'$V=0.694t$: inner vanishes')]
    ax.legend(handles=custom, framealpha=0.9, fontsize=7.3, loc='lower left')
    ax.set_xlabel(r'$k_x / \pi$'); ax.set_ylabel(r'$k_y / \pi$')
    ax.set_title(r'(a) Normal FS around $(\pi,\pi)$ ($n=1.875$)')
    ax.grid(alpha=0.25)

    # (b) Diagonal dispersion eps_pm vs SC Bogoliubov E_pm near (pi,pi) at V=0.562 and 0.694
    ax = axes[1]
    q = np.linspace(0.65*np.pi, np.pi, 300)
    for Vv, col, lbl in [(0.562, C_BLUE, r'$V=0.562t$'), (0.694, C_TEAL, r'$V=0.694t$')]:
        eps_line = -4.0*np.cos(q)
        rashba = Vv*np.sqrt(2.0)*np.abs(np.sin(q))
        e0 = eps_line - rashba; e1 = eps_line + rashba
        # get SC parameters at this V
        rows = t67['task6_V_sweep']['Nk401_per']['rows']
        r = min(rows, key=lambda x: abs(x['V']-Vv))
        D0, DS, muSC = r['D0'], r['DS'], r['mu_SC']
        E0 = np.sqrt((D0 - DS*e0/4.0)**2 + (e0 - muSC)**2)
        ax.plot(q/np.pi, e0, '--', color=col, alpha=0.75, label=rf'{lbl}: $\varepsilon_-(k)$')
        ax.plot(q/np.pi, E0, '-', color=col, lw=2.0, label=rf'{lbl}: $E_-(k)$ (gapped)')
    ax.axhline(4.0, color=C_BLK, ls=':', lw=1.0, label=r'$\varepsilon=4t$ Dirac vertex')
    ax.set_xlabel(r'Diagonal $k_x=k_y$ ($\times \pi$)')
    ax.set_ylabel(r'Energy ($t$)')
    ax.set_title(r'(b) Dirac vertex at $(\pi,\pi)$ gapped by SC $E_-\geq 1.56t$')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.0, loc='lower left')

    # (c) Normal 3rd-order Dirac-Lifshitz signature d3eN/dV3 vs analytic SC d3eSC/dV3
    ax = axes[2]
    # Use clean T=2e-4 Nk=801 normal curve computed fast on the fly!
    from scipy.optimize import brentq
    Vs_f = np.round(np.arange(0.60, 0.781, 0.01), 3)
    eN_f = []
    for V in Vs_f:
        _, _, eps_v, _, _, _ = fs.make_grid(401, 1.0, float(V), periodic=True)
        mu = brentq(lambda m: float(2.0*np.mean(1.0/(1.0+np.exp(np.clip((eps_v-m)/3e-4,-60,60))))-1.875),
                    3.4, 4.6, xtol=1e-12)
        f = 1.0/(1.0+np.exp(np.clip((eps_v-mu)/3e-4,-60,60)))
        eN_f.append(float(2.0*np.mean(eps_v*f)))
    v_f = np.array(Vs_f)
    d2N = np.gradient(np.gradient(eN_f, v_f, edge_order=2), v_f, edge_order=2)
    d3N = np.gradient(d2N, v_f, edge_order=2)
    d4p = t67['task6_V_sweep']['Nk401_per']
    v_s = np.array([r['V'] for r in d4p['rows']])
    d3S = np.gradient(np.array(d4p['d2eSC_dV2']), v_s, edge_order=2)
    m_s = (v_s >= 0.58) & (v_s <= 0.80)
    ax.plot(v_f, d3N, 'o-', ms=3.5, color=C_TEAL, label=r'Normal $d^3e_N/dV^3$ (3rd-order Dirac jump)')
    ax.plot(v_s[m_s], d3S[m_s], '-', lw=2.0, color=C_VERM, label=r'SC $d^3e_{\mathrm{SC}}/dV^3$ (analytic)')
    ax.axvline(0.6936, color=C_TEAL, ls=':', lw=1.2, label=r'$V_L^{\mathrm{norm}}=0.694t$')
    ax.axhline(0, color=C_BLK, lw=0.8)
    ax.set_xlabel(r'$V_{\mathrm{SO}}/t$ ($n=1.875$)')
    ax.set_ylabel(r'$d^3 e_{\mathrm{GS}} / dV_{\mathrm{SO}}^3$ ($t^{-2}$)')
    ax.set_title(r'(c) Normal 3rd-order Lifshitz vs smooth SC')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.2)

    save_all(fig, 'fig11_task6_kspace_topology')


# ---------------------------------------------------------------------------
# FIG 12: Task 7 Numerical Convergence & Robustness Dashboard
# ---------------------------------------------------------------------------
def make_fig12():
    t67 = json.load(open(os.path.join(DATA, 'task6_7_audit.json')))
    t5 = json.load(open(os.path.join(DATA, 'task5_concurrence.json')))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)

    # (a) Convergence of V(muN=4t), V(muSC=4t), and nc(V=0.5) across Nk=101..801
    ax = axes[0]
    nks_v = [201, 401, 801]
    v_sc_repo = [t67['task6_V_sweep'][f'Nk{k}_repo']['V_crossing_muSC_4t'] for k in nks_v]
    v_sc_per  = [t67['task6_V_sweep'][f'Nk{k}_per']['V_crossing_muSC_4t'] for k in nks_v]
    v_n_repo  = [t67['task6_V_sweep'][f'Nk{k}_repo']['V_Lifshitz_muN_4t'] for k in nks_v]
    v_n_per   = [t67['task6_V_sweep'][f'Nk{k}_per']['V_Lifshitz_muN_4t'] for k in nks_v]
    invN = 1.0/np.array(nks_v)
    ax.plot(invN*1e3, v_sc_repo, 'o--', color=C_SKY, label=r'$V(\mu_{\mathrm{SC}}=4t)$ repo')
    ax.plot(invN*1e3, v_sc_per,  's-',  color=C_BLUE, label=r'$V(\mu_{\mathrm{SC}}=4t)$ per (exact $0.5619$)')
    ax.plot(invN*1e3, v_n_repo,  'o--', color=C_AMBR, label=r'$V(\mu_N=4t)$ repo')
    ax.plot(invN*1e3, v_n_per,   's-',  color=C_TEAL, label=r'$V(\mu_N=4t)$ per ($\to 0.694$)')
    ax.set_xlabel(r'$10^3 / N_k$')
    ax.set_ylabel(r'Crossing $V_{\mathrm{SO}}/t$')
    ax.set_title(r'(a) Crossing convergence vs $1/N_k$ ($n=1.875$)')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.2)

    # (b) Benchmark |D0(Nk) - D0_per(801)| across Nk=101..801 for 4 regimes
    ax = axes[1]
    bm = t67['task7_convergence']['benchmarks']
    nks_b = [101, 201, 401]
    for pname, col, lbl in [('n1.875_V0.5', C_BLUE, r'$(1.875, 0.5)$'),
                            ('n1.0_V1.235', C_VERM, r'$(1.0, 1.235)$'),
                            ('n1.4_V0.525', C_TEAL, r'$(1.4, 0.525)$')]:
        b = bm[pname]
        er = [abs(b[f'repo_{k}']['D0'] - b['per_801']['D0']) for k in nks_b]
        ep = [abs(b[f'per_{k}']['D0'] - b['per_801']['D0']) for k in nks_b]
        ax.loglog(nks_b, er, 'o--', color=col, alpha=0.7, label=rf'{lbl} repo')
        ax.loglog(nks_b, ep, 's-',  color=col, label=rf'{lbl} per')
    ax.set_xlabel(r'$N_k$')
    ax.set_ylabel(r'$|\Delta_0(N_k) - \Delta_{0,\mathrm{per}}(801)|$ ($t$)')
    ax.set_title(r'(b) Gap convergence (per $10^{-6}$ vs repo $10^{-3}$)')
    ax.grid(alpha=0.25, which='both'); ax.legend(framealpha=0.9, fontsize=6.8, ncol=2)

    # (c) Derivative step-size artifact in d2C/dn2 (normal grid stair-step noise vs SC smooth)
    ax = axes[2]
    rows = t5['cut_n_V0.5']['rows']
    sub = [r for r in rows if 0.990 <= r['param'] <= 1.060]
    p = np.array([r['param'] for r in sub])
    C = np.array([r['sc_metrics']['spin']['0.25_0']['C'] for r in sub])
    Cp = np.array([r['norm_metrics']['spin']['0.25_0']['C'] for r in sub])
    d2C = np.gradient(np.gradient(C, p), p)
    d2Cp = np.gradient(np.gradient(Cp, p), p)
    ax.plot(p[2:-2], d2C[2:-2], color=C_BLUE, label=r'SC $d^2C/dn^2$ ($\Delta n=0.002, N_k=201$)')
    ax.plot(p[2:-2], d2Cp[2:-2], '--', color=C_VERM, alpha=0.85, label=r'Pure normal $d^2C_N/dn^2$ (FS grid noise!)')
    ax.axvline(1.012, color=C_BLK, ls=':', lw=1.1, label=r'$n_c$ ($N_k=201$)')
    ax.set_xlabel(r'Electron filling $n$')
    ax.set_ylabel(r'$d^2 C_{\mathrm{spin}} / dn^2$')
    ax.set_title(r'(c) Finite-grid FS stair-steps fake $d^2C/dn^2$ peaks')
    ax.grid(alpha=0.25); ax.legend(framealpha=0.9, fontsize=7.2)

    save_all(fig, 'fig12_task7_convergence_audit')


if __name__ == '__main__':
    print("Generating all 12 figure sets (PNG + PDF + SVG)...")
    make_fig01()
    make_fig02()
    make_fig03()
    make_fig04()
    make_fig05()
    make_fig06()
    make_fig07()
    make_fig08()
    make_fig09()
    make_fig10()
    make_fig11()
    make_fig12()
    print("ALL 12 FIGURES GENERATED IN research/figs/")
