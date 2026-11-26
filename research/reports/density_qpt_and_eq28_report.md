# Electron-Density ($n$) Effects on Quantum Phase Transitions: `arXiv:1801.08988` Eq. (28) Ground-State Energy vs. Two-Qubit Concurrence

**Scope:** Follow-up Task 8 density study (`research/task8_density_qpt_eq28.py`, `research/make_density_qpt_figures.py`)  
**Primary Artifacts:** `research/data/task8_density_qpt_eq28.json`, `research/data/csv/task8_density_{full_sweeps,nk801_zooms}.csv`, `research/figs/fig13_density_eq28_phase_landscape.{png,pdf,svg}`, `research/figs/fig14_density_qpt_zooms_eq28_vs_concurrence.{png,pdf,svg}`

---

## Executive Summary

1. **Clarification on Concurrence as a QPT Indicator:** Prior repository scripts (`analysis/plot_qpt_signature_paper_comparison.py`) and manuscript literature (`manuscript.tex` Secs. `sec:vso_concurrence`–`sec:qpt`, citing Osterloh *et al.* 2002; Yang 2005; Ebrahimian *et al.* 2022) tested Wootters concurrence $C$ and its derivatives as candidate quantum phase transition (QPT) markers. Our earlier audit (`research/reports/qpt_and_concurrence_audit_report.md` Section 5) already showed that raw $C$ fails as a universal QPT indicator along $V_{\mathrm{SO}}$ and noisy $N_k=201$ density cuts. Extending that test here across the **entire electron filling $n \in (0, 2)$** on converged **periodic $N_k = 401$ and $N_k = 801$** grids resolves exactly what density $n$ does to **both** Equation (28) ground-state energy and two-qubit concurrence $C(n)$.
2. **Exact Verification of `arXiv:1801.08988` Eq. (28):** Equation (28) of Hutchinson, Hirsch, & Marsiglio (*Phys. Rev. B* **97**, 184513 (2018)) defines the grand-canonical mean-field ground-state energy $E_g = \langle H_{\mathrm{MF}} - \mu\hat{N}\rangle_0 = \frac{1}{2}\sum_{\mathbf{k},s}[(\epsilon_{\mathbf{k}s}-\mu) - E_{\mathbf{k}s} + \bar{\Delta}_{\mathbf{k}s}b_{\mathbf{k}s}^*]$ (with $b_{\mathbf{k}s} = \bar{\Delta}_{\mathbf{k}s}/(2E_{\mathbf{k}s})$ at $T=0$ and **no** extra $\Delta^2/U$ term). At fixed electron density $n$, the canonical ground-state energy per site is $e_{\mathrm{GS}}(n) = E_g/N + \mu n$, which satisfies $\frac{de_{\mathrm{GS}}}{dn} = \mu$ and $\frac{d^2 e_{\mathrm{GS}}}{dn^2} = \frac{d\mu}{dn}$ across $n \in (0, 2)$ to discretization precision ($\le 2.4\times 10^{-3}t$ at $N_k=401$, $\le 1.1\times 10^{-3}t$ on $N_k=801$ zooms, and $\le 2.3\times 10^{-5}t$ inside gapped SC).
3. **New Physics Finding — Dual Density Pairing Boundaries ($n_{c,1}$ and $n_{c,2}$) at $U=75t, \Delta_t=4.5t$:** Because $\Delta_t = 4.5t > 2t$ yields $U_c(n\to 0^+) = (2\Delta_t - 4)\Delta_t |E_{\min}| = 95.46t > 75t$ (at $V_{\mathrm{SO}}=0.5t$), the canonical $U=75t$ model hosts **two** superconducting phases vs. density (`Fig. 13a`): a newly identified **re-entrant dilute nodal ($s_\pm$-like, $\Delta_0\cdot\Delta_S < 0$) pocket** for $n < n_{c,1} = 0.2192$ (missed by `analysis/n_density_scan.py`, which started at $n=0.30$), and the main high-density SC phase for $n > n_{c,2} = 1.0194$. Raising $U$ to $115t > 95.46t$ (`arXiv:1801.08988` Fig. 5 right) closes the dilute pocket and leaves a single transition at $n_c = 1.4602$.
4. **Effect of Density $n$ on QPT Detection — Eq. (28) vs. Concurrence $C(n)$:**
   - **Normal-State van Hove & Dirac–Lifshitz QPTs:** On converged $N_k=801$ grids, both Eq. (28) $d^2e_{\mathrm{GS}}/dn^2 = d\mu_N/dn$ **and** short-range concurrence $dC_N/dn, d^2C_N/dn^2$ detect genuine normal-state Fermi-surface topology QPTs: the logarithmic van Hove saddles ($n_{\mathrm{vH},-} = 0.9065$ at $V_{\mathrm{SO}}=0.5t$; $n_{\mathrm{vH}} = 1.000$ at $V_{\mathrm{SO}}=0$) and the 3rd-order $(\pi,\pi)$ Dirac cone ($n_{\mathrm{Dirac},4} = 1.9290$, where $d^3e_N/dn^3$ jumps from $+11.6t \to -91.6t$ and $d^2C_N/dn^2$ jumps from $+0.23 \to -1.92$). When SC gaps these fillings, both observables smear smoothly (`Fig. 14`).
   - **Superconducting Pairing QPTs ($n_{c,1}, n_{c,2}$):** Because 2D BCS pairing opens with an essential singularity $\Delta_{\mathrm{FS}} \sim \Lambda \exp[-c_1/(\lambda_+ - 1)]$, **both** $\Delta F = e_{\mathrm{SC}} - e_N$ and $\Delta C = C_{\mathrm{SC}} - C_N$ are $C^\infty$-flat right at $n_c^+$ (e.g., at $U=115$, $|\Delta F| < 10^{-8}t$ and $|C_{\mathrm{SC}} - C_N| < 10^{-5}$ all the way to $n = 1.52 = n_c + 0.06$, `Fig. 14c,g`). Moreover, raw $C(n)$ suffers from steep normal background $dC_N/dn$, strong basis sensitivity (`spin` vs `band` ESD at $n = 0.0325$), and distance-dependent Wootters entanglement sudden-death (ESD) kinks at $n_{\mathrm{ESD}}(r) \in \{0.416, 0.737, 0.803, 1.537\}$ (`Fig. 13d`).
   - **Conclusion:** Eq. (28) $e_{\mathrm{GS}}(n)$ (with $\Delta F$ and $d^2e_{\mathrm{GS}}/dn^2 = d\mu/dn$) is an unambiguous, $r$-independent, basis-independent thermodynamic QPT probe across $n$, whereas concurrence $C(n)$ only detects normal van Hove/Dirac restructurings at $N_k \ge 401\text{–}801$ and $r \le 0.35a$ and cannot reliably isolate $n_c$ without normal-baseline subtraction.

---

## 1. Exact Analysis of Equation (28) in `arXiv:1801.08988`

In Hutchinson, Hirsch, & Marsiglio (*Phys. Rev. B* **97**, 184513 (2018), Sec. III), the mean-field Hamiltonian in the helicity basis is diagonalized into $H_{\mathrm{MF}} = \sum_{\mathbf{k},s} E_{\mathbf{k}s}\hat{\alpha}_{\mathbf{k}s}^\dagger\hat{\alpha}_{\mathbf{k}s} + E_g$ (`Eq. (27)`), where **Eq. (28)** defines the $T=0$ grand-canonical energy:
$$
E_g = \frac{1}{2}\sum_{\mathbf{k},s}\left[(\epsilon_{\mathbf{k}s} - \mu) - E_{\mathbf{k}s} + \bar{\Delta}_{\mathbf{k}s} b_{\mathbf{k}s}^*\right],
\qquad
b_{\mathbf{k}s} = -u_{\mathbf{k}s}^* v_{\mathbf{k}s} = \frac{\bar{\Delta}_{\mathbf{k}s}}{2E_{\mathbf{k}s}} \;\;(\text{Eqs. (26), (29)}).
$$
Notice two essential points:
1. **No extra $-\Delta^2/U$ term:** The mean-field constant $+\frac{1}{2}\sum_{\mathbf{k},s}\bar{\Delta}_{\mathbf{k}s} b_{\mathbf{k}s}^* = +\frac{1}{2}\sum_{\mathbf{k},s}\frac{|\bar{\Delta}_{\mathbf{k}s}|^2}{2E_{\mathbf{k}s}}$ already cancels the interaction double counting of the kinetic expectation value. Adding any external $\Delta^2/U$ term would violate self-consistency.
2. **Grand-canonical vs. canonical density-$n$ energy:** Because $E_g = \langle H - \mu\hat{N}\rangle_0 = \Omega_{\mathrm{GS}}(\mu)$ satisfies $\partial \Omega_{\mathrm{GS}}/\partial \mu = -N_{\mathrm{elec}}$ (verified directly by `dphi_id_err <= 2.26e-10` across all 600 full-band points), fixed-density ground-state comparisons require adding $+\mu n$ per site:
   $$
   e_{\mathrm{GS}}(n) = \frac{E_g}{N} + \mu n, \qquad \frac{de_{\mathrm{GS}}}{dn} = \mu(n), \qquad \frac{d^2 e_{\mathrm{GS}}}{dn^2} = \frac{d\mu}{dn} = \frac{1}{n^2\kappa(n)}.
   $$

---

## 2. Full-Band Density ($n$) Phase Map & Critical Points ($N_k = 801$)

Using `lambda_+(n) = \Delta_t^2 \langle |\epsilon - \mu_N|\rangle_{\mathrm{BZ}} / (U - 4\Delta_t \mu_N) = 1` and sorting the $N_k=801$ periodic dispersion gives all pairing and Lifshitz critical densities (`\Delta_t = 4.5t`):

| $V_{\mathrm{SO}}/t$ | $U_c(n\to 0^+)/t$ | $n_{\mathrm{sad},-}$ | $n_{\mathrm{vH},-}$ ($\mu_N=-E_{\mathrm{vH}}$) | $n_{\mathrm{Dirac},0}$ ($\mu_N=0$) | $n_{\mathrm{vH},+}$ ($\mu_N=+E_{\mathrm{vH}}$) | $n_{\mathrm{Dirac},4}$ ($\mu_N=4t$) | $n_c$ at $U=75t$ | $n_c$ at $U=90t$ | $n_c$ at $U=115t$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.0** | 89.999 | 0.00000 | 1.00000 | 1.00000 | 1.00000 | 2.00000 | **0.1254, 1.1966** | 1.3622 | 1.5421 |
| **0.4** | 93.531 | 0.00404 | 0.93100 | 1.00000 | 1.06900 | 1.95280 | **0.1855, 1.0957** | 0.0487, 1.2975 | 1.4890 |
| **0.5** | 95.459 | 0.00743 | **0.90648** | **1.00000** | **1.09347** | **1.92900** | **0.2192, 1.0194** | 0.0746, 1.2594 | **1.4602** |
| **0.7** | 100.422 | 0.01764 | 0.86111 | 1.00000 | 1.13898 | 1.87329 | **0.3162, 0.8713** | 0.1398, 1.1256 | 1.3857 |

### Dilute Re-entrant Nodal SC Pocket ($n < n_{c,1}$) vs. High-$n$ SC Phase ($n > n_{c,2}$)
At $U=75t, \Delta_t=4.5t, V_{\mathrm{SO}}=0.5t$, $\lambda_+(n)$ is non-monotonic (`Fig. 13a`): it reaches a minimum $\lambda_{\min} = 0.8758$ near $n=0.60$, rises above $1$ at high filling ($n > n_{c,2} = 1.0194$, reaching $\min \Delta F = -0.06765t$ at $n=1.77$), **and also exceeds $1$ at low filling ($n < n_{c,1} = 0.2192$, reaching $\min \Delta F = -7.69\times 10^{-5}t$ at $n=0.015$)** via sign-changing nodal $s_\pm$ pairing ($\Delta_0 = +0.125t, \Delta_S = -0.139t, \Delta_{\mathrm{FS}} = -0.0226t$). Crucially, in the ultra-dilute BEC-like corner ($n \le 0.04$, where $|\Delta_{\mathrm{FS}}| \sim E_F$), anomalous pairing $F_{AB}$ **boosts** nearest-neighbor lattice concurrence $C_{\mathrm{SC}}(r=a)$ from $C_N = 0.8409$ (and $C_{\mathrm{norm\text{-}only}} = 0.6282$) up to **$C_{\mathrm{SC}} = 0.9636$** at $n=0.004$ (`Fig. 13e`), opposite to the weak Fermi-smearing reduction observed everywhere above $n > 1.0$.

---

## 3. Head-to-Head Comparison Across All Density-Driven QPT Types (`Fig. 14`)

1. **Normal Logarithmic van Hove QPTs ($n_{\mathrm{vH},-} = 0.9065$ at $V=0.5$; $n_{\mathrm{vH}} = 1.000$ at $V=0$):**
   - **Eq. (28):** $d^2 e_N/dn^2 = d\mu_N/dn = 1/g(\mu_N)$ pinches logarithmically downward to $1.600t$ at $n=0.905$ ($V=0.5$) and $1.216t$ at $n=1.000$ ($V=0$) (`Fig. 14a,b`).
   - **Concurrence ($N_k=801, r=0.25a$):** Because $\frac{dG_N}{dn} = \langle e^{i\mathbf{k}\cdot\mathbf{r}}\rangle_{\mathrm{FS}}$ becomes dominated by the van Hove saddle momentum, $dC_N/dn$ exhibits a genuine continuum cusp reaching $-0.3561$ at $n=0.905$ (`-0.3527` at $n=1.000$ for $V=0$), with $d^2 C_N/dn^2$ swinging cleanly from $-4.00 \to +4.07$ (`-1.66 \to +1.96` for $V=0$) (`Fig. 14e,f`). Comparing `U=115` (Normal at $n_{\mathrm{vH},+} = 1.0935$) vs. `U=75` (SC at $n=1.0935$) shows SC pairing completely smears the upper van Hove cusp.
2. **Superconducting Pairing Onset ($n_c = 1.0194$ at $U=75$; $n_c = 1.4602$ at $U=115$):**
   - Both Eq. (28) $\Delta F, d\mu/dn$ and concurrence $C(n)$ are **exponentially flat** at $n_c^+$ due to the 2D BCS essential singularity (`Fig. 14c,g`). At $U=115$, $|\Delta_{\mathrm{FS}}| \le 0.0033t$ and $|C_{\mathrm{SC}} - C_N| < 10^{-5}$ all the way to $n = 1.54$, so $d^2C/dn^2$ stays buried inside $\pm 0.28$ discrete-grid jitter. Even at $U=75$, the peak of $d^2C_{\mathrm{SC}}/dn^2 = -6.21$ occurs at $n = 1.040$ — shifted $+0.021$ into the SC dome rather than at $n_{c,2} = 1.0194$.
3. **Normal $(\pi,\pi)$ Dirac–Lifshitz QPT ($n_{\mathrm{Dirac},4} = 1.9290$, $\mu_N = 4t$):**
   - At $U=300t$ (Normal, $T=10^{-3}t$), Eq. (28) $d^3 e_N/dn^3$ jumps discontinuously from $+11.56t$ ($n=1.922$) to $-91.63t$ ($n=1.943$), confirming 3rd-order Ehrenfest scaling, and $d^2 C_N/dn^2$ steps from $+0.230 \to -1.918$ (`Fig. 14d,h`). At $U=75t$ (SC, $\min E \ge 0.94t$), both observables remain $C^\infty$-smooth ($d^2C_{\mathrm{SC}}/dn^2 \approx -0.10$).
4. **Spurious Indicators — Distance-Dependent ESD Kinks & Finite-$N_k$ Noise (`Fig. 13d,f`):**
   - Wootters clipping $C = \max(0, C_{\mathrm{raw}})$ produces sharp slope discontinuities at $n_{\mathrm{ESD}}(r) = 0.4162$ ($r=1.0a$), $0.7368$ ($r=0.75a$), $0.8033$ ($r=(0.5,0.5)a$), and $1.5366$ ($r=0.50a$) in `spin` basis (`0.0325` and `0.0618` extra death/revival points in `band` basis), shifting rapidly with $r$ and proving ESD kinks are not QPTs.
   - Coarse grids (`Nk=101, 201`) generate $\pm 15\text{–}23$ background oscillations in $d^2C/dn^2$ (`std=7.49` at `Nk=101`, `3.70` at `Nk=201` vs `0.57` at `Nk=801` in `[0.855, 0.885]`), masking true van Hove inflections unless $N_k \ge 401\text{–}801$.
