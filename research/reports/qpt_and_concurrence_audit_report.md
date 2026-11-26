# Rigorous Audit of Quantum Phase Transitions, Global Phase Structure, and Two-Spin Concurrence in the Rashba Correlated-Hopping BdG Model

**Repository:** `AmirhoseynpowAsghari/quantum-teleportation-and-dense-coding-in-open-quantum-systems`  
**Audit Scope:** Tasks 1–8 numerical revalidation, analytical pairing/Landau theory, two-qubit concurrence decomposition, $n = 1.875$ manuscript audit, and Brillouin-zone discretization benchmarks  
**Primary Artifacts:** `research/data/*.json`, `research/data/csv/*.csv` (11 curated tables, 2.0 MB), `research/figs/fig01`–`fig12` (36 PNG/PDF/SVG files, 8.4 MB)

---

## Executive Summary

We carried out an end-to-end analytical and numerical audit of the zero-temperature ($T=0$) Bogoliubov–de Gennes (BdG) phase structure and conditional two-qubit Wootters concurrence of the two-dimensional square-lattice correlated-hopping + Rashba spin–orbit model (Hutchinson, Hirsch, & Marsiglio, *Phys. Rev. B* **97**, 184513 (2018), `10.1103/PhysRevB.97.184513`) at canonical defaults $t = 1$, $U/t = 75$, $\Delta_t / t = 4.5$. Every prior quantum phase transition (QPT) claim in the repository (`reports/*.json`, `analysis/*.py`) and manuscript (`manuscript.tex` Secs. `sec:vso_concurrence` and `sec:qpt`) was tested on periodic (`endpoint=False`) and repository (`endpoint=True`) grids up to $N_k = 801 \times 801$ ($\Delta V_{\mathrm{SO}} = 10^{-4}t$, machine-precision residuals $\le 10^{-14}$).

### Summary Verdict Table

| Claim / Parameter Point | Prior Claim (Repo / `manuscript.tex`) | Audit Verdict | Exact Continuum Result & Physical Mechanism |
| :--- | :--- | :--- | :--- |
| **$(n=1.0,\, V_{\mathrm{SO}}=1.235t)$** | Candidate 1st-order transition | **Falsified** | Deep inside single-valued SC phase ($N_b^{\max}=1$, $\Delta F = -0.1053t$); $\Delta V = 10^{-4}t$ up/down hysteresis $\le 8.88\times 10^{-16}$. True continuous onset is at $V_c(1.0) = 0.5322t$ (`Fig. 1a,c`, `Fig. 5a`). |
| **$(n=1.4,\, V_{\mathrm{SO}}=0.525t)$** | Candidate 1st-order transition | **Falsified** | Deep inside single-valued SC phase ($N_b^{\max}=1$, $\Delta F = -0.02505t$); $\Delta V = 10^{-4}t$ up/down hysteresis $\le 8.88\times 10^{-16}$; strictly positive compressibility $\min(d\mu/dn) \ge 1.21t$ (`Fig. 1b,d`). |
| **Repo `qpt_parameter_scan.json` hysteresis ($0.96$–$14.92$)** | Large hysteresis in $V_{\mathrm{SO}}$ and $U$ sweeps | **Software Bug** | Stale JSON generated before `[::-1]` patch: `down` stored descending and compared element-wise without reversal. Reversal reduces `_gdist` to $1.33\times 10^{-15}$–$9.40\times 10^{-12}$ (`Fig. 2a`). |
| **$n=1.875$ "Lifshitz" at $V_{\mathrm{SO}}=0.5487t$, $U=118.04t$ (`manuscript.tex` L1257–1300)** | Inner pocket empties ($\mu = 4t$), 2nd-order Ehrenfest transition rounded by SC | **Conflation & Order Error** | Values $(0.54875t, 118.08t)$ are $\mu_{\mathrm{SC}} = 4t$ on $N_k=201$ `endpoint=True` (`0.56187t, 111.71t` continuum), not $\mu_N=4t$. True normal Dirac-cone Lifshitz ($\mu_N=4t$) is at $V_L = 0.6942t$, is **3rd-order** ($d^3 e_N/dV^3$ jump $+0.88 \to -0.74$), and is completely smeared in SC ($\min E_{\mathbf{k}} \ge 0.94t$) (`Fig. 10`, `Fig. 11`). |
| **$n=1.875$, $V_{\mathrm{SO}}=0.97t$ (`manuscript.tex` L1277)** | "Pairing onset at $V_{\mathrm{SO}}/t = 0.97$" | **Contradicted** | $\lambda_+ > 1$ across the entire sweep ($\lambda_+(0) = 8.13$, $|\Delta_S(0)| = 1.49t$); $0.97t$ is merely the smooth inflection peak of $d|\Delta_S|/dV_{\mathrm{SO}}$ ($\mathrm{FWHM} = 1.19t$). True $\Delta_t$ threshold at $V=0$ is $\Delta_{t,c} = 2.527t$ (`2.348t` at $V=0.5t$). |
| **Global SC $\leftrightarrow$ Normal boundary ($\lambda_+ = 1$)** | Uncharacterized / power-law assumption | **Confirmed Continuous Essential Singularity** | Unique global phase boundary $\lambda_+ = \frac{\Delta_t^2 \langle|\epsilon - \mu_N|\rangle_{\mathrm{BZ}}}{U - 4\Delta_t\mu_N} = 1$. At $V=0.50t$, $n_c = 1.0194 \pm 0.0005$. BCS gap obeys $|\Delta_{\mathrm{FS}}| \sim \Lambda \exp[-c_1/(\lambda_+ - 1)]$, not power-law $\beta=0.5$ (`Fig. 3`, `Fig. 4`, `Fig. 7`). |
| **Concurrence $C(r)$ as a QPT detector** | $dC/d\theta$ extrema / peaks mark Lifshitz or transitions | **Refuted as Universal Probe** | $C(r=0.25a)$ is $>95\%$ normal-state Fermi-exchange background; SC reduction is Fermi-step smearing ($|C - C_{\mathrm{norm\text{-}only}}| \le 0.015$ vs $|C - C_N| \le 0.079$). Fine-grid $d^2C/dn^2$ ($\pm 20$) and $d^2C/dV^2$ spikes are discrete-FS grid noise and Wootters $\max(0, C_{\mathrm{raw}})$ sudden-death kinks (`Fig. 8`, `Fig. 9`, `Fig. 12c`). |

---

## 1. Microscopic Model, BdG Equations, and Exact Pairing Kernel

### 1.1 Hamiltonian and Self-Consistency Formulation
Following `kspace_setup.py`, `greens_functions.py`, `self_consistency.py`, and Hutchinson, Hirsch, & Marsiglio (*Phys. Rev. B* **97**, 184513 (2018)), the two helicity bands on an $N_k \times N_k$ 2D square-lattice Brillouin zone are
$$
\epsilon_s(\mathbf{k}) = -2t(\cos k_x + \cos k_y) - 2 s V_{\mathrm{SO}} \sqrt{\sin^2 k_x + \sin^2 k_y}, \qquad s \in \{+1, -1\},
$$
with correlated-hopping pairing kernel (`self_consistency.py`, `research/fast_solver.py` L16–20)
$$
K_s(\mathbf{k}) = U + 4\Delta_t (\cos k_x + \cos k_y) + 4 \frac{\Delta_t}{t} V_{\mathrm{SO}} s' \sqrt{\sin^2 k_x + \sin^2 k_y} \equiv U - 2 \frac{\Delta_t}{t} \epsilon_s(\mathbf{k}), \quad (s' \equiv -s),
$$
where `eps[..., 0]` corresponds to $s=+1$ ($s'=-1$) and `eps[..., 1]` corresponds to $s=-1$ ($s'=+1$). Setting $t=1$, the BdG gap function depends on momentum and helicity purely through $\epsilon_s(\mathbf{k})$:
$$
\Delta_s(\mathbf{k}) = \Delta_0 - \frac{\Delta_S}{4}\,\epsilon_s(\mathbf{k}), \qquad \xi_s(\mathbf{k}) = \epsilon_s(\mathbf{k}) - \mu, \qquad E_s(\mathbf{k}) = \sqrt{\Delta_s(\mathbf{k})^2 + \xi_s(\mathbf{k})^2}.
$$
At $T=0$, the three coupled self-consistency residuals solved to $\max|r_i| \le 10^{-14}$ (`research/fast_solver.py` L81–91) are
$$
\begin{aligned}
r_1 &= \Delta_0 + \frac{1}{4 N_k^2} \sum_{\mathbf{k},s} K_s(\mathbf{k}) \frac{\Delta_s(\mathbf{k})}{E_s(\mathbf{k})} = 0, \\
r_2 &= \Delta_S + \frac{2\Delta_t}{N_k^2} \sum_{\mathbf{k},s} \frac{\Delta_s(\mathbf{k})}{E_s(\mathbf{k})} = 0, \\
r_3 &= \frac{1}{2 N_k^2} \sum_{\mathbf{k},s} \frac{\xi_s(\mathbf{k})}{E_s(\mathbf{k})} - (1 - n) = 0.
\end{aligned}
$$
Under the global $\mathbb{Z}_2$ gauge symmetry $(\Delta_0, \Delta_S, \mu) \leftrightarrow (-\Delta_0, -\Delta_S, \mu)$, we canonicalize solutions by $\Delta_0 \ge 0$ (retaining the physically meaningful relative sign $\Delta_0 \cdot \mathrm{sgn}(\Delta_S)$). Because $K_s(\mathbf{k}) = U - 2\Delta_t \epsilon_s(\mathbf{k})$ is an exact affine function of $\epsilon_s(\mathbf{k})$, substituting $r_2$ into $r_1$ relates the two gap parameters via $\Delta_{\mathrm{FS}} \equiv \Delta_0 - \frac{\Delta_S}{4}\mu_N$.

### 1.2 Exact Analytic Instability Eigenvalue $\lambda_+$
Linearizing $(r_1, r_2)$ around the normal state $(\Delta_0, \Delta_S) \to (0,0)$ at fixed $\mu = \mu_N(n, V_{\mathrm{SO}})$ yields the $2 \times 2$ linearized pairing matrix $M$ (`research/fast_solver.py` L142–163). Define the Brillouin-zone moments summed over both helicity bands $s \in \{+1,-1\}$:
$$
I_0 = \frac{1}{N_k^2}\sum_{\mathbf{k},s} \frac{1}{|\epsilon_s(\mathbf{k}) - \mu_N|}, \qquad J_1 = \frac{1}{N_k^2}\sum_{\mathbf{k},s} \mathrm{sgn}(\epsilon_s(\mathbf{k}) - \mu_N) = 2(1 - n), \qquad J_2 \equiv \langle |\epsilon - \mu_N| \rangle_{\mathrm{BZ}} = \frac{1}{N_k^2}\sum_{\mathbf{k},s} |\epsilon_s(\mathbf{k}) - \mu_N|.
$$
Expanding $\epsilon_s(\mathbf{k}) = \mu_N + \xi_s(\mathbf{k})$ in the characteristic equation $\det(M - \lambda I) = 0$ isolates the leading divergent 2D Cooper-logarithm coefficient $I_0 \sim N(\mu_N)\ln(\Lambda/\delta k) \to +\infty$:
$$
\det(M - \lambda I) = \frac{I_0}{4}\Big[(U - 4\Delta_t \mu_N)\,\lambda - \Delta_t^2 J_2\Big] + \mathcal{O}(I_0^0) = 0.
$$
In the thermodynamic limit $I_0 \to \infty$, the $\mathcal{O}(I_0)$ bracket must vanish for the nontrivial eigenvalue $\lambda_+$ governing $\Delta_{\mathrm{FS}} = \Delta_0 - \frac{\Delta_S}{4}\mu_N \neq 0$ (`Fig. 3`):
$$
\boxed{\lambda_+(n, V_{\mathrm{SO}}, U, \Delta_t) = \frac{\Delta_t^2 \langle |\epsilon - \mu_N| \rangle_{\mathrm{BZ}}}{U - 4\Delta_t \mu_N}} \qquad (U > 4\Delta_t \mu_N),
$$
with $\lambda_+ = +\infty$ (unconditionally unstable even for infinitesimal $\Delta_t$) once $U \le 4\Delta_t \mu_N$. Setting $\lambda_+ = 1$ yields the exact critical repulsion curve
$$
U_c(n, V_{\mathrm{SO}}, \Delta_t) = 4\Delta_t \mu_N(n, V_{\mathrm{SO}}) + \Delta_t^2 \langle |\epsilon - \mu_N| \rangle_{\mathrm{BZ}}.
$$
Superconductivity exists if and only if $U < U_c$ (i.e., $\lambda_+ > 1$). On the square lattice, particle–hole symmetry guarantees $\mu_N(n=1.0, V_{\mathrm{SO}}) = 0$ identically for all $V_{\mathrm{SO}}$, where $U_c(1.0, V_{\mathrm{SO}}) = \Delta_t^2 \langle|\epsilon|\rangle$ rises monotonically with $V_{\mathrm{SO}}$ from $65.659t$ ($V_{\mathrm{SO}}=0$) to $73.924t$ ($0.5t$), $81.583t$ ($0.7t$), and $91.190t$ ($0.9t$) at $\Delta_t = 4.5t$ (`Fig. 3c`). Thus, spin–orbit coupling **monotonically enhances pairing** and expands the superconducting domain by raising $U_c(n, V_{\mathrm{SO}})$.

---

## 2. Task 1: Refutation of Candidate First-Order Transitions and Audit of Repository Bugs

### 2.1 Falsification of Candidate First-Order Points
Two candidate parameter points previously flagged as potential first-order transitions at $U/t=75, \Delta_t/t=4.5$ — **(1) $(n=1.0, V_{\mathrm{SO}}=1.235t)$** and **(2) $(n=1.4, V_{\mathrm{SO}}=0.525t)$** — were tested with multi-seed enumeration (`18` distinct initial guesses spanning $\Delta_0 \in [-0.5, 1.5]$, $\Delta_S \in [-4.0, 4.0]$) plus ascending/descending continuation (`research/task1_candidates.py`, `research/task1_deep.py`, `research/data/csv/task1_adaptive_hysteresis.csv`, `Fig. 1`):

1. **Coarse & Ultra-Fine Hysteresis (`N_k = 201, 401`):**
   - At $n=1.0$: wide continuation over $V_{\mathrm{SO}}/t \in [0.0, 2.5]$ (`51` pts) and adaptive sweeps around $1.235t$ with $\Delta V = 10^{-3}t$ ($V \in [1.22, 1.25]$) and $\Delta V = 10^{-4}t$ ($V \in [1.233, 1.237]$) yield **maximum branch multiplicity $N_b^{\max} = 1$** (`0` metastable branches) and up/down differences at IEEE-754 floating-point noise:
     - `n1.0_Nk201` ($V \in [0, 2.0]$): $\max|\Delta F_{\uparrow} - \Delta F_{\downarrow}| = 1.332\times 10^{-15}t$, $\max|\Delta_{0,\uparrow} - \Delta_{0,\downarrow}| = 8.882\times 10^{-16}t$.
     - `n1.0_dV0.0001_Nk401` ($V \in [1.233, 1.237]$): $\max|\Delta F_{\uparrow} - \Delta F_{\downarrow}| = 8.882\times 10^{-16}t$, $\max|\Delta_{0,\uparrow} - \Delta_{0,\downarrow}| = 4.441\times 10^{-16}t$, $\max|\Delta_{S,\uparrow} - \Delta_{S,\downarrow}| = 1.776\times 10^{-15}t$.
   - At $n=1.4$: wide sweep $V_{\mathrm{SO}}/t \in [0.0, 2.5]$ and adaptive sweeps ($\Delta V = 10^{-3}t$ on $[0.51, 0.54]$, $\Delta V = 10^{-4}t$ on $[0.523, 0.527]$) likewise give **$N_b^{\max} = 1$** and zero hysteresis:
     - `n1.4_Nk401` ($V \in [0, 2.0]$): $\max|\Delta F_{\uparrow} - \Delta F_{\downarrow}| = 8.882\times 10^{-16}t$, $\max|\Delta_{0,\uparrow} - \Delta_{0,\downarrow}| = 1.110\times 10^{-16}t$.
     - `n1.4_dV0.0001_Nk401`: $\max|\Delta F_{\uparrow} - \Delta F_{\downarrow}| = 6.661\times 10^{-16}t$, $\max|\Delta_{0,\uparrow} - \Delta_{0,\downarrow}| = 8.327\times 10^{-17}t$, $\max|\Delta_{S,\uparrow} - \Delta_{S,\downarrow}| = 1.776\times 10^{-15}t$.
2. **Thermodynamic Compressibility & Phase Separation Check (`research/data/csv/task1_compressibility.csv`, `Fig. 1c,d`):**  
   Density sweeps across $n \in [0.10, 1.95]$ for $V_{\mathrm{SO}}/t \in \{0.0, 0.5, 0.7, 1.0, 1.235, 1.4\}$ at $N_k = 201, 401$ show strictly positive compressibility everywhere (`min(dμ/dn) > 0`, `min(d²e/dn²) > 0`), ruling out phase separation (Maxwell construction):
   - $V_{\mathrm{SO}}=0.0t$ ($N_k=201$): $\min(d\mu/dn) = +1.2106t$, $\min(d^2 e/dn^2) = +1.4312t$.
   - $V_{\mathrm{SO}}=0.5t$ ($N_k=401$): $\min(d\mu/dn) = +1.8367t$, $\min(d^2 e/dn^2) = +1.8775t$, thermodynamic consistency $\max|de/dn - \mu| = 1.98\times 10^{-3}t$.
   - $V_{\mathrm{SO}}=1.0t$ ($N_k=401$): $\min(d\mu/dn) = +2.2779t$, $\min(d^2 e/dn^2) = +2.3663t$, $\max|de/dn - \mu| = 2.38\times 10^{-3}t$.
   - $V_{\mathrm{SO}}=1.235t$ ($N_k=201$): $\min(d\mu/dn) = +2.4329t$, $\min(d^2 e/dn^2) = +2.5673t$.
   - $V_{\mathrm{SO}}=1.4t$ ($N_k=201$): $\min(d\mu/dn) = +2.6576t$, $\min(d^2 e/dn^2) = +2.7477t$.

Both $(1.0, 1.235t)$ and $(1.4, 0.525t)$ are deep in the single-valued superconducting phase ($\Delta F = -0.10529t$ and $-0.02505t$, respectively) and exhibit zero thermodynamic instability.

### 2.2 Diagnosis of Three Repository Artifacts (`Fig. 2`)
Our forensic code audit identified three distinct numerical artifacts in the existing repository scripts and cached reports:

1. **Unreversed Hysteresis Indexing Bug in `reports/qpt_parameter_scan.json` (`Fig. 2a`):**  
   The committed `reports/qpt_parameter_scan.json` reports massive apparent hysteresis (`1.200` for `U90_Dt4.5_n1.875`, `0.961` for `U75_Dt3.5_n1.875`, `1.154` for `U75_Dt4.0_n1.875`, `8.787` for `V0_Dt4.5_n1.875`, `13.470` for `V0.7_Dt4.5_n1.875`, `14.918` for `V1.4_Dt4.5_n1.875`, and `5.114` for `V1.0_Dt3.5_n1.875`). Although `analysis/qpt_parameter_scan.py` L137 was later patched with `fam["down"][::-1]`, the JSON file on disk was never regenerated and still contains unreversed `zip(up, down)` comparisons of $V_{\mathrm{up}}[i]$ against $V_{\mathrm{down}}[i] = V_{\max} - \Delta V \cdot i$. Re-evaluating `_gdist(a["x"], b["x"])` on `fam["up"]` vs `fam["down"][::-1]` directly on the stored JSON vectors reduces every single hysteresis value to **$1.33\times 10^{-15}$–$9.40\times 10^{-12}$** (zero hysteresis across all 12 parameter families).
2. **Zone-Edge Double-Counting Bias (`endpoint=True`) in `kspace_setup.py` (`Fig. 2b,c`):**  
   `kspace_setup.py` L19–20 constructs the Brillouin zone using `np.linspace(-np.pi, np.pi, Nk)` (`endpoint=True`). Because $k_x = +\pi$ and $k_x = -\pi$ are the exact same point on the 2-torus $\mathbb{T}^2$, `endpoint=True` double-counts the entire zone boundary ($4N_k - 4$ states) with relative weight $\mathcal{O}(1/N_k)$. At $N_k=100$ (`reports/n_density_scan.json`), this shifts $\mu_N(n=1.0)$ from $0$ to $+0.04000t$ and inflates $\lambda_+(1.0, 0.5)$ from $0.9860$ to $0.98976$. At $(n=1.0, V_{\mathrm{SO}}=0.52t)$, where the continuum $\lambda_+ = 0.9945 < 1$ is strictly Normal, `endpoint=True` generates a **spurious finite gap** $|\Delta_S| = 0.0414t$ at $N_k=101$ and $0.0015t$ at $N_k=201$ that vanishes for $N_k \ge 301$ and never exists on periodic `endpoint=False` grids (`Fig. 2b`).
3. **$\mu_{\mathrm{SC}}$ Plugged into Bare Dispersion $\epsilon_0$ in `analysis/test_qpt_diagnostics.py` L221–235:**  
   Detailed in Section 6 (`Task 6`).

---

## 3. Task 2: Critical Scaling and Essential Singularity at the Pairing Threshold

### 3.1 Continuum Convergence of the Threshold $n_c$ and $V_c(n)$
Solving $\lambda_+(n, V_{\mathrm{SO}}) = 1$ via Brent root-finding (`research/data/lambda_convergence.json`, `research/data/task2_analysis.json`, `Fig. 3b`, `Fig. 4a`) yields:
- **At $V_{\mathrm{SO}} = 0.50t$ ($U/t = 75, \Delta_t/t = 4.5$):**
  - Repository `endpoint=True`: $n_c(201) = 1.01163 \to n_c(401) = 1.01540 \to n_c(801) = 1.01740$ (slow $\mathcal{O}(1/N_k)$ drift from boundary double-counting).
  - Periodic `endpoint=False`: $n_c(201) = 1.02005 \to n_c(401) = 1.01891 \to n_c(801) = \mathbf{1.01944}$ (**continuum limit $n_c = 1.0194 \pm 0.0005$**).
- **Other $V_{\mathrm{SO}}$ slices (periodic $N_k = 401$):** $n_c(0.0) = 1.19626$, $n_c(0.45t) = 1.05315$, $n_c(0.55t) = 0.98887$, $n_c(0.70t) = 0.87121$.
- **Critical Rashba coupling $V_c(n)$ (periodic $N_k = 401$/$801$):**
  $V_c(0.60) = 0.87348t$, $V_c(0.70) = 0.83041t$, $V_c(0.75) = 0.79601t$, $V_c(0.80) = 0.75641t$, $V_c(0.90) = 0.67271t$, $V_c(1.00) = 0.53218t$, $V_c(1.10) = 0.39329t$.

### 3.2 Proof of Essential Singularity Rather Than Mean-Field Power Law
Unlike an analytic Ginzburg–Landau second-order transition where $|\Delta| \propto (n - n_c)^{1/2}$ ($\beta = 1/2$), the $T=0$ BCS gap at the Fermi surface $\Delta_{\mathrm{FS}} \equiv \Delta_0 - \frac{\Delta_S}{4}\mu$ opens with an **essential singularity** controlled by the 2D Cooper logarithm $\ln(\Lambda / |\Delta_{\mathrm{FS}}|)$:
$$
\lambda_+ - 1 = \frac{c_1}{\ln(2t / |\Delta_{\mathrm{FS}}|)} + c_2 \Delta_S^2 \quad \Longleftrightarrow \quad |\Delta_{\mathrm{FS}}| \simeq 2t \exp\!\left[-\frac{c_1}{(\lambda_+ - 1) - c_2 \Delta_S^2}\right].
$$
Because `Δ_FS` is exponentially suppressed near threshold while the self-consistent $\Delta_S$ inherits a linear mixing proportional to $\Delta_{\mathrm{FS}}$ plus higher-order $\mathcal{O}(\Delta_S^2)$ feedback, fitting a naive power law $|\Delta_S| = A(n - n_c)^\beta$ (`research/data/csv/task2_window_fits.csv`, `Fig. 4c`) produces severe **fit-window drift** rather than $\beta = 0.5$:
- At $V_{\mathrm{SO}} = 0.50t$ ($N_k = 801$ periodic): $\beta_{\mathrm{eff}}$ drifts continuously from **$1.65 \pm 0.68$** ($n_{\max}=1.06$) $\to$ **$1.26 \pm 0.32$** ($1.08$) $\to$ **$1.03 \pm 0.16$** ($1.10$) $\to$ **$0.89 \pm 0.09$** ($1.12$) $\to$ **$0.78 \pm 0.05$** ($1.15$).
- At $V_{\mathrm{SO}} = 0.45t$ ($N_k = 401$): $\beta_{\mathrm{eff}}$ drifts from **$1.84 \pm 0.93$** to **$1.19 \pm 0.12$**.
- At $V_{\mathrm{SO}} = 0.55t$ ($N_k = 401$): $\beta_{\mathrm{eff}}$ drifts from **$1.37 \pm 0.52$** to **$0.72 \pm 0.04$**.

Conversely, the coupled essential-singularity law $(\lambda_+ - 1) - c_2 \Delta_S^2 = c_1 / \ln(2t/|\Delta_{\mathrm{FS}}|)$ passes cleanly through the origin $(0,0)$ (`Fig. 4b`) with near-exact determination across all three cuts:
- $V_{\mathrm{SO}} = 0.45t$: $c_1 = 0.09446$, $c_2 = 0.08345$, **$R^2 = 0.99889$**.
- $V_{\mathrm{SO}} = 0.50t$ ($N_k=801$): $c_1 = 0.06900$, $c_2 = 0.06544$, **$R^2 = 0.99055$**.
- $V_{\mathrm{SO}} = 0.55t$: $c_1 = 0.06523$, $c_2 = 0.05497$, **$R^2 = 0.99959$**.

(On $N_k = 801$ at $V=0.50t$, the slight dip near $1/\ln(2/|\Delta_{\mathrm{FS}}|) \approx 0.25$ corresponds to $|\Delta_{\mathrm{FS}}| \sim 3.6\times 10^{-2}t$ crossing the finite $N_k=801$ $k$-spacing scale $\delta\epsilon \sim 8t\pi / N_k \approx 0.03t$, confirming the discrete-grid origin of that bump.)

---

## 4. Tasks 3–4: Global Phase Diagram & Exact Landau Free-Energy Landscape

### 4.1 Global Uniqueness Across $(n, V_{\mathrm{SO}})$, $(V_{\mathrm{SO}}, U)$, $(n, U)$, and $\Delta_t$
We mapped the full phase structure across `research/data/phasediagram_Nk201.json` (`504` adaptive points, all `35` high-$\Delta_S$ initial misses repaired with `fast_solver.py` to `0` misses), `research/data/phasediagram_Nk401_per.json` (`354` periodic $N_k=401$ points, `224` SC / `130` Normal, `0` misses), and `research/data/task3_4_slices.json` (`3835` continuum $N_k=301$ periodic map points plus SC verification slices in $(V_{\mathrm{SO}}, U)$, $(n, U)$, and $\Delta_t \in \{2.5, 3.5\}t$):

- **Zero branch multiplicity everywhere (`Fig. 5`, `Fig. 6`):** Across all `504` points at $N_k=201$, all `354` points at $N_k=401$, and all `256` SC verification points across $(V_{\mathrm{SO}}, U)$, $(n, U)$, and $\Delta_t \in \{2.5, 3.5\}t$, **$N_b^{\max} = 1$** with **zero metastable branches** and **zero positive-$\Delta F$ SC solutions** (the single tiny $+2.76\times 10^{-9}t$ numerical artifact at $(n=1.6, U=140, V=0.5)$ at $N_k=201$ vanishes to $\Delta F < 0$ for periodic $N_k \ge 401$).
- **Two topological crossovers inside the SC phase (`Fig. 5`):**
  1. **Nodal sign-flip crossover ($\Delta_0 = 0$):** Along a smooth curve running from $(n=1.035, V_{\mathrm{SO}}=0.515t)$ through $(1.40, 0.99t)$ to $(1.90, 1.256t)$, the relative sign $\Delta_0 \cdot \mathrm{sgn}(\Delta_S)$ passes smoothly through zero (so the $k$-space gap nodal contour $\Delta(\mathbf{k})=0$ sweeps across $\epsilon = 0$ without any nonanalyticity in $e_{\mathrm{GS}}$ or $|\Delta_S|$).
  2. **Normal Dirac–Lifshitz line ($\mu_N = 4t$):** Running from $(n=2.0, V_{\mathrm{SO}}=0)$ through $(1.875, 0.6942t)$ to $(1.6009, 1.60t)$, this line lies entirely inside the fully gapped SC phase ($\Delta F < 0$, $|\Delta_S| \ge 1.5t$) and is smoothed into a broad crossover (`Section 6`).
- **Global $U$ threshold (`Fig. 6`):** Across $n \in [0.50, 1.95], V_{\mathrm{SO}} \in [0, 1.60]$, the minimum critical repulsion is $\min U_c = 56.165t$ (at low filling near $n=0.55, V=0$), so for $U/t \le 56$ the entire domain is superconducting, whereas for $U/t > 56$ a single continuous boundary $U = U_c(n, V_{\mathrm{SO}})$ separates the SC region ($U < U_c$) from the Normal state ($U > U_c$).

### 4.2 Exact 1D Constrained Landau Potential $F_{\mathrm{eff}}(\psi)$
To prove why **no tricritical point, critical end point (CEP), or first-order transition** can exist anywhere on the global phase boundary, we evaluated the exact constrained fixed-density variational free energy $F_{\mathrm{eff}}(\psi) - F_N$ as a function of the amplitude $\psi \equiv \Delta_S$ (minimizing over $\Delta_0$ and solving the number constraint $n(\Delta_0, \psi, \mu) = n$ at every $\psi$; validated against on-shell `sc_energy` to $-2.22\times 10^{-16}t$) at 14 boundary and interior points (`research/data/csv/task4_landau_potential.csv`, `Fig. 7`).

At $T=0$, the Cooper logarithm adds a strictly nonanalytic logarithmic term to the standard Landau expansion:
$$
\frac{F_{\mathrm{eff}}(\psi) - F_N}{\psi^2} = a + \frac{\kappa}{\ln(4t / |\psi|)} + b \psi^2,
$$
which outperforms the analytic polynomial $a_{\mathrm{p}} + b_{\mathrm{p}} \psi^2 + c_{\mathrm{p}} \psi^4$ in RMSE by **$2\times$ to $11\times$** across every tested point (`Fig. 7a,b`):
- $(n=0.95, V=0.5, U=75)$: $\mathrm{RMSE}_{\log} = 4.75\times 10^{-6}$ vs $\mathrm{RMSE}_{\mathrm{poly}} = 5.20\times 10^{-5}$ ($11\times$ improvement), $\kappa = +1.180\times 10^{-3} > 0$, $b_{\mathrm{poly}} = +2.84\times 10^{-3} > 0$.
- $(n=1.019, V=0.5, U=75)$ (right on $n_c$): $\mathrm{RMSE}_{\log} = 8.85\times 10^{-5}$ vs $\mathrm{RMSE}_{\mathrm{poly}} = 1.60\times 10^{-4}$, $\kappa = +2.622\times 10^{-3} > 0$, $b_{\mathrm{poly}} = +4.10\times 10^{-3} > 0$.
- $(n=1.05, V=0.5, U=75)$: $\mathrm{RMSE}_{\log} = 2.71\times 10^{-5}$ vs $\mathrm{RMSE}_{\mathrm{poly}} = 1.71\times 10^{-4}$, $\kappa = +3.989\times 10^{-3} > 0$, $b_{\mathrm{poly}} = +7.64\times 10^{-3} > 0$.
- $(n=1.40, V=0.525, U=75)$: $\mathrm{RMSE}_{\log} = 3.10\times 10^{-4}$ vs $\mathrm{RMSE}_{\mathrm{poly}} = 1.25\times 10^{-3}$ ($4\times$ improvement), $\kappa = +2.579\times 10^{-2} > 0$, $b_{\mathrm{poly}} = +3.861\times 10^{-2} > 0$.
- $(n=1.0, V=1.235, U=75)$: $\kappa = +5.127\times 10^{-3} > 0$, $b_{\mathrm{poly}} = +7.54\times 10^{-3} > 0$.

Across all 14 points (`Fig. 7c`), **both $\kappa > 0$ and $b_{\mathrm{poly}} > 0$ globally**: $F_{\mathrm{eff}}(\psi)$ is strictly convex in $\psi^2$ with zero potential barrier ($\kappa/\ln(4/|\psi|) \to 0^+$ as $\psi \to 0$ drives a strictly continuous, essential-singularity turn-on wherever $a \le 0$), algebraically precluding any first-order jump or tricritical point anywhere in the phase diagram.

---

## 5. Task 5: Two-Spin & Band Concurrence Decomposition Across True vs. Refuted Boundaries

Evaluating the manuscript's conditional-Wick spin-resolved and helicity-band two-qubit states (`helicity_density_matrix.py`, `density_matrix.py`, `observables.py`) across 9 parameter cuts (`research/data/task5_concurrence.json`, `research/data/csv/task5_concurrence_cuts.csv`, `3142` rows) across separations $\mathbf{r}/a \in \{(0.25,0), (0.35,0), (0.50,0), (0.75,0), (1.0,0), (0.50,0.50)\}$ at $N_k=201, 401$ yields four unambiguous conclusions (`Fig. 8`, `Fig. 9`):

1. **Short-Range Normal Exchange Dominance (`Fig. 8a–c`, `Fig. 9a`):**  
   At lattice separation $r = 1.0a$ and $0.75a$, spin concurrence is small or identically zero ($C(1,0) \equiv 0.0000$ across all 9 cuts). At $r = 0.25a$, $>95\%$ of the SC concurrence comes from **normal-state Fermi-exchange statistics** ($C_N = 0.724149$ spin / $0.694828$ band at $n=1.0, V_{\mathrm{SO}}=0.5t$; $C_N = 0.549040$ spin / $0.548695$ band at $n=1.875, V_{\mathrm{SO}}=0.5t$), consistent with Oh & Kim (*Phys. Rev. A* **69**, 054305 (2004); *Phys. Rev. B* **71**, 144523 (2005)).
2. **Fermi-Smearing Mechanism Dominates Over Anomalous Pairing (`Fig. 9c`):**  
   Decomposing the superconducting change $\Delta C = C_{\mathrm{SC}} - C_N$ into (i) Fermi-step smearing $C_{\mathrm{norm\text{-}only}} - C_N$ (setting anomalous pair amplitudes $F_{AB} = 0$ while retaining BdG quasiparticle occupations $v_{\mathbf{k}}^2$) and (ii) pure anomalous $F_{AB}$ contribution $C_{\mathrm{SC}} - C_{\mathrm{norm\text{-}only}}$ proves that **superconductivity slightly reduces $C(r=0.25a)$ almost entirely via Fermi-surface smearing**:
   - Along `cut_U_n1.0_V0.5` ($U \in [40, 110]t$): $|C - C_N| \le 0.07493$ whereas $|C - C_{\mathrm{norm\text{-}only}}| \le 4.416\times 10^{-3}$ ($<6\%$ of the change).
   - Along `cut_n_V0.5`: $|C - C_N| \le 0.01422$ vs $|C - C_{\mathrm{norm\text{-}only}}| \le 7.952\times 10^{-4}$.
   - Along `cut_V_n1.0`: $|C - C_N| \le 0.07910$ vs $|C - C_{\mathrm{norm\text{-}only}}| \le 0.01545$.
3. **Clean Continuous QPT Detection Only When Normal State Is Fixed (`Fig. 8a,d`):**  
   Along `cut_U_n1.0_V0.5` (fixed $\mu_N = 0$, hence constant $C_N = 0.724149$), $C(U)$ is flat for $U > U_c \approx 73.92t$ and turns smoothly downward for $U < U_c$, producing a clean one-sided onset in $dC/dU$ (peak $2.87 \times 10^{-3}t^{-1}$) and a sharp minimum in $d^2C/dU^2 = -2.170\times 10^{-3}t^{-2}$ near $U_c$. Conversely, along the density cut `cut_n_V0.5` ($n_c \approx 1.019$) and $V_{\mathrm{SO}}$ cuts (`cut_V_n1.0`, `cut_V_n1.4`), $C$ and $dC/d\theta$ vary completely smoothly across the refuted boundaries $V_{\mathrm{SO}} = 0.525t$ and $1.235t$ (`Fig. 8b,c,e,f`), while the essential-singularity threshold at $n_c = 1.019$ is visually dwarfed by the smooth normal background slope $dC_N/dn \approx -0.25$ and discrete-grid fluctuations.
4. **Two Kinds of Spurious Concurrence Derivative Singularities (`Fig. 9b`, `Fig. 12c`):**
   - **Discrete Fermi-Surface Stair-Step Noise (`Fig. 8e,f`, `Fig. 12c`):** On finite $N_k=201$ grids at fine steps ($\Delta n = 0.002$ or $\Delta V = 0.005$), discrete $k$-points crossing the Fermi contour produce jagged oscillations in $d^2C_N/dn^2$ up to **$\pm 20$** even in the **purely normal state** ($\Delta = 0$) below threshold ($n < n_c$ or $V < V_c$), completely unrelated to superconductivity.
   - **Wootters Entanglement Sudden Death (ESD) Cutoff Kinks (`Fig. 9b`):** Because Wootters concurrence $C = \max(0, C_{\mathrm{raw}})$ (`observables.py`, Yu & Eberly, *Science* **323**, 598 (2009)) clips the smooth signed eigenvalue difference $C_{\mathrm{raw}} = \tilde{\lambda}_1 - \tilde{\lambda}_2 - \tilde{\lambda}_3 - \tilde{\lambda}_4$ at zero, whenever $C_{\mathrm{raw}}$ passes smoothly through $0$ inside a completely smooth SC phase it creates artificial slope discontinuities and $d^2C/dV^2$ spikes — explicitly observed at $r=0.50a$ on `cut_V_n1.4` at $V_{\mathrm{SO}} \approx 0.81t$ (spin) and $0.75t$ (band), on `cut_V_n1.0` at $V_{\mathrm{SO}} \approx 1.03t$ (band), and on `cut_n_V0.5` at $r=(0.5, 0.5)$ near $n \approx 0.855$ (band), in direct accord with Yang (*Phys. Rev. A* **71**, 030302(R) (2005)).

---

## 6. Task 6: Revalidation at $n = 1.875$ and Correction of `manuscript.tex` Claims

We systematically compared the manuscript's $n = 1.875$ sweeps (`U/t = 75, V_SO/t \in [0, 1.5]` and `V_SO/t = 0.5, 0.7, U/t \in [20, 160]`) on both repository (`endpoint=True`) and periodic (`endpoint=False`) meshes from $N_k = 201$ to $801$ (`research/data/task6_7_audit.json`, `research/data/lifshitz_V_Nk401.json`, `research/data/csv/task6_n1875_{V,U}sweep.csv`, `Fig. 10`, `Fig. 11`).

### 6.1 Origin of the Manuscript's $(0.5487t, 68.55t, 118.04t)$ Values
Every crossing cited in `manuscript.tex` L1257–1294 is **not** the normal-state Lifshitz point $\mu_N = 4t$, but the finite-grid superconducting chemical potential crossing $\mu_{\mathrm{SC}} = 4t$ on $N_k=201$ `endpoint=True`:

| Sweep at $n=1.875, \Delta_t=4.5t$ | Manuscript Claim | $\mu_{\mathrm{SC}}=4t$ ($N_k=201$ repo) | $\mu_{\mathrm{SC}}=4t$ (Continuum per) | True Normal Lifshitz $\mu_N = 4t$ (Continuum per) |
| :--- | :--- | :--- | :--- | :--- |
| **$V_{\mathrm{SO}}$ sweep ($U=75$)** | $V_{\mathrm{SO}}/t = 0.5487$ (`0.549`) | **$0.54875t$** | **$0.56187t$** | **$0.69421t$** ($0.68485t$ repo 201) |
| **$U$ sweep ($V_{\mathrm{SO}}=0.5t$)** | $U/t = 68.55$ (repo scripts) | **$68.568t$** | **$66.814t$** | *None* ($\mu_N = 3.6477t < 4t$ $\forall U$) |
| **$U$ sweep ($V_{\mathrm{SO}}=0.7t$)** | $U/t = 118.04$ (L1257) | **$118.082t$** | **$111.708t$** | *None* ($\mu_N = 4.0120t > 4t$ $\forall U$) |

In `analysis/test_qpt_diagnostics.py` L221–235, the author evaluated the bare dispersion $\epsilon_0(\mathbf{k}) = -2\cos k_x - 2\cos k_y - 2 V_{\mathrm{SO}}\sqrt{\sin^2 k_x + \sin^2 k_y}$ against `mu = mu_down[idt_fig, i]` ($\mu_{\mathrm{SC}}$ instead of $\mu_N$) and reported `<theta(eps0 - mu_SC)>` vanishing at $0.54875t$. In the true normal state, $\mu_N = 4t$ does not occur until $V_L = 0.6942t$ in the $V_{\mathrm{SO}}$ sweep, and **does not exist at all** along either fixed-$V_{\mathrm{SO}}$ $U$-sweep because $\mu_N$ does not depend on $U$!

### 6.2 Exact Classification of the Normal Dirac–Lifshitz Transition vs. Gapped SC Crossover (`Fig. 11`)
Expanding the helicity dispersions around the Brillouin-zone corner $\mathbf{k} = (\pi - q_x, \pi - q_y)$ gives an **isotropic 2D Dirac cone** centered at $E = 4t$:
$$
\epsilon_{\mp}(\pi - \mathbf{q}) = 4t \mp 2 V_{\mathrm{SO}} |\mathbf{q}| - t |\mathbf{q}|^2 + \mathcal{O}(q^4).
$$
For $\mu_N < 4t$ ($V < V_L$), the lower helicity cone $\epsilon_-$ contains an empty hole pocket centered at $(\pi,\pi)$; at $\mu_N = 4t$ ($V_L = 0.69421t$), this $(\pi,\pi)$ hole pocket shrinks to the Dirac apex and for $\mu_N > 4t$ an electron pocket of $\epsilon_+$ nucleates at $(\pi,\pi)$ (`Fig. 11a,b`). Because a 2D Dirac cone has a **linearly vanishing V-kink density of states** $N_{\mathrm{cone}}(E) \propto |E - 4t| / V_{\mathrm{SO}}^2$ rather than the Heaviside step of a 2D parabolic band edge, its singular grand-potential contribution scales as $\delta \Omega_{\mathrm{cone}} \propto |\mu_N - 4t|^3$.

High-precision low-temperature Fermi–Dirac differentiation at $T = 2\times 10^{-4}t, N_k = 801$ (`V_L = 0.69426t`) directly verifies this **third-order** character (`Fig. 11c`):
- $d^2 e_N / dV_{\mathrm{SO}}^2$ remains **continuous** with a sharp symmetric V-cusp reaching $-0.00391t^{-1}$ at $V_L$ (from $-0.01317$ at $0.680t$ through $-0.00391$ at $0.695t$ to $-0.01331$ at $0.710t$).
- $d^3 e_N / dV_{\mathrm{SO}}^3$ exhibits a **discontinuous jump from $+0.881t^{-2}$ to $-0.735t^{-2}$** across $V_L$.

Thus, `manuscript.tex` L1296–1298 misclassifies the normal transition as a "second-order Ehrenfest" jump in $d^2e/dV^2$. Furthermore, inside the superconducting ground state at $\Delta_t = 4.5t, U = 75t$, the Bogoliubov quasiparticle spectrum remains **deeply gapped** across the entire $V_{\mathrm{SO}} \in [0, 1.5]t$ sweep ($\min_{\mathbf{k},s} E_s(\mathbf{k}) \ge 0.9374t$, and $E_-(\pi,\pi) \ge 1.366t$ at $V_L$, `Fig. 10c`, `Fig. 11b`) and across the $U$ sweeps ($\min E_s(\mathbf{k}) \ge 0.115t$ at $V=0.5t$ and $\ge 0.266t$ at $V=0.7t$). Consequently, no Fermi pocket closes or empties in the superconducting ground state, and $e_{\mathrm{SC}}$, $\Delta_0$, $\Delta_S$, and $C(r)$ remain $C^\infty$-smooth through both $0.562t$ and $0.694t$ (`Fig. 10c`, `Fig. 11c`).

### 6.3 Remaining Textual & Bibliographic Discrepancies in `manuscript.tex`
1. **L1277 ("The pairing onset at $V_{\mathrm{SO}}/t = 0.97$"):** Contradicts L1308 and exact pairing diagnostics (`reports/qpt_diagnostic_tests.json`): at $(n=1.875, U=75, \Delta_t=4.5)$, $\lambda_+(V=0) = 8.125 > 1$ ($|\Delta_S(0)| = 1.49t$), and $0.97t$ is merely the broad inflection maximum of $d|\Delta_S|/dV_{\mathrm{SO}}$ ($\mathrm{FWHM}=1.19t$). By contrast, at $V_{\mathrm{SO}}=0, n=1.875, U=75$, the genuine continuum $\Delta_t$ pairing onset is $\Delta_{t,c}(V=0) = \mathbf{2.5267t}$ (consistent with L1301's $2.52 \pm 0.03t$, shifting down to $2.3483t$ at $V_{\mathrm{SO}} = 0.5t$).
2. **Bibliography (`manuscript.tex` L1501–1515):** `\bibitem{OhKim2005}` (L1501–1503) duplicates `\bibitem{Oh2005}` (`PRB 71, 144523`) with a conflated title; `\bibitem{Osterloh2002}` (L1510) cites `Phys. Rev. Lett. 89, 087902` instead of ***Nature* 416, 608 (2002)** (`10.1038/416608a`); `\bibitem{Yang2004}` (L1513) cites the preprint instead of ***Phys. Rev. A* 71, 030302(R) (2005)** (`10.1103/PhysRevA.71.030302`).

---

## 7. Task 7: Numerical Convergence & Solver Benchmarking (`Fig. 12`)

Comparing periodic (`endpoint=False`) and repository (`endpoint=True`) grids for $N_k \in \{101, 201, 401, 801\}$ and `xtol` $\in \{10^{-8}, 10^{-11}, 10^{-14}\}$ (`research/data/task6_7_audit.json`, `research/data/csv/task7_convergence_benchmarks.csv`) demonstrates:
- **Spectral vs. Algebraic Convergence (`Fig. 12a,b`):** Away from threshold at $(n=1.4, V=0.525t)$ and $(n=1.0, V=1.235t)$, **periodic $N_k = 401$** converges $\Delta_0, \Delta_S, \mu_{\mathrm{SC}}$ to **$10^{-14}$–$10^{-16}t$** and condensation energy $\Delta F$ to **$\le 3.2\times 10^{-7}t$** relative to $N_k = 801$. By contrast, `endpoint=True` decays algebraically as $\mathcal{O}(N_k^{-1})$, leaving $|\Delta_0(401) - \Delta_{0,\mathrm{per}}(801)| \approx 3.8\text{–}4.7\times 10^{-3}t$ and $|\Delta F(401) - \Delta F_{\mathrm{per}}(801)| \approx 3.8\text{–}8.3\times 10^{-4}t$ even at $N_k = 401$.
- **Critical Region Requirement:** Near the continuous threshold ($n=1.05, V=0.5t$, $|\Delta_{\mathrm{FS}}| \approx 1.8\times 10^{-3}t$), $N_k \ge 401$–$801$ periodic is required to resolve the small Fermi-surface gap to $\sim 10^{-5}t$ and eliminate $\Delta_t$-dependent grid resonances (`Fig. 4b`, `Fig. 12b`).
- **Solver Tolerance:** `xtol = 1e-11` with analytical Jacobian (`research/fast_solver.py`) already matches `xtol = 1e-14` to $|\delta \Delta| < 10^{-12}t$ and $|\delta F| < 10^{-13}t$ while delivering a $4\times$ speedup over numerical finite-difference Jacobians.

---

## 8. Inventory of Verified Artifacts & Peer-Reviewed Literature

### 8.1 Generated Datasets (`research/data/csv/`, 11 files, 2.0 MB) & Figures (`research/figs/`, 12 sets / 36 files, 8.4 MB)
- **CSVs:** `task1_adaptive_hysteresis.csv`, `task1_compressibility.csv`, `task2_critical_scaling.csv`, `task2_window_fits.csv`, `task3_phase_diagram_points.csv` (858 points including repaired $N_k=201$ and periodic $N_k=401$ grids), `task3_continuum_nV_maps.csv` (3,835 points), `task4_landau_potential.csv` (210 points), `task5_concurrence_cuts.csv` (3,142 rows), `task6_n1875_Vsweep.csv` (478 rows), `task6_n1875_Usweep.csv` (344 rows), `task7_convergence_benchmarks.csv` (40 rows).
- **Publication Figures (`.png`, `.pdf`, `.svg`):** `fig01_task1_candidate_falsification` through `fig12_task7_convergence_audit`.

### 8.2 Verified Peer-Reviewed References
1. J. Hutchinson, J. E. Hirsch, & F. Marsiglio, *Phys. Rev. B* **97**, 184513 (2018), DOI: `10.1103/PhysRevB.97.184513`.
2. J. E. Hirsch & F. Marsiglio, *Phys. Rev. B* **39**, 11515 (1989), DOI: `10.1103/PhysRevB.39.11515`; F. Marsiglio & J. E. Hirsch, *Phys. Rev. B* **41**, 6435 (1990), DOI: `10.1103/PhysRevB.41.6435`.
3. L. Arrachea & A. A. Aligia, *Phys. Rev. Lett.* **73**, 2240 (1994), DOI: `10.1103/PhysRevLett.73.2240`.
4. L. P. Gor'kov & E. I. Rashba, *Phys. Rev. Lett.* **87**, 037004 (2001), DOI: `10.1103/PhysRevLett.87.037004`.
5. E. Cappelluti, C. Grimaldi, & F. Marsiglio, *Phys. Rev. Lett.* **98**, 167002 (2007), DOI: `10.1103/PhysRevLett.98.167002`.
6. J. P. Vyasanakere, S. Zhang, & V. B. Shenoy, *Phys. Rev. B* **84**, 014512 (2011), DOI: `10.1103/PhysRevB.84.014512`.
7. G. E. Volovik, *Low Temp. Phys.* **43**, 47 (2017), DOI: `10.1063/1.4974185`; *Physics-Uspekhi* **61**, 89 (2018), DOI: `10.3367/UFNe.2017.01.038218`.
8. D. Innocenti *et al.*, *Phys. Rev. B* **82**, 184528 (2010), DOI: `10.1103/PhysRevB.82.184528`; M. V. Mazziotti *et al.*, *Phys. Rev. B* **103**, 024523 (2021), DOI: `10.1103/PhysRevB.103.024523`.
9. S. L. Sondhi, S. M. Girvin, J. P. Carini, & D. Shahar, *Rev. Mod. Phys.* **69**, 315 (1997), DOI: `10.1103/RevModPhys.69.315`.
10. W. K. Wootters, *Phys. Rev. Lett.* **80**, 2245 (1998), DOI: `10.1103/PhysRevLett.80.2245`.
11. A. Osterloh, L. Amico, G. Falci, & R. Fazio, *Nature* **416**, 608 (2002), DOI: `10.1038/416608a`.
12. T. J. Osborne & M. A. Nielsen, *Phys. Rev. A* **66**, 032110 (2002), DOI: `10.1103/PhysRevA.66.032110`.
13. L.-A. Wu, M. S. Sarandy, & D. A. Lidar, *Phys. Rev. Lett.* **93**, 250404 (2004), DOI: `10.1103/PhysRevLett.93.250404`.
14. L. Amico, R. Fazio, A. Osterloh, & V. Vedral, *Rev. Mod. Phys.* **80**, 517 (2008), DOI: `10.1103/RevModPhys.80.517`.
15. M.-F. Yang, *Phys. Rev. A* **71**, 030302(R) (2005), DOI: `10.1103/PhysRevA.71.030302`.
16. T. Yu & J. H. Eberly, *Science* **323**, 598 (2009), DOI: `10.1126/science.1167343`.
17. S. Oh & J. Kim, *Phys. Rev. A* **69**, 054305 (2004), DOI: `10.1103/PhysRevA.69.054305`; *Phys. Rev. B* **71**, 144523 (2005), DOI: `10.1103/PhysRevB.71.144523`.
18. M. Ebrahimian, F. Khosrojerdi, & R. Afzali, *Phys. Rev. B* **106**, 024512 (2022), DOI: `10.1103/PhysRevB.106.024512`.
