# Evidence and reproducibility report

**Repository commit used for the audit:** `6ef5c49faf09494a25777d85591a6be0cf732446`  
**Reproduction command:** `.venv/bin/python analysis/reproduce_and_validate.py`  
**Execution:** completed successfully; see `reports/reproduction_stdout.txt`.  
**Numerical environment:** Python 3.11, NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.2.

## Evidence map

Status labels follow the requested convention: **VERIFIED** (code/data/direct numerical check), **DERIVED** (mathematical consequence of verified definitions), **QUALITATIVE** (restricted interpretation of actual plotted behavior), and **UNVERIFIED** (missing evidence).

| Manuscript claim | Supporting evidence | Source file / equation / figure | Verification status |
|---|---|---|---|
| The BdG residual equations and `U`-dependent interaction kernel are those solved numerically. | Direct source inspection and execution of `run_selfconsistency`. | `self_consistency.py`, `kspace_setup.py`; manuscript Eqs. (1)–(9) | VERIFIED |
| Archived Green's functions correspond to `U/t=75`, `Delta_t/t=4.5`, `V_SO=0`, `n=1.875`, `N_k=100`. | Fresh `U/t=75` calculation matches low-regime arrays to `1.53e-16` and high-regime arrays to `1.24e-12` maximum absolute error. NPZ metadata alone was insufficient. | `config.py`; `data/*_greens.npz`; `reports/numerical_validation.json` | VERIFIED by numerical reproduction |
| Gap parameters at `U/t=40,75,115` are those printed in the manuscript. | Independent self-consistency solutions; maximum residuals `2.18e-11`, `2.36e-12`, and `2.39e-10`. | `analysis/reproduce_and_validate.py`; `data/hubbard_U40_U75_U115_...npz`; Fig. `fig:delta_ks_heatmaps` | VERIFIED |
| The exact signed BdG coherence identity is `-Delta*/(2E)`. | Algebraic BdG coherence-factor relation. | Manuscript Eq. `eq:coherence_factor` | DERIVED |
| Archived numerical results do **not** implement that signed identity in the anomalous integrand. | `u` and `v` are formed as non-negative roots and `greens_functions.py` evaluates `conj(v)*conj(u)`. | `self_consistency.py`, `greens_functions.py` | VERIFIED implementation discrepancy |
| Absolute values of already real normalized ratios do not alter this density matrix. | Only `g^2` and `f^2` enter the matrix and Werner parameter. | `density_matrix.py`; Eqs. `eq:density_matrix`, `eq:werner_parameter` | DERIVED |
| Initial states are normalized Werner states and the `r=0` state is the pure singlet. | Algebraic reduction and direct trace/Hermiticity/eigenvalue tests. `g(0)=1` implies `p(0)=1`. | `density_matrix.py`; Eqs. `eq:werner_state`–`eq:werner_concurrence`; validation JSON | VERIFIED and DERIVED |
| Both reported amplitude-damping regimes are underdamped and independent. | `2 gamma > Gamma`: `3>0.1` and `9>0.01`; filenames and evolution routine specify independent baths. | `noise_channels.py`, `evolution.py`; Eqs. `eq:kraus_evolution`–`eq:d_parameter` | VERIFIED |
| Lowercase `gamma` and uppercase `Gamma` have different roles. | They enter `d^2=2 gamma Gamma-Gamma^2` and the envelope separately. | `noise_channels.py`; Eq. `eq:AD_probability` | VERIFIED |
| The fidelity convention is squared Uhlmann fidelity. | Direct inspection and numerical comparison with the implemented eigensystem routine. | `fidelity.py:fidelity_uhlmann`; Eq. `eq:uhlmann` | VERIFIED |
| Pure-singlet-referenced Uhlmann fidelity equals singlet fraction. | Pure-reference identity; reconstructed arrays agree with archived `F_pure` to `1.47e-8`. | `fidelity.py`, `observables.py`; Eq. `eq:singlet_fidelity`; validation JSON | VERIFIED and DERIVED |
| Initial-state fidelity is a separate quantity and is unity at `t=0`. | Archived `F_init` has maximum initial-time error `7.11e-15` (low regime) and `3.33e-15` (high regime). | `data/*_fidelity.npz`; validation JSON | VERIFIED |
| The plotted singlet fraction is not an optimized teleportation fidelity for evolved states. | Repository contains no local Bell optimization or fully entangled-fraction calculation. The standard relation was therefore not applied to evolved curves. | `fidelity.py`, `observables.py`; Eq. `eq:uhlmann` and teleportation discussion; Horodecki reference | VERIFIED absence; operational values UNVERIFIED |
| High-regime singlet-fidelity maxima at `r=0` are `0.81091`, `0.65751`, and `0.53309` near `t=21,42,63`. | Direct extraction from archived `F_pure[0,:]`. | Mandatory Figure A; high-regime fidelity NPZ | VERIFIED |
| High-regime concurrence occupies repeated spatial/temporal windows and differs from singlet-fidelity persistence at finite separation. | Visual inspection plus reconstruction from the saved Green's functions and exact channel. At the special point `r=0`, `F_s=C=p`; at finite `r`, singlet weight can persist when `C=0`. | Mandatory Figures A and B; Eq. `eq:r0_high` | VERIFIED and QUALITATIVE for map description |
| Low-coupling singlet-fidelity revivals at `r=0` are `0.31136` and `0.09694` at the first two sampled maxima. | Direct extraction from the 261-point low-regime fidelity array. | Mandatory Figure E; low-regime fidelity NPZ | VERIFIED |
| Angular noise coupling is not explicitly introduced. | Kraus operators depend on time, `gamma`, and `Gamma`, not `theta`; `theta` enters only the real-space phase in the Green's functions. | `greens_functions.py`, `noise_channels.py`; Eq. `eq:r_parameterization` | VERIFIED |
| The initial singlet fraction is anisotropic at selected finite radii and obeys the sampled square-lattice symmetries. | At `r=3.3166`, values range from `0.25190` to `0.50801`; symmetry-pair differences are at most `1.40e-8` in fidelity. | Mandatory Figure F; `data/*_theta_obs.npz`; validation JSON | VERIFIED |
| The polar plot does not cover the full angular circle. | Plot wrapper uses seven angles `0,15,...,90` degrees and appends only the first point for closure. | `visualize_theta.py:plot_F_polar`; Mandatory Figure F | VERIFIED |
| The angular Holevo heatmap is angle independent only at `r=0`. | Plot function fixes `r_index=0`; archived rows have zero peak-to-peak angular variation at this radius. | `holevo.py:plot_holevo_theta_t_heatmap`; Mandatory Figure C | VERIFIED; finite-`r` isotropy not claimed |
| Equal-probability Pauli encoding gives `rho_bar=(I_A/2) tensor rho_B` and `chi_P=1+S_2(rho_B)-S_2(rho_AB)`. | Direct Pauli-twirl derivation and comparison with `compute_holevo`; reconstructed archived `chi_all` agrees to `3.68e-11`. | `holevo.py`; Eqs. `eq:pauli_average`, `eq:holevo_entropy`; validation JSON | VERIFIED and DERIVED |
| Entropies in the Holevo calculation use bits. | `von_neumann_entropy_bits` uses `log2`; main observable entropy separately uses natural logarithms. | `holevo.py`, `observables.py` | VERIFIED |
| Initial low-regime Holevo spatial maxima include `1.06735` bits at `r=1.6080` and `0.27961` bits at `r=3.2161`. | Direct extraction from `chi_rt[:,0]` at `theta=45` degrees. | Mandatory Figure D; low-regime Holevo NPZ | VERIFIED |
| Long-time independent amplitude damping gives a pure product state with `F_s=C=0` and `chi_P=1`. | Kraus fixed point and entropy identity. Finite `t=80` Holevo values still differ from one by at most `1.64e-3` across the archived grid. | `evolution.py`; Eq. `eq:holevo_entropy`; validation JSON | DERIVED limit; finite-time values VERIFIED |
| `U` affects open-system measures through the initial state, not through the Kraus map. | `U` appears in the BdG kernel; no noise function accepts `U`. | `self_consistency.py`, `noise_channels.py`, `evolution.py` | VERIFIED |
| A numerical `U`-dependence was established for three interaction strengths. | New full-grid calculations at `U/t=40,75,115`, not inference from one fixed-`U` run. All passed residual and PSD checks. | New interaction-sweep NPZ; Fig. `fig:U_fidelity_holevo` | VERIFIED |
| Interaction ordering is separation dependent, not universally beneficial or detrimental. | At `r=0.50251`, `F_s=(0.53243,0.350996,0.284845)` for `U=(40,75,115)`; at `r=1.50754`, `F_s=(0.717745,0.778551,0.783515)`. Holevo orderings also change. | New interaction-sweep NPZ; manuscript `sec:U_results` | VERIFIED |
| A universal monotonic law or optimal `U` follows from the sweep. | Only three `U` values at fixed filling and zero SOC were calculated, and orderings reverse with radius. | New interaction-sweep NPZ | UNVERIFIED; deliberately omitted |
| Direct signed-versus-unsigned Green's-function dynamics are established. | No such comparative calculation exists in the archived data or new sweep. | `greens_functions.py`; limitations report | UNVERIFIED; deliberately omitted |

## New interaction-sweep configuration

| Field | Value |
|---|---|
| `U/t` | `40, 75, 115` |
| Fixed BdG parameters | `Delta_t/t=4.5`, `V_SO=0`, `n=1.875`, `T=0` |
| Momentum grid | `100 x 100`, including the implementation's `[-pi,pi]` endpoints |
| Radial grid | 200 points, `0 <= r <= 10` |
| Angular grid | `0,15,30,45,60,75,90` degrees |
| Time grid | 261 points, `0 <= t <= 80` |
| Noise | independent non-Markovian amplitude damping on both qubits |
| Noise parameters | `gamma_A=gamma_B=1.5`, `Gamma_A=Gamma_B=0.1` |
| Implementation convention | current magnitude-amplitude Green's-function reduction |
| Saved arrays | `C`, `F_singlet`, `chi`, `S_AB`, `S_B`, plus BdG parameters, residuals, PSD minima, and commit |
| Data file | `data/hubbard_U40_U75_U115_amp_independent_g1.50_G0.100_observables.npz` |

The selected values were not a newly invented range: all three were already used in the supplied unfinished manuscript's gap discussion. No experimental significance is assigned to them.

## Validation summary

- Initial-density-matrix trace, Hermiticity, and positive-semidefiniteness checks passed for every newly generated state.
- Direct evolved-state cross-checks found maximum trace error `2.22e-16`, zero Hermiticity error, and minimum sampled eigenvalue zero within numerical precision.
- Reconstructed versus repository concurrence: maximum error `2.78e-17`.
- Reconstructed versus repository singlet fraction: maximum error `2.22e-16`.
- Reconstructed versus repository Holevo quantity: maximum error `8.54e-14` at direct test points; archived full-array error `3.68e-11`.
- Pure-reference Uhlmann fidelity versus singlet fraction: maximum direct-test error `4.26e-9`.
- The zero-separation identities `F_s(0,t)=p(t)` and `C(0,t)=p(t)` hold with zero floating-point discrepancy in the reconstructed arrays for both archived regimes.
- Archived fidelity values lie in `[0,1]` up to `2.2e-15` floating-point overshoot.
- Archived Holevo values lie in `[0,2]` within numerical precision.
- Complete numerical values are recorded in `reports/numerical_validation.json`.
