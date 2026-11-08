# Microscopic origin and publishability assessment of BdG-resource teleportation

## Executive assessment

The corrected equal-time calculation establishes a **short-range,
continuous-distance teleportation window**, but not a teleportation advantage
at any tested nonzero square-lattice site. At \(U/t=75\) and
\(\theta=45^\circ\), the converged initial boundary is
\[
r_c/a=0.46915899\quad(N_k=400).
\]
Both fixed and optimized average fidelities exceed \(2/3\) for continuous
\(0<r<r_c\), but the boundary is not a thermodynamic critical point. It is the
Werner-weight crossing \(p=1/3\).

Strongly oscillatory non-Markovian amplitude damping can temporarily restore
the advantage after it has been lost, but only within a smaller region that was
initially entangled. The first-revival boundary at \(U/t=75\) is bracketed by
\[
0.3875<r_c^{(\mathrm{rev})}/a\leq0.3900.
\]
Initially separable finite-distance resources did not acquire an above-\(2/3\)
advantage under the implemented local channels.

The microscopic dependence is weak at the threshold. Over
\(40\leq U/t\leq115\), the initial \(N_k=100\) boundary changes only from
0.466083 to 0.468749 lattice constants, while the self-consistent gap
magnitudes change substantially. Direct anomalous-\(F\) contributions to the
Werner weight are very small near the boundary; the normal correlator \(G\)
dominates the calculated threshold.

**Publishability conclusion:** these results do not yet establish a new
superconductivity-enabled teleportation mechanism suitable for a strong Q1
physics claim. Most threshold and revival behavior follows from the Werner
form and standard local amplitude-damping dynamics. The model-specific
continuous-distance map is useful, but the absence of an advantage at tested
nonzero lattice sites and the weak pairing dependence substantially limit the
physical novelty.

## 1. Methods and numerical reliability

### Existing implementations reused

The study uses the unmodified:

- BdG self-consistency solver;
- corrected `equal_time` Green's functions;
- Oh--Kim two-spin density-matrix construction;
- independent/common-bath Kraus evolution;
- explicit Bell-measurement and feedforward protocol;
- SVD fully entangled fraction and optimized fidelity.

No Hamiltonian or new physical noise model was introduced. The exact
overdamped comparison uses the existing hyperbolic continuation in
`fidelity.py`; the analytical exponential limit uses the same amplitude-damping
Kraus operators.

### Parameter coverage

- \(U/t=40,45,\ldots,115\), with all 16 \(N_k=100\) solutions satisfying the
  \(10^{-8}\) residual criterion and remaining nontrivial.
- Fixed model parameters: \(\Delta_t/t=4.5\), \(V_{SO}=0\), \(n=1.875\), and
  \(T=0\).
- Continuous distances up to \(r/a=10\) and seven angles for the full
  \(U/t=40,75,115\) maps.
- Exact lattice sites \((1,0),(1,1),(2,0),(2,1),(2,2),(3,0)\).
- Momentum grids \(N_k=100,200,400\) for threshold convergence.
- Production dynamics at \(U/t=75,N_k=400,\theta=45^\circ\).
- Existing amplitude-damping regimes:
  \((\gamma,\Gamma)=(1.5,0.1)\) and \((4.5,0.01)\).
- Independent and common-bath dephasing controls, plus an overdamped
  \(\Gamma/\gamma=100\) comparison and exact exponential channel.

### Validation and convergence

- The Oh--Kim matrices equal their Werner reconstruction with maximum matrix
  error \(2.22\times10^{-16}\).
- Initial analytic fixed/optimized fidelities and concurrence agree with the
  explicit protocol to at most \(2.23\times10^{-16}\).
- At \(U/t=75\), \(r_c/a\) changes from 0.46822888 (\(N_k=100\)) to
  0.46884656 (\(N_k=200\)) and 0.46915899 (\(N_k=400\)).
- At fixed \(N_k=200\), changing the radial spacing from 0.005 to 0.00125
  changes \(r_c\) by only \(5.6\times10^{-6}\).
- The Simpson occupancy error at \(U/t=75\) decreases from 0.451% to 0.228%
  to 0.115%; the discrete sum rule remains satisfied to roundoff.
- The main underdamped evolution and the exact continuation in `fidelity.py`
  agree with zero stored matrix difference at the tested times.
- The largest numerical optimized-below-fixed violation is
  \(2.22\times10^{-16}\).
- Independent artifact validation passes 86/86 checks, including 30 evolved
  resource samples with full conditional Bell-output positivity, Hermiticity,
  trace, and probability-normalization validation.

At \(U/t=115,N_k=400\), the solver returns a zero-gap fixed point with residual
\(3.00\times10^{-12}\). It is retained only as an explicitly labeled
**zero-gap control**, not as a converged paired branch. Strong production
conclusions use the paired \(U/t=75,N_k=400\) solution.

## 2. What is mathematically inevitable

The initial Oh--Kim state is exactly
\[
\rho_W(p)=(1-p)\frac{I_4}{4}+p|\Psi^-\rangle\langle\Psi^-|,
\]
with
\[
p=\frac{g^2+f^2}{2+f^2-g^2},\qquad
 g^2=\frac{|G(r,\theta)|^2}{|G(0,\theta)|^2},\quad
 f^2=\frac{|F(r,\theta)|^2}{|G(0,\theta)|^2}.
\]
Consequently,
\[
\overline{\mathcal F}_{fixed}
=\overline{\mathcal F}_{opt}=\frac{1+p}{2},\qquad
C=\max\left(0,\frac{3p-1}{2}\right).
\]
Thus the initial quantum advantage condition is exactly
\[
\overline{\mathcal F}>2/3
\iff p>1/3
\iff 2g^2+f^2>1
\iff C>0.
\]
The equality of the initial teleportation and Werner-entanglement thresholds is
standard quantum information theory, not new microscopic physics.

The normal and anomalous derivatives,
\[
\frac{\partial p}{\partial(g^2)}=\frac{2(1+f^2)}{(2+f^2-g^2)^2},\qquad
\frac{\partial p}{\partial(f^2)}=\frac{2(1-g^2)}{(2+f^2-g^2)^2},
\]
are nonnegative in the physical domain. Both correlators can increase \(p\),
but their numerical magnitudes determine which one matters.

Local Kraus channels cannot create entanglement from an initially separable
resource. An above-\(2/3\) optimized fidelity requires fully entangled fraction
above \(1/2\), so an initially separable finite-\(r\) Werner resource cannot
acquire that advantage under the implemented local evolutions.

## 3. New numerical observations

### 3.1 Continuous-distance initial advantage

At \(U/t=75,N_k=400,\theta=45^\circ\):

| \(r/a\) | \(p(0)\) | Fixed and optimized initial fidelity | Above \(2/3\)? |
|---:|---:|---:|:---:|
| 0 | 1 | 1 | Yes |
| 0.30 | 0.611863 | 0.805932 | Yes |
| 0.40 | 0.437050 | 0.718525 | Yes |
| 0.45 | 0.360404 | 0.680202 | Yes |
| 0.50 | 0.292507 | 0.646254 | No |
| 1.00 | 0.008318 | 0.504159 | No |

At \(N_k=100\), no second above-threshold island occurs along the
\(\theta=45^\circ\) line after the first crossing up to \(r/a=1.5\). Across all
seven sampled angles and \(U/t=40,75,115\), no point with \(r/a\geq1\) has
\(p>1/3\); the largest such value is \(p=0.0211801\) at \(U/t=40\).

### 3.2 Exact nonzero lattice sites do not show an advantage

Across all 16 interactions and six exact nonzero sites at \(N_k=100\), the
largest Werner weight is
\[
p_{max}=0.00444889,
\qquad \overline{\mathcal F}_{max}=0.50222444,
\]
at \(U/t=40\) and site \((1,0)\). Repeating the sites at \(U/t=75,N_k=400\)
gives a largest value \(p=0.00229876\), again at \((1,0)\), corresponding to
\(\overline{\mathcal F}=0.50114938\).

Therefore the continuous result \(0<r/a\lesssim0.469\) must not be presented as
an experimentally established nearest-neighbor lattice advantage. On the
underlying discrete square lattice, there is no nonzero site between zero and
the continuous boundary. A physical off-lattice or sub-unit-cell detection
interpretation would require separate justification.

### 3.3 Interaction and pairing dependence at the threshold is weak

The \(N_k=100\) continuous boundary varies from \(r_c/a=0.466083\) at
\(U/t=40\) to 0.468749 at \(U/t=115\), a total change of only 0.002666. At
\(r/a=0.40\), the initial fidelity across the same paired \(N_k=100\) sweep
varies only from 0.716325 to 0.718234.

This occurs despite substantial gap changes: at \(N_k=100\),
\(|\Delta_S|/t\) falls from 1.27978 at \(U/t=40\) to 0.361489 at \(U/t=75\)
and 0.003555 at \(U/t=115\). The zero-gap \(U/t=115,N_k=400\) control gives
\(r_c/a=0.469551\), close to the paired values. There is no numerically
meaningful interaction-driven teleportation transition in this range. A
U-dependent crossing can occur only within the narrow radial interval covered
by these nearly coincident boundaries.

### 3.4 Normal correlations dominate the direct threshold contribution

At \(U/t=75,N_k=100\), setting \(f=0\) only in the Werner formula changes
\(p\) by at most \(4.1413\times10^{-4}\) along \(0\leq r/a\leq1.5\). The
anomalous-only diagnostic \(f^2/(2+f^2)\) has the same small maximum.
Nearest the critical grid point, \(r/a=0.47\),
\[
g^2=0.497246,\qquad f^2=2.17\times10^{-6},\qquad p=0.330891.
\]
The threshold is therefore controlled primarily by the decay of the normal
occupation correlator, not by a large direct anomalous-\(F\) contribution.

This is not a complete "no-superconductivity" counterfactual: pairing also
enters the self-consistent \(v_k^2\) weights that determine \(G\). The precise
conclusion is that the **direct anomalous term is negligible at the tested
boundary**, while the total indirect role of pairing in \(G\) remains model
dependent.

## 4. Noise-induced loss and restoration

### 4.1 Amplitude damping

No above-\(2/3\) regain was found in the low regime
\((\gamma,\Gamma)=(1.5,0.1)\) at any tested point.

The high regime \((4.5,0.01)\) produces bounded restoration:

| \(r/a\) | Protocol | First loss | First regain | Maximum after first loss |
|---:|---|---:|---:|---:|
| 0 | fixed | 5.28414 | 16.53773 | 0.873962 |
| 0 | optimized | 5.30000 | 16.50000 | 0.873962 |
| 0.30 | fixed and optimized | 4.06997 | 18.16005 | 0.726502 |
| 0.3875 | optimized | 3.04402 | 20.51813 | 0.667853 |
| 0.39 | optimized | 3.00431 | none | at most \(2/3\) numerically |
| 0.40 | fixed and optimized | 2.83608 | none | at most \(2/3\) numerically |
| 0.50, 1.00 | either | initially below | none | never above \(2/3\) |

At \(r=0\), three high-noise regain events occur within \(0\leq t\leq80\).
The refined first-revival boundary is between \(r/a=0.3875\) and 0.3900. This
is a sampled dynamical boundary for the stated channel and time grid, not a
new phase transition.

The noise does not create an advantage at the initially separable points
\(r/a=0.50\) or 1.00. Their optimized fidelities can rise toward \(2/3\) as the
state approaches a pure product fixed point, but do not cross above it. A rise
below the classical threshold is not a restored quantum advantage.

### 4.2 Markovian control

For the same \(\gamma=4.5\), the overdamped \(\Gamma/\gamma=100\) evolution and
exact exponential channel show no above-\(2/3\) regain. At \(r=0\), optimized
fidelity first falls below threshold near \(t=0.15625\) in the overdamped run
and \(t=0.15500\) in the exponential limit. At \(r=0.40\), the corresponding
loss times are 0.04312 and 0.04109.

The finite-\(\Gamma\) Lorentzian survival probability differs from
\(e^{-\gamma t}\) by at most 0.00940 on the compared interval; optimized-curve
differences are at most 0.00627. This supports the interpretation that the
above-threshold regains require the nonmonotonic environmental envelope, not
merely the initial Werner state.

The main `noise_channels.py` uses a critically damped expression throughout
\(d^2\leq0\), so it is not reliable for general overdamped calculations. The
comparison deliberately uses the exact continuation in `fidelity.py` after
validating agreement in the underdamped regime.

### 4.3 Dephasing controls

Independent dephasing gives no above-threshold regain. Its optimized fidelity
first crosses below \(2/3\) at \(t=33.45\) for \(r=0\) and \(t=2.52398\) for
\(r=0.40\), with no later regain.

The implemented common-bath collective dephasing leaves the tested initial
Werner resources unchanged: optimized fidelity remains 1 at \(r=0\) and
0.718525 at \(r=0.40\). This is preservation in a decoherence-free sector of
the specific collective channel, not a revival.

### 4.4 No information-backflow claim

This study establishes fidelity threshold crossings only. It does not provide
an operational trace-distance backflow witness or a divisibility proof.
Oscillations or revivals in one fidelity are not used as standalone evidence
of genuine quantum information backflow.

## 5. Established versus new versus speculative

### Established analytically and validated numerically

- Initial Oh--Kim resources are Werner states.
- Fixed and optimized initial fidelities are \((1+p)/2\).
- The initial advantage, concurrence, and \(p=1/3\) thresholds coincide.
- Local channels cannot create entanglement from initially separable resources.
- The exponential amplitude-damping limit gives no above-threshold revival.

### New numerical observations from this study

- The converged continuous initial boundary near \(r/a=0.469\) at \(U/t=75\).
- No advantage at the six tested nonzero lattice sites or any sampled point
  with \(r/a\geq1\) in the full maps.
- Very weak variation of the boundary over \(40\leq U/t\leq115\).
- Direct anomalous contributions below \(4.15\times10^{-4}\) in \(p\) on the
  analyzed radial line.
- High-noise regain at \(r=0.30\), no regain at \(r=0.40\), and a refined
  first-revival boundary between 0.3875 and 0.3900.
- No low-noise or independent-dephasing advantage regain in the tested cases.

### Interpretations that remain limited or speculative

- A physically meaningful sub-lattice-continuous detection protocol is not
  established by the lattice calculation.
- The calculations do not show that pairing is irrelevant altogether; they
  show weak direct \(F\) and weak threshold dependence in this parameter range.
- No claim of experimental Bell measurement, mode addressability, microscopic
  quasiparticle input coupling, or genuine information backflow is made.
- No comprehensive literature proof of priority or novelty was performed.

## 6. Final publishability assessment

### Strongest physically meaningful finding

The strongest result is a **negative and delimiting microscopic finding**:
after the equal-time correction, the model supports only an ultrashort
continuous-distance advantage dominated by normal correlations, with no tested
nonzero lattice-site advantage. Non-Markovian amplitude damping can restore
the advantage only inside a smaller initially entangled continuous-distance
region.

### Responses to the requested assessment

- **Finite-separation advantage:** yes for continuous \(0<r/a\lesssim0.469\)
  at \(U/t=75\); no for any tested nonzero square-lattice site.
- **Noise restoration:** yes in the high non-Markovian amplitude-damping regime
  for a restricted region, demonstrated explicitly at \(r/a=0.30\); no in the
  tested low regime, Markovian limit, independent dephasing, or initially
  separable finite-distance examples.
- **Microscopic dependence:** weak at the threshold over the tested interaction
  range. Normal \(G\) dominates the direct contribution; anomalous \(F\) is
  small near the boundary.
- **Standard quantum-information content:** Werner thresholds, initial
  fidelities, entanglement threshold, local-channel limitations, and monotonic
  Markovian decay are expected standard behavior.
- **Genuinely new microscopic physics:** not established by these calculations.
  The new content is primarily a model-specific numerical map and a negative
  assessment of previously assumed finite-range/pairing-driven behavior.

### Evidence required for a reputable Q1 physics journal

The present results alone are unlikely to support a strong Q1 novelty claim.
A publishable advance would require, at minimum:

1. a physically justified detection geometry that makes the sub-lattice
   continuous-distance window meaningful, or a generalized model exhibiting
   advantage at actual discrete modes;
2. a parameter regime in which anomalous pairing correlations materially alter
   the teleportation threshold beyond normal-correlation decay;
3. converged paired solutions across the claimed parameter range, including a
   clear treatment of zero-gap and multiple-solver branches;
4. a microscopic mode-addressing, input-coupling, Bell-measurement, and
   feedforward model with realistic inefficiencies;
5. validation beyond the restrictive mean-field, zero-temperature,
   zero-spin-orbit-coupling, idealized two-spin reduction where relevant;
6. an operational information-backflow analysis if non-Markovian backflow is
   claimed;
7. a systematic literature comparison demonstrating what is genuinely new;
8. complete regeneration of manuscript figures using the corrected equal-time
   Green's functions, without mixing archived legacy finite-\(r\) results.

## 7. Reproduction and artifacts

```bash
.venv/bin/python analysis/validate_teleportation.py
.venv/bin/python analysis/run_microscopic_teleportation_study.py
.venv/bin/python analysis/refine_microscopic_revival_boundary.py
.venv/bin/python analysis/validate_microscopic_study.py
```

Data:

- `data/microscopic_teleportation_initial.npz`
- `data/microscopic_teleportation_dynamics.npz`
- `data/microscopic_teleportation_convergence.npz`
- `data/microscopic_revival_boundary.npz`

Machine-readable reports:

- `reports/microscopic_teleportation_summary.json`
- `reports/microscopic_revival_boundary.json`
- `reports/microscopic_study_validation.json`

Figures:

- `figs/microscopic_initial_U_r_map.png`
- `figs/microscopic_gf_decomposition.png`
- `figs/microscopic_teleportation_convergence.png`
- `figs/microscopic_teleportation_revival_maps.png`
- `figs/microscopic_markovian_comparison.png`
- `figs/microscopic_revival_boundary.png`
