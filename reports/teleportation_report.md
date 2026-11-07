# Explicit quantum teleportation with the BdG two-spin resource

## Scope

This implementation uses the existing time-evolved two-spin density matrix
`rho_AB(r,theta,t)` as a teleportation resource. It does **not** equate an
entangled Q–A state with teleportation. A separate effective input spin Q is
prepared in `rho_Q`, a Bell measurement is applied to Q and A, the four
classical outcomes are communicated, and outcome-dependent Pauli corrections
are applied to B.

The original BdG solver, corrected equal-time Green's functions,
density-matrix construction, Kraus evolution, fidelity, and Holevo modules
were not changed. The analysis uses the current corrected Green's functions,
not the archived legacy finite-separation manuscript data.

## Implementation

- `teleportation.py`
  - `teleport_input`: forms the 8x8 state `rho_Q tensor rho_AB`, applies the
    four Bell projectors to Q,A, records outcome probabilities, traces out
    Q,A, and conditionally normalizes each 2x2 output on B.
  - `feedforward_corrections`: for the singlet resource, the outcome map is
    Phi+ -> `iY`, Phi- -> `X`, Psi+ -> `Z`, Psi- -> `I`.
  - `average_teleportation_fidelity`: averages over the six eigenstates of
  Pauli X,Y,Z. This uses Uhlmann fidelity, reduced to `<psi|rho_out|psi>`
  for the pure input states; it is distinct from initial-state fidelity.
  - `resource_metrics`: reports concurrence, singlet-reference fidelity,
    fully entangled fraction, fixed-protocol fidelity, optimized fidelity,
    and diagnostics.
- `analysis/validate_teleportation.py`: deterministic validation suite.
- `analysis/run_teleportation_analysis.py`: uses the existing solver and
  `evolve_rho`, evaluates five r,theta points in both existing noise regimes,
  and exports arrays, figures, and JSON summaries.

For a two-qubit resource, the local-unitary fully entangled fraction is
evaluated without a numerical optimizer:

```
T_lm = Tr[rho_AB (sigma_l tensor sigma_m)]
f_FEF = 1/4 * (1 + N),    N = sum of singular values of T
F_avg_opt = (2*f_FEF + 1)/3.
```

This is the standard-protocol optimum over local resource rotations, not a
search over arbitrary LOCC protocols. No optimizer convergence parameter is
therefore required.

## Validation

Run:

```bash
.venv/bin/python analysis/validate_teleportation.py
```

Result: **65 checks passed**. The record is
`reports/teleportation_validation.json`.

Validated cases include:

- Every Bell outcome of an ideal singlet has probability 1/4 and every
  corrected output equals an arbitrary normalized input state.
- Fixed and optimized six-state average fidelities are both 1 for a singlet.
- A maximally mixed resource outputs `I/2`, with fixed and optimized average
  fidelity 1/2 and fully entangled fraction 1/4.
- For the Werner resource

  ```
  rho_W = w |Psi-><Psi-| + (1-w) I/4,
  ```

  both teleportation fidelities equal `(1+w)/2`, while the singlet fraction
  and fully entangled fraction equal `(1+3w)/4`. Concurrence is
  `max(0,(3w-1)/2)`. Six values of w were checked.
- For the pure product resource `|00><00|`, fixed singlet-reference fidelity
  is 1/3, the fully entangled fraction is 1/2, and optimized fidelity is 2/3.
  This explicitly distinguishes the fixed and optimized protocols.
- Hermiticity, unit output trace, outcome-probability normalization,
  positivity, and exact Bell-projector reconstruction were checked for the
  tested states.

## Numerical analysis

Run:

```bash
.venv/bin/python analysis/run_teleportation_analysis.py
```

Parameters are recorded in `reports/teleportation_analysis_summary.json`:
`U/t=75`, `V_SO=0.7`, `n_elec=2`, `Nk=400`, `Delta/t=0.9`, and 261 time points
from 0 to 80. The distance unit is `a` and the time unit is `hbar/t`.
Five points were evaluated: r=0 and two finite distances at theta=45 deg, plus
r=1.5075377 at theta=0 and 90 deg. Both existing low- and high-noise settings
were used.

Initial values at theta=45 deg:

| r (a) | concurrence | singlet fraction | fixed fidelity | optimized fidelity |
|---|---:|---:|---:|---:|
| 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 1.5075377 | 0 | 0.251301 | 0.500867 | 0.500867 |
| 3.0150754 | 0 | 0.250331 | 0.500331 | 0.500331 |

At r=1.5075377, the theta=0 and 90 deg initial fixed/optimized fidelity is
0.509795 and the initial singlet fraction is 0.254898. The three angular
curves overlap in the present model: the anomalous magnitude used here is
angle-independent and the implemented local amplitude damping is also
angle-independent. This is not a claim that the initial BdG state has no
angular structure.

Only r=0 has above-2/3 intervals in this run. In the high-noise regime its
fixed-protocol fidelity has later revivals above the classical benchmark.
All tested finite-r points remain below 2/3 for both fixed and optimized
fidelities at every simulated time. At r=0 the final fixed fidelities are
0.33348348 (low noise) and 0.53582200 (high noise); the optimized values are
numerically 2/3 at the final time. These time-resolved differences show why
fixed-protocol performance cannot be inferred from singlet fraction,
concurrence, or the locally optimized fully entangled fraction alone.

The complete arrays are in `data/teleportation_metrics.npz`. Numerical output
checks reported minimum output eigenvalue 0, maximum output trace error
`2.22e-16`, and probability-sum error `7.77e-16`.

## Figures

- `figs/teleportation_theta45_low.png`: r comparison at theta=45 deg, low
  noise.
- `figs/teleportation_theta45_high.png`: the same comparison at high noise.
- `figs/teleportation_angles_low.png`: angular comparison at
  r=1.5075377, low noise.

Each figure shows concurrence, singlet fraction, fixed-protocol average
fidelity, fully entangled fraction, and optimized average fidelity. The
classical average-fidelity benchmark 2/3 is marked on both fidelity panels.

## Assumptions and limitations

- Q is an effective input spin, not automatically a physical Bogoliubov
  quasiparticle.
- A and B are assumed independently addressable spin modes; Q is initially
  independent of A,B.
- Bell measurement and classical feedforward are ideal and are applied after
  the open-system evolution used to generate rho_AB.
- Conditional B outputs and probabilities are computed from the reduced
  two-spin state; no microscopic Bell-detector model is included.
- The work does not establish a quasiparticle–Cooper-pair interaction,
  individual Cooper-pair addressability, or experimental Bell-measurement
  feasibility.
- The simulation is a fixed-U, selected-grid demonstration, not a complete
  distance, angle, U, or noise-parameter study.
- Nonmonotonicity alone is not interpreted as information backflow.
