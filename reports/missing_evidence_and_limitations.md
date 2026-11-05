# Missing evidence and limitations

## 1. Signed coherence-factor calculation remains unresolved

The manuscript derives the signed BdG identity, but the current `greens_functions.py` does not use it. It forms non-negative Bogoliubov amplitudes and integrates their product. All six mandatory figures and the new `U`-sweep therefore describe the **current magnitude-amplitude implementation**, not a verified signed-coherence calculation.

A scientifically valid comparison requires, at minimum:

1. an anomalous integrand constructed from `-Delta_ks/(2 E_ks)` with the implementation's BdG sign convention;
2. identical self-consistency solutions and quadrature grids for both conventions;
3. direct comparison of `G`, `F`, density matrices, concurrence, singlet fraction, and Holevo quantity;
4. repetition of the interaction sweep if the initial states change.

No signed-result curves or claims of numerical sign-induced enhancement were inserted.

## 2. Operational teleportation fidelity was not calculated for evolved states

The repository calculates Uhlmann fidelity to the initial state and to the fixed singlet. It does not optimize over local unitaries to calculate the fully entangled fraction after evolution. Consequently:

- the singlet-fidelity plots cannot be assigned optimized average teleportation fidelities;
- the `2/3` average-teleportation benchmark cannot be applied indiscriminately to the plotted singlet fraction;
- the dashed horizontal plotting guides have no verified operational meaning.

The relation to fully entangled fraction is stated only for the initial Werner state, where the optimization reduces to the largest Bell weight.

## 3. Archived metadata and data coverage are incomplete

- The archived NPZ files do not contain complete BdG metadata. `U/t=75` and the other current model settings were established by numerical reproduction, not by an embedded provenance record.
- No raw high-regime concurrence NPZ is supplied. Mandatory Figure B was visually inspected and its observable was reproduced from archived Green's functions and the stated channel, but an original high-regime concurrence array is unavailable for a direct file-to-file comparison.
- Raw two-qubit density matrices are not archived. Validation was performed on matrices reconstructed from the saved initial Green's functions.
- The current `config.py` alone cannot establish the settings of every old figure; the reproduction tests and actual array grids were used instead.

## 4. The main pipeline has parameter-routing and channel-consistency risks

`main.py` evolves concurrence/Holevo states with command-line noise arguments, but fidelity uses separate `FID_*` configuration values. Changing only `--gammaA`, `--GammaA`, and their B counterparts can therefore produce different physical channels for different observables. The archived arrays themselves were checked against their stated parameters, and the new sweep used one explicit parameter set for every observable, but the general CLI issue remains.

The standalone Green's-function envelope is also a visualization aid, not a substitute for the two-qubit Kraus evolution. Manuscript conclusions were based on the density-matrix/Kraus arrays.

## 5. Overdamped formulas are inconsistent across modules

`fidelity.py` implements the exact hyperbolic overdamped continuation. `noise_channels.py` and `greens_functions.py` use a critically damped expression throughout the non-oscillatory regime. Both reported regimes are underdamped, so this inconsistency does not alter the presented results. No overdamped conclusions are made.

## 6. Interaction-study scope is limited

- The new numerical dependence was calculated only at `U/t=40,75,115`, with fixed `Delta_t/t=4.5`, `V_SO=0`, `n=1.875`, and `N_k=100`.
- Three points do not determine a continuous monotonic law, an optimal interaction, or asymptotic large-`U` behavior.
- The observed ordering changes with spatial separation. No universal enhancement or suppression is claimed.
- `U/t=150` was excluded after tested solver outputs failed the `1e-8` maximum-residual criterion. The source's `converged` flag alone is insufficient because it does not impose a strict residual check; one seed produced `converged=True` with residual approximately `5.26e-4`.
- Momentum-grid convergence, denser `U` sampling, nonzero SOC, and filling dependence remain unverified.

## 7. Angular evidence is restricted

Only `0,15,30,45,60,75,90` degrees were sampled. The polar plot closes the first-quadrant data with a line and does not provide measurements over the rest of the circle. The angular Holevo heatmap is fixed at `r=0`; its identical rows do not demonstrate finite-radius isotropy. Full-circle behavior and continuous angular extrema remain unverified beyond the stated lattice-symmetry checks.

## 8. Dense-coding interpretation is protocol specific

The calculated quantity is the Holevo quantity of four equal-probability Pauli encodings applied after the shared-state evolution. It is not an optimization over arbitrary encodings, a one-shot accessible-information calculation, or a simulation of additional noise during encoding and transmission. The approach to one bit is a classical-baseline limit, not surviving quantum advantage.

## 9. Unreported repository features

Independent dephasing and common-bath dephasing are implemented, but no selected mandatory figure establishes their results. They are described as available methods only. No common-environment amplitude-damping result is claimed; the evolution routine explicitly rejects that combination.

## 10. Conflicting supplied drafts

The two pasted LaTeX drafts disagree on title/authors, SOC, momentum-grid size, and whether the channel is common or independent. The expanded first draft was retained as the structural baseline; numerical statements were instead governed by the current repository and archived arrays. Claims from the shorter draft such as `V_SO=0.4t`, `N_k=60`, and a common amplitude-damping channel were not imported.

## 11. LaTeX compilation and original support files

No LaTeX compiler was available, so compilation and visual PDF layout remain unverified. The original `Delta_ks_heatmaps.png`, `figs/greens_angular.png`, and `figs/concurrence_U_theta.png` were absent; they were regenerated from the current code. Their new numerical content is verified, but fidelity to the author's missing original images cannot be assessed.
