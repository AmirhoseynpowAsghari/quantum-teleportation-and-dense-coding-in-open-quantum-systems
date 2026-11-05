# Manuscript modification report

## Baseline and scope

- **Completed source:** `manuscript.tex`.
- **Analyzed repository commit:** `6ef5c49faf09494a25777d85591a6be0cf732446`.
- The expanded first LaTeX draft supplied in the prompt was used as the baseline because it contains the extended Hubbard-interaction discussion and stops at the placeholder immediately before the conclusion. Its title, authors, REVTeX class, section notation, core model equations, acknowledgment, and manual bibliography system were retained.
- The second, shorter supplied draft was treated as a conflicting earlier version, not merged indiscriminately. Its different title/author, `V_SO=0.4t`, `N_k=60`, and common-channel narrative do not match the current archived data analyzed here.
- The unfinished placeholder figure (`image.png`) was removed and replaced by the requested scientific results.

## Sections added

1. **Fidelity, teleportation, and Pauli-ensemble Holevo information** (`sec:information_measures`)
   - Squared Uhlmann fidelity and its two distinct references.
   - Exact equality of pure-singlet-referenced fidelity and singlet fraction.
   - Fully entangled fraction and the conditions for relating it to average teleportation fidelity.
   - Equal-probability Pauli encoding on subsystem A, base-two entropies, and the derivation of `chi_P = 1 + S_2(rho_B) - S_2(rho_AB)`.
2. **Numerical grids and validation** (`sec:validation`)
   - Exact angular, radial, and temporal grids for the two archived regimes.
   - Reproduction errors, state-validation checks, and BdG provenance.
3. **Results and Discussion**, containing the original spatial-concurrence subsection and new subsections on:
   - Non-Markovian concurrence and singlet fidelity.
   - Fidelity dynamics at lower coupling.
   - Angular dependence of the singlet fraction.
   - Holevo information and dense coding.
   - Hubbard interaction dependence of fidelity and Holevo information.

## Existing material modified

- **Abstract and introduction:** replaced interchangeable uses of state fidelity, teleportation fidelity, and dense-coding capacity with the quantities actually implemented; added the completed interaction study and the implementation qualification.
- **Self-consistent BdG subsection:** replaced the unreproduced gap parameters with converged values from the current implementation. Added chemical potentials, residuals, fixed parameters, and units.
- **Green's-function subsection:** distinguished the exact signed BdG identity from the unsigned-amplitude implementation in `greens_functions.py`. Corrected the earlier claim that taking the absolute value after integration necessarily changes this density matrix: only squared normalized ratios enter it. The consequential implementation difference is loss of the gap sign before Brillouin-zone integration.
- **Density-matrix subsection:** corrected the undefined citation key `OhKim2005` to `Oh2005`; clarified that the concurrence routine is an X-state reduction, not a general-state Wootters implementation; added the exact zero-separation singlet result.
- **Open-system subsection:** explicitly distinguished lowercase coupling `gamma` from uppercase memory rate `Gamma`; limited the exact overdamped-continuation claim to `fidelity.py`, because the main evolution routines use a different critical-damping fallback.
- **Spatial-concurrence subsection:** removed the nonconverged `U/t=150` example and updated the interpretation to match the reproduced curves. The discussion now notes that stronger `U` shortens the first entangled interval in the sampled cases but can enhance finite-radius revival peaks.
- **Conclusion:** rewritten to reflect the verified fidelity/Holevo/angular/interaction results and remaining implementation limitations.
- **Bibliography:** retained the two original entries and added eight verified references required for teleportation, dense coding, concurrence, Uhlmann fidelity, fully entangled fraction, and non-Markovian dynamics. Every entry is cited.

## Equations added or corrected

- Squared Uhlmann fidelity; initial-state and singlet-referenced definitions.
- Pure-reference reduction to singlet fraction and the initial-Werner relation `F_s=(1+3p)/4`.
- Fully entangled fraction and the cited qubit teleportation relation, with optimization limitations stated.
- Pauli encoding ensemble, averaged state, base-two Holevo definition, and entropy reduction.
- `r=0` amplitude-damping identities `F_s=p(t)` and `C=p(t)`.
- Corrected implementation statements concerning the signed coherence factor, absolute ratios, concurrence, and overdamped evolution.
- Reproduced self-consistent numerical parameters for `U/t=40,75,115`; no curve fitting or phenomenological equations were introduced.

## Figure order

| Order | Label | File | Role |
|---|---|---|---|
| 1 | `fig:delta_ks_heatmaps` | `Delta_ks_heatmaps.png` | Self-consistent gap versus `U` |
| 2 | `fig:greens_angular` | `figs/greens_angular.png` | Spatial/angular correlators |
| 3 | `fig:concurrence_U_theta` | `figs/concurrence_U_theta.png` | Initial concurrence and interaction dependence |
| 4 | `fig:C_high_heatmap` | Mandatory Figure B | High-`gamma`, low-`Gamma` concurrence dynamics |
| 5 | `fig:F_high_heatmap` | Mandatory Figure A | Matching singlet-fraction dynamics |
| 6 | `fig:F_low_time` | Mandatory Figure E | `gamma=1.5`, `Gamma=0.1` temporal slices |
| 7 | `fig:F_angular` | Mandatory Figure F | Initial angular singlet fraction |
| 8 | `fig:holevo_spatial` | Mandatory Figure D | Spatial Holevo profiles |
| 9 | `fig:holevo_angular` | Mandatory Figure C | Zero-separation angular/time consistency |
| 10 | `fig:U_fidelity_holevo` | `figs/U_dependence_fidelity_holevo.png` | New converged `U`-sweep |

Each mandatory figure appears **exactly once**, has a unique label and a scientific caption, and is referenced in the surrounding argument. The first three source figures were absent from the supplied repository and were regenerated from the current implementation rather than treated as pre-existing evidence.

## New reproducibility artifacts

- `analysis/reproduce_and_validate.py`: archived-data checks, new interaction sweep, and regenerated supporting figures.
- `analysis/check_latex.py`: static manuscript checks.
- `data/hubbard_U40_U75_U115_amp_independent_g1.50_G0.100_observables.npz`: full seven-angle interaction-sweep arrays and metadata.
- `reports/numerical_validation.json`: machine-readable validation results.
- `reports/reproduction_stdout.txt`: completed execution log.
- `reports/latex_static_validation.json`: static LaTeX audit results.

No core simulation module was silently rewritten. The implementation conflicts are documented in `missing_evidence_and_limitations.md`.
