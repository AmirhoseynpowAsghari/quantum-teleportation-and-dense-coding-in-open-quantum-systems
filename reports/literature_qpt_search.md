# Literature search: QPT signatures via entanglement in (Rashba + correlated-hopping) superconductors

Deep search (2026-11-05) covering six facets of our problem:
(i) concurrence derivatives as QPT detectors, (ii) the arXiv:2011.14038 research
program, (iii) two-particle-density-matrix concurrence in SCs and Rashba 2DEGs,
(iv) Lifshitz transitions as quantum phase transitions, (v) correlated-hopping
(bond-charge, our Delta_t channel) models, (vi) entanglement sudden death at
QPTs.

## A. Concurrence derivatives as QPT detectors (our dC/dV_SO method)

1. **Osterloh, Amico, Falci, Fazio, "Entanglement and quantum phase
   transitions in spin chains", PRL 89, 087902 (2002).** The classic:
   dC/dlambda diverges logarithmically (~ -0.2702 ln N) at the critical point;
   position of the extremum scales as g_m - g_c ~ N^-theta with theta = 1/nu.
2. **"Renormalization of concurrence", arXiv:0710.5843 (2007).** QRG study of
   the transverse-field Ising chain: minimum of dC/dg diverges as N^(1/nu);
   entanglement exponent theta = 1/nu; data collapse of dC/dg.
3. **"Quantum Phase Transition and Entanglement in Topological Quantum Wires",
   arXiv:1706.01637 / Sci. Rep. 5, 10764 (2017).** SSH/Kitaev: dC/dlambda
   diverges logarithmically at the topological QPT (universal for 1D two-band
   fermions); ALSO finds a second, purely-entanglement transition with
   sudden-death-like nonanalyticity of C that does *not* close the spectral
   gap -- directly relevant to our legacy ESD/EB events.
4. **"Entanglement scaling at first order quantum phase transitions",
   New J. Phys. 20, 063047 (2018) (iopscience aab2db).** DMRG finite-size
   scaling: at 1QPTs the concurrence stays continuous while its FIRST
   derivative is discontinuous; the scaled derivative -- not C itself --
   collapses. Supports analyzing dC/dV, as we do.
5. **M.-F. Yang, "Entanglement and Quantum Phase Transition Revisited",
   arXiv:quant-ph/0407226 (2004).** CRITICAL CAVEAT: an exactly solvable spin
   model with a discontinuity in dC but NO QPT -- "no one-to-one
   correspondence between QPT and nonanalyticity of concurrence". This is the
   literature counterweight to the one-to-one claim of arXiv:2011.14038 and
   the reason our PR #15 adds independent (thermodynamic/scaling) tests.
6. **"Concurrence in collective models", arXiv:quant-ph/0603108.** At a
   first-order transition the rescaled concurrence can remain continuous and
   singular only in its derivatives; classification must use derivatives.

## B. The arXiv:2011.14038 program (the paper the user cited)

7. **Ebrahimian, Khosrojerdi, Afzali, arXiv:2011.14038 -> published as
   Phys. Rev. B 106, 024512 (2022), "Tuning of quantum entanglement in a
   superconductor with transition-metal and rare-earth impurities: Effect of
   potential scattering on quantum phase transitions".** (The journal version
   of the cited preprint; T-matrix/Green's-function Werner state.)
8. **Afzali, Fahimi, Dehghan, Int. J. Quantum Inf. 15, 1750006 (2017),
   "Quantum Entanglement and Correlation Lengths of a S-wave Superconductor in
   the Presence of a Weak Constant External Potential"**
   (link.springer.com/s10773-017-3296-8): same space-spin density matrix,
   concurrence vs potential and distance r; entanglement lengths.
9. **"Quantum Information Aspects on Bulk and Nano Interacting Fermi System:
   Spin-Space Density Matrix Approach", arXiv:1603.01645 (2016)** and Phys.
   Lett. A (ScienceDirect S0375960116305266): d-wave/nano-grain SC, X-state
   two-electron space-spin density matrix from Green's functions, concurrence
   sensitive to the gap -- the same X-state construction as our
   `greens_functions.py` legacy path.

## C. Two-particle-density-matrix / concurrence in SCs and Rashba systems

10. **"Quantum correlations of a two-dimensional electron gas with Rashba
    spin-orbit coupling", Phys. Rev. B 108, 115408 (2023).** Two-particle
    density matrix of a Rashba 2DEG -> exchange hole, concurrence, entanglement
    of formation, quantum discord. The closest methodological neighbor to our
    `helicity_density_matrix.py` (but noninteracting, no pairing/QPT).
11. **"Entanglement modes and topological phase transitions in
    superconductors", arXiv:1503.05154 (2015).** 2D p-wave + Rashba + Zeeman
    lattice SC; von Neumann entropy, entanglement spectrum, fidelity (and
    fidelity spectrum) detect/ distinguish topological transitions -- an
    entanglement-detector paper for SOC superconductors.
12. **"Integer effects in the entanglement and spin fluctuations of a quantum
    Hall system with Rashba interactions", arXiv:1601.07611 (2016).**
    Nonanalytic kinks (discontinuous first derivatives) of spin-orbit
    entanglement entropy at Fermi-level crossings in a Rashba 2DEG -- a
    Lifshitz/level-crossing entanglement singularity of the kind we report.
13. **"Ground-State Entanglement of the BCS Model" (2005) and related BCS
    concurrence literature (Martin-Delgado; Dunning et al.; average local
    concurrence vs order parameter).** Concurrence of BCS states related to
    the order parameter and condensation threshold; entanglement of the
    maximum geminal defines the disappearance point of superconductivity.

## D. Lifshitz transition as a quantum phase transition

14. **G. E. Volovik, "Topology of quantum phase transitions",
    cond-mat/0601372 (2006)** and **"Exotic Lifshitz transitions in
    topological materials", Phys.-Usp. 61, 1 (2018).** FS = vortex in
    (p,omega) space; Lifshitz transitions are topological QPTs without
    symmetry breaking; Ehrenfest order depends on dimensionality/geometry
    (3D band edge ~2.5 order; 2D band edge -> DOS jump = 2nd order in our
    case, consistent with the N(E_F) kink of PR #15).
15. **"Fermi surface as a quantum critical manifold: gaplessness, order
    parameter, and scaling in d dimensions", arXiv:2504.07014 (2025).**
    FS appearance/restructuring = Lifshitz QPT; scaling and hyperscaling
    relations hold; Ehrenfest classification per dimension (2nd order with
    divergent susceptibilities in 1D, 3rd kind in 2D/3D for the generic
    case).
16. **"The Lifshitz nature of the transition between the gap and gapless
    states in the Abrikosov-Gor'kov theory of a superconductor",
    SciPost Phys. Core 5, 009.** Proves the AG gap-gapless transition is a
    Lifshitz (2.5-order) transition -- connects the impurity-superconductor
    context of arXiv:2011.14038 to Lifshitz QPT language.
17. **Experiments where Lifshitz coincides with SC onset/QPT (analogues of
    our pairing-onset alignment):**
    - "Lifshitz transition enabling superconducting dome around a
      charge-order critical point", Sci. Adv. (2024), TiSe2
      (sciadv.adl3921).
    - "Pressure-induced Lifshitz and quantum phase transitions in
      electron-doped cuprate superconductors", PNAS (2026),
      pnas.2622416123.
    - "Quenched nematic criticality and two superconducting domes in an
      iron-based superconductor", Nat. Phys. (2019), s41567-019-0694-2
      (FeSe, Lifshitz between the domes).
    - "Re-entrant superconductivity through a quantum Lifshitz transition in
      twisted trilayer graphene", arXiv:2104.13920.
    - "Quantum Lifshitz points in an altermagnetic superconductor",
      Springer (2025), s43673-025-00178-2 (first/second-order quantum
      Lifshitz points at T=0).

## E. Correlated hopping (our Delta_t channel): bond-charge models

18. **"Bond-charge interaction in the extended Hubbard chain", Phys. Rev. B
    54, 1302 (1996), cond-mat/9603164.** Classic density-dependent hopping
    (correlated hopping X): CDW-SDW transitions, succession of critical
    points; X weakens the first-order character of the transition.
19. **"Entanglement in extended Hubbard models and quantum phase
    transitions", Phys. Rev. B 75, 165106 (2007).** Bipartite/multipartite
    entanglement at the QPTs of a bond-charge extended Hubbard model
    (qubit and qutrit on-site degrees of freedom).
20. **Quantum discord and classical correlations in the bond-charge extended
    Hubbard model, arXiv:1110.6381 (2011).** Compares QD/CC and their
    derivatives in direct and reciprocal space to dissect two-point driven
    QPTs of the correlated-hopping Hubbard model -- the same
    "derivatives of correlation measures" logic as our dC/dV analysis, in
    exactly our interaction class (without SOC/pairing).
21. **"Tricritical behavior in the extended Hubbard chains",
    cond-mat/9909277 (1999).** Correlated hopping creates BCDW/BSDW phases
    and tricritical points (first/second-order structure) in the
    bond-charge model.
22. **"Correlated hopping of electrons: effect on the Brinkman-Rice
    transition ..." (Gutzwiller).** Correlated hopping shifts the
    Mott/Brinkman-Rice QPT -- evidence that the X channel changes
    transition locations (our V* displacement with Delta_t).

## F. Entanglement sudden death during QPTs

23. **arXiv:1706.01637 (see #3):** an SSH "entanglement transition" with
    sudden-death nonanalyticity of C, explicitly NOT closing the spectral
    gap -- the closest published analogue of our legacy ESD/EB events.
24. **Yu & Eberly, "Finite-time disentanglement via spontaneous emission",
    Opt. Commun. 264, 393 (2006)** (and follow-ups): the original SDE/ESD
    phenomenon the legacy curves exhibit as a function of V_SO.

## Where our work sits (gap = opportunity)

No paper found combines ALL of: (a) lattice Hubbard + Rashba SOC + bond-charge
correlated hopping, (b) helicity-band BdG self-consistency, (c) exact
conditional two-detection (non-X-state) concurrence in spin and band bases,
(d) dense 151-point dC/dV_SO QPT diagnostics with Lifshitz/pairing/λ/finite-size
tests. Closest pieces: #10 (Rashba concurrence, no interaction), #19/#20
(correlated hopping + entanglement, no SOC), #7-#9 (SC concurrence with
impurities, Green's-function X-state), #3 (dC divergence at topological QPT,
1D).
