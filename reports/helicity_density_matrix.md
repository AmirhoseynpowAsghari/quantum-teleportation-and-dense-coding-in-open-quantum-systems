# Density matrix and teleportation at nonzero spin–orbit coupling

## 1. Short answer

Yes, a two-qubit density matrix can be constructed when \(V_{\mathrm{SO}}\neq0\), but the Oh–Kim scalar \(g,f\) substitution is not generally sufficient. One must retain \(2\times2\) normal and anomalous correlator matrices, apply Wick's theorem, and normalize only after conditioning on detecting one electron at each of two locations.

There are two legitimate but operationally different choices:

1. **Original-spin qubits.** Transform the helicity correlators back to the original \(\uparrow,\downarrow\) basis. The result is a general two-spin state, usually not an Oh–Kim \(X\) state or a Werner state. Its concurrence and teleportation fidelity describe ordinary spin qubits.
2. **Helicity-pseudospin qubits.** Retain band labels \(s=\pm1\) as the two internal states. Concurrence and teleportation are algebraically defined in exactly the same two-qubit way, but Bell measurements, Pauli corrections, and local unitaries are then **helicity-pseudospin operations**. They are not automatically physical spin operations.

Thus, “spin is not a good quantum number” does **not** imply “spin entanglement is undefined.” It implies that spin-singlets, spin triplets, and spin-off-diagonal normal propagation can mix. Entanglement always requires a precise choice of local modes.

The implementation is in `helicity_density_matrix.py`; the executed checks and demonstration are in `reports/helicity_density_matrix_validation.json` and `data/helicity_density_matrix_demo.npz`. No archived \(V_{\mathrm{SO}}=0\) array is changed.

## 2. Pairing and helicity conventions

Hutchinson, Hirsch, and Marsiglio define [1]

\[
c_{\mathbf ks}^{\dagger}
=\frac{c_{\mathbf k\uparrow}^{\dagger}
+s e^{i\theta_{\mathbf k}}c_{\mathbf k\downarrow}^{\dagger}}{\sqrt2},
\qquad
e^{i\theta_{\mathbf k}}
=\frac{\sin k_y-i\sin k_x}{\lambda_{\mathbf k}},
\]

\[
\lambda_{\mathbf k}=\sqrt{\sin^2k_x+\sin^2k_y},
\qquad s=\pm1.
\]

It is convenient to write \(c_{\mathbf k\sigma}=\sum_sU_{\mathbf k,\sigma s}c_{\mathbf ks}\), with

\[
U_{\mathbf k}=\frac1{\sqrt2}
\begin{pmatrix}
1&1\\
e^{i\theta_{\mathbf k}}&-e^{i\theta_{\mathbf k}}
\end{pmatrix},
\qquad U_{-\mathbf k}=ZU_{\mathbf k},
\]

where \(Z=\operatorname{diag}(1,-1)\), because \(e^{i\theta_{-\mathbf k}}=-e^{i\theta_{\mathbf k}}\).

The model pairs time-reversed electrons within the same helicity band. Its full gap is

\[
\Delta_{\mathbf ks}=s e^{-i\theta_{\mathbf k}}\bar\Delta_{\mathbf ks}.
\]

The repository's `bdg["Delta_ks"]` is the even scalar \(\bar\Delta_{\mathbf ks}\). At \(T=0\), Eq. (29) of Ref. [1] gives \(b_{\mathbf ks}=\bar\Delta_{\mathbf ks}/(2E_{\mathbf ks})\). Up to an irrelevant common sign, the momentum correlators are therefore

\[
N^{\mathrm h}_{\mathbf k}
=\begin{pmatrix}v_{\mathbf k+}^2&0\\0&v_{\mathbf k-}^2\end{pmatrix},
\qquad
F^{\mathrm h}_{\mathbf k}
=\begin{pmatrix}
e^{-i\theta_{\mathbf k}}\bar\Delta_{\mathbf k+}/(2E_{\mathbf k+})&0\\
0&-e^{-i\theta_{\mathbf k}}\bar\Delta_{\mathbf k-}/(2E_{\mathbf k-})
\end{pmatrix}.
\]

Here \(N_{\mathbf k,ss'}=\langle c_{\mathbf ks}^{\dagger}c_{\mathbf ks'}\rangle\) and \(F_{\mathbf k,ss'}=\langle c_{\mathbf ks}c_{-\mathbf ks'}\rangle\). The phase \(s e^{-i\theta_{\mathbf k}}\) is essential: it gives the required off-TRIM relation \(F_{-\mathbf k}=-F_{\mathbf k}\). At a time-reversal-invariant momentum with \(\lambda_{\mathbf k}=0\), the nondegenerate same-band pair amplitude vanishes by Pauli exclusion. The arbitrary helicity gauge at a zero-SOC degeneracy is handled separately in the code.

For original-spin correlators, the creation–annihilation convention gives

\[
N^{\mathrm{spin}}_{\mathbf k}
=U_{\mathbf k}^{*}N^{\mathrm h}_{\mathbf k}U_{\mathbf k}^{T},
\qquad
F^{\mathrm{spin}}_{\mathbf k}
=U_{\mathbf k}F^{\mathrm h}_{\mathbf k}U_{-\mathbf k}^{T}.
\]

Using \(UNU^\dagger\) for the first expression would correspond to a different index convention and gives the wrong phase for \(N_{\uparrow\downarrow}=\langle c_\uparrow^\dagger c_\downarrow\rangle\).

## 3. Real-space correlators and conditional state

Let \(A\) and \(B\) be two positions and \(\mathbf r=\mathbf R_B-\mathbf R_A\). For either internal basis,

\[
N_{AB}(\mathbf r)
=\frac1{N_k^2}\sum_{\mathbf k}e^{i\mathbf k\cdot\mathbf r}N_{\mathbf k},
\qquad
F_{AB}(\mathbf r)
=\frac1{N_k^2}\sum_{\mathbf k}e^{-i\mathbf k\cdot\mathbf r}F_{\mathbf k}.
\]

The local matrices are \(N_A=N_B=N_{AB}(0)\), and \(N_{BA}=N_{AB}^{\dagger}\). Direct anticommutation at the same ordered pair of positions gives

\[
F_{BA,\beta\alpha}=\langle c_{B\beta}c_{A\alpha}\rangle
=-F_{AB,\alpha\beta}.
\]

At \(A=B\), \(F_{AB}(0)=-F_{AB}^{T}(0)\) exactly. The implementation removes only the finite-quadrature symmetric residue at this point; it does not clip density-matrix eigenvalues.

For a Gaussian BdG mean-field state, Wick's theorem gives the unnormalized two-detection matrix \(\widetilde\rho_{AB}\). In the basis \(\{|00\rangle,|01\rangle,|10\rangle,|11\rangle\}\),

\[
\boxed{
\begin{aligned}
\widetilde\rho_{\alpha\beta,\alpha'\beta'}
={}&N_{B,\beta'\beta}N_{A,\alpha'\alpha}
-N_{BA,\beta'\alpha}N_{AB,\alpha'\beta}\\
&+F_{AB,\alpha'\beta'}^{*}F_{AB,\alpha\beta}.
\end{aligned}}
\]

The three terms are respectively the direct, normal-exchange, and anomalous-pair contributions. The creation-side anomalous factor is \(F_{AB,\alpha'\beta'}^*\), equivalently \(-F_{BA,\beta'\alpha'}^*\); this sign is necessary to recover the negative singlet coherence.

The conditional state and its detection-sector weight are

\[
P_{AB}=\operatorname{Tr}\widetilde\rho_{AB},
\qquad
\rho_{AB}=\widetilde\rho_{AB}/P_{AB}.
\]

This is a conditional one-electron-at-\(A\)-and-one-at-\(B\) state, not the complete variable-particle-number BCS state and not a canonical fixed-total-electron-number reduction.

### Zero-SOC check

If \(N_A=N_B=n_0I\), \(N_{AB}=N_{BA}=n_0g(\mathbf r)I\), and \(F_{AB}=n_0f(\mathbf r)i\sigma_y\), the boxed formula reduces exactly to

\[
\rho_X=\frac1{4-2g^2+2f^2}
\begin{pmatrix}
1-g^2&0&0&0\\
0&1+f^2&-(g^2+f^2)&0\\
0&-(g^2+f^2)&1+f^2&0\\
0&0&0&1-g^2
\end{pmatrix},
\]

where \(g^2=|g|^2\) and \(f^2=|f|^2\). The numerical implementation reproduces this matrix to \(5.56\times10^{-17}\) maximum elementwise error in the executed zero-SOC test.

## 4. Entanglement in either qubit basis

After normalization, \(\rho_{AB}\) is a general two-qubit density matrix. Concurrence is calculated using Wootters' spin-flip formula, not the repository's \(X\)-state shortcut:

\[
\widetilde R=(\sigma_y\otimes\sigma_y)\rho_{AB}^{*}
(\sigma_y\otimes\sigma_y),
\]

\[
C=\max\{0,\lambda_1-\lambda_2-\lambda_3-\lambda_4\},
\]

where the \(\lambda_i\) are the decreasingly ordered square roots of the eigenvalues of \(\rho_{AB}\widetilde R\). In the helicity calculation, \(\sigma_y\) acts on **band pseudospin**. The algebraic formula is unchanged; its physical interpretation follows the chosen modes.

Concurrence is basis-dependent between original spin and helicity-band qubits because the helicity transformation is momentum dependent and is not a pair of fixed local spin rotations. Numerical equality of the two would require evidence; it must not be assumed.

## 5. Teleportation fidelity

Let \(F_{\mathrm h}\) be the overlap with a chosen reference Bell state in the same qubit basis. For the fixed protocol referenced to that Bell state,

\[
\overline{\mathcal F}_{\mathrm{ref}}=\frac{2F_{\mathrm h}+1}{3}.
\]

This is distinct from the local-unitary-optimized result for a general non-Werner state. Define

\[
T_{ij}=\operatorname{Tr}[\rho_{AB}(\sigma_i\otimes\sigma_j)],
\qquad
\mathfrak f=\max_{U_A,U_B}
\langle\Phi^+|(U_A\otimes U_B)\rho_{AB}(U_A\otimes U_B)^\dagger|\Phi^+\rangle.
\]

For two qubits,

\[
\mathfrak f=\frac{1+\sum_{i=1}^{3}\nu_i(T)}4,
\qquad
\overline{\mathcal F}_{\mathrm{opt}}=\frac{2\mathfrak f+1}{3},
\]

where \(\nu_i(T)\) are singular values. The standard classical benchmark is exceeded only when

\[
\overline{\mathcal F}_{\mathrm{opt}}>2/3
\quad\Longleftrightarrow\quad
\mathfrak f>1/2
\quad\Longleftrightarrow\quad
\sum_i\nu_i(T)>1.
\]

For a Werner resource, fixed and optimized formulas coincide. The generalized Rashba state need not be Werner, so the implementation reports \(F_{\mathrm h}\), \(\mathfrak f\), fixed-reference fidelity, optimized fidelity, and concurrence separately.

## 6. What helicity-band teleportation means operationally

For band-labeled qubits the code uses the canonical but nonlocal modes

\[
d_{i,s}=\frac1{N_k}\sum_{\mathbf k}e^{i\mathbf k\cdot\mathbf R_i}c_{\mathbf ks}.
\]

One then assumes independently addressable \(A,B\) band-pseudospin modes, an independently prepared input pseudospin qubit \(Q\), an ideal Bell measurement on \(Q,A\) in the band basis, and ideal classical feedforward to \(B\). These are mathematical protocol assumptions. The calculation does **not** establish a microscopic helicity Bell-measurement device, individual Cooper-pair addressability, or experimentally available band-local Pauli corrections.

If those operations are not available, the result remains a well-defined entanglement/FEF calculation for the selected modes but is not a demonstrated experimental teleportation protocol. For ordinary spin teleportation at nonzero SOC, use `basis="spin"`, not `basis="band"`.

## 7. Executed methodological demonstration

The script solves the self-consistency equations independently at \(V_{\mathrm{SO}}/t=0\) and \(0.5\), with \(U/t=75\), \(\Delta_t/t=4.5\), \(n=1.875\), \(T=0\), and an inversion-symmetric \(101\times101\) mesh. The respective maximum residuals are \(2.91\times10^{-12}\) and \(3.35\times10^{-12}\). Six lattice separations, two bases, and two SOC values give 24 conditional matrices.

All checks pass:

- maximum zero-SOC Oh–Kim matrix error: \(5.55\times10^{-17}\);
- maximum trace error: \(1.11\times10^{-16}\);
- maximum Hermiticity error: zero;
- minimum eigenvalue: \(-1.57\times10^{-32}\) (numerical zero);
- maximum same-site singlet error: \(1.11\times10^{-16}\);
- maximum off-TRIM anomalous inversion residual: \(3.30\times10^{-15}\);
- maximum Werner/Bell metric error: \(2.67\times10^{-15}\);
- fixed-protocol identity error for a non-\(X\) nonzero-SOC state: zero to printed precision.

For example, at \(V_{\mathrm{SO}}/t=0.5\), \(\mathbf r=(1,0)\), the original-spin matrix is, to eight decimal places,

\[
\rho_{\mathrm{spin}}\approx
\begin{pmatrix}
0.24961259&-0.00049305&0.00049305&0.00023973\\
-0.00049305&0.25038741&-0.00101455&-0.00049305\\
0.00049305&-0.00101455&0.25038741&0.00049305\\
0.00023973&-0.00049305&0.00049305&0.24961259
\end{pmatrix}.
\]

The code retains the full complex matrix. This example has \(C=0\), \(F_{\mathrm h}=0.251401965\), \(\mathfrak f=0.251881235\), and \(\overline{\mathcal F}_{\mathrm{opt}}=0.501254157\). The band-basis state at the same separation is different, with \(F_{\mathrm h}=0.249609387\) and \(\mathfrak f=0.250454072\). These are only demonstration values at one mesh and parameter set. They are not a converged \(V_{\mathrm{SO}}\)-dependence study and do not support a general claim that SOC suppresses entanglement.

## 8. Usage

From the repository root:

```bash
.venv/bin/python analysis/validate_helicity_density_matrix.py
```

For another already converged BdG solution:

```python
from helicity_density_matrix import density_at_separation, resource_metrics

spin_state = density_at_separation(bdg, [1, 0], basis="spin")
band_state = density_at_separation(bdg, [1, 0], basis="band")

rho_spin = spin_state["rho"]
rho_band = band_state["rho"]
print(resource_metrics(rho_spin))
print(resource_metrics(rho_band))
```

`spin_state["weight"]` and `band_state["weight"]` retain the pre-normalization detection-sector weights. The NPZ stores matrices, correlators, weights, concurrence, fixed-reference fidelity, FEF, and optimized fidelity for both bases.

## 9. Scope requiring further work

- The demonstration is not a production mesh-convergence or seed-stability study.
- The construction assumes a spatially uniform Gaussian BdG state with zero center-of-mass-momentum intraband pairing.
- Disorder, finite-momentum pairing, interband pairing, non-Gaussian correlations, or position-dependent pairing require the corresponding full correlators and, where Wick's theorem fails, more than the present factorization.
- Physical Bell-measurement feasibility and band/spin addressability are separate experimental questions.
- The legacy magnitude-amplitude dynamics are not regenerated by this initial-state calculation.

## References

1. J. Hutchinson, J. E. Hirsch, and F. Marsiglio, *Enhancement of superconducting \(T_c\) due to the spin-orbit interaction*, Phys. Rev. B **97**, 184513 (2018), especially Eqs. (16)–(29), [arXiv:1801.08988](https://arxiv.org/html/1801.08988).
2. S. Oh and J. Kim, *Entanglement of electron spins in superconductors*, Phys. Rev. B **71**, 144523 (2005), [arXiv:quant-ph/0406013](https://arxiv.org/pdf/quant-ph/0406013).
3. W. K. Wootters, *Entanglement of formation of an arbitrary state of two qubits*, Phys. Rev. Lett. **80**, 2245 (1998).
4. M. Horodecki, P. Horodecki, and R. Horodecki, *General teleportation channel, singlet fraction, and quasidistillation*, Phys. Rev. A **60**, 1888 (1999).
