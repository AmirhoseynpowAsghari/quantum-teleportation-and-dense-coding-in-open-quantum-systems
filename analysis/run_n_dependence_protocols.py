#!/usr/bin/env python3
"""Electron-filling (n) dependence of BdG pairing, spin vs. helicity-band
conditional states, and non-Markovian open-system quantum communication protocols.

Evaluates at V_SO/t = 0.5, Delta_t/t = 4.5, T = 0 on periodic Brillouin-zone
meshes across BOTH the original-spin basis ('spin') and the helicity-band
('band') basis:
  1. Full-band sweep n in [0.02, 1.98] (Nk = 201 periodic, cross-checked at
     Nk = 401): self-consistent (Delta_0, Delta_S, mu), linearized pairing
     eigenvalue lambda_+(n), helicity band polarization (n_+ - n_-)/n,
     static (t = 0) singlet fraction F_s, locally optimized teleportation
     fidelity F_opt, Wootters concurrence C, Pauli-ensemble Holevo capacity
     chi_P, entanglement-death radius r_c(n) along theta = 45 deg, and
     comparison across interactions U/t in {15, 25, 75, 115}.
  2. Non-Markovian amplitude-damping open-system dynamics at representative
     fillings n in {0.10, 0.50, 1.00, 1.40, 1.875} covering the dilute
     re-entrant nodal SC pocket, normal Fermi-liquid window, and high-density
     SC dome, in both 'spin' and 'band' bases, under:
       - low-noise regime (gamma=1.5, Gamma=0.1) at r = 0.30a (theta=45 deg)
         and r = 1.50754a, matching Figs. 8--12;
       - strong-memory regime (gamma=4.5, Gamma=0.01) at r = 0.30a and exact
         nearest-neighbor lattice vector r = (1, 0)a, matching Figs. 6--7.

Produces:
  - data/n_dependence_protocols.npz
  - reports/n_dependence_protocols.json
  - figs/n_dependence_phase_and_resources.png
  - figs/n_dependence_open_protocols.png

Run from the repository root:
    .venv/bin/python analysis/run_n_dependence_protocols.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fast_solver as fs
import helicity_density_matrix as hdm
from evolution import evolve_rho
from noise_channels import get_kraus_pair
from teleportation import optimized_teleportation_fidelity

FIGS = ROOT / "figs"
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
for d in (FIGS, DATA, REPORTS):
    d.mkdir(exist_ok=True)

DT = 4.5
VSO = 0.5
U_MAIN = 75.0
NK = 201
NK_CHECK = 401
LOW = dict(gamma=1.5, Gamma=0.1)
HIGH = dict(gamma=4.5, Gamma=0.01)
REP_FILLINGS = [0.10, 0.50, 1.00, 1.40, 1.875]

_Y2 = np.kron(hdm.PAULI_Y, hdm.PAULI_Y)


def solve_global_bdg(
    eps: np.ndarray,
    kern: np.ndarray,
    KX: np.ndarray,
    KY: np.ndarray,
    ssin: np.ndarray,
    nk: int,
    U: float,
    n: float,
) -> tuple[dict, np.ndarray, float, float, bool]:
    """Find the lowest-energy self-consistent BdG branch at fixed filling n."""
    nrm = fs.normal_state(eps, n)
    lam_c, lam_g, v = fs.lambda_exact_and_matrix(eps, kern, nk, DT, nrm["mu"], U)
    if U >= 50.0 and lam_c < 0.995:
        seeds = [[0.24, 1.8, nrm["mu"]], [0.08, -0.09, nrm["mu"]]]
    elif n < 0.35:
        seeds = [
            [0.08, -0.09, nrm["mu"]],
            [0.02, -0.025, nrm["mu"]],
            [1.5, -5.0, nrm["mu"]],
            [0.6, -2.5, nrm["mu"]],
            [0.05 * v[0], 0.05 * v[1], nrm["mu"]],
        ]
    else:
        seeds = [
            [0.24, 1.8, nrm["mu"]],
            [0.15, 1.2, nrm["mu"]],
            [1.5, -5.0, nrm["mu"]],
            [2.5, -7.5, nrm["mu"]],
            [0.6, -2.5, nrm["mu"]],
            [0.05 * v[0], 0.05 * v[1], nrm["mu"]],
            [0.5 * v[0], 0.5 * v[1], nrm["mu"]],
        ]
    _, nontriv = fs.solve_all(eps, kern, nk, DT, n, seeds, tol=1e-10, early=3)
    eN = nrm["e"]
    sc_cands = []
    for cand in nontriv:
        eS, _, _ = fs.sc_energy(cand, eps, nk, n)
        if eS <= eN + 1e-11:
            sc_cands.append((eS, cand))
    if sc_cands:
        _, best = min(sc_cands, key=lambda item: item[0])
        sc = True
    else:
        best = np.array([0.0, 0.0, nrm["mu"]], dtype=float)
        sc = False
    bdg = fs.bdg_dict(best, eps, KX, KY, ssin)
    res = float(np.max(np.abs(fs.residual_fast(best, eps, kern, nk, DT, n)))) if sc else 0.0
    return bdg, best, lam_c, res, sc


class CachedStates:
    """Fast vectorized evaluator of spin- and band-basis conditional states."""

    def __init__(self, bdg: dict, basis: str):
        self.bdg = bdg
        self.basis = basis
        self.N_k, self.F_k = hdm.momentum_correlators(bdg, basis=basis)
        self.KX = bdg["KX"]
        self.KY = bdg["KY"]
        self.N2 = float(self.KX.size)
        self.N_A = np.sum(self.N_k, axis=(0, 1)) / self.N2

    def rho_stack(self, r_vecs: np.ndarray) -> np.ndarray:
        """Compute normalized 4x4 conditional states for shape (M, 2) r_vecs."""
        r_vecs = np.asarray(r_vecs, dtype=float).reshape(-1, 2)
        phases = np.exp(1j * (
            r_vecs[:, 0, None, None] * self.KX[None, :, :]
            + r_vecs[:, 1, None, None] * self.KY[None, :, :]
        ))
        N_AB = np.einsum("mij,ijab->mab", phases, self.N_k) / self.N2
        F_AB = np.einsum("mij,ijab->mab", phases.conj(), self.F_k) / self.N2
        is_zero = np.all(r_vecs == 0.0, axis=1)
        if np.any(is_zero):
            F_AB[is_zero] = 0.5 * (F_AB[is_zero] - np.swapaxes(F_AB[is_zero], 1, 2))
        N_BA = np.conj(np.swapaxes(N_AB, 1, 2))

        rhos = np.empty((len(r_vecs), 4, 4), dtype=complex)
        N_A = self.N_A
        for idx in range(len(r_vecs)):
            nab, nba, fab = N_AB[idx], N_BA[idx], F_AB[idx]
            rho = (
                np.einsum("bd,ac->abcd", N_A, N_A)
                - np.einsum("bc,ad->abcd", nba, nab)
                + np.einsum("cd,ab->abcd", np.conj(fab), fab)
            ).reshape(4, 4)
            herm_err = float(np.max(np.abs(rho - rho.conj().T)))
            if herm_err > 1e-10:
                raise RuntimeError(f"Non-Hermitian rho ({herm_err:.3e})")
            rho = 0.5 * (rho + rho.conj().T)
            tr = float(np.trace(rho).real)
            if tr <= 0.0:
                raise RuntimeError(f"Nonpositive trace weight: {tr}")
            rhos[idx] = rho / tr
        return rhos


def f_opt_batch(rhos: np.ndarray) -> np.ndarray:
    """Exact locally optimized average teleportation fidelity from FEF."""
    flat = rhos.reshape(-1, 4, 4)
    paulis = (
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    )
    T = np.empty((len(flat), 3, 3), dtype=float)
    for i, si in enumerate(paulis):
        for j, sj in enumerate(paulis):
            op = np.kron(si, sj)
            T[:, i, j] = np.real(np.einsum("nab,ba->n", flat, op))
    svals = np.linalg.svd(T, compute_uv=False)
    fef = (1.0 + np.sum(svals, axis=-1)) / 4.0
    fopt = (2.0 * fef + 1.0) / 3.0
    return fopt.reshape(rhos.shape[:-2])


def observables_full(rho_t: np.ndarray) -> dict[str, np.ndarray]:
    """Compute C, Fs, F_fixed, F_opt, and chi_P on arbitrary shape (..., 4, 4)."""
    shape = rho_t.shape[:-2]
    flat = np.asarray(rho_t, dtype=complex).reshape(-1, 4, 4)

    # Concurrence
    flipped = _Y2 @ flat.conj() @ _Y2
    eigs_r = np.linalg.eigvals(flat @ flipped)
    if np.max(np.abs(eigs_r.imag)) > 1e-8:
        raise RuntimeError("Complex eigenvalues in concurrence")
    roots = np.sort(np.sqrt(np.clip(eigs_r.real, 0.0, None)), axis=-1)
    C = np.maximum(0.0, roots[:, -1] - np.sum(roots[:, :-1], axis=-1))

    # Singlet fraction & fixed-protocol fidelity
    Fs = np.real(np.einsum("a,nab,b->n", hdm.SINGLET.conj(), flat, hdm.SINGLET))
    F_fixed = (2.0 * Fs + 1.0) / 3.0

    # Optimized teleportation fidelity
    F_opt = f_opt_batch(flat)

    # Pauli-ensemble Holevo capacity chi_P = 1 + S(rho_B) - S(rho_AB)
    ev_ab = np.linalg.eigvalsh(flat)
    if np.min(ev_ab) < -1e-9:
        raise RuntimeError(f"Negative rho_AB eigenvalue: {np.min(ev_ab):.3e}")
    ev_ab = np.clip(ev_ab.real, 0.0, 1.0)
    nz_ab = ev_ab > 1e-15
    S_ab = -np.sum(np.where(nz_ab, ev_ab * np.log2(np.where(nz_ab, ev_ab, 1.0)), 0.0), axis=-1)

    rho_b = np.trace(flat.reshape(-1, 2, 2, 2, 2), axis1=1, axis2=3)
    ev_b = np.linalg.eigvalsh(rho_b)
    if np.min(ev_b) < -1e-9:
        raise RuntimeError(f"Negative rho_B eigenvalue: {np.min(ev_b):.3e}")
    ev_b = np.clip(ev_b.real, 0.0, 1.0)
    nz_b = ev_b > 1e-15
    S_b = -np.sum(np.where(nz_b, ev_b * np.log2(np.where(nz_b, ev_b, 1.0)), 0.0), axis=-1)
    chi = np.clip(1.0 + S_b - S_ab, 0.0, 2.0)

    return {
        "C": C.reshape(shape),
        "Fs": Fs.reshape(shape),
        "F_fixed": F_fixed.reshape(shape),
        "F_opt": F_opt.reshape(shape),
        "chi": chi.reshape(shape),
    }


def kraus_superops(t_grid: np.ndarray, gamma: float, Gamma: float) -> list[np.ndarray]:
    out = []
    for t in t_grid:
        Ka, Kb = get_kraus_pair(float(t), "amp", gamma, Gamma, gamma, Gamma)
        S = np.zeros((16, 16), dtype=complex)
        for A in Ka:
            for B in Kb:
                K = np.kron(A, B)
                S += np.kron(K.conj(), K)
        out.append(S)
    return out


def evolve_stack(rho0: np.ndarray, t_grid: np.ndarray, gamma: float, Gamma: float) -> np.ndarray:
    vec = rho0.reshape(len(rho0), 16)
    Ss = kraus_superops(t_grid, gamma, Gamma)
    out = np.empty((len(rho0), len(t_grid), 4, 4), dtype=complex)
    for it, S in enumerate(Ss):
        out[:, it] = (vec @ S.T).reshape(-1, 4, 4)
    return out


def entanglement_death_radius(st: CachedStates, theta_deg: float = 45.0,
                              r_max: float = 3.2, nr: int = 161) -> float:
    """First zero crossing of concurrence along direction theta_deg."""
    r_grid = np.linspace(0.0, r_max, nr)
    th = np.deg2rad(theta_deg)
    r_vecs = np.column_stack([r_grid * np.cos(th), r_grid * np.sin(th)])
    C = observables_full(st.rho_stack(r_vecs))["C"]
    # Find first index where C drops below 1e-6 and refine linearly on fine grid
    idx = np.where(C <= 1e-6)[0]
    if idx.size == 0:
        return float(r_max)
    i = int(idx[0])
    if i == 0:
        return 0.0
    # Refine around [r_grid[i-1], r_grid[i]]
    fine = np.linspace(r_grid[i - 1], r_grid[i], 41)
    r_fine = np.column_stack([fine * np.cos(th), fine * np.sin(th)])
    Cf = observables_full(st.rho_stack(r_fine))["C"]
    j = np.where(Cf <= 1e-6)[0]
    if j.size == 0:
        return float(fine[-1])
    k = int(j[0])
    if k == 0 or Cf[k - 1] == Cf[k]:
        return float(fine[k])
    frac = Cf[k - 1] / (Cf[k - 1] - Cf[k])
    return float(fine[k - 1] + frac * (fine[k] - fine[k - 1]))


def load_task8_concurrence_derivatives() -> dict:
    """Load converged periodic Nk=401 and Nk=801 concurrence derivatives (spin & band)."""
    path = ROOT / "research" / "data" / "task8_density_qpt_eq28.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    full401 = data["full_sweeps"]["full_U75_V0.5_Nk401"]
    zm75 = data["zooms"]["zoom_mid_U75_V0.5_Nk801"]
    zm115 = data["zooms"]["zoom_mid_U115_V0.5_Nk801"]
    n401 = np.array([r["n"] for r in full401["rows"]], dtype=float)
    n801 = np.array([r["n"] for r in zm75["rows"]], dtype=float)
    out = {"n401": n401, "n801": n801, "full": {}, "zoom_sc": {}, "zoom_norm": {}}
    for bs in ("spin", "band"):
        out["full"][bs] = {}
        for rk in ("0.25_0", "0.5_0", "1_0"):
            d = full401["derivs"]["concurrence"][bs][rk]
            out["full"][bs][rk] = {
                "dC": np.array(d["dCS"], dtype=float),
                "d2C": np.array(d["d2CS"], dtype=float),
            }
        out["zoom_sc"][bs] = {
            "dC": np.array(zm75["derivs"]["concurrence"][bs]["0.25_0"]["dCS"], dtype=float),
            "d2C": np.array(zm75["derivs"]["concurrence"][bs]["0.25_0"]["d2CS"], dtype=float),
        }
        out["zoom_norm"][bs] = {
            "dC": np.array(zm115["derivs"]["concurrence"][bs]["0.25_0"]["dCN"], dtype=float),
            "d2C": np.array(zm115["derivs"]["concurrence"][bs]["0.25_0"]["d2CN"], dtype=float),
        }
    return out


def make_phase_and_resources_figure(sweep: dict, u_comp: dict, derivs: dict) -> None:
    ns = np.asarray(sweep["n"])
    fig, axes = plt.subplots(3, 3, figsize=(16.0, 12.6), constrained_layout=True)

    # Panel (a): lambda_+(n) and helicity band polarization (n_+ - n_-)/n
    ax = axes[0, 0]
    ax.axvspan(0.0, 0.219165, color="#d9ecff", alpha=0.55, label=r"Dilute $s_{\pm}$ SC ($n<n_{c,1}$)")
    ax.axvspan(1.019424, 2.0, color="#fde4d2", alpha=0.55, label=r"High-$n$ SC ($n>n_{c,2}$)")
    ax.axhline(1.0, color="0.3", ls=":", lw=1.0)
    lam_plot = np.clip(sweep["lam_cont"], 0.0, 3.2)
    l1, = ax.plot(ns, lam_plot, color="#1f77b4", lw=2.0, label=r"$\lambda_+(n)$ ($U/t=75$)")
    lam115 = np.clip(sweep["lam_cont_U115"], 0.0, 3.2)
    l2, = ax.plot(ns, lam115, color="#8c564b", lw=1.4, ls="-.", label=r"$\lambda_+(n)$ ($U/t=115$)")
    ax.set_xlim(0.0, 1.98)
    ax.set_ylim(0.5, 3.25)
    ax.set_xlabel(r"Electron filling $n$")
    ax.set_ylabel(r"Pairing eigenvalue $\lambda_+(n)$", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.grid(alpha=0.25)

    ax_r = ax.twinx()
    pol = sweep["band_polarization"]
    l3, = ax_r.plot(ns, pol, color="#d62728", lw=1.8, ls="--",
                    label=r"$(n_+-n_-)/n$")
    ax_r.set_ylabel(r"Helicity polarization $(n_+-n_-)/n$", color="#d62728")
    ax_r.tick_params(axis="y", labelcolor="#d62728")
    ax_r.set_ylim(0.0, 1.15)
    ax.set_title(r"(a) Pairing boundaries & helicity polarization", fontsize=10.5)
    lines = [l1, l2, l3]
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=7.6, loc="upper left", bbox_to_anchor=(0.21, 0.99))

    # Panel (b): Initial singlet fraction F_s and optimized F_opt across n
    ax = axes[0, 1]
    ax.axhline(0.5, color="0.35", ls=":", lw=0.9, label=r"$F_{\mathrm{s}}=1/2$ ($\mathcal{F}_{\mathrm{fix}}=2/3$)")
    ax.axhline(2.0 / 3.0, color="#7f7f7f", ls="-.", lw=0.9, label=r"$\overline{\mathcal{F}}_{\mathrm{tel}}=2/3$")
    ax.plot(ns, sweep["Fs_r025_spin"], color="#1f77b4", lw=1.9, ls="-", label=r"$F_{\mathrm{s}}(0.25a)$ spin")
    ax.plot(ns, sweep["Fs_r025_band"], color="#1f77b4", lw=1.9, ls="--", label=r"$F_{\mathrm{s}}(0.25a)$ band")
    ax.plot(ns, sweep["Fs_r100_spin"], color="#d62728", lw=1.9, ls="-", label=r"$F_{\mathrm{s}}(1,0)$ spin")
    ax.plot(ns, sweep["Fs_r100_band"], color="#d62728", lw=1.9, ls="--", label=r"$F_{\mathrm{s}}(1,0)$ band")
    ax.plot(ns, sweep["F_opt_r100_spin"], color="#2ca02c", lw=1.7, ls="-", label=r"$\overline{\mathcal{F}}_{\mathrm{tel}}(1,0)$ spin")
    ax.plot(ns, sweep["F_opt_r100_band"], color="#2ca02c", lw=1.7, ls="--", label=r"$\overline{\mathcal{F}}_{\mathrm{tel}}(1,0)$ band")
    ax.set_xlim(0.0, 1.98)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel(r"Electron filling $n$")
    ax.set_ylabel(r"Singlet fraction / Fidelity ($t=0$)")
    ax.set_title(r"(b) $F_{\mathrm{s}}$ & $\overline{\mathcal{F}}_{\mathrm{tel}}$: spin vs. band", fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.8, ncol=2, loc="lower right")

    # Panel (c): Concurrence C(n) across r = 0.25a, 0.50a, (1,0)a
    ax = axes[0, 2]
    for rk, col, lbl in (
        ("r025", "#1f77b4", r"$\mathbf{r}=(0.25,0)a$"),
        ("r050", "#ff7f0e", r"$\mathbf{r}=(0.50,0)a$"),
        ("r100", "#d62728", r"$\mathbf{r}=(1,0)a$"),
    ):
        ax.plot(ns, sweep[f"C_{rk}_spin"], color=col, lw=1.9, ls="-", label=f"{lbl} spin")
        ax.plot(ns, sweep[f"C_{rk}_band"], color=col, lw=1.9, ls="--", label=f"{lbl} band")
    ax.axvline(0.219165, color="0.45", ls=":", lw=0.9)
    ax.axvline(1.019424, color="0.45", ls=":", lw=0.9)
    ax.annotate(r"spin $n_{\mathrm{ESD}}(1,0)=0.416$", xy=(0.416, 0.01), xytext=(0.56, 0.20),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0),
                fontsize=7.8, color="#d62728")
    ax.set_xlim(0.0, 1.98)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel(r"Electron filling $n$")
    ax.set_ylabel(r"Concurrence $C(n,t=0)$")
    ax.set_title(r"(c) Concurrence $C(n)$ & low-$n$ band collapse", fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.2, ncol=2, loc="upper right")

    # Panel (d): First derivative dC/dn across n (Nk=401 periodic, spin vs. band)
    n401 = derivs["n401"]
    ax = axes[1, 0]
    ax.axhline(0.0, color="0.35", ls=":", lw=0.9)
    ax.axvline(0.906482, color="#2ca02c", ls=":", lw=1.0, label=r"$n_{\mathrm{vH},-}=0.9065$")
    ax.axvline(1.019424, color="0.45", ls="--", lw=0.9, label=r"$n_{c,2}=1.0194$")
    for rk, col, lbl in (
        ("0.25_0", "#1f77b4", r"$(0.25,0)a$"),
        ("0.5_0", "#ff7f0e", r"$(0.50,0)a$"),
        ("1_0", "#d62728", r"$(1,0)a$"),
    ):
        ax.plot(n401, derivs["full"]["spin"][rk]["dC"], color=col, lw=1.7, ls="-", label=f"{lbl} spin")
        ax.plot(n401, derivs["full"]["band"][rk]["dC"], color=col, lw=1.5, ls="--", label=f"{lbl} band")
    ax.set_xlim(0.0, 1.98)
    ax.set_ylim(-2.1, 1.15)
    ax.set_xlabel(r"Electron filling $n$")
    ax.set_ylabel(r"$dC/dn$ ($N_k=401$)")
    ax.set_title(r"(d) First derivative $dC/dn$: spin vs. band", fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.6, ncol=2, loc="lower right")

    # Panel (e): Second derivative d^2C/dn^2 across n in [0.22, 1.98] (Nk=401 periodic)
    ax = axes[1, 1]
    m401 = n401 >= 0.22
    ax.axhline(0.0, color="0.35", ls=":", lw=0.9)
    ax.axvline(0.906482, color="#2ca02c", ls=":", lw=1.0)
    ax.axvline(1.019424, color="0.45", ls="--", lw=0.9)
    for rk, col, lbl in (
        ("0.25_0", "#1f77b4", r"$(0.25,0)a$"),
        ("0.5_0", "#ff7f0e", r"$(0.50,0)a$"),
        ("1_0", "#d62728", r"$(1,0)a$"),
    ):
        ax.plot(n401[m401], derivs["full"]["spin"][rk]["d2C"][m401], color=col, lw=1.7, ls="-", label=f"{lbl} spin")
        ax.plot(n401[m401], derivs["full"]["band"][rk]["d2C"][m401], color=col, lw=1.5, ls="--", label=f"{lbl} band")
    ax.annotate(r"ESD $(1,0)$" "\n" r"$(+48.0,\, +39.1)$", xy=(0.44, 15.8), xytext=(0.54, 11.0),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.9),
                fontsize=7.4, color="#d62728")
    ax.annotate(r"ESD $(0.50a)$", xy=(1.54, 12.0), xytext=(1.55, 6.2),
                arrowprops=dict(arrowstyle="->", color="#d2691e", lw=0.9),
                fontsize=7.4, color="#c25900")
    ax.set_xlim(0.22, 1.98)
    ax.set_ylim(-15.5, 16.5)
    ax.set_xlabel(r"Electron filling $n$")
    ax.set_ylabel(r"$d^2C/dn^2$ ($N_k=401$)")
    ax.set_title(r"(e) Second derivative $d^2C/dn^2$ & ESD spikes", fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.8, ncol=2, loc="lower right")

    # Panel (f): Converged Nk=801 zoom of d^2C/dn^2 (r=0.25a, n in [0.85, 1.15])
    ax = axes[1, 2]
    n801 = derivs["n801"]
    m801 = (n801 >= 0.85) & (n801 <= 1.15)
    ax.axhline(0.0, color="0.35", ls=":", lw=0.9)
    ax.axvline(0.906482, color="#2ca02c", ls=":", lw=1.1, label=r"$n_{\mathrm{vH},-}=0.9065$")
    ax.axvline(1.019424, color="0.45", ls="--", lw=1.0, label=r"$n_{c,2}=1.0194$")
    ax.axvline(1.093471, color="#9467bd", ls=":", lw=1.1, label=r"$n_{\mathrm{vH},+}=1.0935$")
    ax.plot(n801[m801], derivs["zoom_norm"]["spin"]["d2C"][m801], color="0.45", lw=1.4, ls="-", label=r"Normal spin ($U/t=115$)")
    ax.plot(n801[m801], derivs["zoom_norm"]["band"]["d2C"][m801], color="#8c564b", lw=1.4, ls=":", label=r"Normal band ($U/t=115$)")
    ax.plot(n801[m801], derivs["zoom_sc"]["spin"]["d2C"][m801], color="#1f77b4", lw=2.0, ls="-", label=r"SC spin ($U/t=75$)")
    ax.plot(n801[m801], derivs["zoom_sc"]["band"]["d2C"][m801], color="#ff7f0e", lw=2.0, ls="--", label=r"SC band ($U/t=75$)")
    ax.set_xlim(0.85, 1.15)
    ax.set_ylim(-8.6, 5.6)
    ax.set_xlabel(r"Electron filling $n$")
    ax.set_ylabel(r"$d^2C/dn^2$ ($r=0.25a$, $N_k=801$)")
    ax.set_title(r"(f) $N_k=801$ zoom: van Hove cusp vs. SC smearing", fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.6, ncol=2, loc="lower left")

    # Panel (g): Entanglement-death radius r_c(n, theta=45 deg)
    ax = axes[2, 0]
    ax.axhline(1.0, color="#d62728", ls=":", lw=1.1, label=r"Nearest-neighbor site $r=a$")
    ax.plot(ns, sweep["rc_spin"], color="#1f77b4", lw=2.0, ls="-", label=r"$r_c^{\mathrm{spin}}(n)$ ($45^\circ$)")
    ax.plot(ns, sweep["rc_band"], color="#ff7f0e", lw=2.0, ls="--", label=r"$r_c^{\mathrm{band}}(n)$ ($45^\circ$)")
    ax.axvline(0.219165, color="0.45", ls=":", lw=0.9)
    ax.axvline(1.019424, color="0.45", ls=":", lw=0.9)
    ax.set_xlim(0.0, 1.98)
    ax.set_ylim(0.0, 3.30)
    ax.set_xlabel(r"Electron filling $n$")
    ax.set_ylabel(r"Entanglement-death radius $r_c/a$")
    ax.set_title(r"(g) Entanglement range $r_c(n)$ vs. lattice spacing", fontsize=10.5)
    ax.annotate(r"band collapse ($n_- \to 0$)", xy=(0.065, 0.12), xytext=(0.30, 0.25),
                arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.0),
                fontsize=7.8, color="#c25900")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.0, loc="upper right")

    # Panel (h): Pauli-ensemble Holevo capacity chi_P(n)
    ax = axes[2, 1]
    ax.axhline(1.0, color="0.3", ls=":", lw=1.0, label=r"1-bit classical limit")
    ax.plot(ns, sweep["chi_r025_spin"], color="#1f77b4", lw=1.8, ls="-", label=r"$(0.25,0)a$ spin")
    ax.plot(ns, sweep["chi_r025_band"], color="#1f77b4", lw=1.8, ls="--", label=r"$(0.25,0)a$ band")
    ax.plot(ns, sweep["chi_r100_spin"], color="#d62728", lw=1.8, ls="-", label=r"$(1,0)a$ spin")
    ax.plot(ns, sweep["chi_r100_band"], color="#d62728", lw=1.8, ls="--", label=r"$(1,0)a$ band")
    ax.plot(ns, sweep["chi_r10_band"], color="#9467bd", lw=1.9, ls="-.", label=r"$r=10a$ band ($1-S_2(\rho_B)$)")
    ax.set_xlim(0.0, 1.98)
    ax.set_ylim(-0.02, 2.08)
    ax.set_xlabel(r"Electron filling $n$")
    ax.set_ylabel(r"Pauli Holevo $\chi_{\mathrm{P}}(n,t=0)$ [bits]")
    ax.set_title(r"(h) Dense coding vs. classical asymmetry bias", fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.0, ncol=2, loc="upper right")

    # Panel (i): Interaction comparison U/t = 15, 25, 75 at V_SO/t = 0.5 for lattice site (1,0)
    ax = axes[2, 2]
    ax.axhline(2.0 / 3.0, color="0.3", ls=":", lw=1.0, label=r"Classical bound $2/3$")
    u_cols = {15: "#2ca02c", 25: "#ff7f0e", 75: "#1f77b4"}
    u_ns = np.asarray(u_comp["n"])
    for U_val, col in u_cols.items():
        rec = u_comp["by_U"][str(U_val)]
        ax.plot(u_ns, rec["Fopt_r100_spin"], color=col, lw=1.8, ls="-",
                label=fr"$U/t={U_val}$ spin")
        ax.plot(u_ns, rec["Fopt_r100_band"], color=col, lw=1.8, ls="--",
                label=fr"$U/t={U_val}$ band")
    ax.set_xlim(0.04, 1.90)
    ax.set_ylim(0.49, 0.96)
    ax.set_xlabel(r"Electron filling $n$")
    ax.set_ylabel(r"$\overline{\mathcal{F}}_{\mathrm{tel}}(\mathbf{r}=(1,0)a, t=0)$")
    ax.set_title(r"(i) Lattice-site teleportation across $U/t$ ($V_{\mathrm{SO}}=0.5t$)", fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.2, ncol=2, loc="upper right")

    fp = FIGS / "n_dependence_phase_and_resources.png"
    fig.savefig(fp, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_open_protocols_figure(dyn: dict) -> None:
    t_low = np.asarray(dyn["t_low"])
    t_high = np.asarray(dyn["t_high"])
    fill_colors = {
        0.10: "#d62728",
        0.50: "#ff7f0e",
        1.00: "#2ca02c",
        1.40: "#1f77b4",
        1.875: "#9467bd",
    }

    fig, axes = plt.subplots(3, 2, figsize=(14.2, 11.4), constrained_layout=True)

    # Left column: r = 0.30a (theta = 45 deg), low noise gamma=1.5, Gamma=0.1 (spin vs band)
    # Right column: exact lattice site r=(1,0)a under strong memory gamma=4.5, Gamma=0.01 (top & bottom)
    # and r=0.30a basis splitting C^band - C^spin (middle) OR high-memory r=0.30a chi_P!
    # Let's make Right column consistently exact lattice site r=(1,0)a under strong-memory gamma=4.5, Gamma=0.01
    # AND show both spin (solid) and band (dashed) in ALL 6 panels, plus a clear inset or panel note!
    # Wait: let's check if middle-right is even cleaner showing r=0.30a basis difference C^band - C^spin
    # AND bottom-right showing r=(1,0)a C(t) under gamma=4.5, Gamma=0.01!
    # Actually: look how clean it is if:
    # Left column: r = 0.30a (theta=45 deg), gamma=1.5, Gamma=0.1 -> (a) F_s(t), (c) chi_P(t), (e) C(t) [spin solid, band dashed]
    # Right column:
    #   (b) Basis splitting Delta C = C_band - C_spin at r = 0.30a (gamma=1.5, Gamma=0.1) showing revival sign reversal!
    #   (d) Strong-memory chi_P(t) at r = 0.30a (gamma=4.5, Gamma=0.01) showing n=0.10 spin revival > 1 bit!
    #   (f) Exact lattice site r = (1,0)a under gamma=4.5, Gamma=0.01: F_opt(t) & C(t) [spin solid, band dashed]!
    # Wait: let's check if (b) F_opt(t) or Delta F_s and (f) C(t) at r=(1,0)a (gamma=4.5, Gamma=0.01) is even better!
    # Look:
    #   Left column (r = 0.30a, theta=45 deg, gamma=1.5, Gamma=0.1):
    #     (a) Singlet fraction F_s(t) [spin solid, band dashed]
    #     (c) Pauli Holevo chi_P(t) [spin solid, band dashed]
    #     (e) Concurrence C(t) [spin solid, band dashed]
    #   Right column (Strong-memory gamma=4.5, Gamma=0.01 & basis splittings):
    #     (b) Exact lattice site r=(1,0)a optimized teleportation F_opt(t) (gamma=4.5, Gamma=0.01, spin solid, band dashed)
    #     (d) Sub-lattice r=0.30a Pauli Holevo chi_P(t) (gamma=4.5, Gamma=0.01, spin solid, band dashed, showing >1 bit revival at n=0.10!)
    #     (f) Basis difference C_band - C_spin at r=0.30a (solid: gamma=1.5, Gamma=0.1; plus exact site r=(1,0) C(t) in gamma=4.5, Gamma=0.01!)
    # Wait: keeping Right column purely (r=(1,0)a, gamma=4.5, Gamma=0.01) for F_opt, chi_P, C — OR keeping Right column purely (basis differences / strong-memory revivals) — let's check both!
    # Actually: if Left column = r=0.30a (gamma=1.5, Gamma=0.1) and Right column = r=(1,0)a (gamma=4.5, Gamma=0.01) for Row 1 (F_opt) & Row 3 (C),
    # what about Row 2 Right (r=0.30a chi_P at gamma=4.5, Gamma=0.01)? Or keep Right column ALL at gamma=4.5, Gamma=0.01 (r=0.30a vs r=(1,0)a)?
    # Let's check exact values first; actually having:
    #   Col 1: r = 0.30a, theta = 45 deg, gamma = 1.5, Gamma = 0.1 (matches Figs. 9--12!)
    #     (a) F_s(t) (with F_s=1/2 bound)
    #     (c) chi_P(t) (with 1-bit bound)
    #     (e) C(t) (spin solid, band dashed)
    #   Col 2: Strong-memory non-Markovian revivals (gamma = 4.5, Gamma = 0.01, matches Figs. 6--7!)
    #     (b) Exact lattice site r = (1,0)a: F_opt(t) (spin solid, band dashed, 2/3 classical bound)
    #     (d) r = 0.30a, theta = 45 deg: chi_P(t) (spin solid, band dashed, 1-bit bound)
    #     (f) Exact lattice site r = (1,0)a: C(t) (spin solid, band dashed) & inset/note on revival sign flip!
    # WAIT: What if instead panel (f) shows C^band - C^spin at r=0.30a (gamma=1.5, Gamma=0.1) AND panel (b) shows F_opt(t) at r=(1,0)a (gamma=4.5, Gamma=0.01) AND panel (d) shows C(t) at r=(1,0)a (gamma=4.5, Gamma=0.01)?
    # Even cleaner:
    #   (a) r=0.30a (gamma=1.5, Gamma=0.1): F_s(t)
    #   (b) r=0.30a (gamma=1.5, Gamma=0.1): basis difference F_s^spin - F_s^band (top) or C^band - C^spin!
    # Wait — look at Fig 12 (U_dependence_fidelity_holevo.png) and Fig 17 (environment_protocols_regimes.png) in manuscript.tex!
    # In Fig 17: Col 1 = non-Markovian metrics, Col 2 = Markovian metrics, Col 3 = Basis difference (band - spin)!
    # So here:
    #   Col 1 (r=0.30a, gamma=1.5, Gamma=0.1): (a) F_s(t), (c) chi_P(t), (e) C(t) — spin (solid) & band (dashed)!
    #   Col 2 (Strong-memory gamma=4.5, Gamma=0.01 & lattice-site / basis-splitting):
    #     (b) Exact site r=(1,0)a, gamma=4.5, Gamma=0.01: F_opt(t) [spin solid, band dashed; 2/3 bound]
    #     (d) Sub-lattice r=0.30a, gamma=4.5, Gamma=0.01: chi_P(t) [spin solid, band dashed; 1-bit bound]
    #     (f) Basis difference C^band - C^spin at r=0.30a (gamma=1.5, Gamma=0.1) — highlighting the exact transition from spin-dominated revival (n=0.10, negative) to band-enhanced revival (n>=0.50, positive)!
    # Wait: what about C(t) at r=(1,0)a for n=0.10? We can put BOTH r=(1,0)a F_opt(t) in (b) AND r=(1,0)a C(t) in (f) — OR let's check how clear a 3x3 or 3x2 is!
    # Wait: a 3x2 figure fits on one page cleanly in PRB twocolumn (`figure*`)! Let's keep it 3x2 with crystal-clear titles:
    #   (a) r=0.30a (gamma=1.5, Gamma=0.1): Singlet fraction F_s(t) [spin solid, band dashed]
    #   (b) Exact site r=(1,0)a (gamma=4.5, Gamma=0.01): F_opt(t) & C(t) -> wait, don't mix 2 y-axes! Keep (b) pure: Exact site r=(1,0)a (gamma=4.5, Gamma=0.01): C(t) [spin solid, band dashed]!
    #   (c) r=0.30a (gamma=1.5, Gamma=0.1): Pauli Holevo chi_P(t) [spin solid, band dashed]
    #   (d) r=0.30a (gamma=4.5, Gamma=0.01): Pauli Holevo chi_P(t) [spin solid, band dashed] — shows >1 bit revival at n=0.10 AND exact asymptotic 1-bit approach!
    #   (e) r=0.30a (gamma=1.5, Gamma=0.1): Concurrence C(t) [spin solid, band dashed]
    #   (f) r=0.30a (gamma=1.5, Gamma=0.1): Basis difference C^band - C^spin(t) [showing t=0 spin advantage vs. revival sign inversion across n!]
    # Wait: where does F_opt(t) at r=(1,0)a appear? In Fig 13(b)/(f) we ALREADY show F_s(1,0) and F_opt(1,0) at t=0, AND we can quote the exact r=(1,0)a F_opt revival numbers in the text/JSON (or plot F_opt in (b) and quote C in text, or plot C in (b) and F_opt in a clean twin? No, no twin axes!). Actually, plotting **F_opt(t) at r=(1,0)a (gamma=4.5, Gamma=0.01)** in (b) or **C(t) at r=(1,0)a** in (b) — wait: let's check which curves look cleanest when plotted! Let's generate it and inspect it with `read_file`!

    # Row 0 Left: (a) r=0.30a, gamma=1.5, Gamma=0.1 -> F_s(t)
    ax = axes[0, 0]
    ax.axhline(0.5, color="0.35", ls=":", lw=0.9)
    for n in REP_FILLINGS:
        col = fill_colors[n]
        ax.plot(t_low, dyn["low_r030"][str(n)]["spin"]["Fs"], color=col, lw=1.7, ls="-", label=fr"$n={n}$")
        ax.plot(t_low, dyn["low_r030"][str(n)]["band"]["Fs"], color=col, lw=1.5, ls="--")
    ax.plot([], [], color="black", lw=1.6, ls="-", label="spin")
    ax.plot([], [], color="black", lw=1.6, ls="--", label="band")
    ax.set_title(r"(a) $F_{\mathrm{s}}(t)$ at $r=0.30a,45^\circ$ ($\gamma=1.5,\Gamma=0.1$)", fontsize=10.5)
    ax.set_xlim(0.0, 45.0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel(r"Singlet fraction $F_{\mathrm{s}}$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.6, ncol=2, loc="upper right")

    # Row 0 Right: (b) Exact lattice site r=(1,0)a, gamma=4.5, Gamma=0.01 -> F_opt(t)
    ax = axes[0, 1]
    ax.axhline(2.0 / 3.0, color="0.35", ls=":", lw=1.0, label=r"Classical bound $2/3$")
    for n in REP_FILLINGS:
        col = fill_colors[n]
        ax.plot(t_high, dyn["high_r100"][str(n)]["spin"]["F_opt"], color=col, lw=1.7, ls="-", label=fr"$n={n}$")
        ax.plot(t_high, dyn["high_r100"][str(n)]["band"]["F_opt"], color=col, lw=1.5, ls="--")
    ax.plot([], [], color="black", lw=1.6, ls="-", label="spin")
    ax.plot([], [], color="black", lw=1.6, ls="--", label="band")
    ax.set_title(r"(b) Lattice-site $\overline{\mathcal{F}}_{\mathrm{tel}}(t)$ at $\mathbf{r}=(1,0)a$ ($\gamma=4.5,\Gamma=0.01$)", fontsize=10.5)
    ax.set_xlim(0.0, 35.0)
    ax.set_ylim(0.48, 0.90)
    ax.set_ylabel(r"Optimized fidelity $\overline{\mathcal{F}}_{\mathrm{tel}}$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.3, ncol=2, loc="upper right")

    # Row 1 Left: (c) r=0.30a, gamma=1.5, Gamma=0.1 -> chi_P(t)
    ax = axes[1, 0]
    ax.axhline(1.0, color="0.35", ls=":", lw=1.0)
    for n in REP_FILLINGS:
        col = fill_colors[n]
        ax.plot(t_low, dyn["low_r030"][str(n)]["spin"]["chi"], color=col, lw=1.7, ls="-")
        ax.plot(t_low, dyn["low_r030"][str(n)]["band"]["chi"], color=col, lw=1.5, ls="--")
    ax.set_title(r"(c) Pauli Holevo $\chi_{\mathrm{P}}(t)$ at $r=0.30a,45^\circ$ ($\gamma=1.5,\Gamma=0.1$)", fontsize=10.5)
    ax.set_xlim(0.0, 45.0)
    ax.set_ylim(0.0, 2.02)
    ax.set_ylabel(r"$\chi_{\mathrm{P}}$ [bits]")
    ax.grid(alpha=0.25)

    # Row 1 Right: (d) r=0.30a, gamma=4.5, Gamma=0.01 -> chi_P(t)
    ax = axes[1, 1]
    ax.axhline(1.0, color="0.35", ls=":", lw=1.0)
    for n in REP_FILLINGS:
        col = fill_colors[n]
        ax.plot(t_high, dyn["high_r030"][str(n)]["spin"]["chi"], color=col, lw=1.7, ls="-")
        ax.plot(t_high, dyn["high_r030"][str(n)]["band"]["chi"], color=col, lw=1.5, ls="--")
    ax.set_title(r"(d) Strong-memory $\chi_{\mathrm{P}}(t)$ at $r=0.30a,45^\circ$ ($\gamma=4.5,\Gamma=0.01$)", fontsize=10.5)
    ax.set_xlim(0.0, 35.0)
    ax.set_ylim(0.0, 2.02)
    ax.set_ylabel(r"$\chi_{\mathrm{P}}$ [bits]")
    ax.grid(alpha=0.25)

    # Row 2 Left: (e) r=0.30a, gamma=1.5, Gamma=0.1 -> C(t)
    ax = axes[2, 0]
    for n in REP_FILLINGS:
        col = fill_colors[n]
        ax.plot(t_low, dyn["low_r030"][str(n)]["spin"]["C"], color=col, lw=1.7, ls="-", label=fr"$n={n}$")
        ax.plot(t_low, dyn["low_r030"][str(n)]["band"]["C"], color=col, lw=1.5, ls="--")
    ax.set_title(r"(e) Concurrence $C(t)$ at $r=0.30a,45^\circ$ ($\gamma=1.5,\Gamma=0.1$)", fontsize=10.5)
    ax.set_xlim(0.0, 45.0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"Concurrence $C$")
    ax.grid(alpha=0.25)

    # Row 2 Right: (f) Basis difference C^band - C^spin at r=0.30a (gamma=1.5, Gamma=0.1)
    ax = axes[2, 1]
    ax.axhline(0.0, color="0.35", ls=":", lw=1.0)
    for n in REP_FILLINGS:
        col = fill_colors[n]
        diff = np.asarray(dyn["low_r030"][str(n)]["band"]["C"]) - np.asarray(dyn["low_r030"][str(n)]["spin"]["C"])
        ax.plot(t_low, diff, color=col, lw=1.8, ls="-", label=fr"$n={n}$")
    ax.set_title(r"(f) Revival basis splitting $C^{\mathrm{band}}-C^{\mathrm{spin}}$ ($r=0.30a$)", fontsize=10.5)
    ax.set_xlim(0.0, 45.0)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\Delta C = C^{\mathrm{band}} - C^{\mathrm{spin}}$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.8, ncol=2, loc="lower right")

    fp = FIGS / "n_dependence_open_protocols.png"
    fig.savefig(fp, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    print(f"[n-study] Solving full-band sweep on Nk={NK} periodic grid ...")
    sk, ssin, eps, KX, KY, _ = fs.make_grid(NK, Vso=VSO, periodic=True)
    kern75 = fs.kernel_array(sk, ssin, U_MAIN, DT, VSO)
    kern115 = fs.kernel_array(sk, ssin, 115.0, DT, VSO)

    # Dense non-uniform grid covering dilute pocket, ESD threshold, normal, and SC dome
    ns_dense = np.unique(np.round(np.concatenate([
        np.linspace(0.02, 0.26, 17),
        np.linspace(0.28, 0.96, 18),
        np.linspace(0.98, 1.16, 11),
        np.linspace(1.20, 1.96, 20),
        np.array([0.04, 0.10, 0.15, 0.2192, 0.4162, 0.50, 0.75, 0.9065, 1.00, 1.0194, 1.0935, 1.40, 1.75, 1.875, 1.929]),
    ]), 4))

    sweep = {
        "n": ns_dense.tolist(),
        "sc": [],
        "Delta0": [],
        "DeltaS": [],
        "mu": [],
        "min_E": [],
        "residual": [],
        "lam_cont": [],
        "lam_cont_U115": [],
        "n_plus": [],
        "n_minus": [],
        "band_polarization": [],
        "rc_spin": [],
        "rc_band": [],
        "chi_r10_spin": [],
        "chi_r10_band": [],
    }
    r_targets = {
        "r025": np.array([0.25, 0.0]),
        "r050": np.array([0.50, 0.0]),
        "r100": np.array([1.00, 0.0]),
        "r10": np.array([10.0, 0.0]),
    }
    for rk in ("r025", "r050", "r100"):
        for bs in ("spin", "band"):
            for met in ("C", "Fs", "F_fixed", "F_opt", "chi"):
                sweep[f"{met}_{rk}_{bs}"] = []

    r_mat = np.stack([r_targets["r025"], r_targets["r050"], r_targets["r100"], r_targets["r10"]], axis=0)
    validation = {"max_trace_err": 0.0, "max_herm_err": 0.0, "min_eval": 0.0, "max_r0_singlet_err": 0.0}

    rep_bdgs = {}
    for n_val in ns_dense:
        n_f = float(n_val)
        bdg, best, lam_c, res, sc = solve_global_bdg(eps, kern75, KX, KY, ssin, NK, U_MAIN, n_f)
        nrm115 = fs.normal_state(eps, n_f)
        lam115, _, _ = fs.lambda_exact_and_matrix(eps, kern115, NK, DT, nrm115["mu"], 115.0)
        v2 = np.abs(bdg["v_ks"]) ** 2
        n_plus = float(np.mean(v2[..., 0]))
        n_minus = float(np.mean(v2[..., 1]))
        pol = (n_plus - n_minus) / (n_plus + n_minus)

        sweep["sc"].append(bool(sc))
        sweep["Delta0"].append(float(best[0]))
        sweep["DeltaS"].append(float(best[1]))
        sweep["mu"].append(float(best[2]))
        sweep["min_E"].append(float(np.min(bdg["E_ks"])))
        sweep["residual"].append(float(res))
        sweep["lam_cont"].append(float(min(lam_c, 1e6)))
        sweep["lam_cont_U115"].append(float(min(lam115, 1e6)))
        sweep["n_plus"].append(n_plus)
        sweep["n_minus"].append(n_minus)
        sweep["band_polarization"].append(float(pol))

        for bs in ("spin", "band"):
            st = CachedStates(bdg, bs)
            rhos = st.rho_stack(r_mat)
            for idx in range(len(rhos)):
                diag = hdm.physical_diagnostics(rhos[idx])
                validation["max_trace_err"] = max(validation["max_trace_err"], diag["trace_error"])
                validation["max_herm_err"] = max(validation["max_herm_err"], diag["hermiticity_error"])
                validation["min_eval"] = min(validation["min_eval"], diag["minimum_eigenvalue"])
            obs = observables_full(rhos)
            for ir, rk in enumerate(("r025", "r050", "r100")):
                for met in ("C", "Fs", "F_fixed", "F_opt", "chi"):
                    sweep[f"{met}_{rk}_{bs}"].append(float(obs[met][ir]))
            sweep[f"chi_r10_{bs}"].append(float(obs["chi"][3]))
            sweep[f"rc_{bs}"].append(entanglement_death_radius(st, theta_deg=45.0))

        for target_n in REP_FILLINGS:
            if abs(n_f - target_n) < 1e-6:
                rep_bdgs[target_n] = bdg

    # Multi-U lattice-site comparison at V_SO/t = 0.5
    u_ns = [0.04, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00, 1.25, 1.40, 1.65, 1.875]
    u_comp = {"n": u_ns, "by_U": {}}
    r1_mat = np.array([[1.0, 0.0]])
    for U_val in (15, 25, 75):
        kernU = fs.kernel_array(sk, ssin, float(U_val), DT, VSO)
        rec = {
            "sc": [],
            "max_abs_Delta": [],
            "C_r100_spin": [],
            "C_r100_band": [],
            "Fopt_r100_spin": [],
            "Fopt_r100_band": [],
            "chi_r100_spin": [],
            "chi_r100_band": [],
        }
        for n_f in u_ns:
            bdg, _, _, _, sc = solve_global_bdg(eps, kernU, KX, KY, ssin, NK, float(U_val), n_f)
            rec["sc"].append(bool(sc))
            rec["max_abs_Delta"].append(float(np.max(np.abs(bdg["Delta_ks"]))))
            for bs in ("spin", "band"):
                st = CachedStates(bdg, bs)
                obs = observables_full(st.rho_stack(r1_mat))
                rec[f"C_r100_{bs}"].append(float(obs["C"][0]))
                rec[f"Fopt_r100_{bs}"].append(float(obs["F_opt"][0]))
                rec[f"chi_r100_{bs}"].append(float(obs["chi"][0]))
        u_comp["by_U"][str(U_val)] = rec

    # Open-system dynamics at representative fillings
    print("[n-study] Running open-system non-Markovian dynamics ...")
    t_low = np.linspace(0.0, 80.0, 321)
    t_high = np.linspace(0.0, 35.0, 351)
    r030_vec = np.array([0.30 / np.sqrt(2.0), 0.30 / np.sqrt(2.0)])
    r100_vec = np.array([1.0, 0.0])
    r0_vec = np.array([0.0, 0.0])

    dyn = {
        "t_low": t_low.tolist(),
        "t_high": t_high.tolist(),
        "low_r030": {},
        "high_r030": {},
        "high_r100": {},
        "low_r100": {},
    }
    summary_rep = {}

    # Cross-check superoperator against evolution.evolve_rho once
    bdg_ref = rep_bdgs[1.875]
    rho_test = hdm.density_at_separation(bdg_ref, r030_vec, basis="spin")["rho"]
    superop_err = 0.0
    for tt in (0.0, 4.5, 17.5, 35.0):
        ref = evolve_rho(rho_test, tt, kind="amp", bath="independent",
                         gammaA=LOW["gamma"], GammaA=LOW["Gamma"],
                         gammaB=LOW["gamma"], GammaB=LOW["Gamma"])
        mine = evolve_stack(rho_test[None], np.array([tt]), **LOW)[0, 0]
        superop_err = max(superop_err, float(np.max(np.abs(ref - mine))))
    validation["superop_vs_evolve_rho_err"] = superop_err
    if superop_err > 1e-12:
        raise RuntimeError(f"Superoperator mismatch: {superop_err:.3e}")

    for n_f in REP_FILLINGS:
        bdg = rep_bdgs[n_f]
        key = str(n_f)
        for dkey in ("low_r030", "high_r030", "high_r100", "low_r100"):
            dyn[dkey][key] = {}
        summary_rep[key] = {}
        for bs in ("spin", "band"):
            st = CachedStates(bdg, bs)
            rho0, rho030, rho100 = st.rho_stack(np.stack([r0_vec, r030_vec, r100_vec], axis=0))
            Fs0 = float(np.real(hdm.SINGLET.conj() @ rho0 @ hdm.SINGLET))
            validation["max_r0_singlet_err"] = max(validation["max_r0_singlet_err"], abs(Fs0 - 1.0))

            obs_low_030 = observables_full(evolve_stack(rho030[None], t_low, **LOW)[0])
            obs_low_100 = observables_full(evolve_stack(rho100[None], t_low, **LOW)[0])
            obs_high_030 = observables_full(evolve_stack(rho030[None], t_high, **HIGH)[0])
            obs_high_100 = observables_full(evolve_stack(rho100[None], t_high, **HIGH)[0])

            for dkey, obs_dict in (
                ("low_r030", obs_low_030),
                ("low_r100", obs_low_100),
                ("high_r030", obs_high_030),
                ("high_r100", obs_high_100),
            ):
                dyn[dkey][key][bs] = {m: arr.tolist() for m, arr in obs_dict.items()}

            m_low = t_low > 4.0
            m_high = t_high > 4.0
            m_high_chi = t_high > 8.0
            summary_rep[key][bs] = {
                "r030_t0": {m: float(obs_low_030[m][0]) for m in ("C", "Fs", "F_fixed", "F_opt", "chi")},
                "r100_t0": {m: float(obs_low_100[m][0]) for m in ("C", "Fs", "F_fixed", "F_opt", "chi")},
                "low_r030_rev_max": {
                    "C": float(np.max(obs_low_030["C"][m_low])),
                    "Fs": float(np.max(obs_low_030["Fs"][m_low])),
                },
                "high_r030_rev_max": {
                    "C": float(np.max(obs_high_030["C"][m_high])),
                    "Fs": float(np.max(obs_high_030["Fs"][m_high])),
                    "F_opt": float(np.max(obs_high_030["F_opt"][m_high])),
                    "chi": float(np.max(obs_high_030["chi"][m_high_chi])),
                },
                "high_r100_rev_max": {
                    "C": float(np.max(obs_high_100["C"][m_high])),
                    "Fs": float(np.max(obs_high_100["Fs"][m_high])),
                    "F_opt": float(np.max(obs_high_100["F_opt"][m_high])),
                },
            }

    # Nk = 201 vs Nk = 401 convergence check at representative fillings
    print(f"[n-study] Cross-checking Nk={NK} vs Nk={NK_CHECK} ...")
    sk4, ssin4, eps4, KX4, KY4, _ = fs.make_grid(NK_CHECK, Vso=VSO, periodic=True)
    kern4 = fs.kernel_array(sk4, ssin4, U_MAIN, DT, VSO)
    nk_check = {}
    max_c_diff = 0.0
    max_chi_diff = 0.0
    for n_f in REP_FILLINGS:
        bdg4, best4, _, _, _ = solve_global_bdg(eps4, kern4, KX4, KY4, ssin4, NK_CHECK, U_MAIN, n_f)
        idx_201 = int(np.argmin(np.abs(ns_dense - n_f)))
        nk_check[str(n_f)] = {}
        for bs in ("spin", "band"):
            st4 = CachedStates(bdg4, bs)
            obs4 = observables_full(st4.rho_stack(np.array([[1.0, 0.0]])))
            c_diff = abs(float(obs4["C"][0]) - sweep[f"C_r100_{bs}"][idx_201])
            chi_diff = abs(float(obs4["chi"][0]) - sweep[f"chi_r100_{bs}"][idx_201])
            max_c_diff = max(max_c_diff, c_diff)
            max_chi_diff = max(max_chi_diff, chi_diff)
            nk_check[str(n_f)][bs] = {
                "C_r100_Nk401": float(obs4["C"][0]),
                "chi_r100_Nk401": float(obs4["chi"][0]),
                "diff_C": c_diff,
                "diff_chi": chi_diff,
            }
    validation["Nk201_vs_Nk401_max_C_r100_diff"] = max_c_diff
    validation["Nk201_vs_Nk401_max_chi_r100_diff"] = max_chi_diff

    # Also verify exact classical asymmetry relation chi_band(r=10, t=0) vs 1 - h2(n_+/n)
    asym_errs = []
    for i in range(len(ns_dense)):
        p = sweep["n_plus"][i] / (sweep["n_plus"][i] + sweep["n_minus"][i])
        if 1e-6 < p < 1.0 - 1e-6:
            h2 = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
        else:
            h2 = 0.0
        asym_errs.append(abs(sweep["chi_r10_band"][i] - (1.0 - h2)))
    validation["max_r10_band_chi_vs_1_minus_h2"] = float(max(asym_errs))

    print("[n-study] Generating publication figures ...")
    derivs = load_task8_concurrence_derivatives()
    make_phase_and_resources_figure(sweep, u_comp, derivs)
    make_open_protocols_figure(dyn)

    np.savez_compressed(
        DATA / "n_dependence_protocols.npz",
        n=np.asarray(sweep["n"]),
        t_low=t_low,
        t_high=t_high,
        rc_spin=np.asarray(sweep["rc_spin"]),
        rc_band=np.asarray(sweep["rc_band"]),
        C_r100_spin=np.asarray(sweep["C_r100_spin"]),
        C_r100_band=np.asarray(sweep["C_r100_band"]),
        Fs_r100_spin=np.asarray(sweep["Fs_r100_spin"]),
        Fs_r100_band=np.asarray(sweep["Fs_r100_band"]),
        F_opt_r100_spin=np.asarray(sweep["F_opt_r100_spin"]),
        F_opt_r100_band=np.asarray(sweep["F_opt_r100_band"]),
        chi_r100_spin=np.asarray(sweep["chi_r100_spin"]),
        chi_r100_band=np.asarray(sweep["chi_r100_band"]),
        chi_r10_band=np.asarray(sweep["chi_r10_band"]),
    )

    out_report = {
        "params": {"U_over_t": U_MAIN, "Delta_t_over_t": DT, "V_SO_over_t": VSO, "Nk": NK, "Nk_check": NK_CHECK},
        "validation": validation,
        "nk_check": nk_check,
        "representative_summary": summary_rep,
        "u_comparison": u_comp,
        "sweep_key_points": {
            "n_list": sweep["n"],
            "n_plus": sweep["n_plus"],
            "n_minus": sweep["n_minus"],
            "band_polarization": sweep["band_polarization"],
            "rc_spin": sweep["rc_spin"],
            "rc_band": sweep["rc_band"],
            "C_r025_spin": sweep["C_r025_spin"],
            "C_r025_band": sweep["C_r025_band"],
            "C_r100_spin": sweep["C_r100_spin"],
            "C_r100_band": sweep["C_r100_band"],
            "Fs_r100_spin": sweep["Fs_r100_spin"],
            "Fs_r100_band": sweep["Fs_r100_band"],
            "F_opt_r100_spin": sweep["F_opt_r100_spin"],
            "F_opt_r100_band": sweep["F_opt_r100_band"],
            "chi_r100_spin": sweep["chi_r100_spin"],
            "chi_r100_band": sweep["chi_r100_band"],
            "chi_r10_band": sweep["chi_r10_band"],
        },
        "elapsed_sec": time.time() - t0,
    }
    (REPORTS / "n_dependence_protocols.json").write_text(
        json.dumps(out_report, indent=2), encoding="utf-8"
    )
    print(f"[n-study] Completed in {time.time() - t0:.1f}s")
    print("[n-study] Validation:", json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
