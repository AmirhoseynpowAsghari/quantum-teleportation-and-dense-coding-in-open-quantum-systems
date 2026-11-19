#!/usr/bin/env python3
"""Regenerate every manuscript figure that was produced at V_SO = 0,
now at V_SO/t = 0.5, using the helicity conditional-Wick density matrix
transformed to the original-spin basis.

Figures produced (paths identical to the archived V_SO=0 files so the
manuscript references stay valid):

  1  Delta_ks_heatmaps.png                       (gap structure, V_SO!=0)
  2  figs/greens_angular.png                     (spin-basis |G|,|F| vs r)
  3  figs/concurrence_U_theta.png                (C(r,theta,0), 4 U values)
  7  figures/<HIGH>_C_heatmap.png                (C(r,theta=0,t), high noise)
  8  figures/<HIGH>_F_heatmap_singlet.png        (F_s(r,theta=0,t), high noise)
  9  figures/<LOW>_F_vs_t_singlet.png            (F_s vs t, theta=45deg)
 10  figures/<LOW>_F_polar_singlet.png           (F_s(r,theta,0) polar)
 11  figures/<LOW>_holevo_vs_r.png               (chi(r,45deg,t) slices)
 12  figures/<LOW>_holevo_theta_t_heatmap.png    (chi(theta,r=0,t))
 13  figs/U_dependence_fidelity_holevo.png       (U sweep at V_SO!=0)

All two-qubit states come from helicity_density_matrix (spin basis).
Evolution uses the repository's exact independent-bath amplitude-damping
Kraus map; observables are the general Wootters concurrence, the
singlet-referenced fidelity F_s = <Psi-|rho|Psi->, and the equal-Pauli
Holevo quantity (identical to 1+S(rho_B)-S(rho_AB)).

Run from the repository root:
    .venv/bin/python analysis/regenerate_vso_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
import kspace_setup
import self_consistency
from helicity_density_matrix import (
    SINGLET,
    conditional_two_qubit_density,
    momentum_correlators,
)
from noise_channels import get_kraus_pair
from self_consistency import run_selfconsistency
from fidelity import eta_amp_lorentz
from evolution import evolve_rho

import importlib.util
spec = importlib.util.spec_from_file_location(
    "reproduce_and_validate", ROOT / "analysis" / "reproduce_and_validate.py")
rv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv)

FIGURES = ROOT / "figures"
FIGS = ROOT / "figs"
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
for d in (FIGURES, FIGS, DATA, REPORTS):
    d.mkdir(exist_ok=True)

VSO = 0.5
NK = 100
R_GRID = np.linspace(config.R_MIN, config.R_MAX, config.N_R)   # 200 on [0,10]
THETA_DEG = list(config.THETA_DEG)                              # 7 angles
T_FULL = np.linspace(config.T_MIN, config.T_MAX, config.N_T)    # 261 on [0,80]
T_HEAT = np.linspace(0.0, 80.0, 161)                            # heatmaps
LOW = dict(gamma=1.5, Gamma=0.1)
HIGH = dict(gamma=4.5, Gamma=0.01)
LOW_PREFIX = "hubbard_amp_independent_gA1.50_GA0.100_gB1.50_GB0.100"
HIGH_PREFIX = "hubbard_amp_independent_gA4.50_GA0.010_gB4.50_GB0.010"
R_VS_T = [0.0, 1.50754, 3.3166, 6.6332, 10.0]                   # fig 9
R_POLAR = [0.0, 3.3166, 6.6332, 10.0]                           # fig 10
U_SWEEP = [40, 75, 115]                                         # figs 1, 13
U_THETA = [20, 44, 79, 115]                                     # fig 3

report: dict = {"V_SO_over_t": VSO, "Nk": NK, "figures": {},
                "validation": {}, "quoted_numbers": {}}


def solve(vso: float, U: float, nk: int = NK) -> dict:
    for mod in (config, kspace_setup, self_consistency):
        mod.V_SO = float(vso)
        mod.Nk = int(nk)
    bdg = run_selfconsistency(seed=42, U_override=float(U),
                              kBT_override=0.0, verbose=False)
    res = float(np.max(np.abs(bdg["residuals"])))
    if not bdg["converged"] or res > 1e-8:
        raise RuntimeError(f"U={U}, V_SO={vso}: residual={res:.3e}")
    return bdg


class States:
    """Spin-basis conditional states for one BdG solution, N_k/F_k cached."""

    def __init__(self, bdg: dict):
        self.bdg = bdg
        self.N_k, self.F_k = momentum_correlators(bdg, basis="spin")
        self.weight = float(self.N_k[..., 0, 0].real.mean())  # unused, keep

    def correlators(self, r_vec) -> dict:
        r_vec = np.asarray(r_vec, dtype=float).reshape(2)
        kx = self.bdg["KX"]
        ky = self.bdg["KY"]
        phase = np.exp(1j * (kx * r_vec[0] + ky * r_vec[1]))
        w = 1.0 / kx.size
        N_AB = w * np.einsum("ij,ijab->ab", phase, self.N_k)
        N_A = w * np.sum(self.N_k, axis=(0, 1))
        F_AB = w * np.einsum("ij,ijab->ab", phase.conj(), self.F_k)
        if not np.any(r_vec):
            F_AB = 0.5 * (F_AB - F_AB.T)
        return {"N_A": N_A, "N_B": N_A.copy(), "N_AB": N_AB,
                "N_BA": N_AB.conj().T, "F_AB": F_AB, "F_BA": -F_AB.T}

    def rho(self, r_vec) -> np.ndarray:
        rho, _ = conditional_two_qubit_density(self.correlators(r_vec))
        return rho

    def ohkim_x(self, r_vec) -> np.ndarray:
        """Scalar Oh-Kim X state built from the same correlators (V_SO=0 limit)."""
        corr = self.correlators(r_vec)
        n0 = float(corr["N_A"][0, 0].real)
        g2 = float(abs(corr["N_AB"][0, 0] / n0) ** 2)
        f2 = float(abs(corr["F_AB"][0, 1] / n0) ** 2)
        norm = 4.0 - 2.0 * g2 + 2.0 * f2
        return np.array([
            [1.0 - g2, 0.0, 0.0, 0.0],
            [0.0, 1.0 + f2, -(g2 + f2), 0.0],
            [0.0, -(g2 + f2), 1.0 + f2, 0.0],
            [0.0, 0.0, 0.0, 1.0 - g2],
        ], dtype=complex) / norm


def r_vec(r: float, theta_deg: float) -> np.ndarray:
    th = np.deg2rad(theta_deg)
    return np.array([r * np.cos(th), r * np.sin(th)])


def kraus_superops(t_grid, gamma, Gamma) -> list[np.ndarray]:
    """Row-major-vec superoperators S(t) with vec(K rho K^dagger)=S vec(rho)."""
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


def evolve_stack(rho0: np.ndarray, t_grid, gamma, Gamma) -> np.ndarray:
    """rho0 (M,4,4) -> (M,T,4,4) under independent amplitude damping."""
    vec = rho0.reshape(len(rho0), 16)
    Ss = kraus_superops(t_grid, gamma, Gamma)
    out = np.empty((len(rho0), len(t_grid), 4, 4), dtype=complex)
    for it, S in enumerate(Ss):
        out[:, it] = (vec @ S.T).reshape(-1, 4, 4)
    return out


def observables(rho_t: np.ndarray) -> dict:
    """General metrics for any stack of two-qubit states (last axes 4,4)."""
    from helicity_density_matrix import concurrence as wootters
    shape = rho_t.shape[:-2]
    flat = rho_t.reshape(-1, 4, 4)
    C = np.array([wootters(rho) for rho in flat]).reshape(shape)
    Fs = np.array([float(np.real(SINGLET.conj() @ rho @ SINGLET))
                   for rho in flat]).reshape(shape)
    from helicity_density_matrix import dense_coding_metrics
    chi = np.array([dense_coding_metrics(rho)["holevo_pauli_bits"]
                    for rho in flat]).reshape(shape)
    return dict(C=C, Fs=Fs, chi=chi)


def rho_grid(states: States, r_values, thetas_deg) -> tuple[np.ndarray, np.ndarray]:
    """(n_theta, n_r, 4,4) initial states."""
    rhos = np.empty((len(thetas_deg), len(r_values), 4, 4), dtype=complex)
    for i, th in enumerate(thetas_deg):
        for j, r in enumerate(r_values):
            rhos[i, j] = states.rho(r_vec(r, th))
    return rhos, np.asarray(r_values, float)


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def check_rho(rho: np.ndarray, tag: str, errors: dict) -> None:
    errors["trace"] = max(errors.get("trace", 0.0),
                          abs(float(np.trace(rho).real) - 1.0))
    errors["hermiticity"] = max(errors.get("hermiticity", 0.0),
                                float(np.max(np.abs(rho - rho.conj().T))))
    errors["min_eigenvalue"] = min(errors.get("min_eigenvalue", 0.0),
                                   float(np.linalg.eigvalsh(rho).min()))
    if errors["trace"] > 1e-10 or errors["hermiticity"] > 1e-10:
        raise RuntimeError(f"{tag}: bad density matrix {errors}")


def validate_superoperator(rho0, gamma, Gamma) -> float:
    worst = 0.0
    for t in (0.0, 3.7, 25.0, 80.0):
        ref = evolve_rho(rho0, t, kind="amp", bath="independent",
                         gammaA=gamma, GammaA=Gamma,
                         gammaB=gamma, GammaB=Gamma)
        mine = evolve_stack(rho0[None], np.array([t]), gamma, Gamma)[0, 0]
        worst = max(worst, float(np.max(np.abs(ref - mine))))
    return worst


# ---------------------------------------------------------------------------
# figure 1: Delta_ks heatmaps at V_SO != 0
# ---------------------------------------------------------------------------
def fig1_delta_heatmaps() -> dict:
    solutions = {}
    quoted = {}
    for U in U_SWEEP:
        bdg = solve(VSO, U)
        solutions[U] = {"bdg": bdg}
        quoted[str(U)] = {
            "Delta0": float(bdg["Delta0"]), "DeltaS": float(bdg["DeltaS"]),
            "mu": float(bdg.get("mu", np.nan)),
            "residual": float(np.max(np.abs(bdg["residuals"])))}
    rv.plot_delta_heatmaps(solutions)
    report["quoted_numbers"]["delta_ks_U"] = quoted
    report["figures"]["Delta_ks_heatmaps.png"] = "ok"
    return solutions


# ---------------------------------------------------------------------------
# figure 2: spin-basis angular correlators
# ---------------------------------------------------------------------------
def fig2_greens_angular(st75: States) -> None:
    sol = {"r": R_GRID,
           "theta": np.deg2rad(np.array([0.0, 30.0, 60.0, 90.0])),
           "G": np.empty((4, len(R_GRID)), dtype=complex),
           "F": np.empty((4, len(R_GRID)), dtype=complex)}
    for i, th in enumerate((0, 30, 60, 90)):
        for j, r in enumerate(R_GRID):
            corr_r = st75.rho(r_vec(r, th))  # rho built from correlators below
            # recompute correlators directly for the plotted components
            kx, ky = st75.bdg["KX"], st75.bdg["KY"]
            ph = np.exp(1j * (kx * r_vec(r, th)[0] + ky * r_vec(r, th)[1]))
            w = 1.0 / kx.size
            N_AB = w * np.einsum("ij,ijab->ab", ph, st75.N_k)
            F_AB = w * np.einsum("ij,ijab->ab", ph.conj(), st75.F_k)
            sol["G"][i, j] = N_AB[0, 0]
            sol["F"][i, j] = F_AB[0, 1]
    rv.plot_greens_angular(75, sol)
    report["figures"]["figs/greens_angular.png"] = "ok"


# ---------------------------------------------------------------------------
# figure 3: C(r,theta,0) for four U values
# ---------------------------------------------------------------------------
def fig3_concurrence_u_theta(U_solutions: dict) -> None:
    solutions = {}
    for U, st in U_solutions.items():
        rhos, _ = rho_grid(st, R_GRID, THETA_DEG)
        C0 = observables(rhos)["C"]                    # (7, 200)
        solutions[U] = {
            "theta": np.deg2rad(np.array(THETA_DEG, dtype=float)),
            "r": R_GRID,
            "initial_obs": {"C": C0[..., None]}}
    rv.plot_concurrence_U_theta(solutions)
    report["figures"]["figs/concurrence_U_theta.png"] = "ok"


# ---------------------------------------------------------------------------
# figure 7/8: heatmaps under the high-noise preset
# ---------------------------------------------------------------------------
def fig7_8_heatmaps(st75: States) -> None:
    t_grid = T_HEAT
    rhos, _ = rho_grid(st75, R_GRID, [0.0])            # theta = 0
    rho0 = rhos[0]
    for i, rho in enumerate(rho0):
        check_rho(rho, f"heatmap r={R_GRID[i]:.3f}", {})
    rho_t = evolve_stack(rho0, t_grid, **HIGH)         # (200,161,4,4)
    obs = observables(rho_t)
    # --- concurrence heatmap (custom, mirrors the fidelity style) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(obs["C"], aspect="auto", origin="lower", cmap="RdYlBu_r",
                   extent=[t_grid[0], t_grid[-1], R_GRID[0], R_GRID[-1]],
                   vmin=0.0, vmax=1.0, interpolation="bilinear")
    levels = np.array([0.1, 0.2, 0.3, 0.5, 0.7])
    levels = levels[levels <= float(np.nanmax(obs["C"]))]
    if levels.size:
        cs = ax.contour(t_grid, R_GRID, obs["C"], levels=levels,
                        colors="white", linewidths=0.9)
        ax.clabel(cs, inline=True, fontsize=9, fmt="C=%.2f")
    div = __import__("mpl_toolkits.axes_grid1", fromlist=["make_axes_locatable"])
    cax = div.make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(r"Concurrence $C$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$r$")
    ax.set_title(rf"Concurrence ($\theta=0^\circ$, $\gamma=4.5$, $\Gamma=0.01$)")
    fig.tight_layout()
    fp = FIGURES / f"{HIGH_PREFIX}_C_heatmap.png"
    fig.savefig(fp, dpi=180, bbox_inches="tight")
    plt.close(fig)
    # --- singlet-fidelity heatmap via the repository's plot style ---
    import visualize_fidelity as vf
    vf.plot_fidelity_heatmap(
        R_GRID, t_grid, obs["Fs"], 0.0, reference_label=r"$\Psi^-$ initial",
        filename=f"{HIGH_PREFIX}_F_heatmap_singlet.png", show=False)
    # envelope maxima of F_s(0,t) = p(t) (eta_amp_lorentz returns p)
    p = np.array([eta_amp_lorentz(t, **HIGH) for t in t_grid])
    from scipy.signal import argrelmax
    peaks = argrelmax(p)[0]
    report["quoted_numbers"]["F0_envelope"] = [
        {"t": float(t_grid[i]), "F": float(p[i])} for i in peaks]
    report["figures"][f"figures/{HIGH_PREFIX}_C_heatmap.png"] = "ok"
    report["figures"][f"figures/{HIGH_PREFIX}_F_heatmap_singlet.png"] = "ok"


# ---------------------------------------------------------------------------
# figure 9: F_s vs t at theta=45 for several r (low noise)
# ---------------------------------------------------------------------------
def fig9_f_vs_t(st75: States) -> None:
    t_grid = T_FULL
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    quoted = {}
    for r in R_VS_T:
        rho0 = st75.rho(r_vec(r, 45.0))
        check_rho(rho0, f"vs_t r={r}", {})
        rho_t = evolve_stack(rho0[None], t_grid, **LOW)[0]
        Fs = observables(rho_t)["Fs"]
        ax.plot(t_grid, Fs, label=fr"$r={r:.5g}$".replace(".0$", ""))
        quoted[f"{r:g}"] = float(Fs[0])
    for y in (0.5, 0.7, 0.8, 0.9):
        ax.axhline(y, color="gray", ls="--", lw=0.7)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\mathcal{F}_{\Psi^-}(r,45^\circ,t)$")
    ax.set_xlim(t_grid[0], t_grid[-1])
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fp = FIGURES / f"{LOW_PREFIX}_F_vs_t_singlet.png"
    fig.savefig(fp, dpi=180, bbox_inches="tight")
    plt.close(fig)
    report["quoted_numbers"]["fig9_initial_Fs"] = quoted
    report["figures"][f"figures/{LOW_PREFIX}_F_vs_t_singlet.png"] = "ok"


# ---------------------------------------------------------------------------
# figure 10: polar initial singlet fraction
# ---------------------------------------------------------------------------
def fig10_f_polar(st75: States) -> None:
    thetas = np.deg2rad(np.arange(0.0, 90.0 + 1e-9, 15.0))
    fig, ax = plt.subplots(figsize=(6.4, 5.4), subplot_kw={"projection": "polar"})
    quoted = {}
    for r in R_POLAR:
        Fs = []
        for th in thetas:
            rho0 = st75.rho(r_vec(r, np.rad2deg(th)))
            Fs.append(float(np.real(SINGLET.conj() @ rho0 @ SINGLET)))
        Fs = np.array(Fs)
        ax.plot(thetas, Fs, label=fr"$r={r:.5g}$")
        quoted[f"{r:g}"] = {"min": float(Fs.min()), "max": float(Fs.max()),
                            "values": [round(float(v), 5) for v in Fs]}
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamax(90)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)
    fp = FIGURES / f"{LOW_PREFIX}_F_polar_singlet.png"
    fig.savefig(fp, dpi=180, bbox_inches="tight")
    plt.close(fig)
    # angular-symmetry checks quoted in the manuscript text
    sym = {}
    for r in (3.3166, 6.6332, 10.0):
        vals = {d: float(np.real(SINGLET.conj()
                @ st75.rho(r_vec(r, d)) @ SINGLET)) for d in
                (0, 15, 30, 45, 60, 75, 90)}
        sym[f"{r:g}"] = vals
        sym[f"{r:g}_sym_err"] = max(
            abs(vals[0] - vals[90]), abs(vals[15] - vals[75]),
            abs(vals[30] - vals[60]))
    report["quoted_numbers"]["fig10_polar"] = quoted
    report["quoted_numbers"]["fig10_symmetry"] = sym
    report["figures"][f"figures/{LOW_PREFIX}_F_polar_singlet.png"] = "ok"


# ---------------------------------------------------------------------------
# figure 11: chi vs r at six times (low noise, theta=45)
# ---------------------------------------------------------------------------
def fig11_holevo_vs_r(st75: States) -> None:
    t_grid = T_FULL
    times = [0, 16, 32, 48, 64, 80]
    rhos, _ = rho_grid(st75, R_GRID, [45.0])
    rho0 = rhos[0]
    rho_t = evolve_stack(rho0, t_grid, **LOW)           # (200,261,4,4)
    obs = observables(rho_t)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    quoted = {}
    for tv in times:
        it = int(np.argmin(np.abs(t_grid - tv)))
        ax.plot(R_GRID, obs["chi"][:, it], label=fr"$t={t_grid[it]:g}$")
    for tv in (16,):
        it = int(np.argmin(np.abs(t_grid - tv)))
        quoted[f"t={tv}"] = [float(obs["chi"][:, it].min()),
                             float(obs["chi"][:, it].max())]
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\chi_{\mathrm{P}}(r,45^\circ,t)$ [bits]")
    ax.set_xlim(R_GRID[0], R_GRID[-1])
    ax.set_ylim(0, 2.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fp = FIGURES / f"{LOW_PREFIX}_holevo_vs_r.png"
    fig.savefig(fp, dpi=180, bbox_inches="tight")
    plt.close(fig)
    report["quoted_numbers"]["fig11_chi_t16_range"] = quoted
    report["figures"][f"figures/{LOW_PREFIX}_holevo_vs_r.png"] = "ok"


# ---------------------------------------------------------------------------
# figure 12: chi(theta, t) at r=0 (low noise)
# ---------------------------------------------------------------------------
def fig12_holevo_theta_t(st75: States) -> None:
    t_grid = T_FULL
    rows = []
    for th in THETA_DEG:
        rho0 = st75.rho(r_vec(0.0, th))
        rho_t = evolve_stack(rho0[None], t_grid, **LOW)[0]
        rows.append(observables(rho_t)["chi"])
    chi = np.array(rows)                                # (7, 261)
    row_ptp = float(np.max(np.ptp(chi, axis=0)))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    im = ax.imshow(chi, aspect="auto", origin="lower", cmap="RdYlBu_r",
                   extent=[t_grid[0], t_grid[-1], -0.5, 6.5],
                   vmin=0.0, vmax=2.0, interpolation="bilinear")
    ax.set_yticks(range(7))
    ax.set_yticklabels([fr"${d}^\circ$" for d in THETA_DEG])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\theta$")
    div = __import__("mpl_toolkits.axes_grid1", fromlist=["make_axes_locatable"])
    cax = div.make_axes_locatable(ax).append_axes("right", size="3%", pad=0.08)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(r"$\chi_{\mathrm{P}}(\theta, r=0, t)$ [bits]")
    fig.tight_layout()
    fp = FIGURES / f"{LOW_PREFIX}_holevo_theta_t_heatmap.png"
    fig.savefig(fp, dpi=180, bbox_inches="tight")
    plt.close(fig)
    report["validation"]["fig12_r0_row_ptp"] = row_ptp
    report["figures"][f"figures/{LOW_PREFIX}_holevo_theta_t_heatmap.png"] = "ok"


# ---------------------------------------------------------------------------
# figure 13: U sweep of F_s and chi (low noise)
# ---------------------------------------------------------------------------
def fig13_u_sweep() -> None:
    t_grid = T_FULL
    nU, nth, nr, nt = len(U_SWEEP), len(THETA_DEG), len(R_GRID), len(t_grid)
    C = np.empty((nU, nth, nr, nt))
    Fs = np.empty_like(C)
    chi = np.empty_like(C)
    quoted = {}
    i_r050 = int(np.argmin(np.abs(R_GRID - 0.50251)))
    i_r150 = int(np.argmin(np.abs(R_GRID - 1.50754)))
    i_th45 = THETA_DEG.index(45)
    for iU, U in enumerate(U_SWEEP):
        st = States(solve(VSO, U))
        rhos, _ = rho_grid(st, R_GRID, THETA_DEG)
        rho_t = evolve_stack(rhos.reshape(-1, 4, 4), t_grid, **LOW)
        obs = observables(rho_t)
        for key, dst in (("C", C), ("Fs", Fs), ("chi", chi)):
            dst[iU] = obs[key].reshape(nth, nr, nt)
        quoted[str(U)] = {
            "Fs_r0p50251": float(Fs[iU, i_th45, i_r050, 0]),
            "chi_r0p50251": float(chi[iU, i_th45, i_r050, 0]),
            "Fs_r1p50754": float(Fs[iU, i_th45, i_r150, 0]),
            "chi_r1p50754": float(chi[iU, i_th45, i_r150, 0])}
    rv.plot_U_sweep(U_SWEEP, R_GRID, t_grid, C, Fs, chi)
    report["quoted_numbers"]["fig13_U_sweep"] = quoted
    report["figures"]["figs/U_dependence_fidelity_holevo.png"] = "ok"
    np.savez_compressed(
        DATA / "vso0p5_regeneration.npz",
        r=R_GRID, theta_deg=np.array(THETA_DEG), t=t_grid,
        Fs_u_sweep=Fs, chi_u_sweep=chi, C_u_sweep=C)


# ---------------------------------------------------------------------------
# internal cross-checks (not shown in the paper)
# ---------------------------------------------------------------------------
def cross_checks(st75: States) -> None:
    errors = {}
    # r=0 must be the pure singlet for every angle at V_SO != 0.
    for th in THETA_DEG:
        rho0 = st75.rho(r_vec(0.0, th))
        check_rho(rho0, f"r0 th={th}", errors)
        Fs = float(np.real(SINGLET.conj() @ rho0 @ SINGLET))
        errors["r0_singlet_err"] = max(errors.get("r0_singlet_err", 0.0),
                                       abs(Fs - 1.0))
    if errors.get("r0_singlet_err", 1.0) > 1e-9:
        raise RuntimeError(f"r=0 not a singlet: {errors}")
    # superoperator vs repository evolve_rho
    rho0 = st75.rho(r_vec(3.3166, 45.0))
    errors["superop_vs_evolve_rho"] = validate_superoperator(rho0, **LOW)
    if errors["superop_vs_evolve_rho"] > 1e-12:
        raise RuntimeError(f"superoperator mismatch: {errors}")
    # At V_SO=0 the helicity construction must reduce to the scalar Oh-Kim
    # X state built from the SAME signed correlators (internal consistency;
    # not a paper figure).  The legacy archived pipeline additionally used
    # the unsigned gap convention, which is a documented, separate gap.
    st0 = States(solve(0.0, 75.0))
    worst = 0.0
    for th in (0.0, 45.0):
        for r in (0.0, 1.0, 3.3166, 6.6332):
            worst = max(worst, float(np.max(np.abs(
                st0.rho(r_vec(r, th)) - st0.ohkim_x(r_vec(r, th))))))
    errors["V0_helicity_vs_OhKim_max_abs"] = worst
    if worst > 1e-10:
        raise RuntimeError(f"V_SO=0 helicity vs Oh-Kim mismatch: {worst}")
    report["validation"] = {**report.get("validation", {}), **errors}


def fig_claims(U_solutions: dict, st75: States) -> None:
    """Extract the numbers quoted in the Results text (figs 2/3)."""
    # fig 3: first zero, revival windows, angular-symmetry errors
    r_sub = R_GRID[R_GRID <= 4.0]
    claims = {"U": {}}
    sym = {"0_vs_90": 0.0, "30_vs_60": 0.0}
    for U, st in U_solutions.items():
        entry = {}
        for th in (0, 30, 60, 90):
            C = observables(np.stack([st.rho(r_vec(r, th)) for r in r_sub]))["C"]
            below = np.where(C < 1e-4)[0]
            first_zero = float(r_sub[below[0]]) if below.size else None
            # revival: local max after first zero with C > 1e-3
            rev = []
            if below.size:
                seg = C[below[0]:]
                for i in range(1, len(seg) - 1):
                    if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 1e-3:
                        rev.append([float(r_sub[below[0] + i]), float(seg[i])])
            entry[str(th)] = {"first_zero": first_zero, "revivals": rev[:3],
                              "C_max_after_zero": float(max(
                                  (v for _, v in rev), default=0.0))}
        claims["U"][str(U)] = entry
        # symmetry of the C(r) curves
        C0 = observables(np.stack([st.rho(r_vec(r, 0)) for r in r_sub]))["C"]
        C90 = observables(np.stack([st.rho(r_vec(r, 90)) for r in r_sub]))["C"]
        C30 = observables(np.stack([st.rho(r_vec(r, 30)) for r in r_sub]))["C"]
        C60 = observables(np.stack([st.rho(r_vec(r, 60)) for r in r_sub]))["C"]
        sym["0_vs_90"] = max(sym["0_vs_90"], float(np.max(np.abs(C0 - C90))))
        sym["30_vs_60"] = max(sym["30_vs_60"], float(np.max(np.abs(C30 - C60))))
    claims["symmetry_max_abs"] = sym
    # fig 2: angular-symmetry errors of |N_upup| and |F_updn|
    gsym = {"0_vs_90": 0.0, "30_vs_60": 0.0}
    for th_a, th_b in ((0, 90), (30, 60)):
        for r in R_GRID:
            ca = st75.correlators(r_vec(r, th_a))
            cb = st75.correlators(r_vec(r, th_b))
            gsym[f"{th_a}_vs_{th_b}"] = max(
                gsym[f"{th_a}_vs_{th_b}"],
                abs(abs(ca["N_AB"][0, 0]) - abs(cb["N_AB"][0, 0])),
                abs(abs(ca["F_AB"][0, 1]) - abs(cb["F_AB"][0, 1])))
    claims["fig2_symmetry_max_abs"] = gsym
    report["quoted_numbers"]["fig3_claims"] = claims


def main() -> None:
    print(f"[regen] solving BdG at V_SO/t={VSO}, Nk={NK} ...")
    solutions = fig1_delta_heatmaps()          # U = 40, 75, 115
    st75 = States(solutions[75]["bdg"])
    print("[regen] fig2 greens_angular")
    fig2_greens_angular(st75)
    print("[regen] fig3 concurrence_U_theta (4 U solves)")
    U_sol = {75: st75}
    for U in U_THETA:
        if U not in U_sol:
            U_sol[U] = States(solve(VSO, U))
    fig3_concurrence_u_theta(U_sol)
    print("[regen] figs7-8 heatmaps (high noise)")
    fig7_8_heatmaps(st75)
    print("[regen] fig9 F_vs_t")
    fig9_f_vs_t(st75)
    print("[regen] fig10 polar")
    fig10_f_polar(st75)
    print("[regen] fig11 holevo_vs_r")
    fig11_holevo_vs_r(st75)
    print("[regen] fig12 holevo_theta_t")
    fig12_holevo_theta_t(st75)
    print("[regen] fig13 U sweep (3 solves)")
    fig13_u_sweep()
    print("[regen] fig3/fig2 claim extraction")
    fig_claims(U_sol, st75)
    print("[regen] cross-checks")
    cross_checks(st75)
    (REPORTS / "vso0p5_regeneration.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print("[regen] done ->", REPORTS / "vso0p5_regeneration.json")
    for k, v in report["figures"].items():
        print("   ", k, v)


if __name__ == "__main__":
    main()
