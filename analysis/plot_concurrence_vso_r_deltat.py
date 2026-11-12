#!/usr/bin/env python3
"""Compute and plot concurrence C(V_SO) across r, V_SO, and Delta_t.

This script evaluates the self-consistent Hubbard-Rashba BdG solutions across
V_SO/t in [0.0, 1.2] and Delta_t/t in {3.5, 4.0, 4.5, 5.0, 6.0} at U/t=75,
n=1.875, T=0, and computes:

1. Exact conditional two-detection Wick concurrence C_spin(V_SO, r, Delta_t)
   and C_band(V_SO, r, Delta_t) from helicity_density_matrix.py:
   - At exact integer lattice vectors r in {(0,0), (1,0), (1,1), (2,0), (0,2), (3,1)}.
   - Along Fourier-interpolated continuous radial separations r in [0, 0.50]
     where C(r) > 0 up to the critical entanglement radius r_c(V_SO, Delta_t).
2. Comparison with the repository's legacy particle-branch scalar X-state
   construction (convention="legacy_unsigned" in greens_functions.py) at
   r in {0.25, 0.50, 1.00, 1.50}.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

import config
import kspace_setup
import self_consistency
from density_matrix import build_density_matrices
from greens_functions import compute_greens_functions
from helicity_density_matrix import (
    concurrence,
    density_at_separation,
    physical_diagnostics,
    resource_metrics,
)
from kspace_setup import build_kgrid
from observables import concurrence_xstate
from self_consistency import residual

DATA = ROOT / "data"
FIGS = ROOT / "figs"
REPORTS = ROOT / "reports"


def bdg_dict(vso: float, dt: float, u: float, n: float, nk: int,
             sol: np.ndarray) -> dict:
    for mod in (config, kspace_setup, self_consistency):
        mod.V_SO = float(vso)
        mod.Delta_t = float(dt)
        mod.U = float(u)
        mod.n_elec = float(n)
        mod.Nk = int(nk)
    kx, ky, KX, KY, eps_ks, ssin = build_kgrid()
    sk = 0.5 * (np.cos(KX) + np.cos(KY))
    D0, DS, mu_val = float(sol[0]), float(sol[1]), float(sol[2])
    if DS > 0.0:
        D0, DS = -D0, -DS
    res = np.asarray(
        residual([D0, DS, mu_val], eps_ks, sk, ssin, nk, 1.0,
                 float(u), float(dt), float(vso), float(n), np.inf),
        dtype=float,
    )
    Dks = D0 - (DS / 4.0) * eps_ks
    Eks = np.clip(np.sqrt(Dks**2 + (eps_ks - mu_val) ** 2), 1e-12, None)
    uks = np.sqrt(np.clip(0.5 * (1.0 + (eps_ks - mu_val) / Eks), 0.0, 1.0))
    vks = np.sqrt(np.clip(0.5 * (1.0 - (eps_ks - mu_val) / Eks), 0.0, 1.0))
    return {
        "Delta0": D0,
        "DeltaS": DS,
        "mu": mu_val,
        "Delta_ks": Dks,
        "E_ks": Eks,
        "u_ks": uks,
        "v_ks": vks,
        "kx": kx,
        "ky": ky,
        "KX": KX,
        "KY": KY,
        "epsilon_ks": eps_ks,
        "sqrt_sin": ssin,
        "residuals": res,
        "max_abs_residual": float(np.max(np.abs(res))),
    }


def solve_branch_grid(vso_values: np.ndarray, dt_values: np.ndarray,
                      u: float = 75.0, n: float = 1.875,
                      nk: int = 101) -> dict[tuple[float, float], dict]:
    solutions: dict[tuple[float, float], dict] = {}
    seed_high_vso = np.array([-0.15, -2.10, 5.80], dtype=float)
    for dt in sorted(dt_values, reverse=True):
        sol_prev = seed_high_vso.copy()
        for vso in sorted(vso_values, reverse=True):
            for mod in (config, kspace_setup, self_consistency):
                mod.V_SO = float(vso)
                mod.Delta_t = float(dt)
                mod.U = float(u)
                mod.n_elec = float(n)
                mod.Nk = int(nk)
            kx, ky, KX, KY, eps_ks, ssin = build_kgrid()
            sk = 0.5 * (np.cos(KX) + np.cos(KY))
            args = (eps_ks, sk, ssin, nk, 1.0, float(u), float(dt),
                    float(vso), float(n), np.inf)
            sol, _, _, _ = fsolve(
                residual, sol_prev, args=args, full_output=True,
                xtol=1e-12, maxfev=6000
            )
            bdg = bdg_dict(vso, dt, u, n, nk, sol)
            if bdg["max_abs_residual"] > 1e-10 or (
                abs(bdg["Delta0"]) < 1e-5 and abs(bdg["DeltaS"]) < 1e-5
            ):
                for guess in (
                    (-0.05, -0.10, sol[2]),
                    (-0.15, -0.45, sol[2]),
                    (-0.25, -0.90, sol[2]),
                ):
                    cand, _, _, _ = fsolve(
                        residual, np.array(guess, float), args=args,
                        full_output=True, xtol=1e-12, maxfev=6000
                    )
                    cand_bdg = bdg_dict(vso, dt, u, n, nk, cand)
                    if cand_bdg["max_abs_residual"] < 1e-10 and (
                        abs(cand_bdg["Delta0"]) > 1e-5
                        or abs(cand_bdg["DeltaS"]) > 1e-5
                    ):
                        bdg = cand_bdg
                        break
            if bdg["max_abs_residual"] > 1e-10:
                raise RuntimeError(
                    f"Failed to converge Dt={dt}, V_SO={vso}: "
                    f"residual={bdg['max_abs_residual']:.3e}"
                )
            sol_prev = np.array([bdg["Delta0"], bdg["DeltaS"], bdg["mu"]])
            if vso == max(vso_values):
                seed_high_vso = sol_prev.copy()
            solutions[(float(dt), float(vso))] = bdg
    return solutions


def critical_radius_interp(r_grid: np.ndarray, c_vals: np.ndarray) -> float:
    positive = np.where(c_vals > 0.0)[0]
    if len(positive) == 0:
        return 0.0
    last = int(positive[-1])
    if last >= len(r_grid) - 1:
        return float(r_grid[-1])
    # Linear interpolation using the two last positive points if available
    if last >= 1 and c_vals[last - 1] > c_vals[last]:
        slope = (c_vals[last] - c_vals[last - 1]) / (r_grid[last] - r_grid[last - 1])
        r_zero = r_grid[last] - c_vals[last] / slope
        return float(np.clip(r_zero, r_grid[last], r_grid[last + 1]))
    return float(0.5 * (r_grid[last] + r_grid[last + 1]))


def main() -> int:
    DATA.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    dt_values = np.array([3.5, 4.0, 4.5, 5.0, 6.0], dtype=float)
    vso_values = np.round(np.linspace(0.0, 1.2, 13), 2)
    r_short = np.array([0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45], dtype=float)
    r_profile = np.linspace(0.0, 0.50, 51)
    lattice_vectors = np.array([[0, 0], [1, 0], [1, 1], [2, 0], [0, 2], [3, 1]], dtype=int)
    legacy_r_targets = np.array([0.25, 0.50, 1.00, 1.50], dtype=float)

    print("Solving BdG self-consistency across (Delta_t, V_SO) at Nk=101 ...")
    solutions = solve_branch_grid(vso_values, dt_values, nk=101)

    # Mesh convergence check at Nk=201 for representative (Delta_t, V_SO)
    print("Running Nk=201 mesh convergence check at representative points ...")
    check_dt = np.array([3.5, 4.5, 6.0])
    check_vso = np.array([0.0, 0.5, 1.0])
    solutions_201 = solve_branch_grid(check_vso, check_dt, nk=201)

    ndt = len(dt_values)
    nvso = len(vso_values)
    nr_short = len(r_short)
    nr_prof = len(r_profile)
    nlat = len(lattice_vectors)

    Delta0_arr = np.zeros((ndt, nvso))
    DeltaS_arr = np.zeros((ndt, nvso))
    mu_arr = np.zeros((ndt, nvso))
    residual_arr = np.zeros((ndt, nvso))

    C_spin_short_x = np.zeros((ndt, nvso, nr_short))
    C_band_short_x = np.zeros((ndt, nvso, nr_short))
    C_spin_short_diag = np.zeros((ndt, nvso, nr_short))
    C_band_short_diag = np.zeros((ndt, nvso, nr_short))

    C_spin_prof_x = np.zeros((ndt, nvso, nr_prof))
    C_band_prof_x = np.zeros((ndt, nvso, nr_prof))
    rc_spin_x = np.zeros((ndt, nvso))
    rc_band_x = np.zeros((ndt, nvso))

    C_spin_lattice = np.zeros((ndt, nvso, nlat))
    C_band_lattice = np.zeros((ndt, nvso, nlat))
    fef_spin_lattice = np.zeros((ndt, nvso, nlat))
    fef_band_lattice = np.zeros((ndt, nvso, nlat))
    Fnorm_spin_lattice = np.zeros((ndt, nvso, nlat))
    Fnorm_band_lattice = np.zeros((ndt, nvso, nlat))

    max_trace_err = 0.0
    max_herm_err = 0.0
    min_eig = np.inf

    for idt, dt in enumerate(dt_values):
        for iv, vso in enumerate(vso_values):
            bdg = solutions[(float(dt), float(vso))]
            Delta0_arr[idt, iv] = bdg["Delta0"]
            DeltaS_arr[idt, iv] = bdg["DeltaS"]
            mu_arr[idt, iv] = bdg["mu"]
            residual_arr[idt, iv] = bdg["max_abs_residual"]

            for ir, r_val in enumerate(r_short):
                for basis, arr_x, arr_d in (
                    ("spin", C_spin_short_x, C_spin_short_diag),
                    ("band", C_band_short_x, C_band_short_diag),
                ):
                    res_x = density_at_separation(bdg, [r_val, 0.0], basis=basis)
                    diag_x = physical_diagnostics(res_x["rho"])
                    max_trace_err = max(max_trace_err, diag_x["trace_error"])
                    max_herm_err = max(max_herm_err, diag_x["hermiticity_error"])
                    min_eig = min(min_eig, diag_x["minimum_eigenvalue"])
                    arr_x[idt, iv, ir] = concurrence(res_x["rho"])

                    r_d = r_val / np.sqrt(2.0)
                    res_d = density_at_separation(bdg, [r_d, r_d], basis=basis)
                    arr_d[idt, iv, ir] = concurrence(res_d["rho"])

            for ir, r_val in enumerate(r_profile):
                C_spin_prof_x[idt, iv, ir] = concurrence(
                    density_at_separation(bdg, [r_val, 0.0], basis="spin")["rho"]
                )
                C_band_prof_x[idt, iv, ir] = concurrence(
                    density_at_separation(bdg, [r_val, 0.0], basis="band")["rho"]
                )
            rc_spin_x[idt, iv] = critical_radius_interp(r_profile, C_spin_prof_x[idt, iv])
            rc_band_x[idt, iv] = critical_radius_interp(r_profile, C_band_prof_x[idt, iv])

            for il, r_vec in enumerate(lattice_vectors):
                for basis, c_arr, fef_arr, fn_arr in (
                    ("spin", C_spin_lattice, fef_spin_lattice, Fnorm_spin_lattice),
                    ("band", C_band_lattice, fef_band_lattice, Fnorm_band_lattice),
                ):
                    out = density_at_separation(bdg, r_vec, basis=basis)
                    m = resource_metrics(out["rho"])
                    c_arr[idt, iv, il] = m["concurrence"]
                    fef_arr[idt, iv, il] = m["fully_entangled_fraction"]
                    fn_arr[idt, iv, il] = float(np.linalg.norm(out["F_AB"]))

    # Compute legacy particle-branch scalar X-state concurrence for Dt in {3.5, 4.5, 6.0}
    legacy_dt_indices = [0, 2, 4]
    C_legacy_x = np.zeros((len(legacy_dt_indices), nvso, len(legacy_r_targets)))
    C_legacy_diag = np.zeros((len(legacy_dt_indices), nvso, len(legacy_r_targets)))
    C_equal_x = np.zeros((len(legacy_dt_indices), nvso, len(legacy_r_targets)))
    r_grid_gf = None
    for idx_pos, idt in enumerate(legacy_dt_indices):
        dt = dt_values[idt]
        for iv, vso in enumerate(vso_values):
            bdg = solutions[(float(dt), float(vso))]
            r_grid_gf, _, G_leg, F_leg = compute_greens_functions(
                bdg, verbose=False, convention="legacy_unsigned"
            )
            _, _, G_eq, F_eq = compute_greens_functions(
                bdg, verbose=False, convention="equal_time"
            )
            rho_leg = np.asarray(build_density_matrices(G_leg, F_leg))
            rho_eq = np.asarray(build_density_matrices(G_eq, F_eq))
            for ir_t, r_target in enumerate(legacy_r_targets):
                j_idx = int(np.argmin(np.abs(r_grid_gf - r_target)))
                C_legacy_x[idx_pos, iv, ir_t] = concurrence_xstate(rho_leg[0, j_idx])
                C_legacy_diag[idx_pos, iv, ir_t] = concurrence_xstate(rho_leg[3, j_idx])
                C_equal_x[idx_pos, iv, ir_t] = concurrence_xstate(rho_eq[0, j_idx])

    # Mesh convergence differences between Nk=101 and Nk=201
    mesh_diffs = []
    for dt in check_dt:
        for vso in check_vso:
            b101 = solutions[(float(dt), float(vso))]
            b201 = solutions_201[(float(dt), float(vso))]
            c101_s = concurrence(density_at_separation(b101, [0.25, 0.0], "spin")["rho"])
            c201_s = concurrence(density_at_separation(b201, [0.25, 0.0], "spin")["rho"])
            c101_b = concurrence(density_at_separation(b101, [0.25, 0.0], "band")["rho"])
            c201_b = concurrence(density_at_separation(b201, [0.25, 0.0], "band")["rho"])
            c101_10 = concurrence(density_at_separation(b101, [1.0, 0.0], "spin")["rho"])
            c201_10 = concurrence(density_at_separation(b201, [1.0, 0.0], "spin")["rho"])
            mesh_diffs.append({
                "Delta_t_over_t": float(dt),
                "V_SO_over_t": float(vso),
                "abs_diff_Delta0": float(abs(b101["Delta0"] - b201["Delta0"])),
                "abs_diff_DeltaS": float(abs(b101["DeltaS"] - b201["DeltaS"])),
                "abs_diff_C_spin_r025": float(abs(c101_s - c201_s)),
                "abs_diff_C_band_r025": float(abs(c101_b - c201_b)),
                "C_spin_r10_Nk101": float(c101_10),
                "C_spin_r10_Nk201": float(c201_10),
            })

    # -------------------------------------------------------------------------
    # Figure 1: Exact conditional Wick concurrence C(V_SO) across r and Delta_t
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.8), constrained_layout=True)
    dt_colors = ["#1f77b4", "#2ca02c", "#d95f02", "#9467bd", "#d62728"]

    # Panel (a): Self-consistent BdG gaps |Delta_0| and |Delta_S| vs V_SO
    ax = axes[0, 0]
    for idt, dt in enumerate(dt_values):
        ax.plot(vso_values, np.abs(DeltaS_arr[idt]), "o-", color=dt_colors[idt],
                label=fr"$\Delta_t/t={dt:.1f}$, $|\Delta_S|$")
        ax.plot(vso_values, np.abs(Delta0_arr[idt]), "s--", color=dt_colors[idt],
                mfc="none", alpha=0.8)
    ax.set_xlabel(r"$V_{SO}/t$")
    ax.set_ylabel(r"Self-consistent gap amplitude ($/t$)")
    ax.set_title(r"(a) BdG gaps $|\Delta_S|$ (solid) & $|\Delta_0|$ (dashed)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, ncol=1)

    # Panel (b): C(V_SO) vs V_SO for different separations r at Delta_t/t = 4.5
    ax = axes[0, 1]
    idt_45 = 2
    r_plot_indices = [0, 2, 4, 6, 7]  # r = 0.0, 0.15, 0.25, 0.35, 0.40
    r_colors = ["#000000", "#1b9e77", "#1f78b4", "#e66101", "#7570b3"]
    for c_col, ir in zip(r_colors, r_plot_indices):
        rv = r_short[ir]
        ax.plot(vso_values, C_spin_short_x[idt_45, :, ir], "o-", color=c_col,
                label=fr"$r={rv:.2f}$ (spin)")
        if rv > 0:
            ax.plot(vso_values, C_band_short_x[idt_45, :, ir], "s--", color=c_col,
                    mfc="none", label=fr"$r={rv:.2f}$ (band)")
    ax.plot(vso_values, C_spin_lattice[idt_45, :, 1], "d:", color="#d62728",
            lw=2.0, label=r"exact $\mathbf{r}=(1,0),(1,1),(2,0)$")
    ax.set_xlabel(r"$V_{SO}/t$")
    ax.set_ylabel(r"Concurrence $C(V_{SO})$")
    ax.set_title(r"(b) $C(V_{SO})$ for different $r$ at $\Delta_t/t=4.5$")
    ax.set_ylim(-0.04, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.0, ncol=2)

    # Panel (c): C(V_SO) vs V_SO across all Delta_t at r=0.25 and r=0.35
    ax = axes[0, 2]
    ir_025 = 4  # r = 0.25
    for idt, dt in enumerate(dt_values):
        ax.plot(vso_values, C_spin_short_x[idt, :, ir_025], "o-",
                color=dt_colors[idt], label=fr"$\Delta_t/t={dt:.1f}$ (spin)")
        ax.plot(vso_values, C_band_short_x[idt, :, ir_025], "s--",
                color=dt_colors[idt], mfc="none",
                label=fr"$\Delta_t/t={dt:.1f}$ (band)")
    ax.set_xlabel(r"$V_{SO}/t$")
    ax.set_ylabel(r"Concurrence $C(V_{SO})$ at $r=0.25$")
    ax.set_title(r"(c) $C(V_{SO})$ vs $\Delta_t/t$ at $r=0.25$ ($\theta=0^\circ$)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.0, ncol=2)

    # Panel (d): Basis splitting C_spin - C_band vs V_SO across Delta_t and r
    ax = axes[1, 0]
    for idt, dt in enumerate(dt_values):
        diff_025 = C_spin_short_x[idt, :, 4] - C_band_short_x[idt, :, 4]
        diff_035 = C_spin_short_x[idt, :, 6] - C_band_short_x[idt, :, 6]
        ax.plot(vso_values, 1e3 * diff_025, "o-", color=dt_colors[idt],
                label=fr"$\Delta_t/t={dt:.1f}$, $r=0.25$")
        ax.plot(vso_values, 1e3 * diff_035, "s--", color=dt_colors[idt],
                mfc="none", label=fr"$\Delta_t/t={dt:.1f}$, $r=0.35$")
    ax.axhline(0.0, color="black", lw=0.8, ls=":")
    ax.set_xlabel(r"$V_{SO}/t$")
    ax.set_ylabel(r"$10^3 \times [C_{\mathrm{spin}} - C_{\mathrm{band}}]$")
    ax.set_title(r"(d) Spin vs. helicity-band concurrence splitting")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.0, ncol=2)

    # Panel (e): Radial profiles C(r) vs r for selected V_SO and Delta_t
    ax = axes[1, 1]
    for idt, ls_dt in ((0, "-"), (2, "--"), (4, ":")):
        dt = dt_values[idt]
        for iv, col in ((0, "#1f77b4"), (5, "#2ca02c"), (10, "#d62728")):
            vso = vso_values[iv]
            ax.plot(r_profile, C_spin_prof_x[idt, iv], ls=ls_dt, color=col,
                    lw=1.8, label=fr"$\Delta_t={dt:.1f}, V_{{SO}}={vso:.1f}$")
    ax.set_xlabel(r"Separation $r$ (lattice constants, $\theta=0^\circ$)")
    ax.set_ylabel(r"Concurrence $C_{\mathrm{spin}}(r)$")
    ax.set_title(r"(e) Radial decay $C_{\mathrm{spin}}(r)$ across $(V_{SO},\Delta_t)$")
    ax.set_xlim(0.0, 0.48)
    ax.set_ylim(-0.02, 1.04)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.0, ncol=2)

    # Panel (f): Entanglement critical radius r_c(V_SO, Delta_t)
    ax = axes[1, 2]
    for idt, dt in enumerate(dt_values):
        ax.plot(vso_values, rc_spin_x[idt], "o-", color=dt_colors[idt],
                label=fr"$\Delta_t/t={dt:.1f}$ (spin)")
        ax.plot(vso_values, rc_band_x[idt], "s--", color=dt_colors[idt],
                mfc="none", label=fr"$\Delta_t/t={dt:.1f}$ (band)")
    ax.set_xlabel(r"$V_{SO}/t$")
    ax.set_ylabel(r"Critical entanglement radius $r_c$")
    ax.set_title(r"(f) Entanglement death radius $r_c(V_{SO},\Delta_t)$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.0, ncol=2)

    fig.suptitle(
        r"Exact conditional Wick concurrence $C(V_{SO}, r, \Delta_t)$ at $U/t=75, n=1.875, T=0$ ($N_k=101$)",
        fontsize=13,
    )
    fig.savefig(FIGS / "concurrence_vso_r_deltat_wick.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 2: Exact lattice site r=(1,0) diagnostics & legacy X-state curves
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=True)

    # Panel (a): Anomalous pair norm ||F_AB(r=(1,0))||_F vs V_SO
    ax = axes[0, 0]
    for idt, dt in enumerate(dt_values):
        ax.plot(vso_values, Fnorm_spin_lattice[idt, :, 1], "o-",
                color=dt_colors[idt], label=fr"$\Delta_t/t={dt:.1f}$ ($\|F_{{AB}}\|_F$)")
    ax.set_xlabel(r"$V_{SO}/t$")
    ax.set_ylabel(r"$\|F_{AB}(\mathbf{r}=(1,0))\|_F$")
    ax.set_title(r"(a) Exact nearest-neighbor pair norm $\|F_{AB}(1,0)\|_F$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    # Panel (b): Fully entangled fraction at r=(1,0) vs V_SO (threshold f=0.5)
    ax = axes[0, 1]
    for idt, dt in enumerate(dt_values):
        ax.plot(vso_values, fef_spin_lattice[idt, :, 1], "o-",
                color=dt_colors[idt], label=fr"$\Delta_t/t={dt:.1f}$ (spin)")
        ax.plot(vso_values, fef_band_lattice[idt, :, 1], "s--",
                color=dt_colors[idt], mfc="none", label=fr"$\Delta_t/t={dt:.1f}$ (band)")
    ax.set_xlabel(r"$V_{SO}/t$")
    ax.set_ylabel(r"Fully entangled fraction $\mathfrak{f}(\mathbf{r}=(1,0))$")
    ax.set_title(r"(b) Nearest-neighbor $\mathfrak{f}(1,0)$ (entangled iff $\mathfrak{f}>0.5$)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, ncol=2)

    # Panel (c): Legacy particle-branch scalar X-state C(V_SO) at theta=0 deg
    ax = axes[1, 0]
    leg_colors = ["#1f77b4", "#d95f02", "#d62728"]
    r_styles = ["-", "--", "-.", ":"]
    for idx_pos, idt in enumerate(legacy_dt_indices):
        dt = dt_values[idt]
        for ir_t, r_target in enumerate(legacy_r_targets):
            rv_actual = r_grid_gf[int(np.argmin(np.abs(r_grid_gf - r_target)))]
            ax.plot(
                vso_values, C_legacy_x[idx_pos, :, ir_t],
                ls=r_styles[ir_t], marker="o", ms=3.5, color=leg_colors[idx_pos],
                label=fr"$\Delta_t={dt:.1f}, r\approx{rv_actual:.2f}$",
            )
    ax.set_xlabel(r"$V_{SO}/t$")
    ax.set_ylabel(r"Legacy particle-branch $C^{\mathrm{legacy}}(V_{SO})$")
    ax.set_title(r"(c) Legacy particle-branch $X$-state $C(V_{SO})$ ($\theta=0^\circ$)")
    ax.set_ylim(-0.03, 0.95)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.0, ncol=2)

    # Panel (d): Legacy particle-branch scalar X-state C(V_SO) at theta=45 deg
    ax = axes[1, 1]
    for idx_pos, idt in enumerate(legacy_dt_indices):
        dt = dt_values[idt]
        for ir_t, r_target in enumerate(legacy_r_targets):
            rv_actual = r_grid_gf[int(np.argmin(np.abs(r_grid_gf - r_target)))]
            ax.plot(
                vso_values, C_legacy_diag[idx_pos, :, ir_t],
                ls=r_styles[ir_t], marker="s", ms=3.5, color=leg_colors[idx_pos],
                label=fr"$\Delta_t={dt:.1f}, r\approx{rv_actual:.2f}$",
            )
    ax.set_xlabel(r"$V_{SO}/t$")
    ax.set_ylabel(r"Legacy particle-branch $C^{\mathrm{legacy}}(V_{SO})$")
    ax.set_title(r"(d) Legacy particle-branch $X$-state $C(V_{SO})$ ($\theta=45^\circ$)")
    ax.set_ylim(-0.03, 0.95)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.0, ncol=2)

    fig.suptitle(
        r"Exact lattice-site diagnostics ($\mathbf{r}=(1,0)$) vs. legacy particle-branch $X$-state reduction",
        fontsize=13,
    )
    fig.savefig(
        FIGS / "concurrence_vso_r_deltat_lattice_and_legacy.png",
        dpi=220, bbox_inches="tight",
    )
    plt.close(fig)

    np.savez_compressed(
        DATA / "concurrence_vso_r_deltat.npz",
        Delta_t_over_t=dt_values,
        V_SO_over_t=vso_values,
        r_short=r_short,
        r_profile=r_profile,
        lattice_vectors=lattice_vectors,
        legacy_r_targets=legacy_r_targets,
        Delta0_over_t=Delta0_arr,
        DeltaS_over_t=DeltaS_arr,
        mu_over_t=mu_arr,
        max_abs_residual=residual_arr,
        C_spin_short_x=C_spin_short_x,
        C_band_short_x=C_band_short_x,
        C_spin_short_diag=C_spin_short_diag,
        C_band_short_diag=C_band_short_diag,
        C_spin_prof_x=C_spin_prof_x,
        C_band_prof_x=C_band_prof_x,
        rc_spin_x=rc_spin_x,
        rc_band_x=rc_band_x,
        C_spin_lattice=C_spin_lattice,
        C_band_lattice=C_band_lattice,
        fef_spin_lattice=fef_spin_lattice,
        fef_band_lattice=fef_band_lattice,
        Fnorm_spin_lattice=Fnorm_spin_lattice,
        Fnorm_band_lattice=Fnorm_band_lattice,
        C_legacy_x=C_legacy_x,
        C_legacy_diag=C_legacy_diag,
        C_equal_x=C_equal_x,
    )

    summary = {
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "parameters": {
            "U_over_t": 75.0,
            "n_elec": 1.875,
            "temperature": 0.0,
            "Nk_primary": 101,
            "Nk_convergence_check": 201,
            "Delta_t_over_t": dt_values.tolist(),
            "V_SO_over_t": vso_values.tolist(),
        },
        "diagnostics": {
            "max_solver_residual_Nk101": float(np.max(residual_arr)),
            "max_trace_error": float(max_trace_err),
            "max_hermiticity_error": float(max_herm_err),
            "min_density_eigenvalue": float(min_eig),
            "max_nonzero_lattice_concurrence_spin": float(np.max(C_spin_lattice[:, :, 1:])),
            "max_nonzero_lattice_concurrence_band": float(np.max(C_band_lattice[:, :, 1:])),
        },
        "mesh_convergence_Nk101_vs_Nk201": mesh_diffs,
        "generated_figures": [
            "figs/concurrence_vso_r_deltat_wick.png",
            "figs/concurrence_vso_r_deltat_lattice_and_legacy.png",
        ],
    }
    (REPORTS / "concurrence_vso_r_deltat.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["diagnostics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
