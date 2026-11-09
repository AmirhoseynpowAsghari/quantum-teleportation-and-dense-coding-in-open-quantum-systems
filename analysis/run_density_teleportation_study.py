#!/usr/bin/env python3
"""Density-resolved BdG Green-function and teleportation study.

Only electron filling is changed.  The lattice, interaction, correlated-hopping
parameter, zero SOC, zero temperature, equal-time Green functions, Oh--Kim
conditional state, and teleportation benchmark are those in the repository.

The production calculation uses Nk=400.  Nk=100, 200, 400, and 800 are compared
along theta=0,45 degrees and at exact lattice sites.  Continuous radial curves
are Fourier interpolations of the lattice model; nonzero physical sites are
reported separately at integer (R_x,R_y).

Run from the repository root:
    .venv/bin/python analysis/run_density_teleportation_study.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson

import config
import greens_functions as gf_module
import kspace_setup
import self_consistency as sc_module
from density_matrix import build_density_matrices
from self_consistency import run_selfconsistency
from teleportation import CLASSICAL_BENCHMARK, resource_metrics, singlet_state

DATA = ROOT / "data"
FIGS = ROOT / "figs"
REPORTS = ROOT / "reports"
for directory in (DATA, FIGS, REPORTS):
    directory.mkdir(exist_ok=True)

DENSITIES = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.875])
NK_VALUES = (100, 200, 400, 800)
NK_PRODUCTION = 400
ANGLES = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
FINE_ANGLES = np.array([0.0, 45.0])
SITES = np.array([
    (1, 0), (1, 1), (2, 0), (2, 1), (2, 2),
    (3, 0), (3, 1), (3, 2), (3, 3), (4, 0),
], dtype=int)
P_THRESHOLD = 1.0 / 3.0
ZERO_GAP_TOL = 1.0e-7
RESIDUAL_TOL = 1.0e-8


def maxabs(value):
    return float(np.max(np.abs(value)))


def json_records(records):
    return np.array([json.dumps(record) for record in records])


@contextmanager
def solver_context(n, nk):
    old = sc_module.n_elec, sc_module.Nk, kspace_setup.Nk
    sc_module.n_elec, sc_module.Nk, kspace_setup.Nk = float(n), int(nk), int(nk)
    try:
        yield
    finally:
        sc_module.n_elec, sc_module.Nk, kspace_setup.Nk = old


def bdg_from_solution(n, nk, d0, ds, mu):
    """Construct arrays and residuals in precisely the solver's convention."""
    with solver_context(n, nk):
        kx, ky, KX, KY, eps, ssin = kspace_setup.build_kgrid()
        sk = 0.5 * (np.cos(KX) + np.cos(KY))
        D = d0 - (ds / (4.0 * config.t_hop)) * eps
        xi = eps - mu
        E = np.clip(np.sqrt(D**2 + xi**2), 1.0e-12, None)
        u = np.sqrt(np.clip(0.5 * (1.0 + xi / E), 0.0, 1.0))
        v = np.sqrt(np.clip(0.5 * (1.0 - xi / E), 0.0, 1.0))
        residuals = np.asarray(sc_module.residual(
            [d0, ds, mu], eps, sk, ssin, nk, config.t_hop, config.U,
            config.Delta_t, config.V_SO, n, np.inf))
    return {
        "Delta0": float(d0), "DeltaS": float(ds), "mu": float(mu),
        "Delta_ks": D, "E_ks": E, "u_ks": u, "v_ks": v,
        "kx": kx, "ky": ky, "KX": KX, "KY": KY,
        "epsilon_ks": eps, "sqrt_sin": ssin, "residuals": residuals,
    }


def normal_stationary_state(n, nk):
    """Best exact zero-gap T=0 stationary state on the finite momentum mesh.

    At a Fermi-level degeneracy the BdG coherence factor is v^2=1/2.  Including
    this fact often gives the requested filling exactly even though the occupied
    number has a discontinuity as a function of mu.  If the requested filling is
    not representable on a mesh, the closest state and its nonzero number
    residual are returned and explicitly labeled.
    """
    with solver_context(n, nk):
        _, _, _, _, eps, _ = kspace_setup.build_kgrid()
    flat = eps[:, :, 0].ravel()
    rounded = np.round(flat, 12)
    levels, counts = np.unique(rounded, return_counts=True)
    below = np.concatenate(([0], np.cumsum(counts)[:-1]))
    n_below = 2.0 * below / nk**2
    n_mid = (2.0 * below + counts) / nk**2
    n_above = 2.0 * (below + counts) / nk**2

    candidates = []
    # Half filling of a degenerate Fermi shell is the natural exact T=0 root.
    for i in range(len(levels)):
        candidates.append((abs(n_mid[i] - n), 0, float(levels[i]),
                           float(n_mid[i]), "half-filled Fermi shell"))
    # A plateau between adjacent energies has a fully occupied/empty shell.
    for i in range(len(levels) - 1):
        mu = 0.5 * (levels[i] + levels[i + 1])
        candidates.append((abs(n_above[i] - n), 1, float(mu),
                           float(n_above[i]), "occupation plateau"))
    error, _, mu, n_actual, mu_type = min(candidates, key=lambda x: (x[0], x[1]))
    # On an even grid the half-filling Fermi contour need not contain an exact
    # grid point.  The selected plateau mu then approaches zero as Nk grows;
    # forcing mu=0 would instead miss the target filling on that finite mesh.
    bdg = bdg_from_solution(n, nk, 0.0, 0.0, mu)
    raw_residuals = np.asarray(bdg["residuals"]).copy()
    # Mathematically degenerate energies can differ by a few ulps after cos(pi)
    # evaluation.  Snap that shell to xi=0 so every member has the required
    # v^2=1/2 instead of an arbitrary sign from roundoff.  The number residual
    # is then evaluated from the same occupations used by the Green functions.
    xi = bdg["epsilon_ks"] - mu
    degenerate = np.abs(xi) < 1.0e-9
    xi = xi.copy()
    xi[degenerate] = 0.0
    E = np.clip(np.abs(xi), 1.0e-12, None)
    bdg["E_ks"] = E
    bdg["u_ks"] = np.sqrt(np.clip(0.5 * (1.0 + xi / E), 0.0, 1.0))
    bdg["v_ks"] = np.sqrt(np.clip(0.5 * (1.0 - xi / E), 0.0, 1.0))
    bdg["residuals"] = np.array([0.0, 0.0, float(n - n_actual)])
    bdg["raw_residuals"] = raw_residuals
    return bdg, {
        "chemical_potential_type": mu_type,
        "discrete_filling": n_actual,
        "absolute_filling_error": float(error),
        "exact_filling_on_mesh": bool(error < 1.0e-12),
        "degeneracy_tolerance_over_t": 1.0e-9,
        "raw_zero_gap_residuals_before_degeneracy_snap": raw_residuals.tolist(),
    }


def paired_solver_state(n, nk, seed=42):
    with solver_context(n, nk):
        result = run_selfconsistency(seed=seed, U_override=config.U,
                                     kBT_override=0.0, verbose=False)
    # Reconstruct to remove any ambiguity about module-state restoration.
    bdg = bdg_from_solution(n, nk, result["Delta0"], result["DeltaS"], result["mu"])
    bdg["converged"] = bool(result["converged"])
    bdg["n_iter"] = int(result["n_iter"])
    return bdg


def solution_record(n, nk, bdg, branch, seed=42, extra=None):
    d0, ds = float(bdg["Delta0"]), float(bdg["DeltaS"])
    discrete_g0 = 0.5 * np.sum(bdg["v_ks"]**2) / nk**2
    record = {
        "n_elec": float(n), "Nk": int(nk), "seed": int(seed),
        "branch": branch,
        "Delta0_over_t": d0, "DeltaS_over_t": ds, "mu_over_t": float(bdg["mu"]),
        "max_abs_Delta_ks_over_t": float(np.max(np.abs(bdg["Delta_ks"]))),
        "min_abs_Delta_ks_over_t": float(np.min(np.abs(bdg["Delta_ks"]))),
        "min_quasiparticle_E_over_t": float(np.min(bdg["E_ks"])),
        "residuals": np.asarray(bdg["residuals"]).tolist(),
        "max_abs_residual": maxabs(bdg["residuals"]),
        "discrete_G0": float(discrete_g0),
        "discrete_filling_from_v2": float(2.0 * discrete_g0),
        "paired": bool(max(abs(d0), abs(ds)) > ZERO_GAP_TOL),
        "global_thermodynamic_stability_established": False,
    }
    if extra:
        record.update(extra)
    return record


def simpson_weights(x):
    # Integrating each unit basis vector reproduces scipy.simpson exactly,
    # including its even-number-of-samples endpoint treatment.
    return simpson(y=np.eye(len(x)), x=x, axis=0)


def fast_greens_equal_time(bdg, r_values, angles_deg):
    """Algebraically batched form of the repository's V_SO=0 GF integral.

    The two degenerate helicity terms and the common 1/2 factor give one spin
    sum.  Simpson quadrature is represented by its exact weight vector.  This is
    checked against greens_functions.compute_greens_functions below.
    """
    kx, ky = bdg["kx"], bdg["ky"]
    wx, wy = simpson_weights(kx), simpson_weights(ky)
    normal = bdg["v_ks"][:, :, 0]**2 * wx[:, None] * wy[None, :]
    anomalous = (bdg["Delta_ks"][:, :, 0] / (2.0 * bdg["E_ks"][:, :, 0])
                 * wx[:, None] * wy[None, :])
    # Fixed-solution channel decomposition, not separately converged states.
    sk = -bdg["epsilon_ks"][:, :, 0] / (4.0 * config.t_hop)
    onsite = (bdg["Delta0"] / (2.0 * bdg["E_ks"][:, :, 0])
              * wx[:, None] * wy[None, :])
    extended = (bdg["DeltaS"] * sk / (2.0 * bdg["E_ks"][:, :, 0])
                * wx[:, None] * wy[None, :])
    r_values = np.asarray(r_values, dtype=float)
    G = np.zeros((len(angles_deg), len(r_values)), complex)
    F = np.zeros_like(G)
    F0 = np.zeros_like(G)
    FS = np.zeros_like(G)
    for ia, angle in enumerate(np.radians(angles_deg)):
        px = np.exp(-1j * np.outer(r_values * np.cos(angle), kx))
        py = np.exp(-1j * np.outer(r_values * np.sin(angle), ky))
        integral_g = np.einsum("ri,ij,rj->r", px, normal, py, optimize=False)
        integral_f = np.einsum("ri,ij,rj->r", px, anomalous, py, optimize=False)
        integral_0 = np.einsum("ri,ij,rj->r", px, onsite, py, optimize=False)
        integral_s = np.einsum("ri,ij,rj->r", px, extended, py, optimize=False)
        G[ia] = 1j * integral_g / (2.0 * np.pi)**2
        F[ia] = 1j * integral_f / (2.0 * np.pi)**2
        F0[ia] = 1j * integral_0 / (2.0 * np.pi)**2
        FS[ia] = 1j * integral_s / (2.0 * np.pi)**2
    return G, F, F0, FS


def normalized_correlators(G, F):
    G0 = np.abs(G[:, [0]])
    g2 = (np.abs(G) / G0)**2
    f2 = (np.abs(F) / G0)**2
    p = (g2 + f2) / (2.0 + f2 - g2)
    return g2, f2, p


def observables(p):
    return np.maximum(0.0, (3.0 * p - 1.0) / 2.0), (1.0 + p) / 2.0


def critical_radius(r, p):
    above = np.asarray(p) > P_THRESHOLD
    for i in range(len(r) - 1):
        if above[i] and not above[i + 1]:
            fraction = (P_THRESHOLD - p[i]) / (p[i + 1] - p[i])
            return float(r[i] + fraction * (r[i + 1] - r[i]))
    return None


def threshold_recrossings(r, p):
    above = np.asarray(p) > P_THRESHOLD
    losses, regains = [], []
    for i in range(len(r) - 1):
        if above[i] == above[i + 1]:
            continue
        fraction = (P_THRESHOLD - p[i]) / (p[i + 1] - p[i])
        crossing = float(r[i] + fraction * (r[i + 1] - r[i]))
        (losses if above[i] else regains).append(crossing)
    return losses, regains


def exact_site_greens(bdg):
    radii = np.hypot(SITES[:, 0], SITES[:, 1])
    angles = np.degrees(np.arctan2(SITES[:, 1], SITES[:, 0]))
    G = np.zeros((1, len(SITES) + 1), complex)
    F = np.zeros_like(G)
    F0 = np.zeros_like(G)
    FS = np.zeros_like(G)
    # G(0) is needed for normalization; each exact vector is then inserted.
    g_origin, f_origin, f0_origin, fs_origin = fast_greens_equal_time(bdg, [0.0], [0.0])
    G[0, 0], F[0, 0] = g_origin[0, 0], f_origin[0, 0]
    F0[0, 0], FS[0, 0] = f0_origin[0, 0], fs_origin[0, 0]
    for i, (radius, angle) in enumerate(zip(radii, angles), start=1):
        g, f, f0, fs = fast_greens_equal_time(bdg, [0.0, radius], [angle])
        G[0, i], F[0, i] = g[0, 1], f[0, 1]
        F0[0, i], FS[0, i] = f0[0, 1], fs[0, 1]
    return radii, angles, G, F, F0, FS


def validate_fast_greens(bdg_paired, bdg_normal):
    old = (gf_module.R_MIN, gf_module.R_MAX, gf_module.N_R, gf_module.THETA_DEG)
    maximum = 0.0
    try:
        for bdg in (bdg_paired, bdg_normal):
            gf_module.R_MIN, gf_module.R_MAX, gf_module.N_R = 0.0, 1.2, 4
            gf_module.THETA_DEG = [0.0, 30.0, 45.0]
            r, theta, G_ref, F_ref = gf_module.compute_greens_functions(
                bdg, verbose=False, convention="equal_time")
            G, F, _, _ = fast_greens_equal_time(bdg, r, np.degrees(theta))
            maximum = max(maximum, maxabs(G - G_ref), maxabs(F - F_ref))
    finally:
        (gf_module.R_MIN, gf_module.R_MAX, gf_module.N_R,
         gf_module.THETA_DEG) = old
    if maximum > 2.0e-13:
        raise RuntimeError(f"Batched Green functions disagree with source: {maximum}")
    return maximum


def werner_matrix(p):
    singlet = np.outer(singlet_state(), singlet_state().conj())
    return (1.0 - p) * np.eye(4) / 4.0 + p * singlet


def protocol_validation(p_samples):
    errors = {"fixed": 0.0, "optimized": 0.0, "concurrence": 0.0,
              "minimum_eigenvalue": np.inf, "maximum_trace_error": 0.0}
    for p in np.asarray(p_samples).ravel():
        rho = werner_matrix(float(p))
        metrics = resource_metrics(rho, validate=True)
        errors["fixed"] = max(errors["fixed"],
                              abs(metrics["fixed_average_fidelity"] - (1+p)/2))
        errors["optimized"] = max(errors["optimized"],
                                  abs(metrics["optimized_average_fidelity"] - (1+p)/2))
        errors["concurrence"] = max(errors["concurrence"],
                                    abs(metrics["concurrence"] - max(0.0, (3*p-1)/2)))
        errors["minimum_eigenvalue"] = min(errors["minimum_eigenvalue"],
                                           float(np.min(np.linalg.eigvalsh(rho))))
        errors["maximum_trace_error"] = max(errors["maximum_trace_error"],
                                            abs(float(np.trace(rho).real)-1.0))
    errors["minimum_eigenvalue"] = float(errors["minimum_eigenvalue"])
    return errors


def plot_solutions(records):
    production = [r for r in records if r["Nk"] == NK_PRODUCTION]
    d0 = [r["Delta0_over_t"] for r in production]
    ds = [r["DeltaS_over_t"] for r in production]
    mu = [r["mu_over_t"] for r in production]
    mu_normal = [r["normal_reference_mu_over_t"] for r in production]
    max_gap = [r["max_abs_Delta_ks_over_t"] for r in production]
    min_e = [r["min_quasiparticle_E_over_t"] for r in production]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), constrained_layout=True)
    axes[0].plot(DENSITIES, d0, "o-", label=r"$\Delta_0/t$")
    axes[0].plot(DENSITIES, ds, "s-", label=r"$\Delta_S/t$")
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set(xlabel="Electron filling $n$", ylabel="Signed gap parameter")
    axes[0].legend()
    axes[1].plot(DENSITIES, mu, "o-", label="selected BdG branch")
    axes[1].plot(DENSITIES, mu_normal, "--", label="normal reference at same $n$")
    axes[1].set(xlabel="Electron filling $n$", ylabel=r"$\mu/t$")
    axes[1].legend()
    axes[2].plot(DENSITIES, max_gap, "o-", label=r"$\max_{\bf k}|\Delta_{\bf k}|/t$")
    axes[2].plot(DENSITIES, min_e, "s--", label=r"$\min_{\bf k}E_{\bf k}/t$")
    axes[2].set(xlabel="Electron filling $n$", ylabel="Energy scale")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(r"Density sweep at fixed $U/t=75$, $\Delta_t/t=4.5$, $V_{SO}=0$, $T=0$, $N_k=400$")
    fig.savefig(FIGS / "density_bdg_solutions.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_greens(r, G, F, g2, f2, p, F0, FS):
    i45 = int(np.where(ANGLES == 45.0)[0][0])
    mask = r <= 4.0
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True)
    for i, n in enumerate(DENSITIES):
        color = plt.get_cmap("viridis")(i/(len(DENSITIES)-1))
        label = fr"$n={n:g}$"
        axes[0, 0].plot(r[mask], np.abs(G[i, i45, mask]), color=color, label=label)
        axes[0, 1].plot(r[mask], np.abs(F[i, i45, mask]), color=color, label=label)
        axes[0, 2].plot(r[mask], g2[i, i45, mask], color=color, label=label)
        axes[1, 0].plot(r[mask], f2[i, i45, mask], color=color, label=label)
        axes[1, 1].plot(r[mask], np.ptp(p[i][:, mask], axis=0), color=color, label=label)
    for ax, title in zip(axes.flat[:5],
                         ("Normal magnitude", "Anomalous magnitude",
                          "Squared normalized normal correlator",
                          "Squared normalized anomalous correlator",
                          r"Angular variation $\max_\theta p-\min_\theta p$")):
        ax.set(xlabel=r"$r/a$", title=title, xlim=(0, 4.0))
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel(r"$|G(r,45^\circ)|$")
    axes[0, 1].set_ylabel(r"$|F(r,45^\circ)|$")
    axes[0, 2].set_ylabel(r"$g^2$")
    axes[1, 0].set_ylabel(r"$f^2$")
    axes[1, 1].set_ylabel(r"$\Delta_\theta p$")
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_ylim(1e-7, 1.2)
    for i, n in enumerate(DENSITIES):
        if n < 1.75:
            continue
        color = plt.get_cmap("viridis")(i/(len(DENSITIES)-1))
        axes[1, 2].plot(r[mask], np.abs(F[i, i45, mask]), color=color,
                        label=fr"$n={n:g}$, total")
        axes[1, 2].plot(r[mask], np.abs(F0[i, i45, mask]), color=color, ls="--",
                        label=fr"$n={n:g}$, $\Delta_0$ part")
        axes[1, 2].plot(r[mask], np.abs(FS[i, i45, mask]), color=color, ls=":",
                        label=fr"$n={n:g}$, $\Delta_S$ part")
    axes[1, 2].set(xlabel=r"$r/a$", ylabel=r"$|F|$", xlim=(0, 4.0),
                   title="Fixed-solution anomalous decomposition")
    axes[1, 2].grid(alpha=0.25)
    axes[0, 0].legend(ncol=2, fontsize=8)
    axes[1, 2].legend(fontsize=7)
    fig.suptitle(r"Density dependence of equal-time lattice Green's functions; $N_k=400$")
    fig.savefig(FIGS / "density_greens_functions.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_resources(r_fine, p_fine, rc_fine, r_full, p_full, rc_full):
    C_fine, F_fine = observables(p_fine)
    C_full, _ = observables(p_full)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True)
    for i, n in enumerate(DENSITIES):
        color = plt.get_cmap("viridis")(i/(len(DENSITIES)-1))
        axes[0, 0].plot(r_fine, C_fine[i, 1], color=color, label=fr"$n={n:g}$")
        axes[0, 1].plot(r_fine, F_fine[i, 1], color=color, label=fr"$n={n:g}$")
        axes[0, 2].plot(r_fine, p_fine[i, 1], color=color, label=fr"$n={n:g}$")
    axes[0, 1].axhline(CLASSICAL_BENCHMARK, color="black", ls="--")
    axes[0, 2].axhline(P_THRESHOLD, color="black", ls="--")
    axes[0, 0].set(xlabel=r"$r/a$", ylabel=r"$C(r,45^\circ)$", xlim=(0, 1.5))
    axes[0, 1].set(xlabel=r"$r/a$", ylabel=r"$\overline{\mathcal{F}}_{tel}$", xlim=(0, 1.5), ylim=(0.48, 1.01))
    axes[0, 2].set(xlabel=r"$r/a$", ylabel="Werner $p$", xlim=(0, 1.5))
    axes[0, 0].legend(ncol=2, fontsize=8)
    axes[1, 0].plot(DENSITIES, rc_fine[:, 0], "o-", label=r"$\theta=0^\circ$")
    axes[1, 0].plot(DENSITIES, rc_fine[:, 1], "s-", label=r"$\theta=45^\circ$")
    axes[1, 0].fill_between(DENSITIES, np.min(rc_full, axis=1),
                            np.max(rc_full, axis=1), alpha=0.2,
                            label="range over seven angles")
    axes[1, 0].set(xlabel="Electron filling $n$", ylabel=r"First continuous $r_c/a$")
    axes[1, 0].legend(fontsize=8)
    image = axes[1, 1].pcolormesh(r_full, DENSITIES, C_full[:, 3],
                                  shading="auto", cmap="viridis", vmin=0, vmax=1)
    axes[1, 1].set(xlabel=r"$r/a$", ylabel="Electron filling $n$", xlim=(0, 2),
                   title=r"Concurrence at $\theta=45^\circ$")
    fig.colorbar(image, ax=axes[1, 1], label="$C$")
    image = axes[1, 2].pcolormesh(r_full, DENSITIES, p_full[:, 3],
                                  shading="auto", cmap="viridis", vmin=0, vmax=1)
    axes[1, 2].contour(r_full, DENSITIES, p_full[:, 3], levels=[P_THRESHOLD],
                       colors=["red"], linewidths=[1.4])
    axes[1, 2].set(xlabel=r"$r/a$", ylabel="Electron filling $n$", xlim=(0, 2),
                   title=r"Werner $p$; red contour $p=1/3$")
    fig.colorbar(image, ax=axes[1, 2], label="$p$")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("Continuous-distance Fourier interpolation; discrete lattice sites are plotted separately")
    fig.savefig(FIGS / "density_entanglement_teleportation.png", dpi=210,
                bbox_inches="tight")
    plt.close(fig)


def plot_sites(site_p, normal_site_p):
    site_r = np.hypot(SITES[:, 0], SITES[:, 1])
    C, F = observables(site_p)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    axes[0].plot(DENSITIES, site_p[:, 0], "o-", label=r"$(1,0)$, $r=a$")
    axes[0].plot(DENSITIES, site_p[:, 1], "s-", label=r"$(1,1)$, $r=\sqrt{2}a$")
    axes[0].plot(DENSITIES, np.max(site_p, axis=1), "^--",
                 label="maximum over 10 tested sites")
    axes[0].axhline(P_THRESHOLD, color="black", ls="--", label=r"$p=1/3$")
    axes[0].set(xlabel="Electron filling $n$", ylabel="Exact-site Werner $p$")
    axes[0].legend(fontsize=8)
    axes[1].plot(DENSITIES, F[:, 0], "o-", label=r"$r=a$")
    axes[1].plot(DENSITIES, F[:, 1], "s-", label=r"$r=\sqrt{2}a$")
    axes[1].axhline(CLASSICAL_BENCHMARK, color="black", ls="--", label="classical $2/3$")
    axes[1].set(xlabel="Electron filling $n$", ylabel=r"$\overline{\mathcal{F}}_{tel}$",
                ylim=(0.49, 0.70))
    axes[1].legend(fontsize=8)
    for i, n in enumerate(DENSITIES):
        color = plt.get_cmap("viridis")(i/(len(DENSITIES)-1))
        axes[2].plot(site_r, site_p[i], "o-", color=color, label=fr"$n={n:g}$")
        if n >= 1.75:
            axes[2].plot(site_r, normal_site_p[i], "x--", color=color,
                         label=fr"$n={n:g}$ normal reference")
    axes[2].axhline(P_THRESHOLD, color="black", ls="--")
    axes[2].set(xlabel=r"Exact site separation $\sqrt{R_x^2+R_y^2}\,a$",
                ylabel="Werner $p$", yscale="log")
    axes[2].legend(ncol=2, fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("Integer lattice vectors only; no continuous radial interpolation")
    fig.savefig(FIGS / "density_lattice_sites.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(convergence):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for i, n in enumerate(DENSITIES):
        color = plt.get_cmap("viridis")(i/(len(DENSITIES)-1))
        rows = [r for r in convergence if r["n_elec"] == n]
        nk = [r["Nk"] for r in rows]
        axes[0].plot(nk, [r["critical_r_over_a_theta45"] for r in rows],
                     "o-", color=color, label=fr"$n={n:g}$")
        axes[1].plot(nk, [r["nearest_site_p"] for r in rows],
                     "o-", color=color, label=fr"$n={n:g}$")
        axes[2].semilogy(nk, [max(r["G0_simpson_relative_error"], 1e-16) for r in rows],
                         "o-", color=color, label=fr"$n={n:g}$")
    axes[0].set(xlabel="$N_k$", ylabel=r"$r_c/a$ at $45^\circ$", xscale="log")
    axes[1].set(xlabel="$N_k$", ylabel=r"$p(r=a)$", xscale="log")
    axes[2].set(xlabel="$N_k$", ylabel=r"$||G(0)|-n/2|/(n/2)$", xscale="log")
    axes[0].legend(ncol=2, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("Momentum-mesh convergence, including the discontinuous normal-state Fermi surface")
    fig.savefig(FIGS / "density_convergence.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_fermi_surfaces(normal_bdg):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    dense = np.linspace(-np.pi, np.pi, 501)
    X, Y = np.meshgrid(dense, dense)
    eps = -2.0 * config.t_hop * (np.cos(X) + np.cos(Y))
    for i, n in enumerate(DENSITIES):
        ax = axes.flat[i]
        mu = normal_bdg[n]["mu"]
        ax.contourf(X, Y, eps <= mu, levels=[-0.5, 0.5, 1.5],
                    colors=["white", "#c7e9c0"])
        ax.contour(X, Y, eps, levels=[mu], colors=["red"], linewidths=[1.6])
        ax.set(xlabel=r"$k_xa$", ylabel=r"$k_ya$", aspect="equal",
               title=fr"$n={n:g}$, normal $\mu/t={mu:.4f}$")
    axes.flat[-1].axis("off")
    fig.suptitle("Exact square-lattice normal-state Fermi surfaces (not a continuum approximation)")
    fig.savefig(FIGS / "density_fermi_surfaces.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_angular_maps(r, p):
    C, _ = observables(p)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
    for i, n in enumerate(DENSITIES):
        ax = axes.flat[i]
        image = ax.pcolormesh(r, ANGLES, C[i], shading="auto", vmin=0, vmax=1,
                              cmap="viridis")
        ax.contour(r, ANGLES, p[i], levels=[P_THRESHOLD], colors=["red"],
                   linewidths=[1.2])
        ax.set(xlabel=r"$r/a$", ylabel=r"$\theta$ (degrees)", xlim=(0, 2),
               title=fr"$n={n:g}$")
    axes.flat[-1].axis("off")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Concurrence", shrink=0.8)
    fig.suptitle("Angular concurrence maps; red contour is the first and any subsequent $p=1/3$ crossing")
    fig.savefig(FIGS / "density_angular_concurrence_maps.png", dpi=210,
                bbox_inches="tight")
    plt.close(fig)


def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    print("Stage 1/7: self-consistent density solutions and multistart checks", flush=True)
    bdg = {}
    normal_bdg = {}
    records = []
    multistart = []
    for nk in NK_VALUES:
        for n in DENSITIES:
            normal, normal_info = normal_stationary_state(n, nk)
            normal_bdg[(float(n), nk)] = normal
            candidate = paired_solver_state(n, nk)
            candidate_residual = maxabs(candidate["residuals"])
            candidate_paired = max(abs(candidate["Delta0"]),
                                   abs(candidate["DeltaS"])) > ZERO_GAP_TOL
            if candidate_paired and candidate_residual < RESIDUAL_TOL:
                selected = candidate
                branch = "nonzero-gap stationary BdG branch"
            else:
                selected = normal
                branch = "zero-gap normal stationary branch"
            extra = dict(normal_info)
            extra.update({
                "normal_reference_mu_over_t": float(normal["mu"]),
                "solver_candidate_max_abs_residual": candidate_residual,
                "solver_candidate_paired": bool(candidate_paired),
                "solver_candidate_converged_flag": bool(candidate.get("converged", False)),
                "solver_function_evaluations": int(candidate.get("n_iter", 0)),
            })
            record = solution_record(n, nk, selected, branch, extra=extra)
            records.append(record)
            bdg[(float(n), nk)] = selected
            print(f"  Nk={nk:3d}, n={n:.3f}: {branch}, D0={selected['Delta0']:+.7f}, "
                  f"DS={selected['DeltaS']:+.7f}, residual={record['max_abs_residual']:.2e}", flush=True)
            if record["max_abs_residual"] > RESIDUAL_TOL:
                print(f"    finite-mesh filling mismatch retained and labeled: {normal_info}", flush=True)

    # Independent seeds at Nk=100.  The built-in solver itself also restarts.
    for n in DENSITIES:
        seed_roots = []
        for seed in (7, 19, 42, 73, 101, 173):
            candidate = paired_solver_state(n, 100, seed=seed)
            residual = maxabs(candidate["residuals"])
            paired = max(abs(candidate["Delta0"]), abs(candidate["DeltaS"])) > ZERO_GAP_TOL
            seed_roots.append({
                "seed": seed, "Delta0_over_t": float(candidate["Delta0"]),
                "DeltaS_over_t": float(candidate["DeltaS"]),
                "mu_over_t": float(candidate["mu"]),
                "max_abs_residual": residual, "paired_candidate": bool(paired),
            })
        paired_roots = [r for r in seed_roots if r["paired_candidate"] and
                        r["max_abs_residual"] < RESIDUAL_TOL]
        multistart.append({
            "n_elec": float(n), "Nk": 100, "seed_results": seed_roots,
            "paired_root_count": len(paired_roots),
            "paired_root_max_Delta0_span": (
                max(r["Delta0_over_t"] for r in paired_roots) -
                min(r["Delta0_over_t"] for r in paired_roots) if paired_roots else None),
            "paired_root_max_DeltaS_span": (
                max(r["DeltaS_over_t"] for r in paired_roots) -
                min(r["DeltaS_over_t"] for r in paired_roots) if paired_roots else None),
            "paired_root_max_mu_span": (
                max(r["mu_over_t"] for r in paired_roots) -
                min(r["mu_over_t"] for r in paired_roots) if paired_roots else None),
        })
        print(f"  multistart n={n:.3f}: {len(paired_roots)} paired roots from six seeds", flush=True)

    print("Stage 2/7: validate batched GF against repository implementation", flush=True)
    gf_implementation_error = validate_fast_greens(
        bdg[(1.875, 100)], bdg[(0.5, 100)])
    print(f"  maximum source-versus-batched GF error: {gf_implementation_error:.3e}", flush=True)

    print("Stage 3/7: production radial/angular Green functions", flush=True)
    r_full = np.linspace(0.0, 10.0, 401)
    r_fine = np.linspace(0.0, 2.0, 801)
    shape_full = (len(DENSITIES), len(ANGLES), len(r_full))
    shape_fine = (len(DENSITIES), len(FINE_ANGLES), len(r_fine))
    arrays = {
        "G_full": np.zeros(shape_full, complex), "F_full": np.zeros(shape_full, complex),
        "F0_full": np.zeros(shape_full, complex), "FS_full": np.zeros(shape_full, complex),
        "G_fine": np.zeros(shape_fine, complex), "F_fine": np.zeros(shape_fine, complex),
        "g2_full": np.zeros(shape_full), "f2_full": np.zeros(shape_full), "p_full": np.zeros(shape_full),
        "g2_fine": np.zeros(shape_fine), "f2_fine": np.zeros(shape_fine), "p_fine": np.zeros(shape_fine),
    }
    normal_comparison = {
        "G_fine": np.zeros(shape_fine, complex), "F_fine": np.zeros(shape_fine, complex),
        "p_fine": np.zeros(shape_fine),
    }
    rc_full = np.zeros((len(DENSITIES), len(ANGLES)))
    rc_fine = np.zeros((len(DENSITIES), len(FINE_ANGLES)))
    rc_normal = np.zeros((len(DENSITIES), len(FINE_ANGLES)))
    recrossing_records = []
    for i, n in enumerate(DENSITIES):
        state = bdg[(float(n), NK_PRODUCTION)]
        G, F, F0, FS = fast_greens_equal_time(state, r_full, ANGLES)
        arrays["G_full"][i], arrays["F_full"][i] = G, F
        arrays["F0_full"][i], arrays["FS_full"][i] = F0, FS
        arrays["g2_full"][i], arrays["f2_full"][i], arrays["p_full"][i] = normalized_correlators(G, F)
        G, F, _, _ = fast_greens_equal_time(state, r_fine, FINE_ANGLES)
        arrays["G_fine"][i], arrays["F_fine"][i] = G, F
        arrays["g2_fine"][i], arrays["f2_fine"][i], arrays["p_fine"][i] = normalized_correlators(G, F)
        normal = normal_bdg[(float(n), NK_PRODUCTION)]
        Gn, Fn, _, _ = fast_greens_equal_time(normal, r_fine, FINE_ANGLES)
        normal_comparison["G_fine"][i], normal_comparison["F_fine"][i] = Gn, Fn
        _, _, normal_comparison["p_fine"][i] = normalized_correlators(Gn, Fn)
        for ia in range(len(ANGLES)):
            rc_full[i, ia] = critical_radius(r_full, arrays["p_full"][i, ia])
            losses, regains = threshold_recrossings(r_full, arrays["p_full"][i, ia])
            recrossing_records.append({
                "n_elec": float(n), "theta_deg": float(ANGLES[ia]),
                "loss_r_over_a": losses, "regain_r_over_a": regains,
                "max_p_after_first_loss": float(np.max(
                    arrays["p_full"][i, ia, r_full > losses[0]])) if losses else None,
            })
        for ia in range(len(FINE_ANGLES)):
            rc_fine[i, ia] = critical_radius(r_fine, arrays["p_fine"][i, ia])
            rc_normal[i, ia] = critical_radius(r_fine, normal_comparison["p_fine"][i, ia])
        print(f"  n={n:.3f}: rc(0)={rc_fine[i,0]:.8f}, rc(45)={rc_fine[i,1]:.8f}", flush=True)

    C_full, Ftel_full = observables(arrays["p_full"])
    C_fine, Ftel_fine = observables(arrays["p_fine"])

    print("Stage 4/7: exact integer lattice vectors and normal-state comparison", flush=True)
    nsite = len(SITES)
    site_arrays = {name: np.zeros((len(DENSITIES), nsite)) for name in
                   ("g2", "f2", "p", "C", "Ftel", "G_abs", "F_abs", "F0_abs", "FS_abs")}
    normal_site_p = np.zeros((len(DENSITIES), nsite))
    site_G = np.zeros((len(DENSITIES), nsite), complex)
    site_F = np.zeros_like(site_G)
    for i, n in enumerate(DENSITIES):
        _, _, G, F, F0, FS = exact_site_greens(bdg[(float(n), NK_PRODUCTION)])
        g2, f2, p = normalized_correlators(G, F)
        C, Ftel = observables(p)
        for key, value in (("g2", g2[0, 1:]), ("f2", f2[0, 1:]), ("p", p[0, 1:]),
                           ("C", C[0, 1:]), ("Ftel", Ftel[0, 1:]),
                           ("G_abs", np.abs(G[0, 1:])), ("F_abs", np.abs(F[0, 1:])),
                           ("F0_abs", np.abs(F0[0, 1:])), ("FS_abs", np.abs(FS[0, 1:]))):
            site_arrays[key][i] = value
        site_G[i], site_F[i] = G[0, 1:], F[0, 1:]
        _, _, Gn, Fn, _, _ = exact_site_greens(normal_bdg[(float(n), NK_PRODUCTION)])
        _, _, pn = normalized_correlators(Gn, Fn)
        normal_site_p[i] = pn[0, 1:]
        print(f"  n={n:.3f}: nearest p={p[0,1]:.9f}, diagonal p={p[0,2]:.9f}, "
              f"max site p={np.max(p[0,1:]):.9f}", flush=True)

    print("Stage 5/7: mesh-convergence curves and exact-site controls", flush=True)
    r_conv = np.linspace(0.0, 1.5, 601)
    p_conv = np.zeros((len(DENSITIES), len(NK_VALUES), len(r_conv)))
    g2_conv = np.zeros_like(p_conv)
    f2_conv = np.zeros_like(p_conv)
    convergence = []
    for i, n in enumerate(DENSITIES):
        for j, nk in enumerate(NK_VALUES):
            state = bdg[(float(n), nk)]
            G, F, _, _ = fast_greens_equal_time(state, r_conv, [45.0])
            g2, f2, p = normalized_correlators(G, F)
            p_conv[i, j], g2_conv[i, j], f2_conv[i, j] = p[0], g2[0], f2[0]
            rc = critical_radius(r_conv, p[0])
            _, _, Gs, Fs, _, _ = exact_site_greens(state)
            gs2, fs2, ps = normalized_correlators(Gs, Fs)
            row = next(r for r in records if r["n_elec"] == n and r["Nk"] == nk)
            convergence.append({
                "n_elec": float(n), "Nk": nk,
                "critical_r_over_a_theta45": rc,
                "nearest_site_g2": float(gs2[0, 1]), "nearest_site_f2": float(fs2[0, 1]),
                "nearest_site_p": float(ps[0, 1]),
                "diagonal_site_p": float(ps[0, 2]),
                "maximum_tested_site_p": float(np.max(ps)),
                "G0_simpson_abs": float(abs(G[0, 0])),
                "G0_occupancy_target_n_over_2": float(n/2),
                "G0_simpson_relative_error": float(abs(abs(G[0, 0])-n/2)/(n/2)),
                "Delta0_over_t": row["Delta0_over_t"], "DeltaS_over_t": row["DeltaS_over_t"],
                "mu_over_t": row["mu_over_t"], "max_abs_residual": row["max_abs_residual"],
                "branch": row["branch"], "exact_filling_on_mesh": row["exact_filling_on_mesh"],
            })
            print(f"  n={n:.3f}, Nk={nk}: rc={rc:.8f}, p(a)={ps[0,1]:.9f}", flush=True)

    print("Stage 6/7: physical-state and protocol validation", flush=True)
    # Sample all branches, both sub- and above-threshold states and exact sites.
    p_samples = list(arrays["p_fine"][:, :, ::100].ravel())
    p_samples += list(site_arrays["p"].ravel())
    protocol_errors = protocol_validation(p_samples)
    if max(protocol_errors[key] for key in ("fixed", "optimized", "concurrence")) > 2e-12:
        raise RuntimeError(f"Werner/protocol identity failure: {protocol_errors}")
    if protocol_errors["minimum_eigenvalue"] < -1e-12:
        raise RuntimeError("Negative sampled Werner eigenvalue")

    # Directly compare the repository X-state constructor with a Werner matrix.
    density_matrix_error = 0.0
    for i, n in enumerate(DENSITIES):
        G = arrays["G_fine"][i, 1:2, :5]
        F = arrays["F_fine"][i, 1:2, :5]
        rho = np.asarray(build_density_matrices(G, F))
        p = arrays["p_fine"][i, 1, :5]
        expected = np.asarray([werner_matrix(value) for value in p])[None, :, :, :]
        density_matrix_error = max(density_matrix_error, maxabs(rho - expected))
    if density_matrix_error > 2e-12:
        raise RuntimeError(f"Density-matrix/Werner mismatch {density_matrix_error}")

    # The fixed-solution anomalous decomposition must add up to the full F.
    decomposition_error = maxabs(arrays["F_full"] - arrays["F0_full"] - arrays["FS_full"])
    if decomposition_error > 2e-12:
        raise RuntimeError("Anomalous channel decomposition does not sum to F")

    print("Stage 7/7: serialize arrays, summaries, and figures", flush=True)
    production_records = [r for r in records if r["Nk"] == NK_PRODUCTION]
    solution_table = []
    correlation_table = []
    site_table = []
    for i, n in enumerate(DENSITIES):
        rec = production_records[i]
        G0 = abs(arrays["G_full"][i, 0, 0])
        F0 = abs(arrays["F_full"][i, 0, 0])
        p_normal_only = arrays["g2_fine"][i] / (2.0 - arrays["g2_fine"][i])
        solution_table.append({
            **rec,
            "G0_simpson_abs": float(G0),
            "G0_simpson_relative_error": float(abs(G0-n/2)/(n/2)),
            "F0_simpson_abs": float(F0),
            "normalized_f_at_origin": float(F0/G0),
            "multistart_Nk100": multistart[i],
        })
        correlation_table.append({
            "n_elec": float(n),
            "critical_r_over_a_theta0": float(rc_fine[i, 0]),
            "critical_r_over_a_theta45": float(rc_fine[i, 1]),
            "critical_r_over_a_min_over_angles": float(np.min(rc_full[i])),
            "critical_r_over_a_max_over_angles": float(np.max(rc_full[i])),
            "critical_r_over_a_normal_reference_theta45": float(rc_normal[i, 1]),
            "max_p_after_first_loss_all_angles": float(max(
                (recrossing_records[i*len(ANGLES)+ia]["max_p_after_first_loss"] or 0.0)
                for ia in range(len(ANGLES)))),
            "any_continuous_concurrence_revival_up_to_r10": bool(any(
                recrossing_records[i*len(ANGLES)+ia]["regain_r_over_a"]
                for ia in range(len(ANGLES)))),
            "max_angular_p_variation": float(np.max(np.ptp(arrays["p_full"][i], axis=0))),
            "max_abs_p_minus_normal_only_diagnostic": float(maxabs(
                arrays["p_fine"][i] - p_normal_only)),
            "max_abs_paired_minus_normal_reference_p": float(maxabs(
                arrays["p_fine"][i] - normal_comparison["p_fine"][i])),
            "nearest_site_p_normal_reference": float(normal_site_p[i, 0]),
            "interpretation_boundary": (
                "The normal-only diagnostic removes f from the Werner formula but retains the paired v_k^2. "
                "The normal-reference comparison instead uses the zero-gap state at the same filling. Neither isolates a separately re-solved Delta0-only or DeltaS-only Hamiltonian."),
        })
        for j, (rx, ry) in enumerate(SITES):
            site_table.append({
                "n_elec": float(n), "Rx": int(rx), "Ry": int(ry),
                "r_over_a": float(np.hypot(rx, ry)),
                "theta_deg": float(np.degrees(np.arctan2(ry, rx))),
                "g2": float(site_arrays["g2"][i, j]), "f2": float(site_arrays["f2"][i, j]),
                "p": float(site_arrays["p"][i, j]), "concurrence": float(site_arrays["C"][i, j]),
                "average_teleportation_fidelity": float(site_arrays["Ftel"][i, j]),
                "above_classical": bool(site_arrays["p"][i, j] > P_THRESHOLD),
                "normal_reference_p": float(normal_site_p[i, j]),
            })

    best_density_index = int(np.argmax(rc_fine[:, 1]))
    strongest_site_index = np.unravel_index(np.argmax(site_arrays["p"]), site_arrays["p"].shape)
    summary = {
        "repository_commit": commit,
        "scope": {
            "density_values": DENSITIES.tolist(),
            "fixed_parameters": {
                "t_hop": config.t_hop, "U_over_t": config.U/config.t_hop,
                "Delta_t_over_t": config.Delta_t/config.t_hop,
                "V_SO_over_t": config.V_SO/config.t_hop, "temperature": 0.0,
                "lattice_constant": 1.0,
            },
            "momentum_meshes": list(NK_VALUES), "production_Nk": NK_PRODUCTION,
            "angles_deg": ANGLES.tolist(), "fine_angles_deg": FINE_ANGLES.tolist(),
            "r_full_max_over_a": float(r_full[-1]), "r_fine_step_over_a": float(r_fine[1]-r_fine[0]),
            "exact_site_coordinates": SITES.tolist(),
            "greens_convention": "equal-time v^2 normal, signed Delta/(2E) anomalous, degenerate-helicity mean 1/2",
            "distance_unit": "r/a; r_code=a=1 in the repository",
            "energy_unit": "t=1",
            "classical_average_fidelity": CLASSICAL_BENCHMARK,
            "Werner_threshold": P_THRESHOLD,
            "continuous_distance_caveat": (
                "Noninteger r/a is a Fourier interpolation, not an additional lattice site. "
                "Exact integer (Rx,Ry) results are the physically allowed separations of the site-based Hamiltonian."),
        },
        "numerical_methods": {
            "paired_solver": "repository fsolve with restarts; six independent seeds at Nk=100",
            "zero_gap_solver": "same T=0 equations evaluated at Delta0=DeltaS=0; degenerate Fermi shells have v^2=1/2",
            "finite_mesh_zero_gap_note": (
                "At Nk=100, n=0.75 and 1.25 are not exactly representable by the zero-gap occupation steps. "
                "Their residuals are reported, not hidden. Production Nk=400 and Nk=800 fillings are exact."),
            "batched_gf_validation_max_abs_error": gf_implementation_error,
            "density_matrix_werner_max_abs_error": density_matrix_error,
            "anomalous_decomposition_max_abs_error": decomposition_error,
            "protocol_validation": protocol_errors,
            "protocol_validation_state_count": len(p_samples),
            "stability_boundary": (
                "A small residual, positive BdG energies, and reproducibility across seeds establish a stationary numerical branch, "
                "not a global free-energy minimum. No global thermodynamic-stability or phase-coexistence calculation is claimed."),
        },
        "solutions_production": solution_table,
        "all_solution_records": records,
        "multistart_checks": multistart,
        "spatial_results": correlation_table,
        "exact_lattice_sites": site_table,
        "exact_lattice_site_overview": {
            "any_above_classical": bool(np.any(site_arrays["p"] > P_THRESHOLD)),
            "maximum_p": float(np.max(site_arrays["p"])),
            "maximum_average_fidelity": float(np.max(site_arrays["Ftel"])),
            "maximum_at_n": float(DENSITIES[strongest_site_index[0]]),
            "maximum_at_coordinate": SITES[strongest_site_index[1]].tolist(),
            "maximum_at_r_over_a": float(np.hypot(*SITES[strongest_site_index[1]])),
            "largest_n_with_any_site_advantage": None,
            "tested_site_count_per_density": len(SITES),
        },
        "continuous_threshold_overview": {
            "longest_rc_theta45_n": float(DENSITIES[best_density_index]),
            "longest_rc_theta45_over_a": float(rc_fine[best_density_index, 1]),
            "all_rc_theta45_over_a": dict(zip(DENSITIES.astype(str), rc_fine[:, 1].tolist())),
            "note": "A continuous rc below a does not create an allowed sublattice detector location.",
        },
        "convergence_records": convergence,
        "recrossings_up_to_r10": recrossing_records,
        "files": {
            "spatial_data": "data/density_teleportation_spatial.npz",
            "convergence_data": "data/density_teleportation_convergence.npz",
            "figures": [
                "figs/density_bdg_solutions.png", "figs/density_greens_functions.png",
                "figs/density_entanglement_teleportation.png", "figs/density_lattice_sites.png",
                "figs/density_convergence.png", "figs/density_fermi_surfaces.png",
                "figs/density_angular_concurrence_maps.png",
            ],
        },
    }

    np.savez_compressed(
        DATA / "density_teleportation_spatial.npz",
        densities=DENSITIES, Nk_production=NK_PRODUCTION,
        r_full=r_full, theta_full_deg=ANGLES, r_fine=r_fine, theta_fine_deg=FINE_ANGLES,
        **arrays, C_full=C_full, Ftel_full=Ftel_full, C_fine=C_fine, Ftel_fine=Ftel_fine,
        rc_full=rc_full, rc_fine=rc_fine, rc_normal_fine=rc_normal,
        normal_G_fine=normal_comparison["G_fine"], normal_F_fine=normal_comparison["F_fine"],
        normal_p_fine=normal_comparison["p_fine"],
        sites=SITES, site_G=site_G, site_F=site_F, normal_site_p=normal_site_p,
        **{f"site_{key}": value for key, value in site_arrays.items()},
        solution_records=json_records(solution_table), repository_commit=np.bytes_(commit))
    np.savez_compressed(
        DATA / "density_teleportation_convergence.npz",
        densities=DENSITIES, Nk_values=np.array(NK_VALUES), r_conv=r_conv,
        p_conv=p_conv, g2_conv=g2_conv, f2_conv=f2_conv,
        convergence_records=json_records(convergence),
        all_solution_records=json_records(records), multistart_records=json_records(multistart),
        repository_commit=np.bytes_(commit))
    (REPORTS / "density_teleportation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    plot_solutions(records)
    plot_greens(r_full, arrays["G_full"], arrays["F_full"], arrays["g2_full"],
                arrays["f2_full"], arrays["p_full"], arrays["F0_full"], arrays["FS_full"])
    plot_resources(r_fine, arrays["p_fine"], rc_fine, r_full, arrays["p_full"], rc_full)
    plot_sites(site_arrays["p"], normal_site_p)
    plot_convergence(convergence)
    plot_fermi_surfaces({n: normal_bdg[(float(n), NK_PRODUCTION)] for n in DENSITIES})
    plot_angular_maps(r_full, arrays["p_full"])

    print(json.dumps({
        "status": "completed",
        "paired_densities": [r["n_elec"] for r in production_records if r["paired"]],
        "rc_theta45": dict(zip(DENSITIES.astype(str), rc_fine[:, 1].tolist())),
        "any_exact_site_above_classical": summary["exact_lattice_site_overview"]["any_above_classical"],
        "maximum_exact_site": summary["exact_lattice_site_overview"],
        "gf_implementation_error": gf_implementation_error,
        "summary": "reports/density_teleportation_summary.json",
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
