#!/usr/bin/env python3
"""Validate the scope of the Oh--Kim two-spin density-matrix reduction.

This script does not modify the manuscript's archived results.  It checks
lattice symmetries and Werner positivity using the archived V_SO=0 arrays,
constructs the signed Delta/(2E) anomalous correlator for comparison, and
shows the spin-off-diagonal normal correlations that appear in the original
spin basis when Rashba coupling is nonzero.

Run from the repository root:
    .venv/bin/python analysis/validate_density_matrix_scope.py
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
from scipy.integrate import simpson

import config
import kspace_setup
import self_consistency
from self_consistency import run_selfconsistency

DATA = ROOT / "data"
FIGS = ROOT / "figs"
REPORTS = ROOT / "reports"
FIGS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

ARCHIVED = DATA / "hubbard_amp_independent_gA1.50_GA0.100_gB1.50_GB0.100_greens.npz"


def integral2(values, kx, ky):
    """Use the same Brillouin-zone quadrature as greens_functions.py."""
    return (simpson(y=simpson(y=values, x=kx, axis=0), x=ky, axis=0)
            / (2.0 * np.pi) ** 2)


def solve_at_vso(vso):
    """Temporarily set V_SO in every module that imported it by value."""
    config.V_SO = float(vso)
    kspace_setup.V_SO = float(vso)
    self_consistency.V_SO = float(vso)
    bdg = run_selfconsistency(seed=42, U_override=75, kBT_override=0.0,
                              verbose=False)
    residual = float(np.max(np.abs(bdg["residuals"])))
    if not bdg["converged"] or residual > 1e-8:
        raise RuntimeError(f"Unreliable solution at V_SO={vso}: {residual}")
    return bdg, residual


def normalized_ratios(G, F):
    """Repository magnitude-amplitude reduction."""
    G0 = np.abs(G[:, [0]])
    return np.abs(G) / G0, np.abs(F) / G0


def werner_parameter(g, f):
    return (f * f + g * g) / (2.0 + f * f - g * g)


def xstate(g, f):
    """Construct Oh--Kim Eq. (21) from real normalized correlators."""
    g2, f2 = np.asarray(g) ** 2, np.asarray(f) ** 2
    norm = 4.0 - 2.0 * g2 + 2.0 * f2
    rho = np.zeros(g2.shape + (4, 4), dtype=complex)
    rho[..., 0, 0] = 1.0 - g2
    rho[..., 1, 1] = 1.0 + f2
    rho[..., 2, 2] = 1.0 + f2
    rho[..., 3, 3] = 1.0 - g2
    rho[..., 1, 2] = -(g2 + f2)
    rho[..., 2, 1] = -(g2 + f2)
    return rho / norm[..., None, None]


def werner_state(p):
    p = np.asarray(p)
    rho = np.broadcast_to(np.eye(4) / 4.0, p.shape + (4, 4)).copy()
    singlet = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2.0)
    projector = np.outer(singlet, singlet.conj())
    return (1.0 - p[..., None, None]) * rho + p[..., None, None] * projector


def signed_anomalous(bdg, r_vals, theta_vals):
    """F using u v* = -Delta*/(2E), retaining the sign before BZ integration."""
    kx, ky, KX, KY = (bdg[key] for key in ("kx", "ky", "KX", "KY"))
    Delta, E = bdg["Delta_ks"], bdg["E_ks"]
    coherence = -np.conj(Delta) / (2.0 * E)
    F = np.zeros((len(theta_vals), len(r_vals)), dtype=complex)
    for i, theta in enumerate(theta_vals):
        for j, r in enumerate(r_vals):
            phase = np.exp(1j * r * (KX * np.cos(theta) + KY * np.sin(theta)))
            integrand = np.sum(1j * coherence * np.conj(phase[..., None]), axis=2)
            F[i, j] = integral2(integrand, kx, ky)
    return F


def original_spin_normal_correlators(bdg, r_vals, theta=0.0):
    """Equal-time normal correlators after helicity -> original spin rotation.

    Hutchinson et al. Eq. (16) gives
      c_up = (c_+ + c_-)/sqrt(2),
      c_down = exp(i theta_k)(c_+ - c_-)/sqrt(2).
    For n_s = <c_{k,s}^dagger c_{k,s}>, the k-resolved spin-off-diagonal
    coefficient is exp(i theta_k)(n_+ - n_-)/2.  Its spatial Fourier sum is
    evaluated directly along the requested direction.
    """
    kx, ky, KX, KY = (bdg[key] for key in ("kx", "ky", "KX", "KY"))
    n = bdg["v_ks"] ** 2
    lam = bdg["sqrt_sin"]
    phase_spin = np.zeros_like(lam, dtype=complex)
    nonzero = lam > 1e-14
    phase_spin[nonzero] = ((np.sin(KY[nonzero]) - 1j * np.sin(KX[nonzero]))
                           / lam[nonzero])
    diagonal_weight = 0.5 * (n[:, :, 0] + n[:, :, 1])
    offdiag_weight = 0.5 * phase_spin * (n[:, :, 0] - n[:, :, 1])
    Guu, Gud = np.zeros(len(r_vals), complex), np.zeros(len(r_vals), complex)
    for j, r in enumerate(r_vals):
        spatial = np.exp(1j * r * (KX * np.cos(theta) + KY * np.sin(theta)))
        Guu[j] = integral2(diagonal_weight * spatial, kx, ky)
        Gud[j] = integral2(offdiag_weight * spatial, kx, ky)
    return Guu, Gud


def plot_spatial_checks(r, theta_deg, G, F, g, f, metrics):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)
    for i, label in [(0, "0"), (6, "90"), (2, "30"), (4, "60")]:
        axes[0, 0].plot(r, np.abs(G[i]), label=fr"$\theta={label}^\circ$")
        axes[0, 1].plot(r, np.abs(F[i]), label=fr"$\theta={label}^\circ$")
    axes[0, 0].set_ylabel(r"$|G_{\uparrow\uparrow}(r,\theta)|$")
    axes[0, 1].set_ylabel(r"$|F_{\uparrow\downarrow}(r,\theta)|$")
    for ax in axes[0]:
        ax.set_xlim(0, 6)
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)

    i45 = int(np.argmin(np.abs(theta_deg - 45)))
    axes[1, 0].plot(r, g[i45] ** 2, label=r"$g^2$")
    axes[1, 0].plot(r, f[i45] ** 2, label=r"$f^2$")
    axes[1, 0].set_ylabel("Squared normalized correlator")
    axes[1, 0].set_xlim(0, 6)
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    floor = 1e-18
    axes[1, 1].semilogy(r, np.maximum(np.abs(G[0] - G[6]), floor),
                        label=r"$G:0^\circ-90^\circ$")
    axes[1, 1].semilogy(r, np.maximum(np.abs(G[2] - G[4]), floor),
                        label=r"$G:30^\circ-60^\circ$")
    axes[1, 1].semilogy(r, np.maximum(np.abs(F[0] - F[6]), floor),
                        "--", label=r"$F:0^\circ-90^\circ$")
    axes[1, 1].semilogy(r, np.maximum(np.abs(F[2] - F[4]), floor),
                        "--", label=r"$F:30^\circ-60^\circ$")
    axes[1, 1].set_ylabel("Absolute symmetry residual")
    axes[1, 1].set_ylim(1e-18, 1e-14)
    axes[1, 1].set_xlim(0, 6)
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel(r"$r$ (lattice constant)")
    fig.suptitle("Spatial-correlator checks for the archived $V_{SO}=0$, $U/t=75$ state")
    fig.text(0.5, -0.015,
             "Discrete $C_4$ symmetry is a consistency check, not a substitute for the analytical translation proof.",
             ha="center", fontsize=9)
    fig.savefig(FIGS / "density_matrix_spatial_checks.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def plot_werner_checks(r, theta_deg, g, f, p, rho, rho_w, metrics):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)
    for angle in (0, 30, 45, 60, 90):
        i = int(np.argmin(np.abs(theta_deg - angle)))
        axes[0, 0].plot(r, p[i], label=fr"$\theta={angle}^\circ$")
    axes[0, 0].set_ylabel(r"Werner parameter $p(r,\theta)$")
    axes[0, 0].set_ylim(-0.02, 1.02)
    axes[0, 0].set_xlim(0, 6)
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(ncol=2)

    i45 = int(np.argmin(np.abs(theta_deg - 45)))
    lam_s = (1 + 3 * p[i45]) / 4
    lam_t = (1 - p[i45]) / 4
    axes[0, 1].plot(r, lam_s, label=r"$\lambda_s=(1+3p)/4$")
    axes[0, 1].plot(r, lam_t, label=r"$\lambda_t=(1-p)/4$ (threefold)")
    axes[0, 1].axhline(0, color="black", lw=0.8)
    axes[0, 1].set_ylabel(r"Eigenvalues at $\theta=45^\circ$")
    axes[0, 1].set_xlim(0, 6)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()

    min_triplet = np.min((1 - p) / 4, axis=0)
    axes[1, 0].plot(r, min_triplet)
    axes[1, 0].axhline(0, color="black", lw=0.8)
    axes[1, 0].set_ylabel(r"$\min_\theta \lambda_t(r,\theta)$")
    axes[1, 0].set_xlim(0, 6)
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].axis("off")
    summary = (
        "Independent algebraic/numerical checks\n\n"
        fr"$\max|\rho_{{X}}-\rho_{{W}}| = {metrics['werner_equivalence_max_abs_error']:.3e}$" "\n"
        fr"$\max|\mathrm{{Tr}}\rho-1| = {metrics['trace_max_abs_error']:.3e}$" "\n"
        fr"$\max|\rho-\rho^\dagger| = {metrics['hermiticity_max_abs_error']:.3e}$" "\n"
        fr"$\min\lambda = {metrics['minimum_eigenvalue']:.3e}$" "\n"
        fr"$p_{{\min}} = {metrics['p_min']:.6f}$" "\n"
        fr"$p_{{\max}} = {metrics['p_max']:.6f}$" "\n"
        fr"$\max|\rho(0)-|\Psi^-\rangle\langle\Psi^-|| = {metrics['r0_singlet_max_abs_error']:.3e}$" "\n\n"
        "These tests verify the reduction and positivity;\n"
        "they do not prove the microscopic input correlators."
    )
    axes[1, 1].text(0.02, 0.97, summary, va="top", fontsize=11,
                    bbox=dict(boxstyle="round", facecolor="#f4f6f8", edgecolor="#667788"))
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel(r"$r$ (lattice constant)")
    fig.suptitle("Oh--Kim X state and its exact Werner-state representation")
    fig.savefig(FIGS / "density_matrix_werner_validation.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def plot_rashba_scope(bdgs, r, spin_corr, metrics):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), constrained_layout=True)
    images = []
    for ax, vso in zip(axes[0], (0.0, 0.5)):
        bdg = bdgs[vso]
        off = 0.5 * np.abs(bdg["v_ks"][:, :, 0] ** 2 - bdg["v_ks"][:, :, 1] ** 2)
        im = ax.imshow(off, origin="lower", cmap="magma",
                       extent=[-np.pi, np.pi, -np.pi, np.pi], aspect="equal")
        images.append(im)
        ax.set_title(fr"$V_{{SO}}/t={vso}$")
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")
    fig.colorbar(images[-1], ax=axes[0, :], shrink=0.82,
                 label=r"$|n_{\mathbf{k}+}-n_{\mathbf{k}-}|/2$")

    for vso, style in [(0.0, "--"), (0.5, "-")]:
        Guu, Gud = spin_corr[vso]
        axes[1, 0].plot(r, np.abs(Gud), style,
                        label=fr"$V_{{SO}}/t={vso}$")
    axes[1, 0].set_ylabel(r"$|\mathcal{G}_{\uparrow\downarrow}(r,0^\circ)|$")
    axes[1, 0].set_xlabel(r"$r$ (lattice constant)")
    axes[1, 0].set_xlim(0, 6)
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    Guu, Gud = spin_corr[0.5]
    axes[1, 1].plot(r, np.abs(Guu), label=r"$|\mathcal{G}_{\uparrow\uparrow}|$")
    axes[1, 1].plot(r, np.abs(Gud), label=r"$|\mathcal{G}_{\uparrow\downarrow}|$")
    axes[1, 1].set_ylabel(r"Correlator magnitude, $V_{SO}/t=0.5$")
    axes[1, 1].set_xlabel(r"$r$ (lattice constant)")
    axes[1, 1].set_xlim(0, 6)
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()
    fig.suptitle("Why Rashba coupling requires a spin-basis generalization")
    fig.text(0.5, -0.015,
             "Oh--Kim Eq. (8b) assumes a spin-diagonal normal correlator.  "
             "The nonzero off-diagonal Fourier sum at $V_{SO}=0.5t$ violates that assumption.",
             ha="center", fontsize=9)
    fig.savefig(FIGS / "density_matrix_rashba_scope.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def plot_coherence_convention(bdg, r, theta_deg, f_current, f_signed,
                              p_current, p_signed):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), constrained_layout=True)
    gap = bdg["Delta_ks"][:, :, 0]
    lim = float(np.max(np.abs(gap)))
    im = axes[0, 0].imshow(gap, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                           extent=[-np.pi, np.pi, -np.pi, np.pi], aspect="equal")
    axes[0, 0].contour(bdg["kx"], bdg["ky"], gap, levels=[0], colors="black",
                       linewidths=0.8)
    axes[0, 0].set_title(r"Signed gap $\Delta_{\mathbf{k},+}/t$, $U/t=75$")
    axes[0, 0].set_xlabel(r"$k_x$")
    axes[0, 0].set_ylabel(r"$k_y$")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.8)

    i45 = int(np.argmin(np.abs(theta_deg - 45)))
    axes[0, 1].plot(r, f_current[i45] ** 2, label="unsigned $uv$ implementation")
    axes[0, 1].plot(r, f_signed[i45] ** 2, "--", label=r"signed $-\Delta^*/(2E)$")
    axes[0, 1].set_ylabel(r"$f^2(r,45^\circ)$")
    axes[0, 1].set_xlim(0, 6)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(fontsize=9)

    axes[1, 0].plot(r, p_current[i45], label="unsigned")
    axes[1, 0].plot(r, p_signed[i45], "--", label="signed")
    axes[1, 0].set_ylabel(r"$p(r,45^\circ)$")
    axes[1, 0].set_xlim(0, 6)
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    for angle in (0, 30, 45, 60, 90):
        i = int(np.argmin(np.abs(theta_deg - angle)))
        axes[1, 1].plot(r, p_signed[i] - p_current[i], label=fr"${angle}^\circ$")
    axes[1, 1].axhline(0, color="black", lw=0.8)
    axes[1, 1].set_ylabel(r"$p_{\mathrm{signed}}-p_{\mathrm{unsigned}}$")
    axes[1, 1].set_xlim(0, 6)
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(ncol=2)
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.set_xlabel(r"$r$ (lattice constant)")
    fig.suptitle("Separate issue: the anomalous-coherence convention used as input to Eq. (21)")
    fig.savefig(FIGS / "density_matrix_coherence_convention.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    with np.load(ARCHIVED) as d:
        G = d["G_t"][:, :, 0].copy()
        F = d["F_t"][:, :, 0].copy()
        r = d["r_vals"].copy()
        theta = d["theta_vals"].copy()
    theta_deg = np.degrees(theta)

    g, f = normalized_ratios(G, F)
    p = werner_parameter(g, f)
    rho = xstate(g, f)
    rho_w = werner_state(p)
    singlet = np.array([0, 1, -1, 0], complex) / np.sqrt(2)
    projector = np.outer(singlet, singlet.conj())
    eigvals = np.linalg.eigvalsh(rho)

    metrics = {
        "repository_commit_at_analysis": commit,
        "archived_source": str(ARCHIVED.relative_to(ROOT)),
        "archived_parameters": {"U_over_t": 75, "V_SO_over_t": 0,
                                "Delta_t_over_t": 4.5, "n": 1.875, "Nk": 100},
        "C4_symmetry_max_abs_error": {
            "G_0_minus_90": float(np.max(np.abs(G[0] - G[6]))),
            "G_30_minus_60": float(np.max(np.abs(G[2] - G[4]))),
            "F_0_minus_90": float(np.max(np.abs(F[0] - F[6]))),
            "F_30_minus_60": float(np.max(np.abs(F[2] - F[4]))),
        },
        "werner_equivalence_max_abs_error": float(np.max(np.abs(rho - rho_w))),
        "trace_max_abs_error": float(np.max(np.abs(np.trace(rho, axis1=-2, axis2=-1) - 1))),
        "hermiticity_max_abs_error": float(np.max(np.abs(rho - np.conj(np.swapaxes(rho, -1, -2))))),
        "minimum_eigenvalue": float(np.min(eigvals)),
        "p_min": float(np.min(p)),
        "p_max": float(np.max(p)),
        "g_max": float(np.max(g)),
        "r0_singlet_max_abs_error": float(np.max(np.abs(rho[:, 0] - projector))),
    }

    bdgs, residuals = {}, {}
    for vso in (0.0, 0.5):
        bdgs[vso], residuals[vso] = solve_at_vso(vso)
    metrics["self_consistency"] = {
        str(vso): {"converged": bool(bdgs[vso]["converged"]),
                   "max_abs_residual": residuals[vso],
                   "Delta0_over_t": float(bdgs[vso]["Delta0"]),
                   "DeltaS_over_t": float(bdgs[vso]["DeltaS"]),
                   "mu_over_t": float(bdgs[vso]["mu"])}
        for vso in (0.0, 0.5)
    }

    spin_corr = {vso: original_spin_normal_correlators(bdgs[vso], r, theta=0.0)
                 for vso in (0.0, 0.5)}
    metrics["original_spin_basis_normal_offdiagonal"] = {
        str(vso): {"max_abs_G_up_down": float(np.max(np.abs(spin_corr[vso][1]))),
                   "max_abs_k_resolved_weight": float(np.max(0.5 * np.abs(
                       bdgs[vso]["v_ks"][:, :, 0] ** 2 - bdgs[vso]["v_ks"][:, :, 1] ** 2)))}
        for vso in (0.0, 0.5)
    }

    F_signed = signed_anomalous(bdgs[0.0], r, theta)
    g_signed = np.real(G / G[:, [0]])
    f_signed = np.real(F_signed / G[:, [0]])
    p_signed = werner_parameter(g_signed, f_signed)
    rho_signed = xstate(g_signed, f_signed)
    metrics["signed_coherence_comparison"] = {
        "convention": "u v* = -Delta*/(2E), with the same archived normal G and quadrature",
        "max_abs_F_signed_minus_archived": float(np.max(np.abs(F_signed - F))),
        "max_abs_p_signed_minus_unsigned": float(np.max(np.abs(p_signed - p))),
        "minimum_signed_density_eigenvalue": float(np.min(np.linalg.eigvalsh(rho_signed))),
        "signed_trace_max_abs_error": float(np.max(np.abs(
            np.trace(rho_signed, axis1=-2, axis2=-1) - 1))),
        "interpretation": "A separate input-correlator issue; it does not invalidate the Oh--Kim matrix algebra."
    }

    plot_spatial_checks(r, theta_deg, G, F, g, f, metrics)
    plot_werner_checks(r, theta_deg, g, f, p, rho, rho_w, metrics)
    plot_rashba_scope(bdgs, r, spin_corr, metrics)
    plot_coherence_convention(bdgs[0.0], r, theta_deg, f, f_signed, p, p_signed)

    np.savez_compressed(
        DATA / "density_matrix_validity_checks.npz",
        r_vals=r, theta_vals=theta, G_archived=G, F_archived=F,
        F_signed=F_signed, g_unsigned=g, f_unsigned=f,
        g_signed=g_signed, f_signed=f_signed,
        p_unsigned=p, p_signed=p_signed,
        kx=bdgs[0.0]["kx"], ky=bdgs[0.0]["ky"],
        Delta_U75=bdgs[0.0]["Delta_ks"],
        rashba_k_offdiag_V0=(0.5 * np.abs(bdgs[0.0]["v_ks"][:, :, 0] ** 2
                                           - bdgs[0.0]["v_ks"][:, :, 1] ** 2)),
        rashba_k_offdiag_V05=(0.5 * np.abs(bdgs[0.5]["v_ks"][:, :, 0] ** 2
                                            - bdgs[0.5]["v_ks"][:, :, 1] ** 2)),
        spin_Guu_V0=spin_corr[0.0][0], spin_Gud_V0=spin_corr[0.0][1],
        spin_Guu_V05=spin_corr[0.5][0], spin_Gud_V05=spin_corr[0.5][1],
    )
    with open(REPORTS / "density_matrix_validity_checks.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
