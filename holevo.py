# =============================================================================
# holevo.py
# Holevo quantity and related entropy analysis for Cooper-pair density matrices.
#
# Reuses:
#   config.py            – physical parameters
#   kspace_setup.py      – k-grid
#   self_consistency.py  – BdG solver
#   greens_functions.py  – G(r,theta), F(r,theta)
#   density_matrix.py    – rho(theta, r)
#   evolution.py         – non-Markovian Kraus evolution
#   observables.py       – von_neumann_entropy
#   utils.py             – figure_path, data_path
#
# New in this file:
#   compute_holevo          – Holevo quantity for one density matrix
#   compute_holevo_grid     – chi(r, t) for one theta slice
#   compute_holevo_all_theta– chi(theta, r, t) for all theta
#   save_holevo_data        – save to data/
#   plot_holevo_vs_t        – chi(t) at selected r
#   plot_holevo_vs_r        – chi(r) at selected t
#   plot_holevo_heatmap     – 2-D heatmap chi(r,t)
#   plot_entropy_panel      – S(rho) and S(rho_tilde) together
#   plot_holevo_all_theta   – chi vs theta at fixed (r,t)
#   plot_holevo_all         – convenience wrapper
#   run_holevo_analysis     – full pipeline entry point
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import linalg

from utils import figure_path, data_path


# ===========================================================================
# Internal helpers
# ===========================================================================

def _save(fig, filename, show=True):
    fp = figure_path(filename)
    fig.savefig(fp, dpi=180, bbox_inches="tight")
    print(f"[holevo] Saved -> {fp}")
    if show:
        plt.show()
    plt.close(fig)


def _cbar(ax, im, label=""):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.08)
    cb  = plt.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=11)
    cb.ax.tick_params(labelsize=9)
    return cb


def _minor_grid(ax):
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", ls="--", lw=0.5, alpha=0.5)
    ax.grid(which="minor", ls=":",  lw=0.3, alpha=0.3)


# ===========================================================================
# Core: Holevo quantity
# ===========================================================================

def von_neumann_entropy_bits(rho):
    """
    S(rho) = -Tr(rho log2 rho)  [bits, base-2 logarithm].

    Parameters
    ----------
    rho : (N, N) complex array

    Returns
    -------
    float >= 0
    """
    evals = linalg.eigvalsh(rho)
    evals = evals[evals > 1e-12]
    return float(-np.sum(evals * np.log2(evals)))


def compute_holevo(rho_AB):
    """
    Holevo quantity for a two-qubit state rho_AB.

    Definition
    ----------
    chi(rho_AB) = S(rho_tilde_AB) - S(rho_AB)

    where the dephased state is:
        rho_tilde_AB = (1/4) * SUM_{i=0}^{3} (sigma_i x I) rho (sigma_i x I)^dag

    and sigma_0..sigma_3 = {I, X, Y, Z}.

    This measures the classical information extractable from
    subsystem A when subsystem B is traced out.

    Parameters
    ----------
    rho_AB : (4, 4) complex array   two-qubit density matrix

    Returns
    -------
    chi    : float   Holevo quantity  (>= 0)
    S_rho  : float   S(rho_AB)        [bits]
    S_tilde: float   S(rho_tilde_AB)  [bits]
    """
    # Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]],   dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]],  dtype=complex)

    rho_tilde = np.zeros_like(rho_AB, dtype=complex)
    for sigma in [I, X, Y, Z]:
        U          = np.kron(sigma, I)
        rho_tilde += U @ rho_AB @ U.conj().T
    rho_tilde /= 4.0

    # Enforce Hermiticity and unit trace
    rho_tilde  = 0.5 * (rho_tilde + rho_tilde.conj().T)
    tr         = rho_tilde.trace().real
    rho_tilde /= max(tr, 1e-15)

    S_rho   = von_neumann_entropy_bits(rho_AB)
    S_tilde = von_neumann_entropy_bits(rho_tilde)
    chi     = float(max(0.0, S_tilde - S_rho))   # non-negative by construction

    return chi, S_rho, S_tilde


# ===========================================================================
# Grid computations
# ===========================================================================

def compute_holevo_grid(rho_rt, r_vals, t_grid, theta_idx=0, verbose=True):
    """
    Compute chi(r, t), S_rho(r,t), S_tilde(r,t) for one theta slice.

    Parameters
    ----------
    rho_rt    : list [N_theta][N_r][N_t] of (4,4) arrays
    r_vals    : 1-D float array
    t_grid    : 1-D float array
    theta_idx : int

    Returns
    -------
    chi_rt    : float array (N_r, N_t)
    S_rho_rt  : float array (N_r, N_t)
    S_tilde_rt: float array (N_r, N_t)
    """
    Nr = len(r_vals)
    Nt = len(t_grid)

    chi_rt     = np.zeros((Nr, Nt))
    S_rho_rt   = np.zeros((Nr, Nt))
    S_tilde_rt = np.zeros((Nr, Nt))

    for j in range(Nr):
        if verbose and j % max(1, Nr // 5) == 0:
            print(f"  [holevo_grid] r-index {j+1}/{Nr}")
        for k in range(Nt):
            rho               = rho_rt[theta_idx][j][k]
            chi, S, St        = compute_holevo(rho)
            chi_rt    [j, k]  = chi
            S_rho_rt  [j, k]  = S
            S_tilde_rt[j, k]  = St

    return chi_rt, S_rho_rt, S_tilde_rt


def compute_holevo_all_theta(rho_rt, r_vals, t_grid, verbose=True):
    """
    Compute chi(theta, r, t) for ALL theta slices.

    Parameters
    ----------
    rho_rt  : list [N_theta][N_r][N_t]
    r_vals  : 1-D float array
    t_grid  : 1-D float array

    Returns
    -------
    chi_all     : float array (N_theta, N_r, N_t)
    S_rho_all   : float array (N_theta, N_r, N_t)
    S_tilde_all : float array (N_theta, N_r, N_t)
    """
    Nth = len(rho_rt)
    Nr  = len(r_vals)
    Nt  = len(t_grid)

    chi_all     = np.zeros((Nth, Nr, Nt))
    S_rho_all   = np.zeros((Nth, Nr, Nt))
    S_tilde_all = np.zeros((Nth, Nr, Nt))

    for i in range(Nth):
        if verbose:
            print(f"  [holevo_all_theta] theta {i+1}/{Nth}")
        chi_all[i], S_rho_all[i], S_tilde_all[i] = compute_holevo_grid(
            rho_rt, r_vals, t_grid,
            theta_idx=i, verbose=False,
        )

    return chi_all, S_rho_all, S_tilde_all


# ===========================================================================
# Save
# ===========================================================================

def save_holevo_data(chi_rt, S_rho_rt, S_tilde_rt,
                     r_vals, t_grid, theta_deg,
                     chi_all=None, S_rho_all=None, S_tilde_all=None,
                     theta_vals=None, prefix="holevo"):
    """
    Save Holevo arrays to data/<prefix>_holevo.npz

    Parameters
    ----------
    chi_rt, S_rho_rt, S_tilde_rt : (N_r, N_t)   primary theta slice
    chi_all, S_rho_all, S_tilde_all : (Nth,Nr,Nt) or None
    """
    out  = data_path(f"{prefix}_holevo.npz")
    save = dict(
        chi_rt     = chi_rt,
        S_rho_rt   = S_rho_rt,
        S_tilde_rt = S_tilde_rt,
        r_vals     = r_vals,
        t_grid     = t_grid,
        theta_deg  = float(theta_deg),
    )
    if chi_all is not None:
        save["chi_all"]     = chi_all
        save["S_rho_all"]   = S_rho_all
        save["S_tilde_all"] = S_tilde_all
    if theta_vals is not None:
        save["theta_vals"] = theta_vals

    np.savez_compressed(out, **save)
    print(f"[holevo] Saved -> {out}")
    return out


# ===========================================================================
# Plots
# ===========================================================================

# ---------------------------------------------------------------------------
# 1.  chi(t) at selected r values
# ---------------------------------------------------------------------------

def plot_holevo_vs_t(chi_rt, r_vals, t_grid, theta_deg,
                     r_indices=None,
                     filename="holevo_vs_t.png",
                     show=True, figsize=(9, 5)):
    """
    Line plot of Holevo quantity chi(t) at several r values.

    Parameters
    ----------
    chi_rt   : float array (N_r, N_t)
    r_indices: list of int or None
    """
    Nr = len(r_vals)
    ri = (r_indices
          or list(np.linspace(0, Nr - 1, min(6, Nr), dtype=int)))
    cmap = plt.get_cmap("plasma", len(ri))

    fig, ax = plt.subplots(figsize=figsize)
    for idx, j in enumerate(ri):
        ax.plot(t_grid, chi_rt[j, :],
                color=cmap(idx), lw=2,
                label=fr"$r = {r_vals[j]:.2f}$")

    ax.set_xlim(t_grid[0], t_grid[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Time $t$", fontsize=13)
    ax.set_ylabel(r"Holevo quantity $\chi$", fontsize=13)
    ax.set_title(
        fr"$\chi(t)$ at selected $r$  ($\theta = {theta_deg:.0f}°$)",
        fontsize=13, pad=8,
    )
    ax.legend(fontsize=9, ncol=2, framealpha=0.7)
    _minor_grid(ax)
    fig.tight_layout()
    _save(fig, filename, show)


# ---------------------------------------------------------------------------
# 2.  chi(r) at selected time snapshots
# ---------------------------------------------------------------------------

def plot_holevo_vs_r(chi_rt, r_vals, t_grid, theta_deg,
                     t_indices=None,
                     filename="holevo_vs_r.png",
                     show=True, figsize=(9, 5)):
    """
    Line plot of Holevo quantity chi(r) at several time snapshots.
    """
    Nt = len(t_grid)
    ti = (t_indices
          or list(np.linspace(0, Nt - 1, min(6, Nt), dtype=int)))
    cmap = plt.get_cmap("viridis", len(ti))

    fig, ax = plt.subplots(figsize=figsize)
    for idx, k in enumerate(ti):
        ax.plot(r_vals, chi_rt[:, k],
                color=cmap(idx), lw=2,
                label=fr"$t = {t_grid[k]:.1f}$")

    ax.set_xlim(r_vals[0], r_vals[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Distance $r$", fontsize=13)
    ax.set_ylabel(r"Holevo quantity $\chi$", fontsize=13)
    ax.set_title(
        fr"$\chi(r)$ at selected $t$  ($\theta = {theta_deg:.0f}°$)",
        fontsize=13, pad=8,
    )
    ax.legend(fontsize=9, ncol=2, framealpha=0.7)
    _minor_grid(ax)
    fig.tight_layout()
    _save(fig, filename, show)


# ---------------------------------------------------------------------------
# 3.  2-D heatmap chi(r, t)
# ---------------------------------------------------------------------------

def plot_holevo_heatmap(chi_rt, r_vals, t_grid, theta_deg,
                         filename="holevo_heatmap.png",
                         show=True, figsize=(9, 6),
                         cmap="magma"):
    """
    2-D false-colour heatmap of chi(r, t).
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        chi_rt.T,                    # (N_t, N_r) for imshow
        origin        = "lower",
        aspect        = "auto",
        extent        = [r_vals[0], r_vals[-1],
                         t_grid[0], t_grid[-1]],
        cmap          = cmap,
        interpolation = "bilinear",
    )

    # Contour overlay
    vmin, vmax = chi_rt.min(), chi_rt.max()
    if vmax > vmin:
        levels = np.linspace(vmin, vmax, 8)[1:-1]
        cs = ax.contour(
            chi_rt.T, levels=levels,
            colors="white", linewidths=0.7, alpha=0.6,
            extent=[r_vals[0], r_vals[-1],
                    t_grid[0],  t_grid[-1]],
            origin="lower",
        )
        ax.clabel(cs, fmt="%.3f", fontsize=8)

    _cbar(ax, im, r"Holevo $\chi(r,t)$")
    ax.set_xlabel(r"Distance $r$", fontsize=13)
    ax.set_ylabel(r"Time $t$",     fontsize=13)
    ax.set_title(
        fr"Holevo Quantity Heatmap  ($\theta = {theta_deg:.0f}°$)",
        fontsize=13, pad=8,
    )
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    _save(fig, filename, show)


# ---------------------------------------------------------------------------
# 4.  Entropy panel: S(rho) and S(rho_tilde) vs t at fixed r
# ---------------------------------------------------------------------------

def plot_entropy_panel(S_rho_rt, S_tilde_rt, chi_rt,
                        r_vals, t_grid, theta_deg,
                        r_indices=None,
                        filename="holevo_entropy_panel.png",
                        show=True, figsize=(14, 10)):
    """
    2x2 panel showing:
      [chi(t) at selected r  |  chi(r,t) heatmap       ]
      [S(rho) and S(tilde)   |  S_tilde - S_rho surface]
    """
    Nr = len(r_vals)
    ri = (r_indices
          or list(np.linspace(0, Nr - 1, min(5, Nr), dtype=int)))
    cmap_lines = plt.get_cmap("plasma", len(ri))
    ext        = [r_vals[0], r_vals[-1], t_grid[0], t_grid[-1]]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # --- top-left: chi(t) ---
    ax = axes[0, 0]
    for idx, j in enumerate(ri):
        ax.plot(t_grid, chi_rt[j, :],
                color=cmap_lines(idx), lw=2,
                label=fr"$r={r_vals[j]:.2f}$")
    ax.set_xlabel(r"$t$", fontsize=12)
    ax.set_ylabel(r"$\chi(t)$", fontsize=12)
    ax.set_title(r"Holevo $\chi(t)$", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8); _minor_grid(ax)

    # --- top-right: chi(r,t) heatmap ---
    ax  = axes[0, 1]
    im  = ax.imshow(chi_rt.T, origin="lower", aspect="auto",
                     extent=ext, cmap="magma",
                     interpolation="bilinear")
    _cbar(ax, im, r"$\chi$")
    ax.set_xlabel(r"$r$", fontsize=12)
    ax.set_ylabel(r"$t$",  fontsize=12)
    ax.set_title(r"Holevo Heatmap $\chi(r,t)$", fontsize=12)

    # --- bottom-left: S(rho) and S(rho_tilde) at r=0 ---
    ax = axes[1, 0]
    j0 = 0
    ax.plot(t_grid, S_rho_rt  [j0, :], "b-",  lw=2,
            label=r"$S(\rho_{AB})$")
    ax.plot(t_grid, S_tilde_rt[j0, :], "r--", lw=2,
            label=r"$S(\tilde{\rho}_{AB})$")
    ax.fill_between(t_grid,
                    S_rho_rt[j0, :], S_tilde_rt[j0, :],
                    alpha=0.15, color="purple",
                    label=r"$\chi$ region")
    ax.set_xlabel(r"$t$", fontsize=12)
    ax.set_ylabel("Entropy [bits]", fontsize=12)
    ax.set_title(fr"Entropies at $r={r_vals[j0]:.2f}$", fontsize=12)
    ax.legend(fontsize=9); _minor_grid(ax)

    # --- bottom-right: S_tilde(r,t) heatmap ---
    ax  = axes[1, 1]
    im  = ax.imshow(S_tilde_rt.T, origin="lower", aspect="auto",
                     extent=ext, cmap="cividis",
                     interpolation="bilinear")
    _cbar(ax, im, r"$S(\tilde{\rho})$ [bits]")
    ax.set_xlabel(r"$r$", fontsize=12)
    ax.set_ylabel(r"$t$",  fontsize=12)
    ax.set_title(r"$S(\tilde{\rho}_{AB})(r,t)$", fontsize=12)

    fig.suptitle(
        fr"Holevo Quantity & Entropy Analysis  "
        fr"($\theta = {theta_deg:.0f}°$)",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    _save(fig, filename, show)


# ---------------------------------------------------------------------------
# 5.  chi vs theta at fixed (r, t)
# ---------------------------------------------------------------------------

def plot_holevo_vs_theta(chi_all, theta_vals, r_vals, t_grid,
                          r_indices=None, t_index=-1,
                          filename="holevo_vs_theta.png",
                          show=True, figsize=(9, 5)):
    """
    Line plot of chi(theta) at several r values for a fixed time.

    Parameters
    ----------
    chi_all   : float array (N_theta, N_r, N_t)
    t_index   : int   which time to slice
    """
    Nr   = len(r_vals)
    th_d = np.degrees(theta_vals)
    ri   = (r_indices
            or list(np.linspace(0, Nr - 1, min(6, Nr), dtype=int)))
    cmap = plt.get_cmap("plasma", len(ri))

    fig, ax = plt.subplots(figsize=figsize)
    for idx, j in enumerate(ri):
        ax.plot(th_d, chi_all[:, j, t_index],
                color=cmap(idx), lw=2, marker="o", ms=6,
                label=fr"$r = {r_vals[j]:.2f}$")

    ax.set_xlabel(r"Angle $\theta$ (degrees)", fontsize=13)
    ax.set_ylabel(r"Holevo quantity $\chi$", fontsize=13)
    ax.set_title(
        fr"$\chi(\theta)$ at $t = {t_grid[t_index]:.1f}$",
        fontsize=13, pad=8,
    )
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, framealpha=0.7)
    _minor_grid(ax)
    fig.tight_layout()
    _save(fig, filename, show)


# ---------------------------------------------------------------------------
# 6.  chi(theta, t) heatmap at fixed r
# ---------------------------------------------------------------------------

def plot_holevo_theta_t_heatmap(chi_all, theta_vals, r_vals, t_grid,
                                 r_index=0,
                                 filename="holevo_theta_t_heatmap.png",
                                 show=True, figsize=(9, 5)):
    """
    Heatmap of chi(theta, t) at a fixed radial distance.
    """
    data = chi_all[:, r_index, :]    # (Nth, Nt)
    th_d = np.degrees(theta_vals)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data,
        origin  = "lower",
        aspect  = "auto",
        extent  = [t_grid[0], t_grid[-1], 0, len(theta_vals) - 1],
        cmap    = "magma",
        interpolation = "bilinear",
    )
    _cbar(ax, im, r"$\chi(\theta, t)$")
    ax.set_yticks(range(len(theta_vals)))
    ax.set_yticklabels([f"{d:.0f}°" for d in th_d])
    ax.set_xlabel(r"Time $t$", fontsize=12)
    ax.set_ylabel(r"Angle $\theta$", fontsize=12)
    ax.set_title(
        fr"Holevo $\chi(\theta, t)$ at $r = {r_vals[r_index]:.2f}$",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, filename, show)


# ---------------------------------------------------------------------------
# 7.  Grid of chi(r,t) – one panel per theta
# ---------------------------------------------------------------------------

def plot_holevo_heatgrid_all_theta(chi_all, theta_vals, r_vals, t_grid,
                                    filename="holevo_heatgrid_all_theta.png",
                                    show=True, figsize=None):
    """
    Grid layout: one chi(r,t) heatmap per theta value.
    """
    Nth   = len(theta_vals)
    ncols = min(4, Nth)
    nrows = int(np.ceil(Nth / ncols))
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    ext  = [r_vals[0], r_vals[-1], t_grid[0], t_grid[-1]]
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=figsize, squeeze=False)

    vmin = chi_all.min()
    vmax = chi_all.max()

    for idx in range(Nth):
        row, col = divmod(idx, ncols)
        ax  = axes[row][col]
        im  = ax.imshow(
            chi_all[idx].T,
            origin="lower", aspect="auto",
            extent=ext, cmap="magma",
            vmin=vmin, vmax=vmax,
            interpolation="bilinear",
        )
        _cbar(ax, im, r"$\chi$")
        ax.set_title(
            fr"$\theta = {np.degrees(theta_vals[idx]):.0f}°$",
            fontsize=11,
        )
        ax.set_xlabel(r"$r$", fontsize=10)
        ax.set_ylabel(r"$t$", fontsize=10)

    for idx in range(Nth, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        r"Holevo Quantity $\chi(r,t)$ for all $\theta$",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    _save(fig, filename, show)


# ---------------------------------------------------------------------------
# 8.  Convenience wrapper for one theta slice
# ---------------------------------------------------------------------------

def plot_holevo_all(chi_rt, S_rho_rt, S_tilde_rt,
                    r_vals, t_grid, theta_deg,
                    chi_all=None, theta_vals=None,
                    t_indices=None, r_indices=None,
                    prefix="holevo", show=True):
    """
    Generate and save the full suite of Holevo plots.

    Parameters
    ----------
    chi_rt, S_rho_rt, S_tilde_rt : (N_r, N_t)   primary theta slice
    chi_all    : (Nth, Nr, Nt) or None   all-theta array
    theta_vals : 1-D array (radians) or None
    t_indices  : list of int or None
    r_indices  : list of int or None
    prefix     : str   filename prefix
    show       : bool
    """
    Nt = len(t_grid)
    Nr = len(r_vals)
    ti = t_indices or list(np.linspace(0, Nt - 1, min(6, Nt), dtype=int))
    ri = r_indices or list(np.linspace(0, Nr - 1, min(6, Nr), dtype=int))

    print("[holevo] chi vs t ...")
    plot_holevo_vs_t(
        chi_rt, r_vals, t_grid, theta_deg,
        r_indices = ri,
        filename  = f"{prefix}_holevo_vs_t.png",
        show      = show,
    )

    print("[holevo] chi vs r ...")
    plot_holevo_vs_r(
        chi_rt, r_vals, t_grid, theta_deg,
        t_indices = ti,
        filename  = f"{prefix}_holevo_vs_r.png",
        show      = show,
    )

    print("[holevo] chi heatmap ...")
    plot_holevo_heatmap(
        chi_rt, r_vals, t_grid, theta_deg,
        filename = f"{prefix}_holevo_heatmap.png",
        show     = show,
    )

    print("[holevo] entropy panel ...")
    plot_entropy_panel(
        S_rho_rt, S_tilde_rt, chi_rt,
        r_vals, t_grid, theta_deg,
        r_indices = ri,
        filename  = f"{prefix}_holevo_entropy_panel.png",
        show      = show,
    )

    if chi_all is not None and theta_vals is not None:
        print("[holevo] chi vs theta ...")
        plot_holevo_vs_theta(
            chi_all, theta_vals, r_vals, t_grid,
            r_indices = ri,
            t_index   = -1,
            filename  = f"{prefix}_holevo_vs_theta.png",
            show      = show,
        )

        print("[holevo] chi(theta,t) heatmap ...")
        plot_holevo_theta_t_heatmap(
            chi_all, theta_vals, r_vals, t_grid,
            r_index  = 0,
            filename = f"{prefix}_holevo_theta_t_heatmap.png",
            show     = show,
        )

        print("[holevo] chi heatgrid (all theta) ...")
        plot_holevo_heatgrid_all_theta(
            chi_all, theta_vals, r_vals, t_grid,
            filename = f"{prefix}_holevo_heatgrid_all_theta.png",
            show     = show,
        )

    print("[holevo] All Holevo plots done.")


# ===========================================================================
# Full pipeline entry point
# ===========================================================================

def run_holevo_analysis(rho_rt, r_vals, t_grid, theta_vals,
                         theta_idx=0,
                         compute_all_theta=True,
                         prefix="holevo",
                         show=True,
                         verbose=True):
    """
    Full Holevo analysis pipeline.

    Parameters
    ----------
    rho_rt            : list [N_theta][N_r][N_t] of (4,4) arrays
                        Output of evolution.evolve_grid()
    r_vals, t_grid    : 1-D float arrays
    theta_vals        : 1-D float array (radians)
    theta_idx         : int   primary theta for single-slice plots
    compute_all_theta : bool  also compute chi for every theta
    prefix            : str   filename prefix
    show              : bool
    verbose           : bool

    Returns
    -------
    dict with keys:
        chi_rt, S_rho_rt, S_tilde_rt   – (N_r, N_t) primary slice
        chi_all, S_rho_all, S_tilde_all – (Nth,Nr,Nt) or None
    """
    theta_deg = float(np.degrees(theta_vals[theta_idx]))

    # --- primary theta slice ---
    print(f"[holevo] Computing chi(r,t) for theta={theta_deg:.0f} deg ...")
    chi_rt, S_rho_rt, S_tilde_rt = compute_holevo_grid(
        rho_rt, r_vals, t_grid,
        theta_idx = theta_idx,
        verbose   = verbose,
    )

    # --- all theta ---
    chi_all = S_rho_all = S_tilde_all = None
    if compute_all_theta:
        print("[holevo] Computing chi for all theta ...")
        chi_all, S_rho_all, S_tilde_all = compute_holevo_all_theta(
            rho_rt, r_vals, t_grid, verbose=verbose,
        )

    # --- save ---
    save_holevo_data(
        chi_rt, S_rho_rt, S_tilde_rt,
        r_vals, t_grid, theta_deg,
        chi_all     = chi_all,
        S_rho_all   = S_rho_all,
        S_tilde_all = S_tilde_all,
        theta_vals  = theta_vals,
        prefix      = prefix,
    )

    # --- plots ---
    if show:
        ti = list(np.linspace(0, len(t_grid)-1,
                               min(6, len(t_grid)), dtype=int))
        ri = list(np.linspace(0, len(r_vals)-1,
                               min(6, len(r_vals)), dtype=int))
        plot_holevo_all(
            chi_rt, S_rho_rt, S_tilde_rt,
            r_vals, t_grid, theta_deg,
            chi_all    = chi_all,
            theta_vals = theta_vals,
            t_indices  = ti,
            r_indices  = ri,
            prefix     = prefix,
            show       = show,
        )

    # --- statistics ---
    print("=" * 50)
    print("Holevo Analysis Summary")
    print("=" * 50)
    print(f"  chi(t=0)   : min={chi_rt[:,0].min():.4f} "
          f"max={chi_rt[:,0].max():.4f}")
    print(f"  chi(t=tmax): min={chi_rt[:,-1].min():.4f} "
          f"max={chi_rt[:,-1].max():.4f}")
    print(f"  chi global : min={chi_rt.min():.4f} "
          f"max={chi_rt.max():.4f}")

    return {
        "chi_rt"    : chi_rt,
        "S_rho_rt"  : S_rho_rt,
        "S_tilde_rt": S_tilde_rt,
        "chi_all"   : chi_all,
        "S_rho_all" : S_rho_all,
        "S_tilde_all": S_tilde_all,
    }


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    print("=== holevo.py self-test ===")

    # Bell state |Phi+>
    rho_bell = np.array([
        [0.5, 0, 0, 0.5],
        [0,   0, 0, 0  ],
        [0,   0, 0, 0  ],
        [0.5, 0, 0, 0.5],
    ], dtype=complex)

    chi, S, St = compute_holevo(rho_bell)
    print(f"Bell |Phi+>: chi={chi:.4f}  S(rho)={S:.4f}  S(tilde)={St:.4f}")
    # S(rho)=0 (pure), S(tilde)=1 (maximally mixed on A), chi=1

    # Maximally mixed
    rho_mix = np.eye(4, dtype=complex) / 4
    chi, S, St = compute_holevo(rho_mix)
    print(f"Mixed:       chi={chi:.4f}  S(rho)={S:.4f}  S(tilde)={St:.4f}")
    # chi should be 0 (already maximally mixed)

    print("Self-test done.")