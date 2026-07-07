# =============================================================================
# visualize_theta.py
# Theta-dependence visualisations for concurrence and fidelity.
#
# All figures are saved to FIG_DIR/ (config.py).
#
# Functions
# ---------
#   compute_obs_all_theta        – C, Fs, purity, S for every (theta, r, t)
#   compute_fidelity_all_theta   – F(r,t) for every theta slice
#
#   plot_C_vs_theta_fixed_r_t    – C(theta) at fixed r and t
#   plot_C_vs_theta_fixed_r      – C(theta, t) heatmap at fixed r
#   plot_C_vs_theta_fixed_t      – C(theta, r) heatmap at fixed t
#   plot_C_polar                 – polar plot C(theta) at selected r, t
#   plot_C_all_theta_heatgrid    – grid of heatmaps, one per theta
#
#   plot_F_vs_theta_fixed_r_t    – F(theta) at fixed r and t
#   plot_F_vs_theta_fixed_r      – F(theta, t) heatmap at fixed r
#   plot_F_vs_theta_fixed_t      – F(theta, r) heatmap at fixed t
#   plot_F_polar                 – polar plot F(theta) at selected r, t
#   plot_F_all_theta_heatgrid    – grid of heatmaps, one per theta
#
#   plot_C_F_vs_theta            – side-by-side C and F vs theta
#   plot_theta_summary           – full summary panel
#
#   run_theta_analysis           – convenience wrapper (all plots)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import figure_path
from observables import (concurrence_xstate, singlet_fraction,
                          purity, von_neumann_entropy)
from fidelity import compute_fidelity_grid


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save(fig, filename, show=True):
    fp = figure_path(filename)
    fig.savefig(fp, dpi=180, bbox_inches="tight")
    print(f"[visualize_theta] Saved -> {fp}")
    if show:
        plt.show()
    plt.close(fig)


def _cbar(ax, im, label=""):
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.08)
    cb  = plt.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=9)
    return cb


def _minor_grid(ax):
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", ls="--", lw=0.5, alpha=0.5)
    ax.grid(which="minor", ls=":",  lw=0.3, alpha=0.3)


def _theta_ticks(ax, theta_vals_deg, axis="x"):
    labels = [f"{d:.0f}°" for d in theta_vals_deg]
    if axis == "x":
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
    else:
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------

def compute_obs_all_theta(rho_rt, r_vals, t_grid):
    """
    Compute C, Fs, purity, entropy for EVERY theta slice.

    Parameters
    ----------
    rho_rt  : list [N_theta][N_r][N_t] of (4,4) arrays
    r_vals  : 1-D float array
    t_grid  : 1-D float array

    Returns
    -------
    dict with keys 'C', 'Fs', 'pur', 'S'
        each a float array of shape (N_theta, N_r, N_t)
    """
    Nth = len(rho_rt)
    Nr  = len(r_vals)
    Nt  = len(t_grid)

    C   = np.zeros((Nth, Nr, Nt))
    Fs  = np.zeros((Nth, Nr, Nt))
    pur = np.zeros((Nth, Nr, Nt))
    S   = np.zeros((Nth, Nr, Nt))

    for i in range(Nth):
        print(f"  [obs_all_theta] theta {i+1}/{Nth}")
        for j in range(Nr):
            for k in range(Nt):
                rho         = rho_rt[i][j][k]
                C  [i,j,k]  = concurrence_xstate(rho)
                Fs [i,j,k]  = singlet_fraction(rho)
                pur[i,j,k]  = purity(rho)
                S  [i,j,k]  = von_neumann_entropy(rho)

    return {"C": C, "Fs": Fs, "pur": pur, "S": S}


def compute_fidelity_all_theta(rho_matrices, r_vals, t_grid,
                                reference_type="initial",
                                verbose=True, **fid_noise):
    """
    Compute fidelity F(r,t) for every theta slice.

    Parameters
    ----------
    rho_matrices   : list [N_theta][N_r] of (4,4) arrays
    r_vals, t_grid : 1-D arrays
    reference_type : 'initial' or 'pure'
    **fid_noise    : noise kwargs forwarded to compute_fidelity_grid

    Returns
    -------
    F_all : float array, shape (N_theta, N_r, N_t)
    """
    Nth   = len(rho_matrices)
    Nr    = len(r_vals)
    Nt    = len(t_grid)
    F_all = np.zeros((Nth, Nr, Nt))

    for i in range(Nth):
        if verbose:
            print(f"  [fidelity_all_theta] theta {i+1}/{Nth}")
        F_rt, _ = compute_fidelity_grid(
            rho_matrices, t_grid,
            theta_idx      = i,
            reference_type = reference_type,
            verbose        = False,
            **fid_noise,
        )
        F_all[i] = F_rt

    return F_all


# ===========================================================================
# CONCURRENCE vs THETA
# ===========================================================================

# ---------------------------------------------------------------------------
# 1.  C(theta) line plot at fixed r and t
# ---------------------------------------------------------------------------

def plot_C_vs_theta_fixed_r_t(obs_all, theta_vals, r_vals, t_grid,
                                r_indices=None, t_indices=None,
                                filename="C_vs_theta_fixed_rt.png",
                                show=True, figsize=(9, 5)):
    """
    Line plot of C(theta) at several (r, t) combinations.

    Parameters
    ----------
    obs_all    : dict  output of compute_obs_all_theta()
    theta_vals : 1-D float array (radians)
    r_indices  : list of int or None
    t_indices  : list of int or None
    """
    C    = obs_all["C"]          # (Nth, Nr, Nt)
    Nth  = len(theta_vals)
    Nr   = len(r_vals)
    Nt   = len(t_grid)
    th_d = np.degrees(theta_vals)

    ri = r_indices or [0, Nr//4, Nr//2, 3*Nr//4]
    ti = t_indices or [0, Nt//4, Nt//2, Nt-1]

    cmap_r = plt.get_cmap("plasma",  len(ri))
    cmap_t = plt.get_cmap("viridis", len(ti))

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # left: fix t, vary r
    ax = axes[0]
    t_fix = t_grid[ti[len(ti)//2]]
    k_fix = ti[len(ti)//2]
    for idx, j in enumerate(ri):
        ax.plot(th_d, C[:, j, k_fix],
                color=cmap_r(idx), lw=2, marker="o", ms=5,
                label=fr"$r={r_vals[j]:.2f}$")
    ax.set_xlabel(r"Angle $\theta$ (degrees)", fontsize=12)
    ax.set_ylabel(r"Concurrence $\mathcal{C}$", fontsize=12)
    ax.set_title(fr"$\mathcal{{C}}(\theta)$ at $t={t_fix:.1f}$",
                 fontsize=12)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9); _minor_grid(ax)

    # right: fix r, vary t
    ax = axes[1]
    j_fix = ri[len(ri)//2]
    for idx, k in enumerate(ti):
        ax.plot(th_d, C[:, j_fix, k],
                color=cmap_t(idx), lw=2, marker="s", ms=5,
                label=fr"$t={t_grid[k]:.1f}$")
    ax.set_xlabel(r"Angle $\theta$ (degrees)", fontsize=12)
    ax.set_title(
        fr"$\mathcal{{C}}(\theta)$ at $r={r_vals[j_fix]:.2f}$",
        fontsize=12,
    )
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9); _minor_grid(ax)

    fig.suptitle(r"Concurrence vs Angle $\theta$", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, filename, show)


# ---------------------------------------------------------------------------
# 2.  C(theta, t) heatmap at fixed r
# ---------------------------------------------------------------------------

def plot_C_vs_theta_fixed_r(obs_all, theta_vals, r_vals, t_grid,
                              r_index=0,
                              filename="C_vs_theta_t_heatmap.png",
                              show=True, figsize=(9, 5)):
    """
    Heatmap  C(theta, t)  at a fixed radial distance r_index.
    """
    C    = obs_all["C"]           # (Nth, Nr, Nt)
    data = C[:, r_index, :]       # (Nth, Nt)
    th_d = np.degrees(theta_vals)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data,
        origin  = "lower",
        aspect  = "auto",
        extent  = [t_grid[0], t_grid[-1], 0, len(theta_vals)-1],
        cmap    = "inferno",
        vmin    = 0.0, vmax = 1.0,
        interpolation = "bilinear",
    )
    _cbar(ax, im, r"$\mathcal{C}(\theta, t)$")

    # y-axis: theta ticks
    ax.set_yticks(range(len(theta_vals)))
    ax.set_yticklabels([f"{d:.0f}°" for d in th_d])

    ax.set_xlabel(r"Time $t$", fontsize=12)
    ax.set_ylabel(r"Angle $\theta$", fontsize=12)
    ax.set_title(
        fr"Concurrence $\mathcal{{C}}(\theta, t)$ "
        fr"at $r = {r_vals[r_index]:.2f}$",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, filename, show)


# ---------------------------------------------------------------------------
# 3.  C(theta, r) heatmap at fixed t
# ---------------------------------------------------------------------------

def plot_C_vs_theta_fixed_t(obs_all, theta_vals, r_vals, t_grid,
                              t_index=-1,
                              filename="C_vs_theta_r_heatmap.png",
                              show=True, figsize=(9, 5)):
    """
    Heatmap  C(theta, r)  at a fixed time t_index.
    """
    C    = obs_all["C"]
    data = C[:, :, t_index]      # (Nth, Nr)
    th_d = np.degrees(theta_vals)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data,
        origin  = "lower",
        aspect  = "auto",
        extent  = [r_vals[0], r_vals[-1], 0, len(theta_vals)-1],
        cmap    = "inferno",
        vmin    = 0.0, vmax = 1.0,
        interpolation = "bilinear",
    )
    _cbar(ax, im, r"$\mathcal{C}(\theta, r)$")

    ax.set_yticks(range(len(theta_vals)))
    ax.set_yticklabels([f"{d:.0f}°" for d in th_d])

    ax.set_xlabel(r"Distance $r$", fontsize=12)
    ax.set_ylabel(r"Angle $\theta$", fontsize=12)
    ax.set_title(
        fr"Concurrence $\mathcal{{C}}(\theta, r)$ "
        fr"at $t = {t_grid[t_index]:.1f}$",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, filename, show)


# ---------------------------------------------------------------------------
# 4.  Polar plot  C(theta) at selected (r, t)
# ---------------------------------------------------------------------------

def plot_C_polar(obs_all, theta_vals, r_vals, t_grid,
                  r_indices=None, t_index=0,
                  filename="C_polar.png",
                  show=True, figsize=(7, 7)):
    """
    Polar plot of C vs theta at a fixed time for several r values.
    """
    C    = obs_all["C"]
    Nr   = len(r_vals)
    ri   = r_indices or list(np.linspace(0, Nr-1, min(5,Nr), dtype=int))
    cmap = plt.get_cmap("plasma", len(ri))

    # Close the polar curve by appending the first point
    theta_closed = np.append(theta_vals, theta_vals[0])

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=figsize)

    for idx, j in enumerate(ri):
        vals = np.append(C[:, j, t_index], C[0, j, t_index])
        ax.plot(theta_closed, vals,
                color=cmap(idx), lw=2,
                label=fr"$r={r_vals[j]:.2f}$")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_title(
        fr"Polar: $\mathcal{{C}}(\theta)$ at $t={t_grid[t_index]:.1f}$",
        fontsize=13, pad=15,
    )
    ax.legend(fontsize=9, loc="lower right",
              bbox_to_anchor=(1.3, -0.1))
    fig.tight_layout()
    _save(fig, filename, show)


# ---------------------------------------------------------------------------
# 5.  Grid of C(r,t) heatmaps – one panel per theta
# ---------------------------------------------------------------------------

def plot_C_all_theta_heatgrid(obs_all, theta_vals, r_vals, t_grid,
                                filename="C_heatgrid_all_theta.png",
                                show=True, figsize=None):
    """
    Grid layout: each panel shows the C(r,t) heatmap for one theta.
    """
    C   = obs_all["C"]      # (Nth, Nr, Nt)
    Nth = len(theta_vals)
    ncols = min(4, Nth)
    nrows = int(np.ceil(Nth / ncols))

    if figsize is None:
        figsize = (5*ncols, 4*nrows)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=figsize, squeeze=False)
    ext = [r_vals[0], r_vals[-1], t_grid[0], t_grid[-1]]

    for idx in range(Nth):
        row, col = divmod(idx, ncols)
        ax  = axes[row][col]
        im  = ax.imshow(
            C[idx].T,
            origin="lower", aspect="auto",
            extent=ext, cmap="inferno",
            vmin=0.0, vmax=1.0,
            interpolation="bilinear",
        )
        _cbar(ax, im, r"$\mathcal{C}$")
        ax.set_title(
            fr"$\theta = {np.degrees(theta_vals[idx]):.0f}°$",
            fontsize=11,
        )
        ax.set_xlabel(r"$r$", fontsize=10)
        ax.set_ylabel(r"$t$", fontsize=10)

    # Hide unused panels
    for idx in range(Nth, nrows*ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        r"Concurrence $\mathcal{C}(r,t)$ for all angles $\theta$",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    _save(fig, filename, show)


# ===========================================================================
# FIDELITY vs THETA
# ===========================================================================

# ---------------------------------------------------------------------------
# 6.  F(theta) line plot at fixed r and t
# ---------------------------------------------------------------------------

def plot_F_vs_theta_fixed_r_t(F_all, theta_vals, r_vals, t_grid,
                                reference_label="initial",
                                r_indices=None, t_indices=None,
                                filename=None,
                                show=True, figsize=(9, 5)):
    """
    Line plot of F(theta) at several (r, t) combinations.

    Parameters
    ----------
    F_all          : float array (N_theta, N_r, N_t)
    reference_label: str  for titles/filenames
    """
    filename = filename or f"F_vs_theta_fixed_rt_{reference_label}.png"
    Nth  = len(theta_vals)
    Nr   = len(r_vals)
    Nt   = len(t_grid)
    th_d = np.degrees(theta_vals)

    ri = r_indices or [0, Nr//4, Nr//2, 3*Nr//4]
    ti = t_indices or [0, Nt//4, Nt//2, Nt-1]

    cmap_r = plt.get_cmap("plasma",  len(ri))
    cmap_t = plt.get_cmap("viridis", len(ti))

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # left: fix t at midpoint, vary r
    k_fix = ti[len(ti)//2]
    ax    = axes[0]
    for idx, j in enumerate(ri):
        ax.plot(th_d, F_all[:, j, k_fix],
                color=cmap_r(idx), lw=2, marker="o", ms=5,
                label=fr"$r={r_vals[j]:.2f}$")
    ax.set_xlabel(r"Angle $\theta$ (degrees)", fontsize=12)
    ax.set_ylabel(r"Fidelity $\mathcal{F}$", fontsize=12)
    ax.set_title(
        fr"$\mathcal{{F}}(\theta)$ at $t={t_grid[k_fix]:.1f}$  "
        fr"(ref: {reference_label})",
        fontsize=11,
    )
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9); _minor_grid(ax)

    # right: fix r at midpoint, vary t
    j_fix = ri[len(ri)//2]
    ax    = axes[1]
    for idx, k in enumerate(ti):
        ax.plot(th_d, F_all[:, j_fix, k],
                color=cmap_t(idx), lw=2, marker="s", ms=5,
                label=fr"$t={t_grid[k]:.1f}$")
    ax.set_xlabel(r"Angle $\theta$ (degrees)", fontsize=12)
    ax.set_title(
        fr"$\mathcal{{F}}(\theta)$ at $r={r_vals[j_fix]:.2f}$  "
        fr"(ref: {reference_label})",
        fontsize=11,
    )
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9); _minor_grid(ax)

    fig.suptitle(
        fr"Fidelity vs Angle $\theta$  (ref: {reference_label})",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    _save(fig, filename, show)


# ---------------------------------------------------------------------------
# 7.  F(theta, t) heatmap at fixed r
# ---------------------------------------------------------------------------

def plot_F_vs_theta_fixed_r(F_all, theta_vals, r_vals, t_grid,
                              reference_label="initial",
                              r_index=0,
                              filename=None,
                              show=True, figsize=(9, 5)):
    """
    Heatmap  F(theta, t)  at a fixed radial distance.
    """
    filename = filename or f"F_vs_theta_t_heatmap_{reference_label}.png"
    data = F_all[:, r_index, :]      # (Nth, Nt)
    th_d = np.degrees(theta_vals)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data,
        origin="lower", aspect="auto",
        extent=[t_grid[0], t_grid[-1], 0, len(theta_vals)-1],
        cmap="RdYlBu_r",
        vmin=0.0, vmax=1.0,
        interpolation="bilinear",
    )
    _cbar(ax, im, r"Fidelity $\mathcal{F}$")

    ax.set_yticks(range(len(theta_vals)))
    ax.set_yticklabels([f"{d:.0f}°" for d in th_d])
    ax.set_xlabel(r"Time $t$", fontsize=12)
    ax.set_ylabel(r"Angle $\theta$", fontsize=12)
    ax.set_title(
        fr"Fidelity $\mathcal{{F}}(\theta,t)$ "
        fr"at $r={r_vals[r_index]:.2f}$  (ref: {reference_label})",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, filename, show)


# ---------------------------------------------------------------------------
# 8.  F(theta, r) heatmap at fixed t
# ---------------------------------------------------------------------------

def plot_F_vs_theta_fixed_t(F_all, theta_vals, r_vals, t_grid,
                              reference_label="initial",
                              t_index=-1,
                              filename=None,
                              show=True, figsize=(9, 5)):
    """
    Heatmap  F(theta, r)  at a fixed time.
    """
    filename = filename or f"F_vs_theta_r_heatmap_{reference_label}.png"
    data = F_all[:, :, t_index]      # (Nth, Nr)
    th_d = np.degrees(theta_vals)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data,
        origin="lower", aspect="auto",
        extent=[r_vals[0], r_vals[-1], 0, len(theta_vals)-1],
        cmap="RdYlBu_r",
        vmin=0.0, vmax=1.0,
        interpolation="bilinear",
    )
    _cbar(ax, im, r"Fidelity $\mathcal{F}$")

    ax.set_yticks(range(len(theta_vals)))
    ax.set_yticklabels([f"{d:.0f}°" for d in th_d])
    ax.set_xlabel(r"Distance $r$", fontsize=12)
    ax.set_ylabel(r"Angle $\theta$", fontsize=12)
    ax.set_title(
        fr"Fidelity $\mathcal{{F}}(\theta,r)$ "
        fr"at $t={t_grid[t_index]:.1f}$  (ref: {reference_label})",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, filename, show)


# ---------------------------------------------------------------------------
# 9.  Polar plot  F(theta) at selected (r, t)
# ---------------------------------------------------------------------------

def plot_F_polar(F_all, theta_vals, r_vals, t_grid,
                  reference_label="initial",
                  r_indices=None, t_index=0,
                  filename=None,
                  show=True, figsize=(7, 7)):
    """
    Polar plot of F(theta) at a fixed time for several r values.
    """
    filename = filename or f"F_polar_{reference_label}.png"
    Nr   = len(r_vals)
    ri   = r_indices or list(np.linspace(0, Nr-1, min(5,Nr), dtype=int))
    cmap = plt.get_cmap("plasma", len(ri))

    theta_closed = np.append(theta_vals, theta_vals[0])

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"},
                            figsize=figsize)
    for idx, j in enumerate(ri):
        vals = np.append(F_all[:, j, t_index], F_all[0, j, t_index])
        ax.plot(theta_closed, vals,
                color=cmap(idx), lw=2,
                label=fr"$r={r_vals[j]:.2f}$")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_title(
        fr"Polar: $\mathcal{{F}}(\theta)$ at $t={t_grid[t_index]:.1f}$  "
        fr"(ref: {reference_label})",
        fontsize=12, pad=15,
    )
    ax.legend(fontsize=9, loc="lower right",
              bbox_to_anchor=(1.3, -0.1))
    fig.tight_layout()
    _save(fig, filename, show)


# ---------------------------------------------------------------------------
# 10.  Grid of F(r,t) heatmaps – one panel per theta
# ---------------------------------------------------------------------------

def plot_F_all_theta_heatgrid(F_all, theta_vals, r_vals, t_grid,
                                reference_label="initial",
                                filename=None,
                                show=True, figsize=None):
    """
    Grid layout: each panel shows the F(r,t) heatmap for one theta.
    """
    filename = filename or f"F_heatgrid_all_theta_{reference_label}.png"
    Nth   = len(theta_vals)
    ncols = min(4, Nth)
    nrows = int(np.ceil(Nth / ncols))

    if figsize is None:
        figsize = (5*ncols, 4*nrows)

    ext = [r_vals[0], r_vals[-1], t_grid[0], t_grid[-1]]
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=figsize, squeeze=False)

    for idx in range(Nth):
        row, col = divmod(idx, ncols)
        ax  = axes[row][col]
        im  = ax.imshow(
            F_all[idx],
            origin="lower", aspect="auto",
            extent=ext, cmap="RdYlBu_r",
            vmin=0.0, vmax=1.0,
            interpolation="bilinear",
        )
        _cbar(ax, im, r"$\mathcal{F}$")
        ax.set_title(
            fr"$\theta = {np.degrees(theta_vals[idx]):.0f}°$",
            fontsize=11,
        )
        ax.set_xlabel(r"$r$", fontsize=10)
        ax.set_ylabel(r"$t$", fontsize=10)

    for idx in range(Nth, nrows*ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        fr"Fidelity $\mathcal{{F}}(r,t)$ for all $\theta$  "
        fr"(ref: {reference_label})",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    _save(fig, filename, show)


# ===========================================================================
# COMBINED: C and F side-by-side vs theta
# ===========================================================================

def plot_C_F_vs_theta(obs_all, F_all_init, F_all_pure,
                       theta_vals, r_vals, t_grid,
                       r_index=0, t_index=-1,
                       filename="CF_vs_theta_combined.png",
                       show=True, figsize=(14, 5)):
    """
    Side-by-side bar/line comparison of C and F vs theta
    at a fixed (r, t).

    Parameters
    ----------
    obs_all     : dict  output of compute_obs_all_theta()
    F_all_init  : (Nth, Nr, Nt)  fidelity vs initial
    F_all_pure  : (Nth, Nr, Nt)  fidelity vs singlet
    r_index     : int   which r to slice
    t_index     : int   which t to slice
    """
    C    = obs_all["C"][:, r_index, t_index]   # (Nth,)
    Fi   = F_all_init[:, r_index, t_index]
    Fp   = F_all_pure[:, r_index, t_index]
    th_d = np.degrees(theta_vals)
    x    = np.arange(len(theta_vals))
    w    = 0.25

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- bar chart ---
    ax = axes[0]
    ax.bar(x - w, C,  width=w, label=r"$\mathcal{C}$",        color="steelblue",  alpha=0.85)
    ax.bar(x,     Fi, width=w, label=r"$\mathcal{F}$ (init)", color="darkorange", alpha=0.85)
    ax.bar(x + w, Fp, width=w, label=r"$\mathcal{F}$ (sing)", color="seagreen",   alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.0f}°" for d in th_d])
    ax.set_xlabel(r"Angle $\theta$", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title(
        fr"$\mathcal{{C}}$ and $\mathcal{{F}}$ vs $\theta$  "
        fr"($r={r_vals[r_index]:.2f}$, $t={t_grid[t_index]:.1f}$)",
        fontsize=11,
    )
    ax.legend(fontsize=9); ax.grid(axis="y", ls="--", alpha=0.5)

    # --- line chart ---
    ax = axes[1]
    ax.plot(th_d, C,  "o-", lw=2, ms=7, color="steelblue",
            label=r"$\mathcal{C}$")
    ax.plot(th_d, Fi, "s-", lw=2, ms=7, color="darkorange",
            label=r"$\mathcal{F}$ vs initial")
    ax.plot(th_d, Fp, "^-", lw=2, ms=7, color="seagreen",
            label=r"$\mathcal{F}$ vs singlet")
    ax.set_xlabel(r"Angle $\theta$ (degrees)", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        fr"Line: $\mathcal{{C}}$ and $\mathcal{{F}}$ vs $\theta$",
        fontsize=11,
    )
    ax.legend(fontsize=9); _minor_grid(ax)

    fig.suptitle(
        fr"Angular dependence of $\mathcal{{C}}$ and $\mathcal{{F}}$",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    _save(fig, filename, show)


# ---------------------------------------------------------------------------
# 11.  Full summary panel  (4 rows x N_theta columns)
# ---------------------------------------------------------------------------

def plot_theta_summary(obs_all, F_all_init, F_all_pure,
                        theta_vals, r_vals, t_grid,
                        filename="theta_summary.png",
                        show=True, figsize=None):
    """
    Large summary grid:
      Row 0: C(r,t) heatmap for each theta
      Row 1: F_init(r,t) heatmap for each theta
      Row 2: F_pure(r,t) heatmap for each theta
      Row 3: C, F_init, F_pure vs r at t=0 for each theta
    """
    Nth   = len(theta_vals)
    if figsize is None:
        figsize = (4*Nth, 16)

    ext = [r_vals[0], r_vals[-1], t_grid[0], t_grid[-1]]
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(4, Nth, figure=fig,
                   hspace=0.45, wspace=0.35)

    row_titles = [
        r"$\mathcal{C}(r,t)$",
        r"$\mathcal{F}_\mathrm{init}(r,t)$",
        r"$\mathcal{F}_\mathrm{sing}(r,t)$",
        r"vs $r$ at $t=0$",
    ]
    cmaps = ["inferno", "RdYlBu_r", "RdYlBu_r", None]

    for col, (ith, theta) in enumerate(zip(range(Nth), theta_vals)):
        th_str = fr"$\theta={np.degrees(theta):.0f}°$"

        # row 0: C heatmap
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(obs_all["C"][ith].T, origin="lower",
                        aspect="auto", extent=ext,
                        cmap="inferno", vmin=0, vmax=1,
                        interpolation="bilinear")
        _cbar(ax, im, r"$\mathcal{C}$")
        ax.set_title(th_str, fontsize=11)
        if col == 0: ax.set_ylabel(row_titles[0], fontsize=10)
        ax.set_xlabel(r"$r$", fontsize=9)

        # row 1: F_init heatmap
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(F_all_init[ith], origin="lower",
                        aspect="auto", extent=ext,
                        cmap="RdYlBu_r", vmin=0, vmax=1,
                        interpolation="bilinear")
        _cbar(ax, im, r"$\mathcal{F}$")
        if col == 0: ax.set_ylabel(row_titles[1], fontsize=10)
        ax.set_xlabel(r"$r$", fontsize=9)

        # row 2: F_pure heatmap
        ax = fig.add_subplot(gs[2, col])
        im = ax.imshow(F_all_pure[ith], origin="lower",
                        aspect="auto", extent=ext,
                        cmap="RdYlBu_r", vmin=0, vmax=1,
                        interpolation="bilinear")
        _cbar(ax, im, r"$\mathcal{F}$")
        if col == 0: ax.set_ylabel(row_titles[2], fontsize=10)
        ax.set_xlabel(r"$r$", fontsize=9)

        # row 3: line plot vs r at t=0
        ax = fig.add_subplot(gs[3, col])
        ax.plot(r_vals, obs_all["C"][ith, :, 0],
                "b-", lw=2, label=r"$\mathcal{C}$")
        ax.plot(r_vals, F_all_init[ith, :, 0],
                "r--", lw=2, label=r"$\mathcal{F}_\mathrm{i}$")
        ax.plot(r_vals, F_all_pure[ith, :, 0],
                "g:", lw=2, label=r"$\mathcal{F}_\mathrm{s}$")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel(r"$r$", fontsize=9)
        if col == 0:
            ax.set_ylabel(row_titles[3], fontsize=10)
            ax.legend(fontsize=8)
        _minor_grid(ax)

    fig.suptitle(
        r"Full $\theta$-dependence summary: "
        r"$\mathcal{C}(r,t)$ and $\mathcal{F}(r,t)$",
        fontsize=15, y=1.01,
    )
    _save(fig, filename, show)


# ===========================================================================
# Convenience wrapper
# ===========================================================================

def run_theta_analysis(rho_rt, rho_matrices, r_vals, t_grid,
                        theta_vals, prefix="theta",
                        show=True, fid_noise=None):
    """
    Run the full theta-dependence analysis.

    Parameters
    ----------
    rho_rt       : list [Nth][Nr][Nt]   evolved density matrices
    rho_matrices : list [Nth][Nr]       initial density matrices
    r_vals       : 1-D float array
    t_grid       : 1-D float array
    theta_vals   : 1-D float array (radians)
    prefix       : str   filename prefix
    show         : bool
    fid_noise    : dict or None   noise kwargs for fidelity
                   (keys: kind, bath, nuA, GammaA, nuB, GammaB,
                          gammaA, GammaAD_A, gammaB, GammaAD_B)
    """
    if fid_noise is None:
        import config as cfg
        fid_noise = dict(
            kind      = cfg.NOISE_KIND,
            bath      = cfg.NOISE_BATH,
            nuA       = cfg.FID_NU_A,
            GammaA    = cfg.FID_GAMMA_A,
            nuB       = cfg.FID_NU_B,
            GammaB    = cfg.FID_GAMMA_B,
            gammaA    = cfg.FID_GAMMA_AD_A,
            GammaAD_A = cfg.FID_BIGAMMA_AD_A,
            gammaB    = cfg.FID_GAMMA_AD_B,
            GammaAD_B = cfg.FID_BIGAMMA_AD_B,
        )

    Nr  = len(r_vals)
    Nt  = len(t_grid)
    Nth = len(theta_vals)

    # snapshot indices
    ri = list(np.linspace(0, Nr-1,  min(4, Nr),  dtype=int))
    ti = list(np.linspace(0, Nt-1,  min(4, Nt),  dtype=int))

    # --- compute observables for ALL theta --------------------------------
    print("[theta] Computing observables for all theta ...")
    obs_all = compute_obs_all_theta(rho_rt, r_vals, t_grid)

    # --- compute fidelity for ALL theta -----------------------------------
    print("[theta] Computing fidelity (vs initial) for all theta ...")
    F_all_init = compute_fidelity_all_theta(
        rho_matrices, r_vals, t_grid,
        reference_type="initial", verbose=True, **fid_noise,
    )

    print("[theta] Computing fidelity (vs singlet) for all theta ...")
    F_all_pure = compute_fidelity_all_theta(
        rho_matrices, r_vals, t_grid,
        reference_type="pure", verbose=True, **fid_noise,
    )

    # ---- CONCURRENCE PLOTS -----------------------------------------------
    print("[theta] C vs theta (fixed r,t) ...")
    plot_C_vs_theta_fixed_r_t(
        obs_all, theta_vals, r_vals, t_grid,
        r_indices=ri, t_indices=ti,
        filename=f"{prefix}_C_vs_theta_fixed_rt.png", show=show,
    )

    print("[theta] C(theta,t) heatmap ...")
    plot_C_vs_theta_fixed_r(
        obs_all, theta_vals, r_vals, t_grid,
        r_index=0,
        filename=f"{prefix}_C_vs_theta_t_heatmap.png", show=show,
    )

    print("[theta] C(theta,r) heatmap ...")
    plot_C_vs_theta_fixed_t(
        obs_all, theta_vals, r_vals, t_grid,
        t_index=-1,
        filename=f"{prefix}_C_vs_theta_r_heatmap.png", show=show,
    )

    print("[theta] C polar plot ...")
    plot_C_polar(
        obs_all, theta_vals, r_vals, t_grid,
        r_indices=ri, t_index=0,
        filename=f"{prefix}_C_polar.png", show=show,
    )

    print("[theta] C heatgrid (all theta) ...")
    plot_C_all_theta_heatgrid(
        obs_all, theta_vals, r_vals, t_grid,
        filename=f"{prefix}_C_heatgrid_all_theta.png", show=show,
    )

    # ---- FIDELITY PLOTS --------------------------------------------------
    for F_all, ref_lbl in [(F_all_init, "initial"), (F_all_pure, "singlet")]:

        print(f"[theta] F({ref_lbl}) vs theta (fixed r,t) ...")
        plot_F_vs_theta_fixed_r_t(
            F_all, theta_vals, r_vals, t_grid,
            reference_label=ref_lbl, r_indices=ri, t_indices=ti,
            filename=f"{prefix}_F_vs_theta_fixed_rt_{ref_lbl}.png",
            show=show,
        )

        print(f"[theta] F({ref_lbl})(theta,t) heatmap ...")
        plot_F_vs_theta_fixed_r(
            F_all, theta_vals, r_vals, t_grid,
            reference_label=ref_lbl, r_index=0,
            filename=f"{prefix}_F_vs_theta_t_heatmap_{ref_lbl}.png",
            show=show,
        )

        print(f"[theta] F({ref_lbl})(theta,r) heatmap ...")
        plot_F_vs_theta_fixed_t(
            F_all, theta_vals, r_vals, t_grid,
            reference_label=ref_lbl, t_index=-1,
            filename=f"{prefix}_F_vs_theta_r_heatmap_{ref_lbl}.png",
            show=show,
        )

        print(f"[theta] F({ref_lbl}) polar plot ...")
        plot_F_polar(
            F_all, theta_vals, r_vals, t_grid,
            reference_label=ref_lbl, r_indices=ri, t_index=0,
            filename=f"{prefix}_F_polar_{ref_lbl}.png",
            show=show,
        )

        print(f"[theta] F({ref_lbl}) heatgrid ...")
        plot_F_all_theta_heatgrid(
            F_all, theta_vals, r_vals, t_grid,
            reference_label=ref_lbl,
            filename=f"{prefix}_F_heatgrid_all_theta_{ref_lbl}.png",
            show=show,
        )

    # ---- COMBINED PLOTS --------------------------------------------------
    print("[theta] C & F combined vs theta ...")
    plot_C_F_vs_theta(
        obs_all, F_all_init, F_all_pure,
        theta_vals, r_vals, t_grid,
        r_index=0, t_index=-1,
        filename=f"{prefix}_CF_vs_theta_combined.png",
        show=show,
    )

    print("[theta] Full summary panel ...")
    plot_theta_summary(
        obs_all, F_all_init, F_all_pure,
        theta_vals, r_vals, t_grid,
        filename=f"{prefix}_theta_summary.png",
        show=show,
    )

    print("[theta] All theta-dependence plots done.")

    return obs_all, F_all_init, F_all_pure