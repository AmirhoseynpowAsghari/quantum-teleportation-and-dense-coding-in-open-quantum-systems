# =============================================================================
# visualization.py
# Plotting utilities for the Hubbard Cooper Pair evolution results.
#
# Provides
# --------
#   plot_concurrence_vs_r        – C(r) curves at selected time snapshots
#   plot_concurrence_heatmap     – 2-D heatmap C(r, t)
#   plot_concurrence_vs_t        – C(t) curves at selected r values
#   plot_multi_observable        – 2×2 panel for all four observables
#   run_all_plots                – convenience wrapper that calls all of the above
# =============================================================================

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Use a non-interactive backend when no display is available
# (comment out the next line if you are running in a Jupyter notebook)
# matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_and_show(fig, save_path, show):
    """Optionally save and/or display a figure, then close it."""
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"[visualization] Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def _add_colorbar(ax, im, label=""):
    """Attach a neat colorbar that matches the axes height exactly."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cb  = plt.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=11)
    cb.ax.tick_params(labelsize=9)
    return cb


# ---------------------------------------------------------------------------
# 1.  C(r) – line plot at selected time snapshots
# ---------------------------------------------------------------------------

def plot_concurrence_vs_r(
    obs,
    r_vals,
    t_grid,
    theta_deg,
    t_indices=None,
    *,
    save_path=None,
    show=True,
    figsize=(8, 5),
    cmap_name="plasma",
    linewidth=2.0,
    alpha=0.85,
):
    """
    Plot concurrence C(r) at several time snapshots on the same axes.

    Parameters
    ----------
    obs       : dict   output of observables.compute_observables_grid()
                       must contain key 'C' with shape (N_r, N_t)
    r_vals    : 1-D array   radial grid
    t_grid    : 1-D array   time grid
    theta_deg : float       angle in degrees (used in title only)
    t_indices : list of int or None
                Which time indices to plot.
                Default: 5 evenly spaced snapshots including t=0 and t=t_max.
    save_path : str or None
    show      : bool
    figsize   : tuple
    cmap_name : str   matplotlib colormap for distinguishing curves
    linewidth : float
    alpha     : float   line transparency
    """
    C = obs["C"]                          # (N_r, N_t)
    Nr, Nt = C.shape

    # Default: 5 evenly spaced snapshots
    if t_indices is None:
        t_indices = np.linspace(0, Nt - 1, min(5, Nt), dtype=int).tolist()

    # Colour palette – one colour per snapshot
    cmap   = plt.get_cmap(cmap_name, len(t_indices))
    colors = [cmap(i) for i in range(len(t_indices))]

    fig, ax = plt.subplots(figsize=figsize)

    for color, k in zip(colors, t_indices):
        t_val = t_grid[k]
        ax.plot(
            r_vals,
            C[:, k],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=fr"$t = {t_val:.1f}$",
        )

    # Formatting
    ax.set_xlim(r_vals[0], r_vals[-1])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"Distance  $r$", fontsize=13)
    ax.set_ylabel(r"Concurrence  $\mathcal{C}(r,\,t)$", fontsize=13)
    ax.set_title(
        fr"Concurrence vs Distance  ($\theta = {theta_deg:.0f}°$)",
        fontsize=14,
        pad=10,
    )
    ax.legend(
        fontsize=10,
        framealpha=0.7,
        loc="upper right",
        ncol=2,
    )
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.grid(which="minor", linestyle=":",  linewidth=0.3, alpha=0.4)
    fig.tight_layout()

    _save_and_show(fig, save_path, show)


# ---------------------------------------------------------------------------
# 2.  C(r, t) heatmap
# ---------------------------------------------------------------------------

def plot_concurrence_heatmap(
    obs,
    r_vals,
    t_grid,
    theta_deg,
    *,
    save_path=None,
    show=True,
    figsize=(9, 6),
    cmap_name="inferno",
    n_levels=256,
    contour_overlay=True,
    n_contours=8,
):
    """
    2-D false-colour heatmap of concurrence C(r, t).

    Parameters
    ----------
    obs              : dict   must contain key 'C' with shape (N_r, N_t)
    r_vals           : 1-D array
    t_grid           : 1-D array
    theta_deg        : float  angle in degrees (title)
    save_path        : str or None
    show             : bool
    figsize          : tuple
    cmap_name        : str    matplotlib colormap
    n_levels         : int    colour resolution
    contour_overlay  : bool   draw iso-concurrence contour lines on top
    n_contours       : int    number of contour levels
    """
    C = obs["C"]          # (N_r, N_t)

    # axes: x = r,  y = t
    # imshow needs shape (N_t, N_r) with origin='lower'
    C_plot = C.T          # (N_t, N_r)

    vmin, vmax = C_plot.min(), C_plot.max()

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        C_plot,
        origin="lower",
        aspect="auto",
        extent=[r_vals[0], r_vals[-1], t_grid[0], t_grid[-1]],
        cmap=cmap_name,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )

    # Optional iso-concurrence contours
    if contour_overlay and vmax > vmin:
        levels = np.linspace(vmin, vmax, n_contours + 2)[1:-1]
        ax.contour(
            C_plot,
            levels=levels,
            colors="white",
            linewidths=0.6,
            alpha=0.55,
            extent=[r_vals[0], r_vals[-1], t_grid[0], t_grid[-1]],
            origin="lower",
        )

    _add_colorbar(ax, im, label=r"Concurrence  $\mathcal{C}(r,\,t)$")

    ax.set_xlabel(r"Distance  $r$", fontsize=13)
    ax.set_ylabel(r"Time  $t$",     fontsize=13)
    ax.set_title(
        fr"Concurrence Heatmap  ($\theta = {theta_deg:.0f}°$)",
        fontsize=14,
        pad=10,
    )
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()

    _save_and_show(fig, save_path, show)


# ---------------------------------------------------------------------------
# 3.  C(t) – line plot at selected r values
# ---------------------------------------------------------------------------

def plot_concurrence_vs_t(
    obs,
    t_grid,
    r_vals,
    theta_deg,
    r_indices=None,
    *,
    save_path=None,
    show=True,
    figsize=(8, 5),
    cmap_name="viridis",
    linewidth=2.0,
    alpha=0.85,
):
    """
    Plot concurrence C(t) at several spatial points on the same axes.

    Parameters
    ----------
    obs       : dict   must contain key 'C' with shape (N_r, N_t)
    t_grid    : 1-D array
    r_vals    : 1-D array
    theta_deg : float
    r_indices : list of int or None
                Which r indices to plot.
                Default: 5 evenly spaced points.
    """
    C = obs["C"]          # (N_r, N_t)
    Nr, Nt = C.shape

    if r_indices is None:
        r_indices = np.linspace(0, Nr - 1, min(5, Nr), dtype=int).tolist()

    cmap   = plt.get_cmap(cmap_name, len(r_indices))
    colors = [cmap(i) for i in range(len(r_indices))]

    fig, ax = plt.subplots(figsize=figsize)

    for color, j in zip(colors, r_indices):
        r_val = r_vals[j]
        ax.plot(
            t_grid,
            C[j, :],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=fr"$r = {r_val:.2f}$",
        )

    ax.set_xlim(t_grid[0], t_grid[-1])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"Time  $t$", fontsize=13)
    ax.set_ylabel(r"Concurrence  $\mathcal{C}(r,\,t)$", fontsize=13)
    ax.set_title(
        fr"Concurrence vs Time  ($\theta = {theta_deg:.0f}°$)",
        fontsize=14,
        pad=10,
    )
    ax.legend(fontsize=10, framealpha=0.7, loc="upper right", ncol=2)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.grid(which="minor", linestyle=":",  linewidth=0.3, alpha=0.4)
    fig.tight_layout()

    _save_and_show(fig, save_path, show)


# ---------------------------------------------------------------------------
# 4.  2×2 multi-observable heatmap panel
# ---------------------------------------------------------------------------

def plot_multi_observable(
    obs,
    r_vals,
    t_grid,
    theta_deg,
    *,
    save_path=None,
    show=True,
    figsize=(13, 9),
    cmaps=None,
):
    """
    2×2 panel of heatmaps: concurrence, singlet fraction, purity, VN entropy.

    Parameters
    ----------
    obs     : dict  keys 'C', 'Fs', 'pur', 'S'  each (N_r, N_t)
    cmaps   : list of 4 colormap names or None (uses defaults)
    """
    keys   = ["C",                    "Fs",
              "pur",                  "S"]
    titles = [r"Concurrence $\mathcal{C}$",
              r"Singlet Fraction $F_s$",
              r"Purity $\mathrm{Tr}[\rho^2]$",
              r"VN Entropy $S(\rho)$"]
    if cmaps is None:
        cmaps = ["inferno", "cividis", "plasma", "magma"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    extent = [r_vals[0], r_vals[-1], t_grid[0], t_grid[-1]]

    for ax, key, title, cmap in zip(axes.flat, keys, titles, cmaps):
        data = obs[key].T          # (N_t, N_r) for imshow
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            interpolation="bilinear",
        )
        _add_colorbar(ax, im, label=title)
        ax.set_xlabel(r"$r$", fontsize=12)
        ax.set_ylabel(r"$t$", fontsize=12)
        ax.set_title(title, fontsize=12, pad=6)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.suptitle(
        fr"All Observables  ($\theta = {theta_deg:.0f}°$)",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()

    _save_and_show(fig, save_path, show)


# ---------------------------------------------------------------------------
# 5.  Convenience wrapper – call every plot in one go
# ---------------------------------------------------------------------------

def run_all_plots(
    obs,
    r_vals,
    t_grid,
    theta_deg,
    *,
    t_indices=None,
    r_indices=None,
    prefix="hubbard",
    show=True,
):
    """
    Generate and save all four standard plots.

    Parameters
    ----------
    obs       : dict   output of observables.compute_observables_grid()
    r_vals    : 1-D array
    t_grid    : 1-D array
    theta_deg : float
    t_indices : list of int or None   passed to plot_concurrence_vs_r
    r_indices : list of int or None   passed to plot_concurrence_vs_t
    prefix    : str   filename prefix for saved figures
    show      : bool  display windows interactively
    """
    print("[visualization] Plotting C(r) line plot …")
    plot_concurrence_vs_r(
        obs, r_vals, t_grid, theta_deg,
        t_indices=t_indices,
        save_path=f"{prefix}_C_vs_r.png",
        show=show,
    )

    print("[visualization] Plotting C(r,t) heatmap …")
    plot_concurrence_heatmap(
        obs, r_vals, t_grid, theta_deg,
        save_path=f"{prefix}_C_heatmap.png",
        show=show,
    )

    print("[visualization] Plotting C(t) line plot …")
    plot_concurrence_vs_t(
        obs, t_grid, r_vals, theta_deg,
        r_indices=r_indices,
        save_path=f"{prefix}_C_vs_t.png",
        show=show,
    )

    print("[visualization] Plotting multi-observable panel …")
    plot_multi_observable(
        obs, r_vals, t_grid, theta_deg,
        save_path=f"{prefix}_multi_obs.png",
        show=show,
    )

    print("[visualization] All plots done.")


# ---------------------------------------------------------------------------
# Self-test (run: python visualization.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(42)

    # Synthetic data that looks vaguely physical
    Nr, Nt = 80, 60
    r_test  = np.linspace(0, 10, Nr)
    t_test  = np.linspace(0, 80, Nt)

    # C decays in time and space – simple Gaussian envelope
    R, T     = np.meshgrid(r_test, t_test, indexing="ij")
    C_fake   = np.exp(-0.15 * R) * np.exp(-0.04 * T)
    C_fake  += 0.05 * rng.standard_normal((Nr, Nt))
    C_fake   = np.clip(C_fake, 0.0, 1.0)

    obs_fake = {
        "C":   C_fake,
        "Fs":  C_fake * 0.5,
        "pur": 0.25 + 0.75 * C_fake,
        "S":   np.log(4) * (1.0 - C_fake),
    }

    run_all_plots(
        obs_fake,
        r_vals=r_test,
        t_grid=t_test,
        theta_deg=0,
        prefix="test",
        show=True,          # set False for headless testing
    )