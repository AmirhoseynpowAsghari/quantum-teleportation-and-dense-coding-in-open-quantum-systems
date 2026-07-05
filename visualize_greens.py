# =============================================================================
# visualize_greens.py
# Visualisation of Green's functions G(r,t) and F(r,t).
# ALL figures are saved to FIG_DIR/ (config.py).
#
# Functions
# ---------
#   plot_greens_vs_r        – |G|,|F| vs r at selected time snapshots
#   plot_greens_heatmap     – 2-D heatmap |G(r,t)| and |F(r,t)|
#   plot_greens_real_imag   – Re and Im parts vs r
#   plot_greens_phase       – phase angle vs r
#   plot_greens_vs_theta    – angular comparison at fixed t
#   plot_greens_all         – convenience wrapper (calls all of the above)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import figure_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save(fig, filename, show=True):
    """Save figure to FIG_DIR/filename and optionally display."""
    fp = figure_path(filename)
    fig.savefig(fp, dpi=180, bbox_inches="tight")
    print(f"[visualize_greens] Saved -> {fp}")
    if show:
        plt.show()
    plt.close(fig)


def _cbar(ax, im, label=""):
    """Attach a tight colorbar to ax."""
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.08)
    cb  = plt.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=11)
    cb.ax.tick_params(labelsize=9)
    return cb


def _theta_label(theta_vals, idx):
    return fr"$\theta = {np.degrees(theta_vals[idx]):.0f}°$"


def _minor_grid(ax):
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", ls="--", lw=0.5, alpha=0.5)
    ax.grid(which="minor", ls=":",  lw=0.3, alpha=0.3)


# ---------------------------------------------------------------------------
# 1.  |G(r)| and |F(r)| vs distance at several time snapshots
# ---------------------------------------------------------------------------

def plot_greens_vs_r(G_t, F_t, r_vals, t_grid, theta_vals,
                     theta_idx=0, t_indices=None,
                     filename=None, show=True, figsize=(13, 5)):
    """
    Plot |G_uu(r,t)| and |F_ud(r,t)| vs distance r
    at several time snapshots (side-by-side panels).

    Parameters
    ----------
    G_t, F_t    : complex arrays, shape (N_theta, N_r, N_t)
    r_vals      : 1-D float array
    t_grid      : 1-D float array
    theta_vals  : 1-D float array (radians)
    theta_idx   : int
    t_indices   : list of int or None   (default: 6 evenly spaced)
    filename    : str or None
    show        : bool
    figsize     : tuple
    """
    filename = filename or "greens_GF_vs_r.png"
    Nt = len(t_grid)
    ti = (t_indices
          or list(np.linspace(0, Nt - 1, min(6, Nt), dtype=int)))

    cmap   = plt.get_cmap("plasma", len(ti))
    colors = [cmap(i) for i in range(len(ti))]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    for ax, data, ylabel in zip(
        axes,
        [G_t, F_t],
        [r"$|G_{uu}(r,t)|$", r"$|F_{ud}(r,t)|$"],
    ):
        for col, k in zip(colors, ti):
            ax.plot(r_vals, np.abs(data[theta_idx, :, k]),
                    color=col, lw=2, alpha=0.85,
                    label=fr"$t = {t_grid[k]:.1f}$")
        ax.set_xlabel(r"Distance $r$", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(
            fr"{ylabel}  ({_theta_label(theta_vals, theta_idx)})",
            fontsize=12, pad=7,
        )
        ax.legend(fontsize=9, ncol=2, framealpha=0.7)
        _minor_grid(ax)

    fig.suptitle(
        r"Normal and Anomalous Green's Functions vs $r$",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    _save(fig, filename, show)


# ---------------------------------------------------------------------------
# 2.  2-D heatmaps  |G(r,t)|  and  |F(r,t)|
# ---------------------------------------------------------------------------

def plot_greens_heatmap(G_t, F_t, r_vals, t_grid, theta_vals,
                         theta_idx=0,
                         filename=None, show=True, figsize=(14, 5),
                         cmap_G="inferno", cmap_F="viridis",
                         contour_overlay=True, n_contours=6):
    """
    Side-by-side 2-D heatmaps of |G_uu(r,t)| and |F_ud(r,t)|.

    Parameters
    ----------
    G_t, F_t        : complex arrays, shape (N_theta, N_r, N_t)
    r_vals, t_grid  : 1-D float arrays
    theta_vals      : 1-D float array (radians)
    theta_idx       : int
    cmap_G, cmap_F  : colormap names
    contour_overlay : bool   draw iso-value contours
    n_contours      : int
    """
    filename = filename or "greens_GF_heatmap.png"

    # imshow: x = r, y = t  → need shape (N_t, N_r) with origin='lower'
    absG = np.abs(G_t[theta_idx, :, :]).T   # (N_t, N_r)
    absF = np.abs(F_t[theta_idx, :, :]).T

    extent = [r_vals[0], r_vals[-1], t_grid[0], t_grid[-1]]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, data, cmap, clabel in zip(
        axes,
        [absG, absF],
        [cmap_G, cmap_F],
        [r"$|G_{uu}(r,t)|$", r"$|F_{ud}(r,t)|$"],
    ):
        im = ax.imshow(
            data, origin="lower", aspect="auto",
            extent=extent, cmap=cmap,
            interpolation="bilinear",
        )
        if contour_overlay and data.max() > data.min():
            levels = np.linspace(data.min(), data.max(),
                                 n_contours + 2)[1:-1]
            cs = ax.contour(
                data, levels=levels, colors="white",
                linewidths=0.6, alpha=0.55,
                extent=extent, origin="lower",
            )
            ax.clabel(cs, fmt="%.3f", fontsize=8)

        _cbar(ax, im, clabel)
        ax.set_xlabel(r"Distance $r$", fontsize=13)
        ax.set_ylabel(r"Time $t$",     fontsize=13)
        ax.set_title(
            fr"{clabel}  ({_theta_label(theta_vals, theta_idx)})",
            fontsize=12, pad=7,
        )
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.suptitle(
        r"Green's Function Heatmaps $|G_{uu}(r,t)|$ and $|F_{ud}(r,t)|$",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    _save(fig, filename, show)


# ---------------------------------------------------------------------------
# 3.  Real and imaginary parts vs r
# ---------------------------------------------------------------------------

def plot_greens_real_imag(G_t, F_t, r_vals, t_grid, theta_vals,
                           theta_idx=0, t_indices=None,
                           filename=None, show=True, figsize=(14, 10)):
    """
    2×2 panel: Re(G), Re(F), Im(G), Im(F) vs r at selected times.

    Layout
    ------
    [ Re(G) | Re(F) ]
    [ Im(G) | Im(F) ]
    """
    filename = filename or "greens_real_imag.png"
    Nt = len(t_grid)
    ti = (t_indices
          or list(np.linspace(0, Nt - 1, min(5, Nt), dtype=int)))

    cmap   = plt.get_cmap("coolwarm", len(ti))
    colors = [cmap(i) for i in range(len(ti))]

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)

    cfg = [
        (axes[0, 0], G_t, np.real, r"$\mathrm{Re}[G_{uu}(r,t)]$"),
        (axes[0, 1], F_t, np.real, r"$\mathrm{Re}[F_{ud}(r,t)]$"),
        (axes[1, 0], G_t, np.imag, r"$\mathrm{Im}[G_{uu}(r,t)]$"),
        (axes[1, 1], F_t, np.imag, r"$\mathrm{Im}[F_{ud}(r,t)]$"),
    ]

    for ax, data, func, ylabel in cfg:
        for col, k in zip(colors, ti):
            ax.plot(r_vals, func(data[theta_idx, :, k]),
                    color=col, lw=2, alpha=0.85,
                    label=fr"$t = {t_grid[k]:.1f}$")
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=8, ncol=2, framealpha=0.6)
        _minor_grid(ax)

    axes[1, 0].set_xlabel(r"Distance $r$", fontsize=13)
    axes[1, 1].set_xlabel(r"Distance $r$", fontsize=13)

    fig.suptitle(
        fr"Real & Imaginary Parts of Green's Functions  "
        fr"({_theta_label(theta_vals, theta_idx)})",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    _save(fig, filename, show)


# ---------------------------------------------------------------------------
# 4.  Phase angle vs r
# ---------------------------------------------------------------------------

def plot_greens_phase(G_t, F_t, r_vals, t_grid, theta_vals,
                       theta_idx=0, t_indices=None,
                       filename=None, show=True, figsize=(13, 5)):
    """
    Plot the complex phase angle of G_uu and F_ud vs r
    at selected time snapshots.
    """
    filename = filename or "greens_phase.png"
    Nt = len(t_grid)
    ti = (t_indices
          or list(np.linspace(0, Nt - 1, min(5, Nt), dtype=int)))

    cmap   = plt.get_cmap("rainbow", len(ti))
    colors = [cmap(i) for i in range(len(ti))]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, data, tag in zip(
        axes,
        [G_t, F_t],
        [r"$\angle G_{uu}(r,t)$", r"$\angle F_{ud}(r,t)$"],
    ):
        for col, k in zip(colors, ti):
            ax.plot(r_vals, np.angle(data[theta_idx, :, k]),
                    color=col, lw=2, alpha=0.85,
                    label=fr"$t = {t_grid[k]:.1f}$")
        ax.axhline( np.pi, color="gray", lw=0.6, ls=":")
        ax.axhline(-np.pi, color="gray", lw=0.6, ls=":")
        ax.set_ylim(-np.pi - 0.3, np.pi + 0.3)
        ax.set_xlabel(r"Distance $r$", fontsize=13)
        ax.set_ylabel(tag + r"  [rad]", fontsize=13)
        ax.set_title(fr"Phase of {tag}", fontsize=12, pad=7)
        ax.legend(fontsize=9, ncol=2, framealpha=0.7)
        _minor_grid(ax)

    fig.suptitle(
        fr"Phase of Green's Functions  "
        fr"({_theta_label(theta_vals, theta_idx)})",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    _save(fig, filename, show)


# ---------------------------------------------------------------------------
# 5.  Angular dependence: |G(r)| and |F(r)| for all theta at fixed t
# ---------------------------------------------------------------------------

def plot_greens_vs_theta(G_t, F_t, r_vals, theta_vals,
                          t_idx=0,
                          filename=None, show=True, figsize=(13, 5)):
    """
    Compare |G_uu(r)| and |F_ud(r)| for every angle at a fixed time.

    Parameters
    ----------
    t_idx : int   time index to plot (default 0 = initial state)
    """
    filename = filename or "greens_vs_theta.png"
    Nth  = len(theta_vals)
    cmap = plt.get_cmap("tab10", Nth)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, data, tag in zip(
        axes,
        [G_t, F_t],
        [r"$|G_{uu}(r)|$", r"$|F_{ud}(r)|$"],
    ):
        for i in range(Nth):
            ax.plot(r_vals, np.abs(data[i, :, t_idx]),
                    color=cmap(i), lw=2, alpha=0.85,
                    label=fr"$\theta = {np.degrees(theta_vals[i]):.0f}°$")
        ax.set_xlabel(r"Distance $r$", fontsize=13)
        ax.set_ylabel(tag, fontsize=13)
        ax.set_title(
            fr"{tag}  at  $t = {0:.1f}$",
            fontsize=12, pad=7,
        )
        ax.legend(fontsize=9, framealpha=0.7, loc="upper right")
        _minor_grid(ax)

    fig.suptitle(
        r"Angular Dependence of Green's Functions",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    _save(fig, filename, show)


# ---------------------------------------------------------------------------
# 6.  Convenience wrapper
# ---------------------------------------------------------------------------

def plot_greens_all(G_t, F_t, r_vals, t_grid, theta_vals,
                    theta_idx=0, t_indices=None,
                    prefix="greens", show=True):
    """
    Generate and save the complete suite of Green's function plots.

    Parameters
    ----------
    G_t, F_t    : complex arrays, shape (N_theta, N_r, N_t)
    r_vals      : 1-D float array
    t_grid      : 1-D float array
    theta_vals  : 1-D float array (radians)
    theta_idx   : int   primary angle slice
    t_indices   : list of int or None   snapshots for line plots
    prefix      : str   filename prefix for saved figures
    show        : bool
    """
    Nt = len(t_grid)
    ti = (t_indices
          or list(np.linspace(0, Nt - 1, min(6, Nt), dtype=int)))

    print("[visualize_greens] |G|,|F| vs r ...")
    plot_greens_vs_r(
        G_t, F_t, r_vals, t_grid, theta_vals,
        theta_idx=theta_idx, t_indices=ti,
        filename=f"{prefix}_GF_vs_r.png", show=show,
    )

    print("[visualize_greens] |G|,|F| heatmaps ...")
    plot_greens_heatmap(
        G_t, F_t, r_vals, t_grid, theta_vals,
        theta_idx=theta_idx,
        filename=f"{prefix}_GF_heatmap.png", show=show,
    )

    print("[visualize_greens] Re, Im parts ...")
    plot_greens_real_imag(
        G_t, F_t, r_vals, t_grid, theta_vals,
        theta_idx=theta_idx, t_indices=ti,
        filename=f"{prefix}_GF_real_imag.png", show=show,
    )

    print("[visualize_greens] Phase ...")
    plot_greens_phase(
        G_t, F_t, r_vals, t_grid, theta_vals,
        theta_idx=theta_idx, t_indices=ti,
        filename=f"{prefix}_GF_phase.png", show=show,
    )

    print("[visualize_greens] Angular dependence ...")
    plot_greens_vs_theta(
        G_t, F_t, r_vals, theta_vals,
        t_idx=0,
        filename=f"{prefix}_GF_vs_theta.png", show=show,
    )

    print("[visualize_greens] All Green's function plots done.")


# ---------------------------------------------------------------------------
# Self-test (run: python visualize_greens.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng     = np.random.default_rng(0)
    Nr, Nt  = 80, 60
    Nth     = 7
    r_t     = np.linspace(0, 10, Nr)
    t_t     = np.linspace(0, 80, Nt)
    th_t    = np.radians([0, 15, 30, 45, 60, 75, 90])

    G_fake  = np.zeros((Nth, Nr, Nt), dtype=complex)
    F_fake  = np.zeros((Nth, Nr, Nt), dtype=complex)

    R, T = np.meshgrid(r_t, t_t, indexing="ij")
    for i in range(Nth):
        ph         = np.exp(1j*(R*np.cos(th_t[i]) + 0.3*T))
        G_fake[i]  = np.exp(-0.3*R)*np.exp(-0.05*T)*ph
        F_fake[i]  = np.exp(-0.5*R)*np.exp(-0.08*T)*ph*0.4

    plot_greens_all(
        G_fake, F_fake, r_t, t_t, th_t,
        theta_idx=0, prefix="test_greens", show=False,
    )
    print("Self-test finished. Check figures/ for test_greens_*.png")