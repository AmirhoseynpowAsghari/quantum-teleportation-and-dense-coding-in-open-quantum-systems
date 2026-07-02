# =============================================================================
# visualize_greens.py
# Visualize Green's functions G(r) and F(r) as functions of
# distance r and time t.
#
# Functions
# ---------
#   plot_greens_vs_r          – |G| and |F| vs r at fixed t snapshots
#   plot_greens_heatmap       – 2-D heatmap |G(r,t)| and |F(r,t)|
#   plot_greens_real_imag     – Re and Im parts vs r
#   plot_greens_phase         – phase angle of G and F vs r
#   plot_greens_all           – convenience wrapper for all plots
# =============================================================================

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_and_show(fig, save_path, show):
    """Optionally save and/or display a figure, then close it."""
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"[visualize_greens] Saved -> {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def _add_colorbar(ax, im, label=""):
    """Attach a neat colorbar that matches the axes height exactly."""
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="4%", pad=0.08)
    cb      = plt.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=11)
    cb.ax.tick_params(labelsize=9)
    return cb


def _theta_label(theta_vals, theta_idx):
    """Return a formatted theta label string."""
    return fr"$\theta = {np.degrees(theta_vals[theta_idx]):.0f}°$"


# ---------------------------------------------------------------------------
# 1.  |G(r)| and |F(r)| vs distance at selected time snapshots
# ---------------------------------------------------------------------------

def plot_greens_vs_r(
    G_t,
    F_t,
    r_vals,
    t_grid,
    theta_vals,
    theta_idx=0,
    t_indices=None,
    *,
    save_path=None,
    show=True,
    figsize=(12, 5),
    cmap_name="plasma",
    linewidth=2.0,
    alpha=0.85,
):
    """
    Plot |G(r,t)| and |F(r,t)| vs distance r at several time snapshots.

    Parameters
    ----------
    G_t       : complex array, shape (N_theta, N_R, N_t)
    F_t       : complex array, shape (N_theta, N_R, N_t)
    r_vals    : 1-D float array
    t_grid    : 1-D float array
    theta_vals: 1-D float array (radians)
    theta_idx : int    which angle to plot
    t_indices : list of int or None
                Time indices to plot. Default: 6 evenly spaced.
    """
    Nt = len(t_grid)
    if t_indices is None:
        t_indices = np.linspace(0, Nt - 1, min(6, Nt), dtype=int).tolist()

    cmap   = plt.get_cmap(cmap_name, len(t_indices))
    colors = [cmap(i) for i in range(len(t_indices))]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    # --- |G(r)| ---
    ax = axes[0]
    for color, k in zip(colors, t_indices):
        ax.plot(
            r_vals,
            np.abs(G_t[theta_idx, :, k]),
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=fr"$t = {t_grid[k]:.1f}$",
        )
    ax.set_xlabel(r"Distance  $r$", fontsize=13)
    ax.set_ylabel(r"$|G_{uu}(r,t)|$", fontsize=13)
    ax.set_title(
        fr"Normal Green's Function  ({_theta_label(theta_vals, theta_idx)})",
        fontsize=13,
        pad=8,
    )
    ax.legend(fontsize=9, framealpha=0.7, ncol=2)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.grid(which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)

    # --- |F(r)| ---
    ax = axes[1]
    for color, k in zip(colors, t_indices):
        ax.plot(
            r_vals,
            np.abs(F_t[theta_idx, :, k]),
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=fr"$t = {t_grid[k]:.1f}$",
        )
    ax.set_xlabel(r"Distance  $r$", fontsize=13)
    ax.set_ylabel(r"$|F_{ud}(r,t)|$", fontsize=13)
    ax.set_title(
        fr"Anomalous Green's Function  ({_theta_label(theta_vals, theta_idx)})",
        fontsize=13,
        pad=8,
    )
    ax.legend(fontsize=9, framealpha=0.7, ncol=2)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.grid(which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)

    fig.suptitle(
        r"Green's Functions  $|G_{uu}|$ and $|F_{ud}|$ vs Distance",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    _save_and_show(fig, save_path, show)


# ---------------------------------------------------------------------------
# 2.  Heatmaps: |G(r,t)| and |F(r,t)|
# ---------------------------------------------------------------------------

def plot_greens_heatmap(
    G_t,
    F_t,
    r_vals,
    t_grid,
    theta_vals,
    theta_idx=0,
    *,
    save_path=None,
    show=True,
    figsize=(14, 5),
    cmap_G="inferno",
    cmap_F="viridis",
    contour_overlay=True,
    n_contours=6,
):
    """
    2-D heatmaps of |G(r,t)| and |F(r,t)|.

    Parameters
    ----------
    G_t            : complex array, shape (N_theta, N_R, N_t)
    F_t            : complex array, shape (N_theta, N_R, N_t)
    r_vals         : 1-D float array
    t_grid         : 1-D float array
    theta_vals     : 1-D float array (radians)
    theta_idx      : int
    cmap_G         : str   colormap for |G|
    cmap_F         : str   colormap for |F|
    contour_overlay: bool  draw iso-value contours on top
    n_contours     : int
    """
    absG = np.abs(G_t[theta_idx, :, :]).T   # (N_t, N_R)
    absF = np.abs(F_t[theta_idx, :, :]).T   # (N_t, N_R)

    extent = [r_vals[0], r_vals[-1], t_grid[0], t_grid[-1]]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, data, cmap, ylabel_tag, title_tag in zip(
        axes,
        [absG, absF],
        [cmap_G, cmap_F],
        [r"$|G_{uu}(r,t)|$", r"$|F_{ud}(r,t)|$"],
        ["Normal", "Anomalous"],
    ):
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            interpolation="bilinear",
        )

        if contour_overlay and data.max() > data.min():
            levels = np.linspace(data.min(), data.max(), n_contours + 2)[1:-1]
            ax.contour(
                data,
                levels=levels,
                colors="white",
                linewidths=0.6,
                alpha=0.5,
                extent=extent,
                origin="lower",
            )

        _add_colorbar(ax, im, label=ylabel_tag)
        ax.set_xlabel(r"Distance  $r$", fontsize=13)
        ax.set_ylabel(r"Time  $t$",     fontsize=13)
        ax.set_title(
            fr"{title_tag} Green's Function  "
            fr"({_theta_label(theta_vals, theta_idx)})",
            fontsize=12,
            pad=8,
        )
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.suptitle(
        r"Green's Functions Heatmap  $|G_{uu}(r,t)|$  and  $|F_{ud}(r,t)|$",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    _save_and_show(fig, save_path, show)


# ---------------------------------------------------------------------------
# 3.  Real and imaginary parts vs r
# ---------------------------------------------------------------------------

def plot_greens_real_imag(
    G_t,
    F_t,
    r_vals,
    t_grid,
    theta_vals,
    theta_idx=0,
    t_indices=None,
    *,
    save_path=None,
    show=True,
    figsize=(14, 10),
    cmap_name="coolwarm",
    linewidth=2.0,
    alpha=0.85,
):
    """
    Plot Re and Im parts of G and F vs r at selected time snapshots.

    Layout: 2 rows x 2 columns
        [Re(G) | Re(F)]
        [Im(G) | Im(F)]
    """
    Nt = len(t_grid)
    if t_indices is None:
        t_indices = np.linspace(0, Nt - 1, min(5, Nt), dtype=int).tolist()

    cmap   = plt.get_cmap(cmap_name, len(t_indices))
    colors = [cmap(i) for i in range(len(t_indices))]

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)

    plot_cfg = [
        (axes[0, 0], G_t, np.real, r"$\mathrm{Re}[G_{uu}(r,t)]$"),
        (axes[0, 1], F_t, np.real, r"$\mathrm{Re}[F_{ud}(r,t)]$"),
        (axes[1, 0], G_t, np.imag, r"$\mathrm{Im}[G_{uu}(r,t)]$"),
        (axes[1, 1], F_t, np.imag, r"$\mathrm{Im}[F_{ud}(r,t)]$"),
    ]

    for ax, data, func, ylabel in plot_cfg:
        for color, k in zip(colors, t_indices):
            ax.plot(
                r_vals,
                func(data[theta_idx, :, k]),
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=fr"$t={t_grid[k]:.1f}$",
            )
        ax.set_ylabel(ylabel, fontsize=12)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.legend(fontsize=8, framealpha=0.6, ncol=2)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)

    axes[1, 0].set_xlabel(r"Distance  $r$", fontsize=13)
    axes[1, 1].set_xlabel(r"Distance  $r$", fontsize=13)

    fig.suptitle(
        fr"Real and Imaginary Parts of Green's Functions  "
        fr"({_theta_label(theta_vals, theta_idx)})",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    _save_and_show(fig, save_path, show)


# ---------------------------------------------------------------------------
# 4.  Phase angle of G and F vs r
# ---------------------------------------------------------------------------

def plot_greens_phase(
    G_t,
    F_t,
    r_vals,
    t_grid,
    theta_vals,
    theta_idx=0,
    t_indices=None,
    *,
    save_path=None,
    show=True,
    figsize=(12, 5),
    cmap_name="rainbow",
    linewidth=2.0,
    alpha=0.85,
):
    """
    Plot the phase angle (in radians) of G and F vs r at selected times.
    """
    Nt = len(t_grid)
    if t_indices is None:
        t_indices = np.linspace(0, Nt - 1, min(5, Nt), dtype=int).tolist()

    cmap   = plt.get_cmap(cmap_name, len(t_indices))
    colors = [cmap(i) for i in range(len(t_indices))]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, data, tag in zip(
        axes,
        [G_t, F_t],
        [r"$\angle G_{uu}(r,t)$", r"$\angle F_{ud}(r,t)$"],
    ):
        for color, k in zip(colors, t_indices):
            ax.plot(
                r_vals,
                np.angle(data[theta_idx, :, k]),
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=fr"$t = {t_grid[k]:.1f}$",
            )
        ax.set_xlabel(r"Distance  $r$", fontsize=13)
        ax.set_ylabel(tag + r"  [rad]", fontsize=13)
        ax.set_title(fr"Phase of  {tag}", fontsize=12, pad=8)
        ax.axhline( np.pi, color="gray", linewidth=0.6, linestyle=":")
        ax.axhline(-np.pi, color="gray", linewidth=0.6, linestyle=":")
        ax.set_ylim(-np.pi - 0.3, np.pi + 0.3)
        ax.legend(fontsize=9, framealpha=0.7, ncol=2)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.suptitle(
        fr"Phase of Green's Functions  "
        fr"({_theta_label(theta_vals, theta_idx)})",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    _save_and_show(fig, save_path, show)


# ---------------------------------------------------------------------------
# 5.  Multi-angle comparison: |G(r, t=0)| for all theta
# ---------------------------------------------------------------------------

def plot_greens_vs_theta(
    G_t,
    F_t,
    r_vals,
    theta_vals,
    t_idx=0,
    *,
    save_path=None,
    show=True,
    figsize=(12, 5),
    cmap_name="tab10",
    linewidth=2.0,
    alpha=0.85,
):
    """
    Compare |G(r)| and |F(r)| for all angles at a fixed time.

    Parameters
    ----------
    t_idx : int   which time step to plot (default: 0 = initial state)
    """
    N_theta = len(theta_vals)
    cmap    = plt.get_cmap(cmap_name, N_theta)
    colors  = [cmap(i) for i in range(N_theta)]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, data, tag in zip(
        axes,
        [G_t, F_t],
        [r"$|G_{uu}(r)|$", r"$|F_{ud}(r)|$"],
    ):
        for color, i, theta in zip(colors, range(N_theta), theta_vals):
            ax.plot(
                r_vals,
                np.abs(data[i, :, t_idx]),
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=fr"$\theta = {np.degrees(theta):.0f}°$",
            )
        ax.set_xlabel(r"Distance  $r$", fontsize=13)
        ax.set_ylabel(tag, fontsize=13)
        ax.set_title(fr"{tag}  at  $t = {0:.1f}$", fontsize=12, pad=8)
        ax.legend(fontsize=9, framealpha=0.7, loc="upper right")
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.suptitle(
        r"Angular Dependence of Green's Functions",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    _save_and_show(fig, save_path, show)


# ---------------------------------------------------------------------------
# 6.  Convenience wrapper
# ---------------------------------------------------------------------------

def plot_greens_all(
    G_t,
    F_t,
    r_vals,
    t_grid,
    theta_vals,
    theta_idx=0,
    *,
    t_indices=None,
    prefix="greens",
    show=True,
):
    """
    Generate and save all Green's function plots.

    Parameters
    ----------
    G_t       : complex array, shape (N_theta, N_R, N_t)
    F_t       : complex array, shape (N_theta, N_R, N_t)
    r_vals    : 1-D float array
    t_grid    : 1-D float array
    theta_vals: 1-D float array (radians)
    theta_idx : int    which angle to use as the primary slice
    t_indices : list of int or None   snapshots for line plots
    prefix    : str    filename prefix for all saved figures
    show      : bool
    """
    print("[visualize_greens] Plotting |G|, |F| vs r ...")
    plot_greens_vs_r(
        G_t, F_t, r_vals, t_grid, theta_vals,
        theta_idx=theta_idx,
        t_indices=t_indices,
        save_path=f"{prefix}_GF_vs_r.png",
        show=show,
    )

    print("[visualize_greens] Plotting |G|, |F| heatmaps ...")
    plot_greens_heatmap(
        G_t, F_t, r_vals, t_grid, theta_vals,
        theta_idx=theta_idx,
        save_path=f"{prefix}_GF_heatmap.png",
        show=show,
    )

    print("[visualize_greens] Plotting Re, Im parts ...")
    plot_greens_real_imag(
        G_t, F_t, r_vals, t_grid, theta_vals,
        theta_idx=theta_idx,
        t_indices=t_indices,
        save_path=f"{prefix}_GF_real_imag.png",
        show=show,
    )

    print("[visualize_greens] Plotting phase ...")
    plot_greens_phase(
        G_t, F_t, r_vals, t_grid, theta_vals,
        theta_idx=theta_idx,
        t_indices=t_indices,
        save_path=f"{prefix}_GF_phase.png",
        show=show,
    )

    print("[visualize_greens] Plotting angular dependence ...")
    plot_greens_vs_theta(
        G_t, F_t, r_vals, theta_vals,
        t_idx=0,
        save_path=f"{prefix}_GF_vs_theta.png",
        show=show,
    )

    print("[visualize_greens] All Green's function plots done.")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    Nr      = 80
    Nt      = 60
    N_theta = 7

    r_t     = np.linspace(0, 10, Nr)
    t_t     = np.linspace(0, 80, Nt)
    theta_t = np.radians([0, 15, 30, 45, 60, 75, 90])

    # Synthetic decaying Green's functions
    R, T   = np.meshgrid(r_t, t_t, indexing="ij")
    G_fake = np.zeros((N_theta, Nr, Nt), dtype=complex)
    F_fake = np.zeros((N_theta, Nr, Nt), dtype=complex)

    for i in range(N_theta):
        phase           = np.exp(1j * (R * np.cos(theta_t[i]) + 0.5 * T))
        G_fake[i]       = np.exp(-0.3 * R) * np.exp(-0.05 * T) * phase
        F_fake[i]       = np.exp(-0.5 * R) * np.exp(-0.08 * T) * phase * 0.5

    plot_greens_all(
        G_fake, F_fake,
        r_vals=r_t,
        t_grid=t_t,
        theta_vals=theta_t,
        theta_idx=0,
        prefix="test_greens",
        show=True,
    )