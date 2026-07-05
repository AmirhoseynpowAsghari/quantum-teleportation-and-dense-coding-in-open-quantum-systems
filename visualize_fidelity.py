# =============================================================================
# visualize_fidelity.py
# ALL figures saved to FIG_DIR/
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import figure_path
from fidelity import analyze_fidelity_decay


def _save(fig, filename, show=True):
    fp = figure_path(filename)
    fig.savefig(fp, dpi=180, bbox_inches="tight")
    print(f"[visualize_fidelity] Saved -> {fp}")
    if show:
        plt.show()
    plt.close(fig)


def _cbar(ax, im, label=""):
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.08)
    cb  = plt.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=11)
    return cb


def plot_fidelity_heatmap(r_vals, t_grid, F_rt, theta_deg,
                           reference_label="initial",
                           filename=None, show=True, figsize=(10,6)):
    filename = filename or f"fidelity_heatmap_{reference_label}.png"
    fig, ax  = plt.subplots(figsize=figsize)
    im = ax.imshow(
        F_rt, aspect="auto", origin="lower", cmap="RdYlBu_r",
        extent=[t_grid[0], t_grid[-1], r_vals[0], r_vals[-1]],
        vmin=0.0, vmax=1.0, interpolation="bilinear",
    )
    levels = np.array([0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
    levels = levels[levels <= float(np.nanmax(F_rt))]
    if levels.size:
        cs = ax.contour(t_grid, r_vals, F_rt, levels=levels,
                         colors="white", linewidths=0.9)
        ax.clabel(cs, inline=True, fontsize=9, fmt="F=%.2f")
    _cbar(ax, im, r"Fidelity $\mathcal{F}$")
    ax.set_xlabel(r"$t$", fontsize=13)
    ax.set_ylabel(r"$r$", fontsize=13)
    ax.set_title(
        fr"Fidelity  (ref: {reference_label}, $\theta={theta_deg:.0f}°$)",
        fontsize=13,
    )
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    _save(fig, filename, show)


def plot_fidelity_vs_t(r_vals, t_grid, F_rt, theta_deg,
                        reference_label="initial",
                        r_indices=None, filename=None,
                        show=True, figsize=(8,5)):
    filename = filename or f"fidelity_vs_t_{reference_label}.png"
    Nr = len(r_vals)
    ri = r_indices or list(np.linspace(0, Nr-1, min(6,Nr), dtype=int))
    cm = plt.get_cmap("viridis", len(ri))

    fig, ax = plt.subplots(figsize=figsize)
    for idx, j in enumerate(ri):
        ax.plot(t_grid, F_rt[j,:], color=cm(idx), lw=2,
                label=fr"$r={r_vals[j]:.2f}$")
    for th in [0.9, 0.8, 0.7, 0.5]:
        ax.axhline(th, ls="--", lw=0.7, color="red", alpha=0.35)
    ax.set_xlim(t_grid[0], t_grid[-1]); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$t$", fontsize=13)
    ax.set_ylabel(r"Fidelity $\mathcal{F}$", fontsize=13)
    ax.set_title(
        fr"$\mathcal{{F}}(t)$  (ref: {reference_label}, $\theta={theta_deg:.0f}°$)"
    )
    ax.legend(fontsize=9, ncol=2)
    ax.grid(ls="--", alpha=0.4)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    _save(fig, filename, show)


def plot_fidelity_vs_r(r_vals, t_grid, F_rt, theta_deg,
                        reference_label="initial",
                        t_indices=None, filename=None,
                        show=True, figsize=(8,5)):
    filename = filename or f"fidelity_vs_r_{reference_label}.png"
    Nt = len(t_grid)
    ti = t_indices or list(np.linspace(0, Nt-1, min(6,Nt), dtype=int))
    cm = plt.get_cmap("plasma", len(ti))

    fig, ax = plt.subplots(figsize=figsize)
    for idx, k in enumerate(ti):
        ax.plot(r_vals, F_rt[:,k], color=cm(idx), lw=2,
                label=fr"$t={t_grid[k]:.1f}$")
    for th in [0.9, 0.8, 0.7, 0.5]:
        ax.axhline(th, ls="--", lw=0.7, color="gray", alpha=0.45)
    ax.set_xlim(r_vals[0], r_vals[-1]); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$r$", fontsize=13)
    ax.set_ylabel(r"Fidelity $\mathcal{F}$", fontsize=13)
    ax.set_title(
        fr"$\mathcal{{F}}(r)$  (ref: {reference_label}, $\theta={theta_deg:.0f}°$)"
    )
    ax.legend(fontsize=9, ncol=2)
    ax.grid(ls="--", alpha=0.4)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    _save(fig, filename, show)


def plot_fidelity_comparison(r_vals, t_grid, F_init, F_pure,
                              theta_deg, filename=None,
                              show=True, figsize=(14,10)):
    filename = filename or "fidelity_comparison.png"
    ext = [t_grid[0], t_grid[-1], r_vals[0], r_vals[-1]]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for ax, Fdata, lbl in zip(
        [axes[0,0], axes[0,1]],
        [F_init, F_pure],
        ["vs Initial", "vs Singlet"],
    ):
        im = ax.imshow(Fdata, aspect="auto", origin="lower", cmap="RdYlBu_r",
                        extent=ext, vmin=0, vmax=1, interpolation="bilinear")
        _cbar(ax, im, r"$\mathcal{F}$")
        ax.set_title(f"Fidelity {lbl}", fontsize=12)
        ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$r$")

    axes[1,0].plot(t_grid, F_init[0,:], "b-", lw=2, label="vs initial")
    axes[1,0].plot(t_grid, F_pure[0,:], "r-", lw=2, label="vs singlet")
    for th in [0.9,0.8,0.7,0.5]:
        axes[1,0].axhline(th, ls="--", lw=0.7, color="gray", alpha=0.5)
    axes[1,0].set_xlim(t_grid[0], t_grid[-1])
    axes[1,0].set_ylim(-0.02, 1.05)
    axes[1,0].set_xlabel(r"$t$"); axes[1,0].set_ylabel(r"$\mathcal{F}$")
    axes[1,0].set_title(r"$\mathcal{F}(t)$ at $r=0$", fontsize=12)
    axes[1,0].legend(); axes[1,0].grid(ls="--", alpha=0.4)

    axes[1,1].plot(r_vals, F_init[:,-1], "b-", lw=2, label="vs initial")
    axes[1,1].plot(r_vals, F_pure[:,-1], "r-", lw=2, label="vs singlet")
    for th in [0.9,0.8,0.7,0.5]:
        axes[1,1].axhline(th, ls="--", lw=0.7, color="gray", alpha=0.5)
    axes[1,1].set_xlim(r_vals[0], r_vals[-1])
    axes[1,1].set_ylim(-0.02, 1.05)
    axes[1,1].set_xlabel(r"$r$"); axes[1,1].set_ylabel(r"$\mathcal{F}$")
    axes[1,1].set_title(
        fr"$\mathcal{{F}}(r)$ at $t={t_grid[-1]:.1f}$", fontsize=12
    )
    axes[1,1].legend(); axes[1,1].grid(ls="--", alpha=0.4)

    fig.suptitle(
        fr"Fidelity Comparison  ($\theta={theta_deg:.0f}°$)",
        fontsize=15, y=1.01,
    )
    fig.tight_layout()
    _save(fig, filename, show)


def plot_fidelity_decay_times(r_vals, t_grid, F_rt, theta_deg,
                               reference_label="initial",
                               thresholds=(0.9, 0.8, 0.7, 0.5),
                               filename=None, show=True, figsize=(8,5)):
    filename = filename or f"fidelity_decay_times_{reference_label}.png"
    analysis = analyze_fidelity_decay(r_vals, t_grid, F_rt, thresholds)
    cm = plt.get_cmap("coolwarm", len(thresholds))

    fig, ax = plt.subplots(figsize=figsize)
    for idx, th in enumerate(thresholds):
        ax.plot(r_vals, analysis["decay_times"][th],
                color=cm(idx), lw=2, label=fr"$\mathcal{{F}}<{th}$")
    ax.set_xlim(r_vals[0], r_vals[-1])
    ax.set_xlabel(r"$r$", fontsize=13)
    ax.set_ylabel(r"Crossing time $t^*$", fontsize=13)
    ax.set_title(
        fr"Fidelity Decay Times  "
        fr"(ref: {reference_label}, $\theta={theta_deg:.0f}°$)"
    )
    ax.legend(fontsize=10); ax.grid(ls="--", alpha=0.5)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    _save(fig, filename, show)


def plot_fidelity_all(r_vals, t_grid, F_init, F_pure,
                       theta_deg, t_indices=None, r_indices=None,
                       prefix="fidelity", show=True):
    """Generate and save the full suite of fidelity plots."""
    Nt = len(t_grid); Nr = len(r_vals)
    ti = t_indices or list(np.linspace(0, Nt-1, min(6,Nt), dtype=int))
    ri = r_indices or list(np.linspace(0, Nr-1, min(6,Nr), dtype=int))

    print("[visualize_fidelity] heatmap (initial) ...")
    plot_fidelity_heatmap(r_vals, t_grid, F_init, theta_deg,
                           "initial",
                           filename=f"{prefix}_F_heatmap_initial.png",
                           show=show)

    print("[visualize_fidelity] heatmap (singlet) ...")
    plot_fidelity_heatmap(r_vals, t_grid, F_pure, theta_deg,
                           "singlet",
                           filename=f"{prefix}_F_heatmap_singlet.png",
                           show=show)

    print("[visualize_fidelity] F(t) (initial) ...")
    plot_fidelity_vs_t(r_vals, t_grid, F_init, theta_deg, "initial",
                        r_indices=ri,
                        filename=f"{prefix}_F_vs_t_initial.png", show=show)

    print("[visualize_fidelity] F(t) (singlet) ...")
    plot_fidelity_vs_t(r_vals, t_grid, F_pure, theta_deg, "singlet",
                        r_indices=ri,
                        filename=f"{prefix}_F_vs_t_singlet.png", show=show)

    print("[visualize_fidelity] F(r) (initial) ...")
    plot_fidelity_vs_r(r_vals, t_grid, F_init, theta_deg, "initial",
                        t_indices=ti,
                        filename=f"{prefix}_F_vs_r_initial.png", show=show)

    print("[visualize_fidelity] F(r) (singlet) ...")
    plot_fidelity_vs_r(r_vals, t_grid, F_pure, theta_deg, "singlet",
                        t_indices=ti,
                        filename=f"{prefix}_F_vs_r_singlet.png", show=show)

    print("[visualize_fidelity] comparison panel ...")
    plot_fidelity_comparison(r_vals, t_grid, F_init, F_pure, theta_deg,
                              filename=f"{prefix}_F_comparison.png",
                              show=show)

    print("[visualize_fidelity] decay times (initial) ...")
    plot_fidelity_decay_times(r_vals, t_grid, F_init, theta_deg, "initial",
                               filename=f"{prefix}_F_decay_initial.png",
                               show=show)

    print("[visualize_fidelity] decay times (singlet) ...")
    plot_fidelity_decay_times(r_vals, t_grid, F_pure, theta_deg, "singlet",
                               filename=f"{prefix}_F_decay_singlet.png",
                               show=show)

    print("[visualize_fidelity] All fidelity plots done.")