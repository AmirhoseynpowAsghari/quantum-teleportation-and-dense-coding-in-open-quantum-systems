# =============================================================================
# visualization.py
# ALL figures saved to FIG_DIR/
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import figure_path


def _save(fig, filename, show=True):
    fp = figure_path(filename)
    fig.savefig(fp, dpi=180, bbox_inches="tight")
    print(f"[visualization] Saved -> {fp}")
    if show:
        plt.show()
    plt.close(fig)


def _cbar(ax, im, label=""):
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.08)
    cb  = plt.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=11)
    return cb


# ---------------------------------------------------------------------------

def plot_concurrence_vs_r(obs, r_vals, t_grid, theta_deg,
                           t_indices=None, filename="C_vs_r.png",
                           show=True, figsize=(8,5)):
    C  = obs["C"]; Nt = C.shape[1]
    ti = t_indices or list(np.linspace(0, Nt-1, min(6,Nt), dtype=int))
    cm = plt.get_cmap("plasma", len(ti))
    fig, ax = plt.subplots(figsize=figsize)
    for idx, k in enumerate(ti):
        ax.plot(r_vals, C[:,k], color=cm(idx), lw=2,
                label=fr"$t={t_grid[k]:.1f}$")
    ax.set_xlim(r_vals[0], r_vals[-1]); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"$r$", fontsize=13)
    ax.set_ylabel(r"$\mathcal{C}(r,t)$", fontsize=13)
    ax.set_title(fr"Concurrence vs $r$  ($\theta={theta_deg:.0f}°$)")
    ax.legend(fontsize=9, ncol=2); ax.grid(ls="--", alpha=0.5)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    _save(fig, filename, show)


def plot_concurrence_heatmap(obs, r_vals, t_grid, theta_deg,
                               filename="C_heatmap.png",
                               show=True, figsize=(9,6)):
    C  = obs["C"].T
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(C, origin="lower", aspect="auto",
                    extent=[r_vals[0], r_vals[-1], t_grid[0], t_grid[-1]],
                    cmap="inferno", interpolation="bilinear")
    if C.max() > C.min():
        lvl = np.linspace(C.min(), C.max(), 9)[1:-1]
        cs  = ax.contour(C, levels=lvl, colors="w", linewidths=0.6, alpha=0.5,
                          extent=[r_vals[0], r_vals[-1],
                                  t_grid[0],  t_grid[-1]], origin="lower")
        ax.clabel(cs, fmt="%.2f", fontsize=8)
    _cbar(ax, im, r"$\mathcal{C}(r,t)$")
    ax.set_xlabel(r"$r$", fontsize=13); ax.set_ylabel(r"$t$", fontsize=13)
    ax.set_title(fr"Concurrence Heatmap  ($\theta={theta_deg:.0f}°$)")
    fig.tight_layout()
    _save(fig, filename, show)


def plot_concurrence_vs_t(obs, t_grid, r_vals, theta_deg,
                           r_indices=None, filename="C_vs_t.png",
                           show=True, figsize=(8,5)):
    C  = obs["C"]; Nr = C.shape[0]
    ri = r_indices or list(np.linspace(0, Nr-1, min(6,Nr), dtype=int))
    cm = plt.get_cmap("viridis", len(ri))
    fig, ax = plt.subplots(figsize=figsize)
    for idx, j in enumerate(ri):
        ax.plot(t_grid, C[j,:], color=cm(idx), lw=2,
                label=fr"$r={r_vals[j]:.2f}$")
    ax.set_xlim(t_grid[0], t_grid[-1]); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"$t$", fontsize=13)
    ax.set_ylabel(r"$\mathcal{C}(r,t)$", fontsize=13)
    ax.set_title(fr"Concurrence vs $t$  ($\theta={theta_deg:.0f}°$)")
    ax.legend(fontsize=9, ncol=2); ax.grid(ls="--", alpha=0.5)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    _save(fig, filename, show)


def plot_multi_observable(obs, r_vals, t_grid, theta_deg,
                            filename="multi_obs.png",
                            show=True, figsize=(14,9)):
    keys  = ["C","Fs","pur","S"]
    labs  = [r"$\mathcal{C}$", r"$F_s$",
             r"Purity $\mathrm{Tr}[\rho^2]$", r"$S(\rho)$"]
    cmaps = ["inferno","cividis","plasma","magma"]
    ext   = [r_vals[0], r_vals[-1], t_grid[0], t_grid[-1]]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax, key, lab, cm in zip(axes.flat, keys, labs, cmaps):
        im = ax.imshow(obs[key].T, origin="lower", aspect="auto",
                        extent=ext, cmap=cm, interpolation="bilinear")
        _cbar(ax, im, lab)
        ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$t$")
        ax.set_title(lab, fontsize=12)

    fig.suptitle(fr"All observables  ($\theta={theta_deg:.0f}°$)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, filename, show)


def run_all_plots(obs, r_vals, t_grid, theta_deg,
                   t_indices=None, r_indices=None,
                   prefix="hubbard", show=True):
    Nt = len(t_grid); Nr = len(r_vals)
    ti = t_indices or list(np.linspace(0, Nt-1, min(6,Nt), dtype=int))
    ri = r_indices or list(np.linspace(0, Nr-1, min(6,Nr), dtype=int))

    plot_concurrence_vs_r(obs, r_vals, t_grid, theta_deg,
                           t_indices=ti,
                           filename=f"{prefix}_C_vs_r.png", show=show)
    plot_concurrence_heatmap(obs, r_vals, t_grid, theta_deg,
                               filename=f"{prefix}_C_heatmap.png", show=show)
    plot_concurrence_vs_t(obs, t_grid, r_vals, theta_deg,
                           r_indices=ri,
                           filename=f"{prefix}_C_vs_t.png", show=show)
    plot_multi_observable(obs, r_vals, t_grid, theta_deg,
                            filename=f"{prefix}_multi_obs.png", show=show)