# =============================================================================
# visualization.py
# Plotting utilities for the Hubbard Cooper Pair evolution results.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from config import THETA_DEG


# ---------------------------------------------------------------------------
# Heatmap: C(r, t)
# ---------------------------------------------------------------------------

def plot_concurrence_heatmap(obs, r_vals, t_grid, theta_deg,
                              save_path=None, show=True):
    """
    2-D false-colour plot of concurrence C(r, t) for a fixed angle.

    Parameters
    ----------
    obs       : dict returned by observables.compute_observables_grid
    r_vals    : 1-D array
    t_grid    : 1-D array
    theta_deg : float   angle in degrees (for title)
    save_path : str or None   if given, save figure to this path
    show      : bool
    """
    C = obs["C"]          # (Nr, Nt)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(C, origin="lower", aspect="auto",
                   extent=[r_vals[0], r_vals[-1], t_grid[0], t_grid[-1]],
                   cmap="inferno")
    plt.colorbar(im, ax=ax, label=r"Concurrence $\mathcal{C}(r,t)$")
    ax.set_xlabel(r"Distance $r$")
    ax.set_ylabel(r"Time $t$")
    ax.set_title(fr"Concurrence Heatmap  ($\theta = {theta_deg:.0f}°$)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[visualization] Saved heatmap → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Line plots at fixed r or fixed t
# ---------------------------------------------------------------------------

def plot_concurrence_vs_r(obs, r_vals, t_indices, t_grid,
                           theta_deg, save_path=None, show=True):
    """
    C(r) curves for selected time snapshots.

    Parameters
    ----------
    t_indices : list of int   which time indices to plot
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for k in t_indices:
        ax.plot(r_vals, obs["C"][:, k],
                label=fr"$t = {t_grid[k]:.1f}$")
    ax.set_xlabel(r"Distance $r$")
    ax.set_ylabel(r"Concurrence $\mathcal{C}$")
    ax.set_title(fr"$\mathcal{{C}}(r)$ at selected times  ($\theta={theta_deg:.0f}°$)")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[visualization] Saved C(r) plot → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_concurrence_vs_t(obs, t_grid, r_indices, r_vals,
                           theta_deg, save_path=None, show=True):
    """
    C(t) curves for selected spatial points.

    Parameters
    ----------
    r_indices : list of int   which r indices to plot
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for j in r_indices:
        ax.plot(t_grid, obs["C"][j, :],
                label=fr"$r = {r_vals[j]:.2f}$")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Concurrence $\mathcal{C}$")
    ax.set_title(fr"$\mathcal{{C}}(t)$ at selected distances  ($\theta={theta_deg:.0f}°$)")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[visualization] Saved C(t) plot → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-observable panel
# ---------------------------------------------------------------------------

def plot_multi_observable(obs, r_vals, t_grid, theta_deg,
                           save_path=None, show=True):
    """
    2×2 panel: C, F_s, purity, von-Neumann entropy – all as heatmaps.
    """
    keys   = ["C",   "Fs",              "pur",     "S"]
    labels = [r"$\mathcal{C}$",
              r"$F_s$",
              r"Purity $\mathrm{Tr}[\rho^2]$",
              r"Entropy $S(\rho)$"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key, label in zip(axes.flat, keys, labels):
        data = obs[key]
        im   = ax.imshow(data, origin="lower", aspect="auto",
                         extent=[r_vals[0], r_vals[-1], t_grid[0], t_grid[-1]],
                         cmap="viridis")
        plt.colorbar(im, ax=ax, label=label)
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$t$")
        ax.set_title(label)
    fig.suptitle(fr"Observables   ($\theta = {theta_deg:.0f}°$)", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[visualization] Saved multi-observable panel → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # Tiny smoke test
    rng  = np.random.default_rng(1)
    r    = np.linspace(0, 10, 20)
    t    = np.linspace(0, 80, 30)
    fake = {"C":   rng.random((20, 30)),
            "Fs":  rng.random((20, 30)),
            "pur": rng.random((20, 30)),
            "S":   rng.random((20, 30))}
    plot_concurrence_heatmap(fake, r, t, theta_deg=0, show=False)
    print("[visualization] smoke test passed.")