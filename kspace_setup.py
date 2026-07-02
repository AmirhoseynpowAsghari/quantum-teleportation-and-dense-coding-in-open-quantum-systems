# =============================================================================
# kspace_setup.py
# Build 2-D k-space grid and spin-resolved dispersion ε_{k,s}.
# =============================================================================

import numpy as np
from config import t_hop, V_SO, Nk


def build_kgrid():
    """
    Returns
    -------
    kx, ky : 1-D arrays of length Nk
    KX, KY : 2-D meshgrid arrays  (Nk × Nk)
    epsilon_ks : array, shape (Nk, Nk, 2)
        epsilon_ks[:,:,0]  →  s = +1
        epsilon_ks[:,:,1]  →  s = -1
    sqrt_sin : array, shape (Nk, Nk)
        sqrt(sin²(kx) + sin²(ky))
    """
    kx = np.linspace(-np.pi, np.pi, Nk)
    ky = np.linspace(-np.pi, np.pi, Nk)
    KX, KY = np.meshgrid(kx, ky)

    sqrt_sin = np.sqrt(np.sin(KX)**2 + np.sin(KY)**2)
    cos_sum  = np.cos(KX) + np.cos(KY)

    epsilon_ks = np.empty((Nk, Nk, 2))
    epsilon_ks[:, :, 0] = -2*t_hop*cos_sum - 2*V_SO*sqrt_sin  # s = +1
    epsilon_ks[:, :, 1] = -2*t_hop*cos_sum + 2*V_SO*sqrt_sin  # s = -1

    return kx, ky, KX, KY, epsilon_ks, sqrt_sin


if __name__ == "__main__":
    kx, ky, KX, KY, eps, ss = build_kgrid()
    print(f"k-grid: {Nk}×{Nk}")
    print(f"ε_ks   min={eps.min():.4f}  max={eps.max():.4f}")
    print(f"sqrt_sin min={ss.min():.4f}  max={ss.max():.4f}")