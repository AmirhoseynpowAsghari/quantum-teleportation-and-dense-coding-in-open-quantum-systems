# =============================================================================
# kspace_setup.py
# =============================================================================

import numpy as np
from config import t_hop, V_SO, Nk


def build_kgrid():
    """
    Build 2-D k-space grid and spin-resolved dispersion.

    Returns
    -------
    kx, ky          : 1-D arrays, length Nk
    KX, KY          : 2-D meshgrid arrays (Nk x Nk)
    epsilon_ks      : array (Nk, Nk, 2)
    sqrt_sin        : array (Nk, Nk)
    """
    kx = np.linspace(-np.pi, np.pi, Nk)
    ky = np.linspace(-np.pi, np.pi, Nk)
    KX, KY = np.meshgrid(kx, ky)

    sqrt_sin = np.sqrt(np.sin(KX)**2 + np.sin(KY)**2)
    cos_sum  = np.cos(KX) + np.cos(KY)

    epsilon_ks = np.empty((Nk, Nk, 2))
    epsilon_ks[:, :, 0] = -2*t_hop*cos_sum - 2*V_SO*sqrt_sin   # s = +1
    epsilon_ks[:, :, 1] = -2*t_hop*cos_sum + 2*V_SO*sqrt_sin   # s = -1

    return kx, ky, KX, KY, epsilon_ks, sqrt_sin


if __name__ == "__main__":
    kx, ky, KX, KY, eps, ss = build_kgrid()
    print(f"k-grid : {Nk}x{Nk}")
    print(f"eps    : min={eps.min():.4f}  max={eps.max():.4f}")