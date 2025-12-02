
import numpy as np
from scipy.integrate import simpson


def compute_greens_functions(
        u_ks, v_ks, KX, KY, kx, ky,
        r_max=10, Nr=200,
        theta_deg=[0, 15, 30, 45, 60, 75, 90]
    ):
    """
    Compute real-space equal-time Green's functions:
        G_uu(r, θ)
        F_ud(r, θ)

    Arguments:
        u_ks, v_ks : arrays (Nk × Nk × 2)
        KX, KY     : k-mesh arrays (Nk × Nk)
        kx, ky     : 1D arrays
        r_max      : maximum real-space distance
        Nr         : number of radial points
        theta_deg  : list of angles (degrees)

    Returns:
        r_vals, theta_vals, G_uu_polar, F_ud_polar
    """

    Nk = len(kx)
    dkx = kx[1] - kx[0]
    dky = ky[1] - ky[0]

    r_vals = np.linspace(0, r_max, Nr)
    theta_vals = np.radians(theta_deg)

    # Storage
    G_uu_polar = np.zeros((len(theta_vals), Nr), dtype=complex)
    F_ud_polar = np.zeros((len(theta_vals), Nr), dtype=complex)

    # Loop over angles
    for i, theta in enumerate(theta_vals):

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Loop over radial distances
        for j, r in enumerate(r_vals):
            rx = r * cos_t
            ry = r * sin_t

            # phase = e^{i k·r}
            phase_factor = np.exp(1j * (KX * rx + KY * ry))

            # spin sum
            sum_G = np.zeros((Nk, Nk), dtype=complex)
            sum_F = np.zeros((Nk, Nk), dtype=complex)

            for s in range(2):
                u = u_ks[:, :, s]
                v = v_ks[:, :, s]

                # Normal Green's function at t = 0+
                sum_G += -1j * u * np.conj(u) * phase_factor

                # Anomalous Green's function (pairing)
                sum_F += 1j * np.conj(v) * np.conj(u) * np.conj(phase_factor)

            # Integrate over k-space using Simpson's rule
            G_val = simpson(
                y=simpson(sum_G, x=kx, axis=0),
                x=ky,
                axis=0
            )

            F_val = simpson(
                y=simpson(sum_F, x=kx, axis=0),
                x=ky,
                axis=0
            )

            # Normalize Fourier transform in 2D
            G_uu_polar[i, j] = G_val / (2 * np.pi)**2
            F_ud_polar[i, j] = F_val / (2 * np.pi)**2

    return r_vals, theta_vals, G_uu_polar, F_ud_polar


# Debug mode: run directly
if __name__ == "__main__":
    print("This module computes real-space Green's functions.")
    print("Use it by importing compute_greens_functions().")
