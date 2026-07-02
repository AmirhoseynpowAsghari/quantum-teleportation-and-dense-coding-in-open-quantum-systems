# =============================================================================
# config.py
# Configuration and physical parameters for the Hubbard Cooper Pair evolution.
# =============================================================================

import numpy as np

# -----------------------------------------------------------------------------
# Physical parameters
# -----------------------------------------------------------------------------
t_hop   = 0.1          # hopping amplitude (eV or arbitrary units)
U       = 75 * t_hop   # on-site Hubbard interaction
Delta_t = 0.45         # triplet pairing amplitude
V_SO    = 0.0 * t_hop  # spin-orbit coupling strength
n_elec  = 1.875        # average electron density per site

# -----------------------------------------------------------------------------
# k-space grid
# -----------------------------------------------------------------------------
Nk = 400               # number of k-points per direction (reduce for quick tests)

# -----------------------------------------------------------------------------
# Self-consistency loop
# -----------------------------------------------------------------------------
SC_MAX_ITER  = 900_000  # maximum BdG self-consistency iterations
SC_TOL       = 1e-8     # convergence tolerance on Δ
SC_DAMP      = 0.1      # damping factor λ  (0 < λ ≤ 1)

# -----------------------------------------------------------------------------
# Real-space (r, θ) grid for Green's functions
# -----------------------------------------------------------------------------
R_MIN, R_MAX, N_R = 0.0, 10.0, 200       # radial grid
THETA_DEG         = [0, 15, 30, 45,
                     60, 75, 90]          # angles in degrees

# -----------------------------------------------------------------------------
# Time evolution
# -----------------------------------------------------------------------------
T_MIN, T_MAX, N_T = 0.0, 80.0, 161      # time grid

# Noise channel
NOISE_KIND = "amp"           # "dephasing"  or  "amp" (amplitude damping)
NOISE_BATH = "independent"   # "independent" or "common" (dephasing only)

GAMMA_A = 4.5    # coupling strength, qubit A
BIG_GAMMA_A = 0.01  # reservoir memory rate, qubit A
GAMMA_B = 4.5    # coupling strength, qubit B
BIG_GAMMA_B = 0.01  # reservoir memory rate, qubit B

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
OUT_RHO   = "rho_rt_results.npz"
OUT_HEATMAP = "C_rt_heatmap.npz"
MAKE_PLOTS  = False   # set True for interactive plots
THETA_PLOT_IDX = 0    # which θ slice to use for the heatmap