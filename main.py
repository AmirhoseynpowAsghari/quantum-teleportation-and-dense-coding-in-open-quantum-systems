# =============================================================================
# main.py
# Orchestration script: self-consistency → Green's functions →
#   density matrices → non-Markovian evolution → observables → save/plot.
#
# Usage
# -----
#   python main.py
#   python main.py --kind dephasing --bath common --plot
# =============================================================================

import argparse
import numpy as np

import config
from self_consistency  import run_selfconsistency
from greens_functions  import compute_greens_functions
from density_matrix    import build_density_matrices, verify_density_matrices
from evolution         import evolve_grid
from observables       import compute_observables_grid
from visualization     import (plot_concurrence_heatmap,
                                plot_concurrence_vs_r,
                                plot_concurrence_vs_t,
                                plot_multi_observable)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Hubbard Cooper Pair non-Markovian evolution"
    )
    p.add_argument("--kind",    default=config.NOISE_KIND,
                   choices=["dephasing", "amp"],
                   help="Noise channel type")
    p.add_argument("--bath",    default=config.NOISE_BATH,
                   choices=["independent", "common"],
                   help="Bath topology")
    p.add_argument("--gammaA",  type=float, default=config.GAMMA_A)
    p.add_argument("--GammaA",  type=float, default=config.BIG_GAMMA_A)
    p.add_argument("--gammaB",  type=float, default=config.GAMMA_B)
    p.add_argument("--GammaB",  type=float, default=config.BIG_GAMMA_B)
    p.add_argument("--plot",    action="store_true",
                   help="Show interactive plots")
    p.add_argument("--seed",    type=int, default=None,
                   help="Random seed for reproducibility")
    p.add_argument("--theta",   type=int, default=config.THETA_PLOT_IDX,
                   help="θ index for observable output (0-based)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    show = args.plot or config.MAKE_PLOTS

    # ------------------------------------------------------------------
    # 1. BdG self-consistency
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1 – BdG self-consistency")
    print("=" * 60)
    bdg = run_selfconsistency(seed=args.seed)

    # ------------------------------------------------------------------
    # 2. Green's functions on (θ, r) grid
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 – Green's functions")
    print("=" * 60)
    r_vals, theta_vals, G_uu, F_ud = compute_greens_functions(bdg, verbose=True)

    # ------------------------------------------------------------------
    # 3. Density matrices
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 – Density matrices")
    print("=" * 60)
    rho_matrices = build_density_matrices(G_uu, F_ud)
    ok = verify_density_matrices(rho_matrices)
    print(f"[density_matrix] Verification: {'PASSED' if ok else 'FAILED'}")

    # ------------------------------------------------------------------
    # 4. Time evolution
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 – Non-Markovian time evolution")
    print(f"         kind={args.kind}  bath={args.bath}")
    print(f"         γA={args.gammaA}  ΓA={args.GammaA}"
          f"  γB={args.gammaB}  ΓB={args.GammaB}")
    print("=" * 60)
    t_grid = np.linspace(config.T_MIN, config.T_MAX, config.N_T)

    rho_rt = evolve_grid(
        rho_matrices, t_grid,
        kind=args.kind, bath=args.bath,
        gammaA=args.gammaA, GammaA=args.GammaA,
        gammaB=args.gammaB, GammaB=args.GammaB,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 5. Observables
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5 – Computing observables")
    print("=" * 60)
    obs = compute_observables_grid(rho_rt, r_vals, t_grid,
                                    theta_idx=args.theta)
    theta_deg = np.degrees(theta_vals[args.theta])

    C_r_final = obs["C"][:, -1]
    print(f"Concurrence at t_max: min={C_r_final.min():.4f}  "
          f"max={C_r_final.max():.4f}")

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6 – Saving results")
    print("=" * 60)
    np.savez_compressed(
        config.OUT_RHO,
        t_grid=t_grid,
        theta_vals=theta_vals,
        r_vals=r_vals,
        C_rt=obs["C"],
        Fs_rt=obs["Fs"],
        purity_rt=obs["pur"],
        entropy_rt=obs["S"],
        C_r_final=C_r_final,
        noise_kind=np.bytes_(args.kind),
        noise_bath=np.bytes_(args.bath),
        gammaA=args.gammaA, GammaA=args.GammaA,
        gammaB=args.gammaB, GammaB=args.GammaB,
    )
    print(f"[main] Saved → {config.OUT_RHO}")

    # Legacy heatmap file (backwards compatible)
    np.savez_compressed(
        config.OUT_HEATMAP,
        C_rt=obs["C"],
        r_vals=r_vals,
        t_grid=t_grid,
        theta_index=args.theta,
    )
    print(f"[main] Saved → {config.OUT_HEATMAP}")

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    if show:
        print("\n" + "=" * 60)
        print("STEP 7 – Plotting")
        print("=" * 60)
        plot_concurrence_heatmap(obs, r_vals, t_grid, theta_deg,
                                  save_path="concurrence_heatmap.png",
                                  show=True)

        t_snaps = [0, config.N_T // 4,
                   config.N_T // 2, config.N_T - 1]
        plot_concurrence_vs_r(obs, r_vals, t_snaps, t_grid, theta_deg,
                               save_path="C_vs_r.png", show=True)

        r_snaps = [0, len(r_vals) // 4,
                   len(r_vals) // 2, len(r_vals) - 1]
        plot_concurrence_vs_t(obs, t_grid, r_snaps, r_vals, theta_deg,
                               save_path="C_vs_t.png", show=True)

        plot_multi_observable(obs, r_vals, t_grid, theta_deg,
                               save_path="multi_obs.png", show=True)

    print("\n[main] All done.")


if __name__ == "__main__":
    main()