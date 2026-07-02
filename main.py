# =============================================================================
# main.py
# Orchestration script: self-consistency → Green's functions →
#   density matrices → non-Markovian evolution → observables → save/plot.
#
# Usage
# -----
#   python main.py
#   python main.py --kind dephasing --bath common --plot
#   python main.py --seed 42 --gammaA 2.0 --GammaA 0.05
# =============================================================================

import argparse
import numpy as np

import config
from self_consistency import run_selfconsistency
from greens_functions import compute_greens_functions
from density_matrix import build_density_matrices, verify_density_matrices
from evolution import evolve_grid
from observables import compute_observables_grid
from visualization import run_all_plots


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
    """
    p = argparse.ArgumentParser(
        description="Hubbard Cooper Pair non-Markovian evolution pipeline"
    )

    # Noise parameters
    p.add_argument(
        "--kind",
        default=config.NOISE_KIND,
        choices=["dephasing", "amp"],
        help="Noise channel type (default: %(default)s)",
    )
    p.add_argument(
        "--bath",
        default=config.NOISE_BATH,
        choices=["independent", "common"],
        help="Bath topology (default: %(default)s)",
    )
    p.add_argument(
        "--gammaA",
        type=float,
        default=config.GAMMA_A,
        help="Coupling strength for qubit A (default: %(default)s)",
    )
    p.add_argument(
        "--GammaA",
        type=float,
        default=config.BIG_GAMMA_A,
        help="Memory rate for qubit A (default: %(default)s)",
    )
    p.add_argument(
        "--gammaB",
        type=float,
        default=config.GAMMA_B,
        help="Coupling strength for qubit B (default: %(default)s)",
    )
    p.add_argument(
        "--GammaB",
        type=float,
        default=config.BIG_GAMMA_B,
        help="Memory rate for qubit B (default: %(default)s)",
    )

    # Simulation control
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    p.add_argument(
        "--theta",
        type=int,
        default=config.THETA_PLOT_IDX,
        help="Theta index for observable extraction (default: %(default)s)",
    )

    # Output control
    p.add_argument(
        "--plot",
        action="store_true",
        help="Generate and show/save plots (overrides config.MAKE_PLOTS)",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting even if config.MAKE_PLOTS is True",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Filename prefix for saved figures (default: auto-generated)",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    """
    Run the full Hubbard Cooper Pair evolution pipeline.
    """
    args = parse_args()

    # Determine whether to show plots
    show_plots = config.MAKE_PLOTS
    if args.plot:
        show_plots = True
    if args.no_plot:
        show_plots = False

    print("=" * 70)
    print("HUBBARD COOPER PAIR – NON-MARKOVIAN EVOLUTION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. BdG self-consistency
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 1: BdG self-consistency")
    print("-" * 70)
    bdg = run_selfconsistency(seed=args.seed)

    # ------------------------------------------------------------------
    # 2. Compute Green's functions
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 2: Computing Green's functions G(r,θ) and F(r,θ)")
    print("-" * 70)
    r_vals, theta_vals, G_uu, F_ud = compute_greens_functions(bdg, verbose=True)

    # ------------------------------------------------------------------
    # 3. Build density matrices
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 3: Building density matrices ρ(θ,r)")
    print("-" * 70)
    rho_matrices = build_density_matrices(G_uu, F_ud)
    is_ok = verify_density_matrices(rho_matrices, atol=1e-6)
    if is_ok:
        print("[main] Density matrix verification: PASSED")
    else:
        print("[main] Density matrix verification: FAILED (check warnings above)")

    # ------------------------------------------------------------------
    # 4. Time evolution
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 4: Non-Markovian time evolution")
    print("-" * 70)
    print(f"  Channel: kind = {args.kind}, bath = {args.bath}")
    print(
        f"  Parameters: γA = {args.gammaA}, ΓA = {args.GammaA}, "
        f"γB = {args.gammaB}, ΓB = {args.GammaB}"
    )

    # Build time grid
    t_grid = np.linspace(config.T_MIN, config.T_MAX, config.N_T)

    # Evolve
    rho_rt = evolve_grid(
        rho_matrices,
        t_grid,
        kind=args.kind,
        bath=args.bath,
        gammaA=args.gammaA,
        GammaA=args.GammaA,
        gammaB=args.gammaB,
        GammaB=args.GammaB,
        verbose=True,
    )

    print("[main] Time evolution complete.")

    # ------------------------------------------------------------------
    # 5. Compute observables
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 5: Computing observables")
    print("-" * 70)

    obs = compute_observables_grid(
        rho_rt, r_vals, t_grid, theta_idx=args.theta
    )

    theta_deg = float(np.degrees(theta_vals[args.theta]))

    # Quick summary
    C_final = obs["C"][:, -1]
    print(
        f"  Concurrence at t_max: min = {C_final.min():.4f}, "
        f"  max = {C_final.max():.4f}, "
        f"  mean = {C_final.mean():.4f}"
    )

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 6: Saving results")
    print("-" * 70)

    # Main results file
    np.savez_compressed(
        config.OUT_RHO,
        t_grid=t_grid,
        theta_vals=theta_vals,
        r_vals=r_vals,
        C_rt=obs["C"],
        Fs_rt=obs["Fs"],
        purity_rt=obs["pur"],
        entropy_rt=obs["S"],
        C_r_final=C_final,
        noise_kind=args.kind,
        noise_bath=args.bath,
        gammaA=args.gammaA,
        GammaA=args.GammaA,
        gammaB=args.gammaB,
        GammaB=args.GammaB,
        theta_index=args.theta,
        theta_deg=theta_deg,
    )
    print(f"[main] Saved → {config.OUT_RHO}")

    # Backwards-compatible heatmap file
    np.savez_compressed(
        config.OUT_HEATMAP,
        C_rt=obs["C"],
        r_vals=r_vals,
        t_grid=t_grid,
        theta_index=args.theta,
    )
    print(f"[main] Saved → {config.OUT_HEATMAP}")

    # ------------------------------------------------------------------
    # 7. Generate plots
    # ------------------------------------------------------------------
    if show_plots:
        print("\n" + "-" * 70)
        print("STEP 7: Generating plots")
        print("-" * 70)

        from greens_functions import evolve_greens_functions
        from visualize_greens import plot_greens_all

        print("[main] Evolving Green's functions in time ...")
        G_t, F_t = evolve_greens_functions(
            G_uu, F_ud, t_grid,
            gamma=args.gammaA,
            Gamma=args.GammaA,
            kind=args.kind,
            verbose=True,
        )

        # Choose snapshot indices
        t_snap = np.linspace(0, config.N_T - 1, 6, dtype=int).tolist()

        plot_greens_all(
            G_t, F_t,
            r_vals=r_vals,
            t_grid=t_grid,
            theta_vals=theta_vals,
            theta_idx=args.theta,
            t_indices=t_snap,
            prefix=f"greens_{args.kind}_{args.bath}",
            show=True,
        )

        # Determine filename prefix
        if args.prefix is not None:
            prefix = args.prefix
        else:
            prefix = (
                f"hubbard_{args.kind}_{args.bath}_"
                f"gA{args.gammaA:.2f}_GA{args.GammaA:.3f}_"
                f"gB{args.gammaB:.2f}_GB{args.GammaB:.3f}"
            )

        # Choose snapshot indices for line plots
        Nr_total = len(r_vals)
        Nt_total = len(t_grid)

        # Evenly spaced time snapshots
        t_indices = np.linspace(0, Nt_total - 1, 6, dtype=int).tolist()
        # Evenly spaced r snapshots
        r_indices = np.linspace(0, Nr_total - 1, 6, dtype=int).tolist()

        # Generate all plots
        run_all_plots(
            obs,
            r_vals=r_vals,
            t_grid=t_grid,
            theta_deg=theta_deg,
            t_indices=t_indices,
            r_indices=r_indices,
            prefix=prefix,
            show=True,
        )

    else:
        print("\n" + "-" * 70)
        print("STEP 7: Skipping plots (disabled)")
        print("-" * 70)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Output files: {config.OUT_RHO}, {config.OUT_HEATMAP}")
    if show_plots:
        print("  Figures saved with prefix:", args.prefix or "auto")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()