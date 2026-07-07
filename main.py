# =============================================================================
# main.py
# Full pipeline:
#   1. BdG self-consistency (fsolve, T=0)
#   2. Green's functions G(r,theta), F(r,theta)
#   3. Density matrices rho(theta, r)
#   4. Non-Markovian time evolution
#   5. Observables (C, Fs, purity, entropy)
#   6. Save observable data  -> data/
#   7. Concurrence plots     -> figures/
#   8. Green's function plots-> figures/
#   9. Fidelity analysis     -> data/ + figures/
#
# Usage
# -----
#   python main.py                    # uses config.py defaults
#   python main.py --plot             # force plots on
#   python main.py --no-plot          # force plots off
#   python main.py --kind dephasing --bath common --plot
#   python main.py --seed 42 --prefix my_run --plot
# =============================================================================

import argparse
import numpy as np

import config
from utils             import ensure_dir, data_path
from self_consistency  import run_selfconsistency
from greens_functions  import compute_greens_functions, evolve_greens_functions
from density_matrix    import build_density_matrices, verify_density_matrices
from evolution         import evolve_grid
from observables       import compute_observables_grid
from visualization     import run_all_plots
from visualize_greens  import plot_greens_all
from fidelity          import (compare_fidelity_references,
                                save_fidelity_data)
from visualize_fidelity import plot_fidelity_all


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Hubbard Cooper Pair non-Markovian evolution"
    )
    # noise
    p.add_argument("--kind",   default=config.NOISE_KIND,
                   choices=["dephasing", "amp"],
                   help="Noise channel type (default: %(default)s)")
    p.add_argument("--bath",   default=config.NOISE_BATH,
                   choices=["independent", "common"],
                   help="Bath topology (default: %(default)s)")
    p.add_argument("--gammaA", type=float, default=config.GAMMA_A)
    p.add_argument("--GammaA", type=float, default=config.BIG_GAMMA_A)
    p.add_argument("--gammaB", type=float, default=config.GAMMA_B)
    p.add_argument("--GammaB", type=float, default=config.BIG_GAMMA_B)
    # misc
    p.add_argument("--seed",    type=int,  default=None)
    p.add_argument("--theta",   type=int,  default=config.THETA_PLOT_IDX,
                   help="Theta index for observable slice")
    p.add_argument("--plot",    action="store_true",
                   help="Enable plots (overrides config.MAKE_PLOTS)")
    p.add_argument("--no-plot", action="store_true",
                   help="Disable plots (overrides config.MAKE_PLOTS)")
    p.add_argument("--prefix",  type=str,  default=None,
                   help="Filename prefix for figures and data")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEP  = "=" * 68
STEP = "-" * 68


def _header(n, title):
    print(f"\n{STEP}")
    print(f"STEP {n}: {title}")
    print(STEP)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- output directories ----------------------------------------------
    ensure_dir(config.FIG_DIR)
    ensure_dir(config.DATA_DIR)

    # ---- plot flag -------------------------------------------------------
    show = config.MAKE_PLOTS
    if args.plot:    show = True
    if args.no_plot: show = False

    # ---- filename prefix -------------------------------------------------
    prefix = args.prefix or (
        f"hubbard_{args.kind}_{args.bath}"
        f"_gA{args.gammaA:.2f}_GA{args.GammaA:.3f}"
        f"_gB{args.gammaB:.2f}_GB{args.GammaB:.3f}"
    )

    print(SEP)
    print("HUBBARD COOPER PAIR – NON-MARKOVIAN EVOLUTION")
    print(f"  kind   = {args.kind}   bath = {args.bath}")
    print(f"  gammaA = {args.gammaA}  GammaA = {args.GammaA}")
    print(f"  gammaB = {args.gammaB}  GammaB = {args.GammaB}")
    print(f"  prefix = {prefix}")
    print(f"  figures -> {config.FIG_DIR}/")
    print(f"  data    -> {config.DATA_DIR}/")
    print(SEP)

    # =========================================================
    # STEP 1 – BdG self-consistency
    # =========================================================
    _header(1, "BdG self-consistency (fsolve, T=0)")
    bdg = run_selfconsistency(seed=args.seed)
    if not bdg["converged"]:
        print("  WARNING: solver did not fully converge!")

    # =========================================================
    # STEP 2 – Green's functions
    # =========================================================
    _header(2, "Green's functions  G(r,θ) and F(r,θ)")
    r_vals, theta_vals, G_uu, F_ud = compute_greens_functions(
        bdg, verbose=True
    )

    # =========================================================
    # STEP 3 – Density matrices
    # =========================================================
    _header(3, "Density matrices  ρ(θ, r)")
    rho_matrices = build_density_matrices(G_uu, F_ud)
    ok = verify_density_matrices(rho_matrices)
    print(f"  Verification: {'PASSED' if ok else 'FAILED'}")

    # =========================================================
    # STEP 4 – Time evolution
    # =========================================================
    _header(4, f"Non-Markovian time evolution  "
               f"(kind={args.kind}, bath={args.bath})")
    t_grid = np.linspace(config.T_MIN, config.T_MAX, config.N_T)
    print(f"  t grid: [{t_grid[0]:.1f}, {t_grid[-1]:.1f}]  "
          f"N_T={config.N_T}")

    rho_rt = evolve_grid(
        rho_matrices, t_grid,
        kind=args.kind, bath=args.bath,
        gammaA=args.gammaA, GammaA=args.GammaA,
        gammaB=args.gammaB, GammaB=args.GammaB,
        verbose=True,
    )

    # =========================================================
    # STEP 5 – Observables
    # =========================================================
    _header(5, "Computing observables  (C, Fs, purity, entropy)")
    obs = compute_observables_grid(
        rho_rt, r_vals, t_grid, theta_idx=args.theta
    )
    theta_deg = float(np.degrees(theta_vals[args.theta]))
    C_fin     = obs["C"][:, -1]
    print(f"  theta   = {theta_deg:.0f} deg")
    print(f"  C(t_max): min={C_fin.min():.4f}  "
          f"max={C_fin.max():.4f}  mean={C_fin.mean():.4f}")

    # =========================================================
    # STEP 6 – Save observable data
    # =========================================================
    _header(6, "Saving observable data -> data/")

    out_rho = data_path(config.OUT_RHO)
    np.savez_compressed(
        out_rho,
        t_grid      = t_grid,
        theta_vals  = theta_vals,
        r_vals      = r_vals,
        C_rt        = obs["C"],
        Fs_rt       = obs["Fs"],
        purity_rt   = obs["pur"],
        entropy_rt  = obs["S"],
        C_r_final   = C_fin,
        noise_kind  = np.bytes_(args.kind),
        noise_bath  = np.bytes_(args.bath),
        gammaA      = args.gammaA,
        GammaA      = args.GammaA,
        gammaB      = args.gammaB,
        GammaB      = args.GammaB,
        theta_index = args.theta,
        theta_deg   = theta_deg,
        Delta0      = bdg["Delta0"],
        DeltaS      = bdg["DeltaS"],
        mu          = bdg["mu"],
    )
    print(f"  Saved -> {out_rho}")

    out_hm = data_path(config.OUT_HEATMAP)
    np.savez_compressed(
        out_hm,
        C_rt        = obs["C"],
        r_vals      = r_vals,
        t_grid      = t_grid,
        theta_index = args.theta,
        theta_deg   = theta_deg,
    )
    print(f"  Saved -> {out_hm}")

    # snapshot indices shared by several plot functions
    ti = list(np.linspace(0, config.N_T - 1,
                           min(6, config.N_T), dtype=int))
    ri = list(np.linspace(0, len(r_vals) - 1,
                           min(6, len(r_vals)), dtype=int))

    # =========================================================
    # STEP 7 – Concurrence / observable plots
    # =========================================================
    if show:
        _header(7, "Concurrence & observable plots -> figures/")
        run_all_plots(
            obs, r_vals, t_grid, theta_deg,
            t_indices = ti,
            r_indices = ri,
            prefix    = prefix,
            show      = show,
        )
    else:
        _header(7, "Concurrence plots SKIPPED (--no-plot)")

    # =========================================================
    # STEP 8 – Green's function plots
    # =========================================================
    if show:
        _header(8, "Green's function plots -> figures/")
        print("  Evolving Green's functions in time ...")
        G_t, F_t = evolve_greens_functions(
            G_uu, F_ud, t_grid,
            gamma  = args.gammaA,
            GammaA = args.GammaA,
            kind   = args.kind,
            verbose = True,
        )

        # Save Green's function arrays
        out_gf = data_path(f"{prefix}_greens.npz")
        np.savez_compressed(
            out_gf,
            G_t        = G_t,
            F_t        = F_t,
            r_vals     = r_vals,
            t_grid     = t_grid,
            theta_vals = theta_vals,
        )
        print(f"  Saved Green's functions -> {out_gf}")

        plot_greens_all(
            G_t, F_t, r_vals, t_grid, theta_vals,
            theta_idx = args.theta,
            t_indices = ti,
            prefix    = prefix,
            show      = show,
        )
    else:
        _header(8, "Green's function plots SKIPPED (--no-plot)")

    # =========================================================
    # STEP 9 – Fidelity analysis
    # =========================================================
    _header(9, "Fidelity analysis")

    fid_noise = dict(
        kind      = args.kind,
        bath      = args.bath,
        nuA       = config.FID_NU_A,
        GammaA    = config.FID_GAMMA_A,
        nuB       = config.FID_NU_B,
        GammaB    = config.FID_GAMMA_B,
        gammaA    = config.FID_GAMMA_AD_A,
        GammaAD_A = config.FID_BIGAMMA_AD_A,
        gammaB    = config.FID_GAMMA_AD_B,
        GammaAD_B = config.FID_BIGAMMA_AD_B,
    )

    F_init, F_pure = compare_fidelity_references(
        rho_matrices, r_vals, t_grid,
        theta_idx = args.theta,
        verbose   = True,
        **fid_noise,
    )

    save_fidelity_data(
        F_init, F_pure, r_vals, t_grid,
        theta_deg = theta_deg,
        prefix    = prefix,
    )

    if show:
        print("  Generating fidelity plots -> figures/")
        plot_fidelity_all(
            r_vals, t_grid, F_init, F_pure,
            theta_deg = theta_deg,
            t_indices = ti,
            r_indices = ri,
            prefix    = prefix,
            show      = show,
        )
    else:
        print("  Fidelity plots SKIPPED (--no-plot)")

        # =========================================================
    # STEP 10 – Theta-dependence analysis
    # =========================================================
    _header(10, "Theta-dependence analysis (C and F vs theta)")

    from visualize_theta import run_theta_analysis
    from utils import data_path as _dp

    obs_all, F_all_init, F_all_pure = run_theta_analysis(
        rho_rt       = rho_rt,
        rho_matrices = rho_matrices,
        r_vals       = r_vals,
        t_grid       = t_grid,
        theta_vals   = theta_vals,
        prefix       = prefix,
        show         = show,
        fid_noise    = fid_noise,   # reuse dict from STEP 9
    )

    # Save theta-dependence data
    out_theta = _dp(f"{prefix}_theta_obs.npz")
    np.savez_compressed(
        out_theta,
        C_all    = obs_all["C"],
        Fs_all   = obs_all["Fs"],
        pur_all  = obs_all["pur"],
        S_all    = obs_all["S"],
        F_init   = F_all_init,
        F_pure   = F_all_pure,
        r_vals   = r_vals,
        t_grid   = t_grid,
        theta_vals = theta_vals,
    )
    print(f"  Saved -> {out_theta}")

        # =========================================================
    # STEP 11 – Holevo quantity analysis
    # =========================================================
    _header(11, "Holevo quantity analysis")

    from holevo import run_holevo_analysis

    holevo_results = run_holevo_analysis(
        rho_rt            = rho_rt,
        r_vals            = r_vals,
        t_grid            = t_grid,
        theta_vals        = theta_vals,
        theta_idx         = args.theta,
        compute_all_theta = True,
        prefix            = prefix,
        show              = show,
        verbose           = True,
    )

    # =========================================================
    # DONE
    # =========================================================
    print(f"\n{SEP}")
    print("PIPELINE COMPLETE")
    print(f"  data/    :")
    print(f"    {config.OUT_RHO}")
    print(f"    {config.OUT_HEATMAP}")
    print(f"    {prefix}_fidelity.npz")
    print(f"    {prefix}_greens.npz")
    print(f"  figures/ : {prefix}_*.png")
    print(SEP)


if __name__ == "__main__":
    main()