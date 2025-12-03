import numpy as np
from scipy.optimize import minimize, least_squares, basinhopping
from scipy.integrate import simpson
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time

# k-space Green's functions (Unchanged)
def compute_greens_functions(u_ks, v_ks, KX, KY, r, theta):
    Nk = KX.shape[0]
    kx = KX[0, :]
    ky = KY[:, 0]
    rx = r * np.cos(theta)
    ry = r * np.sin(theta)
    phase = np.exp(1j * (KX * rx + KY * ry))

    G_sum = np.zeros((Nk, Nk), dtype=complex)
    F_sum = np.zeros((Nk, Nk), dtype=complex)
    for s in range(2):
        G_sum += -1j * u_ks[..., s] * np.conj((u_ks[..., s])) * phase
        F_sum += 1j * np.conj(v_ks[..., s]) * np.conj(u_ks[..., s]) * np.conj(phase)

    # integrate with Simpson's rule
    Gx = simpson(G_sum, x=kx, axis=1)
    Fx = simpson(F_sum, x=kx, axis=1)
    G = simpson(Gx, x=ky) / (4 * np.pi**2)
    F = simpson(Fx, x=ky) / (4 * np.pi**2)
    return G, F

# Concurrence calculation (Unchanged)
def compute_concurrence(G, F, G0):
    # Ensure G0 is treated as real number from imag part
    G0_real = G0.imag
    if abs(G0_real) < 1e-12:
        print(f"Warning: G0.imag is very small ({G0_real}). May lead to instability.")
        return np.nan # Return NaN if G0 is too small

    g = G.imag / G0_real
    f = F.imag / G0_real
    p_numerator = (f**2 + g**2)
    p_denominator = (2 + f**2 - g**2)

    # Avoid division by zero or near-zero
    if abs(p_denominator) < 1e-12:
        print(f"Warning: Denominator in 'p' calculation is very small ({p_denominator}).")
        # Decide how to handle this - maybe related to a phase transition?
        # For now, return NaN, but might need physical interpretation.
        return np.nan

    p = p_numerator / p_denominator
    C = max(0, (3 * p - 1) / 2)
    return C

# --- Improved self-consistent calculation with robust solver ---
def compute_self_consistent_improved(V_SO, U, n, Nk, Delta_t, initial_guess=None, verbose=False):
    """
    Self-consistent solution for the superconducting gap equations.
    Modified to better handle negative U values.
    """
    t = 0.1
    kx = np.linspace(-np.pi, np.pi, Nk)
    ky = np.linspace(-np.pi, np.pi, Nk)
    KX, KY = np.meshgrid(kx, ky)
    sqrt_sin = np.sqrt(np.sin(KX)**2 + np.sin(KY)**2)
    cos_sum = np.cos(KX) + np.cos(KY)
    epsilon_ks = np.stack([
        -2 * t * cos_sum - 2 * V_SO * sqrt_sin,
        -2 * t * cos_sum + 2 * V_SO * sqrt_sin
    ], axis=-1)
    s_k = 0.5 * (np.cos(KX) + np.cos(KY))
    norm_factor = 4 * Nk**2

    # For tracking
    final_sum_DS = None

    def residuals(x):
        nonlocal final_sum_DS
        Delta0, DeltaS, mu = x
        Delta_ks = Delta0 - (DeltaS / (4 * t)) * epsilon_ks
        E_ks_squared = Delta_ks**2 + (epsilon_ks - mu)**2
        E_ks = np.sqrt(np.maximum(E_ks_squared, 1e-12)) # Avoid sqrt(0) or negative
        inv_E_ks = np.zeros_like(E_ks)
        non_zero_mask = E_ks > 1e-12 # Use a small tolerance
        inv_E_ks[non_zero_mask] = 1.0 / E_ks[non_zero_mask]

        term_n = (epsilon_ks - mu) * inv_E_ks
        rhs = 1 - (1 / (2 * Nk**2)) * np.sum(term_n)
        eq_n = rhs - n

        sum_D0 = 0.0
        sum_DS = 0.0
        for s in [0, 1]:
            s_prime = 1 if s == 0 else -1
            # Modified weight calculation for negative U
            # The key term U is now handled differently - using abs(U) but maintaining sign
            if U < 0:
                # For negative U (attractive interaction), adjust the weight calculation
                # This is the key change to handle negative U properly
                weight = U + 8 * Delta_t * s_k + 4 * (Delta_t / t) * V_SO * s_prime * sqrt_sin
                # Add stabilization term for negative U regime if needed
                # weight = weight - 0.1 * abs(U) * (s_k**2)  # Example stabilization
            else:
                # Original weight calculation for positive U
                weight = U + 8 * Delta_t * s_k + 4 * (Delta_t / t) * V_SO * s_prime * sqrt_sin

            term_gap = Delta_ks[..., s] * inv_E_ks[..., s]
            sum_D0 += np.sum(weight * term_gap)
            sum_DS += np.sum(term_gap)

        eq_D0 = Delta0 + sum_D0 / norm_factor
        eq_DS = DeltaS + 8 * Delta_t * sum_DS / norm_factor

        final_sum_DS = sum_DS

        return [eq_D0, eq_DS, eq_n]

    # Function for least_squares solver (returns array)
    def residuals_array(x):
        res = residuals(x)
        return np.array(res, dtype=float)

    # Cost function for minimize (sum of squares)
    def cost_function(x):
        res = residuals(x)
        return np.sum(np.array(res)**2)

    # Improved initial guess generation
    if initial_guess is None:
        # For negative U, we need different initial guesses
        if U < 0:
            # For attractive interaction (negative U),
            # Delta0 is expected to be positive regardless of sign of U
            delta0_guess = 0.1
            # DeltaS might need different scaling for negative U
            deltaS_guess = 0.05 * np.sign(U)  # Maintain sign proportional to U
            # Chemical potential might differ in negative U regime
            mu_guess = np.mean(epsilon_ks) - 0.1 * abs(U)  # Adjust based on U magnitude
        else:
            # Original initialization for positive U
            sgn = int(np.sign(U)) if U != 0 else 1
            delta0_guess = sgn * 0.2
            deltaS_guess = sgn * 0.2
            mu_guess = np.mean(epsilon_ks)

        x0 = [delta0_guess, deltaS_guess, mu_guess]
        if verbose: print(f"Generated initial guess: {x0}")
    else:
        x0 = initial_guess
        if verbose: print(f"Using provided initial guess: {x0}")

    # --- Solver Attempts ---
    sol = None
    success = False
    final_result_obj = None

    # 1. Try least_squares methods
    ls_methods = ['lm', 'trf', 'dogbox']
    for method in ls_methods:
        try:
            if verbose: print(f"Trying least_squares with method {method}...")
            result = least_squares(
                residuals_array,
                x0,
                method=method,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
                max_nfev=5000,
                verbose=0
            )
            if result.success and np.max(np.abs(result.fun)) < 1e-5:
                sol = result.x
                success = True
                final_result_obj = result
                if verbose: print(f"Converged with least_squares ({method})! Cost: {result.cost:.2e}, Residuals: {result.fun}")
                break
            elif verbose:
                 print(f"least_squares ({method}) finished but success flag is {result.success} or residuals too high: {result.fun}")
        except Exception as e:
            if verbose: print(f"least_squares method {method} failed with error: {str(e)}")

    # 2. If least_squares fails, try minimize methods
    if not success:
        # For negative U, try different methods first
        if U < 0:
            min_methods = ['Powell', 'Nelder-Mead', 'BFGS', 'CG']  # Reordered for negative U
        else:
            min_methods = ['Nelder-Mead', 'Powell', 'BFGS', 'CG']

        for method in min_methods:
            try:
                if verbose: print(f"Trying minimize with method {method}...")
                result = minimize(
                    cost_function,
                    x0,
                    method=method,
                    tol=1e-7,
                    options={'maxiter': 5000, 'disp': False}
                )
                if result.success and result.fun < 1e-8:
                    sol = result.x
                    success = True
                    final_result_obj = result
                    if verbose: print(f"Converged with minimize ({method})! Cost: {result.fun:.2e}")
                    break
                elif verbose:
                     print(f"minimize ({method}) finished but success flag is {result.success} or cost too high: {result.fun:.2e}")
            except Exception as e:
                if verbose: print(f"minimize method {method} failed with error: {str(e)}")

    # 3. Try basinhopping for global optimization
    if not success:
        if verbose: print("Trying basin-hopping global optimization...")
        try:
            # For negative U, use different minimizer and parameters
            if U < 0:
                minimizer_kwargs = {'method': 'Powell', "options": {"ftol": 1e-7}}
                T_param = 1.0  # Higher temperature for more exploration in negative U
                stepsize = 0.7  # Larger stepsize
            else:
                minimizer_kwargs = {'method': 'Nelder-Mead', "options": {"fatol": 1e-7}}
                T_param = 0.5
                stepsize = 0.5

            result = basinhopping(
                cost_function,
                x0,
                niter=50,
                T=T_param,
                stepsize=stepsize,
                minimizer_kwargs=minimizer_kwargs,
                disp=verbose
            )
            if result.fun < 1e-8:
                sol = result.x
                success = True
                final_result_obj = result
                if verbose: print(f"Converged with basin-hopping! Cost: {result.fun:.2e}")
            elif verbose:
                 print(f"Basin-hopping finished but final cost {result.fun:.2e} is too high.")

        except Exception as e:
            if verbose: print(f"Basin-hopping failed with error: {str(e)}")

    # --- Final Checks ---
    if not success or sol is None:
        if verbose:
            print(f"All optimization methods failed for V_SO={V_SO}, U={U}")
            if final_result_obj: print(f"Last attempt status: {final_result_obj.message}")
        return None, None, None, None, None, None

    # Check final residuals
    final_residuals = residuals_array(sol)
    if np.max(np.abs(final_residuals)) > 1e-5:
        if verbose:
            print(f"Warning: Solution found for V_SO={V_SO}, U={U}, but residuals are relatively large: {final_residuals}")
            print(f"Solver message: {final_result_obj.message if final_result_obj else 'N/A'}")

    # Calculate u_ks, v_ks with the converged solution
    Delta0, DeltaS, mu = sol
    Delta_ks = Delta0 - (DeltaS / (4 * t)) * epsilon_ks
    E_ks_squared = Delta_ks**2 + (epsilon_ks - mu)**2
    E_ks = np.sqrt(np.maximum(E_ks_squared, 1e-24))

    # Avoid division by zero
    E_ks_safe = np.maximum(E_ks, 1e-12)
    ratio = (epsilon_ks - mu) / E_ks_safe

    # Calculate u_ks and v_ks
    u_ks_arg = 0.5 * (1 + ratio)
    v_ks_arg = 0.5 * (1 - ratio)
    u_ks = np.sqrt(np.maximum(u_ks_arg, 0))
    v_ks = np.sqrt(np.maximum(v_ks_arg, 0))

    # Check normalization
    norm_check = u_ks**2 + v_ks**2
    if not np.allclose(norm_check, 1.0, atol=1e-4):
        if verbose:
            max_diff = np.max(np.abs(norm_check - 1.0))
            print(f"Warning: Normalization check failed for V_SO={V_SO}, U={U}. Max diff: {max_diff:.2e}")

    if verbose: print(f"Successful solution found for V_SO={V_SO}, U={U}: Delta0={Delta0:.4f}, DeltaS={DeltaS:.4f}, mu={mu:.4f}")

    # Recalculate final_sum_DS
    residuals(sol)

    return u_ks, v_ks, KX, KY, sol, final_sum_DS


# --- Improved helper for adaptive continuation method ---
def solve_with_continuation(U_target, V_SO, n, Nk, Delta_t,
                            initial_guess_sol=None,
                            start_U=None,
                            verbose=False,
                            continuation_steps=10,
                            max_recursion=2):
    """
    Improved continuation method that handles negative U better.
    """
    if initial_guess_sol is None or start_U is None:
        # Choose better starting point based on sign of U_target
        if U_target < 0:
            # For negative U, it might be better to start at a small negative value
            # rather than zero, to avoid crossing potentially problematic regions
            U_start_attempt = -0.1
        else:
            U_start_attempt = 0.0

        if verbose: print(f"Continuation: No start point provided. Attempting solve at U={U_start_attempt}...")
        u_ks, v_ks, KX, KY, sol, sum_DS = compute_self_consistent_improved(
            V_SO, U_start_attempt, n, Nk, Delta_t, initial_guess=None, verbose=verbose
        )
        if sol is None:
            if verbose: print(f"Continuation failed: Cannot solve even at starting U={U_start_attempt}")
            # Try additional backup starting points if primary fails
            if U_target < 0:
                backup_starts = [-0.01, -0.5, -1.0]
            else:
                backup_starts = [0.01, 0.5, 1.0]

            for backup_U in backup_starts:
                if verbose: print(f"Trying backup starting point U={backup_U}...")
                u_ks, v_ks, KX, KY, sol, sum_DS = compute_self_consistent_improved(
                    V_SO, backup_U, n, Nk, Delta_t, initial_guess=None, verbose=verbose
                )
                if sol is not None:
                    start_U = backup_U
                    current_guess = sol
                    break
            else:  # No backup worked
                return None, None, None, None, None, None
        else:
            start_U = U_start_attempt
            current_guess = sol
    else:
        start_U = start_U
        current_guess = initial_guess_sol

    if verbose: print(f"Continuation: Starting from U={start_U} towards U_target={U_target}")

    # Create adaptive path from starting U to target U
    # For negative U, use more points in regions where convergence might be difficult
    if (start_U < 0 and U_target < 0) or (start_U > 0 and U_target > 0):
        # Same sign - linear spacing should work
        U_path = np.linspace(start_U, U_target, continuation_steps + 1)[1:]
    elif start_U == 0 or U_target == 0:
        # Zero crossing - use denser spacing near zero
        if start_U < U_target:  # Negative to positive
            mid = np.linspace(start_U, 0, continuation_steps//2 + 1)[1:]
            end = np.linspace(0, U_target, continuation_steps//2 + 1)[1:]
            U_path = np.concatenate([mid, end])
        else:  # Positive to negative
            mid = np.linspace(start_U, 0, continuation_steps//2 + 1)[1:]
            end = np.linspace(0, U_target, continuation_steps//2 + 1)[1:]
            U_path = np.concatenate([mid, end])
    else:
        # Sign change - be even more careful
        # Create more points near zero and in the negative region
        if start_U < U_target:  # Negative to positive
            before_zero = np.linspace(start_U, -0.01, continuation_steps//3 + 1)[1:]
            near_zero = np.linspace(-0.01, 0.01, continuation_steps//3 + 1)[1:]
            after_zero = np.linspace(0.01, U_target, continuation_steps//3 + 1)[1:]
            U_path = np.concatenate([before_zero, near_zero, after_zero])
        else:  # Positive to negative
            before_zero = np.linspace(start_U, 0.01, continuation_steps//3 + 1)[1:]
            near_zero = np.linspace(0.01, -0.01, continuation_steps//3 + 1)[1:]
            after_zero = np.linspace(-0.01, U_target, continuation_steps//3 + 1)[1:]
            U_path = np.concatenate([before_zero, near_zero, after_zero])

    # Step through the path
    for step, U_step in enumerate(U_path):
        if verbose:
            print(f"Continuation step {step+1}/{len(U_path)}: Stepping from U={start_U:.6f} -> U={U_step:.6f}")

        u_ks, v_ks, KX, KY, sol, sum_DS = compute_self_consistent_improved(
            V_SO, U_step, n, Nk, Delta_t, initial_guess=current_guess, verbose=False
        )

        if sol is None:
            if verbose:
                print(f"Continuation failed at U={U_step:.6f} (step {step+1}).")
            # Try recursion with smaller steps
            if continuation_steps < 10 * (2**max_recursion) and step > 0:
                new_steps = continuation_steps * 2
                if verbose:
                    print(f"Attempting finer continuation ({new_steps} steps) from last good point U={start_U:.6f}...")
                return solve_with_continuation(
                    U_target, V_SO, n, Nk, Delta_t,
                    initial_guess_sol=current_guess,
                    start_U=start_U,
                    verbose=verbose,
                    continuation_steps=new_steps,
                    max_recursion=max_recursion - 1
                )
            else:
                 if verbose: print("Max recursion depth reached or failed at first step. Aborting continuation.")
                 return None, None, None, None, None, None

        # Update for next step
        current_guess = sol
        start_U = U_step

    if verbose: print(f"Continuation successful: Reached target U={U_target}")
    return u_ks, v_ks, KX, KY, sol, sum_DS


# Smoothing function (Unchanged)
def smooth_data(x_vals, y_vals, window_size=3):
    """Smooth data with missing values (NaNs) using a rolling average after interpolation."""
    y_vals = np.array(y_vals, dtype=float)
    valid_indices = ~np.isnan(y_vals)

    if np.sum(valid_indices) <= 1:
        return y_vals

    x_valid = x_vals[valid_indices]
    y_valid = y_vals[valid_indices]

    try:
        interp_func = interp1d(x_valid, y_valid, kind='linear', bounds_error=False, fill_value="extrapolate")
        y_interp = interp_func(x_vals)

        smoothed = np.copy(y_interp)
        half_window = window_size // 2
        for i in range(len(y_interp)):
            start = max(0, i - half_window)
            end = min(len(y_interp), i + half_window + 1)
            window_indices = np.arange(start, end)
            valid_in_window = ~np.isnan(y_interp[window_indices])
            if np.any(valid_in_window):
                 smoothed[i] = np.nanmean(y_interp[window_indices[valid_in_window]])
            else:
                 smoothed[i] = np.nan

        smoothed[~valid_indices] = np.nan
        return smoothed

    except ValueError as e:
        print(f"Smoothing interpolation failed: {e}. Returning original data.")
        return y_vals


# Numerical derivative (Unchanged)
def compute_numerical_derivative(x_vals, y_vals, window_size=5):
    """Compute numerical derivative using centered difference on smoothed data."""
    if np.sum(~np.isnan(y_vals)) <= 2:
        return np.full_like(y_vals, np.nan)

    y_smoothed = smooth_data(x_vals, y_vals, window_size=window_size)
    deriv = np.full_like(y_vals, np.nan)

    for i in range(1, len(x_vals) - 1):
        if not (np.isnan(y_smoothed[i-1]) or np.isnan(y_smoothed[i+1]) or np.isnan(y_smoothed[i])):
            dx = x_vals[i+1] - x_vals[i-1]
            dy = y_smoothed[i+1] - y_smoothed[i-1]
            if abs(dx) > 1e-12:
                deriv[i] = dy / dx
            else:
                deriv[i] = np.nan

    if len(x_vals) > 1 and not (np.isnan(y_smoothed[0]) or np.isnan(y_smoothed[1])):
         dx = x_vals[1] - x_vals[0]
         dy = y_smoothed[1] - y_smoothed[0]
         if abs(dx) > 1e-12: deriv[0] = dy / dx

    if len(x_vals) > 1 and not (np.isnan(y_smoothed[-1]) or np.isnan(y_smoothed[-2])):
         dx = x_vals[-1] - x_vals[-2]
         dy = y_smoothed[-1] - y_smoothed[-2]
         if abs(dx) > 1e-12: deriv[-1] = dy / dx

    return deriv



# --- Main Loop Modified for Both Positive and Negative U ---
if __name__ == "__main__":
    # Parameters - Include both positive and negative U values
    # Modified to include both positive and negative U with special attention to transitions
    #U_negative = np.linspace(-3, -0.099, 40)  # More points in negative region
    #U_near_zero = np.linspace(-0.1, 0.1, 41)  # Dense sampling near zero
    U_positive = np.linspace(0.0, 15, 150)     # Positive region
    U_vals = np.concatenate([ U_positive])

    fixed_params = {
        'V_SO': 0.0,      # Fixed Spin-Orbit coupling
        'Delta_t': 0.45,
        'n': 1.875,
        'Nk': 600,
    }
    r_fixed = 0.2
    theta_fixed = 0.0

    C_vals = np.full(len(U_vals), np.nan)
    dC_dU_vals = np.full(len(U_vals), np.nan)

    # Store solutions for reuse in continuation - separate dictionaries for positive and negative U
    pos_solutions = {}  # For positive U
    neg_solutions = {}  # For negative U

    # Progress tracking
    start_time = time.time()
    successful_points = 0
    last_successful_U = None
    last_sol = None

    for i, u_val in enumerate(U_vals):
        print(f"\n--- Processing U = {u_val:.6f} [{i+1}/{len(U_vals)}] ---")

        # Try to choose a good initial guess based on previously solved points
        current_initial_guess = None
        closest_U = None
        closest_sol_for_cont = None

        # Find the closest solved point with the same sign
        if u_val >= 0:
            if pos_solutions:
                solved_Us = np.array(list(pos_solutions.keys()))
                if len(solved_Us) > 0:
                    closest_idx = np.argmin(np.abs(solved_Us - u_val))
                    closest_U = solved_Us[closest_idx]
                    current_initial_guess = pos_solutions[closest_U]
                    print(f"Using solution from U={closest_U:.6f} as initial guess.")
        else:  # u_val < 0
            if neg_solutions:
                solved_Us = np.array(list(neg_solutions.keys()))
                if len(solved_Us) > 0:
                    closest_idx = np.argmin(np.abs(solved_Us - u_val))
                    closest_U = solved_Us[closest_idx]
                    current_initial_guess = neg_solutions[closest_U]
                    print(f"Using solution from U={closest_U:.6f} as initial guess.")

        # If no suitable previous solution was found, use the last successful one
        # (even if it had different sign, but only if close enough)
        if current_initial_guess is None and last_sol is not None:
            if last_successful_U is not None and abs(u_val - last_successful_U) < 2.0:
                print(f"Using last successful solution from U={last_successful_U:.6f} as initial guess.")
                current_initial_guess = last_sol
                closest_U = last_successful_U
                closest_sol_for_cont = last_sol

        # Try direct solution first
        u_ks, v_ks, KX, KY, sol, _ = compute_self_consistent_improved(
            fixed_params['V_SO'], u_val, fixed_params['n'], fixed_params['Nk'],
            fixed_params['Delta_t'], initial_guess=current_initial_guess, verbose=True
        )

        # If direct solution fails, try continuation method
        if sol is None:
            print(f"Direct solution failed for U={u_val:.6f}. Trying continuation...")

            # Find the closest successfully solved point to use as starting point
            same_sign_dict = pos_solutions if u_val >= 0 else neg_solutions
            if same_sign_dict:
                solved_Us = np.array(list(same_sign_dict.keys()))
                closest_idx = np.argmin(np.abs(solved_Us - u_val))
                closest_U = solved_Us[closest_idx]
                closest_sol_for_cont = same_sign_dict[closest_U]
                print(f"Continuation starting from U={closest_U:.6f}")
            elif last_successful_U is not None and abs(u_val - last_successful_U) < 5.0:
                # Fall back to last successful solution if it's not too far
                closest_U = last_successful_U
                closest_sol_for_cont = last_sol
                print(f"Continuation starting from last successful U={closest_U:.6f}")
            else:
                print(f"No suitable starting point for continuation at U={u_val:.6f}. Skipping.")
                continue

            # Apply continuation method
            u_ks, v_ks, KX, KY, sol, _ = solve_with_continuation(
                u_val, fixed_params['V_SO'], fixed_params['n'], fixed_params['Nk'],
                fixed_params['Delta_t'],
                initial_guess_sol=closest_sol_for_cont,
                start_U=closest_U,
                verbose=True,
                continuation_steps=10,
                max_recursion=2
            )

        # Process the solution if successful
        if sol is not None:
            successful_points += 1
            last_successful_U = u_val
            last_sol = sol

            # Store solution in appropriate dictionary
            if u_val >= 0:
                pos_solutions[u_val] = sol
            else:
                neg_solutions[u_val] = sol

            # Compute Green's functions and concurrence
            try:
                G, F = compute_greens_functions(u_ks, v_ks, KX, KY, r_fixed, theta_fixed)
                G0 = compute_greens_functions(u_ks, v_ks, KX, KY, 0.0, 0.0)[0]
                C = compute_concurrence(G, F, G0)
                C_vals[i] = C
                print(f"Concurrence C = {C:.6f} for U = {u_val:.6f}")
            except Exception as e:
                print(f"Error computing concurrence for U={u_val:.6f}: {str(e)}")
                C_vals[i] = np.nan

        else:
            print(f"Failed to converge for U={u_val:.6f}")
            C_vals[i] = np.nan

    # Compute numerical derivative of concurrence
    dC_dU_vals = compute_numerical_derivative(U_vals, C_vals, window_size=5)

    # Plotting results
    plt.figure(figsize=(12, 6))

    # Plot Concurrence
    plt.subplot(1, 2, 1)
    plt.plot(U_vals, C_vals, 'b-', label='Concurrence')
    plt.scatter(U_vals, C_vals, color='blue', s=10)
    plt.xlabel('U')
    plt.ylabel('Concurrence')
    plt.title(f'Concurrence vs U (V_SO={fixed_params["V_SO"]}, n={fixed_params["n"]})')
    plt.grid(True)
    plt.legend()

    # Plot Derivative
    plt.subplot(1, 2, 2)
    plt.plot(U_vals, dC_dU_vals, 'r-', label='dC/dU')
    plt.scatter(U_vals, dC_dU_vals, color='red', s=10)
    plt.xlabel('U')
    plt.ylabel('dC/dU')
    plt.title(f'Derivative of Concurrence vs U')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig("concurrence_vs_U.png", dpi=75, bbox_inches='tight')

    plt.show()

    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\n--- Summary ---")
    print(f"Total points attempted: {len(U_vals)}")
    print(f"Successful points: {successful_points} ({successful_points/len(U_vals)*100:.1f}%)")
    print(f"Total computation time: {elapsed_time:.2f} seconds")