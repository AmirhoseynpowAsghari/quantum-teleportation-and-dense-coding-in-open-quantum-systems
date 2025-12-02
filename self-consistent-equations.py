import numpy as np
from scipy.optimize import bisect

def solve_self_consistent(Nk=200, t=0.1, U=4.5, Delta_t=0.45, V_SO=0.0, n=1.875,
                          tol=1e-8, max_iter=5000, lambda_damp=0.1):
    """
    Returns:
        u_ks, v_ks, Delta_ks, E_ks, mu, KX, KY, epsilon_ks
    """

    # ---- Same code as before, but NO plotting, NO print unless needed ----

    kx = np.linspace(-np.pi, np.pi, Nk)
    ky = np.linspace(-np.pi, np.pi, Nk)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')

    sqrt_sin = np.sqrt(np.sin(KX)**2 + np.sin(KY)**2)
    cos_sum = np.cos(KX) + np.cos(KY)

    epsilon_ks = np.zeros((Nk, Nk, 2))
    epsilon_ks[..., 0] = -2*t*cos_sum - 2*V_SO*sqrt_sin
    epsilon_ks[..., 1] = -2*t*cos_sum + 2*V_SO*sqrt_sin

    rng = np.random.default_rng(123)
    Delta0 = 0.8 + 0.1*rng.random()
    DeltaS = 0.8 + 0.1*rng.random()
    mu = float(np.mean(epsilon_ks))

    s_k = 0.5 * (np.cos(KX) + np.cos(KY))
    s_prime_map = np.array([+1.0, -1.0])

    for it in range(max_iter):

        Delta_ks = Delta0 - (DeltaS/(4*t)) * epsilon_ks

        def number_eq(mu_trial):
            E_trial = np.sqrt(Delta_ks**2 + (epsilon_ks - mu_trial)**2)
            E_trial = np.clip(E_trial, 1e-12, None)
            density = 1 - (1/(2*Nk**2))*np.sum((epsilon_ks - mu_trial)/E_trial)
            return float(density - n)

        eps_min = float(np.min(epsilon_ks))
        eps_max = float(np.max(epsilon_ks))
        mu_low = eps_min - 10*t
        mu_high = eps_max + 10*t

        mu = bisect(number_eq, mu_low, mu_high, xtol=1e-6)

        E_ks = np.sqrt(Delta_ks**2 + (epsilon_ks - mu)**2)
        E_ks = np.clip(E_ks, 1e-12, None)

        term0 = U + 8*Delta_t*s_k
        term_sp = 4 * (Delta_t/t) * V_SO * sqrt_sin

        pref = np.zeros_like(epsilon_ks)
        pref[..., 0] = term0 + term_sp * s_prime_map[0]
        pref[..., 1] = term0 + term_sp * s_prime_map[1]

        ratio = Delta_ks / E_ks
        sum_D0 = 0.5 * np.sum(pref * ratio) / (2*Nk**2)
        sum_DS = 0.5 * np.sum(ratio) / (2*Nk**2)

        Delta0_new = -sum_D0 * 2
        DeltaS_new = -8*Delta_t * sum_DS

        # damping
        Delta0 = (1-lambda_damp)*Delta0 + lambda_damp*Delta0_new
        DeltaS = (1-lambda_damp)*DeltaS + lambda_damp*DeltaS_new

        if abs(Delta0_new-Delta0) < tol and abs(DeltaS_new-DeltaS) < tol:
            break

    # final objects
    Delta_ks = Delta0 - (DeltaS/(4*t)) * epsilon_ks
    E_ks = np.sqrt(Delta_ks**2 + (epsilon_ks - mu)**2)
    E_ks = np.clip(E_ks, 1e-12, None)

    u_ks = np.sqrt(0.5 * (1 + (epsilon_ks - mu)/E_ks))
    v_ks = np.sqrt(0.5 * (1 - (epsilon_ks - mu)/E_ks))

    return u_ks, v_ks, Delta_ks, E_ks, mu, KX, KY, epsilon_ks

if __name__ == "__main__":
    u, v, D, E, mu, KX, KY, eps = solve_self_consistent()
    print("Self-consistency complete. mu =", mu)
