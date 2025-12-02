import numpy as np
import matplotlib.pyplot as plt
from Greens_function import compute_greens_functions
from selfconsistent_equations import solve_self_consistent



u_ks, v_ks, Delta_ks, E_ks, mu, KX, KY, epsilon_ks = solve_self_consistent(
    Nk=200,
    t=0.1,
    U=45*0.1,
    Delta_t=0.45,
    V_SO=0.0,
    n=1.875
)

kx = np.linspace(-np.pi, np.pi, 200)
ky = np.linspace(-np.pi, np.pi, 200)

r_vals, theta_vals, G, F = compute_greens_functions(    u_ks, v_ks, KX, KY, kx, ky,
    r_max=10,
    Nr=200
)

C_polar = np.zeros((len(theta_vals), len(r_vals)))

G0 = np.abs(G[:, 0])  # G(0) for normalization


for i, theta in enumerate(theta_vals):
    g = (G[i, :]).imag / G0[i]
    f = (F[i, :]).imag / G0[i]

    p = (f**2 + g**2) / (2 + f**2 - g**2)
    plt.plot(r_vals, p, label=f"θ = {np.degrees(theta):.0f}°")
    C_polar[i, :] = np.maximum(0, (3*p - 1)/2)
    #print(C_polar)
    #print(g[0:20])
    #print(C_polar[i, :])

plt.xlabel("r")
plt.ylabel("p(r, θ)")
plt.legend()
plt.grid(True)
plt.title("p values vs r for fixed θ")
plt.show()

# Plot concurrence
plt.figure(figsize=(8, 6))
for i, theta in enumerate(theta_vals):
    plt.plot(r_vals, C_polar[i, :], label=f"θ = {np.degrees(theta):.0f}°")

plt.xlabel("r")
plt.ylabel("Concurrence C(r, θ)")
plt.legend()
plt.grid(True)
plt.title("Concurrence vs r for fixed θ")
plt.show()