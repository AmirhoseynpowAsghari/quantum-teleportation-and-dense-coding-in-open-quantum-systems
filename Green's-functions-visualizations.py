import numpy as np
import matplotlib.pyplot as plt
from selfconsistent_equations import solve_self_consistent
from Greens_function import compute_greens_functions

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

# -----------------------
# Step 2: compute G(r,θ), F(r,θ)
# -----------------------
r_vals, theta_vals, G_polar, F_polar = compute_greens_functions(
    u_ks, v_ks, KX, KY, kx, ky,
    r_max=10,
    Nr=200
)


# -----------------------
# Step 3: plot results
# -----------------------

plt.figure(figsize=(8, 6))
for i, theta in enumerate(theta_vals):
    plt.plot(r_vals, (G_polar[i, :]).imag, label=f"θ = {np.degrees(theta):.0f}°")
plt.xlabel("r")
plt.ylabel("$G_{uu}(r, θ)$")
plt.legend()
plt.grid(True)
plt.title("Normal Green's Function in Polar Coordinates")

plt.savefig("G_polar.png", dpi=75, bbox_inches='tight')
plt.show()


plt.figure(figsize=(8, 6))
for i, theta in enumerate(theta_vals):
    plt.plot(r_vals, (F_polar[i, :]).imag, label=f"θ = {np.degrees(theta):.0f}°")
plt.xlabel("r")
plt.ylabel("$F_{ud}(r, θ)$")
plt.legend()
plt.grid(True)
plt.title("Anomalous Green's Function in Polar Coordinates")
plt.savefig("F_polar.png", dpi=75, bbox_inches='tight')
plt.show()
