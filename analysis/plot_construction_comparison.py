"""Isolate WHY the legacy V_SO=0 figures looked richer than the new ones.

Three panels, same observable, same U/t=75, n=1.875, T=0:
  (a) signed helicity construction at V_SO=0
  (b) signed helicity construction at V_SO=0.5
  (c) legacy unsigned X-state construction at V_SO=0   <-- the old figures

If (a) ~= (b) but (c) is rich, the visible structure of the old figures came
from the construction (unsigned amplitudes + scalar X-state threshold), not
from V_SO=0 itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from regenerate_vso_figures import States, observables, r_vec, solve  # noqa: E402
from reproduce_and_validate import solve_and_build  # noqa: E402
from observables import concurrence_xstate, singlet_fraction  # noqa: E402

R_C = np.linspace(0.0, 4.0, 161)
R_F = np.linspace(0.0, 6.0, 121)


def signed_curves(vso: float):
    st = States(solve(vso, 75.0))
    C = observables(np.stack([st.rho(r_vec(x, 0.0)) for x in R_C]))["C"]
    Fs = observables(np.stack([st.rho(r_vec(x, 45.0)) for x in R_F]))["Fs"]
    return np.asarray(C), np.asarray(Fs)


def unsigned_curves():
    _, r_u, th_u, _, _, rho0, _, _ = solve_and_build(75.0)
    i0 = int(np.argmin(np.abs(th_u - 0.0)))
    i45 = int(np.argmin(np.abs(th_u - np.pi / 4)))
    C = np.array([concurrence_xstate(rho0[i0, j]) for j in range(len(r_u))])
    Fs = np.array([singlet_fraction(rho0[i45, j]) for j in range(len(r_u))])
    return r_u, C, Fs


print("[1/3] signed V_SO=0 ...", flush=True)
C_a, F_a = signed_curves(0.0)
print("[2/3] signed V_SO=0.5 ...", flush=True)
C_b, F_b = signed_curves(0.5)
print("[3/3] legacy unsigned V_SO=0 ...", flush=True)
r_u, C_c, F_c = unsigned_curves()


def rev(r, C, lo=1.0, hi=4.0):
    m = (r >= lo) & (r <= hi)
    return float(C[m].max()) if m.any() else 0.0


panels = [
    (R_C, C_a, R_F, F_a,
     "signed helicity,\n$V_{SO}=0$", rev(R_C, C_a)),
    (R_C, C_b, R_F, F_b,
     "signed helicity,\n$V_{SO}=0.5$", rev(R_C, C_b)),
    (r_u, C_c, r_u, F_c,
     "legacy unsigned X-state,\n$V_{SO}=0$ (old figures)", rev(r_u, C_c)),
]

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), constrained_layout=True)
for col, (rc, Cc, rf, ff, ttl, revmax) in enumerate(panels):
    ax = axes[0, col]
    ax.plot(rc, Cc, color="tab:red", lw=2.0)
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.set_title(ttl, fontsize=11)
    ax.annotate(rf"max $C$ on $1\leq r\leq4$: {revmax:.3f}",
                xy=(0.03, 0.78), xycoords="axes fraction", fontsize=10)
    if col == 0:
        ax.set_ylabel(r"$C(r,\ \theta=0^\circ,\ t=0)$", fontsize=11)

    ax = axes[1, col]
    ax.plot(rf, ff, color="tab:blue", lw=2.0)
    ax.axhline(0.25, color="0.5", ls=":", lw=1)
    ax.set_xlim(0, 6)
    ax.grid(alpha=0.3)
    exc = float(np.maximum(ff[rf > 1.0] - 0.25, 0).max())
    ax.annotate(rf"max$(F_s-1/4)$ for $r>1$: {exc:.3f}",
                xy=(0.03, 0.20), xycoords="axes fraction", fontsize=10)
    if col == 0:
        ax.set_ylabel(r"$F_s(r,\ \theta=45^\circ,\ t=0)$", fontsize=11)
    ax.set_xlabel(r"$r$ (lattice units)", fontsize=11)

fig.suptitle(
    r"Construction, not $V_{SO}$, creates the visible structure:"
    r" signed helicity at $V_{SO}=0$ and $0.5$ look alike;"
    r" the legacy unsigned X-state at $V_{SO}=0$ (old figures) shows"
    r" revival windows and large amplitude modulation"
    r" ($U/t=75$, $\Delta_t/t=4.5$)", fontsize=12.5)
out = ROOT / "figs" / "construction_vs_vso.png"
fig.savefig(out, dpi=170, bbox_inches="tight")
print(f"saved {out}", flush=True)
print("revival max C (1<=r<=4): signed V=0: %.4f | signed V=0.5: %.4f | "
      "unsigned V=0: %.4f" % (rev(R_C, C_a), rev(R_C, C_b), rev(r_u, C_c)))
