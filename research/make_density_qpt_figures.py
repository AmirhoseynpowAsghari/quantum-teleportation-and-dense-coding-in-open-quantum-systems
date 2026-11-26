"""Generate curated CSVs and publication figures (fig13, fig14) for Task 8:
Density (n) QPT analysis comparing arXiv:1801.08988 Eq. (28) vs. Concurrence.
"""
from __future__ import annotations

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CSV_DIR = os.path.join(DATA_DIR, "csv")
FIG_DIR = os.path.join(HERE, "figs")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.dpi": 180,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
})


def save_fig(fig, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(FIG_DIR, f"{stem}.{ext}"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.{{png,pdf,svg}}", flush=True)


def export_csvs(d: dict) -> None:
    # 1. Full sweeps CSV
    p1 = os.path.join(CSV_DIR, "task8_density_full_sweeps.csv")
    cols1 = [
        "sweep", "U", "V_SO", "Nk", "n", "sc", "lam_cont", "D0", "DS", "D_FS",
        "E_min", "muN", "muS", "eN", "eS", "dF", "d2eN_dn2", "d2eS_dn2",
        "C_spin_025", "CN_spin_025", "C0_spin_025", "dCS_spin_025", "d2CS_spin_025",
        "C_band_025", "CN_band_025",
        "C_spin_050", "C_raw_spin_050", "C_spin_075", "C_spin_100", "C0_spin_100", "C_spin_050_050",
    ]
    n_rows1 = 0
    with open(p1, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols1)
        for name, sw in d["full_sweeps"].items():
            dv = sw["derivs"]
            cs_d = dv["concurrence"]["spin"]["0.25_0"]
            for i, r in enumerate(sw["rows"]):
                ms = r["sc_metrics"]["spin"]
                mn = r["norm_metrics"]["spin"]
                mb = r["sc_metrics"]["band"]
                mnb = r["norm_metrics"]["band"]
                w.writerow([
                    name, sw["U"], sw["V_SO"], sw["Nk"], r["n"], int(r["sc"]),
                    r["lam_cont"], r["D0"], r["DS"], r["D_FS"], r["E_min"],
                    r["muN"], r["muS"], r["eN"], r["eS"], r["dF"],
                    dv["d2eN_dn2"][i], dv["d2eS_dn2"][i],
                    ms["0.25_0"]["C"], mn["0.25_0"]["C"], ms["0.25_0"]["C_normonly"],
                    cs_d["dCS"][i], cs_d["d2CS"][i],
                    mb["0.25_0"]["C"], mnb["0.25_0"]["C"],
                    ms["0.5_0"]["C"], ms["0.5_0"]["C_raw"],
                    ms["0.75_0"]["C"], ms["1_0"]["C"], ms["1_0"]["C_normonly"],
                    ms["0.5_0.5"]["C"],
                ])
                n_rows1 += 1

    # 2. Nk=801 zoom CSV
    p2 = os.path.join(CSV_DIR, "task8_density_nk801_zooms.csv")
    cols2 = [
        "zoom", "U", "V_SO", "T_norm", "Nk", "n", "sc", "lam_cont", "D0", "DS", "D_FS",
        "muN", "muS", "eN", "eS", "dF", "d2eN_dn2", "d2eS_dn2", "d3eS_dn3",
        "C_spin_025", "CN_spin_025", "dCS_spin_025", "d2CS_spin_025",
        "dCN_spin_025", "d2CN_spin_025", "C_band_025", "dCS_band_025", "d2CS_band_025",
    ]
    n_rows2 = 0
    with open(p2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols2)
        for name, sw in d["zooms"].items():
            dv = sw["derivs"]
            cs = dv["concurrence"]["spin"]["0.25_0"]
            cb = dv["concurrence"]["band"]["0.25_0"]
            for i, r in enumerate(sw["rows"]):
                ms = r["sc_metrics"]["spin"]["0.25_0"]
                mn = r["norm_metrics"]["spin"]["0.25_0"]
                mb = r["sc_metrics"]["band"]["0.25_0"]
                w.writerow([
                    name, sw["U"], sw["V_SO"], sw["T_norm"], sw["Nk"], r["n"], int(r["sc"]),
                    r["lam_cont"], r["D0"], r["DS"], r["D_FS"],
                    r["muN"], r["muS"], r["eN"], r["eS"], r["dF"],
                    dv["d2eN_dn2"][i], dv["d2eS_dn2"][i], dv["d3eS_dn3"][i],
                    ms["C"], mn["C"], cs["dCS"][i], cs["d2CS"][i],
                    cs["dCN"][i], cs["d2CN"][i], mb["C"], cb["dCS"][i], cb["d2CS"][i],
                ])
                n_rows2 += 1
    print(f"Exported {p1} ({n_rows1} rows) and {p2} ({n_rows2} rows)", flush=True)


def make_fig13(d: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.6))
    s75 = d["full_sweeps"]["full_U75_V0.5_Nk401"]
    s115 = d["full_sweeps"]["full_U115_V0.5_Nk401"]
    s0 = d["full_sweeps"]["full_U75_V0.0_Nk401"]

    def arr(sw, key):
        return np.array([r[key] for r in sw["rows"]])

    n75, n115, n0 = arr(s75, "n"), arr(s115, "n"), arr(s0, "n")

    # (a) Pairing instability lambda_+(n)
    ax = axes[0, 0]
    ax.plot(n75, np.clip(arr(s75, "lam_cont"), 0, 4.0), color="#1f77b4", lw=2.0,
            label=r"$U=75, V_{\mathrm{SO}}=0.5$ ($n_{c,1}=0.219, n_{c,2}=1.019$)")
    ax.plot(n115, np.clip(arr(s115, "lam_cont"), 0, 4.0), color="#d62728", lw=2.0,
            label=r"$U=115, V_{\mathrm{SO}}=0.5$ ($n_c=1.460$)")
    ax.plot(n0, np.clip(arr(s0, "lam_cont"), 0, 4.0), color="#2ca02c", lw=1.8, ls="--",
            label=r"$U=75, V_{\mathrm{SO}}=0.0$ ($n_{c,1}=0.125, n_{c,2}=1.197$)")
    ax.axhline(1.0, color="k", lw=1.1, ls=":")
    ax.axvspan(0.0, 0.2192, color="#1f77b4", alpha=0.08)
    ax.axvspan(1.0194, 2.0, color="#1f77b4", alpha=0.06)
    ax.set_xlim(0, 1.98)
    ax.set_ylim(0.55, 2.8)
    ax.set_xlabel(r"Electron density $n$")
    ax.set_ylabel(r"Linearized eigenvalue $\lambda_+(n)$")
    ax.set_title(r"(a) Exact Pairing Instability $\lambda_+(n)$ & Re-entrant Pocket")
    ax.legend(loc="upper left", fontsize=7.3)
    ax.text(0.05, 1.32, "Re-entrant\nnodal SC\n$n<0.219$", fontsize=7.5, color="#1f77b4")

    # (b) Eq. (28) canonical energy & condensation energy dF
    ax = axes[0, 1]
    eN75, eS75 = arr(s75, "eN"), arr(s75, "eS")
    dF75 = arr(s75, "dF")
    dF115 = arr(s115, "dF")
    dF0 = arr(s0, "dF")
    ax.plot(n75, eS75, color="#1f77b4", lw=2.0, label=r"Eq. (28) $e_{\mathrm{GS}}(n)$ ($U=75, V=0.5$)")
    ax.plot(n75, eN75, color="0.45", lw=1.3, ls="--", label=r"Normal $e_N(n)$")
    ax.set_xlim(0, 1.98)
    ax.set_xlabel(r"Electron density $n$")
    ax.set_ylabel(r"$e_{\mathrm{GS}}(n) = E_g/N + \mu n$ [$t$]")
    ax.set_title(r"(b) Eq. (28) Ground-State Energy & Condensation $-\Delta F$")
    ax2 = ax.twinx()
    ax2.plot(n75, -dF75 * 1e3, color="#1f77b4", lw=1.4, ls="-.", alpha=0.85, label=r"$-\Delta F$ ($U=75,V=0.5$)")
    ax2.plot(n115, -dF115 * 1e3, color="#d62728", lw=1.4, ls="-.", alpha=0.85, label=r"$-\Delta F$ ($U=115,V=0.5$)")
    ax2.plot(n0, -dF0 * 1e3, color="#2ca02c", lw=1.3, ls=":", alpha=0.85, label=r"$-\Delta F$ ($U=75,V=0$)")
    ax2.set_ylabel(r"Condensation $-\Delta F$ [$10^{-3} t$]", color="#d62728")
    ax2.tick_params(axis="y", colors="#d62728")
    ax2.grid(False)
    ax.legend(loc="upper left", fontsize=7.5)
    ax2.legend(loc="center left", bbox_to_anchor=(0.02, 0.42), fontsize=7.2)

    # (c) Eq. (28) inverse compressibility d^2 e_GS / dn^2 = dmu / dn
    ax = axes[0, 2]
    m75 = n75 >= 0.50
    m115 = n115 >= 0.50
    m0 = n0 >= 0.50
    ax.plot(n115[m115], np.array(s115["derivs"]["d2eS_dn2"])[m115], color="#d62728", lw=1.8,
            label=r"$U=115, V=0.5$ ($n_{\mathrm{vH},\pm}$ normal)")
    ax.plot(n75[m75], np.array(s75["derivs"]["d2eS_dn2"])[m75], color="#1f77b4", lw=1.8,
            label=r"$U=75, V=0.5$ ($n_{\mathrm{vH},+}$ SC-smeared)")
    ax.plot(n0[m0], np.array(s0["derivs"]["d2eS_dn2"])[m0], color="#2ca02c", lw=1.5, ls="--",
            label=r"$U=75, V=0.0$ ($n_{\mathrm{vH}}=1.0$)")
    for nv, lbl, col in [(0.9065, r"$n_{\mathrm{vH},-}$", "0.35"),
                         (1.0194, r"$n_{c,2}$", "#1f77b4"),
                         (1.0935, r"$n_{\mathrm{vH},+}$", "#d62728"),
                         (1.4602, r"$n_c^{(115)}$", "#d62728")]:
        ax.axvline(nv, color=col, lw=0.9, ls=":")
    ax.set_xlim(0.55, 1.96)
    ax.set_ylim(1.0, 10.5)
    ax.set_xlabel(r"Electron density $n$")
    ax.set_ylabel(r"$d^2 e_{\mathrm{GS}}/dn^2 = d\mu/dn$ [$t$]")
    ax.set_title(r"(c) Eq. (28) Compressibility Inverse vs. $n$")
    ax.legend(loc="upper left", fontsize=7.3)

    # (d) Concurrence C(n) across separations (U=75, V=0.5)
    ax = axes[1, 0]
    r_styles = [
        ("0.25_0", r"$r=(0.25,0)a$", "#1f77b4", "-"),
        ("0.35_0", r"$r=(0.35,0)a$", "#ff7f0e", "-"),
        ("0.5_0", r"$r=(0.50,0)a$", "#2ca02c", "-"),
        ("0.75_0", r"$r=(0.75,0)a$", "#9467bd", "--"),
        ("1_0", r"$r=(1.00,0)a$", "#d62728", "-."),
        ("0.5_0.5", r"$r=(0.5,0.5)a$", "#8c564b", ":"),
    ]
    for rk, lbl, col, ls in r_styles:
        c_sc = np.array([r["sc_metrics"]["spin"][rk]["C"] for r in s75["rows"]])
        ax.plot(n75, c_sc, color=col, ls=ls, lw=1.7, label=lbl)
    # mark ESD arrows
    for n_esd, xt, txt in [(0.416, 0.34, "ESD(1,0)"), (0.737, 0.68, "ESD(.75)"), (1.537, 1.54, "ESD(.50)")]:
        ax.annotate(txt, xy=(n_esd, 0.0), xytext=(xt, 0.14),
                    ha="center", fontsize=7.0,
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
    ax.set_xlim(0, 1.98)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel(r"Electron density $n$")
    ax.set_ylabel(r"Spin concurrence $C_{\mathrm{SC}}(n)$")
    ax.set_title(r"(d) Concurrence $C(n)$ & Distance-Dependent ESD Kinks")
    ax.legend(loc="upper right", fontsize=7.2, ncol=2)

    # (e) SC - Normal concurrence shift & anomalous F_AB isolation
    ax = axes[1, 1]
    c25_sc = np.array([r["sc_metrics"]["spin"]["0.25_0"]["C"] for r in s75["rows"]])
    c25_n = np.array([r["norm_metrics"]["spin"]["0.25_0"]["C"] for r in s75["rows"]])
    c25_0 = np.array([r["sc_metrics"]["spin"]["0.25_0"]["C_normonly"] for r in s75["rows"]])
    c1_sc = np.array([r["sc_metrics"]["spin"]["1_0"]["C"] for r in s75["rows"]])
    c1_n = np.array([r["norm_metrics"]["spin"]["1_0"]["C"] for r in s75["rows"]])
    c1_0 = np.array([r["sc_metrics"]["spin"]["1_0"]["C_normonly"] for r in s75["rows"]])
    ax.plot(n75, c25_sc - c25_n, color="#1f77b4", lw=1.9, label=r"$r=0.25a$: $C_{\mathrm{SC}}-C_N$")
    ax.plot(n75, c25_sc - c25_0, color="#1f77b4", lw=1.4, ls="--", label=r"$r=0.25a$: $C_{\mathrm{SC}}-C_{\mathrm{norm\text{-}only}}$")
    ax.plot(n75, c1_sc - c1_n, color="#d62728", lw=1.9, label=r"$r=1.00a$: $C_{\mathrm{SC}}-C_N$")
    ax.plot(n75, c1_sc - c1_0, color="#d62728", lw=1.4, ls="--", label=r"$r=1.00a$: $C_{\mathrm{SC}}-C_{\mathrm{norm\text{-}only}}$")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(0.2192, color="0.4", lw=0.9, ls=":")
    ax.axvline(1.0194, color="0.4", lw=0.9, ls=":")
    ax.set_xlim(0, 1.98)
    ax.set_xlabel(r"Electron density $n$")
    ax.set_ylabel(r"Concurrence difference $\Delta C$")
    ax.set_title(r"(e) Dilute BEC Pairing Boost vs. High-$n$ Fermi Smearing")
    ax.legend(loc="lower right", fontsize=7.2)

    # (f) Nk convergence of discrete-k noise in d^2 C / dn^2
    ax = axes[1, 2]
    for nk, col, lw, ls in [("Nk101", "0.70", 1.1, "-"),
                            ("Nk201", "#ff7f0e", 1.3, "-"),
                            ("Nk401", "#2ca02c", 1.6, "-"),
                            ("Nk801", "#1f77b4", 2.1, "-")]:
        sw = d["nk_grid_audit"][nk]
        nn = np.array([r["n"] for r in sw["rows"]])
        d2c = np.array(sw["derivs"]["concurrence"]["spin"]["0.25_0"]["d2CS"])
        ax.plot(nn[2:-2], d2c[2:-2], color=col, lw=lw, ls=ls, label=nk.replace("Nk", r"$N_k=$"))
    ax.axvline(0.9065, color="k", lw=1.0, ls="--", label=r"$n_{\mathrm{vH},-}$")
    ax.axvline(1.0194, color="#d62728", lw=1.0, ls=":", label=r"$n_{c,2}$")
    ax.set_xlim(0.855, 1.055)
    ax.set_ylim(-18, 20)
    ax.set_xlabel(r"Electron density $n$ ($\Delta n=0.003$)")
    ax.set_ylabel(r"$d^2 C_{\mathrm{SC}}/dn^2$ ($r=0.25a$)")
    ax.set_title(r"(f) $N_k$ Convergence: Stair-Step Noise vs. True Singularity")
    ax.legend(loc="upper right", fontsize=7.3, ncol=2)

    fig.suptitle(
        r"Electron-Density ($n$) QPT Landscape: $\mathrm{arXiv:1801.08988}$ Eq. (28) Energy vs. Concurrence ($\Delta_t/t=4.5$, Periodic $N_k=401$–$801$)",
        fontsize=11.5, fontweight="bold", y=0.995
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, "fig13_density_eq28_phase_landscape")


def make_fig14(d: dict) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18.5, 8.6))
    z = d["zooms"]

    # Column 0: Mid-band V=0.5 (U=75 SC vs U=115 Normal vs T=1e-3)
    z75 = z["zoom_mid_U75_V0.5_Nk801"]
    z115 = z["zoom_mid_U115_V0.5_Nk801"]
    zT = z["zoom_mid_Normal_T1e3_V0.5_Nk801"]
    n0 = np.array([r["n"] for r in z75["rows"]])
    sl = slice(2, -2)

    ax = axes[0, 0]
    ax.plot(n0[sl], np.array(z115["derivs"]["d2eS_dn2"])[sl], color="0.35", lw=1.7, label="Normal ($U=115$)")
    ax.plot(n0[sl], np.array(zT["derivs"]["d2eS_dn2"])[sl], color="#2ca02c", lw=1.3, ls="--", label="Normal $T=10^{-3}t$")
    ax.plot(n0[sl], np.array(z75["derivs"]["d2eS_dn2"])[sl], color="#1f77b4", lw=2.1, label="SC ($U=75$)")
    for xv, lbl, c in [(0.9065, r"$n_{\mathrm{vH},-}$", "0.3"), (1.0194, r"$n_{c,2}$", "#d62728"), (1.0935, r"$n_{\mathrm{vH},+}$", "0.3")]:
        ax.axvline(xv, color=c, lw=1.0, ls=":")
    ax.set_title(r"(a) $V_{\mathrm{SO}}=0.5$: Eq. (28) $d^2e_{\mathrm{GS}}/dn^2$")
    ax.set_ylabel(r"$d^2e_{\mathrm{GS}}/dn^2 = d\mu/dn$ [$t$]")
    ax.legend(fontsize=7.2, loc="upper right")

    ax = axes[1, 0]
    ax.plot(n0[sl], np.array(z115["derivs"]["concurrence"]["spin"]["0.25_0"]["dCN"])[sl], color="0.35", lw=1.6, label=r"$dC_N/dn$ (spin)")
    ax.plot(n0[sl], np.array(z75["derivs"]["concurrence"]["spin"]["0.25_0"]["dCS"])[sl], color="#1f77b4", lw=2.0, label=r"$dC_{\mathrm{SC}}/dn$ (spin)")
    ax.plot(n0[sl], np.array(z75["derivs"]["concurrence"]["band"]["0.25_0"]["dCS"])[sl], color="#ff7f0e", lw=1.5, ls="--", label=r"$dC_{\mathrm{SC}}/dn$ (band)")
    for xv, c in [(0.9065, "0.3"), (1.0194, "#d62728"), (1.0935, "0.3")]:
        ax.axvline(xv, color=c, lw=1.0, ls=":")
    ax.set_xlabel(r"Electron density $n$")
    ax.set_ylabel(r"$dC/dn$ ($r=0.25a$)")
    ax.set_title(r"(e) $V_{\mathrm{SO}}=0.5$: $n_{\mathrm{vH},-}$ Cusp vs. $n_{c,2}$ Lag")
    ax.legend(fontsize=7.0, loc="lower right")

    # Column 1: V_SO=0.0 separated n_vH=1.000 & n_c2=1.1966
    zv0 = z["zoom_mid_U75_V0.0_Nk801"]
    n1 = np.array([r["n"] for r in zv0["rows"]])
    ax = axes[0, 1]
    ax.plot(n1[sl], np.array(zv0["derivs"]["d2eN_dn2"])[sl], color="0.4", lw=1.5, ls="--", label="Normal")
    ax.plot(n1[sl], np.array(zv0["derivs"]["d2eS_dn2"])[sl], color="#1f77b4", lw=2.0, label="SC ($U=75$)")
    ax.axvline(1.0000, color="0.3", lw=1.0, ls=":", label=r"$n_{\mathrm{vH}}=1.0$")
    ax.axvline(1.1966, color="#d62728", lw=1.0, ls=":", label=r"$n_{c,2}=1.197$")
    ax.set_title(r"(b) $V_{\mathrm{SO}}=0$: Eq. (28) $d^2e_{\mathrm{GS}}/dn^2$")
    ax.legend(fontsize=7.2, loc="upper left")

    ax = axes[1, 1]
    ax.plot(n1[sl], np.array(zv0["derivs"]["concurrence"]["spin"]["0.25_0"]["dCN"])[sl], color="0.4", lw=1.5, ls="--", label=r"$dC_N/dn$")
    ax.plot(n1[sl], np.array(zv0["derivs"]["concurrence"]["spin"]["0.25_0"]["dCS"])[sl], color="#1f77b4", lw=2.0, label=r"$dC_{\mathrm{SC}}/dn$")
    ax.axvline(1.0000, color="0.3", lw=1.0, ls=":")
    ax.axvline(1.1966, color="#d62728", lw=1.0, ls=":")
    ax.set_xlabel(r"Electron density $n$")
    ax.set_title(r"(f) $V_{\mathrm{SO}}=0$: $dC/dn$ ($n_{\mathrm{vH}}$ vs. $n_{c,2}$)")
    ax.legend(fontsize=7.2, loc="lower right")

    # Column 2: Isolated high-density pairing QPT n_c=1.4602 at U=115, V=0.5
    zh = z["zoom_high_U115_V0.5_Nk801"]
    n2 = np.array([r["n"] for r in zh["rows"]])
    ax = axes[0, 2]
    ax.plot(n2[sl], np.array(zh["derivs"]["d2eN_dn2"])[sl], color="0.4", lw=1.5, ls="--", label="Normal")
    ax.plot(n2[sl], np.array(zh["derivs"]["d2eS_dn2"])[sl], color="#d62728", lw=2.0, label="SC ($U=115$)")
    ax.axvline(1.4602, color="#d62728", lw=1.1, ls=":", label=r"$n_c=1.460$")
    ax.set_title(r"(c) $U=115$ Isolated $n_c$: Eq. (28)")
    ax.legend(fontsize=7.2, loc="upper left")

    ax = axes[1, 2]
    ax.plot(n2[sl], np.array(zh["derivs"]["concurrence"]["spin"]["0.25_0"]["d2CN"])[sl], color="0.4", lw=1.5, ls="--", label=r"$d^2C_N/dn^2$")
    ax.plot(n2[sl], np.array(zh["derivs"]["concurrence"]["spin"]["0.25_0"]["d2CS"])[sl], color="#d62728", lw=2.0, label=r"$d^2C_{\mathrm{SC}}/dn^2$")
    ax.axvline(1.4602, color="#d62728", lw=1.1, ls=":", label=r"$n_c=1.460$")
    ax.set_xlabel(r"Electron density $n$")
    ax.set_ylabel(r"$d^2C/dn^2$ ($r=0.25a$)")
    ax.set_title(r"(g) $U=115$: Essential-Singularity Flatness")
    ax.legend(fontsize=7.2, loc="lower left")

    # Column 3: (pi, pi) Dirac cone at n=1.9290 (Normal 3rd-order vs SC U=75)
    zdn = z["zoom_dirac_Normal_V0.5_Nk801"]
    zds = z["zoom_dirac_SC_U75_V0.5_Nk801"]
    n3 = np.array([r["n"] for r in zdn["rows"]])
    ax = axes[0, 3]
    ax.plot(n3[sl], np.array(zdn["derivs"]["d3eS_dn3"])[sl], color="0.3", lw=2.0, label=r"Normal $d^3e_N/dn^3$")
    ax.plot(n3[sl], np.array(zds["derivs"]["d3eS_dn3"])[sl], color="#1f77b4", lw=2.0, ls="--", label=r"SC ($U=75$) $d^3e_{\mathrm{SC}}/dn^3$")
    ax.axvline(1.9290, color="#2ca02c", lw=1.1, ls=":", label=r"$n_{\mathrm{Dirac},4}=1.929$")
    ax.set_title(r"(d) $(\pi,\pi)$ Dirac Cone: Eq. (28) $d^3e_{\mathrm{GS}}/dn^3$")
    ax.set_ylabel(r"$d^3e_{\mathrm{GS}}/dn^3$ [$t$]")
    ax.legend(fontsize=7.2, loc="lower left")

    ax = axes[1, 3]
    ax.plot(n3[sl], np.array(zdn["derivs"]["concurrence"]["spin"]["0.25_0"]["d2CS"])[sl], color="0.3", lw=2.0, label=r"Normal $d^2C_N/dn^2$")
    ax.plot(n3[sl], np.array(zds["derivs"]["concurrence"]["spin"]["0.25_0"]["d2CS"])[sl], color="#1f77b4", lw=2.0, ls="--", label=r"SC ($U=75$) $d^2C_{\mathrm{SC}}/dn^2$")
    ax.axvline(1.9290, color="#2ca02c", lw=1.1, ls=":")
    ax.set_xlabel(r"Electron density $n$")
    ax.set_ylabel(r"$d^2C/dn^2$ ($r=0.25a$)")
    ax.set_title(r"(h) $(\pi,\pi)$ Dirac Cone: $d^2C/dn^2$ Jump vs. SC")
    ax.legend(fontsize=7.2, loc="lower left")

    fig.suptitle(
        r"Converged $N_k=801$ Periodic Zooms: Eq. (28) Ground-State Energy vs. Concurrence Across All Density-Driven Transitions",
        fontsize=11.5, fontweight="bold", y=0.995
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, "fig14_density_qpt_zooms_eq28_vs_concurrence")


def main() -> int:
    d = json.load(open(os.path.join(DATA_DIR, "task8_density_qpt_eq28.json")))
    export_csvs(d)
    make_fig13(d)
    make_fig14(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
