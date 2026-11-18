"""Analyze and visualize analysis/qpt_parameter_scan.py outputs.

Reads reports/qpt_parameter_scan.json, prints a verdict table per family
(first-order: hysteresis / branch crossings / energy kinks; continuous:
d^2e kinks; Lifshitz: mu_N crossing 4t with binding -> normal GS), and
writes figs/qpt_scan_<family>.png.

Run: .venv/bin/python analysis/plot_qpt_parameter_scan.py
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_JSON = "reports/qpt_parameter_scan.json"


def load():
    with open(OUT_JSON) as f:
        return json.load(f)


def _d2(p, e):
    de = np.gradient(e, p, edge_order=2)
    return np.gradient(de, p, edge_order=2)


def _kinks(p, e, span, thr=1e-4):
    d2 = _d2(p, e)
    out = []
    for i in range(2, len(p) - 2):
        win = (p >= p[i] - span) & (p <= p[i] + span)
        m, s = np.mean(d2[win]), np.std(d2[win])
        if s > 0 and d2[i] > m + 3 * s and d2[i] > thr:
            out.append(round(float(p[i]), 4))
    return out, d2


def plot_v(data, ax):
    rec = data["v"]["raw"]
    for fam in rec.get("combos", []):
        up = fam["up"]
        p = np.array([q["V"] for q in up])
        e = np.array([q["e"] for q in up])
        ax[0].plot(p, e, ".-", label=fam["combo"])
        # binding on inset-style second axis handled in figure layout
    ax[0].set_xlabel("V_SO/t")
    ax[0].set_ylabel("e_GS [t]")
    ax[0].legend(fontsize=7)
    ax[0].set_title("e_GS vs V_SO (paper figure params + n=0.5)")


def plot_u(data, ax):
    rec = data["u"]["raw"]
    for fam in rec.get("combos", []):
        up = fam["up"]
        p = np.array([q["U"] for q in up])
        e = np.array([q["e"] for q in up])
        # binding: e - e_N; e_N stored only in points; recompute via points
        ax[0].plot(p, e, ".-", label=fam["combo"], ms=3)
    ax[0].set_xlabel("U [t]")
    ax[0].set_ylabel("e_GS [t]")
    ax[0].legend(fontsize=7)
    ax[0].set_title("e_GS vs U")


def plot_u_binding(data, ax):
    # binding curves from raw points (param, binding)
    pts = data["u"]["raw"].get("points", [])
    # points were appended combo-by-combo; group by contiguous param resets
    groups, cur = [], []
    for q in pts:
        if cur and q["param"] <= cur[-1]["param"]:
            groups.append(cur)
            cur = []
        cur.append(q)
    if cur:
        groups.append(cur)
    combos = data["u"]["raw"].get("combos", [])
    for g, fam in zip(groups, combos):
        p = np.array([q["param"] for q in g])
        b = np.array([q["binding"] for q in g])
        ax.plot(p, np.abs(b), ".-", ms=3, label=fam["combo"])
    ax.set_yscale("log")
    ax.set_xlabel("U [t]")
    ax.set_ylabel("|e_GS - e_N| [t]")
    ax.legend(fontsize=7)
    ax.set_title("pairing binding vs U (log)")


def plot_n(data, ax):
    rec = data["n"]["raw"]
    for fam in rec.get("combos", []):
        pts = fam["pts"]
        nA = np.array([q["n"] for q in pts])
        e = np.array([q["e"] for q in pts])
        mu = np.array([q["mu_N"] for q in pts])
        ax[0].plot(nA, e, ".-", ms=4, label=fam["combo"])
        cross = np.where((mu[:-1] - 4) * (mu[1:] - 4) < 0)[0]
        for i in cross:
            ax[0].axvline(nA[i], color="gray", ls=":", lw=0.8)
        ax[1].plot(nA, mu, ".-", ms=4, label=fam["combo"])
    ax[0].set_xlabel("n")
    ax[0].set_ylabel("e_GS [t]")
    ax[0].legend(fontsize=7)
    ax[0].set_title("e_GS vs n (dotted: mu_N crosses 4t)")
    ax[1].axhline(4.0, color="r", ls="--", lw=0.8)
    ax[1].set_xlabel("n")
    ax[1].set_ylabel("mu_N")
    ax[1].legend(fontsize=7)
    ax[1].set_title("normal-state mu vs n")


def plot_dt(data, ax):
    rec = data["dt"]["raw"]
    for fam in rec.get("combos", []):
        pts = fam["pts"]
        dA = np.array([q["Dt"] for q in pts])
        e = np.array([q["e"] for q in pts])
        b = np.array([abs(q["binding"]) for q in pts])
        ax[0].plot(dA, e, ".-", ms=4, label=fam["combo"])
        ax[1].plot(dA, b, ".-", ms=4, label=fam["combo"])
    ax[0].set_xlabel("Delta_t [t]")
    ax[0].set_ylabel("e_GS [t]")
    ax[0].legend(fontsize=7)
    ax[0].set_title("e_GS vs Delta_t")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Delta_t [t]")
    ax[1].set_ylabel("|binding| [t]")
    ax[1].legend(fontsize=7)
    ax[1].set_title("binding vs Delta_t (log)")


def verdicts(data):
    lines = []
    for kind in ("v", "u", "n", "dt"):
        if kind not in data:
            continue
        det = data[kind]["detected"]
        lines.append(f"== family {kind} (Nk={det['Nk']}) ==")
        for c in det["combos"]:
            hyst = c.get("hysteresis_max")
            kinks = c.get("kinks")
            jumps = c.get("gap_jumps")
            lines.append(
                f"  {c.get('combo')}: kinks={kinks} gap_jumps={jumps} "
                f"hysteresis={hyst} mu4t={c.get('mu_crosses_4t')} "
                f"min_d2e_dn2={c.get('min_d2e_dn2')}"
            )
        # raw extras: coexistence of solutions
        for fam in data[kind]["raw"].get("combos", []):
            if kind in ("v", "u"):
                multi = 0
                for q in fam.get("up", []):
                    pass
                lines.append(f"  [raw] {fam['combo']} hyst="
                             f"{fam.get('hysteresis_max')}")
    return "\n".join(lines)


def main():
    data = load()
    print(verdicts(data))

    if "v" in data:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        plot_v(data, ax)
        # binding panel
        pts = data["v"]["raw"].get("points", [])
        groups, cur = [], []
        for q in pts:
            if cur and q["param"] <= cur[-1]["param"]:
                groups.append(cur)
                cur = []
            cur.append(q)
        if cur:
            groups.append(cur)
        for g, fam in zip(groups, data["v"]["raw"].get("combos", [])):
            p = np.array([q["param"] for q in g])
            b = np.array([q["binding"] for q in g])
            ax[1].plot(p, b, ".-", ms=3, label=fam["combo"])
        ax[1].axhline(0, color="r", ls=":", lw=0.8)
        ax[1].set_xlabel("V_SO/t")
        ax[1].set_ylabel("e_GS - e_N [t]")
        ax[1].legend(fontsize=7)
        ax[1].set_title("binding vs V_SO")
        fig.tight_layout()
        fig.savefig("figs/qpt_scan_v.png", dpi=150)
        print("wrote figs/qpt_scan_v.png")

    if "u" in data:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        plot_u(data, ax)
        plot_u_binding(data, ax[1])
        fig.tight_layout()
        fig.savefig("figs/qpt_scan_u.png", dpi=150)
        print("wrote figs/qpt_scan_u.png")

    if "n" in data:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        plot_n(data, ax)
        fig.tight_layout()
        fig.savefig("figs/qpt_scan_n.png", dpi=150)
        print("wrote figs/qpt_scan_n.png")

    if "dt" in data:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        plot_dt(data, ax)
        fig.tight_layout()
        fig.savefig("figs/qpt_scan_dt.png", dpi=150)
        print("wrote figs/qpt_scan_dt.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
