#!/usr/bin/env python3
"""Generate report tables, matrices, macros, and figures from validated arrays."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "helicity_density_matrix_demo.npz"
VALIDATION = ROOT / "reports" / "helicity_density_matrix_validation.json"
REPORTS = ROOT / "reports"
FIGS = ROOT / "figs"


def fmt(value: float, digits: int = 9) -> str:
    if abs(value) < 0.5 * 10 ** (-digits):
        value = 0.0
    return f"{value:.{digits}f}"


def matrix_latex(matrix: np.ndarray) -> str:
    rows = []
    for row in matrix:
        rows.append(" & ".join(fmt(float(np.real(value)), 8) for value in row))
    return "\\begin{pmatrix}\n" + " \\\\\n".join(rows) + "\n\\end{pmatrix}"


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    if not validation["checks_passed"]:
        raise RuntimeError("The helicity validation record has not passed")
    with np.load(DATA) as data:
        r_vectors = data["r_vectors"].copy()
        vso_values = data["V_SO_over_t"].copy()
        arrays = {key: data[key].copy() for key in data.files if key not in {"r_vectors", "V_SO_over_t"}}

    r_labels = [f"({x},{y})" for x, y in r_vectors]
    # All nonzero-SOC resource rows, generated directly from the NPZ arrays.
    lines = [
        "\\begin{tabular}{lllllll}",
        "\\toprule",
        "Basis & $\\rr$ & $C$ & $\\Fbar_{\\rm ref}$ & $\\Fbar_{\\rm opt}$ & $\\chi_{\\rm P}$ & $S_2(\\rho_B)$ \\\\",
        "\\midrule",
    ]
    for basis in ("spin", "band"):
        for ir, label in enumerate(r_labels):
            lines.append(
                f"{basis.capitalize()} & ${label}$ & {fmt(arrays[f'concurrence_{basis}'][1, ir])} & "
                f"{fmt(arrays[f'fixed_fidelity_{basis}'][1, ir])} & "
                f"{fmt(arrays[f'optimized_fidelity_{basis}'][1, ir])} & "
                f"{fmt(arrays[f'holevo_{basis}'][1, ir])} & "
                f"{fmt(arrays[f'entropy_B_{basis}'][1, ir])} \\\\"
            )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
    ])
    (REPORTS / "helicity_demo_resources.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    matrices = (
        "\\begin{equation}\n\\rho_{\\mathrm{spin}}(V_{\\mathrm{SO}}=0.5t,\\mathbf r=(1,0))\n\\approx\n"
        + matrix_latex(arrays["rho_spin"][1, 1])
        + ",\n\\end{equation}\n\n"
        "\\begin{equation}\n\\rho_{\\mathrm{band}}(V_{\\mathrm{SO}}=0.5t,\\mathbf r=(1,0))\n\\approx\n"
        + matrix_latex(arrays["rho_band"][1, 1])
        + ".\n\\end{equation}\n"
    )
    (REPORTS / "helicity_demo_matrices.tex").write_text(matrices, encoding="utf-8")

    checks = validation["checks"]
    check_rows = [
        ("Zero-SOC solver residual", validation["parameters"]["0.0"]["max_abs_residual"]),
        ("Nonzero-SOC solver residual", validation["parameters"]["0.5"]["max_abs_residual"]),
        ("Zero-SOC Oh--Kim matrix error", checks["maximum_zero_SOC_Oh_Kim_matrix_error"]),
        ("Trace error", checks["maximum_trace_error"]),
        ("Hermiticity error", checks["maximum_hermiticity_error"]),
        ("Same-site singlet error", checks["maximum_same_site_singlet_error"]),
        ("Off-TRIM anomalous inversion residual", checks["maximum_anomalous_inversion_error"]),
        ("Pauli-average identity error", checks["maximum_pauli_average_identity_error"]),
        ("Holevo repository cross-check", checks["maximum_holevo_repository_crosscheck_error"]),
        ("Bell/Werner metric error", checks["maximum_werner_metric_error"]),
    ]
    validation_lines = [
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Diagnostic & Maximum absolute value \\\\",
        "\\midrule",
    ]
    validation_lines.extend(
        f"{name} & ${value:.3e}$ \\\\" for name, value in check_rows
    )
    validation_lines.extend([
        f"Minimum density-matrix eigenvalue & ${checks['minimum_density_eigenvalue']:.3e}$ \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ])
    (REPORTS / "helicity_demo_validation.tex").write_text(
        "\n".join(validation_lines) + "\n", encoding="utf-8")

    singlet = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2.0)
    macros = {
        "SpinChiZero": arrays["holevo_spin"][0, 1],
        "BandChiZero": arrays["holevo_band"][0, 1],
        "SpinChiNonzero": arrays["holevo_spin"][1, 1],
        "BandChiNonzero": arrays["holevo_band"][1, 1],
        "SpinFrefNonzero": arrays["fixed_fidelity_spin"][1, 1],
        "SpinFoptNonzero": arrays["optimized_fidelity_spin"][1, 1],
        "BandFrefNonzero": arrays["fixed_fidelity_band"][1, 1],
        "BandFoptNonzero": arrays["optimized_fidelity_band"][1, 1],
        "SpinFefNonzero": arrays["fef_spin"][1, 1],
        "BandFefNonzero": arrays["fef_band"][1, 1],
        "SpinSingletNonzero": float(np.real(
            singlet.conj() @ arrays["rho_spin"][1, 1] @ singlet)),
        "BandSingletNonzero": float(np.real(
            singlet.conj() @ arrays["rho_band"][1, 1] @ singlet)),
    }
    macro_text = "\n".join(
        f"\\newcommand{{\\{name}}}{{{value:.9f}}}" for name, value in macros.items()
    ) + "\n"
    (REPORTS / "helicity_demo_macros.tex").write_text(macro_text, encoding="utf-8")

    x = np.arange(len(r_labels))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    for basis, color, marker in (("spin", "#1764ab", "o"), ("band", "#d97706", "s")):
        axes[0, 0].plot(x, arrays[f"concurrence_{basis}"][1], marker=marker,
                        color=color, label=basis)
        axes[0, 1].plot(x, arrays[f"optimized_fidelity_{basis}"][1], marker=marker,
                        color=color, label=f"{basis}, optimized")
        axes[0, 1].plot(x, arrays[f"fixed_fidelity_{basis}"][1], marker=marker,
                        color=color, ls="--", mfc="none", label=f"{basis}, fixed")
        axes[1, 0].semilogy(x, np.maximum(arrays[f"holevo_{basis}"][1], 1e-16),
                            marker=marker, color=color, label=basis)
    axes[0, 0].set_ylabel("Concurrence $C$")
    axes[0, 0].set_ylim(-0.04, 1.08)
    axes[0, 1].axhline(2 / 3, color="black", lw=0.9, ls=":", label="classical $2/3$")
    axes[0, 1].set_ylabel("Average teleportation fidelity")
    axes[0, 1].set_ylim(0.47, 1.04)
    axes[1, 0].axhline(1.0, color="black", lw=0.9, ls=":", label="one bit")
    axes[1, 0].set_ylabel(r"$\chi_P$ (bits)")
    axes[1, 1].plot(x, arrays["entropy_AB_spin"][1], "o-", color="#1764ab", label=r"$S(\rho_{AB})$, spin")
    axes[1, 1].plot(x, arrays["entropy_B_spin"][1], "o--", color="#1764ab", mfc="none", label=r"$S(\rho_B)$, spin")
    axes[1, 1].plot(x, arrays["entropy_AB_band"][1], "s-", color="#d97706", label=r"$S(\rho_{AB})$, band")
    axes[1, 1].plot(x, arrays["entropy_B_band"][1], "s--", color="#d97706", mfc="none", label=r"$S(\rho_B)$, band")
    axes[1, 1].set_ylabel("Entropy (bits)")
    for ax in axes.flat:
        ax.set_xticks(x, r_labels)
        ax.set_xlabel("Exact lattice vector $(r_x,r_y)$")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle(r"Validated demonstration at $V_{SO}=0.5t$; points are discrete lattice vectors")
    fig.savefig(FIGS / "helicity_density_matrix_resources.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), constrained_layout=True)
    images = []
    for ax, basis in zip(axes, ("spin", "band")):
        rho = np.abs(arrays[f"rho_{basis}"][1, 1])
        off_diagonal = rho.copy()
        np.fill_diagonal(off_diagonal, np.nan)
        image = ax.imshow(off_diagonal, cmap="viridis",
                          norm=LogNorm(vmin=1e-5, vmax=2e-3))
        images.append(image)
        ax.set_title(f"{basis.capitalize()} basis")
        ax.set_xticks(range(4), ["00", "01", "10", "11"])
        ax.set_yticks(range(4), ["00", "01", "10", "11"])
        for i in range(4):
            for j in range(4):
                if i != j:
                    ax.text(j, i, f"{rho[i, j]:.1e}", ha="center", va="center",
                            color="white" if rho[i, j] > 2e-4 else "black", fontsize=8)
    fig.colorbar(images[-1], ax=axes, shrink=0.85, label=r"$|\rho_{ij}|$, $i\neq j$")
    fig.suptitle(r"Off-diagonal structure at $V_{SO}=0.5t$, $r=(1,0)$ (diagonal removed only for display)")
    fig.savefig(FIGS / "helicity_density_matrix_offdiagonal.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "source_data": str(DATA.relative_to(ROOT)),
        "source_data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "validation_passed": validation["checks_passed"],
        "generated_files": [
            "reports/helicity_demo_resources.tex",
            "reports/helicity_demo_matrices.tex",
            "reports/helicity_demo_validation.tex",
            "reports/helicity_demo_macros.tex",
            "figs/helicity_density_matrix_resources.png",
            "figs/helicity_density_matrix_offdiagonal.png",
        ],
        "resource_rows": 12,
        "matrix_count": 2,
    }
    (REPORTS / "helicity_report_assets.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
