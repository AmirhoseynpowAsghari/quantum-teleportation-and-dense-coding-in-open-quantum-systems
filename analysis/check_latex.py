#!/usr/bin/env python3
"""Dependency-free static checks for manuscript.tex (not a LaTeX compile)."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manuscript.tex"
REPORT = ROOT / "reports" / "latex_static_validation.json"
MANDATORY = [
    "figures/hubbard_amp_independent_gA4.50_GA0.010_gB4.50_GB0.010_F_heatmap_singlet.png",
    "figures/hubbard_amp_independent_gA4.50_GA0.010_gB4.50_GB0.010_C_heatmap.png",
    "figures/hubbard_amp_independent_gA1.50_GA0.100_gB1.50_GB0.100_holevo_theta_t_heatmap.png",
    "figures/hubbard_amp_independent_gA1.50_GA0.100_gB1.50_GB0.100_holevo_vs_r.png",
    "figures/hubbard_amp_independent_gA1.50_GA0.100_gB1.50_GB0.100_F_vs_t_singlet.png",
    "figures/hubbard_amp_independent_gA1.50_GA0.100_gB1.50_GB0.100_F_polar_singlet.png",
]

text = SOURCE.read_text(encoding="utf-8")
labels = re.findall(r"\\label\{([^}]+)\}", text)
refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text)
cites = [item.strip() for group in re.findall(r"\\cite\{([^}]+)\}", text)
         for item in group.split(",")]
bibitems = re.findall(r"\\bibitem\{([^}]+)\}", text)
graphics = re.findall(r"\\includegraphics(?:\[[^]]*\])?\s*\{([^}]+)\}", text)
# Remove comments before the basic delimiter checks.
uncommented = "\n".join(line.split("%", 1)[0] for line in text.splitlines())
environments = ["document", "equation", "align", "figure", "figure*",
                "table", "table*", "thebibliography"]
result = {
    "source": SOURCE.name,
    "compilation_performed": False,
    "check_type": "static source audit only",
    "label_count": len(labels),
    "duplicate_labels": sorted(k for k, v in Counter(labels).items() if v > 1),
    "undefined_references": sorted(set(refs) - set(labels)),
    "citation_key_count": len(bibitems),
    "duplicate_bibitems": sorted(k for k, v in Counter(bibitems).items() if v > 1),
    "undefined_citations": sorted(set(cites) - set(bibitems)),
    "uncited_bibitems": sorted(set(bibitems) - set(cites)),
    "brace_balance": uncommented.count("{") - uncommented.count("}"),
    "environment_balance": {
        env: [uncommented.count("\\begin{" + env + "}"),
              uncommented.count("\\end{" + env + "}")]
        for env in environments
    },
    "graphics": [{"path": path, "exists": (ROOT / path).is_file()}
                 for path in graphics],
    "mandatory_figure_include_counts": {path: graphics.count(path)
                                        for path in MANDATORY},
}
result["all_graphics_found"] = all(item["exists"] for item in result["graphics"])
result["all_mandatory_figures_exactly_once"] = all(
    count == 1 for count in result["mandatory_figure_include_counts"].values())
result["static_checks_passed"] = (
    not result["duplicate_labels"] and not result["undefined_references"]
    and not result["duplicate_bibitems"] and not result["undefined_citations"]
    and not result["uncited_bibitems"] and result["brace_balance"] == 0
    and all(begin == end for begin, end in result["environment_balance"].values())
    and result["all_graphics_found"]
    and result["all_mandatory_figures_exactly_once"]
)
REPORT.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["static_checks_passed"] else 1)
