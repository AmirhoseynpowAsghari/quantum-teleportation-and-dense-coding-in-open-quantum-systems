#!/usr/bin/env python3
"""Static and numerical source checks for the helicity PDF report."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "helicity_density_matrix_report.tex"
REPORT = ROOT / "reports" / "helicity_report_static_validation.json"


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text)
    citations = [key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", text)
                 for key in group.split(",")]
    bibitems = re.findall(r"\\bibitem\{([^}]+)\}", text)
    graphics = re.findall(r"\\includegraphics(?:\[[^]]*\])?\s*\{([^}]+)\}", text)
    inputs = re.findall(r"\\input\{([^}]+)\}", text)
    uncommented = "\n".join(line.split("%", 1)[0] for line in text.splitlines())
    environments = ["document", "equation", "figure", "table", "itemize",
                    "enumerate", "pmatrix", "thebibliography", "verbatim"]
    manifest = json.loads((ROOT / "reports/helicity_report_assets.json").read_text())
    data_path = ROOT / manifest["source_data"]
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    with np.load(data_path) as data:
        numeric_errors = []
        resource_text = (ROOT / "reports/helicity_demo_resources.tex").read_text()
        row_count = sum(1 for line in resource_text.splitlines() if "&" in line)
        for basis in ("spin", "band"):
            for ir in range(6):
                for key in ("concurrence", "fixed_fidelity", "optimized_fidelity",
                            "holevo", "entropy_B"):
                    expected = f"{float(data[f'{key}_{basis}'][1, ir]):.9f}"
                    if expected not in resource_text:
                        numeric_errors.append(f"Missing {key}_{basis}[1,{ir}]={expected}")
        macro_text = (ROOT / "reports/helicity_demo_macros.tex").read_text()
        for name, basis in (("Spin", "spin"), ("Band", "band")):
            expected = f"{float(data[f'holevo_{basis}'][1, 1]):.9f}"
            if f"\\newcommand{{\\{name}ChiNonzero}}{{{expected}}}" not in macro_text:
                numeric_errors.append(f"Macro {name}ChiNonzero mismatch")
    result = {
        "source": SOURCE.name,
        "compilation_performed": False,
        "check_type": "Static source, generated-asset hash, and NPZ transcription checks",
        "duplicate_labels": sorted(k for k, v in Counter(labels).items() if v > 1),
        "undefined_references": sorted(set(refs) - set(labels)),
        "duplicate_bibitems": sorted(k for k, v in Counter(bibitems).items() if v > 1),
        "undefined_citations": sorted(set(citations) - set(bibitems)),
        "uncited_bibitems": sorted(set(bibitems) - set(citations)),
        "brace_balance": uncommented.count("{") - uncommented.count("}"),
        "environment_balance": {
            env: [uncommented.count("\\begin{" + env + "}"),
                  uncommented.count("\\end{" + env + "}")]
            for env in environments
        },
        "graphics": [{"path": path, "exists": (ROOT / path).is_file()} for path in graphics],
        "inputs": [{"path": path, "exists": (ROOT / path).is_file()}
                   for path in inputs],
        "resource_table_rows": row_count,
        "source_data_sha256_matches_manifest": data_hash == manifest["source_data_sha256"],
        "numeric_transcription_errors": numeric_errors,
    }
    result["static_checks_passed"] = (
        not result["duplicate_labels"] and not result["undefined_references"]
        and not result["duplicate_bibitems"] and not result["undefined_citations"]
        and not result["uncited_bibitems"] and result["brace_balance"] == 0
        and all(begin == end for begin, end in result["environment_balance"].values())
        and all(item["exists"] for item in result["graphics"])
        and all(item["exists"] for item in result["inputs"])
        and row_count == 12 and result["source_data_sha256_matches_manifest"]
        and not numeric_errors
    )
    REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["static_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
