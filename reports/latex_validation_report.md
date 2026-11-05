# LaTeX validation report

## Compilation

- **LaTeX compilation was not performed.** `pdflatex`, `latexmk`, and `tectonic` were not installed in the execution environment.
- No PDF was generated, and the source is therefore **not compilation-tested**.
- Figure placement, overfull boxes, font rendering, and final page layout require a subsequent REVTeX 4.2 compilation.

## Static source validation

The command `.venv/bin/python analysis/check_latex.py` completed successfully and wrote `reports/latex_static_validation.json`.

| Check | Result |
|---|---|
| Document class | Original `revtex4-2` class and options retained |
| Braces | Balanced (net balance zero) |
| Environment begin/end counts | Balanced for document, equations, figures, and bibliography |
| Labels | 58; no duplicates |
| Undefined `\ref`/`\eqref` targets | None detected |
| Bibliography items | 10; no duplicates |
| Undefined citation keys | None detected |
| Uncited bibliography items | None detected |
| Included graphics | 10; all files found |
| Mandatory figures | All six included exactly once |
| Placeholder `image.png` | Removed |

## Figure validation

All six mandatory PNG files were opened and visually inspected. Their axes, fixed slices, legends, and color scales were checked against the plotting functions and NPZ arrays. The three missing supporting figures and the new interaction-sweep figure were generated and visually inspected as well.

Important caption corrections include:

- singlet reference versus initial-state reference;
- no teleportation meaning for automatic horizontal guide lines;
- first-quadrant-only polar data and its artificial closing segment;
- `r=0` restriction of the angular Holevo heatmap;
- separate noise parameters and time grids for the two archived regimes.

## Bibliography validation

The two original Oh–Kim entries were retained. The added bibliographic details were checked against publisher or scholarly records: Bennett *et al.* (1993), Bennett and Wiesner (1992), Wootters (1998), Jozsa (1994), Horodecki *et al.* (1999), Bruß *et al.* (2004), Breuer and Petruccione (2002), and Bellomo *et al.* (2007). No unverified DOI was inserted. All bibliography keys are used.

## Remaining technical issues

1. A real REVTeX compilation is still required to detect package-, class-, or layout-specific errors not covered by static checks.
2. The generated figures are PNG files, consistent with the supplied mandatory figures; journal submission may require higher-resolution or vector replacements.
3. `\date{\today}` remains unchanged from the original, so the compiled date will be the eventual compilation date.
