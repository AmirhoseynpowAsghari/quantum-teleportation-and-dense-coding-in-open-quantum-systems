# LaTeX validation report

## Compilation

- **LaTeX compilation was performed successfully** with TeX Live in GitHub Actions run `42100176333` using `xu-cheng/latex-action@v4`.
- **Output:** `manuscript.pdf`, 13 pages and 1,823,972 bytes.
- The complete compiler output is preserved in `reports/manuscript.log`.
- The workflow `.github/workflows/build-manuscript.yml` reproduces the PDF and writes it back to the triggering branch through the GitHub Contents API.

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

1. The successful build log contains no undefined references or citations and no fatal errors. It does contain several underfull-box warnings, four overfull-box warnings associated with wide equations, and one REVTeX stuck-float warning. All ten graphics nevertheless appear in the output log and the PDF was produced completely.
2. REVTeX also warns that no affiliation was supplied and that `showpacs` is required to display PACS codes; these conventions were retained from the original manuscript.
3. The generated figures are PNG files, consistent with the supplied mandatory figures; journal submission may require higher-resolution or vector replacements.
4. `\date{\today}` remains unchanged from the original, so the PDF carries the CI compilation date.
5. Final aesthetic review of line breaks and float placement is recommended before journal submission.
