# TDCA ICLR 2027 paper draft

This directory contains an anonymous, English-language ICLR 2027 submission draft for ?TDCA: Adaptive Test-Time Computation over Dynamic Reasoning Hypergraphs.?

## Status

- Uses the unmodified official ICLR 2027 LaTeX style and bibliography files.
- Keeps the camera-ready switch commented for anonymous review.
- Separates the paper into section files under sections/.
- Marks missing frozen-evaluation values with a visible red TBD.
- Uses boxed placeholders for figures that have not yet been produced.
- Labels current v2.2/v2.3 numbers as development diagnostics rather than final evidence.
- Places implementation details, gates, metric definitions, and reproducibility requirements in the appendix.

No missing result has been fabricated.

## Build

Run from this directory:

    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

Fallback:

    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    bibtex main
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex

The generated main.pdf is ignored by Git.

## Before submission

1. Replace every TBD using frozen experiment artifacts.
2. Replace the architecture, Pareto-curve, per-hop, and trace placeholders.
3. Run matched-compute baselines on identical IDs and corpus snapshots.
4. Add confidence intervals, paired tests, and negative results.
5. Numerically freeze every threshold currently described as acceptable.
6. Confirm that the main text remains at most 9 pages, excluding references.
7. Re-run leakage, graph invariant, controller-only mutation, and unsupported-answer audits.
8. Complete the official ICLR author/reproducibility checklist.
9. Keep the paper anonymous until the camera-ready stage.

See PAPER_TODO.md for the exact artifact-to-placeholder map.
