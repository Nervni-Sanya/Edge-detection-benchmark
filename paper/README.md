# Paper draft

`article.tex` — draft skeleton for **Pattern Recognition and Image Analysis (PRIA)**.

## Status

- **Final** (carry the actual measured numbers): all theorems, equations, and
  the five result tables (synthetic, BSDS500, correspondence, RGB-D).
- **To be written by the author** (marked `\todo{...}` in red): abstract
  headline, introduction motivation, related-work positioning, proof
  formalizations, conclusion takeaway.
- **Draft prose** (plain text, edit freely): section lead-ins.

The numbers were not invented — they are reproduced by the repository:

| Table | Source |
|---|---|
| Synthetic (Tab. 1) | `python tests/testing.py --baseline-only` |
| Isoluminant (§6) | `python tests/testing.py --orientation` |
| BSDS500, 200 img (Tab. 2) | `python tests/testing.py --bsds <BSDS500>` |
| Correspondence (Tab. 3) | `python tests/correspondence_analysis.py` |
| RGB-D (Tab. 4) | `python tests/rgbd_synthetic_test.py` |

## Build

Uses the standard `article` class so it compiles with any basic TeX install
(it was **not** compile-tested in the authoring environment — no TeX toolchain
there):

```bash
latexmk -pdf article.tex      # or: pdflatex article.tex (run twice for refs)
```

## Before submission

1. Port the body into the PRIA / Pleiades template (`pleiades.cls` /
   `spr-pria` style) — keep the math/tables, restyle the front matter.
2. Confirm author block (co-authorship/order); no e-mail/ORCID embedded by
   default, per the repository's privacy preference.
3. Resolve all `\todo{}` markers; add the BSDS SOTA / extra color-edge
   references flagged in the bibliography.
