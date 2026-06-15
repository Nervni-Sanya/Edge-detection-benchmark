
# Edge Detection Benchmark — Vector Lighting & color baselines

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-orange.svg)]()
[![DOI](https://zenodo.org/badge/1201389926.svg)](https://doi.org/10.5281/zenodo.19420796)
[![ru](https://img.shields.io/badge/lang-ru-green.svg)](README.ru.md)

A small, reproducible edge-detection benchmark. It implements classical grayscale operators (Sobel, Prewitt, Canny), color operators (the Di Zenzo color structure tensor and variants), and `vector_lighting` — a method derived here from a "virtual illumination" model of the RGB channels.

> ### ⚠️ Honest status of the `vector_lighting` method
> This project began as an attempt at an original color edge detector. A rigorous comparison against the proper **color** baseline — the **Di Zenzo (1986) structure tensor** — showed that:
> - **The core of `vector_lighting` is mathematically equivalent to the Di Zenzo color structure tensor.** With its height term disabled, its response map is rank-identical to Di Zenzo (Spearman ρ ≈ 0.99 on real images, 1.00 on isoluminant tests). The "virtual illumination + channel permutations" construction is an **alternative derivation of an existing operator**.
> - Its one distinctive ingredient, a channel-asymmetric *height* term, perturbs the result by only ~2–3% and **does not improve** general-RGB accuracy (on BSDS500 it slightly lowers it).
> - The genuine "color sensitivity" advantage over grayscale Canny is **real but not unique** — Di Zenzo shares it.
>
> So `vector_lighting` is **not** a superior new detector. This repository is therefore best understood as a **benchmark and an independent re-derivation**, not a novelty claim. See [`docs/benchmark_report_2026-06.md`](docs/benchmark_report_2026-06.md) and reproduce the equivalence with `python tests/correspondence_analysis.py`.

## Author

**Panchenko Aleksandr Alekseevich** — Radiophysicist, student  
[Email](mailto:sascha.panchenko2018@yandex.ru) · [GitHub](https://github.com/Nervni-Sanya) · [ORCID](https://orcid.org/0009-0009-9104-1214)

## What's here

- **Grayscale detectors:** Sobel, Prewitt, Canny (NumPy/SciPy implementations).
- **Color detectors:** Di Zenzo structure tensor, a Di-Zenzo-based color Canny, multichannel-Sobel, and `vector_lighting`.
- **A reproducible benchmark harness** (`tests/testing.py`): synthetic set, orientation/anisotropy study, image-scaling study, and BSDS500 real-image evaluation.
- **A correspondence analysis** (`tests/correspondence_analysis.py`) quantifying the `vector_lighting` ↔ Di Zenzo equivalence.

All numbers below are reproducible. Absolute timings are machine-dependent.

## 📊 Benchmark results

### Synthetic set (F1, 2-px tolerance, 256×256)

`python tests/testing.py --baseline-only`. Sobel/Prewitt/Di Zenzo are binarized at their 95th percentile; single-image synthetic F1 is sensitive to that choice.

| Image | Sobel | Canny | DiZenzo | VectorLight |
|-------|:-----:|:-----:|:-------:|:-----------:|
| `checkerboard` | 0.000 | 0.945 | 0.739 | 0.933 |
| `concentric_circles` | 1.000 | 1.000 | 1.000 | 1.000 |
| `color_patches_equal_brightness` | 1.000 | 0.000 | 0.909 | 0.909 |
| `color_wheel` | 0.510 | 0.471 | 0.340 | 0.887 |
| `blurred_disk` | 1.000 | 1.000 | 1.000 | 1.000 |
| `gray_scale` | 1.000 | 1.000 | 0.000 | 0.000 |
| **Mean F1** | 0.752 | 0.736 | 0.665 | 0.788 |

On this tiny synthetic set `vector_lighting` and Di Zenzo trade wins per image (the two are the same underlying operator, so per-image gaps come from threshold brittleness, not a real quality difference). The decisive comparison is on real images below.

### Color sensitivity (isoluminant edges) — the advantage is shared

Equal-brightness color stripes, 12 orientations (`python tests/testing.py --orientation`):

| Method | F1 on isoluminant edges |
|--------|:-----------------------:|
| VectorLight | ~0.99 |
| **DiZenzo** | **~0.99** |
| Canny | 0.000 |

Both color methods see isoluminant edges; grayscale Canny cannot. This is the corrected version of the earlier "unique to our method" claim — it is **not** unique.

### Real images — BSDS500 (200-image test split)

`python tests/testing.py --bsds /path/to/BSDS500`. Union-of-annotators ground truth, tolerance 0.0075×diagonal, fixed thresholds.

| Method | F1 | Precision | Recall |
|--------|:--:|:---------:|:------:|
| **DiZenzo** | **0.565** | **0.637** | 0.526 |
| Canny | 0.566 | 0.482 | 0.774 |
| VL-default | 0.552 | 0.605 | 0.504 |
| Prewitt | 0.543 | 0.582 | 0.529 |
| Sobel | 0.541 | 0.577 | 0.530 |

On real photos Di Zenzo slightly leads `vector_lighting` (0.565 vs 0.552). All hand-crafted detectors sit far below modern learned methods (BSDS ODS ≈ 0.83–0.84).

### Method equivalence (`tests/correspondence_analysis.py`)

| Comparison | Spearman ρ (structured images) | Meaning |
|---|:--:|---|
| VectorLight(α=0) ↔ DiZenzo | **0.99–1.00** | Rank-identical ⇒ the same operator |
| height term (α=1 vs α=0) | ~0.98 | Adds only a ~2–3% perturbation |

## Library usage

```python
from vector_lighting import sobel, prewitt, canny, vector_lighting, EdgeDetector

edges = vector_lighting(image)            # color edge map (uint8)
edges = canny(image, low_threshold=50, high_threshold=100)
```

The detectors are usable as a small library; `vector_lighting`'s parameters are documented in [the source](vector_lighting/core.py). For a standard, well-understood color edge detector prefer the Di Zenzo structure tensor (`tests/color_baselines.py`).

## Reproduce

```bash
pip install numpy scipy pandas pillow tqdm matplotlib openpyxl

python tests/testing.py --baseline-only         # synthetic table + timing
python tests/testing.py --orientation           # anisotropy / diagonal edges
python tests/testing.py --scaling               # behavior vs image size
git clone --depth 1 https://github.com/BIDS/BSDS500 /path/to/BSDS500
python tests/testing.py --bsds /path/to/BSDS500 # real images
python tests/correspondence_analysis.py --bsds /path/to/BSDS500  # VL vs Di Zenzo
```

## References

- S. Di Zenzo, "A note on the gradient of a multi-image", *CVGIP* 33(1):116–125, 1986.
- J. Canny, "A computational approach to edge detection", *IEEE TPAMI* 8(6):679–698, 1986.

## Acknowledgments

Generative AI tools were used for code refactoring, benchmarking, and documentation.

```bash
pip install git+https://github.com/Nervni-Sanya/Edge-detection-benchmark.git
```
