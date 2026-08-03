
# Edge Detection Benchmark — Vector Lighting & color baselines

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-orange.svg)]()
[![DOI](https://zenodo.org/badge/1201389926.svg)](https://doi.org/10.5281/zenodo.19420796)
[![ru](https://img.shields.io/badge/lang-ru-green.svg)](README.ru.md)

A small Python edge-detection toolkit and reproducible benchmark. The package ships **seven** edge detectors usable out of the box from a single API: classical grayscale operators (**Sobel**, **Prewitt**, **Canny**), color operators (the **Di Zenzo** color structure tensor, a Di-Zenzo-based **color Canny**, **multichannel Sobel**), and **`vector_lighting`** — a method derived here from a "virtual illumination" model of the RGB channels.

> ### ⚠️ Honest status of the `vector_lighting` method
> This project began as an attempt at an original color edge detector. A rigorous comparison against the proper **color** baseline — the **Di Zenzo (1986) structure tensor** — showed that:
> - **The core of `vector_lighting` is mathematically equivalent to the Di Zenzo color structure tensor.** With its height term disabled, its response map is rank-identical to Di Zenzo (Spearman ρ ≈ 0.99 on real images, 1.00 on isoluminant tests). The "virtual illumination + channel permutations" construction is an **alternative derivation of an existing operator**.
> - Its one distinctive ingredient, a channel-asymmetric *height* term, perturbs the result by only ~2–3% and **does not improve** general-RGB accuracy (on BSDS500 it slightly lowers it).
> - The genuine "color sensitivity" advantage over grayscale Canny is **real but not unique** — Di Zenzo shares it.
>
> So `vector_lighting` is **not** a superior new detector. This repository is therefore best understood as a **benchmark and an independent re-derivation**, not a novelty claim. The algebraic proof of equivalence is in [`docs/equivalence.md`](docs/equivalence.md); empirical verification is reproducible with `python tests/correspondence_analysis.py`.

## Author

**Panchenko Aleksandr Alekseevich** — Radiophysicist, student  
[GitHub](https://github.com/Nervni-Sanya)

## What's here

- **Grayscale detectors:** Sobel, Prewitt, Canny (NumPy/SciPy implementations).
- **Color detectors:** Di Zenzo structure tensor, a Di-Zenzo-based color Canny, multichannel-Sobel, and `vector_lighting`.
- **A reproducible benchmark harness** (`tests/testing.py`): synthetic set, orientation/anisotropy study, image-scaling study, and BSDS500 real-image evaluation.
- **A correspondence analysis** (`tests/correspondence_analysis.py`) quantifying the `vector_lighting` ↔ Di Zenzo equivalence.

All numbers below are reproducible. Absolute timings are machine-dependent.

## 📊 Benchmark results

### Synthetic set (F1, 2-px tolerance, 256×256)

> ⚠️ **These numbers are not a reliable ranking.** See the tie warning below the table.

`python tests/testing.py --baseline-only`. Sobel/Prewitt/Di Zenzo are binarized at their 95th percentile.

| Image | Sobel | Canny | DiZenzo | VectorLight |
|-------|:-----:|:-----:|:-------:|:-----------:|
| `checkerboard` | 0.000 | 0.958 | 0.505 | 0.933 |
| `concentric_circles` | 1.000 | 1.000 | 1.000 | 1.000 |
| `color_patches_equal_brightness` | 1.000 | 0.000 | 0.909 | 0.909 |
| `color_wheel` | 0.510 | 0.473 | 0.340 | 0.887 |
| `blurred_disk` | 1.000 | 1.000 | 1.000 | 1.000 |
| `gray_scale` | 1.000 | 1.000 | 0.000 | 0.000 |
| **Mean F1** | 0.752 | 0.739 | 0.626 | 0.788 |

**Why these are unreliable: threshold ties.** Synthetic images contain large areas of identical gradient magnitude, so thousands of pixels land *exactly* on the percentile threshold and are all included or excluded together. Switching the comparison from `>` to `>=` — a tie-breaking choice with no physical meaning — moves F1 by up to **0.45**:

| Image | Pixels exactly at threshold | F1 with `>` | F1 with `>=` |
|---|:--:|:--:|:--:|
| `checkerboard` | 4690 (28.6%) | 0.505 | **0.954** |
| `color_patches_equal_brightness` | 992 | 0.909 | 0.772 |
| `color_wheel` | 527 | 0.340 | 0.469 |
| Real photo (BSDS500) | 173 (0.11%) | — | negligible |

The same sensitivity makes these numbers change under floating-point rounding at the 1e-16 level. **Draw conclusions from the real-image results below**, where ties are 0.11% of pixels and the numbers are stable.

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
| **DiZenzo** | **0.575** | **0.654** | 0.531 |
| Canny | 0.566 | 0.482 | 0.774 |
| VL-default | 0.552 | 0.630 | 0.509 |
| Prewitt | 0.543 | 0.582 | 0.529 |
| Sobel | 0.541 | 0.577 | 0.530 |

On real photos Di Zenzo leads `vector_lighting` (0.575 vs 0.552). All hand-crafted detectors sit far below modern learned methods (BSDS ODS ≈ 0.83–0.84).

### Method equivalence (`tests/correspondence_analysis.py`)

| Comparison | Spearman ρ (structured images) | Meaning |
|---|:--:|---|
| VectorLight(α=0) ↔ DiZenzo | **0.99–1.00** | Rank-identical ⇒ the same operator |
| height term (α=1 vs α=0) | ~0.98 | Adds only a ~2–3% perturbation |

### Runtime — configuration, not operator, drives the differences

All operators now use separable convolutions (`ndimage.correlate1d`), which is mathematically identical to the 2D form (agreement to ~1e-13) but roughly twice as fast. `python tests/timing_robust.py` (256×256, 3 content types, 7 interleaved rounds):

| Detector | Median | vs. Sobel |
|---|:--:|:--:|
| Sobel / Prewitt | 2.6 ms | 1.0× |
| Di Zenzo, **matched config** (σ=0, central differences) | 4.9 ms | 1.9× |
| VL, 2 vectors, σ=0 | 5.8 ms | 2.3× |
| VL, 8 vectors, σ=0 | 8.5 ms | 3.3× |
| Canny | 11.5 ms | 4.5× |
| Di Zenzo, standard (Sobel gradients + σ=1) | 16.4 ms | 6.4× |
| VL default (8 vectors + permutations) | 36.7 ms | 14.4× |

**`vector_lighting` is not faster than Di Zenzo at equal settings.** Comparing VL's fast preset against Di Zenzo's *standard* configuration suggests a 2.8× advantage, but that gap is the configuration (Gaussian pre-smoothing + Sobel gradients vs. neither), not the operator. Give Di Zenzo the same cheap settings and it wins on **both** axes at once — 1.17× faster *and* markedly more accurate (mean F1 0.908 vs 0.703 on the synthetic set).

This follows from the [equivalence proof](docs/equivalence.md): both compute the same quantity, but the tensor obtains it in closed form while virtual illumination reconstructs it by sampling N directions. Sampling a closed-form expression cannot be cheaper than evaluating it.

Two further measurement notes:

1. **Dropping light vectors buys little.** Runtime is linear with a dominant fixed cost — `T(N) ≈ 8.3 ms + 0.47 ms · N` — so fixed work (gradients, percentile, normalization) is 69% of the total at N=8. Going 8 → 2 vectors gives ~1.5×, and no vector-count reduction can exceed ~1.45×.
2. **Naive block-wise timing is unreliable** (all repetitions of A, then all of B): identical calls measured **8.0 ms cold vs 4.0 ms after allocation churn**, a 2× swing exceeding most differences above. `tests/timing_robust.py` interleaves detectors and shuffles order to remove it; `tests/timing_scaling_analysis.py` breaks down the cost.

### Image size (256 → 2048)

`python tests/testing.py --scaling`. Median ms (left) and mean F1 (right), 3 patterns per size:

| Detector | 256 | 512 | 1024 | 2048 | | F1@256 | F1@512 | F1@1024 | F1@2048 |
|---|---:|---:|---:|---:|---|:--:|:--:|:--:|:--:|
| Sobel | 2.9 | 12.6 | 59 | 396 | | 0.667 | 0.667 | 0.989 | 0.989 |
| Prewitt | 2.9 | 12.3 | 59 | 393 | | 0.667 | 0.667 | 0.989 | 0.989 |
| Canny | 11.9 | 49 | 207 | 1697 | | 0.653 | 0.652 | 0.652 | 0.652 |
| VL-turbo | 5.3 | 24 | 140 | 823 | | **0.990** | **0.989** | **0.989** | **0.989** |
| VL-fast | 7.5 | 32 | 171 | 1042 | | 0.945 | 0.945 | 0.927 | 0.901 |
| VL-default | 35.9 | 168 | 1056 | 9753 | | 0.947 | 0.859 | 0.759 | 0.710 |

The default configuration's F1 **degrades with size** (0.947 → 0.710) while VL-turbo stays flat. This is parameter calibration, not the operator: `sigma=1.0` blurs a fixed pixel width regardless of resolution, and `binary_percentile=0.05` keeps a fixed 5% of pixels while the true edge fraction falls as ~1/size. **For images above ~512 px use `sigma=0–0.5` and `binary_percentile=0.0`.**

## Library usage

All seven detectors share a simple `(H, W, 3) uint8 → (H, W) uint8` signature:

```python
import numpy as np
from vector_lighting import sobel, prewitt, canny, vector_lighting   # grayscale + VL
from tests.color_baselines import dizenzo, dizenzo_canny, multichannel_sobel  # color

img = np.asarray(some_PIL_image)  # (H, W, 3) uint8

# Grayscale operators
e_sobel   = sobel(img)
e_prewitt = prewitt(img)
e_canny   = canny(img, low_threshold=50, high_threshold=100)

# Color operators (see them in tests/color_baselines.py)
e_dz      = dizenzo(img,        sigma=1.0)
e_dzc     = dizenzo_canny(img,  sigma=1.0)
e_mcs     = multichannel_sobel(img, sigma=1.0)

# vector_lighting (parameters documented in vector_lighting/core.py)
e_vl      = vector_lighting(img, mode=3, sigma=1.0)
```

For a standard, well-understood color edge detector, prefer the Di Zenzo structure tensor — it is ~1.5× faster than `vector_lighting`'s default configuration (though slower than its 2-vector setting), slightly more accurate on real photos (BSDS500), and the documented baseline in the literature.

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
