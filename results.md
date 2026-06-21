# Experiment Results

This file summarizes the current quantitative results for **LGDU-Splatting**:

```text
Line-Guided Densification and Unpooling for 3D Gaussian Splatting
```

All values are computed on rendered `test` views.

Metric convention:

- PSNR: higher is better.
- SSIM: higher is better.
- LPIPS: lower is better.

The reported local-region metrics use `region_metrics.py` with the vanilla 3DGS render as the hole-mask reference. This makes all methods compare on the same dark / edge / hole masks.

## Main Quantitative Results

This table is the current main result summary. It combines whole-image metrics with hole-region metrics, because LGDU is designed to preserve global quality while repairing line-related failure regions.

| Scene | Res. | Method | Global PSNR ↑ | Global SSIM ↑ | Global LPIPS ↓ | Hole PSNR ↑ | Hole SSIM ↑ | Hole LPIPS ↓ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| drjohnson | r2 | 3DGS | 29.6548 | 0.91286 | 0.13901 | 10.3297 | 0.97418 | 0.03113 |
| drjohnson | r2 | Mini-Splatting | **29.7897** | 0.91110 | 0.15960 | **12.8215** | **0.97867** | **0.02499** |
| drjohnson | r2 | **LGDU** | 29.7431 | **0.91284** | **0.13807** | 11.9538 | 0.97859 | 0.02791 |
| playroom | r2 | 3DGS | 30.4723 | 0.92675 | **0.13717** | 11.7252 | 0.98029 | 0.02893 |
| playroom | r2 | Mini-Splatting | **30.6869** | **0.92816** | 0.14955 | 12.4723 | 0.98061 | 0.02987 |
| playroom | r2 | **LGDU** | 30.4856 | 0.92600 | 0.13802 | **12.7615** | **0.98153** | **0.02471** |
| counter | r2 | 3DGS | **30.3106** | 0.94141 | 0.06218 | 14.0211 | 0.98625 | 0.01173 |
| counter | r2 | Mini-Splatting | 29.2275 | 0.92343 | 0.08112 | **15.7869** | 0.99001 | **0.00719** |
| counter | r2 | **LGDU** | 30.2934 | **0.94173** | **0.06201** | 15.3808 | **0.99005** | 0.00784 |
| kitchen | r1 | 3DGS | 32.1701 | 0.94846 | 0.06685 | 9.8581 | 0.97854 | 0.02707 |
| kitchen | r1 | Mini-Splatting | 31.0743 | 0.93358 | 0.08861 | **16.5058** | **0.99157** | **0.01069** |
| kitchen | r1 | **LGDU** | **32.3517** | **0.94916** | **0.06613** | 11.4957 | 0.98466 | 0.01993 |

## LGDU vs 3DGS

LGDU is not intended to maximize hole-only scores at the cost of the whole image. The target behavior is:

```text
preserve or improve global 3DGS quality
+ improve hole / edge / dark failure regions
```

| Scene | Delta Global PSNR | Delta Global SSIM | Delta Global LPIPS | Delta Hole PSNR | Delta Hole SSIM | Delta Hole LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| drjohnson | +0.0884 | -0.00002 | -0.00095 | +1.6241 | +0.00441 | -0.00322 |
| playroom | +0.0134 | -0.00075 | +0.00085 | +1.0363 | +0.00124 | -0.00422 |
| counter | -0.0172 | +0.00032 | -0.00017 | +1.3597 | +0.00380 | -0.00389 |
| kitchen | +0.1816 | +0.00070 | -0.00071 | +1.6376 | +0.00612 | -0.00714 |

Summary:

- LGDU improves hole-region PSNR on all four scenes.
- LGDU improves or nearly preserves whole-image PSNR/SSIM/LPIPS on all four scenes.
- The largest global gain appears on `kitchen`.
- The strongest hole-region win over Mini-Splatting appears on `playroom`.

## Local Dark / Edge Region Metrics

These local regions are useful because line-guided methods often affect structural or under-covered areas more than the full image average.

### Dark Region

| Scene | Method | Dark PSNR ↑ | Dark SSIM ↑ | Dark LPIPS ↓ |
|---|---|---:|---:|---:|
| drjohnson | 3DGS | 29.7285 | **0.94739** | **0.07443** |
| drjohnson | LGDU | **29.7859** | 0.94712 | **0.07443** |
| playroom | 3DGS | 28.6304 | 0.97302 | 0.03853 |
| playroom | LGDU | **28.7072** | **0.97328** | **0.03822** |
| counter | 3DGS | **33.3057** | 0.96921 | **0.03599** |
| counter | LGDU | 33.2867 | **0.96925** | 0.03606 |
| kitchen | 3DGS | 31.2971 | 0.98848 | 0.01730 |
| kitchen | LGDU | **31.5306** | **0.98888** | **0.01670** |

### Edge Region

| Scene | Method | Edge PSNR ↑ | Edge SSIM ↑ | Edge LPIPS ↓ |
|---|---|---:|---:|---:|
| drjohnson | 3DGS | 26.3984 | 0.95084 | 0.06109 |
| drjohnson | LGDU | **26.4778** | **0.95135** | **0.06002** |
| playroom | 3DGS | 27.2588 | 0.95791 | 0.05883 |
| playroom | LGDU | **27.2719** | **0.95829** | **0.05769** |
| counter | 3DGS | **27.1027** | **0.97114** | **0.02897** |
| counter | LGDU | 27.0554 | **0.97114** | 0.02903 |
| kitchen | 3DGS | 29.8555 | 0.97909 | 0.02919 |
| kitchen | LGDU | **30.0036** | **0.97950** | **0.02882** |

## Scene-Level Analysis

### drjohnson

LGDU improves global PSNR and LPIPS over 3DGS while keeping SSIM essentially tied. Locally, LGDU improves both edge and hole regions. Mini-Splatting has stronger hole PSNR/LPIPS, but its global LPIPS is substantially worse.

### playroom

LGDU gives the strongest hole-region result among the three methods. Qualitative inspection also shows LGDU better preserves line structures than Mini-Splatting. Whole-image metrics remain close to 3DGS.

### counter

LGDU is globally near-tied with 3DGS and clearly improves the hole region. Mini-Splatting repairs holes aggressively, but degrades global, dark-region, and edge-region quality on this scene.

### kitchen

This is the strongest LGDU scene so far. LGDU improves all global metrics over 3DGS and also improves dark, edge, and hole local regions. Mini-Splatting has very strong hole metrics, but performs much worse globally and on dark/edge regions.

## Current Conclusion

Across four tested scenes, LGDU-Splatting shows a consistent pattern:

```text
LGDU preserves or slightly improves whole-image 3DGS reconstruction quality,
while consistently improving hole/failure regions and often improving structural edge regions.
```

Compared with Mini-Splatting:

- Mini-Splatting can be stronger on hole-only PSNR/LPIPS in some scenes.
- Mini-Splatting often degrades global or dark/edge quality.
- LGDU provides a better balance between global fidelity and local failure-region repair.

This supports using LGDU as a structure-aware 3DGS improvement rather than a hole-only repair method.

## Reproducibility Notes

Fair comparison settings used in these experiments:

```text
--eval
--data_device cpu
--test_iterations -1
--checkpoint_iterations 7000 15000 30000
```

Resolution:

```text
drjohnson: r2
playroom : r2
counter  : r2
kitchen  : r1
```

LGDU uses:

```text
LIMAP nv6 alltracks.txt
line-aware densification score
line-guided unpooling
```

Evaluation commands:

```bash
python metrics_stream.py --split test -m <model_paths...>

python region_metrics.py \
  --split test \
  --regions dark edge hole \
  --hole_reference_model <3dgs_baseline> \
  -m <model_paths...>
```
