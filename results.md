# Experiment Results

This file records the current local experiment results for EVP-LineSplat variants. Values are computed on rendered `test` views using:

```bash
python metrics_stream.py --split test -m <model_paths...>
```

Metric convention:

- PSNR: higher is better.
- SSIM: higher is better.
- LPIPS: lower is better.

## DrJohnson

Dataset path used locally:

```text
/home/ddsm/DownLoad/Confidence-Aware-LineSplat/gaussian-splatting/data/drjohnson
```

LIMAP line tracks:

```text
/home/ddsm/DownLoad/limap/outputs/drjohnson/drjohnson_nv6/alltracks.txt
```

Evaluation split:

```text
--eval, resolution 2, test views: 33
```

| Method | SSIM ↑ | PSNR ↑ | LPIPS ↓ | Notes |
|---|---:|---:|---:|---|
| 3DGS | 0.9128616 | 29.6547909 | 0.1390142 | Original 3DGS baseline |
| No-line pure3DGS | 0.9128566 | 29.6300488 | 0.1383264 | This codebase with line guidance disabled |
| Mini-Splatting | 0.9110953 | **29.7897034** | 0.1596022 | Best PSNR, weaker perceptual score |
| Edge-center adaptive | **0.9130062** | 29.5559807 | 0.1388844 | Best SSIM, but PSNR drops |
| EVP-LineSplat-v1 | 0.9122745 | 29.6463909 | 0.1388366 | Projected line gradient loss |
| EVP-LineSplat-v2 | 0.9128430 | 29.6245480 | **0.1381441** | Current main variant; best LPIPS |

Current interpretation:

- Mini-Splatting achieves the highest PSNR, but LPIPS is substantially worse.
- Edge-center adaptive improves SSIM slightly, but harms PSNR.
- EVP-LineSplat-v2 gives the best LPIPS, indicating better perceptual quality, but does not yet dominate PSNR/SSIM.
- Whole-image metrics may under-represent improvements around projected line structures.

## Playroom

Dataset path used locally:

```text
/home/ddsm/DownLoad/Confidence-Aware-LineSplat/gaussian-splatting/data
```

Evaluation split:

```text
--eval, resolution 2, test views: 29
```

| Method | SSIM ↑ | PSNR ↑ | LPIPS ↓ | Notes |
|---|---:|---:|---:|---|
| 3DGS | 0.9267502 | 30.4722576 | 0.1371734 | Original 3DGS baseline |
| No-line pure3DGS | 0.9262590 | 30.4242325 | 0.1371497 | This codebase with line guidance disabled |
| Mini-Splatting | **0.9281572** | **30.6869240** | 0.1495531 | Best PSNR/SSIM, weaker LPIPS |
| Visibility adaptive | 0.9260339 | 30.4887085 | 0.1375082 | Earlier confidence-aware line variant |
| Edge-orient | 0.9266110 | 30.3845959 | 0.1375450 | Orientation loss was not stable enough |
| Edge-center adaptive | 0.9274006 | 30.4953423 | **0.1366614** | Best LPIPS among listed methods |

Current interpretation:

- Mini-Splatting is strongest on PSNR/SSIM for this scene, but LPIPS is worse.
- Edge-center adaptive improves LPIPS and slightly improves PSNR over 3DGS.
- Orientation alignment is not recommended as a default main loss.

## Method Notes

### Why LPIPS matters here

Line-guided methods primarily target structural regions: edges, contours, and object boundaries. Whole-image PSNR can penalize small edge shifts and may prefer smoother images. LPIPS is usually more aligned with perceptual visual quality, especially when comparing local structure and sharpness.

For paper reporting, use all three metrics, but include local crop comparisons around line-rich regions.

### Known limitation: white holes on dark surfaces

Viewer inspection showed that Mini-Splatting can produce cleaner dark surfaces than 3DGS/Ours. This is likely because Mini-Splatting explicitly repairs low-alpha / low-coverage pixels by sampling from:

```text
prob = 1 - accum_alpha
```

and reinitializing Gaussians on under-covered regions.

Current EVP-LineSplat variants do not yet include an alpha coverage repair mechanism. This motivates the next step:

```text
Coverage-aware EVP-LineSplat
```

Expected components:

- Surface alpha coverage regularization.
- Optional low-alpha region repair.
- Keep EVP-v2 photometric reweighting as the line-structure guidance.

## Reproducibility Checklist

For fair comparison, use:

```text
--eval
--resolution 2
--data_device cpu
--test_iterations -1
--checkpoint_iterations 7000 15000 30000
--densify_max_points_per_stage 0
```

Render before metrics:

```bash
python render.py \
  -s <scene_path> \
  -m <model_path> \
  --iteration 30000 \
  --skip_train \
  --resolution 2 \
  --data_device cpu
```

Evaluate:

```bash
python metrics_stream.py \
  --split test \
  -m <model_path_1> <model_path_2> ...
```
