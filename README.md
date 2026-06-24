# LGDU-Splatting

**Line-Guided Densification and Unpooling for 3D Gaussian Splatting**

LGDU-Splatting is a research fork of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). It studies how reliable 3D line structures from LIMAP can guide Gaussian growth, repair local reconstruction failures, and preserve the global rendering quality of 3DGS.

The current main method is:

```text
edge-verified line confidence
+ projected line photometric reweighting
+ line-aware densification score
+ line-guided unpooling
```

Unlike early LineSplat variants that directly pulled Gaussian centers toward noisy 3D line segments, LGDU uses verified line tracks as a structural signal for **where Gaussians should densify or be newly inserted**.

## Method Overview

LGDU-Splatting uses the following pipeline:

1. LIMAP extracts 3D line tracks from a COLMAP reconstruction.
2. Line confidence is estimated from visibility and multi-view image-edge support.
3. Verified projected line regions receive confidence-aware photometric supervision.
4. During densification, Gaussians near reliable line structures receive a soft densification score boost.
5. Line-guided unpooling actively inserts new Gaussians along reliable line segments in sparse or low-opacity regions.

This makes the method especially useful for 3DGS failure regions such as:

- white holes on dark surfaces,
- under-covered structural edges,
- line-like geometry where vanilla densification is insufficient.

## Current Version

Stable branch and tag:

```text
branch: lgdu-splatting-v1
tag:    lgdu-splatting-v1.1
```

The latest tagged version includes the LGDU-Splatting v1 method and the 2026-06-24 comparison against 3DGS, Mini-Splatting, and Mip-Splatting.

| Component | Status |
|---|---|
| LIMAP `alltracks.txt` line loading | Implemented |
| Visibility-aware line confidence | Implemented |
| Multi-view edge verification | Implemented |
| Projected line photometric reweighting | Implemented |
| Line-aware densification score | Implemented |
| Line-guided unpooling | Implemented |
| Local dark / edge / hole metrics | Implemented |
| Crop exporter for qualitative comparison | Implemented |
| 7-scene comparison with Mip-Splatting | Added |

## Repository Layout

Important files added or modified for LGDU:

```text
train.py                         # 3DGS training with line-guided losses, densification, and unpooling
scene/gaussian_model.py           # Line-aware densification and line-guided unpooling
utils/line_utils.py               # LIMAP/OBJ loading, confidence, projected masks, edge support
arguments/__init__.py             # LGDU command-line arguments
metrics_stream.py                 # Streaming PSNR/SSIM/LPIPS evaluator
region_metrics.py                 # Local dark / edge / hole metrics and crop exporter
region_compare_report.py          # Markdown report generator for local metrics
check_init_ply.py                 # Utility for checking initial point cloud state
scene/dataset_readers.py          # Dataset reading adjustments
```

Original 3DGS files and license are retained. See [LICENSE.md](LICENSE.md).

## Requirements

The base environment follows 3DGS. The local experiments used an environment named `3DGS`.

```bash
conda env create --file environment.yml
conda activate 3DGS
```

OpenCV is required for edge-supported line confidence. LIMAP is used in a separate environment named `Limap` in the local experiments.

Required scene format:

```text
<scene>/
  images/
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```

Required LIMAP output:

```text
<limap_output>/alltracks.txt
```

`triangulated_lines_*.obj` is useful for visualization, but LGDU training uses `alltracks.txt` because it contains track-level visibility information.

## Generate LIMAP Lines

Example command for `nv6` line tracks:

```bash
conda activate Limap

cd "/home/ddsm/DownLoad/limap"

python runners/colmap_triangulation.py \
  -a "$(which colmap)" \
  -m "/path/to/scene/sparse/0" \
  -i "/path/to/scene/images" \
  --max_image_dim 1280 \
  -nv 6 \
  --output_dir "/path/to/limap/outputs/scene_nv6"
```

Important: `-c` / `--config_file` is for a YAML config file. Use `--output_dir` for the output folder.

## Train LGDU-Splatting

This is the current recommended LGDU-Splatting v1 configuration:

```bash
conda activate 3DGS

cd "/home/ddsm/DownLoad/Confidence-Aware-LineSplat/gaussian-splatting"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py \
  --source_path "/path/to/scene" \
  --model_path "/path/to/output/lgdu_nv6" \
  --eval \
  --disable_viewer \
  --data_device cpu \
  --resolution 2 \
  --test_iterations -1 \
  --checkpoint_iterations 7000 15000 30000 \
  --densify_max_points_per_stage 0 \
  --line_tracks_path "/path/to/limap/outputs/scene_nv6/alltracks.txt" \
  --line_min_visible_views 6 \
  --line_confidence_mode visibility \
  --line_confidence_power 1.0 \
  --line_confidence_min 0.05 \
  --line_multiview_edge_support_enable \
  --line_multiview_edge_max_views 32 \
  --line_multiview_edge_min_views 2 \
  --line_confidence_percentile 70 \
  --line_edge_sigma_px 2.0 \
  --line_edge_sample_count 16 \
  --line_edge_min_valid_ratio 0.5 \
  --line_edge_min_projected_length 4.0 \
  --line_photo_reweight_enable \
  --line_photo_lambda_adaptive \
  --line_photo_lambda_target_ratio 0.03 \
  --line_photo_lambda_adaptive_min 0.0 \
  --line_photo_lambda_adaptive_max 0.02 \
  --line_photo_start_iter 3000 \
  --line_photo_end_iter 15000 \
  --line_photo_sample_count 32 \
  --line_photo_mask_dilation_px 2 \
  --line_photo_min_valid_ratio 0.5 \
  --line_photo_min_projected_length 4.0 \
  --line_photo_charbonnier_eps 0.001 \
  --line_densify_mode score \
  --line_densify_start_iter 500 \
  --line_densify_end_iter 15000 \
  --line_densify_score_boost 0.5 \
  --line_densify_confidence_power 1.0 \
  --line_densify_confidence_max 3.0 \
  --line_densify_low_alpha_boost 0.5 \
  --line_unpool_enable \
  --line_unpool_start_iter 3000 \
  --line_unpool_end_iter 12000 \
  --line_unpool_interval 500 \
  --line_unpool_samples_per_line 2 \
  --line_unpool_max_points 512 \
  --line_unpool_candidate_factor 4 \
  --line_unpool_score_threshold 0.10 \
  --line_unpool_opacity_init 0.04 \
  --line_unpool_scale_factor 0.6 \
  --line_unpool_alpha_target 0.08 \
  --line_unpool_low_alpha_boost 0.5
```

Use the same `--resolution` as the 3DGS, Mini-Splatting, and Mip-Splatting baselines for fair comparison.

## Render

Render test views:

```bash
conda activate 3DGS

cd "/home/ddsm/DownLoad/Confidence-Aware-LineSplat/gaussian-splatting"

python render.py \
  -s "/path/to/scene" \
  -m "/path/to/output/lgdu_nv6" \
  --iteration 30000 \
  --skip_train \
  --resolution 2 \
  --data_device cpu
```

The metrics scripts expect:

```text
<model_path>/test/ours_30000/renders/
<model_path>/test/ours_30000/gt/
```

## Evaluation

Global metrics:

```bash
conda activate 3DGS

cd "/home/ddsm/DownLoad/Confidence-Aware-LineSplat/gaussian-splatting"

python metrics_stream.py \
  --split test \
  -m "/path/to/3dgs_baseline" \
     "/path/to/mini_splatting" \
     "/path/to/mip_splatting" \
     "/path/to/lgdu"
```

Local dark / edge / hole metrics and crop export:

```bash
python region_metrics.py \
  --split test \
  --regions dark edge hole \
  --hole_reference_model "/path/to/3dgs_baseline" \
  --output_dir "output/region_metrics/scene_lgdu_test" \
  --crop_export_dir "output/region_crops/scene_lgdu_test" \
  --num_crops 12 \
  --crop_region hole \
  -m "/path/to/3dgs_baseline" \
     "/path/to/mini_splatting" \
     "/path/to/mip_splatting" \
     "/path/to/lgdu"
```

Generate a Markdown report:

```bash
python region_compare_report.py \
  --results "output/region_metrics/scene_lgdu_test/region_results_test.json" \
  --per_view "output/region_metrics/scene_lgdu_test/region_per_view_test.json" \
  --baseline "scene_eval_baseline_3dgs" \
  --methods \
    "scene_eval_baseline_3dgs" \
    "scene_eval_mini_splatting" \
    "scene_eval_mip_splatting" \
    "scene_eval_lgdu" \
  --regions dark edge hole \
  --crop_dir "output/region_crops/scene_lgdu_test" \
  --output "output/region_metrics/scene_lgdu_test/region_comparison_report.md"
```

Metrics:

- PSNR: higher is better.
- SSIM: higher is better.
- LPIPS: lower is better.

Whole-image averages can dilute structural improvements. For line-guided methods, the local dark / edge / hole metrics and exported crops are important.

## Experiment Summary

The following results summarize the 2026-06-24 fair comparison. All methods use the same scene, split, resolution, and 30k iteration evaluation for each scene.

```text
3DGS: vanilla 3D Gaussian Splatting
Mini: Mini-Splatting
Mip : Mip-Splatting
LGDU: LGDU-Splatting v1.1
```

Detailed reports used to update this README:

```text
/home/ddsm/files/train/global_comparison_2026-06-24.md
/home/ddsm/files/train/local_region_comparison_2026-06-24.md
```

### Global Metrics

| Scene | Resolution | Method | PSNR | SSIM | LPIPS |
|---|---:|---|---:|---:|---:|
| drjohnson | r2 | 3DGS | 29.6548 | 0.91286 | 0.13901 |
| drjohnson | r2 | Mini | 29.7897 | 0.91110 | 0.15960 |
| drjohnson | r2 | Mip | 28.9641 | 0.90432 | 0.15175 |
| drjohnson | r2 | LGDU | 29.7431 | 0.91284 | 0.13807 |
| playroom | r2 | 3DGS | 30.4723 | 0.92675 | 0.13717 |
| playroom | r2 | Mini | 30.6869 | 0.92816 | 0.14955 |
| playroom | r2 | Mip | 30.5454 | 0.92634 | 0.13459 |
| playroom | r2 | LGDU | 30.4856 | 0.92600 | 0.13802 |
| counter | r2 | 3DGS | 30.3106 | 0.94141 | 0.06218 |
| counter | r2 | Mini | 29.2275 | 0.92343 | 0.08112 |
| counter | r2 | Mip | 30.3475 | 0.94315 | 0.05832 |
| counter | r2 | LGDU | 30.2934 | 0.94173 | 0.06201 |
| room | r2 | 3DGS | 33.1566 | 0.96414 | 0.04934 |
| room | r2 | Mini | 32.1084 | 0.95551 | 0.06329 |
| room | r2 | Mip | 33.2677 | 0.96315 | 0.05022 |
| room | r2 | LGDU | 33.2952 | 0.96493 | 0.04894 |
| train | r2 | 3DGS | 22.6612 | 0.86717 | 0.12462 |
| train | r2 | Mini | 22.0646 | 0.84028 | 0.17184 |
| train | r2 | Mip | 22.5844 | 0.87292 | 0.12094 |
| train | r2 | LGDU | 22.8251 | 0.86792 | 0.12289 |
| truck | r1 | 3DGS | 25.5370 | 0.88511 | 0.14174 |
| truck | r1 | Mini | 25.1162 | 0.87396 | 0.15896 |
| truck | r1 | Mip | 25.6489 | 0.89245 | 0.12324 |
| truck | r1 | LGDU | 25.4312 | 0.88497 | 0.14036 |
| kitchen | r1 | 3DGS | 32.1701 | 0.94846 | 0.06685 |
| kitchen | r1 | Mini | 31.0743 | 0.93358 | 0.08861 |
| kitchen | r1 | Mip | 31.9186 | 0.94943 | 0.06381 |
| kitchen | r1 | LGDU | 32.3517 | 0.94916 | 0.06613 |

Global metric best-count summary:

| Metric | 3DGS | Mini | Mip | LGDU |
|---|---:|---:|---:|---:|
| PSNR best count | 0 | 2 | 2 | 3 |
| SSIM best count | 1 | 1 | 4 | 1 |
| LPIPS best count | 0 | 0 | 5 | 2 |

### LGDU vs 3DGS Global Delta

| Scene | Delta PSNR | Delta SSIM | Delta LPIPS | Summary |
|---|---:|---:|---:|---|
| drjohnson | +0.0884 | -0.00002 | -0.00095 | PSNR/LPIPS improve, SSIM nearly tied |
| playroom | +0.0134 | -0.00075 | +0.00085 | Global metrics nearly tied; local hole quality improves strongly |
| counter | -0.0172 | +0.00032 | -0.00017 | Nearly tied; SSIM/LPIPS slightly improve |
| room | +0.1385 | +0.00079 | -0.00041 | All global metrics improve |
| train | +0.1639 | +0.00074 | -0.00173 | PSNR/SSIM/LPIPS improve |
| truck | -0.1058 | -0.00014 | -0.00138 | LPIPS improves, PSNR/SSIM slightly decrease |
| kitchen | +0.1816 | +0.00070 | -0.00071 | PSNR/SSIM/LPIPS improve |

### Hole-Region Metrics

| Scene | Method | Hole PSNR | Hole SSIM | Hole LPIPS |
|---|---|---:|---:|---:|
| drjohnson | 3DGS | 10.3297 | 0.97418 | 0.03113 |
| drjohnson | Mini | 12.8215 | 0.97867 | 0.02499 |
| drjohnson | Mip | 11.4775 | 0.97749 | 0.02805 |
| drjohnson | LGDU | 11.9538 | 0.97859 | 0.02791 |
| playroom | 3DGS | 11.7252 | 0.98029 | 0.02893 |
| playroom | Mini | 12.4723 | 0.98061 | 0.02987 |
| playroom | Mip | 13.2995 | 0.98168 | 0.02518 |
| playroom | LGDU | 12.7615 | 0.98153 | 0.02471 |
| counter | 3DGS | 14.0211 | 0.98625 | 0.01173 |
| counter | Mini | 15.7869 | 0.99001 | 0.00719 |
| counter | Mip | 16.4027 | 0.99151 | 0.00621 |
| counter | LGDU | 15.3808 | 0.99005 | 0.00784 |
| room | 3DGS | 14.0549 | 0.98580 | 0.01631 |
| room | Mini | 13.9122 | 0.98629 | 0.01394 |
| room | Mip | 13.6803 | 0.98417 | 0.01504 |
| room | LGDU | 17.2360 | 0.98618 | 0.01230 |
| train | 3DGS | 12.4730 | 0.99034 | 0.00841 |
| train | Mini | 12.8318 | 0.99059 | 0.00882 |
| train | Mip | 13.5427 | 0.99170 | 0.00751 |
| train | LGDU | 13.0838 | 0.99114 | 0.00801 |
| truck | 3DGS | 12.3615 | 0.99269 | 0.00642 |
| truck | Mini | 13.1713 | 0.99355 | 0.00635 |
| truck | Mip | 13.4450 | 0.99385 | 0.00555 |
| truck | LGDU | 12.7707 | 0.99318 | 0.00614 |
| kitchen | 3DGS | 9.8581 | 0.97854 | 0.02707 |
| kitchen | Mini | 16.5058 | 0.99157 | 0.01069 |
| kitchen | Mip | 16.7310 | 0.99151 | 0.01025 |
| kitchen | LGDU | 11.4957 | 0.98466 | 0.01993 |

LGDU improves hole-region PSNR and reduces hole-region LPIPS over 3DGS on all seven scenes:

```text
drjohnson: +1.6241
playroom : +1.0363
counter  : +1.3597
room     : +3.1811
train    : +0.6108
truck    : +0.4092
kitchen  : +1.6376
```

### Local Region Findings

| Scene | Main local observation |
|---|---|
| drjohnson | LGDU improves edge and hole regions; global PSNR/LPIPS also improve over 3DGS. |
| playroom | LGDU has the best hole LPIPS and edge LPIPS, while Mip-Splatting has stronger hole PSNR/SSIM. |
| counter | Mip-Splatting leads most local metrics; LGDU still improves hole quality over 3DGS. |
| room | LGDU is the strongest local repair case, with large hole PSNR/LPIPS gains over 3DGS and Mip-Splatting. |
| train | Mip-Splatting leads local metrics, but LGDU improves hole/dark/edge LPIPS over 3DGS. |
| truck | Mip-Splatting is strongest; LGDU is a mild but positive local improvement over 3DGS. |
| kitchen | LGDU leads edge-region metrics and improves holes over 3DGS, but Mip/Mini repair holes more aggressively. |

## Current Conclusion

LGDU-Splatting is best understood as a structure-guided improvement to vanilla 3DGS:

```text
It preserves or improves whole-image 3DGS quality on most scenes,
and consistently improves local hole-region quality over 3DGS.
```

The 2026-06-24 comparison shows:

- LGDU improves PSNR over 3DGS on 5/7 scenes.
- LGDU improves LPIPS over 3DGS on 6/7 scenes.
- LGDU improves hole-region PSNR and hole-region LPIPS over 3DGS on 7/7 scenes.
- LGDU obtains the most PSNR wins among the four methods.
- Mip-Splatting remains a very strong baseline, especially for SSIM/LPIPS and dark/hole-region perceptual metrics.
- LGDU and Mip-Splatting are complementary: LGDU is strongest as a line-structure-aware local repair method, while Mip-Splatting is strongest as a scale-aware perceptual baseline.

Recommended paper positioning:

```text
LGDU-Splatting consistently improves vanilla 3DGS in local hole/edge failure regions,
while preserving competitive global rendering quality. It is complementary to
Mip-Splatting, whose scale-aware filtering remains strong for perceptual metrics.
```

## Git Remotes

This local fork uses:

```text
origin   git@github.com:Ddsmmmm/LineSplat.git
upstream https://github.com/graphdeco-inria/gaussian-splatting.git
```

## Citation and License

This project is derived from 3D Gaussian Splatting. The original license is retained in [LICENSE.md](LICENSE.md).

If you use this repository, cite the original 3DGS paper:

```bibtex
@Article{kerbl3Dgaussians,
  author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimkuehler, Thomas and Drettakis, George},
  title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal = {ACM Transactions on Graphics},
  number  = {4},
  volume  = {42},
  month   = {July},
  year    = {2023},
  url     = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
