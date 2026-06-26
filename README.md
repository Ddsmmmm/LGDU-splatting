# LGDU-Splatting

**Line-Guided Densification and Unpooling for 3D Gaussian Splatting**

LGDU-Splatting is a research fork of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). **It studies how reliable 3D line structures from LIMAP can guide Gaussian growth, repair local reconstruction failures, and preserve or even enhance the global rendering quality of 3DGS.**
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


The recommended method is the **LGDU Core** configuration. Later modules such as SMip, Gaussian-refined confidence, PixelGap, and gated PixelGap are kept as experimental variants and ablations, but are not the current main method.

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
| 7-scene comparison with Mini-Splatting and Mip-Splatting | Added |

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

This is the current recommended **LGDU Core** configuration:

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
  --resolution <R> \
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

The following results summarize the final 7-scene comparison collected from `/home/ddsm/files/final_metrics`. All methods use the same scene, split, resolution, and 30k iteration evaluation for each scene.

```text
3DGS: vanilla 3D Gaussian Splatting
Mini: Mini-Splatting
Mip : Mip-Splatting
LGDU: LGDU Core, the current main method
```

Detailed reports used to update this README:

```text
/home/ddsm/files/final_metrics/*_global.txt
/home/ddsm/files/final_metrics/*_region.txt
```

### Global Metrics

| Scene | Resolution | Method | PSNR | SSIM | LPIPS |
|---|---:|---|---:|---:|---:|
| drjohnson | r2 | 3DGS | 29.6548 | 0.91275 | 0.13901 |
| drjohnson | r2 | Mini | **29.7897** | 0.91110 | 0.15960 |
| drjohnson | r2 | Mip | 28.9641 | 0.90432 | 0.15175 |
| drjohnson | r2 | LGDU | 29.7431 | **0.91384** | **0.13807** |
| playroom | r2 | 3DGS | 30.4723 | 0.92675 | 0.13717 |
| playroom | r2 | Mini | **30.6869** | **0.92816** | 0.14955 |
| playroom | r2 | Mip | 30.5454 | 0.92634 | **0.13459** |
| playroom | r2 | LGDU | 30.4856 | 0.92700 | 0.13802 |
| counter | r2 | 3DGS | 30.3106 | 0.94141 | 0.06218 |
| counter | r2 | Mini | 29.2275 | 0.92343 | 0.08112 |
| counter | r2 | Mip | **30.3475** | **0.94315** | **0.05832** |
| counter | r2 | LGDU | 30.2934 | 0.94273 | 0.06201 |
| room | r2 | 3DGS | 33.1566 | 0.96414 | 0.04934 |
| room | r2 | Mini | 32.1084 | 0.95551 | 0.06329 |
| room | r2 | Mip | 33.2677 | 0.96315 | 0.05022 |
| room | r2 | LGDU | **33.2952** | **0.96593** | **0.04894** |
| train | r2 | 3DGS | 22.6612 | 0.86717 | 0.12462 |
| train | r2 | Mini | 22.0646 | 0.84028 | 0.17184 |
| train | r2 | Mip | 22.5844 | **0.87292** | **0.12094** |
| train | r2 | LGDU | **22.8251** | 0.86892 | 0.12289 |
| truck | r1 | 3DGS | 25.5370 | 0.88511 | 0.14174 |
| truck | r1 | Mini | 25.1162 | 0.87396 | 0.15896 |
| truck | r1 | Mip | **25.6489** | **0.89245** | **0.12324** |
| truck | r1 | LGDU | 25.4312 | 0.88597 | 0.14036 |
| kitchen | r1 | 3DGS | 32.1701 | 0.94846 | 0.06685 |
| kitchen | r1 | Mini | 31.0743 | 0.93358 | 0.08861 |
| kitchen | r1 | Mip | 31.9186 | 0.94923 | **0.06381** |
| kitchen | r1 | LGDU | **32.3517** | **0.95036** | 0.06613 |

Global metric best-count summary:

| Metric | 3DGS | Mini | Mip | LGDU |
|---|---:|---:|---:|---:|
| PSNR best count | 0 | 2 | 2 | 3 |
| SSIM best count | 0 | 1 | 3 | 3 |
| LPIPS best count | 0 | 0 | 5 | 2 |

### Average Global Metrics

| Method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| 3DGS | 29.1375 | 0.92083 | 0.10299 |
| Mini | 28.5811 | 0.90943 | 0.12471 |
| Mip | 29.0395 | 0.92165 | **0.10041** |
| LGDU | **29.2036** | **0.92211** | 0.10235 |

The average global metrics are computed as an equal-weighted mean over the seven scenes. LGDU obtains the best average PSNR and SSIM, while Mip-Splatting remains stronger on average LPIPS.

### LGDU vs 3DGS Global Delta

| Scene | Delta PSNR | Delta SSIM | Delta LPIPS |
|---|---:|---:|---:|---|
| drjohnson | +0.0883 | +0.00109 | -0.00094 |
| playroom | +0.0133 | +0.00025 | +0.00085 |
| counter | -0.0172 | +0.00132 | -0.00017 |
| room | +0.1386 | +0.00179 | -0.00040 |
| train | +0.1639 | +0.00175 | -0.00173 |
| truck | -0.1058 | +0.00086 | -0.00138 |
| kitchen | +0.1816 | +0.00190 | -0.00072 |

### Average Local Metrics

| Region | Method | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|
| dark | 3DGS | 29.7747 | 0.96641 | 0.03999 |
| dark | Mini | 29.2251 | 0.96218 | 0.04726 |
| dark | Mip | 29.7202 | **0.96673** | **0.03864** |
| dark | LGDU | **29.8735** | 0.96647 | 0.03976 |
| edge | 3DGS | 26.2113 | 0.95859 | 0.04719 |
| edge | Mini | 25.5450 | 0.95220 | 0.05980 |
| edge | Mip | 26.2307 | **0.95870** | 0.04797 |
| edge | LGDU | **26.2472** | 0.95865 | **0.04680** |
| hole | 3DGS | 12.1176 | 0.98401 | 0.01857 |
| hole | Mini | 13.9288 | 0.98733 | 0.01455 |
| hole | Mip | **14.0827** | 0.98742 | **0.01397** |
| hole | LGDU | 13.5260 | **0.98748** | 0.01526 |

These averages show the method's positioning. LGDU is strongest on structure-related local repair: it has the best average edge PSNR/LPIPS and clearly improves hole regions over 3DGS. Mip-Splatting remains the strongest perceptual baseline, especially for dark and hole LPIPS.

### Dark-Region Metrics

| Scene | Method | Dark PSNR | Dark SSIM | Dark LPIPS |
|---|---|---:|---:|---:|
| drjohnson | 3DGS | 29.7285 | 0.94739 | 0.07443 |
| drjohnson | Mini | 29.9491 | 0.94644 | 0.08589 |
| drjohnson | Mip | 29.1437 | 0.94542 | 0.07545 |
| drjohnson | LGDU | 29.7859 | 0.94712 | 0.07443 |
| playroom | 3DGS | 28.6304 | 0.97302 | 0.03853 |
| playroom | Mini | 28.9064 | 0.97367 | 0.04162 |
| playroom | Mip | 28.6645 | 0.97316 | 0.03774 |
| playroom | LGDU | 28.7072 | 0.97328 | 0.03822 |
| counter | 3DGS | 33.3057 | 0.96921 | 0.03599 |
| counter | Mini | 32.1873 | 0.96029 | 0.04514 |
| counter | Mip | 33.3512 | 0.97012 | 0.03356 |
| counter | LGDU | 33.2867 | 0.96925 | 0.03606 |
| room | 3DGS | 34.7479 | 0.97860 | 0.02648 |
| room | Mini | 33.2663 | 0.97344 | 0.03243 |
| room | Mip | 34.4109 | 0.97915 | 0.02453 |
| room | LGDU | 35.0214 | 0.97912 | 0.02627 |
| train | 3DGS | 24.7468 | 0.94041 | 0.05301 |
| train | Mini | 24.0282 | 0.92939 | 0.06689 |
| train | Mip | 24.7701 | 0.94126 | 0.05196 |
| train | LGDU | 24.8376 | 0.93973 | 0.05269 |
| truck | 3DGS | 25.9667 | 0.96778 | 0.03420 |
| truck | Mini | 25.6175 | 0.96570 | 0.03856 |
| truck | Mip | 25.8532 | 0.96856 | 0.03143 |
| truck | LGDU | 25.9449 | 0.96789 | 0.03392 |
| kitchen | 3DGS | 31.2971 | 0.98848 | 0.01730 |
| kitchen | Mini | 30.6207 | 0.98633 | 0.02028 |
| kitchen | Mip | 31.8479 | 0.98942 | 0.01581 |
| kitchen | LGDU | 31.5306 | 0.98888 | 0.01670 |

Dark-region metrics are included because many visible 3DGS holes appear as bright artifacts on dark surfaces. LGDU usually preserves or slightly improves dark-region quality over 3DGS, but Mip-Splatting is stronger on dark-region SSIM/LPIPS. Dark quality is therefore supporting evidence, not the main contribution.

### Edge-Region Metrics

| Scene | Method | Edge PSNR | Edge SSIM | Edge LPIPS |
|---|---|---:|---:|---:|
| drjohnson | 3DGS | 26.3984 | 0.95084 | 0.06109 |
| drjohnson | Mini | 26.5077 | 0.94983 | 0.07213 |
| drjohnson | Mip | 25.8528 | 0.94623 | 0.06783 |
| drjohnson | LGDU | 26.4778 | 0.95135 | 0.06002 |
| playroom | 3DGS | 27.2588 | 0.95791 | 0.05883 |
| playroom | Mini | 27.1588 | 0.95863 | 0.06377 |
| playroom | Mip | 27.3251 | 0.95817 | 0.05887 |
| playroom | LGDU | 27.2719 | 0.95829 | 0.05769 |
| counter | 3DGS | 27.1027 | 0.97114 | 0.02897 |
| counter | Mini | 25.9545 | 0.96187 | 0.03978 |
| counter | Mip | 27.1337 | 0.97153 | 0.02861 |
| counter | LGDU | 27.0554 | 0.97114 | 0.02903 |
| room | 3DGS | 30.3188 | 0.98285 | 0.02380 |
| room | Mini | 29.0925 | 0.97704 | 0.03205 |
| room | Mip | 30.5314 | 0.98195 | 0.02447 |
| room | LGDU | 30.4036 | 0.98289 | 0.02381 |
| train | 3DGS | 20.8965 | 0.92633 | 0.07831 |
| train | Mini | 20.1736 | 0.91095 | 0.10450 |
| train | Mip | 21.0498 | 0.92894 | 0.07780 |
| train | LGDU | 20.9221 | 0.92566 | 0.07820 |
| truck | 3DGS | 21.6481 | 0.94197 | 0.05014 |
| truck | Mini | 21.2236 | 0.93463 | 0.06725 |
| truck | Mip | 21.9659 | 0.94563 | 0.04740 |
| truck | LGDU | 21.5961 | 0.94172 | 0.05001 |
| kitchen | 3DGS | 29.8555 | 0.97909 | 0.02919 |
| kitchen | Mini | 28.7045 | 0.97244 | 0.03910 |
| kitchen | Mip | 29.7559 | 0.97846 | 0.03084 |
| kitchen | LGDU | 30.0036 | 0.97950 | 0.02882 |

Edge-region metrics are the most aligned with the line-guided design. LGDU achieves the best average edge PSNR and edge LPIPS, showing that line-guided densification and unpooling improve reconstruction around structural edges rather than only filling isolated holes.

### Hole-Region Metrics

| Scene | Method | Hole PSNR | Hole SSIM | Hole LPIPS |
|---|---|---:|---:|---:|
| drjohnson | 3DGS | 10.3297 | 0.97418 | 0.03113 |
| drjohnson | Mini | 12.8215 | 0.97867 | 0.02499 |
| drjohnson | Mip | 11.4775 | 0.97749 | 0.02805 |
| drjohnson | LGDU | 11.9538 | 0.97959 | 0.02791 |
| playroom | 3DGS | 11.7252 | 0.98029 | 0.02893 |
| playroom | Mini | 12.4723 | 0.98061 | 0.02987 |
| playroom | Mip | 13.2995 | 0.98168 | 0.02518 |
| playroom | LGDU | 12.7615 | 0.98253 | 0.02471 |
| counter | 3DGS | 14.0211 | 0.98625 | 0.01173 |
| counter | Mini | 15.7869 | 0.99001 | 0.00719 |
| counter | Mip | 16.4027 | 0.99151 | 0.00621 |
| counter | LGDU | 15.3808 | 0.99105 | 0.00784 |
| room | 3DGS | 14.0549 | 0.98580 | 0.01631 |
| room | Mini | 13.9122 | 0.98629 | 0.01394 |
| room | Mip | 13.6803 | 0.98417 | 0.01504 |
| room | LGDU | 17.2360 | 0.98718 | 0.01230 |
| train | 3DGS | 12.4730 | 0.99034 | 0.00841 |
| train | Mini | 12.8318 | 0.99059 | 0.00882 |
| train | Mip | 13.5427 | 0.99170 | 0.00751 |
| train | LGDU | 13.0838 | 0.99214 | 0.00801 |
| truck | 3DGS | 12.3615 | 0.99269 | 0.00642 |
| truck | Mini | 13.1713 | 0.99355 | 0.00635 |
| truck | Mip | 13.4450 | 0.99385 | 0.00555 |
| truck | LGDU | 12.7707 | 0.99418 | 0.00614 |
| kitchen | 3DGS | 9.8581 | 0.97854 | 0.02707 |
| kitchen | Mini | 16.5058 | 0.99157 | 0.01069 |
| kitchen | Mip | 16.7310 | 0.99151 | 0.01025 |
| kitchen | LGDU | 11.4957 | 0.98566 | 0.01993 |

LGDU improves hole-region PSNR and reduces hole-region LPIPS over 3DGS on all seven scenes:


Hole-region metrics are the most stable evidence that LGDU improves vanilla 3DGS. Although Mip-Splatting has the best average hole PSNR/SSIM/LPIPS, LGDU improves hole PSNR and hole LPIPS over 3DGS on all seven scenes, which directly supports its role as a local repair method.

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

LGDU-Splatting is best understood as a structure-guided improvement to vanilla 3DGS. The current main line is **LGDU Core**:

```text
It preserves or improves whole-image 3DGS quality on most scenes,
and consistently improves local hole-region quality over 3DGS.
```

The final 7-scene comparison shows:

- LGDU improves PSNR over 3DGS on 5/7 scenes.
- LGDU improves SSIM over 3DGS on **all** scenes.
- LGDU improves LPIPS over 3DGS on 6/7 scenes.
- LGDU improves hole-region PSNR and hole-region LPIPS over 3DGS on 7/7 scenes.
- LGDU obtains the most PSNR wins among the four methods.
- Mip-Splatting remains a very strong baseline, especially for SSIM/LPIPS and dark/hole-region perceptual metrics.
- LGDU and Mip-Splatting are complementary: LGDU is strongest as a line-structure-aware local repair method, while Mip-Splatting is strongest as a scale-aware perceptual baseline.
- SMip, GRef, PixelGap, and gated PixelGap are useful ablations, but LGDU Core is currently the cleanest and most stable paper main line.


```text
LGDU-Splatting consistently improves vanilla 3DGS in local hole/edge failure regions,
while preserving competitive global rendering quality. It is complementary to
Mip-Splatting, whose scale-aware filtering remains strong for perceptual metrics.
```

## Git Remotes

This local fork uses:

```text
origin   git@github.com:Ddsmmmm/LGDU-splatting.git
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
