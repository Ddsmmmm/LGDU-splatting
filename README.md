# EVP-LineSplat

EVP-LineSplat is a research fork of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) for studying line-guided Gaussian Splatting reconstruction.

The current implementation focuses on **edge-verified projected line guidance**:

- LIMAP extracts 3D line tracks from COLMAP reconstructions.
- Multi-view image edge support filters noisy 3D lines.
- Reliable projected line masks guide 3DGS optimization.
- The current main variant, **EVP-LineSplat-v2**, uses confidence-aware photometric reweighting near verified projected line regions.

This repository is experimental. The current goal is not to replace 3DGS with a fully finalized method, but to provide a controlled implementation for comparing line-guided losses, confidence filtering, and structure-aware reconstruction behavior.

## Method Status

Implemented variants:

| Variant | Main idea | Status |
|---|---|---|
| No-line | Run this codebase with all line losses disabled | Used as codebase sanity baseline |
| Edge-center adaptive | Weight Gaussian center-to-line loss by line confidence and projected edge support | Improves some SSIM/LPIPS cases, but can hurt PSNR |
| EVP-LineSplat-v1 | Multi-view verified projected line mask + image gradient loss | More conservative than center loss, but weak global metric gains |
| EVP-LineSplat-v2 | Multi-view verified projected line mask + photometric reweighting | Current main variant; best observed LPIPS on `drjohnson` |

Planned direction:

- Coverage-aware alpha regularization to reduce transparent holes on dark or low-texture surfaces.
- Line-region metrics to evaluate projected-line neighborhoods separately from whole-image metrics.
- More scenes beyond Deep Blending / Tanks and Temples style indoor scenes.

## Repository Layout

Important files added or modified for this project:

```text
train.py                       # 3DGS training with line-guided losses
utils/line_utils.py             # LIMAP/OBJ line loading, confidence, projection masks
arguments/__init__.py           # EVP-LineSplat command-line arguments
metrics_stream.py               # Streaming PSNR/SSIM/LPIPS evaluation
check_init_ply.py               # Utility for checking initial point cloud state
scene/gaussian_model.py         # Densification-related extensions
scene/dataset_readers.py        # Dataset reading adjustments
```

Original 3DGS files and license are retained. See [LICENSE.md](LICENSE.md).

## Requirements

The base environment follows 3DGS. The current `environment.yml` also includes OpenCV, which is required for edge-supported line confidence.

```bash
conda env create --file environment.yml
conda activate gaussian_splatting
```

In the local experiments for this project, the environment name was `3DGS`. If your environment has a different name, replace `conda activate 3DGS` in the examples below.

Required external inputs:

- COLMAP-style dataset:

```text
<scene>/
  images/
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```

- LIMAP 3D line tracks, usually:

```text
<limap_output>/alltracks.txt
```

## Generating LIMAP Lines

Example LIMAP command:

```bash
conda activate Limap

cd /home/ddsm/DownLoad/limap

python runners/colmap_triangulation.py \
  -a "$(which colmap)" \
  -m "/path/to/scene/sparse/0" \
  -i "/path/to/scene/images" \
  --max_image_dim 1280 \
  -nv 6 \
  -c "/path/to/limap/output/scene_nv6"
```

The important output for EVP-LineSplat is:

```text
/path/to/limap/output/scene_nv6/alltracks.txt
```

`triangulated_lines_*.obj` can be useful for visualization, but the current recommended training path uses `alltracks.txt` because it contains track-level visibility information.

## Training

### 1. Pure no-line baseline using this codebase

Use this to confirm the fork behaves close to original 3DGS when line guidance is disabled.

```bash
conda activate 3DGS

cd "/home/ddsm/DownLoad/Confidence-Aware-LineSplat/gaussian-splatting"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py \
  --source_path "/path/to/scene" \
  --model_path "/path/to/output/no_line_pure3dgs_res2" \
  --eval \
  --disable_viewer \
  --data_device cpu \
  --resolution 2 \
  --test_iterations -1 \
  --checkpoint_iterations 7000 15000 30000 \
  --densify_max_points_per_stage 0
```

### 2. EVP-LineSplat-v2

This is the current recommended Ours configuration. It disables the old 3D center-to-line loss and uses projected line photometric reweighting.

```bash
conda activate 3DGS

cd "/home/ddsm/DownLoad/Confidence-Aware-LineSplat/gaussian-splatting"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py \
  --source_path "/path/to/scene" \
  --model_path "/path/to/output/evp_v2_photo_nv6_nocap_res2" \
  --eval \
  --disable_viewer \
  --data_device cpu \
  --resolution 2 \
  --test_iterations -1 \
  --checkpoint_iterations 7000 15000 30000 \
  --densify_max_points_per_stage 0 \
  --line_tracks_path "/path/to/limap/output/scene_nv6/alltracks.txt" \
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
  --line_photo_charbonnier_eps 0.001
```

For full debug logging every 100 iterations:

```bash
--progress_full_log_interval 100
```

## Rendering

Render test views:

```bash
conda activate 3DGS

cd "/home/ddsm/DownLoad/Confidence-Aware-LineSplat/gaussian-splatting"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render.py \
  -s "/path/to/scene" \
  -m "/path/to/output/evp_v2_photo_nv6_nocap_res2" \
  --iteration 30000 \
  --skip_train \
  --resolution 2 \
  --data_device cpu
```

The metrics script expects:

```text
<model_path>/test/ours_30000/renders/
<model_path>/test/ours_30000/gt/
```

## Evaluation

Use the streaming evaluator to avoid loading all images at once:

```bash
conda activate 3DGS

cd "/home/ddsm/DownLoad/Confidence-Aware-LineSplat/gaussian-splatting"

python metrics_stream.py \
  --split test \
  -m "/path/to/3dgs_baseline" \
     "/path/to/no_line_pure3dgs" \
     "/path/to/evp_v2_photo_nv6_nocap_res2"
```

Metrics:

- PSNR: higher is better.
- SSIM: higher is better.
- LPIPS: lower is better.

For perceptual quality, LPIPS and local crop comparisons are usually more informative than PSNR alone. For line-guided methods, whole-image averages can dilute improvements that occur mainly around structural edges.

See [results.md](results.md) for current experiment records.

## Current Findings

Observed behavior so far:

- Direct 3D center-to-line loss can improve some structure scores but may reduce PSNR.
- Image-space projected line losses are safer than forcing Gaussian centers onto noisy 3D line segments.
- EVP-LineSplat-v2 achieved the best LPIPS among tested variants on `drjohnson`, but did not yet dominate PSNR/SSIM.
- Mini-Splatting can look visually cleaner in the viewer because it explicitly repairs low-alpha / low-coverage regions; EVP-LineSplat does not yet include that mechanism.

Next implementation target:

```text
Coverage-aware EVP-LineSplat:
  EVP-v2 line photometric reweighting
  + alpha coverage regularization
  + optional low-alpha surface repair
```

This target is motivated by observed white holes on dark table surfaces in 3DGS/Ours viewer outputs.

## Git Remotes

This local fork uses:

```text
origin   https://github.com/Ddsmmmm/LineSplat.git
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
