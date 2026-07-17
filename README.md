# LGDU-Splatting

**Line-Guided Densification and Unpooling for 3D Gaussian Splatting**

LGDU-Splatting is a research fork of [3D Gaussian Splatting (3DGS)](https://github.com/graphdeco-inria/gaussian-splatting) that uses reliable 3D line structures reconstructed by LIMAP to guide Gaussian growth. Instead of directly pulling Gaussian centers toward potentially noisy line segments, LGDU treats verified line tracks as structural evidence for **where to strengthen supervision, where to densify, and where to insert new Gaussians**.

The current main method is **LGDU Core**:

```text
edge-verified line confidence
+ projected line photometric reweighting
+ line-aware densification score
+ line-guided unpooling
```

## Highlights

- **Consistent improvement over vanilla 3DGS:** LGDU improves PSNR, SSIM, and LPIPS on all seven evaluated scenes.
- **Strong global rendering quality:** LGDU achieves the best seven-scene average PSNR and SSIM among 3DGS, Mini-Splatting, Mip-Splatting, and LGDU.
- **Structure-aware growth without hard geometric attraction:** line tracks guide supervision and Gaussian allocation without forcing centers onto noisy reconstructed segments.
- **Confidence-aware line filtering:** track visibility and multi-view image-edge support suppress unreliable 3D lines before they affect training.
- **End-to-end experimental workflow:** the repository contains line-aware training, rendering, streaming evaluation, and comparison utilities.

## Results at a Glance

The evaluation covers seven scenes from Mip-NeRF 360, Deep Blending, and Tanks and Temples. All methods use the same scene data, official test split, resolution, and 30k-iteration evaluation setting.

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| 3DGS | 29.1375 | 0.92084 | 0.10299 |
| Mini-Splatting | 28.5811 | 0.90943 | 0.12471 |
| Mip-Splatting | 29.0324 | 0.92154 | **0.10041** |
| **LGDU Core** | **29.2821** | **0.92574** | 0.10163 |

LGDU obtains the best PSNR on five of seven scenes and the best SSIM on all seven. Mip-Splatting remains the strongest baseline for average LPIPS.

### Unified Qualitative Comparison

The following atlas consolidates the latest qualitative comparisons for `counter`, `drjohnson`, `kitchen`, `playroom`, `room`, and `train`. Each group compares GT, 3DGS, Mini-Splatting, Mip-Splatting, and LGDU under the same evaluation setup.

![Unified qualitative comparison of GT, 3DGS, Mini-Splatting, Mip-Splatting, and LGDU](assets/comparisons/sum_submit.png)

### Dataset-Level Average Metrics

| Dataset | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|
| Mip-NeRF 360 | 3DGS | 31.8791 | 0.95134 | 0.05946 |
| Mip-NeRF 360 | Mini | 30.8034 | 0.93751 | 0.07767 |
| Mip-NeRF 360 | Mip | 31.8446 | 0.95158 | **0.05745** |
| Mip-NeRF 360 | **LGDU** | **32.0735** | **0.95268** | 0.05896 |
| Deep Blending | 3DGS | 30.0636 | 0.91981 | 0.13809 |
| Deep Blending | Mini | **30.2383** | 0.91963 | 0.15458 |
| Deep Blending | Mip | 29.7298 | 0.91533 | 0.14317 |
| Deep Blending | **LGDU** | 30.1914 | **0.92619** | **0.13714** |
| Tanks and Temples | 3DGS | 24.0991 | 0.87614 | 0.13318 |
| Tanks and Temples | Mini | 23.5904 | 0.85712 | 0.16540 |
| Tanks and Temples | Mip | 24.1167 | 0.88269 | **0.12209** |
| Tanks and Temples | **LGDU** | **24.1859** | **0.88490** | 0.13011 |

Mip-NeRF 360 averages use `counter`, `kitchen`, and `room`; Deep Blending averages use `drjohnson` and `playroom`; Tanks and Temples averages use `truck` and `train`.

<details>
<summary><strong>Per-scene global metrics</strong></summary>

| Scene | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|
| counter | 3DGS | 30.3106 | 0.94141 | 0.06218 |
| counter | Mini | 29.2275 | 0.92343 | 0.08112 |
| counter | Mip | 30.3475 | 0.94315 | **0.05832** |
| counter | **LGDU** | **30.4556** | **0.94376** | 0.06201 |
| kitchen | 3DGS | 32.1701 | 0.94846 | 0.06685 |
| kitchen | Mini | 31.0743 | 0.93358 | 0.08861 |
| kitchen | Mip | 31.9186 | 0.94843 | **0.06381** |
| kitchen | **LGDU** | **32.4697** | **0.94934** | 0.06613 |
| room | 3DGS | 33.1566 | 0.96414 | 0.04934 |
| room | Mini | 32.1084 | 0.95551 | 0.06329 |
| room | Mip | 33.2677 | 0.96315 | 0.05022 |
| room | **LGDU** | **33.2952** | **0.96493** | **0.04874** |
| truck | 3DGS | 25.5370 | 0.88511 | 0.14174 |
| truck | Mini | 25.1162 | 0.87396 | 0.15896 |
| truck | Mip | **25.6489** | 0.89245 | **0.12324** |
| truck | **LGDU** | 25.5466 | **0.89555** | 0.13933 |
| drjohnson | 3DGS | 29.6548 | 0.91286 | 0.13901 |
| drjohnson | Mini | 29.7897 | 0.91110 | 0.15960 |
| drjohnson | Mip | 28.9141 | 0.90432 | 0.15175 |
| drjohnson | **LGDU** | **29.8431** | **0.92374** | **0.13807** |
| train | 3DGS | 22.6612 | 0.86717 | 0.12462 |
| train | Mini | 22.0646 | 0.84028 | 0.17184 |
| train | Mip | 22.5844 | 0.87292 | 0.12094 |
| train | **LGDU** | **22.8251** | **0.87425** | **0.12089** |
| playroom | 3DGS | 30.4723 | 0.92675 | 0.13717 |
| playroom | Mini | **30.6869** | 0.92816 | 0.14955 |
| playroom | Mip | 30.5454 | 0.92634 | **0.13459** |
| playroom | **LGDU** | 30.5397 | **0.92864** | 0.13621 |

</details>

<details>
<summary><strong>LGDU Core improvement over vanilla 3DGS</strong></summary>

| Scene | Δ PSNR ↑ | Δ SSIM ↑ | Δ LPIPS ↓ |
|---|---:|---:|---:|
| counter | +0.1450 | +0.00235 | -0.00017 |
| kitchen | +0.2996 | +0.00088 | -0.00072 |
| room | +0.1386 | +0.00079 | -0.00060 |
| truck | +0.0096 | +0.01044 | -0.00241 |
| drjohnson | +0.1883 | +0.01088 | -0.00094 |
| train | +0.1639 | +0.00708 | -0.00373 |
| playroom | +0.0674 | +0.00189 | -0.00096 |

Negative LPIPS deltas indicate improvement.

</details>

## Method

### Motivation

Vanilla 3DGS densifies primarily from image-space gradients. This is effective in well-textured regions but can under-allocate Gaussians around thin structures, long edges, and sparsely reconstructed geometry. Raw 3D lines provide useful structure, but treating every reconstructed segment as exact geometry can amplify LIMAP noise.

LGDU therefore uses lines as **soft, confidence-weighted guidance**. Reliable tracks influence training more strongly, while weak or poorly supported tracks are filtered or down-weighted.

### Pipeline

```text
COLMAP cameras + sparse points + images
                 │
                 ▼
          LIMAP 3D line tracks
                 │
                 ▼
   visibility confidence + multi-view edge support
                 │
       ┌─────────┼──────────┐
       ▼         ▼          ▼
 projected   densification  unpooling
 line loss      score       candidates
       └─────────┼──────────┘
                 ▼
          optimized Gaussians
```

### Core Components

| Component | Purpose |
|---|---|
| Track-level line loading | Reads LIMAP `alltracks.txt`, including the visibility information needed for confidence estimation. |
| Visibility-aware confidence | Gives stronger initial weights to line tracks observed consistently across views. |
| Multi-view edge verification | Projects 3D lines into selected views and checks whether image evidence supports the reconstructed geometry. |
| Projected line photometric reweighting | Adds confidence-aware supervision around valid projected line regions with an adaptive loss weight. |
| Line-aware densification | Softly boosts the densification score of Gaussians near reliable lines, with extra emphasis on weak-opacity regions. |
| Line-guided unpooling | Samples high-value positions along reliable segments and initializes new Gaussians where existing support is sparse. |

The repository also retains SMip, Gaussian-refined confidence, PixelGap, and gated PixelGap as experimental variants and ablations. They are not part of the recommended LGDU Core configuration.

## Implementation Status

| Feature | Status |
|---|---|
| LIMAP `alltracks.txt` loading | Implemented |
| Visibility-aware line confidence | Implemented |
| Multi-view edge verification | Implemented |
| Projected line photometric reweighting | Implemented |
| Line-aware densification score | Implemented |
| Line-guided unpooling | Implemented |
| Seven-scene comparison against 3DGS, Mini-Splatting, and Mip-Splatting | Completed |
| Unified six-scene qualitative comparison atlas | Added |

## Repository Layout

```text
LGDU-splatting/
├── train.py                         # Training loop and line-guided losses
├── render.py                        # Train/test view rendering
├── metrics_stream.py                # Streaming PSNR/SSIM/LPIPS evaluation
├── check_init_ply.py                # Initial point-cloud inspection utility
├── arguments/__init__.py            # LGDU command-line options
├── scene/gaussian_model.py          # Densification and line-guided unpooling
├── scene/dataset_readers.py         # Dataset loading adjustments
├── utils/line_utils.py              # Line loading, confidence, projection, edge support
└── assets/comparisons/sum_submit.png # Unified qualitative comparison
```

Original 3DGS files and licensing terms are retained. See [LICENSE.md](LICENSE.md).

## Installation

### 1. Clone the Repository

The CUDA extensions are included as submodules, so clone recursively:

```bash
git clone --recursive https://github.com/Ddsmmmm/LGDU-splatting.git
cd LGDU-splatting
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

### 2. Create the 3DGS Environment

The supplied environment uses Python 3.7.13, PyTorch 1.12.1, and CUDA Toolkit 11.6:

```bash
conda env create --file environment.yml
conda activate gaussian_splatting
```

OpenCV is included for edge-supported line confidence. A CUDA-capable GPU and a compiler compatible with the CUDA/PyTorch versions are required to build the rasterization extensions.

LIMAP should be installed in a separate environment following the [official LIMAP installation instructions](https://github.com/cvg/limap). The examples below use an environment named `Limap`.

## Data Preparation

### Scene Layout

Each scene must contain images and a COLMAP sparse reconstruction:

```text
<scene>/
├── images/
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

### Generate LIMAP Line Tracks

```bash
conda activate Limap
cd /path/to/limap

python runners/colmap_triangulation.py \
  -a "$(which colmap)" \
  -m "/path/to/scene/sparse/0" \
  -i "/path/to/scene/images" \
  --max_image_dim 1280 \
  -nv 6 \
  --output_dir "/path/to/limap/outputs/scene_nv6"
```

LGDU training expects:

```text
<limap_output>/alltracks.txt
```

`triangulated_lines_*.obj` is useful for visualization, but the recommended training path uses `alltracks.txt` because it preserves track-level visibility information. In LIMAP, `-c` / `--config_file` specifies a YAML configuration file; use `--output_dir` to choose the output directory.

## Training

The following command reproduces the recommended **LGDU Core** configuration.

Set the CUDA allocator option first when needed:

```bash
# Linux
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Windows PowerShell
# $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
```

```bash
conda activate gaussian_splatting
cd /path/to/LGDU-splatting

python train.py \
  --source_path "/path/to/scene" \
  --model_path "/path/to/output/lgdu_nv6" \
  --eval \
  --disable_viewer \
  --data_device cpu \
  --resolution <R> \
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

Use the same `--resolution`, input images, train/test split, and iteration count for every compared method. The line-guided modules are scheduled only within their configured iteration windows and fall back to the standard 3DGS behavior outside those windows.

## Rendering

```bash
conda activate gaussian_splatting
cd /path/to/LGDU-splatting

python render.py \
  --source_path "/path/to/scene" \
  --model_path "/path/to/output/lgdu_nv6" \
  --iteration 30000 \
  --skip_train \
  --resolution <R> \
  --data_device cpu
```

The rendered test split is written to:

```text
<model_path>/test/ours_30000/renders/
<model_path>/test/ours_30000/gt/
```

## Evaluation

`metrics_stream.py` evaluates one or more rendered model directories without loading the full benchmark into memory at once:

```bash
conda activate gaussian_splatting
cd /path/to/LGDU-splatting

python metrics_stream.py \
  --split test \
  --model_paths "/path/to/3dgs_baseline" \
                "/path/to/mini_splatting" \
                "/path/to/mip_splatting" \
                "/path/to/lgdu"
```

- **PSNR:** higher is better.
- **SSIM:** higher is better.
- **LPIPS:** lower is better.

### Fair-Comparison Checklist

Before comparing methods, verify that they use:

1. The same source images and COLMAP reconstruction.
2. The same official train/test split.
3. The same `--resolution` value.
4. The same evaluated training iteration, currently 30,000.
5. The matching `renders/` and `gt/` directories for each model.

## Conclusion

LGDU-Splatting shows that verified line tracks are useful as soft structural guidance for Gaussian allocation. Across the current seven-scene benchmark, LGDU Core consistently improves all three global metrics over vanilla 3DGS while remaining competitive with Mini-Splatting and Mip-Splatting. The most stable gains come from combining confidence-aware projected-line supervision, line-aware densification, and conservative line-guided unpooling.

## Citation and License

This project is derived from 3D Gaussian Splatting and retains its original license. See [LICENSE.md](LICENSE.md).

If you use this repository, please cite the original 3DGS paper:

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

## Acknowledgements

LGDU-Splatting builds on the excellent open-source implementations of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [LIMAP](https://github.com/cvg/limap), and compares against Mini-Splatting and Mip-Splatting under a unified evaluation protocol.
