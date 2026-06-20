#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/ddsm/DownLoad/Confidence-Aware-LineSplat/gaussian-splatting"
SOURCE_PATH="${PROJECT_DIR}/data/drjohnson"
MODEL_PATH="${PROJECT_DIR}/output/drjohnson_eval_ca_evp_v2_photo_darkcov_nv6_nocap_res2"
LINE_TRACKS_PATH="/home/ddsm/DownLoad/limap/outputs/drjohnson/drjohnson_nv6/alltracks.txt"
START_CHECKPOINT="${MODEL_PATH}/chkpnt7000.pth"

cd "${PROJECT_DIR}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py \
  --source_path "${SOURCE_PATH}" \
  --model_path "${MODEL_PATH}" \
  --eval \
  --disable_viewer \
  --data_device cpu \
  --resolution 2 \
  --test_iterations -1 \
  --checkpoint_iterations 15000 30000 \
  --progress_full_log_interval 100 \
  --densify_max_points_per_stage 0 \
  --start_checkpoint "${START_CHECKPOINT}" \
  --line_tracks_path "${LINE_TRACKS_PATH}" \
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
  --coverage_loss_enable \
  --coverage_backward_mode opacity \
  --coverage_lambda_adaptive \
  --coverage_lambda_target_ratio 0.02 \
  --coverage_lambda_adaptive_min 0.0 \
  --coverage_lambda_adaptive_max 0.01 \
  --coverage_loss_start_iter 3000 \
  --coverage_loss_end_iter 20000 \
  --coverage_loss_interval 4 \
  --coverage_render_downsample 2 \
  --coverage_opacity_target 0.8 \
  --coverage_alpha_target 0.98 \
  --coverage_mask_mode dark \
  --coverage_dark_threshold 0.35
