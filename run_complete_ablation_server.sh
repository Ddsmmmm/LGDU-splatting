#!/usr/bin/env bash
# LGDU-Splatting 完整消融批量训练脚本
#
# 实验范围：
#   1. component：固定完整置信度 C，对 L、D、U 做 2^3=8 种完整因子组合。
#   2. confidence：固定 L+D+U，补充 C0、C1、C2；A7 同时作为完整 C3。
#   3. sensitivity：以 A7 为中心，每次只改变一个 Core 超参数。
#
# 第三个及后续位置参数是真实随机种子；默认值为 0。脚本逐个把种子传给
# train.py 的 --seed，并把结果写入对应的 seed_<N> 目录。
#
# 常用命令：
#   DRY_RUN=1 bash run_complete_ablation_server.sh everything all 1 2
#   bash run_complete_ablation_server.sh everything all 1 2  # 21×7×2=294 次
#   bash run_complete_ablation_server.sh all all 1
#   bash run_complete_ablation_server.sh component all 1
#   bash run_complete_ablation_server.sh component_missing all 1
#   bash run_complete_ablation_server.sh confidence all 1
#   bash run_complete_ablation_server.sh sensitivity all 1
#   bash run_complete_ablation_server.sh required all 1
#   bash run_complete_ablation_server.sh all all 1
#   bash run_complete_ablation_server.sh everything all 1
#   bash run_complete_ablation_server.sh a7 room 2
#
# group/config 参数：
#   component   A0--A7，共 8 配置
#   component_missing 仅 A1--A6；已有 A0 和 A7 时使用
#   confidence  C0--C2，共 3 个新配置；A7 是 C3/Core，不重复训练
#   sensitivity 10 个非中心参数配置；A7 是所有中心值，不重复训练
#   all         component + confidence，共 11 个唯一配置
#   required    A1--A6 + C0--C2 + sensitivity；仅当相同种子的 A0、A7 已存在时使用
#   everything  component + confidence + sensitivity，共 21 个唯一配置
#   a0..a7/c0..c2/p50/p90/l001/l005/d025/d100/u256/u1024/t005/t020
#
# scene 参数：counter kitchen room drjohnson playroom train truck all

set -euo pipefail

# =========================
# 1. 服务器路径：运行前必须核对
# =========================

REPO_ROOT="${REPO_ROOT:-/data1/x3d/CYN/LGDU-core}"
SCENE_ROOT="${SCENE_ROOT:-/data1/x3d/CYN/scenes}"
LINE_ROOT="${LINE_ROOT:-/data1/x3d/CYN/lines_file}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/ablation_20260901}"

# 多 GPU 并行时，在不同 tmux 会话中分别设置 GPU_ID 和不同配置即可。
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

# 磁盘保护：每次启动新训练前，输出盘可用空间低于该值时停止。
MIN_FREE_GB="${MIN_FREE_GB:-30}"

# 默认只保存最终 30k PLY，不保存训练 checkpoint，以降低批量实验磁盘占用。
# 如确实需要 7k/15k PLY 和 30k checkpoint，运行时设置 SAVE_INTERMEDIATE=1。
SAVE_INTERMEDIATE="${SAVE_INTERMEDIATE:-0}"

# 渲染图和 GT 只用于计算指标，默认在 results/per_view JSON 生成后删除，避免
# 多配置、多场景和多种子运行重复保存大量 PNG。需要局部指标或定性图时设置
# KEEP_RENDER_IMAGES=1。
KEEP_RENDER_IMAGES="${KEEP_RENDER_IMAGES:-0}"

# nvidia-smi 轮询间隔。进程显存按训练 PID 统计，不会把同卡其他公司的进程
# 算入方法峰值；整卡占用另存为背景信息，不能作为论文中的方法显存。
VRAM_SAMPLE_INTERVAL="${VRAM_SAMPLE_INTERVAL:-0.5}"

declare -A SCENE_PATH=(
  [counter]="${SCENE_ROOT}/360_v2/counter"
  [kitchen]="${SCENE_ROOT}/360_v2/kitchen"
  [room]="${SCENE_ROOT}/360_v2/room"
  [drjohnson]="${SCENE_ROOT}/tandt_db/tandt_db.1/db/drjohnson"
  [playroom]="${SCENE_ROOT}/tandt_db/tandt_db.1/db/playroom"
  [train]="${SCENE_ROOT}/tandt_db/tandt_db.1/tandt/train"
  [truck]="${SCENE_ROOT}/tandt_db/tandt_db.1/tandt/truck"
)

declare -A LINE_PATH=(
  [counter]="${LINE_ROOT}/counter/counter_nv6/alltracks.txt"
  [kitchen]="${LINE_ROOT}/kitchen/kitchen_nv6/alltracks.txt"
  [room]="${LINE_ROOT}/room/room_nv6/alltracks.txt"
  [drjohnson]="${LINE_ROOT}/drjohnson/drjohnson_nv6/alltracks.txt"
  [playroom]="${LINE_ROOT}/playroom/playroom_nv6/alltracks.txt"
  [train]="${LINE_ROOT}/train/train_nv6/alltracks.txt"
  [truck]="${LINE_ROOT}/truck/truck_nv6/alltracks.txt"
)

# 必须和已有 3DGS baseline 及完整 LGDU 使用相同分辨率。
declare -A RESOLUTION=(
  [counter]=2
  [kitchen]=1
  [room]=2
  [drjohnson]=2
  [playroom]=2
  [train]=2
  [truck]=1
)

SCENES=(counter kitchen room drjohnson playroom train truck)
COMPONENT_CONFIGS=(a0 a1 a2 a3 a4 a5 a6 a7)
MISSING_COMPONENT_CONFIGS=(a1 a2 a3 a4 a5 a6)
CONFIDENCE_CONFIGS=(c0 c1 c2)
SENSITIVITY_CONFIGS=(p50 p90 l001 l005 d025 d100 u256 u1024 t005 t020)
ALL_CONFIGS=("${COMPONENT_CONFIGS[@]}" "${CONFIDENCE_CONFIGS[@]}")
REQUIRED_CONFIGS=("${MISSING_COMPONENT_CONFIGS[@]}" "${CONFIDENCE_CONFIGS[@]}" "${SENSITIVITY_CONFIGS[@]}")
EVERYTHING_CONFIGS=("${ALL_CONFIGS[@]}" "${SENSITIVITY_CONFIGS[@]}")

# =========================
# 2. 固定公共参数和 Core 参数块
# =========================

COMMON_ARGS=(
  --eval
  --disable_viewer
  --data_device cpu
  --iterations 30000
  --test_iterations -1
  --densify_max_points_per_stage 0
)

if [[ "$SAVE_INTERMEDIATE" == "1" ]]; then
  COMMON_ARGS+=(
    --save_iterations 7000 15000 30000
    --checkpoint_iterations 30000
  )
else
  COMMON_ARGS+=(--save_iterations 30000)
fi

# C3/Core：可见性置信度 + P70 + 多视图边缘验证。
C_CORE=(
  --line_min_visible_views 6
  --line_confidence_mode visibility
  --line_confidence_power 1.0
  --line_confidence_min 0.05
  --line_multiview_edge_support_enable
  --line_multiview_edge_max_views 32
  --line_multiview_edge_min_views 2
  --line_confidence_percentile 70
  --line_edge_sigma_px 2.0
  --line_edge_sample_count 16
  --line_edge_min_valid_ratio 0.5
  --line_edge_min_projected_length 4.0
)

# C0：统一权重，无 P70，无多视图边缘验证。
C_UNIFORM=(
  --line_min_visible_views 6
  --line_confidence_mode none
  --line_confidence_percentile 0
)

# C1：仅使用可见性置信度，无 P70，无多视图边缘验证。
C_VISIBILITY=(
  --line_min_visible_views 6
  --line_confidence_mode visibility
  --line_confidence_power 1.0
  --line_confidence_min 0.05
  --line_confidence_percentile 0
)

# C2：可见性置信度 + P70，无多视图边缘验证。
C_VISIBILITY_P70=(
  --line_min_visible_views 6
  --line_confidence_mode visibility
  --line_confidence_power 1.0
  --line_confidence_min 0.05
  --line_confidence_percentile 70
)

# P50/P90：只改变 Core 的置信度百分位，仍保留多视图边缘验证。
C_CORE_P50=("${C_CORE[@]/70/50}")
C_CORE_P90=("${C_CORE[@]/70/90}")

L_CORE=(
  --line_photo_reweight_enable
  --line_photo_lambda_adaptive
  --line_photo_lambda_target_ratio 0.03
  --line_photo_lambda_adaptive_min 0.0
  --line_photo_lambda_adaptive_max 0.02
  --line_photo_start_iter 3000
  --line_photo_end_iter 15000
  --line_photo_sample_count 32
  --line_photo_mask_dilation_px 2
  --line_photo_min_valid_ratio 0.5
  --line_photo_min_projected_length 4.0
  --line_photo_charbonnier_eps 0.001
)

L_RATIO_001=(
  --line_photo_reweight_enable
  --line_photo_lambda_adaptive
  --line_photo_lambda_target_ratio 0.01
  --line_photo_lambda_adaptive_min 0.0
  --line_photo_lambda_adaptive_max 0.02
  --line_photo_start_iter 3000
  --line_photo_end_iter 15000
  --line_photo_sample_count 32
  --line_photo_mask_dilation_px 2
  --line_photo_min_valid_ratio 0.5
  --line_photo_min_projected_length 4.0
  --line_photo_charbonnier_eps 0.001
)

L_RATIO_005=(
  --line_photo_reweight_enable
  --line_photo_lambda_adaptive
  --line_photo_lambda_target_ratio 0.05
  --line_photo_lambda_adaptive_min 0.0
  --line_photo_lambda_adaptive_max 0.02
  --line_photo_start_iter 3000
  --line_photo_end_iter 15000
  --line_photo_sample_count 32
  --line_photo_mask_dilation_px 2
  --line_photo_min_valid_ratio 0.5
  --line_photo_min_projected_length 4.0
  --line_photo_charbonnier_eps 0.001
)

D_CORE=(
  --line_densify_mode score
  --line_densify_start_iter 500
  --line_densify_end_iter 15000
  --line_densify_score_boost 0.5
  --line_densify_confidence_power 1.0
  --line_densify_confidence_max 3.0
  --line_densify_low_alpha_boost 0.5
)

D_BOOST_025=(
  --line_densify_mode score
  --line_densify_start_iter 500
  --line_densify_end_iter 15000
  --line_densify_score_boost 0.25
  --line_densify_confidence_power 1.0
  --line_densify_confidence_max 3.0
  --line_densify_low_alpha_boost 0.5
)

D_BOOST_100=(
  --line_densify_mode score
  --line_densify_start_iter 500
  --line_densify_end_iter 15000
  --line_densify_score_boost 1.0
  --line_densify_confidence_power 1.0
  --line_densify_confidence_max 3.0
  --line_densify_low_alpha_boost 0.5
)

U_CORE=(
  --line_unpool_enable
  --line_unpool_start_iter 3000
  --line_unpool_end_iter 12000
  --line_unpool_interval 500
  --line_unpool_samples_per_line 2
  --line_unpool_max_points 512
  --line_unpool_candidate_factor 4
  --line_unpool_score_threshold 0.10
  --line_unpool_opacity_init 0.04
  --line_unpool_scale_factor 0.6
  --line_unpool_alpha_target 0.08
  --line_unpool_low_alpha_boost 0.5
)

U_MAX_256=(
  --line_unpool_enable
  --line_unpool_start_iter 3000
  --line_unpool_end_iter 12000
  --line_unpool_interval 500
  --line_unpool_samples_per_line 2
  --line_unpool_max_points 256
  --line_unpool_candidate_factor 4
  --line_unpool_score_threshold 0.10
  --line_unpool_opacity_init 0.04
  --line_unpool_scale_factor 0.6
  --line_unpool_alpha_target 0.08
  --line_unpool_low_alpha_boost 0.5
)

U_MAX_1024=(
  --line_unpool_enable
  --line_unpool_start_iter 3000
  --line_unpool_end_iter 12000
  --line_unpool_interval 500
  --line_unpool_samples_per_line 2
  --line_unpool_max_points 1024
  --line_unpool_candidate_factor 4
  --line_unpool_score_threshold 0.10
  --line_unpool_opacity_init 0.04
  --line_unpool_scale_factor 0.6
  --line_unpool_alpha_target 0.08
  --line_unpool_low_alpha_boost 0.5
)

U_THRESHOLD_005=(
  --line_unpool_enable
  --line_unpool_start_iter 3000
  --line_unpool_end_iter 12000
  --line_unpool_interval 500
  --line_unpool_samples_per_line 2
  --line_unpool_max_points 512
  --line_unpool_candidate_factor 4
  --line_unpool_score_threshold 0.05
  --line_unpool_opacity_init 0.04
  --line_unpool_scale_factor 0.6
  --line_unpool_alpha_target 0.08
  --line_unpool_low_alpha_boost 0.5
)

U_THRESHOLD_020=(
  --line_unpool_enable
  --line_unpool_start_iter 3000
  --line_unpool_end_iter 12000
  --line_unpool_interval 500
  --line_unpool_samples_per_line 2
  --line_unpool_max_points 512
  --line_unpool_candidate_factor 4
  --line_unpool_score_threshold 0.20
  --line_unpool_opacity_init 0.04
  --line_unpool_scale_factor 0.6
  --line_unpool_alpha_target 0.08
  --line_unpool_low_alpha_boost 0.5
)

# =========================
# 3. 工具函数
# =========================

usage() {
  cat <<'EOF'
用法：bash run_complete_ablation_server.sh <group|config> <scene|all> [seed ...]

seed:
  一个或多个非负整数，默认值为 0；各值分别传给 train.py --seed，
  并写入对应的 seed_<N> 输出目录

完整补充种子 1 和 2（21 个唯一配置 × 7 个场景 × 2 个种子）：
  bash run_complete_ablation_server.sh everything all 1 2

group:
  component    A0--A7，完整 L/D/U 因子消融
  component_missing 仅 A1--A6；已有 A0、A7 时使用
  confidence   C0--C2；A7 是完整 C3，不重复训练
  sensitivity 参数敏感性实验；A7 是中心值，不重复训练
  all          component + confidence
  required     A1--A6 + C0--C2 + sensitivity；仅当相同种子的 A0、A7 已存在时使用
  everything   component + confidence + sensitivity

config:
  a0 a1 a2 a3 a4 a5 a6 a7
  c0 c1 c2
  p50 p90 l001 l005 d025 d100 u256 u1024 t005 t020

scene:
  counter kitchen room drjohnson playroom train truck all
EOF
}

contains() {
  local wanted="$1"
  shift
  local item
  for item in "$@"; do
    [[ "$item" == "$wanted" ]] && return 0
  done
  return 1
}

save_command() {
  local output_file="$1"
  shift
  printf '%q ' "$@" > "$output_file"
  printf '\n' >> "$output_file"
}

check_free_space() {
  local output_root="$1"
  mkdir -p "$output_root"
  local free_kb free_gb
  free_kb="$(df -Pk "$output_root" | awk 'NR == 2 {print $4}')"
  [[ "$free_kb" =~ ^[0-9]+$ ]] || {
    echo "[ERROR] 无法读取输出盘剩余空间：$output_root" >&2
    return 1
  }
  free_gb=$((free_kb / 1024 / 1024))
  echo "[DISK] 输出盘剩余约 ${free_gb} GiB；安全下限 ${MIN_FREE_GB} GiB"
  if (( free_gb < MIN_FREE_GB )); then
    echo "[STOP] 磁盘剩余空间低于安全下限，停止启动新训练。" >&2
    return 1
  fi
}

validate_scene_inputs() {
  local scene="$1"
  local scene_path="$2"
  local tracks_path="$3"
  local needs_lines="$4"
  local image_count stem

  if [[ ! -d "$scene_path" ]]; then
    echo "[ERROR] $scene 场景目录不存在：$scene_path" >&2
    return 1
  fi
  if [[ ! -d "$scene_path/images" ]]; then
    echo "[ERROR] $scene 缺少 images 目录：$scene_path/images" >&2
    return 1
  fi
  if [[ ! -d "$scene_path/sparse/0" ]]; then
    echo "[ERROR] $scene 缺少 COLMAP sparse/0：$scene_path/sparse/0" >&2
    return 1
  fi

  image_count="$(
    find "$scene_path/images" -maxdepth 1 -type f \
      \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
         -o -iname '*.JPG' -o -iname '*.JPEG' -o -iname '*.PNG' \) |
    wc -l
  )"
  if (( image_count <= 0 )); then
    echo "[ERROR] $scene 的 images 目录中没有可识别图像。" >&2
    return 1
  fi

  for stem in cameras images points3D; do
    if [[ ! -s "$scene_path/sparse/0/${stem}.bin" && \
          ! -s "$scene_path/sparse/0/${stem}.txt" ]]; then
      echo "[ERROR] $scene 缺少非空 COLMAP ${stem}.bin/.txt" >&2
      return 1
    fi
  done

  if (( needs_lines == 1 )); then
    if [[ ! -s "$tracks_path" ]]; then
      echo "[ERROR] $scene 的 LIMAP 轨迹不存在或为空：$tracks_path" >&2
      return 1
    fi
  fi

  echo "[INPUT] $scene: ${image_count} images; COLMAP model OK; lines_required=${needs_lines}"
}

record_gaussian_counts() {
  local out="$1"
  local result_file="${out}/efficiency/gaussian_counts.csv"
  printf 'iteration,gaussians\n' > "$result_file"
  local iteration ply count
  for iteration in 7000 15000 30000; do
    ply="${out}/point_cloud/iteration_${iteration}/point_cloud.ply"
    if [[ -f "$ply" ]]; then
      count="$(grep -a -m 1 '^element vertex ' "$ply" | awk '{print $3}')"
      printf '%s,%s\n' "$iteration" "${count:-NA}" >> "$result_file"
    else
      printf '%s,NA\n' "$iteration" >> "$result_file"
    fi
  done
}

record_storage_summary() {
  local out="$1"
  local summary="${out}/efficiency/storage_bytes.csv"
  printf 'item,bytes\n' > "$summary"
  local item path bytes
  for item in total point_cloud logs metrics metadata test; do
    if [[ "$item" == "total" ]]; then
      path="$out"
    else
      path="${out}/${item}"
    fi
    if [[ -e "$path" ]]; then
      bytes="$(du -sb "$path" | awk '{print $1}')"
      printf '%s,%s\n' "$item" "$bytes" >> "$summary"
    else
      printf '%s,0\n' "$item" >> "$summary"
    fi
  done
}

run_with_vram_monitor() {
  local out="$1"
  shift
  local -a cmd=("$@")
  local train_log="${out}/logs/train.log"
  local trace_file="${out}/efficiency/vram_trace.csv"
  local peak_file="${out}/efficiency/peak_vram_process_mb.txt"
  local legacy_peak_file="${out}/efficiency/peak_vram_nvidia_smi_mb.txt"
  local start_epoch end_epoch elapsed peak_mb used_mb train_pid exit_code
  local gpu_used_mb other_used_mb peak_gpu_used_mb nonzero_samples sample_count

  printf 'timestamp,train_pid,process_used_mb,gpu_total_used_mb,other_used_estimate_mb\n' > "$trace_file"
  start_epoch="$(date +%s)"
  peak_mb=0
  peak_gpu_used_mb=0
  nonzero_samples=0
  sample_count=0

  set +e
  "${cmd[@]}" > >(tee "$train_log") 2>&1 &
  train_pid=$!

  while kill -0 "$train_pid" 2>/dev/null; do
    used_mb="$(
      nvidia-smi \
        --query-compute-apps=pid,used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null |
      awk -F',' -v target="$train_pid" '
        {
          gsub(/ /, "", $1);
          gsub(/ /, "", $2);
          if ($1 == target) total += $2;
        }
        END { print total + 0 }
      '
    )"
    [[ "$used_mb" =~ ^[0-9]+$ ]] || used_mb=0

    # 整卡占用仅用于判断采样时是否存在其他任务，不作为方法显存结果。
    gpu_used_mb="$(
      nvidia-smi -i "$GPU_ID" --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' '
    )"
    [[ "$gpu_used_mb" =~ ^[0-9]+$ ]] || gpu_used_mb=0
    other_used_mb=$((gpu_used_mb - used_mb))
    (( other_used_mb < 0 )) && other_used_mb=0

    (( used_mb > peak_mb )) && peak_mb="$used_mb"
    (( gpu_used_mb > peak_gpu_used_mb )) && peak_gpu_used_mb="$gpu_used_mb"
    (( used_mb > 0 )) && nonzero_samples=$((nonzero_samples + 1))
    sample_count=$((sample_count + 1))
    printf '%s,%s,%s,%s,%s\n' \
      "$(date --iso-8601=seconds)" "$train_pid" "$used_mb" \
      "$gpu_used_mb" "$other_used_mb" >> "$trace_file"
    sleep "$VRAM_SAMPLE_INTERVAL"
  done

  wait "$train_pid"
  exit_code=$?
  set -e

  end_epoch="$(date +%s)"
  elapsed=$((end_epoch - start_epoch))
  printf '%s\n' "$peak_mb" > "$peak_file"
  # 保留旧文件名，便于已有汇总脚本兼容；其值同样是按训练 PID 统计。
  printf '%s\n' "$peak_mb" > "$legacy_peak_file"
  printf '%s\n' "$peak_gpu_used_mb" > "${out}/efficiency/peak_gpu_total_used_mb.txt"
  printf '%s\n' "$train_pid" > "${out}/efficiency/training_pid.txt"
  printf '%s\n' "$sample_count" > "${out}/efficiency/vram_sample_count.txt"
  printf '%s\n' "$nonzero_samples" > "${out}/efficiency/vram_nonzero_sample_count.txt"
  if (( nonzero_samples == 0 )); then
    printf '%s\n' \
      'WARNING: no non-zero process VRAM sample; possible PID namespace mismatch or nvidia-smi limitation.' \
      > "${out}/efficiency/vram_measurement_warning.txt"
  fi
  printf '%s\n' "$elapsed" > "${out}/efficiency/train_seconds.txt"
  printf '%s\n' "$exit_code" > "${out}/metadata/exit_code.txt"
  return "$exit_code"
}

# =========================
# 4. 配置生成与单次运行
# =========================

run_one() {
  local config="$1"
  local scene="$2"
  local scene_path="${SCENE_PATH[$scene]}"
  local tracks_path="${LINE_PATH[$scene]}"
  local resolution="${RESOLUTION[$scene]}"
  local config_dir description render_dir gt_dir transient_dir needs_lines=1
  local -a module_args=()

  case "$config" in
    # 完整因子消融：C 固定为 Core，仅改变 L/D/U。
    a0)
      config_dir="01_component_factorial/A0_3DGS"
      description="A0: pure 3DGS; C=0 L=0 D=0 U=0"
      needs_lines=0
      module_args=(--line_densify_mode off)
      ;;
    a1)
      config_dir="01_component_factorial/A1_C_L"
      description="A1: C+L; D=0 U=0"
      module_args=("${C_CORE[@]}" "${L_CORE[@]}" --line_densify_mode off)
      ;;
    a2)
      config_dir="01_component_factorial/A2_C_D"
      description="A2: C+D; L=0 U=0"
      module_args=("${C_CORE[@]}" "${D_CORE[@]}")
      ;;
    a3)
      config_dir="01_component_factorial/A3_C_U"
      description="A3: C+U; L=0 D=0"
      module_args=("${C_CORE[@]}" --line_densify_mode off "${U_CORE[@]}")
      ;;
    a4)
      config_dir="01_component_factorial/A4_C_L_D"
      description="A4: C+L+D; U=0 (w/o U)"
      module_args=("${C_CORE[@]}" "${L_CORE[@]}" "${D_CORE[@]}")
      ;;
    a5)
      config_dir="01_component_factorial/A5_C_L_U"
      description="A5: C+L+U; D=0 (w/o D)"
      module_args=("${C_CORE[@]}" "${L_CORE[@]}" --line_densify_mode off "${U_CORE[@]}")
      ;;
    a6)
      config_dir="01_component_factorial/A6_C_D_U"
      description="A6: C+D+U; L=0 (w/o L)"
      module_args=("${C_CORE[@]}" "${D_CORE[@]}" "${U_CORE[@]}")
      ;;
    a7)
      config_dir="01_component_factorial/A7_C_L_D_U_CORE"
      description="A7: C+L+D+U; complete LGDU Core; also C3 and sensitivity center"
      module_args=("${C_CORE[@]}" "${L_CORE[@]}" "${D_CORE[@]}" "${U_CORE[@]}")
      ;;

    # 置信度消融：L+D+U 固定。C3 与 A7 相同，因此不重复创建 c3。
    c0)
      config_dir="02_confidence_ablation/C0_uniform"
      description="C0: uniform confidence; no P70; no multiview edge; L+D+U on"
      module_args=("${C_UNIFORM[@]}" "${L_CORE[@]}" "${D_CORE[@]}" "${U_CORE[@]}")
      ;;
    c1)
      config_dir="02_confidence_ablation/C1_visibility"
      description="C1: visibility confidence; no P70; no multiview edge; L+D+U on"
      module_args=("${C_VISIBILITY[@]}" "${L_CORE[@]}" "${D_CORE[@]}" "${U_CORE[@]}")
      ;;
    c2)
      config_dir="02_confidence_ablation/C2_visibility_p70"
      description="C2: visibility confidence + P70; no multiview edge; L+D+U on"
      module_args=("${C_VISIBILITY_P70[@]}" "${L_CORE[@]}" "${D_CORE[@]}" "${U_CORE[@]}")
      ;;

    # 参数敏感性：每次只改变一个值，其余保持 A7/Core。
    p50)
      config_dir="03_sensitivity/C_percentile/p50"
      description="Sensitivity: confidence percentile=50; Core center=70"
      module_args=("${C_CORE_P50[@]}" "${L_CORE[@]}" "${D_CORE[@]}" "${U_CORE[@]}")
      ;;
    p90)
      config_dir="03_sensitivity/C_percentile/p90"
      description="Sensitivity: confidence percentile=90; Core center=70"
      module_args=("${C_CORE_P90[@]}" "${L_CORE[@]}" "${D_CORE[@]}" "${U_CORE[@]}")
      ;;
    l001)
      config_dir="03_sensitivity/L_target_ratio/r0.01"
      description="Sensitivity: line photo target ratio=0.01; Core center=0.03"
      module_args=("${C_CORE[@]}" "${L_RATIO_001[@]}" "${D_CORE[@]}" "${U_CORE[@]}")
      ;;
    l005)
      config_dir="03_sensitivity/L_target_ratio/r0.05"
      description="Sensitivity: line photo target ratio=0.05; Core center=0.03"
      module_args=("${C_CORE[@]}" "${L_RATIO_005[@]}" "${D_CORE[@]}" "${U_CORE[@]}")
      ;;
    d025)
      config_dir="03_sensitivity/D_score_boost/b0.25"
      description="Sensitivity: densify score boost=0.25; Core center=0.50"
      module_args=("${C_CORE[@]}" "${L_CORE[@]}" "${D_BOOST_025[@]}" "${U_CORE[@]}")
      ;;
    d100)
      config_dir="03_sensitivity/D_score_boost/b1.00"
      description="Sensitivity: densify score boost=1.00; Core center=0.50"
      module_args=("${C_CORE[@]}" "${L_CORE[@]}" "${D_BOOST_100[@]}" "${U_CORE[@]}")
      ;;
    u256)
      config_dir="03_sensitivity/U_max_points/m256"
      description="Sensitivity: unpool max points=256; Core center=512"
      module_args=("${C_CORE[@]}" "${L_CORE[@]}" "${D_CORE[@]}" "${U_MAX_256[@]}")
      ;;
    u1024)
      config_dir="03_sensitivity/U_max_points/m1024"
      description="Sensitivity: unpool max points=1024; Core center=512"
      module_args=("${C_CORE[@]}" "${L_CORE[@]}" "${D_CORE[@]}" "${U_MAX_1024[@]}")
      ;;
    t005)
      config_dir="03_sensitivity/U_score_threshold/t0.05"
      description="Sensitivity: unpool score threshold=0.05; Core center=0.10"
      module_args=("${C_CORE[@]}" "${L_CORE[@]}" "${D_CORE[@]}" "${U_THRESHOLD_005[@]}")
      ;;
    t020)
      config_dir="03_sensitivity/U_score_threshold/t0.20"
      description="Sensitivity: unpool score threshold=0.20; Core center=0.10"
      module_args=("${C_CORE[@]}" "${L_CORE[@]}" "${D_CORE[@]}" "${U_THRESHOLD_020[@]}")
      ;;
    *)
      echo "[ERROR] 未知配置：$config" >&2
      return 2
      ;;
  esac

  local out="${OUTPUT_ROOT}/${config_dir}/${scene}/${SEED_NAME}"

  validate_scene_inputs "$scene" "$scene_path" "$tracks_path" "$needs_lines" || return 3

  check_free_space "$OUTPUT_ROOT" || return 7

  mkdir -p \
    "$out/metadata" "$out/logs" "$out/metrics/global" \
    "$out/metrics/regions" "$out/efficiency" "$out/visuals"

  if [[ -f "$out/results_test.json" ]]; then
    echo "[SKIP] 已完成：$out/results_test.json"
    return 0
  fi
  if [[ -d "$out/point_cloud" && "${ALLOW_PARTIAL_OVERWRITE:-0}" != "1" ]]; then
    echo "[STOP] 检测到未完成的旧训练目录：$out" >&2
    echo "请人工检查；确认可以覆盖时设置 ALLOW_PARTIAL_OVERWRITE=1。" >&2
    return 5
  fi

  local -a train_cmd=(
    python train.py
    --source_path "$scene_path"
    --model_path "$out"
    --resolution "$resolution"
    --seed "$SEED"
    "${COMMON_ARGS[@]}"
  )
  if (( needs_lines == 1 )); then
    train_cmd+=(--line_tracks_path "$tracks_path" "${module_args[@]}")
  else
    train_cmd+=("${module_args[@]}")
  fi

  printf '%s\n' "$description" > "$out/metadata/description.txt"
  printf '%s\n' "$scene" > "$out/metadata/scene.txt"
  printf '%s\n' "$config" > "$out/metadata/config.txt"
  printf '%s\n' "$resolution" > "$out/metadata/resolution.txt"
  printf '%s\n' "$SEED" > "$out/metadata/seed.txt"
  save_command "$out/metadata/command.sh" "${train_cmd[@]}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY RUN] $config / $scene"
    printf '  '
    printf '%q ' "${train_cmd[@]}"
    printf '\n'
    return 0
  fi

  git --git-dir="$REPO_ROOT/.git" --work-tree="$REPO_ROOT" rev-parse HEAD > "$out/metadata/git_commit.txt"
  git --git-dir="$REPO_ROOT/.git" --work-tree="$REPO_ROOT" status --porcelain > "$out/metadata/git_status.txt"
  if (( needs_lines == 1 )); then
    sha256sum "$tracks_path" > "$out/metadata/line_tracks_sha256.txt"
  fi
  nvidia-smi -i "$GPU_ID" --query-gpu=index,uuid,name,driver_version,memory.total \
    --format=csv,noheader > "$out/metadata/gpu.txt"
  python -c 'import sys, torch; print(sys.version); print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda))' \
    > "$out/metadata/python_torch_cuda.txt"
  date --iso-8601=seconds > "$out/metadata/start_time.txt"

  echo
  echo "============================================================"
  echo "开始：config=$config scene=$scene resolution=$resolution GPU_ID=$GPU_ID"
  echo "$description"
  echo "输出：$out"
  echo "============================================================"

  if ! run_with_vram_monitor "$out" "${train_cmd[@]}"; then
    date --iso-8601=seconds > "$out/metadata/end_time.txt"
    printf 'FAILED\n' > "$out/metadata/status.txt"
    echo "[FAILED] 请检查 $out/logs/train.log" >&2
    return 6
  fi

  record_gaussian_counts "$out"

  local -a render_cmd=(
    python render.py
    --source_path "$scene_path"
    --model_path "$out"
    --iteration 30000
    --skip_train
    --resolution "$resolution"
    --data_device cpu
  )
  save_command "$out/metadata/render_command.sh" "${render_cmd[@]}"
  "${render_cmd[@]}" 2>&1 | tee "$out/logs/render.log"

  local -a metric_cmd=(
    python metrics_stream.py
    --split test
    --model_paths "$out"
  )
  save_command "$out/metadata/metrics_command.sh" "${metric_cmd[@]}"
  "${metric_cmd[@]}" 2>&1 | tee "$out/logs/metrics.log"

  [[ -f "$out/results_test.json" ]] && \
    cp "$out/results_test.json" "$out/metrics/global/results_test.json"
  [[ -f "$out/per_view_test.json" ]] && \
    cp "$out/per_view_test.json" "$out/metrics/global/per_view_test.json"

  if [[ "$KEEP_RENDER_IMAGES" != "1" ]]; then
    render_dir="${out}/test/ours_30000/renders"
    gt_dir="${out}/test/ours_30000/gt"
    for transient_dir in "$render_dir" "$gt_dir"; do
      if [[ -d "$transient_dir" && "$transient_dir" == "$out"/* ]]; then
        rm -rf -- "$transient_dir"
      fi
    done
    printf 'Removed transient test renders and duplicated GT after metric evaluation.\n' \
      > "$out/metadata/render_cleanup.txt"
  else
    printf 'Kept test renders and GT because KEEP_RENDER_IMAGES=1.\n' \
      > "$out/metadata/render_cleanup.txt"
  fi

  record_storage_summary "$out"

  date --iso-8601=seconds > "$out/metadata/end_time.txt"
  printf 'COMPLETED\n' > "$out/metadata/status.txt"
  echo "[DONE] $config / $scene"
}

# =========================
# 5. 参数解析与批量运行
# =========================

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "[ERROR] 仓库不存在：$REPO_ROOT" >&2
  exit 10
fi
cd "$REPO_ROOT"

REQUESTED_GROUP="${1:-}"
REQUESTED_SCENE="${2:-}"
if [[ -z "$REQUESTED_GROUP" || -z "$REQUESTED_SCENE" ]]; then
  usage
  exit 1
fi

if (( $# > 2 )); then
  SEEDS=("${@:3}")
else
  SEEDS=(0)
fi

for seed in "${SEEDS[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]] || (( seed > 4294967295 )); then
    echo "[ERROR] seed 必须是 0 到 4294967295 之间的整数：$seed" >&2
    exit 2
  fi
done

case "$REQUESTED_GROUP" in
  component) selected_configs=("${COMPONENT_CONFIGS[@]}") ;;
  component_missing) selected_configs=("${MISSING_COMPONENT_CONFIGS[@]}") ;;
  confidence) selected_configs=("${CONFIDENCE_CONFIGS[@]}") ;;
  sensitivity) selected_configs=("${SENSITIVITY_CONFIGS[@]}") ;;
  all) selected_configs=("${ALL_CONFIGS[@]}") ;;
  required) selected_configs=("${REQUIRED_CONFIGS[@]}") ;;
  everything) selected_configs=("${EVERYTHING_CONFIGS[@]}") ;;
  *)
    if contains "$REQUESTED_GROUP" "${EVERYTHING_CONFIGS[@]}"; then
      selected_configs=("$REQUESTED_GROUP")
    else
      usage
      exit 1
    fi
    ;;
esac

if [[ "$REQUESTED_SCENE" == "all" ]]; then
  selected_scenes=("${SCENES[@]}")
elif contains "$REQUESTED_SCENE" "${SCENES[@]}"; then
  selected_scenes=("$REQUESTED_SCENE")
else
  usage
  exit 1
fi

TOTAL_RUNS=$(( ${#selected_configs[@]} * ${#selected_scenes[@]} * ${#SEEDS[@]} ))
echo "计划运行：${#selected_configs[@]} configurations × ${#selected_scenes[@]} scenes × ${#SEEDS[@]} seeds = ${TOTAL_RUNS} runs"
echo "随机种子：${SEEDS[*]}"
for SEED in "${SEEDS[@]}"; do
  SEED_NAME="seed_${SEED}"
  echo
  echo "===== 开始 seed=$SEED ====="
  for config in "${selected_configs[@]}"; do
    for scene in "${selected_scenes[@]}"; do
      run_one "$config" "$scene"
    done
  done
done

echo
echo "全部指定实验已完成。"
