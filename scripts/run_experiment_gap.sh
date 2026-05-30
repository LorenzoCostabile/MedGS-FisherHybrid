#!/usr/bin/env bash
set -euo pipefail

CASE_NAME="${CASE_NAME:-014_P3_1_right}"
IMG_ROOT="${IMG_ROOT:-data/real_${CASE_NAME}_img}"
SEG_ROOT="${SEG_ROOT:-data/real_${CASE_NAME}_seg}"
ITERATIONS="${ITERATIONS:-30000}"
SAVE_ITERS="${SAVE_ITERS:-${ITERATIONS}}"
CHECKPOINT_ITERS="${CHECKPOINT_ITERS:-${ITERATIONS}}"
TEST_ITERS="${TEST_ITERS:-${ITERATIONS}}"
GAP_START_FRAC="${GAP_START_FRAC:-0.4}"
GAP_END_FRAC="${GAP_END_FRAC:-0.6}"
LABEL="${LABEL:-gap20}"

echo "[Gap] baseline"
docker compose run --rm medgs \
  python -u train.py \
  -s "${IMG_ROOT}" \
  -m "output/expC_${CASE_NAME}_${LABEL}_baseline" \
  --pipeline joint \
  --seg_source_path "${SEG_ROOT}" \
  --iterations "${ITERATIONS}" \
  --holdout_stride 8 \
  --holdout_offset 0 \
  --gap_start_frac "${GAP_START_FRAC}" \
  --gap_end_frac "${GAP_END_FRAC}" \
  --test_split primary \
  --density_mode heuristic \
  --save_iterations "${SAVE_ITERS}" \
  --checkpoint_iterations "${CHECKPOINT_ITERS}" \
  --test_iterations "${TEST_ITERS}"

echo "[Gap] fisher"
docker compose run --rm medgs \
  python -u train.py \
  -s "${IMG_ROOT}" \
  -m "output/expC_${CASE_NAME}_${LABEL}_fisher" \
  --pipeline joint \
  --seg_source_path "${SEG_ROOT}" \
  --iterations "${ITERATIONS}" \
  --holdout_stride 8 \
  --holdout_offset 0 \
  --gap_start_frac "${GAP_START_FRAC}" \
  --gap_end_frac "${GAP_END_FRAC}" \
  --test_split primary \
  --density_mode fisher_hybrid \
  --fisher_views_per_update 4 \
  --fisher_ema_decay 0.8 \
  --fisher_weight_xyz 0.5 \
  --fisher_weight_deform 0.5 \
  --fisher_keep_quantile 0.5 \
  --fisher_prune_quantile 0.1 \
  --fisher_prune_opacity 0.05 \
  --fisher_prune_patience 3 \
  --save_iterations "${SAVE_ITERS}" \
  --checkpoint_iterations "${CHECKPOINT_ITERS}" \
  --test_iterations "${TEST_ITERS}"
