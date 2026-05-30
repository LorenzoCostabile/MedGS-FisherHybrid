#!/usr/bin/env bash
set -euo pipefail

CASE_NAME="${CASE_NAME:-014_P3_1_right}"
IMG_ROOT="${IMG_ROOT:-data/real_${CASE_NAME}_img}"
SEG_ROOT="${SEG_ROOT:-data/real_${CASE_NAME}_seg}"
ITERATIONS="${ITERATIONS:-30000}"
SAVE_ITERS="${SAVE_ITERS:-${ITERATIONS}}"
CHECKPOINT_ITERS="${CHECKPOINT_ITERS:-${ITERATIONS}}"
TEST_ITERS="${TEST_ITERS:-${ITERATIONS}}"

run_baseline() {
  local stride="$1"
  local label="$2"
  docker compose run --rm medgs \
    python -u train.py \
    -s "${IMG_ROOT}" \
    -m "output/expB_${CASE_NAME}_${label}_baseline" \
    --pipeline joint \
    --seg_source_path "${SEG_ROOT}" \
    --iterations "${ITERATIONS}" \
    --holdout_stride 8 \
    --holdout_offset 0 \
    --second_holdout_offset 4 \
    --test_split primary \
    --train_pool_stride "${stride}" \
    --density_mode heuristic \
    --save_iterations "${SAVE_ITERS}" \
    --checkpoint_iterations "${CHECKPOINT_ITERS}" \
    --test_iterations "${TEST_ITERS}"
}

run_fisher() {
  local stride="$1"
  local label="$2"
  docker compose run --rm medgs \
    python -u train.py \
    -s "${IMG_ROOT}" \
    -m "output/expB_${CASE_NAME}_${label}_fisher" \
    --pipeline joint \
    --seg_source_path "${SEG_ROOT}" \
    --iterations "${ITERATIONS}" \
    --holdout_stride 8 \
    --holdout_offset 0 \
    --second_holdout_offset 4 \
    --test_split primary \
    --train_pool_stride "${stride}" \
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
}

echo "[Sparse] dense baseline"
run_baseline 1 dense

echo "[Sparse] dense fisher"
run_fisher 1 dense

echo "[Sparse] x2 baseline"
run_baseline 2 x2

echo "[Sparse] x2 fisher"
run_fisher 2 x2

echo "[Sparse] x4 baseline"
run_baseline 4 x4

echo "[Sparse] x4 fisher"
run_fisher 4 x4
