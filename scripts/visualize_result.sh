#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Uso:"
  echo "  bash scripts/visualize_result.sh <model_path> <source_path> <iteration> <test_split> <holdout_stride> <holdout_offset> [second_holdout_offset]"
  exit 1
fi

MODEL_PATH="$1"
SOURCE_PATH="$2"
ITERATION="$3"
TEST_SPLIT="$4"
HOLDOUT_STRIDE="$5"
HOLDOUT_OFFSET="$6"
SECOND_HOLDOUT_OFFSET="${7:--1}"
GAP_START_FRAC="${GAP_START_FRAC:--1}"
GAP_END_FRAC="${GAP_END_FRAC:--1}"
FPS="${FPS:-8}"

docker compose run --rm medgs \
  python render.py \
  -s "${SOURCE_PATH}" \
  -m "${MODEL_PATH}" \
  --pipeline both \
  --iteration "${ITERATION}" \
  --holdout_stride "${HOLDOUT_STRIDE}" \
  --holdout_offset "${HOLDOUT_OFFSET}" \
  --second_holdout_offset "${SECOND_HOLDOUT_OFFSET}" \
  --test_split "${TEST_SPLIT}" \
  --gap_start_frac "${GAP_START_FRAC}" \
  --gap_end_frac "${GAP_END_FRAC}"

if compgen -G "${MODEL_PATH}/render_img/*.png" > /dev/null; then
  ffmpeg -y -framerate "${FPS}" \
    -i "${MODEL_PATH}/render_img/%05d_0.png" \
    -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
    -pix_fmt yuv420p \
    "${MODEL_PATH}/render_img.mp4"
fi

if compgen -G "${MODEL_PATH}/render_mask/*.png" > /dev/null; then
  ffmpeg -y -framerate "${FPS}" \
    -i "${MODEL_PATH}/render_mask/%05d_0.png" \
    -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
    -pix_fmt yuv420p \
    "${MODEL_PATH}/render_mask.mp4"
fi

echo "Render terminado en: ${MODEL_PATH}"
echo "- ${MODEL_PATH}/render_img"
echo "- ${MODEL_PATH}/render_mask"
echo "- ${MODEL_PATH}/render_img.mp4"
echo "- ${MODEL_PATH}/render_mask.mp4"
