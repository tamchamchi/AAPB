#!/bin/bash
# Run both prompt versions on all 8 RareBench categories.
#
# Usage:
#   bash src/run_rarebench.sh
#   CUDA_VISIBLE_DEVICES=1 bash src/run_rarebench.sh

set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BENCHMARK_DIR="$(dirname "$(dirname "$PROJECT_DIR")")/benchmark/rewrite_gepa"
device="cuda:0"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

SEED=${SEED:-42}
OUT_PATH=${OUT_PATH:-"${PROJECT_DIR}/output/rewrite_gepa"}

echo "=============================="
echo "AAPB RareBench Inference"
echo "Input: ${BENCHMARK_DIR}"
echo "Output: ${OUT_PATH}"
echo "Seed: ${SEED}"
echo "=============================="

python "${SCRIPT_DIR}/inference.py" \
    --test_dir "${BENCHMARK_DIR}" \
    --prompt_keys r2f_prompt r2f_prompt_rewrite \
    --out_path "${OUT_PATH}" \
    --seed "${SEED}" \
    --device "${device}"

echo ""
echo "Done. Results saved to ${OUT_PATH}"
