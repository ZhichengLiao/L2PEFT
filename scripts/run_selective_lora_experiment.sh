#!/bin/bash
# ============================================================================
# run_selective_lora_experiment.sh
#
# End-to-end workflow on AutoDL:
#   1. Sweep top-k × rank to find parameter-matched config
#   2. Generate selective mask from step 5 scores
#   3. Run Selective LoRA training
#   4. (Optional) Run Classic LoRA baseline for comparison
#
# Usage:
#   # Step 1: Sweep (read-only, no training)
#   bash scripts/run_selective_lora_experiment.sh sweep
#
#   # Step 2: Generate mask at chosen top-k, print matching rank
#   bash scripts/run_selective_lora_experiment.sh mask --top_k 25 --classic_rank 16
#
#   # Step 3: Run selective LoRA training
#   bash scripts/run_selective_lora_experiment.sh train_selective
#
#   # Step 4: Run classic LoRA baseline
#   bash scripts/run_selective_lora_experiment.sh train_classic
# ============================================================================
set -euo pipefail
export OMP_NUM_THREADS=1

# ── Locate scripts relative to this file (independent of PROJECT_ROOT) ────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Config ─────────────────────────────────────────────────────────────────
PROJECT_ROOT="/root/autodl-tmp/pruning"
BASE_MODEL="/root/autodl-tmp/models/Qwen3-0.6B"
SCORES_DIR="${PROJECT_ROOT}/results/scores"
MASKS_DIR="${PROJECT_ROOT}/masks"
SCORE_STEP=${SCORE_STEP:-5}    # use step 5 (early-locked L2 ranking)

# These get set by the "mask" step and used by "train_selective"
TOP_K=${TOP_K:-25}
CLASSIC_RANK=${CLASSIC_RANK:-16}
N_GPUS=${N_GPUS:-2}
# ───────────────────────────────────────────────────────────────────────────

SCORE_FILE="${SCORES_DIR}/scores_global_step_${SCORE_STEP}_HF.json"
MASK_FILE="${MASKS_DIR}/mask_step${SCORE_STEP}_l2_top${TOP_K}.json"

cd "${PROJECT_ROOT}"

read_recommended_rank() {
    python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    mask = json.load(f)
budget = mask.get("metadata", {}).get("selective_lora_budget", {})
rank = budget.get("recommended_rank")
print("" if rank is None else int(rank))
PY
}

case "${1:-help}" in
  sweep)
    echo "=== Parameter Budget Sweep ==="
    echo "Using scores from: ${SCORE_FILE}"
    python3 "${SCRIPT_DIR}/prepare_selective_lora.py" \
        --scores "${SCORE_FILE}" \
        --model_path "${BASE_MODEL}" \
        --sweep
    ;;

  mask)
    shift
    # Parse optional overrides
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --top_k) TOP_K="$2"; shift 2 ;;
            --classic_rank) CLASSIC_RANK="$2"; shift 2 ;;
            *) echo "Unknown arg: $1"; exit 1 ;;
        esac
    done
    MASK_FILE="${MASKS_DIR}/mask_step${SCORE_STEP}_l2_top${TOP_K}.json"

    echo "=== Generating Mask ==="
    echo "  Scores:       ${SCORE_FILE}"
    echo "  Top-k:        ${TOP_K}%"
    echo "  Classic rank: ${CLASSIC_RANK}"
    echo "  Output:       ${MASK_FILE}"
    echo ""

    mkdir -p "${MASKS_DIR}"
    python3 "${SCRIPT_DIR}/prepare_selective_lora.py" \
        --scores "${SCORE_FILE}" \
        --model_path "${BASE_MODEL}" \
        --top_k_percent "${TOP_K}" \
        --classic_rank "${CLASSIC_RANK}" \
        --output "${MASK_FILE}"
    ;;

  train_selective)
    shift
    CUSTOM_MASK=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --top_k) TOP_K="$2"; shift 2 ;;
            --rank) SELECTIVE_RANK="$2"; shift 2 ;;
            --mask) MASK_FILE="$2"; CUSTOM_MASK=1; shift 2 ;;
            *) echo "Unknown arg: $1"; exit 1 ;;
        esac
    done
    if [[ "${CUSTOM_MASK}" -eq 0 ]]; then
        MASK_FILE="${MASKS_DIR}/mask_step${SCORE_STEP}_l2_top${TOP_K}.json"
    fi

    # Read rank from mask metadata or use SELECTIVE_RANK env var
    if [[ ! -f "${MASK_FILE}" ]]; then
        echo "ERROR: Mask file not found: ${MASK_FILE}"
        echo "Run 'bash $0 mask' first."
        exit 1
    fi

    if [[ -z "${SELECTIVE_RANK:-}" ]]; then
        SELECTIVE_RANK=$(read_recommended_rank "${MASK_FILE}")
        if [[ -z "${SELECTIVE_RANK}" ]]; then
            echo "SELECTIVE_RANK not set and mask has no metadata.selective_lora_budget.recommended_rank."
            echo "Run 'bash $0 mask --top_k ${TOP_K} --classic_rank ${CLASSIC_RANK}' with the updated prepare script."
            exit 1
        fi
    fi

    echo "=== Selective LoRA Training ==="
    echo "  Mask:  ${MASK_FILE}"
    echo "  Rank:  ${SELECTIVE_RANK}"
    echo ""

    MASK_PATH="${MASK_FILE}" \
    LORA_RANK="${SELECTIVE_RANK}" \
    LORA_ALPHA=$(( SELECTIVE_RANK * 2 )) \
    MODEL_PATH="${BASE_MODEL}" \
    CKPT_DIR="${CKPT_DIR:-$HOME/checkpoints/grpo_selective_lora_step${SCORE_STEP}_top${TOP_K}_r${SELECTIVE_RANK}}" \
    N_GPUS="${N_GPUS}" \
        bash "${SCRIPT_DIR}/train_grpo_selective_lora.sh"
    ;;

  train_classic)
    shift
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --classic_rank) CLASSIC_RANK="$2"; shift 2 ;;
            *) echo "Unknown arg: $1"; exit 1 ;;
        esac
    done
    echo "=== Classic LoRA Baseline Training ==="
    echo "  Rank: ${CLASSIC_RANK}"
    echo ""

    LORA_RANK="${CLASSIC_RANK}" \
    MODEL_PATH="${BASE_MODEL}" \
    N_GPUS="${N_GPUS}" \
        bash "${SCRIPT_DIR}/train_grpo_classic_lora.sh"
    ;;

  help|*)
    echo "Usage: bash $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  sweep                           Sweep top-k × rank combinations (read-only)"
    echo "  mask [--top_k N] [--classic_rank R]  Generate selective mask"
    echo "  train_selective [--top_k N] [--rank R] [--mask PATH]"
    echo "                                  Run Selective LoRA"
    echo "  train_classic [--classic_rank R] Run Classic LoRA baseline"
    echo ""
    echo "Environment variables:"
    echo "  TOP_K          Top-k percent (default: 25)"
    echo "  CLASSIC_RANK   Classic LoRA rank to match (default: 16)"
    echo "  SELECTIVE_RANK Rank for selective LoRA training"
    echo "  SCORE_STEP     Which FFT checkpoint step to use (default: 5)"
    echo ""
    echo "Typical workflow:"
    echo "  1. bash $0 sweep                    # see all (top-k, rank) combos"
    echo "  2. bash $0 mask --top_k 25 --classic_rank 16"
    echo "  3. bash $0 train_selective --top_k 25  # rank read from mask metadata"
    echo "  4. bash $0 train_classic --classic_rank 16"
    ;;
esac
