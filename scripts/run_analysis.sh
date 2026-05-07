#!/bin/bash
# ============================================================================
# run_analysis.sh — Run the full early-locking analysis pipeline
#
# Usage:
#   bash scripts/run_analysis.sh
#
# Assumes:
#   - Base model at /root/autodl-tmp/models/Qwen3-0.6B (adjust below)
#   - Checkpoints at checkpoints/grpo_qwen3_0.6B_numina_FFT/
#     (mix of FSDP and _HF dirs, only _HF dirs are auto-discovered)
# ============================================================================
set -euo pipefail

# ── Config (edit these) ─────────────────────────────────────────────────────
PROJECT_ROOT="/root/autodl-tmp/pruning"
BASE_MODEL="/root/autodl-tmp/models/Qwen3-0.6B"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/grpo_qwen3_0.6B_numina_FFT"
OUTPUT_DIR="${PROJECT_ROOT}/results"

# The label of the final checkpoint (= directory name of the _HF folder)
FINAL_LABEL="global_step_272_HF"

# Scoring params
THRESHOLD="1e-5"
TOP_K_PERCENTS="5,10,20,30,40"
TOP_K_DEFAULT="20"

# Optional: path to training metrics CSV/JSONL (leave empty to skip)
METRICS=""

# ── Derived paths ───────────────────────────────────────────────────────────
ANALYSIS_DIR="${PROJECT_ROOT}/analysis"
SCORES_DIR="${OUTPUT_DIR}/scores"
CONVERGENCE_JSON="${OUTPUT_DIR}/convergence.json"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Early-Locking Analysis Pipeline"
echo "  Base model:     ${BASE_MODEL}"
echo "  Checkpoint dir: ${CHECKPOINT_DIR}"
echo "  Final label:    ${FINAL_LABEL}"
echo "  Threshold:      ${THRESHOLD}"
echo "  Top-k sweep:    ${TOP_K_PERCENTS}"
echo "  Output:         ${OUTPUT_DIR}"
echo "============================================================"
echo ""

# ── Step 1: Score all checkpoints ───────────────────────────────────────────
echo "=== Step 1: Scoring checkpoints ==="
python "${ANALYSIS_DIR}/score_checkpoints.py" \
    --base_model "${BASE_MODEL}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --threshold "${THRESHOLD}" \
    --output_dir "${SCORES_DIR}"
echo ""

# ── Step 2: Compute convergence (IoU + Spearman vs final) ──────────────────
echo "=== Step 2: Computing convergence metrics ==="
python "${ANALYSIS_DIR}/compute_convergence.py" \
    --scores_dir "${SCORES_DIR}" \
    --final_label "${FINAL_LABEL}" \
    --top_k_percent "${TOP_K_DEFAULT}" \
    --top_k_percents "${TOP_K_PERCENTS}" \
    --output "${CONVERGENCE_JSON}"
echo ""

# ── Step 3: Plot convergence curves ────────────────────────────────────────
echo "=== Step 3: Plotting convergence ==="
python "${ANALYSIS_DIR}/plot_convergence.py" \
    --input "${CONVERGENCE_JSON}" \
    --top_k_percent "${TOP_K_DEFAULT}" \
    --output "${OUTPUT_DIR}/convergence_plot.pdf"
echo ""

# ── Step 4: Module-layer decomposition on final checkpoint ─────────────────
echo "=== Step 4: Module-layer decomposition ==="
FINAL_SCORE="${SCORES_DIR}/scores_${FINAL_LABEL}.json"
if [[ -f "${FINAL_SCORE}" ]]; then
    for METHOD in fraction l2; do
        # LoRA-eligible modules only (7 types)
        python "${ANALYSIS_DIR}/decompose_module_layer.py" \
            --inputs "final=${FINAL_SCORE}" \
            --method "${METHOD}" \
            --normalize none \
            --output_dir "${OUTPUT_DIR}/decomposition_${METHOD}" \
            --plot
        # All modules including norms (11 types)
        python "${ANALYSIS_DIR}/decompose_module_layer.py" \
            --inputs "final=${FINAL_SCORE}" \
            --method "${METHOD}" \
            --normalize none \
            --output_dir "${OUTPUT_DIR}/decomposition_${METHOD}_all_modules" \
            --plot \
            --all_modules
    done
else
    echo "  WARNING: ${FINAL_SCORE} not found, skipping decomposition"
fi
echo ""

# ── Step 5: Coupling with training dynamics (optional) ─────────────────────
if [[ -n "${METRICS}" && -f "${METRICS}" ]]; then
    echo "=== Step 5: Coupling with training dynamics ==="
    python "${ANALYSIS_DIR}/couple_training_dynamics.py" \
        --convergence "${CONVERGENCE_JSON}" \
        --metrics "${METRICS}" \
        --top_k_percent "${TOP_K_DEFAULT}" \
        --output "${OUTPUT_DIR}/dynamics_coupling.json"
else
    echo "=== Step 5: Skipping dynamics coupling (no metrics file) ==="
fi
echo ""

# ── Summary ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo "Done. Outputs:"
echo "  Scores:        ${SCORES_DIR}/"
echo "  Convergence:   ${CONVERGENCE_JSON}"
echo "  Plot:          ${OUTPUT_DIR}/convergence_plot.pdf"
echo "  Decomposition: ${OUTPUT_DIR}/decomposition_*/"
echo "============================================================"
