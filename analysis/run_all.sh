#!/bin/bash
# ============================================================================
# run_all.sh — Run the full analysis pipeline
#
# Usage:
#   bash analysis/run_all.sh \
#       --base_model /path/to/Qwen3-0.6B \
#       --checkpoint_dir /path/to/checkpoints \
#       --final_label global_step_272 \
#       --output_dir ./results
#
# Optional:
#   --threshold 1e-5          (fraction scoring threshold, default 1e-5)
#   --top_k_percents 5,10,20,30,40  (top-k sweep, default "10,20,30")
#   --metrics /path/to/trainer_metrics.csv  (for dynamics coupling)
#   --skip_plot               (skip matplotlib plots)
# ============================================================================
set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
THRESHOLD="1e-5"
TOP_K_PERCENTS="10,20,30"
TOP_K_DEFAULT="20"
FINAL_LABEL="final"
OUTPUT_DIR="./results"
METRICS=""
SKIP_PLOT=false

# ── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base_model)       BASE_MODEL="$2"; shift 2;;
        --checkpoint_dir)   CHECKPOINT_DIR="$2"; shift 2;;
        --final_label)      FINAL_LABEL="$2"; shift 2;;
        --output_dir)       OUTPUT_DIR="$2"; shift 2;;
        --threshold)        THRESHOLD="$2"; shift 2;;
        --top_k_percents)   TOP_K_PERCENTS="$2"; shift 2;;
        --top_k_default)    TOP_K_DEFAULT="$2"; shift 2;;
        --metrics)          METRICS="$2"; shift 2;;
        --skip_plot)        SKIP_PLOT=true; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [[ -z "${BASE_MODEL:-}" || -z "${CHECKPOINT_DIR:-}" ]]; then
    echo "Error: --base_model and --checkpoint_dir are required"
    exit 1
fi

# ── Resolve paths ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCORES_DIR="${OUTPUT_DIR}/scores"
CONVERGENCE_JSON="${OUTPUT_DIR}/convergence.json"

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Analysis Pipeline"
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
python "${SCRIPT_DIR}/score_checkpoints.py" \
    --base_model "${BASE_MODEL}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --threshold "${THRESHOLD}" \
    --output_dir "${SCORES_DIR}"
echo ""

# ── Step 2: Compute convergence (IoU + Spearman vs final) ──────────────────
echo "=== Step 2: Computing convergence metrics ==="
python "${SCRIPT_DIR}/compute_convergence.py" \
    --scores_dir "${SCORES_DIR}" \
    --final_label "${FINAL_LABEL}" \
    --top_k_percent "${TOP_K_DEFAULT}" \
    --top_k_percents "${TOP_K_PERCENTS}" \
    --output "${CONVERGENCE_JSON}"
echo ""

# ── Step 3: Plot convergence curves ────────────────────────────────────────
if [[ "${SKIP_PLOT}" == false ]]; then
    echo "=== Step 3: Plotting convergence ==="
    python "${SCRIPT_DIR}/plot_convergence.py" \
        --input "${CONVERGENCE_JSON}" \
        --top_k_percent "${TOP_K_DEFAULT}" \
        --output "${OUTPUT_DIR}/convergence_plot.pdf"
    echo ""
fi

# ── Step 4: Module-layer decomposition on final checkpoint ─────────────────
echo "=== Step 4: Module-layer decomposition ==="
# Find the final checkpoint score file
FINAL_SCORE=$(find "${SCORES_DIR}" -name "scores_*${FINAL_LABEL}*.json" | head -1)
if [[ -n "${FINAL_SCORE}" ]]; then
    PLOT_FLAG=""
    if [[ "${SKIP_PLOT}" == false ]]; then
        PLOT_FLAG="--plot"
    fi

    for METHOD in fraction l2; do
        python "${SCRIPT_DIR}/decompose_module_layer.py" \
            --inputs "final=${FINAL_SCORE}" \
            --method "${METHOD}" \
            --normalize sum1 \
            --output_dir "${OUTPUT_DIR}/decomposition_${METHOD}" \
            ${PLOT_FLAG}
    done
    echo ""
else
    echo "  WARNING: Could not find final score file for decomposition, skipping"
    echo ""
fi

# ── Step 5: Projection audit (if veRL selective_peft is available) ─────────
echo "=== Step 5: Projection audit ==="
if python -c "import sys; sys.path.insert(0,'${SCRIPT_DIR}/../verl'); from verl.utils.selective_peft import classify_active_params" 2>/dev/null; then
    if [[ -n "${FINAL_SCORE}" ]]; then
        PLOT_FLAG=""
        if [[ "${SKIP_PLOT}" == false ]]; then
            PLOT_FLAG="--plot"
        fi
        python "${SCRIPT_DIR}/audit_projection_loss.py" \
            --inputs "final=${FINAL_SCORE}" \
            --method l2 \
            --top_k_percent "${TOP_K_DEFAULT}" \
            --output_dir "${OUTPUT_DIR}/projection_audit" \
            ${PLOT_FLAG}
    fi
else
    echo "  Skipping (verl.utils.selective_peft not available)"
fi
echo ""

# ── Step 6: Couple with training dynamics (if metrics provided) ────────────
if [[ -n "${METRICS}" ]]; then
    echo "=== Step 6: Coupling with training dynamics ==="
    python "${SCRIPT_DIR}/couple_training_dynamics.py" \
        --convergence "${CONVERGENCE_JSON}" \
        --metrics "${METRICS}" \
        --top_k_percent "${TOP_K_DEFAULT}" \
        --output "${OUTPUT_DIR}/dynamics_coupling.json"
    echo ""
else
    echo "=== Step 6: Skipping dynamics coupling (no --metrics provided) ==="
    echo ""
fi

# ── Summary ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo "Pipeline complete. Outputs:"
echo "  Scores:         ${SCORES_DIR}/"
echo "  Convergence:    ${CONVERGENCE_JSON}"
if [[ "${SKIP_PLOT}" == false ]]; then
    echo "  Plot:           ${OUTPUT_DIR}/convergence_plot.pdf"
fi
echo "  Decomposition:  ${OUTPUT_DIR}/decomposition_*/"
if [[ -n "${METRICS}" ]]; then
    echo "  Dynamics:       ${OUTPUT_DIR}/dynamics_coupling.json"
fi
echo "============================================================"
