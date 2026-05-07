#!/bin/bash
# Batch convert veRL FSDP checkpoints to standard HuggingFace format.
#
# Handles both FFT and LoRA runs automatically:
#   - FFT    → runs `verl.model_merger` to produce a full HF model directory
#   - LoRA   → copies the already-correct `actor/lora_adapter/` saved by PEFT
#             during training (avoids the model_merger alpha=0 bug and
#             skips wasteful base-model duplication per step)
#
# Usage:
#   # Convert all global_step_* under a directory:
#   bash scripts/convert_checkpoints.sh /path/to/checkpoints
#
#   # Convert specific checkpoints:
#   bash scripts/convert_checkpoints.sh /path/to/.../global_step_100/actor [...]

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage:"
    echo "  $0 <checkpoint_root>              # auto-find all global_step_*/actor/ under root"
    echo "  $0 <actor_dir1> [actor_dir2] ...   # convert specific actor directories"
    exit 1
fi

# Collect actor directories to convert
ACTOR_DIRS=()

if [ $# -eq 1 ] && [ -d "$1" ] && [ ! -f "$1/fsdp_config.json" ]; then
    # Single arg that is a parent directory (not an actor dir itself) → auto-discover
    ROOT="$1"
    echo "Scanning ${ROOT} for global_step_*/actor/ ..."
    for step_dir in "${ROOT}"/global_step_*/; do
        actor_dir="${step_dir}actor"
        if [ -f "${actor_dir}/fsdp_config.json" ]; then
            ACTOR_DIRS+=("${actor_dir}")
        fi
    done
    if [ ${#ACTOR_DIRS[@]} -eq 0 ]; then
        echo "No FSDP checkpoints found under ${ROOT}"
        exit 1
    fi
else
    # Explicit list of actor directories
    ACTOR_DIRS=("$@")
fi

echo "Found ${#ACTOR_DIRS[@]} checkpoint(s) to convert."
echo ""

FAILED=0
for actor_dir in "${ACTOR_DIRS[@]}"; do
    # Validate
    if [ ! -f "${actor_dir}/fsdp_config.json" ]; then
        echo "[SKIP] ${actor_dir} — no fsdp_config.json found"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Derive output path: /path/to/global_step_100/actor → /path/to/global_step_100_HF
    step_dir="$(dirname "${actor_dir}")"
    target_dir="${step_dir}_HF"

    if [ -d "${target_dir}" ]; then
        echo "[SKIP] ${target_dir} — already exists"
        continue
    fi

    # ── Detect mode: LoRA vs FFT ──────────────────────────────────────────
    if [ -d "${actor_dir}/lora_adapter" ] && [ -f "${actor_dir}/lora_adapter/adapter_config.json" ]; then
        # ────────── LoRA path: trust the PEFT-saved adapter ──────────────
        echo "[LORA]     ${actor_dir}"
        echo "           → ${target_dir}/lora_adapter/"
        mkdir -p "${target_dir}"
        cp -r "${actor_dir}/lora_adapter" "${target_dir}/lora_adapter"

        # Stage base model config + tokenizer for a self-contained HF dir
        if [ -d "${actor_dir}/huggingface" ]; then
            cp -n "${actor_dir}/huggingface/"* "${target_dir}/" 2>/dev/null || true
        fi

        # Sanity-check: alpha must be non-zero (otherwise scaling=0, adapter dead)
        alpha=$(python3 -c "import json; print(json.load(open('${target_dir}/lora_adapter/adapter_config.json')).get('lora_alpha', 0))")
        rank=$(python3 -c "import json; print(json.load(open('${target_dir}/lora_adapter/adapter_config.json')).get('r', 0))")
        n_modules=$(python3 -c "import json; print(len(json.load(open('${target_dir}/lora_adapter/adapter_config.json')).get('target_modules', [])))")
        adapter_size=$(du -h "${target_dir}/lora_adapter/adapter_model.safetensors" | cut -f1)
        if [ "${alpha}" = "0" ]; then
            echo "           [WARN] lora_alpha=0 — adapter effectively disabled; edit adapter_config.json"
            FAILED=$((FAILED + 1))
        else
            echo "           rank=${rank}  alpha=${alpha}  modules=${n_modules}  size=${adapter_size}"
            echo "[OK]       ${target_dir}"
        fi

    else
        # ────────── FFT path: run verl.model_merger to get full HF weights
        echo "[FFT]      ${actor_dir}"
        echo "           → ${target_dir}"
        if python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "${actor_dir}" \
            --target_dir "${target_dir}"; then
            echo "[OK]       ${target_dir}"
        else
            echo "[FAIL]     ${actor_dir}"
            FAILED=$((FAILED + 1))
        fi
    fi
    echo ""
done

echo "Done. ${#ACTOR_DIRS[@]} total, ${FAILED} failed."
