# Analysis Suite

This directory contains checkpoint-driven analysis scripts for the selective PEFT project.

## Existing core flow

1. `score_checkpoints.py`
2. `compute_convergence.py`
3. `plot_convergence.py`

## New supplementary analyses

### 1. Early-locking robustness

`compute_convergence.py` now supports:

- tie-safe Spearman computation
- top-k sweeps via `--top_k_percents`
- boundary ambiguity diagnostics
- step-to-step drift / birth / death / churn metrics

Example:

```bash
python analysis/compute_convergence.py \
  --scores_dir ./scores \
  --final_label final \
  --top_k_percents 5,10,20,30,40 \
  --output ./analysis/convergence.json
```

### 2. Cross-run consensus

Use `analyze_consensus.py` for seed stability or cross-task consensus.

```bash
python analysis/analyze_consensus.py \
  --inputs seed1=./run1/scores_final.json,seed2=./run2/scores_final.json,seed3=./run3/scores_final.json \
  --method l2 \
  --top_k_percent 20 \
  --consensus_min_fraction 0.8 \
  --output_dir ./analysis/consensus \
  --plot
```

Outputs:

- `consensus_summary.json`
- `param_stats.json`
- `consensus_mask.json`
- optional `consensus_frequency.png`

### 3. Profiling to LoRA projection audit

```bash
python analysis/audit_projection_loss.py \
  --inputs step3=./masks/mask_step3_l2.json,final=./masks/mask_final_l2.json \
  --output_dir ./analysis/projection \
  --plot
```

### 4. Module-only vs layer-only vs additive decomposition

```bash
python analysis/decompose_module_layer.py \
  --inputs gsm8k=./scores/gsm8k_final.json,code=./scores/code_final.json \
  --method l2 \
  --normalize sum1 \
  --output_dir ./analysis/decomposition \
  --plot
```

Key outputs:

- `r2_module_only`
- `r2_layer_only`
- `r2_additive_module_plus_layer`
- interaction residual heatmap

### 5. Generic importance comparison

For Spectrum-vs-RL or cross-task ranking transfer:

```bash
python analysis/compare_importance.py \
  --left ./scores/gsm8k_step5.json \
  --right ./scores/code_final.json \
  --left_method l2 \
  --right_method l2 \
  --top_k_percents 10,20,30 \
  --output ./analysis/gsm8k_step5_vs_code_final.json
```

### 6. Coupling with reward / KL / entropy logs

```bash
python analysis/couple_training_dynamics.py \
  --convergence ./analysis/convergence.json \
  --metrics ./trainer_metrics.csv \
  --step_field step \
  --output ./analysis/dynamics_coupling.json
```

The metrics file can be CSV or JSONL. The script will try to auto-detect reward / KL / entropy columns if not specified explicitly.
