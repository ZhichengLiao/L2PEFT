#!/bin/bash
# Classic LoRA baseline: GRPO training with LoRA on ALL linear modules.
#
# This is the "uniform allocation" baseline against which Selective LoRA
# is compared. Same total parameter budget, distributed evenly.
#
# Usage:
#   LORA_RANK=16 bash scripts/train_grpo_classic_lora.sh
#   LORA_RANK=8 N_GPUS=4 bash scripts/train_grpo_classic_lora.sh

set -x
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export REWARD_PRINT_FREQ=0
export RAY_INCLUDE_DASHBOARD=0

# ── Configurable ──
N_GPUS=${N_GPUS:-2}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-$(( LORA_RANK * 2 ))}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
TRAIN_FILE=${TRAIN_FILE:-${PROJECT_ROOT}/data/NemotronCascadeMath/train.parquet}
VAL_FILE=${VAL_FILE:-${PROJECT_ROOT}/data/NemotronCascadeMath/test.parquet}
CKPT_DIR=${CKPT_DIR:-$HOME/checkpoints/grpo_nemotron_cascade_math_classic_lora_r${LORA_RANK}}
REWARD_FN_PATH=${REWARD_FN_PATH:-${PROJECT_ROOT}/rewards/nemotron_math_reward.py}

mkdir -p "${CKPT_DIR}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=${LORA_RANK} \
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.min_p=0 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.min_p=0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    custom_reward_function.path=${REWARD_FN_PATH} \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=grpo_classic_lora \
    trainer.experiment_name=qwen3_0.6b_nemotron_cascade_math_classic_lora_r${LORA_RANK} \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 \
    trainer.val_before_train=False \
    trainer.default_local_dir=${CKPT_DIR}
