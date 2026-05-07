#!/bin/bash
set -x
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY='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'
export OMP_NUM_THREADS=1
export REWARD_PRINT_FREQ=0

N_GPUS=${N_GPUS:-4}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}
TRAIN_FILE=${TRAIN_FILE:-/root/autodl-tmp/pruning/data/Numina/train.parquet}
VAL_FILE=${VAL_FILE:-/root/autodl-tmp/pruning/data/Numina/test.parquet}
CKPT_DIR=${CKPT_DIR:-/root/autodl-tmp/pruning/checkpoints/grpo_qwen3_0.6B_numina_FFT}
REWARD_FN_PATH=${REWARD_FN_PATH:-$(cd "$(dirname "$0")" && pwd)/../rewards/math_reward.py}

mkdir -p ${CKPT_DIR}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +ray_kwargs.ray_init.include_dashboard=False \
    algorithm.use_kl_in_reward=False \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    custom_reward_function.path=${REWARD_FN_PATH} \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='Pruning' \
    trainer.experiment_name='grpo_numina_FFT_qwen3_0.6B' \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.total_epochs=2 \
    trainer.val_before_train=False \
    trainer.default_local_dir=${CKPT_DIR}
