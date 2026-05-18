#!/bin/bash
set -x
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY='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'
export OMP_NUM_THREADS=1
export REWARD_PRINT_FREQ=0
export RAY_INCLUDE_DASHBOARD=0
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

SEED=${SEED:-42}
DATA_SEED=${DATA_SEED:-$SEED}
ROLLOUT_SEED=${ROLLOUT_SEED:-$SEED}
export PYTHONHASHSEED=${SEED}

N_GPUS=${N_GPUS:-4}
MODEL_PATH=${MODEL_PATH:-/nfs-stor/zhengqing.gao/yuhao.wu/lzc/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
TRAIN_FILE=${TRAIN_FILE:-${PROJECT_ROOT}/data/NemotronCascadeMath/train.parquet}
VAL_FILE=${VAL_FILE:-${PROJECT_ROOT}/data/NemotronCascadeMath/test.parquet}
CKPT_DIR=${CKPT_DIR:-/nfs-stor/zhengqing.gao/yuhao.wu/lzc/L2PEFT/checkpoints/grpo_qwen3_1.7B_nemotron_cascade_math_FFT_d${DATA_SEED}_r${ROLLOUT_SEED}}
REWARD_FN_PATH=${REWARD_FN_PATH:-${PROJECT_ROOT}/rewards/nemotron_math_reward.py}

mkdir -p ${CKPT_DIR}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.train_batch_size=128 \
    data.seed=${DATA_SEED} \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.seed=${ROLLOUT_SEED} \
    actor_rollout_ref.rollout.n=8 \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    custom_reward_function.path=${REWARD_FN_PATH} \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=PEFT \
    trainer.experiment_name=qwen3_1.7B_grpo_nemotron_cascade_math_d${DATA_SEED}_r${ROLLOUT_SEED} \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 \
    trainer.val_before_train=True \
    trainer.default_local_dir=${CKPT_DIR}
