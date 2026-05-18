#!/bin/bash
set -x
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY='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'
export OMP_NUM_THREADS=1
export REWARD_PRINT_FREQ=0
export HF_HOME=/nfs-stor/zhengqing.gao/yuhao.wu/lzc/hf_cache
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

SEED=${SEED:-42}
DATA_SEED=${DATA_SEED:-$SEED}
ROLLOUT_SEED=${ROLLOUT_SEED:-$SEED}
export PYTHONHASHSEED=${SEED}

N_GPUS=${N_GPUS:-4}
ACTOR_MODEL=${ACTOR_MODEL:-/nfs-stor/zhengqing.gao/yuhao.wu/lzc/hf_cache/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08}
RM_MODEL=${RM_MODEL:-/nfs-stor/zhengqing.gao/yuhao.wu/lzc/hf_cache/hub/models--Skywork--Skywork-Reward-Llama-3.1-8B-v0.2/snapshots/d4117fbfd81b72f41b96341238baa1e3e90a4ce1}
TRAIN_FILE=${TRAIN_FILE:-/nfs-stor/zhengqing.gao/yuhao.wu/lzc/L2PEFT/data/hh_rlhf/train.parquet}
VAL_FILE=${VAL_FILE:-/nfs-stor/zhengqing.gao/yuhao.wu/lzc/L2PEFT/data/hh_rlhf/test.parquet}
CKPT_DIR=${CKPT_DIR:-/nfs-stor/zhengqing.gao/yuhao.wu/lzc/checkpoints/ppo_llama32_1B_hhrlhf_skywork8B_FFT_d${DATA_SEED}_r${ROLLOUT_SEED}}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-1024}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-256}
# veRL's PPO validation path returns empty metrics for model-style reward_model
# data, so keep built-in validation disabled for Skywork-RM training.
TEST_FREQ=${TEST_FREQ:-0}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}
FILTER_OVERLONG_PROMPTS=${FILTER_OVERLONG_PROMPTS:-True}
FILTER_WORKERS=${FILTER_WORKERS:-16}
RM_MICRO_BATCH_SIZE_PER_GPU=${RM_MICRO_BATCH_SIZE_PER_GPU:-4}

# Llama-3.2-1B is a base model in this cache and its tokenizer has no
# chat_template. veRL's RLHFDataset calls tokenizer.apply_chat_template on
# data.prompt, so provide a Llama-3 compatible template here.
LLAMA3_CHAT_TEMPLATE=${LLAMA3_CHAT_TEMPLATE:-"{{ bos_token }}{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}{{ message['content'] | trim }}{{ '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"}
export LLAMA3_CHAT_TEMPLATE

mkdir -p ${CKPT_DIR}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    +ray_kwargs.ray_init.include_dashboard=False \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=0.02 \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.prompt_key=prompt \
    data.train_batch_size=512 \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.val_max_samples=${VAL_MAX_SAMPLES} \
    data.seed=${DATA_SEED} \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=${FILTER_OVERLONG_PROMPTS} \
    data.filter_overlong_prompts_workers=${FILTER_WORKERS} \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.chat_template=\${oc.env:LLAMA3_CHAT_TEMPLATE} \
    actor_rollout_ref.model.path=${ACTOR_MODEL} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.seed=${ROLLOUT_SEED} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    critic.strategy=fsdp2 \
    critic.model.path=${ACTOR_MODEL} \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=False \
    critic.optim.lr=1e-5 \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.ppo_micro_batch_size_per_gpu=32 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    reward_model.enable=True \
    reward_model.strategy=fsdp2 \
    reward_model.model.path=${RM_MODEL} \
    reward_model.model.input_tokenizer=${ACTOR_MODEL} \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.max_length=2048 \
    reward_model.micro_batch_size_per_gpu=${RM_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.checkpoint.save_contents='[model,extra]' \
    critic.checkpoint.save_contents='[model,extra]' \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='Pruning' \
    trainer.experiment_name=ppo_hhrlhf_llama32_1B_skywork8B_FFT_d${DATA_SEED}_r${ROLLOUT_SEED} \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=${TEST_FREQ} \
    trainer.log_val_generations=${LOG_VAL_GENERATIONS} \
    trainer.total_epochs=2 \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.default_local_dir=${CKPT_DIR}
