#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4  # 根据脚本中trainer.n_gpus_per_node=8调整为8个GPU
#SBATCH --nodes=1          # trainer.nnodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10 # 由于是多GPU训练，增加CPU核心数
#SBATCH --mem=256GB        # 增加内存到256GB以避免OOM
#SBATCH --time=24:00:00
#SBATCH --job-name=vl
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.log # %x是job-name，%j是job-id
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dingjiesong.cs@gmail.com

# 设置环境变量
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

PROJECT_NAME='verl_grpo_qwenvl_vtorl'
EXPERIMENT_NAME='qwen2_5_vl_3b_vtorl_baseline'

############### Model and Data ###############
# TRAIN_FILE='/home/songdingjie/data/verl/virl/train.parquet'
TRAIN_FILE='/home/songdingjie/data/verl/virl_8192/train.parquet'
VAL_FILE='/home/songdingjie/data/verl/virl_8192/val.parquet'
# TRAIN_FILE='/home/songdingjie/data/verl/virl/val.parquet'
# VAL_FILE='/home/songdingjie/data/verl/virl/train.parquet'
MODEL_PATH='/home/songdingjie/models/hf/Qwen2.5-VL-3B-Instruct'

############### GPU Config ###############
TENSOR_MODEL_PARALLEL_SIZE=1


############### Training Config ###############

#==== Core Config ====#
N_GPUS_PER_NODE=4   # GPU 数
TRAIN_BATCH_SIZE=$(( N_GPUS_PER_NODE * 64 ))   # rollout 采样用的batch size，N_GPUS_PER_NODE的64倍
TRAIN_MINI_BATCH_SIZE=$(( N_GPUS_PER_NODE * 8 ))   # Actor 训练用的batch size
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=$(( N_GPUS_PER_NODE * 8 ))    # 设置为TRAIN_BATCH_SIZE的因数


#==== Actor and Critic Config ====#
ADV_ESTIMATOR='grpo'
TRAIN_MICRO_BATCH_SIZE_PER_GPU=1  # 每个GPU的batch size，gradient acc为自动计算。3B模型可以用2
LR=1e-6
TOTAL_EPOCHS=3 # VL-Thinker用了3个epoch

#==== Data Config ====#
MAX_PROMPT_LENGTH=24000   # 减小最大prompt长度
MAX_RESPONSE_LENGTH=4096 # 减小最大response长度
FILTER_OVERLONG_PROMPTS=True  # 开启过长prompt过滤
PROMPT_TRUNCATION='right'   # 右截断过长prompt
IMAGE_KEY='images'         # 数据集图像键名


#==== Reference Config ====#
USE_KL_LOSS=True
KL_LOSS_TYPE='low_var_kl'
KL_LOSS_COEFF=0.01
ENTROPY_COEFF=0

#==== Rollout Config ====#
DISABLE_LOG_STATS=False     # 记录rollout的统计信息
ROLLOUT_GPU_MEMORY_UTILIZATION=0.6  # 预分配 GPU KVCache
ROLLOUT_ENABLE_CHUNKED_PREFILL=False
ROLLOUT_ENFORCE_EAGER=False
ROLLOUT_FREE_CACHE_ENGINE=False
ROLLOUT_LIMIT_IMAGES=100
ROLLOUT_N=$N_GPUS_PER_NODE


############### Run ###############

/blue/yonghui.wu/songdingjie/.conda/envs/torl/bin/python -m verl.trainer.main_ppo \
    \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=$FILTER_OVERLONG_PROMPTS \
    data.truncation=$PROMPT_TRUNCATION \
    data.image_key=$IMAGE_KEY \
    \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$TRAIN_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEFF \
    actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.disable_log_stats=$DISABLE_LOG_STATS \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.enable_chunked_prefill=$ROLLOUT_ENABLE_CHUNKED_PREFILL \
    actor_rollout_ref.rollout.enforce_eager=$ROLLOUT_ENFORCE_EAGER \
    actor_rollout_ref.rollout.free_cache_engine=$ROLLOUT_FREE_CACHE_ENGINE \
    +actor_rollout_ref.rollout.limit_images=$ROLLOUT_LIMIT_IMAGES \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=$TOTAL_EPOCHS $@
