#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2   # 每个节点4张A100 GPU
#SBATCH --nodes=2           # 使用2个节点
#SBATCH --ntasks-per-node=1 # 每个节点1个任务
#SBATCH --cpus-per-task=2  # 从16减少到8
#SBATCH --mem=256GB         # 每个节点内存
#SBATCH --time=32:00:00
#SBATCH --job-name=vl_multinode
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dingjiesong.cs@gmail.com
#SBATCH --exclusive         # 独占节点以确保资源可用性

# 设置环境变量
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

# Ray相关设置
export RAY_BACKEND_LOG_LEVEL=warning
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

# 项目配置
PROJECT_NAME='verl_grpo_qwenvl_vtorl_multinode'
EXPERIMENT_NAME='qwen2_5_vl_3b_vtorl_multinode'

############### Model and Data ###############
TRAIN_FILE='/home/songdingjie/data/verl/virl_8192/train.parquet'
VAL_FILE='/home/songdingjie/data/verl/virl_8192/val.parquet'
MODEL_PATH='/home/songdingjie/models/hf/Qwen2.5-VL-3B-Instruct'

############### GPU Config ###############
TENSOR_MODEL_PARALLEL_SIZE=1

############### Training Config ###############

#==== Multi-node Config ====#
# 获取SLURM分配的节点列表
NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
NODES_ARRAY=($NODES)
HEAD_NODE=${NODES_ARRAY[0]}
NUM_NODES=${#NODES_ARRAY[@]}
MASTER_ADDR=$HEAD_NODE
MASTER_PORT=6000

# 打印节点信息
echo "Head node: $HEAD_NODE"
echo "All nodes: $NODES"
echo "Number of nodes: $NUM_NODES"

#==== Core Config ====#
N_GPUS_PER_NODE=2                                  # 每个节点的GPU数量
TOTAL_GPUS=$(( N_GPUS_PER_NODE * NUM_NODES ))      # 总GPU数量
TRAIN_BATCH_SIZE=$(( TOTAL_GPUS * 32 ))            # 增加总批量大小以利用更多GPU
TRAIN_MINI_BATCH_SIZE=$(( TOTAL_GPUS * 8 ))        # 调整Actor训练批量大小
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=$(( N_GPUS_PER_NODE * 8 ))

#==== Actor and Critic Config ====#
ADV_ESTIMATOR='grpo'
TRAIN_MICRO_BATCH_SIZE_PER_GPU=1
LR=1e-6
TOTAL_EPOCHS=3

#==== Data Config ====#
MAX_PROMPT_LENGTH=24000
MAX_RESPONSE_LENGTH=4096
FILTER_OVERLONG_PROMPTS=True
PROMPT_TRUNCATION='right'
IMAGE_KEY='images'

#==== Reference Config ====#
USE_KL_LOSS=True
KL_LOSS_TYPE='low_var_kl'
KL_LOSS_COEFF=0.01
ENTROPY_COEFF=0

#==== Rollout Config ====#
DISABLE_LOG_STATS=False
ROLLOUT_GPU_MEMORY_UTILIZATION=0.6
ROLLOUT_ENABLE_CHUNKED_PREFILL=False
ROLLOUT_ENFORCE_EAGER=False
ROLLOUT_FREE_CACHE_ENGINE=False
ROLLOUT_LIMIT_IMAGES=100
ROLLOUT_N=$N_GPUS_PER_NODE

############### 启动分布式训练 ###############

# 如果当前节点是主节点，启动Ray head
if [ "$(hostname)" = "$HEAD_NODE" ]; then
    echo "Starting Ray Head on $HEAD_NODE"
    ray start --head \
        --port=6379 \
        --num-cpus=$SLURM_CPUS_PER_TASK \
        --num-gpus=$N_GPUS_PER_NODE \
        --block

    # 等待Ray head启动完成
    sleep 10

    # 获取Ray head的地址
    RAY_ADDRESS=$(ray address)
    echo "Ray head address: $RAY_ADDRESS"

    # 将Ray地址写入共享文件，供其他节点读取
    echo "$RAY_ADDRESS" > /tmp/ray_address.txt

    # 等待所有其他节点加入
    EXPECTED_WORKERS=$(( NUM_NODES - 1 ))
    MAX_WAIT=60
    COUNTER=0

    while [ $COUNTER -lt $MAX_WAIT ]; do
        WORKERS=$(ray status | grep "Workers" | awk '{print $2}')
        if [ "$WORKERS" = "$EXPECTED_WORKERS" ]; then
            echo "All $EXPECTED_WORKERS worker nodes have joined."
            break
        fi
        echo "Waiting for worker nodes to join... ($WORKERS/$EXPECTED_WORKERS)"
        COUNTER=$((COUNTER+1))
        sleep 5
    done

    if [ $COUNTER -eq $MAX_WAIT ]; then
        echo "Timeout waiting for worker nodes to join. Proceeding anyway."
    fi

    # 启动分布式训练任务
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
        trainer.nnodes=$NUM_NODES \
        trainer.master_addr=$MASTER_ADDR \
        trainer.master_port=$MASTER_PORT \
        trainer.distributed_init_method="tcp://$MASTER_ADDR:$MASTER_PORT" \
        trainer.distributed_backend="nccl" \
        trainer.save_freq=-1 \
        trainer.test_freq=5 \
        trainer.total_epochs=$TOTAL_EPOCHS \
        trainer.use_ray=True \
        trainer.ray_address="auto" $@
else
    # 工作节点: 等待Ray head启动并加入集群
    # 等待主节点创建Ray地址文件
    MAX_WAIT=120
    COUNTER=0

    while [ ! -f /tmp/ray_address.txt ] && [ $COUNTER -lt $MAX_WAIT ]; do
        echo "Waiting for Ray head to start..."
        COUNTER=$((COUNTER+1))
        sleep 2
    done

    if [ ! -f /tmp/ray_address.txt ]; then
        echo "Timeout waiting for Ray head to start. Exiting."
        exit 1
    fi

    # 读取Ray head地址
    RAY_ADDRESS=$(cat /tmp/ray_address.txt)
    echo "Connecting to Ray head at $RAY_ADDRESS"

    # 连接到Ray集群
    ray start --address=$RAY_ADDRESS \
        --num-cpus=$SLURM_CPUS_PER_TASK \
        --num-gpus=$N_GPUS_PER_NODE

    # 保持工作节点运行直到作业完成
    while true; do
        if ! ray status &>/dev/null; then
            echo "Ray head is no longer accessible. Exiting."
            break
        fi
        sleep 60
    done
fi

# 清理Ray进程
ray stop
