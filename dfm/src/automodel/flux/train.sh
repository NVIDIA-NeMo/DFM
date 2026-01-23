#!/bin/bash
# FLUX 训练启动脚本 (TensorBoard + 分片 ckpt 保存)

export NCCL_TIMEOUT=1800
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_BLOCKING_WAIT=1

echo "DEBUG in train.sh:"
echo "  NCCL_TIMEOUT=$NCCL_TIMEOUT"
echo "  NCCL_BLOCKING_WAIT=$NCCL_BLOCKING_WAIT"
echo "  TORCH_NCCL_BLOCKING_WAIT=$TORCH_NCCL_BLOCKING_WAIT"

# GPU 数量
NUM_GPUS=8

# 数据路径
META_FOLDER="./data/processed_256_256_3meta"

# 模型路径（你本地的 FLUX 基础模型位置）
MODEL_ID="./models/FLUX.1-dev"

# 训练参数
BATCH_SIZE_PER_GPU=1
NUM_EPOCHS=10000
LEARNING_RATE=1e-5
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0

# Flow Matching 参数
LOGIT_MEAN=0.0
LOGIT_STD=1.0
FLOW_SHIFT=3.0
MIX_UNIFORM_RATIO=0.1
SIGMA_MIN=0.0
SIGMA_MAX=1.0
NUM_TRAIN_TIMESTEPS=1000

# 输出参数
OUTPUT_DIR="./outputs/flux_pretraining_ckpt"
LOG_EVERY=10
NUM_WORKERS=4

# 每多少个 epoch 保存一次分片 ckpt（对应 main_flux.py 里的 --save_every）
SAVE_EVERY=100

# 验证频率（iteration），目前关闭
VALIDATE_EVERY=0
VAL_NUM_INFERENCE_STEPS=28
VAL_GUIDANCE_SCALE=3.5
VAL_HEIGHT=256
VAL_WIDTH=256

echo "=========================================="
echo "FLUX 训练 (TensorBoard + 分片 ckpt 保存)"
echo "=========================================="
echo ""

# 检查 GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ 错误: 未检测到 NVIDIA GPU"
    exit 1
fi

echo "检测到的 GPU:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo ""
echo "配置信息:"
echo "  GPU 数量: $NUM_GPUS"
echo "  每 GPU Batch Size: $BATCH_SIZE_PER_GPU"
echo "  全局 Batch Size: $((BATCH_SIZE_PER_GPU * NUM_GPUS))"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo ""
echo "Flow Matching 参数:"
echo "  Flow Shift: $FLOW_SHIFT"
echo "  Mix Uniform Ratio: $MIX_UNIFORM_RATIO"
echo "  Sigma Range: [$SIGMA_MIN, $SIGMA_MAX]"
echo ""
echo "验证推理参数:"
echo "  validate_every: $VALIDATE_EVERY"
echo "  val_num_inference_steps: $VAL_NUM_INFERENCE_STEPS"
echo "  val_guidance_scale: $VAL_GUIDANCE_SCALE"
echo "  val_resolution: ${VAL_HEIGHT}x${VAL_WIDTH}"
echo ""
echo "ckpt 保存:"
echo "  save_every (epochs): $SAVE_EVERY"
echo ""

# 检查数据
if [ ! -d "$META_FOLDER" ]; then
    echo "❌ 错误: 数据文件夹不存在: $META_FOLDER"
    exit 1
fi

echo "✓ 所有检查通过"
echo ""

# NCCL 环境变量（可选，用于优化通信）
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# 平台会自动注入 TENSORBOARD_LOG_PATH，若想调试可打印：
echo "TENSORBOARD_LOG_PATH=${TENSORBOARD_LOG_PATH}"

echo "=========================================="
echo "启动训练..."
echo "=========================================="
echo ""

torchrun \
    --nproc-per-node=$NUM_GPUS \
    --master_port=29500 \
    scripts/main_flux.py \
    --meta_folder "$META_FOLDER" \
    --model_id "$MODEL_ID" \
    --batch_size_per_gpu $BATCH_SIZE_PER_GPU \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --max_grad_norm $MAX_GRAD_NORM \
    --logit_mean $LOGIT_MEAN \
    --logit_std $LOGIT_STD \
    --flow_shift $FLOW_SHIFT \
    --mix_uniform_ratio $MIX_UNIFORM_RATIO \
    --sigma_min $SIGMA_MIN \
    --sigma_max $SIGMA_MAX \
    --num_train_timesteps $NUM_TRAIN_TIMESTEPS \
    --output_dir "$OUTPUT_DIR" \
    --log_every $LOG_EVERY \
    --num_workers $NUM_WORKERS \
    --validate_every $VALIDATE_EVERY \
    --val_num_inference_steps $VAL_NUM_INFERENCE_STEPS \
    --val_guidance_scale $VAL_GUIDANCE_SCALE \
    --val_height $VAL_HEIGHT \
    --val_width $VAL_WIDTH \
    --save_every $SAVE_EVERY

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 训练成功完成!"
    echo "=========================================="
    echo ""
    echo "输出目录: $OUTPUT_DIR"
else
    echo ""
    echo "=========================================="
    echo "❌ 训练失败"
    echo "=========================================="
    echo ""
    echo "请检查日志: $OUTPUT_DIR/train.log"
    exit 1
fi