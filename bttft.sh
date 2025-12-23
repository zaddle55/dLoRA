#!/bin/bash

# ================= 配置区域 =================
# 环境设置
source ~/.bashrc
conda activate dlora
PROJECT_ROOT="/home/lizz_lab/cse12310823/dLoRA-artifact"
cd $PROJECT_ROOT
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# 参数配置
MODEL_PATH="/home/lizz_lab/cse12310823/models_cache/models/facebook--opt-125m"
PORT=8000
HOST="127.0.0.1"
NUM_MODELS=$3
NUM_GROUPS=4
TP_SIZE=1
TRACE_NAME="azure_v2"
TRACE_PATH="${PROJECT_ROOT}/trace/"
REQUEST_RATE=$2
POLICY=$1  # scheduling policy
OUTPUT_STYLE=6
OUTPUT_PATH=$4
LOAD_BALANCE_VIS_PATH="${PROJECT_ROOT}/ae_scripts/vis_loadbal.py"
SCHEDULER_VIS_PATH="${PROJECT_ROOT}/ae_scripts/vis_sched.py"

# ================= 第一步：启动 Ray =================
echo "🚀 [1/4] Starting Ray instance on node $(hostname)..."
ray stop --force > /dev/null 2>&1
# 启动本地 Ray 头节点
ray start --head --num-cpus=32 --num-gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l) --disable-usage-stats --include-dashboard=False

# ================= 第二步：后台启动 Server =================
echo "🚀 [2/4] Starting vLLM API Server in BACKGROUND..."
python -m vllm.entrypoints.api_server \
    --model $MODEL_PATH \
    --host $HOST \
    --port $PORT \
    --num-models $NUM_MODELS \
    --num-groups $NUM_GROUPS \
    --tensor-parallel-size $TP_SIZE \
    --swap-space 16 \
    --disable-log-requests \
    --worker-use-ray \
    --engine-use-ray \
    --trust-remote-code \
    --policy $POLICY \
    > logs/server_output_$(date +%s).log 2>&1 &

# 获取刚才启动的 Server 进程 ID (PID)，用于稍后关闭它
SERVER_PID=$!
echo "   -> Server PID is: $SERVER_PID"

# ================= 第三步：等待 Server 就绪 =================
echo "⏳ [3/4] Waiting for Server to be ready..."

# 循环检查 /health 接口，直到返回 200 OK
MAX_RETRIES=60  # 最多等待 5 分钟
COUNT=0
while true; do
    # 检查进程是否还活着
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ Server process died unexpectedly! Check logs."
        cat logs/server_output_${SLURM_JOB_ID}.log
        exit 1
    fi

    # 发送健康检查请求
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://$HOST:$PORT/health)
    
    if [ "$HTTP_CODE" == "200" ]; then
        echo "✅ Server is UP and READY!"
        break
    fi

    COUNT=$((COUNT+1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "❌ Timeout waiting for server."
        kill $SERVER_PID
        exit 1
    fi

    echo "   ... waiting ($COUNT/$MAX_RETRIES)"
    sleep 5
done

# ================= 第四步：运行 Benchmark =================
echo "🚀 [4/4] Starting Benchmark Client..."

python benchmarks/benchmark_serving.py \
    --backend vllm \
    --host $HOST \
    --port $PORT \
    --dataset "${PROJECT_ROOT}/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json" \
    --tokenizer $MODEL_PATH \
    --num-models $NUM_MODELS \
    --num-prompts 600 \
    --request-rate $REQUEST_RATE \
    --trust-remote-code \
    --output_style $OUTPUT_STYLE \
    --output $OUTPUT_PATH \
    --policy $POLICY \
    --trace_name $TRACE_NAME \
    --trace_path $TRACE_PATH 

# 记录 benchmark 的退出码
BENCH_EXIT_CODE=$?

# ================= 收尾工作 =================
echo "🛑 Benchmark finished. Cleaning up..."
# 杀掉 Server 进程
kill $SERVER_PID
# 停止 Ray
ray stop --force

echo "All done. Log saved to logs/server_output_$(date +%s).log"
exit $BENCH_EXIT_CODE