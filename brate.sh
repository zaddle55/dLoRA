#!/bin/bash

# ================= 配置区域 =================
# req-rate 网格
REQUEST_RATES=(4 8 12 16 20)
# num-models 网格
NUM_MODELS_LIST=(4 6 8 10)
# EMA 平滑参数 alpha 网格
ALPHAS=(0.1 0.2 0.3 0.35 1.0)
# benchmark serving 脚本路径
BENCHMARK_SCRIPT="bserve.sh"
# =========================================

# 遍历所有参数组合
for REQUEST_RATE in "${REQUEST_RATES[@]}"; do
    for NUM_MODELS in "${NUM_MODELS_LIST[@]}"; do
        for ALPHA in "${ALPHAS[@]}"; do
            echo "=============================================="
            echo "🚀 Running benchmark with Request Rate: $REQUEST_RATE, Num Models: $NUM_MODELS, Alpha: $ALPHA"
            echo "=============================================="
            ./$BENCHMARK_SCRIPT $REQUEST_RATE $NUM_MODELS $ALPHA
        done
    done
done

# ================= 收尾工作 =================
echo "✅ All benchmarks completed!"
