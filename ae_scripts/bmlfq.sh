#!/bin/bash

# ================= 配置区域 =================
# policy 网格
POLICIES=("mlfq" "credit")
# num-models 网格
NUM_MODELS_LIST=(6)
# req-rate 网格
REQUEST_RATES=(4 8 12 16 20)
# benchmark serving 脚本路径
BENCHMARK_SCRIPT="bttft.sh"
# 输出路径
OUTPUT_PATH="/home/lizz_lab/cse12310823/dLoRA-artifact/benchmark_results/po_rr_nm_ttft.csv"
# =========================================

# 如果没有输出文件，添加表头
if [ ! -f $OUTPUT_PATH ]; then
    echo "Policy,Request_Rate,Num_Models,TTFT(ms)" > $OUTPUT_PATH
fi

# 遍历所有参数组合
for NUM_MODELS in "${NUM_MODELS_LIST[@]}"; do
    for REQUEST_RATE in "${REQUEST_RATES[@]}"; do
        for POLICY in "${POLICIES[@]}"; do
            echo "🔄 Running benchmark with Policy=$POLICY, Request_Rate=$REQUEST_RATE, Num_Models=$NUM_MODELS"
            ./$BENCHMARK_SCRIPT $POLICY $REQUEST_RATE $NUM_MODELS $OUTPUT_PATH
        done
    done
done

echo "✅ All benchmarks completed!"