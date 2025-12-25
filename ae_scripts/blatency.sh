# benchmark serving 脚本路径
BENCHMARK_SCRIPT="bttft.sh"
# 输出路径
OUTPUT_PATH="/home/lizz_lab/cse12310823/dLoRA-artifact/benchmark_results/fig9.csv"

if [ ! -f $OUTPUT_PATH ]; then
    echo "Exec Type,Request_Rate,Latency(s)" > $OUTPUT_PATH
fi

EXEC_TYPE=1 # PEFT

./$BENCHMARK_SCRIPT "credit" 2 8 $OUTPUT_PATH 4 $EXEC_TYPE 1.0
./$BENCHMARK_SCRIPT "credit" 4 8 $OUTPUT_PATH 4 $EXEC_TYPE 1.0
./$BENCHMARK_SCRIPT "credit" 6 8 $OUTPUT_PATH 4 $EXEC_TYPE 1.0
./$BENCHMARK_SCRIPT "credit" 8 8 $OUTPUT_PATH 4 $EXEC_TYPE 1.0
./$BENCHMARK_SCRIPT "credit" 10 8 $OUTPUT_PATH 4 $EXEC_TYPE 1.0
./$BENCHMARK_SCRIPT "credit" 12 8 $OUTPUT_PATH 4 $EXEC_TYPE 1.0

echo "-------------------------------------"
echo "Completed benchmarks for EXEC_TYPE=1 (PEFT)"
echo "-------------------------------------"

EXEC_TYPE=2 # vLLM

./$BENCHMARK_SCRIPT "credit" 4 8 $OUTPUT_PATH 4 $EXEC_TYPE 1.0
./$BENCHMARK_SCRIPT "credit" 8 8 $OUTPUT_PATH 4 $EXEC_TYPE 1.0
./$BENCHMARK_SCRIPT "credit" 12 8 $OUTPUT_PATH 4 $EXEC_TYPE 1.0
./$BENCHMARK_SCRIPT "credit" 16 8 $OUTPUT_PATH 4 $EXEC_TYPE 1.0
./$BENCHMARK_SCRIPT "credit" 20 8 $OUTPUT_PATH 4 $EXEC_TYPE 1.0
./$BENCHMARK_SCRIPT "credit" 24 8 $OUTPUT_PATH 4 $EXEC_TYPE 1.0

echo "-------------------------------------"
echo "Completed benchmarks for EXEC_TYPE=2 (vLLM)"
echo "-------------------------------------"

EXEC_TYPE=3 #dLora
./$BENCHMARK_SCRIPT "credit" 8 8 $OUTPUT_PATH 4 $EXEC_TYPE 1.0
./$BENCHMARK_SCRIPT "credit" 16 8 $OUTPUT_PATH 4 $EXEC_TYPE 1.0
./$BENCHMARK_SCRIPT "credit" 24 8 $OUTPUT_PATH 4 $EXEC_TYPE 1.0
./$BENCHMARK_SCRIPT "credit" 32 8 $OUTPUT_PATH 4 $EXEC_TYPE 1.0
./$BENCHMARK_SCRIPT "credit" 40 8 $OUTPUT_PATH 4 $EXEC_TYPE 1.0
./$BENCHMARK_SCRIPT "credit" 48 8 $OUTPUT_PATH 4 $EXEC_TYPE 1.0

echo "-------------------------------------"
echo "Completed benchmarks for EXEC_TYPE=3 (dLora)"
echo "-------------------------------------"

./$BENCHMARK_SCRIPT "mlfq" 8 8 $OUTPUT_PATH 4 $EXEC_TYPE 0.3
./$BENCHMARK_SCRIPT "mlfq" 16 8 $OUTPUT_PATH 4 $EXEC_TYPE 0.3
./$BENCHMARK_SCRIPT "mlfq" 24 8 $OUTPUT_PATH 4 $EXEC_TYPE 0.3
./$BENCHMARK_SCRIPT "mlfq" 32 8 $OUTPUT_PATH 4 $EXEC_TYPE 0.3
./$BENCHMARK_SCRIPT "mlfq" 40 8 $OUTPUT_PATH 4 $EXEC_TYPE 0.3
./$BENCHMARK_SCRIPT "mlfq" 48 8 $OUTPUT_PATH 4 $EXEC_TYPE 0.3

echo "✅ All benchmarks completed!"
