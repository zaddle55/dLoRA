import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# ================= 配置区域 =================
JSON_FILE = "metrics/engine/engine_metric_*.json" 
output_path = "benchmark_results/dlora_serving_load_balance_report5.png"
# ===========================================

def load_data(pattern):
    files = glob.glob(pattern)
    if not files:
        print("❌ 未找到 metric JSON 文件")
        return None
    latest_file = max(files, key=os.path.getctime)
    print(f"📂 正在加载文件: {latest_file}")

    with open(latest_file, 'r') as f:
        raw_data = json.load(f)

    all_records = []
    if isinstance(raw_data, dict) and "records" in raw_data:
        # 单个 Engine 的情况
        eng_id = raw_data.get("engine_id", 0)
        for r in raw_data["records"]:
            r["engine_id"] = eng_id
            all_records.append(r)
    elif isinstance(raw_data, list):
        # 多个 Engine Combine 的情况
        for engine_block in raw_data:
            if "records" in engine_block:
                eng_id = engine_block["engine_id"]
                for r in engine_block["records"]:
                    r["engine_id"] = eng_id
                    all_records.append(r)
            else:
                all_records.append(engine_block)

    df = pd.DataFrame(all_records)
    
    # 将时间戳转换为相对时间（从 0 秒开始）
    if "timestamp" in df.columns:
        start_time = df["timestamp"].min()
        df["relative_time"] = df["timestamp"] - start_time
    else:
        # 如果没有时间戳，就用 index 模拟
        df["relative_time"] = df.index
        
    return df

def plot_dashboard(df, args):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'dLoRA Load Balancing Dashboard at req_rate = {args.req_rate} req/s, num_models = {args.num_models}', fontsize=20)

    # ------------------------------------------------------
    # 图 1: 负载均衡情况 (Pending + Running Requests)
    # ------------------------------------------------------
    df["total_load"] = df["running_requests"] + df["pending_requests"]
    sns.lineplot(ax=axes[0, 0], data=df, x="relative_time", y="total_load", hue="engine_id", 
                 palette="tab10", linewidth=2.5)
    axes[0, 0].set_title("Load Distribution (Queue Depth)", fontsize=14)
    axes[0, 0].set_ylabel("Total Requests (Running + Pending)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].grid(False)

    # ------------------------------------------------------
    # 图 2: KV Cache 利用率 (GPU Memory Balance)
    # ------------------------------------------------------
    sns.lineplot(ax=axes[0, 1], data=df, x="relative_time", y="gpu_cache_usage", hue="engine_id", 
                 palette="tab10", linewidth=2.5)
    axes[0, 1].set_title("GPU KV Cache Utilization", fontsize=14)
    axes[0, 1].set_ylabel("Cache Usage (0-1.0)")
    axes[0, 1].set_ylim(0, 0.6)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].grid(False)

    # ------------------------------------------------------
    # 图 3: 生成吞吐量 (Throughput Stability)
    # ------------------------------------------------------
    sns.lineplot(ax=axes[1, 0], data=df, x="relative_time", y="avg_generation_throughput", hue="engine_id", 
                 palette="tab10")
    axes[1, 0].set_title("Generation Throughput per Engine", fontsize=14)
    axes[1, 0].set_ylabel("Tokens / sec")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].grid(False)

    # ------------------------------------------------------
    # 图 4: JCT 分布对比 (Latency Fairness)
    # ------------------------------------------------------
    # 过滤掉 JCT 为 0 的初始数据
    valid_jct = df[df["avg_JCT"] > 0]
    sns.boxplot(ax=axes[1, 1], data=valid_jct, x="engine_id", y="avg_JCT", palette="tab10")
    axes[1, 1].set_title("Average JCT Distribution by Engine", fontsize=14)
    axes[1, 1].set_ylabel("Avg Job Completion Time (s)")
    axes[1, 1].set_xlabel("Engine ID")
    axes[1, 1].grid(False)

    plt.tight_layout()
    output_path = f"benchmark_results/serving_load_balance_report_rr{args.req_rate}_nm{args.num_models}.png"
    plt.savefig(output_path)
    print(f"✅ 图表已保存为 {output_path}")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="dLoRA Load Balancing Visualization")
    parser.add_argument("--req-rate", type=int, default=12, help="请求速率 (requests per second)")
    parser.add_argument("--num-models", type=int, default=4, help="模型数量")
    args = parser.parse_args()
    df = load_data(JSON_FILE)
    if df is not None and not df.empty:
        df["engine_id"] = df["engine_id"].astype(str)
        plot_dashboard(df, args)
    else:
        print("没有足够的数据进行绘图。")