import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

# ================= 配置区域 =================
# 采样时间窗口大小 (秒)，用于对齐不同 Engine 的数据
TIME_RESAMPLE_RULE = '1S' 
ENGINE_METRIC_PATTERN = "metrics/engine/engine_metric_*.json"
SCHEDULER_METRIC_PATTERN = "metrics/scheduler/scheduler_metric_*.json" 
# ===========================================

def load_json_data(pattern: str) -> List[Dict]:
    """加载最新的 JSON 指标文件"""
    files = glob.glob(pattern)
    if not files:
        print(f"⚠️ 未找到匹配的文件: {pattern}")
        return []
    
    # 找最新的文件
    latest_file = max(files, key=os.path.getctime)
    print(f"📂 正在加载: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
        
    # 兼容 combine 后的格式 [{"engine_id": x, "records": [...]}, ...]
    # 或者单个文件的格式 {"records": [...]}
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "records" in data:
        # 如果是单个文件，为了统一格式，包一层 list
        data.setdefault("engine_id", 0) # 默认 ID
        return [data]
    else:
        print("❌ 未知的数据格式")
        return []

def preprocess_metrics(engine_data: List[Dict], scheduler_data: List[Dict]) -> pd.DataFrame:
    """
    将原始 JSON 数据转换为按时间对齐的 Pandas DataFrame。
    """
    # 1. 转换 Engine 数据
    engine_records = []
    for item in engine_data:
        eid = item['engine_id']
        for r in item['records']:
            # 计算压力值 P_i = Running + Pending + Swapped
            pressure = r.get('running_requests', 0) + \
                       r.get('pending_requests', 0) + \
                       r.get('swapped_requests', 0)
            
            engine_records.append({
                'timestamp': pd.to_datetime(r['timestamp'], unit='s'),
                'engine_id': eid,
                'pressure': pressure,
                'gpu_usage': r.get('gpu_cache_usage', 0),
                'throughput': r.get('avg_generation_throughput', 0),
            })
    
    df_engine = pd.DataFrame(engine_records)
    if df_engine.empty:
        print("❌ Engine 数据为空")
        return pd.DataFrame()

    # 2. 转换 Scheduler 数据
    sched_records = []
    for item in scheduler_data:
        eid = item['engine_id']
        for r in item['records']:
            sched_records.append({
                'timestamp': pd.to_datetime(r['timestamp'], unit='s'),
                'engine_id': eid,
                'merge_right': r.get('merge_right_thresh', 0),
                'merge_left': r.get('merge_left_thresh', 0),
                'tput_merge': r.get('tput_merge', 0),
                'tput_unmerge': r.get('tput_unmerge', 0),
                'num_iters': r.get('num_iters', 0)
            })
            
    df_sched = pd.DataFrame(sched_records)

    # 3. 数据重采样与对齐
    # 我们需要构建一个系统级的视图：在时刻 T，Engine 0 的状态是什么，Engine 1 的状态是什么...
    
    # 设置索引
    df_engine.set_index('timestamp', inplace=True)
    if not df_sched.empty:
        df_sched.set_index('timestamp', inplace=True)

    # 按 Engine ID 分组并重采样
    # 目标结构: Index=Time, Columns=[(Engine0_Pressure, Engine0_Tput...), (Engine1_Pressure...)]
    
    def resample_group(df):
        # 取时间窗口内的平均值
        return df.resample(TIME_RESAMPLE_RULE).mean().interpolate(method='time')

    df_eng_resampled = df_engine.groupby('engine_id').apply(resample_group)
    # 此时索引是 (engine_id, timestamp)，我们需要把它变成宽表方便计算横向指标
    
    # Reset index to extract engine_id, then pivot
    df_eng_resampled = df_eng_resampled.drop(columns=['engine_id'], errors='ignore').reset_index()
    
    # 4. 计算系统级指标 (System-Level Metrics)
    # 我们按 Timestamp 分组，聚合所有 Engine 的数据
    
    system_stats = []
    
    # 获取所有唯一的时间戳
    timestamps = df_eng_resampled['timestamp'].unique()
    timestamps = timestamps[np.argsort(timestamps)]
    
    total_engines = df_eng_resampled['engine_id'].nunique()
    
    prev_merge_right = 0
    prev_merge_left = 0
    for ts in timestamps:
        slice_df = df_eng_resampled[df_eng_resampled['timestamp'] == ts]
        
        # 提取向量
        pressures = slice_df['pressure'].values
        usages = slice_df['gpu_usage'].values
        throughputs = slice_df['throughput'].values
        
        if len(pressures) < total_engines:
            pad_len = total_engines - len(pressures)
            pressures = np.pad(pressures, (0, pad_len), 'constant')
            usages = np.pad(usages, (0, pad_len), 'constant')
            throughputs = np.pad(throughputs, (0, pad_len), 'constant')

        # === 核心计算：负载失衡度 (Jain's Index based) ===
        IMBALANCE_THRESHOLD = 4 # 最小 Pressure 才计算失衡度
        sum_P = np.sum(pressures)
        if sum_P <= IMBALANCE_THRESHOLD:
            raw_imbalance = 0.0 # 绝对平衡
        else:
            sum_sq_P = np.sum(pressures ** 2)
            # Jain's Index J = (Σx)^2 / (N * Σx^2)
            # Imbalance = 1 - J
            jain_index = (sum_P ** 2) / (total_engines * sum_sq_P)
            raw_imbalance = 1.0 - jain_index
            
        # 收集 Scheduler 数据 (如果存在)
        avg_merge_right = 0
        avg_merge_left = 0
        if not df_sched.empty:
            # 同样做重采样处理逻辑，这里为了简化，假设 Scheduler 数据与 Engine 数据时间点相近
            # 实际生产中建议同样做 pivot 处理
            sched_slice = df_sched[(df_sched.index >= ts) & (df_sched.index < ts + pd.Timedelta(TIME_RESAMPLE_RULE))]
            if not sched_slice.empty:
                avg_merge_right = sched_slice['merge_right'].mean()
                avg_merge_left = sched_slice['merge_left'].mean()
            else:
                avg_merge_right = prev_merge_right
                avg_merge_left = prev_merge_left
        else:
            avg_merge_right = prev_merge_right
            avg_merge_left = prev_merge_left

        prev_merge_left = avg_merge_left
        prev_merge_right = avg_merge_right
        system_stats.append({
            'timestamp': ts,
            'load_imbalance': raw_imbalance,
            'total_throughput': np.sum(throughputs),
            'avg_gpu_usage': np.mean(usages),
            'total_pressure': sum_P,
            'avg_merge_right_thresh': avg_merge_right,
            'avg_merge_left_thresh': avg_merge_left
        })
        
    df_system = pd.DataFrame(system_stats)
    # 计算相对时间（秒）
    if not df_system.empty:
        start_time = df_system['timestamp'].min()
        df_system['relative_time'] = (df_system['timestamp'] - start_time).dt.total_seconds()
    df_sched['relative_time'] = (df_sched.index - df_sched.index.min()).total_seconds()
        
    return df_system, df_sched

def plot_analysis(df: pd.DataFrame, df_sched: pd.DataFrame = None):
    """绘制分析图表"""
    if df.empty:
        print("❌ 没有数据可绘图")
        return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
    
    # --- 图表 1: 负载失衡度与总请求压力 ---
    ax1 = axes[0]
    color = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Load Imbalance (0=Balanced, 1=Imbalanced)', color=color, fontweight='bold')
    sns.lineplot(data=df, x='relative_time', y='load_imbalance', ax=ax1, color=color, linewidth=2.5, label='Imbalance (LIC)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # 双轴：显示总压力，看失衡是在高负载还是低负载下发生的
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Total System Pressure (Reqs)', color=color, fontweight='bold')  
    sns.lineplot(data=df, x='relative_time', y='total_pressure', ax=ax2, color=color, linestyle='--', alpha=0.6, label='Total Pressure')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.fill_between(df['relative_time'], df['total_pressure'], color=color, alpha=0.1)
    
    plt.title('System Load Balance & Request Pressure', fontsize=16)
    
    # --- 图表 2: 系统吞吐量与 GPU 利用率 ---
    ax3 = axes[1]
    color = 'tab:green'
    ax3.set_ylabel('Total Throughput (tokens/s)', color=color, fontweight='bold')
    sns.lineplot(data=df, x='relative_time', y='total_throughput', ax=ax3, color=color, linewidth=2, label='Throughput')
    ax3.tick_params(axis='y', labelcolor=color)
    
    ax4 = ax3.twinx()
    color = 'tab:orange'
    ax4.set_ylabel('Avg GPU Cache Usage', color=color, fontweight='bold')
    sns.lineplot(data=df, x='relative_time', y='avg_gpu_usage', ax=ax4, color=color, linestyle='-.', label='GPU Usage')
    ax4.tick_params(axis='y', labelcolor=color)
    ax4.set_ylim(0, 1.05)
    
    plt.title('Throughput & Resource Utilization', fontsize=16)

    # --- 图表 3: 调度器阈值动态变化 ---
    ax5 = axes[2]
    ax5.set_ylabel('Merge Thresholds', fontweight='bold')
    sns.lineplot(data=df, x='relative_time', y='avg_merge_right_thresh', ax=ax5, label='Right Threshold (Enter Merge)', color='purple')
    sns.lineplot(data=df, x='relative_time', y='avg_merge_left_thresh', ax=ax5, label='Left Threshold (Keep Merge)', color='magenta', linestyle='--')
    
    # 标注相关性：如果失衡度高的时候，阈值是否有变化？
    # 这里简单画出失衡度高阈值线作为参考背景
    ax5_bg = ax5.twinx()
    ax5_bg.set_ylabel('Imbalance (Ref)', color='grey')
    sns.lineplot(data=df, x='relative_time', y='load_imbalance', ax=ax5_bg, color='grey', alpha=0.2, linewidth=0)
    ax5_bg.fill_between(df['relative_time'], df['load_imbalance'], color='grey', alpha=0.1) # 背景阴影表示失衡
    ax5_bg.set_yticks([]) # 隐藏刻度，只做背景

    # --- 图表 4: 不同调度器合并吞吐量与非合并吞吐量的比值, 1.0 为标准线
    if df_sched is not None and not df_sched.empty:
        ax6 = axes[3]
        df_sched['merge_unmerge_ratio'] = df_sched.apply(
            lambda row: (row['tput_merge'] / row['tput_unmerge']) if row['tput_unmerge'] > 0 else np.nan, axis=1)
        sns.lineplot(data=df_sched, x='relative_time', y='merge_unmerge_ratio', ax=ax6, color='teal', linewidth=2, hue='engine_id', palette='tab10')
        ax6.axhline(1.0, color='red', linestyle='--', label='Baseline Ratio = 1.0')
        ax6.set_title('Scheduler Merge vs Unmerge Throughput Ratio', fontsize=16)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Merge/Unmerge Throughput Ratio')
        ax6.legend()
    else:
        axes[3].set_visible(False)
    
    plt.title('Scheduler Merge Thresholds Dynamics', fontsize=16)
    ax5.set_xlabel('Time (seconds) from Start', fontsize=14)



    plt.tight_layout()
    output_path = "benchmark_results/system_load_analysis.png"
    plt.savefig(output_path)
    print(f"✅ 图表已保存至: {output_path}")
    plt.show()

# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 加载数据
    engine_data = load_json_data(ENGINE_METRIC_PATTERN)
    scheduler_data = load_json_data(SCHEDULER_METRIC_PATTERN)

    if engine_data:
        # 2. 处理与计算指标
        df_system, df_sched = preprocess_metrics(engine_data, scheduler_data)
        
        # 3. 打印统计摘要
        print("\n=== System Load Summary ===")
        print(df_system[['load_imbalance', 'total_throughput', 'avg_merge_right_thresh']].describe())
        
        # 4. 生成图表
        plot_analysis(df_system, df_sched)
        
        # 5. 可选：保存处理后的数据以便进一步分析
        df_system.to_csv("processed_system_metrics.csv", index=False)
    else:
        print("程序终止：无数据。")