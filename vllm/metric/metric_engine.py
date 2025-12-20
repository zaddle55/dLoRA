from typing import List
from vllm.metric.metric_base import MetricPrototype
class EngineMetric(MetricPrototype):
    def __init__(self, output_dir: str, step_per_log: int = 1):
        """
        引擎级别的数据收集器。

        Args:
            output_dir (str): 数据输出的文件夹路径。
            step_per_log (int): 每隔多少步记录一次数据，默认为 1。
        """
        super().__init__(output_dir, "engine_metric", step_per_log)
        self.engine_id = 0
        self.engine_args = {}

    def init_engine(self, engine_id: int, engine_args: dict):
        """
        初始化引擎ID。

        Args:
            engine_id (int): 引擎的唯一标识符。
        """
        self.engine_id = engine_id
        self.engine_args = engine_args

    def step(self, 
             avg_promopt_throughput: float, 
             avg_generation_throughput: float,
             avg_JCT: float,
             ready_requests: int,
             running_requests: int,
             swapped_requests: int,
             pending_requests: int,
             gpu_cache_usage: float,
             cpu_cache_usage: float,
             **kwargs
             ):
        """
        记录引擎级别的数据。

        Args:
            avg_promopt_throughput (float): 平均提示词处理吞吐量。
            avg_generation_throughput (float): 平均生成吞吐量。
            avg_JCT (float): 平均作业完成时间。
            ready_requests (int): 准备就绪的请求数量。
            running_requests (int): 正在运行的请求数量。
            swapped_requests (int): 已交换的请求数量。
            pending_requests (int): 待处理的请求数量。
            gpu_cache_usage (float): GPU缓存使用率。
            cpu_cache_usage (float): CPU缓存使用率。
            **kwargs: 其他可选的引擎相关数据。
        """
        super().step()
        if self.should_log():
            self.log({
                "avg_promopt_throughput": avg_promopt_throughput,
                "avg_generation_throughput": avg_generation_throughput,
                "avg_JCT": avg_JCT,
                "ready_requests": ready_requests,
                "running_requests": running_requests,
                "swapped_requests": swapped_requests,
                "pending_requests": pending_requests,
                "gpu_cache_usage": gpu_cache_usage,
                "cpu_cache_usage": cpu_cache_usage,
                **kwargs
            })

    def summary(self):
        """
        记录引擎的配置信息作为总结数据。

        Returns:
            Dict[str, Any]: 包含引擎ID和引擎参数的字典。
        """
        self.log({
            "engine_id": self.engine_id,
            "engine_args": self.engine_args,
        })

    @staticmethod
    def combine(metrics: List["EngineMetric"]) -> "EngineMetric":
        combined_metric = EngineMetric(output_dir=metrics[0].output_dir)
        for metric in metrics:
            combined_metric.records.append({
                "engine_id": metric.engine_id,
                "records": metric.records,
            })
        return combined_metric