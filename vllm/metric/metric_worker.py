from vllm.metric.metric_base import MetricPrototype

class InstanceMetric(MetricPrototype):
    def __init__(self, output_dir: str, step_per_log: int = 1):
        """
        GPU实例级别的数据收集器。

        Args:
            output_dir (str): 数据输出的文件夹路径。
            step_per_log (int): 每隔多少步记录一次数据，默认为 1。
        """
        super().__init__(output_dir, "instance_metric", step_per_log)

    def step(self, kv_cache: int, latency: float, merge: int, unmerge: int):
        """
        记录GPU实例级别的数据。

        Args:
            kv_cache (int): 当前KV缓存大小。
            latency (float): 当前请求的延迟时间。
            merge (int): 当前合并的LoRA实例数量。
            unmerge (int): 当前拆分的LoRA实例数量。
        """
        super().step()
        if self.should_log():
            self.log({
                "kv_cache": kv_cache,
                "latency": latency,
                "merge": merge,
                "unmerge": unmerge,
            })
    