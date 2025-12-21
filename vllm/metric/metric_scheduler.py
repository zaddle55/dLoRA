from typing import Dict, Any, List
from vllm.config import MetricOutput, SchedulerConfig
from vllm.metric.metric_base import MetricPrototype

class SchedulerMetric(MetricPrototype):
    def __init__(self, scheduler_config: SchedulerConfig, engine_id: int, step_per_log: int = 1):
        """
        调度器指标收集器，专注于队列状态和 LoRA 负载分布。
        """
        super().__init__(output_dir=MetricOutput.SCHEDULER, name="scheduler_metric", step_per_log=step_per_log)
        self.engine_id = engine_id
        self.scheduler_config = scheduler_config

    def step(self, 
             waiting_len: int,
             running_len: int,
             swapped_len: int,
             free_gpu_blocks: int,
             active_loras: List[int],
             waiting_lora_dist: Dict[str, int],
             running_lora_dist: Dict[str, int],
             lora_credit: Dict[int, float],
             merge_right_thresh: float,
             merge_left_thresh: float,
             tput_merge: float,
             tput_unmerge: float,
             num_iters: int,
             **kwargs):
        """
        Args:
            waiting_len: 等待队列长度
            running_len: 运行队列长度
            swapped_len: 交换(CPU)队列长度
            free_gpu_blocks: 当前空闲的 GPU 显存块数
            active_loras: 当前 GPU 上已加载的 LoRA ID 列表
            waiting_lora_dist: 等待队列中各 LoRA 的请求数分布 {model_id: count}
            running_lora_dist: 运行队列中各 LoRA 的请求数分布 {model_id: count}
        """
        super().step()
        if self.should_log():
            self.log({
                "waiting_len": waiting_len,
                "running_len": running_len,
                "swapped_len": swapped_len,
                "free_gpu_blocks": free_gpu_blocks,
                "active_loras": active_loras,
                "waiting_lora_dist": waiting_lora_dist,
                "running_lora_dist": running_lora_dist,
                "lora_credit": lora_credit,
                "merge_right_thresh": merge_right_thresh,
                "merge_left_thresh": merge_left_thresh,
                "tput_merge": tput_merge,
                "tput_unmerge": tput_unmerge,
                "num_iters": num_iters,
                **kwargs
            })

    def summarize(self) -> Dict[str, Any]:
        """
        返回调度器指标的总结信息。
        """
        return {
            "engine_id": self.engine_id,
            "scheduler_config": vars(self.scheduler_config),
        }
    
    @staticmethod
    def combine(metrics: List['SchedulerMetric']) -> "SchedulerMetric":
        """
        合并多个调度器指标的总结信息。

        Args:
            metrics: 调度器指标列表。

        Returns:
            SchedulerMetric: 包含所有调度器指标总结信息的合并结果。
        """
        combined_metric = SchedulerMetric(
            engine_id=metrics[0].engine_id,
            scheduler_config=metrics[0].scheduler_config,
        )
        for metric in metrics:
            combined_metric.records.append({
                "engine_id": metric.engine_id,
                "scheduler_config": vars(metric.scheduler_config),
                "records": metric.records,
            })
        return combined_metric