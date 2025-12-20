import os
import json
import time
from typing import Any, Dict, List, Optional, Union

class MetricPrototype:
    def __init__(self,
                 output_dir: str,
                 name: str="default", 
                 step_per_log: int = 1,
                 engine_args: Optional[Dict[str, Any]] = None
                 ):
        """
        数据收集原型接口。

        Args:
            output_dir (str): 数据输出的文件夹路径。
            step_per_log (int): 每隔多少步记录一次数据，默认为 1。
        """
        self.output_dir = output_dir
        self.step_per_log = step_per_log
        
        # 内置计数器
        self.nums_step = 0
        
        # 数据缓存区
        self.records: List[Dict[str, Any]] = []
        self.engine_args: Optional[Dict[str, Any]] = None
        # 初始化时自动确保目录存在
        self._ensure_dir()

    def _ensure_dir(self):
        """确保输出目录存在"""
        if self.output_dir and not os.path.exists(self.output_dir):
            try:
                os.makedirs(self.output_dir, exist_ok=True)
            except OSError as e:
                print(f"[MetricPrototype] Error creating directory {self.output_dir}: {e}")

    def step(self, *args, **kwargs):
        """
        【需要重写】记录步骤的核心方法。
        
        子类在重写此方法时，建议调用 super().step() 以保持 nums_step 的自动计数功能。
        
        Example:
            def step(self, loss):
                super().step()  # 计数器 +1
                if self.should_log():
                    self.log({"loss": loss})
        """
        self.nums_step += 1

    def should_log(self) -> bool:
        """
        判断当前 step 是否满足记录条件（基于 step_per_log）。
        通常在子类的 step() 方法中调用。
        """
        return self.nums_step % self.step_per_log == 0

    def log(self, data: Dict[str, Any]):
        """
        【便捷开发方法】将单条数据添加到缓存中。
        会自动附加 step 和 timestamp 信息。
        """
        record = {
            "step": self.nums_step,
            "timestamp": time.time(),
            **data  # 解包用户数据
        }
        self.records.append(record)

    def save(self, filename: str = "metrics.json"):
        """
        【便捷开发方法】将缓存的所有数据导出为 JSON 文件。
        
        Args:
            filename (str): 保存的文件名，默认为 metrics.json。
        """
        if not self.records:
            print("[MetricPrototype] No records to save.")
            return

        file_path = os.path.join(self.output_dir, filename)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.records, f, indent=4, ensure_ascii=False)
            print(f"[MetricPrototype] Successfully saved {len(self.records)} records to {file_path}")
        except Exception as e:
            print(f"[MetricPrototype] Failed to save metrics: {e}")

    def reset(self):
        """【便捷开发方法】重置计数器和数据缓存"""
        self.nums_step = 0
        self.records = []
        print("[MetricPrototype] Metrics reset.")