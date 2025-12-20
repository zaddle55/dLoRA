"""
Central Configuration System with Hardware Awareness
"""
from pathlib import Path
from typing import Literal, Optional
from enum import Enum

import torch
from pydantic import BaseModel, Field, BaseSettings
from loguru import logger


class GPUArchitecture(str, Enum):
    """GPU Architecture Types"""
    AMPERE_PLUS = "ampere+"  # L40, A100, H100 (Arch >= 8.0)
    VOLTA = "volta"  # Titan V, V100 (Arch 7.x)
    UNKNOWN = "unknown"


class DistributedStrategy(str, Enum):
    """Distributed Training Strategy"""
    NONE = "none"
    DDP = "ddp"  # Data Parallel
    TP = "tp"    # Tensor Parallel
    FSDP = "fsdp"  # Fully Sharded Data Parallel


class TaskType(str, Enum):
    """Model Task Types"""
    CAUSAL_LM = "causal_lm"
    SEQ2SEQ = "seq2seq"
    CLASSIFICATION = "classification"


class HardwareConfig(BaseModel):
    """Hardware-aware configuration that auto-detects GPU capabilities"""
    
    device: str = Field(default="cuda:0", description="Primary device")
    gpu_architecture: GPUArchitecture = Field(default=GPUArchitecture.UNKNOWN)
    supports_bf16: bool = Field(default=False)
    supports_fp16: bool = Field(default=True)
    torch_dtype: str = Field(default="float16")
    num_gpus: int = Field(default=1, ge=0)
    gpu_memory_gb: float = Field(default=24.0, gt=0)
    distributed_strategy: DistributedStrategy = Field(default=DistributedStrategy.NONE)
    
    # Multi-GPU settings
    device_map: Optional[str] = Field(default=None, description="Device map for multi-GPU (auto/balanced/sequential)")
    
    # model_config = SettingsConfigDict(use_enum_values=False)
    
    @classmethod
    def auto_detect(cls, device: str = "cuda:0") -> "HardwareConfig":
        """Auto-detect GPU capabilities"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return cls(
                device="cpu",
                gpu_architecture=GPUArchitecture.UNKNOWN,
                supports_bf16=False,
                supports_fp16=False,
                torch_dtype="float32",
                num_gpus=0
            )
        
        device_idx = int(device.split(":")[-1]) if ":" in device else 0
        capability = torch.cuda.get_device_capability(device_idx)
        arch_version = capability[0] + capability[1] / 10.0
        
        # Determine architecture
        if arch_version >= 8.0:
            gpu_arch = GPUArchitecture.AMPERE_PLUS
            supports_bf16 = True
            torch_dtype = "bfloat16"
        elif 7.0 <= arch_version < 8.0:
            gpu_arch = GPUArchitecture.VOLTA
            supports_bf16 = False
            torch_dtype = "float16"
        else:
            gpu_arch = GPUArchitecture.UNKNOWN
            supports_bf16 = False
            torch_dtype = "float16"
        
        # Get GPU memory
        props = torch.cuda.get_device_properties(device_idx)
        gpu_memory_gb = props.total_memory / (1024 ** 3)
        
        # Determine distributed strategy
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            distributed_strategy = DistributedStrategy.DDP if arch_version >= 7.0 else DistributedStrategy.NONE
        else:
            distributed_strategy = DistributedStrategy.NONE
        
        config = cls(
            device=device,
            gpu_architecture=gpu_arch,
            supports_bf16=supports_bf16,
            supports_fp16=True,
            torch_dtype=torch_dtype,
            num_gpus=num_gpus,
            gpu_memory_gb=gpu_memory_gb,
            distributed_strategy=distributed_strategy,
            device_map="auto" if num_gpus > 1 else None
        )
        
        logger.info(f"🔍 Hardware Detection:")
        logger.info(f"  - GPU: {torch.cuda.get_device_name(device_idx)}")
        logger.info(f"  - Architecture: {gpu_arch.value} (Compute {arch_version})")
        logger.info(f"  - Memory: {gpu_memory_gb:.1f} GB")
        logger.info(f"  - BF16 Support: {supports_bf16}")
        logger.info(f"  - Dtype: {torch_dtype}")
        logger.info(f"  - Num GPUs: {num_gpus}")
        logger.info(f"  - Strategy: {distributed_strategy.value}")
        
        return config


class LoRAHyperparameters(BaseModel):
    """LoRA fine-tuning hyperparameters"""
    r: int = Field(default=16, ge=1, le=256, description="LoRA rank")
    alpha: int = Field(default=32, ge=1, description="LoRA alpha scaling")
    dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    )
    bias: str = Field(default="none", pattern="^(none|all|lora_only)$")
    
    # @field_validator('alpha')
    @classmethod
    def validate_alpha(cls, v, info):
        if 'r' in info.data and v < info.data['r']:
            logger.warning(f"Alpha ({v}) < rank ({info.data['r']}), this may reduce effectiveness")
        return v


class TaskConfig(BaseModel):
    """Task-specific configurations"""
    task_type: TaskType = Field(default=TaskType.CAUSAL_LM)
    lora: LoRAHyperparameters = Field(default_factory=LoRAHyperparameters)
    
    # Training parameters
    learning_rate: float = Field(default=2e-4, gt=0)
    batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    num_epochs: int = Field(default=1, ge=1)
    max_length: int = Field(default=1024, ge=32, le=8192)
    use_quantization: bool = Field(default=True)
    quantization_type: Literal["nf4", "fp4"] = Field(default="nf4")
    double_quant: bool = Field(default=True)
    
    # Optimization
    use_gradient_checkpointing: bool = Field(default=True)
    use_mixed_precision: bool = Field(default=True)


class BenchmarkConfig(BaseModel):
    """Benchmarking and stress testing configuration"""
    
    # Request distribution
    num_requests: int = Field(default=1000, ge=1)
    concurrent_users: int = Field(default=32, ge=1, le=256)
    
    # LoRA popularity distribution (Zipfian parameter)
    popularity_distribution: Literal["uniform", "zipf:1.5", "zipf:2.0", "zipf:2.5"] = "zipf:1.5"
    
    # LoRA variety
    num_lora_models: int = Field(default=8, ge=1, le=64)
    lora_rank_variety: list[int] = Field(default_factory=lambda: [8, 16, 32, 64])
    
    # Sequence lengths
    min_prompt_len: int = Field(default=16, ge=1)
    max_prompt_len: int = Field(default=512, ge=1)
    min_output_len: int = Field(default=32, ge=1)
    max_output_len: int = Field(default=256, ge=1)
    
    # Cache simulation
    max_active_adapters: int = Field(default=16, ge=1, description="Max LoRAs in GPU memory")
    
    # @field_validator('max_prompt_len')
    @classmethod
    def validate_prompt_len(cls, v, info):
        if 'min_prompt_len' in info.data and v < info.data['min_prompt_len']:
            raise ValueError("max_prompt_len must be >= min_prompt_len")
        return v


class GlobalConfig(BaseSettings):
    """Global application configuration with path management"""
    
    # Paths
    project_root: Path = Field(default_factory=lambda: Path.cwd())
    models_cache: Path = Field(default_factory=lambda: Path.home() / "models_cache" / "models")
    datasets_cache: Path = Field(default_factory=lambda: Path.home() / "models_cache" / "datasets")
    lora_output: Path = Field(default_factory=lambda: Path.cwd() / "lora_adapters")
    punica_weights: Path = Field(default_factory=lambda: Path.cwd() / "punica_weights")
    logs_dir: Path = Field(default_factory=lambda: Path.cwd() / "logs")
    benchmark_results: Path = Field(default_factory=lambda: Path.cwd() / "benchmark_results")
    
    # Punica conversion script path
    punica_converter_script: Path = Field(
        default_factory=lambda: Path("third_party/punica/src/punica/utils/convert_lora_weight.py")
    )
    
    # HuggingFace settings
    hf_token: Optional[str] = Field(default=None, description="HuggingFace API token")
    
    # Logging
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    
    # Component configs
    hardware: HardwareConfig = Field(default_factory=lambda: HardwareConfig.auto_detect())
    task: TaskConfig = Field(default_factory=TaskConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    sharegpt_path: Path = Field(default_factory=lambda: Path.cwd() / "sharegpt" / "ShareGPT_V3_unfiltered_cleaned_split.json")
    
    # model_config = SettingsConfigDict(
    #     env_prefix="LORA_",
    #     env_file=".env",
    #     env_file_encoding="utf-8",
    #     extra="ignore",
    #     env_nested_delimiter="__",
    # )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()
        self._setup_logging()
    
    def _create_directories(self):
        """Ensure all required directories exist"""
        for path_field in ['models_cache', 'datasets_cache', 'lora_output', 
                           'punica_weights', 'logs_dir', 'benchmark_results']:
            path = getattr(self, path_field)
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"📁 Ensured directory: {path}")
    
    def _setup_logging(self):
        """Configure loguru logger"""
        logger.remove()  # Remove default handler
        
        # Console handler
        logger.add(
            lambda msg: print(msg, end=""),
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            level=self.log_level,
            colorize=True
        )
        
        # File handler
        log_file = self.logs_dir / "multi_lora_{time}.log"
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            rotation="100 MB",
            retention="7 days",
            compression="zip"
        )
        
        logger.info(f"✅ Logging configured: {self.log_level}")
    
    def get_model_path(self, model_name: str) -> Path:
        """Get full path for a model"""
        safe_name = model_name.replace("/", "--")
        return self.models_cache / safe_name
    
    def get_dataset_path(self, dataset_name: str) -> Path:
        """Get full path for a dataset"""
        safe_name = dataset_name.replace("/", "--")
        return self.datasets_cache / safe_name


# Global configuration instance
_config_instance: Optional[GlobalConfig] = None


def get_config(reload: bool = False) -> GlobalConfig:
    """Get or create global configuration singleton"""
    global _config_instance
    if _config_instance is None or reload:
        _config_instance = GlobalConfig()
    return _config_instance


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    logger.info("Configuration loaded successfully")
    logger.info(f"Huggingface Token: {f'Set {config.hf_token}' if config.hf_token else 'Not Set'}")
    logger.info(f"Hardware: {config.hardware.gpu_architecture.value}")
    logger.info(f"Dtype: {config.hardware.torch_dtype}")
    logger.info(f"LoRA rank: {config.task.lora.r}")
    logger.info(f"Use Quantization: {config.task.use_quantization}")
    logger.info(f"Quantization Type: {config.task.quantization_type}")