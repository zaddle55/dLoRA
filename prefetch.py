#!/usr/bin/env python3
import os
import shutil
import logging
import time
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from datasets import load_dataset
import subprocess

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置环境变量优化下载 (针对集群环境)
os.environ["HF_HUB_DISABLE_IMPLICIT_FILE_LOCK"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

class DatasetInstance:
    def __init__(self, name, split, subset=None):
        self.name = name
        self.subset = subset
        self.split = split

    def __str__(self):
        if self.subset:
            return f"{self.name}-{self.subset}-{self.split}"
        else:
            return f"{self.name}-{self.split}"
        
    def __repr__(self):
        return self.__str__()

class ResourceDownloader:
    def __init__(self, hf_token, base_tmp_dir="/tmp"):
        self.hf_token = hf_token
        self.user = os.environ.get("USER", "user")
        
        # 临时下载目录 (高速本地盘)
        self.tmp_root = Path(base_tmp_dir) / f"{self.user}_hf_download_tmp"
        self.tmp_models = self.tmp_root / "models"
        self.tmp_datasets = self.tmp_root / "datasets"
        
        # 最终存储目录 (NFS/持久化存储)
        self.final_root = Path(os.path.expanduser("~/models_cache"))
        self.final_models = self.final_root / "models"
        self.final_datasets = self.final_root / "datasets"

        # 记录成功下载的项目
        self.success_items = []

    def _prepare_dirs(self):
        """清理并重建临时目录，确保干净"""
        if self.tmp_root.exists():
            logger.info(f"清理旧临时目录: {self.tmp_root}")
            shutil.rmtree(self.tmp_root, ignore_errors=True)
        
        self.tmp_models.mkdir(parents=True, exist_ok=True)
        self.tmp_datasets.mkdir(parents=True, exist_ok=True)
        
        # 确保最终目录存在
        self.final_models.mkdir(parents=True, exist_ok=True)
        self.final_datasets.mkdir(parents=True, exist_ok=True)

    def _get_safe_name(self, repo_id):
        """将 repo_id (a/b) 转换为文件名安全格式 (a--b)"""
        return repo_id.replace("/", "--")

    def download_single_model(self, model_name):
        """下载单个模型"""
        safe_name = self._get_safe_name(model_name)
        download_path = self.tmp_models / safe_name
        
        logger.info(f"⬇️  开始下载模型: {model_name} -> {download_path}")
        
        try:
            snapshot_download(
                repo_id=model_name,
                local_dir=download_path,
                local_dir_use_symlinks=False, # 必须为False，否则无法移动
                resume_download=True,
                # token=self.hf_token
            )
            
            # 简单验证
            logger.info(f"🔍 验证模型文件: {model_name}")
            AutoTokenizer.from_pretrained(str(download_path), trust_remote_code=True)
            
            logger.info(f"✅ 模型下载并验证成功: {model_name}")
            return {"type": "model", "name": model_name, "src": download_path, "dest_name": safe_name}
            
        except Exception as e:
            logger.error(f"❌ 模型 {model_name} 下载失败: {e}")
            return None

    def download_single_dataset(self, dataset_instance):
        """下载单个数据集"""
        dataset_name = str(dataset_instance)
        safe_name = self._get_safe_name(dataset_name)
        download_path = self.tmp_datasets / safe_name
        
        logger.info(f"⬇️  开始下载数据集: {dataset_name} -> {download_path}")
        
        try:
            # 下载并处理
            ds = load_dataset(dataset_instance.name, name=dataset_instance.subset, split=dataset_instance.split, trust_remote_code=True)
            # 保存到磁盘 (这会生成箭头文件等)
            ds.save_to_disk(str(download_path))
            
            logger.info(f"✅ 数据集下载并保存成功: {dataset_name} (条数: {len(ds)})")
            return {"type": "dataset", "name": dataset_name, "src": download_path, "dest_name": safe_name}
            
        except Exception as e:
            logger.error(f"❌ 数据集 {dataset_name} 下载失败: {e}")
            return None

    def move_artifacts(self):
        """将成功下载的项目移动到最终目录"""
        if not self.success_items:
            logger.warning("⚠️  没有成功下载的项目，跳过移动步骤。")
            return

        logger.info("="*50)
        logger.info("🚚 开始将文件从临时目录移动到最终存储...")
        logger.info("="*50)

        for item in self.success_items:
            src = item['src']
            
            if item['type'] == 'model':
                dest = self.final_models / item['dest_name']
            else:
                dest = self.final_datasets / item['dest_name']

            try:
                # 如果目标已存在，先删除旧的，确保版本最新
                if dest.exists():
                    logger.warning(f"  目标已存在，正在覆盖: {dest}")
                    shutil.rmtree(dest)
                
                logger.info(f"  Moving: {item['name']}...")
                shutil.move(str(src), str(dest))
                logger.info(f"  ✅ 已就位: {dest}")
                
            except Exception as e:
                logger.error(f"  ❌ 移动 {item['name']} 失败: {e}")

    def cleanup(self):
        """清理临时目录"""
        if self.tmp_root.exists():
            logger.info(f"🧹 清理临时目录: {self.tmp_root}")
            shutil.rmtree(self.tmp_root)

    def run(self, models=None, datasets=None):
        self._prepare_dirs()
        
        # 1. 下载模型
        if models:
            for model in models:
                res = self.download_single_model(model)
                if res:
                    self.success_items.append(res)
        
        # 2. 下载数据集
        if datasets:
            for ds in datasets:
                res = self.download_single_dataset(ds)
                if res:
                    self.success_items.append(res)
        
        # 3. 移动文件
        self.move_artifacts()
        
        # 4. 磁盘统计
        self.show_disk_usage()
        
        # 5. 清理
        self.cleanup()
        
        logger.info("\n🎉 所有任务处理完成!")

    def show_disk_usage(self):
        try:
            logger.info("\n📊 最终目录磁盘占用:")
            subprocess.run(['du', '-sh', str(self.final_root)])
        except:
            pass

# ==========================================
# 入口函数
# ==========================================
def run_downloads(
    model_list=None, 
    dataset_list=None, 
    hf_token=None
):
    """
    配置并启动下载任务
    :param model_list: list of strings, e.g. ["meta-llama/Llama-2-7b-hf"]
    :param dataset_list: list of strings, e.g. ["tatsu-lab/alpaca"]
    :param hf_token: HuggingFace User Access Token
    """

    downloader = ResourceDownloader(hf_token=hf_token)
    
    print(f"🚀 启动下载任务")
    print(f"📦 计划模型: {model_list}")
    print(f"📄 计划数据集: {dataset_list}")
    print("-" * 30)
    
    # 双重确认
    if input("确认开始下载? (y/n): ").lower() != 'y':
        print("已取消")
        return

    downloader.run(models=model_list, datasets=dataset_list)

# ==========================================
# 使用示例 (Main)
# ==========================================
if __name__ == "__main__":
    
    # 配置你想下载的内容
    TARGET_MODELS = [
        # "EleutherAI/gpt-j-6b",
        "NousResearch/Llama-2-7b-hf",
        "NousResearch/Llama-2-13b-hf",
    ]
    
    TARGET_DATASETS = [
        
    ]

    # 运行
    run_downloads(
        model_list=TARGET_MODELS,
        dataset_list=TARGET_DATASETS,
    )