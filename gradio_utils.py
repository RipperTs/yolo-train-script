"""
Gradio应用的工具函数
提供日志监控、状态管理等功能
"""

import os
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import io
import base64
from PIL import Image
import json

from config import MODELS_DIR, LOG_CONFIG, DATASETS_DIR


class LogMonitor:
    """日志监控器"""
    
    def __init__(self):
        self.log_queue = queue.Queue()
        self.is_monitoring = False
        self.monitor_thread = None
        self.current_log_file = None
        
    def start_monitoring(self, log_file_path: str = None):
        """开始监控日志文件"""
        if self.is_monitoring:
            self.stop_monitoring()
            
        if log_file_path is None:
            # 查找最新的日志文件
            log_file_path = self.find_latest_log()
            
        if log_file_path and Path(log_file_path).exists():
            self.current_log_file = log_file_path
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_log_file)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            return True
        return False
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def find_latest_log(self) -> Optional[str]:
        """查找最新的日志文件"""
        # 查找可能的日志位置
        possible_dirs = [
            LOG_CONFIG["log_dir"],  # 配置的日志目录
            Path("logs"),           # 项目logs目录
            Path("runs"),           # YOLO runs目录
            Path("models"),         # 模型目录
            Path("."),              # 当前目录
        ]

        log_files = []
        for log_dir in possible_dirs:
            if log_dir.exists():
                # 查找各种日志文件
                patterns = ["*.log", "**/*.log", "**/train.log", "**/results.csv"]
                for pattern in patterns:
                    log_files.extend(log_dir.glob(pattern))

        if not log_files:
            return None

        # 返回最新修改的日志文件
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        return str(latest_log)
    
    def _monitor_log_file(self):
        """监控日志文件的变化"""
        try:
            with open(self.current_log_file, 'r', encoding='utf-8') as f:
                # 移动到文件末尾
                f.seek(0, 2)
                
                while self.is_monitoring:
                    line = f.readline()
                    if line:
                        self.log_queue.put(line.strip())
                    else:
                        time.sleep(0.1)
        except Exception as e:
            self.log_queue.put(f"日志监控错误: {e}")
    
    def get_new_logs(self) -> List[str]:
        """获取新的日志行"""
        logs = []
        while not self.log_queue.empty():
            try:
                logs.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        return logs
    
    def get_recent_logs(self, num_lines: int = 100) -> List[str]:
        """获取最近的日志内容"""
        if not self.current_log_file or not Path(self.current_log_file).exists():
            return ["没有找到日志文件"]

        try:
            with open(self.current_log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                recent_lines = lines[-num_lines:] if len(lines) > num_lines else lines
                return [line.strip() for line in recent_lines if line.strip()]
        except Exception as e:
            return [f"读取日志文件错误: {e}"]

    def get_recent_logs_as_string(self, num_lines: int = 100) -> str:
        """获取最近的日志内容作为字符串"""
        logs = self.get_recent_logs(num_lines)
        return '\n'.join(logs)


class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self):
        self.current_training_dir = None
        
    def find_latest_training(self) -> Optional[Path]:
        """查找最新的训练目录"""
        train_dirs = list(MODELS_DIR.glob("train*"))
        if not train_dirs:
            return None
        return max(train_dirs, key=lambda x: x.stat().st_mtime)
    
    def get_training_status(self) -> Dict:
        """获取训练状态"""
        latest_dir = self.find_latest_training()
        if not latest_dir:
            return {"status": "no_training", "message": "没有找到训练记录"}
        
        results_file = latest_dir / "results.csv"
        if not results_file.exists():
            return {"status": "no_results", "message": "训练结果文件不存在"}
        
        try:
            df = pd.read_csv(results_file)
            if len(df) == 0:
                return {"status": "empty_results", "message": "训练结果为空"}
            
            latest = df.iloc[-1]
            return {
                "status": "active",
                "epochs": len(df),
                "latest_metrics": {
                    "box_loss": latest.get('train/box_loss', 0),
                    "cls_loss": latest.get('train/cls_loss', 0),
                    "dfl_loss": latest.get('train/dfl_loss', 0),
                    "map50": latest.get('metrics/mAP50(B)', 0),
                    "map50_95": latest.get('metrics/mAP50-95(B)', 0)
                },
                "training_dir": str(latest_dir)
            }
        except Exception as e:
            return {"status": "error", "message": f"读取训练结果错误: {e}"}
    
    def generate_training_plot(self) -> Optional[str]:
        """生成训练曲线图"""
        latest_dir = self.find_latest_training()
        if not latest_dir:
            return None
            
        results_file = latest_dir / "results.csv"
        if not results_file.exists():
            return None
        
        try:
            df = pd.read_csv(results_file)
            if len(df) == 0:
                return None
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            epochs = df.index + 1
            
            # Box Loss
            if 'train/box_loss' in df.columns:
                ax1.plot(epochs, df['train/box_loss'], 'b-', linewidth=2)
                ax1.set_title('Box Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.grid(True, alpha=0.3)
            
            # Class Loss
            if 'train/cls_loss' in df.columns:
                ax2.plot(epochs, df['train/cls_loss'], 'r-', linewidth=2)
                ax2.set_title('Class Loss')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.grid(True, alpha=0.3)
            
            # DFL Loss
            if 'train/dfl_loss' in df.columns:
                ax3.plot(epochs, df['train/dfl_loss'], 'g-', linewidth=2)
                ax3.set_title('DFL Loss')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Loss')
                ax3.grid(True, alpha=0.3)
            
            # mAP50
            if 'metrics/mAP50(B)' in df.columns:
                ax4.plot(epochs, df['metrics/mAP50(B)'], 'purple', linewidth=2)
                ax4.set_title('mAP50')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('mAP50')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存为base64字符串
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            plt.close()
            
            # 转换为base64
            plot_base64 = base64.b64encode(plot_data).decode()
            return f"data:image/png;base64,{plot_base64}"
            
        except Exception as e:
            print(f"生成训练曲线错误: {e}")
            return None


class DatasetManager:
    """数据集管理器"""
    
    def get_dataset_info(self) -> Dict:
        """获取数据集信息"""
        info = {
            "train": {"images": 0, "labels": 0},
            "val": {"images": 0, "labels": 0},
            "test": {"images": 0, "labels": 0}
        }
        
        for split in ["train", "val", "test"]:
            images_dir = DATASETS_DIR / "images" / split
            labels_dir = DATASETS_DIR / "labels" / split
            
            if images_dir.exists():
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    image_files.extend(images_dir.glob(ext))
                info[split]["images"] = len(image_files)
            
            if labels_dir.exists():
                label_files = list(labels_dir.glob("*.txt"))
                info[split]["labels"] = len(label_files)
        
        return info
    
    def get_sample_images(self, split: str = "train", num_samples: int = 5) -> List[str]:
        """获取样本图片路径"""
        images_dir = DATASETS_DIR / "images" / split
        if not images_dir.exists():
            return []
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(images_dir.glob(ext))
        
        if len(image_files) > num_samples:
            import random
            image_files = random.sample(image_files, num_samples)
        
        return [str(f) for f in image_files]


class ModelManager:
    """模型管理器"""
    
    def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        model_files = []
        for pattern in ["**/*.pt", "**/*best.pt", "**/best.pt"]:
            model_files.extend(MODELS_DIR.glob(pattern))

        # 返回相对于项目根目录的路径
        project_root = Path(__file__).parent  # 项目根目录
        relative_paths = []

        for model_file in model_files:
            try:
                # 计算相对路径
                relative_path = model_file.relative_to(project_root)
                relative_paths.append(str(relative_path))
            except ValueError:
                # 如果无法计算相对路径，使用绝对路径
                relative_paths.append(str(model_file))

        return relative_paths

    def get_absolute_path(self, relative_path: str) -> str:
        """将相对路径转换为绝对路径"""
        if Path(relative_path).is_absolute():
            # 如果已经是绝对路径，直接返回
            return relative_path

        # 相对于项目根目录的路径
        project_root = Path(__file__).parent
        absolute_path = project_root / relative_path

        return str(absolute_path)

    def get_model_info(self, model_path: str) -> Dict:
        """获取模型信息"""
        model_path = Path(model_path)
        if not model_path.exists():
            return {"error": "模型文件不存在"}
        
        try:
            stat = model_path.stat()
            return {
                "name": model_path.name,
                "size": f"{stat.st_size / (1024*1024):.2f} MB",
                "modified": time.ctime(stat.st_mtime),
                "path": str(model_path)
            }
        except Exception as e:
            return {"error": f"获取模型信息失败: {e}"}


# 全局实例
log_monitor = LogMonitor()
training_monitor = TrainingMonitor()
dataset_manager = DatasetManager()
model_manager = ModelManager()
