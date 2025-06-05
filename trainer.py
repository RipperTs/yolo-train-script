"""
YOLOv8 训练模块
提供模型训练功能
"""

import os
import sys
from pathlib import Path
import yaml
from datetime import datetime
import torch

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from config import (
    TRAINING_CONFIG, MODEL_CONFIG, DATASETS_DIR, MODELS_DIR,
    LOG_CONFIG, AUGMENTATION_CONFIG, ensure_directories
)
from training_logger import training_log_manager


class YOLOv8Trainer:
    """YOLOv8训练器类"""
    
    def __init__(self):
        ensure_directories()
        self.model = None
        self.dataset_yaml = DATASETS_DIR.parent / "dataset.yaml"
        
        # 检查数据集配置文件是否存在
        if not self.dataset_yaml.exists():
            raise FileNotFoundError(
                f"数据集配置文件不存在: {self.dataset_yaml}\n"
                "请先运行 data_converter.py 转换数据"
            )
    
    def setup_model(self):
        """设置模型"""
        try:
            from ultralytics import YOLO
            
            model_name = MODEL_CONFIG["model_name"]
            pretrained = MODEL_CONFIG["pretrained"]
            
            if pretrained:
                # 使用预训练模型
                self.model = YOLO(model_name)
                print(f"加载预训练模型: {model_name}")
            else:
                # 从头开始训练
                # 需要配置文件，这里使用默认的YOLOv8n配置
                self.model = YOLO("yolov8n.yaml")
                print("使用随机权重初始化模型")
                
        except ImportError:
            raise ImportError(
                "未安装ultralytics库，请运行: pip install ultralytics"
            )
    
    def train(self, resume: bool = False, resume_path: str = None):
        """
        开始训练
        
        Args:
            resume: 是否恢复训练
            resume_path: 恢复训练的模型路径
        """
        if self.model is None:
            self.setup_model()
        
        # 训练参数
        train_args = {
            "data": str(self.dataset_yaml),
            "epochs": TRAINING_CONFIG["epochs"],
            "batch": TRAINING_CONFIG["batch_size"],
            "imgsz": TRAINING_CONFIG["img_size"],
            "lr0": TRAINING_CONFIG["learning_rate"],
            "patience": TRAINING_CONFIG["patience"],
            "save_period": TRAINING_CONFIG["save_period"],
            "workers": TRAINING_CONFIG["workers"],
            "device": "cpu",  # 强制使用CPU
            "project": str(MODELS_DIR),
            "name": f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "exist_ok": True,
            "verbose": True,
        }
        
        # 添加数据增强参数
        train_args.update(AUGMENTATION_CONFIG)
        
        # 如果需要恢复训练
        if resume:
            if resume_path and Path(resume_path).exists():
                train_args["resume"] = resume_path
                print(f"从 {resume_path} 恢复训练")
            else:
                train_args["resume"] = True
                print("自动查找最新的检查点恢复训练")
        
        print("开始训练...")
        print(f"训练参数: {train_args}")

        try:
            # 启动训练日志捕获
            log_file = training_log_manager.start_training_logging()
            print(f"📁 训练日志将保存到: {log_file}")

            # 开始训练
            results = self.model.train(**train_args)

            print("训练完成!")
            print(f"最佳模型保存在: {results.save_dir}")

            return results

        except Exception as e:
            error_msg = f"训练过程中出错: {e}"
            print(error_msg)
            raise
        finally:
            # 停止日志捕获
            training_log_manager.stop_training_logging()
    
    def validate(self, model_path: str = None):
        """
        验证模型
        
        Args:
            model_path: 模型路径，如果为None则使用当前模型
        """
        if model_path:
            from ultralytics import YOLO
            model = YOLO(model_path)
        else:
            if self.model is None:
                raise ValueError("没有可用的模型进行验证")
            model = self.model
        
        print("开始验证...")
        
        # 验证参数
        val_args = {
            "data": str(self.dataset_yaml),
            "imgsz": TRAINING_CONFIG["img_size"],
            "batch": TRAINING_CONFIG["batch_size"],
            "device": "cpu",  # 强制使用CPU
            "verbose": True,
        }
        
        try:
            results = model.val(**val_args)
            print("验证完成!")
            
            # 打印主要指标
            if hasattr(results, 'box'):
                metrics = results.box
                print(f"mAP50: {metrics.map50:.4f}")
                print(f"mAP50-95: {metrics.map:.4f}")
                print(f"Precision: {metrics.mp:.4f}")
                print(f"Recall: {metrics.mr:.4f}")
            
            return results
            
        except Exception as e:
            print(f"验证过程中出错: {e}")
            raise
    
    def export_model(self, model_path: str, format: str = "onnx"):
        """
        导出模型为其他格式
        
        Args:
            model_path: 模型路径
            format: 导出格式 (onnx, torchscript, tflite等)
        """
        try:
            from ultralytics import YOLO
            
            model = YOLO(model_path)
            print(f"导出模型为 {format} 格式...")
            
            export_path = model.export(format=format)
            print(f"模型已导出到: {export_path}")
            
            return export_path
            
        except Exception as e:
            print(f"导出模型时出错: {e}")
            raise
    
    def get_device_info(self):
        """获取设备信息"""
        print("设备信息:")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA设备数量: {torch.cuda.device_count()}")
            print(f"当前CUDA设备: {torch.cuda.current_device()}")
            print(f"设备名称: {torch.cuda.get_device_name()}")
        print(f"CPU核心数: {os.cpu_count()}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv8训练脚本")
    parser.add_argument("--action", choices=["train", "val", "export"], 
                       default="train", help="执行的操作")
    parser.add_argument("--model", type=str, help="模型路径")
    parser.add_argument("--resume", action="store_true", help="恢复训练")
    parser.add_argument("--format", type=str, default="onnx", 
                       help="导出格式")
    
    args = parser.parse_args()
    
    trainer = YOLOv8Trainer()
    trainer.get_device_info()
    
    if args.action == "train":
        trainer.train(resume=args.resume, resume_path=args.model)
    elif args.action == "val":
        trainer.validate(model_path=args.model)
    elif args.action == "export":
        if not args.model:
            print("导出模型需要指定 --model 参数")
            return
        trainer.export_model(args.model, format=args.format)


if __name__ == "__main__":
    main()
