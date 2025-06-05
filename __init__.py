"""
YOLOv8 钢筋图纸分析系统

这个包提供了完整的YOLOv8训练和推理功能，专门用于钢筋图纸分析。

主要模块:
- config: 配置管理
- data_converter: 数据格式转换
- trainer: 模型训练
- inference: 模型推理
- utils: 工具函数
- run: 主运行脚本

使用示例:
    from yolov8.inference import YOLOv8Inference
    from yolov8.trainer import YOLOv8Trainer
    from yolov8.data_converter import DataConverter
"""

__version__ = "1.0.0"
__author__ = "YOLOv8 钢筋图纸分析系统"

# 导入主要类
from .config import ensure_directories
from .data_converter import DataConverter
from .trainer import YOLOv8Trainer
from .inference import YOLOv8Inference

# 初始化时确保目录结构存在
ensure_directories()

__all__ = [
    "DataConverter",
    "YOLOv8Trainer", 
    "YOLOv8Inference",
    "ensure_directories"
]
