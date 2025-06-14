"""
YOLOv8 配置文件
包含训练、推理和数据处理的相关配置
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
YOLO_ROOT = Path(__file__).parent

# 数据路径配置
YOLO_POINT_DIR = PROJECT_ROOT / "labeling_data"  # JSON标注文件目录
DATASETS_DIR = YOLO_ROOT / "datasets"  # YOLO格式数据集目录
IMAGES_DIR = DATASETS_DIR / "images"  # 图片目录
LABELS_DIR = DATASETS_DIR / "labels"  # 标签目录
MODELS_DIR = YOLO_ROOT / "models"  # 模型保存目录

# 训练/验证/测试数据分割比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# 类别配置（已废弃 - 现在使用智能类别管理器）
# 类别信息现在完全基于标注数据自动检测和管理
# 请使用 class_manager 模块获取最新的类别信息

# 训练配置
TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 16,
    "img_size": 640,
    "learning_rate": 0.01,
    "patience": 50,  # 早停耐心值
    "save_period": 10,  # 每10个epoch保存一次模型
    "workers": 2,  # 数据加载器工作进程数
    "device": None,  # 设备将由设备管理器自动选择最佳可用设备
}

# 推理配置
INFERENCE_CONFIG = {
    "conf_threshold": 0.25,  # 置信度阈值
    "iou_threshold": 0.45,   # NMS IoU阈值
    "max_det": 1000,         # 最大检测数量
    "img_size": 640,         # 推理图片尺寸
}

# 数据增强配置
AUGMENTATION_CONFIG = {
    "hsv_h": 0.015,      # 色调增强
    "hsv_s": 0.7,        # 饱和度增强
    "hsv_v": 0.4,        # 明度增强
    "degrees": 0.0,      # 旋转角度
    "translate": 0.1,    # 平移
    "scale": 0.5,        # 缩放
    "shear": 0.0,        # 剪切
    "perspective": 0.0,  # 透视变换
    "flipud": 0.0,       # 上下翻转
    "fliplr": 0.5,       # 左右翻转
    "mosaic": 1.0,       # 马赛克增强
    "mixup": 0.0,        # 混合增强
}

# 模型配置
MODEL_CONFIG = {
    "model_name": "yolov8n.pt",  # 预训练模型名称 (n/s/m/l/x)
    "pretrained": True,          # 是否使用预训练权重
}

# 日志配置
LOG_CONFIG = {
    "log_dir": YOLO_ROOT / "logs",
    "tensorboard": True,
    "wandb": False,  # 是否使用wandb记录
}

# 确保目录存在
def ensure_directories():
    """确保所有必要的目录存在"""
    directories = [
        DATASETS_DIR,
        IMAGES_DIR / "train",
        IMAGES_DIR / "val",
        IMAGES_DIR / "test",
        LABELS_DIR / "train",
        LABELS_DIR / "val",
        LABELS_DIR / "test",
        MODELS_DIR,
        LOG_CONFIG["log_dir"]
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_default_device():
    """获取默认训练设备"""
    # 延迟导入避免循环依赖
    try:
        from device_manager import device_manager
        return device_manager.get_best_available_device()
    except ImportError:
        # 如果设备管理器不可用，返回CPU作为安全选择
        return "cpu"

if __name__ == "__main__":
    ensure_directories()
    print("目录结构创建完成")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"YOLO根目录: {YOLO_ROOT}")
    print(f"数据集目录: {DATASETS_DIR}")
    print(f"模型目录: {MODELS_DIR}")
