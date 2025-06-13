"""
配置管理模块
提供动态配置管理功能
"""

import json
import copy
from pathlib import Path
from typing import Dict, Any, Optional

from config import (
    TRAINING_CONFIG, INFERENCE_CONFIG, AUGMENTATION_CONFIG,
    MODEL_CONFIG, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)
from device_manager import device_manager


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.config_file = Path("gradio_config.json")
        self.current_config = self._load_default_config()
        self.load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
            "training": copy.deepcopy(TRAINING_CONFIG),
            "inference": copy.deepcopy(INFERENCE_CONFIG),
            "augmentation": copy.deepcopy(AUGMENTATION_CONFIG),
            "model": copy.deepcopy(MODEL_CONFIG),
            "dataset": {
                "class_names": [],  # 类别名称现在由ClassManager管理
                "train_ratio": TRAIN_RATIO,
                "val_ratio": VAL_RATIO,
                "test_ratio": TEST_RATIO
            },
            "smart_training": {
                "target_box_loss": 0.05,
                "target_cls_loss": 1.0,
                "target_dfl_loss": 0.8,
                "target_map50": 0.7,
                "patience": 20,
                "min_improvement": 0.001,
                "max_total_epochs": 500,
                "continue_epochs": 50
            }
        }
    
    def load_config(self):
        """从文件加载配置"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                
                # 合并配置，保留默认值
                self._merge_config(self.current_config, saved_config)
            except Exception as e:
                print(f"加载配置文件失败: {e}")
    
    def save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def _merge_config(self, target: Dict, source: Dict):
        """递归合并配置"""
        for key, value in source.items():
            if key in target:
                if isinstance(value, dict) and isinstance(target[key], dict):
                    self._merge_config(target[key], value)
                else:
                    target[key] = value
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.current_config["training"]
    
    def update_training_config(self, **kwargs):
        """更新训练配置"""
        for key, value in kwargs.items():
            if key in self.current_config["training"]:
                self.current_config["training"][key] = value

                # 如果更新了设备，同时更新设备管理器
                if key == "device":
                    device_manager.set_device(value)

                    # 根据设备自动调整其他参数
                    recommended = device_manager.get_optimal_batch_size(value)
                    if "batch_size" not in kwargs:  # 如果没有手动设置批次大小
                        self.current_config["training"]["batch_size"] = recommended

        self.save_config()
    
    def get_inference_config(self) -> Dict[str, Any]:
        """获取推理配置"""
        return self.current_config["inference"]
    
    def update_inference_config(self, **kwargs):
        """更新推理配置"""
        for key, value in kwargs.items():
            if key in self.current_config["inference"] or key == "device":
                self.current_config["inference"][key] = value
                
                # 如果更新了设备，同时更新设备管理器
                if key == "device":
                    device_manager.set_device(value)
                    
        self.save_config()
    
    def get_augmentation_config(self) -> Dict[str, Any]:
        """获取数据增强配置"""
        return self.current_config["augmentation"]
    
    def update_augmentation_config(self, **kwargs):
        """更新数据增强配置"""
        for key, value in kwargs.items():
            if key in self.current_config["augmentation"]:
                self.current_config["augmentation"][key] = value
        self.save_config()
    
    def get_smart_training_config(self) -> Dict[str, Any]:
        """获取智能训练配置"""
        return self.current_config["smart_training"]
    
    def update_smart_training_config(self, **kwargs):
        """更新智能训练配置"""
        for key, value in kwargs.items():
            if key in self.current_config["smart_training"]:
                self.current_config["smart_training"][key] = value
        self.save_config()
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.current_config["model"]
    
    def update_model_config(self, **kwargs):
        """更新模型配置"""
        for key, value in kwargs.items():
            if key in self.current_config["model"]:
                self.current_config["model"][key] = value
        self.save_config()
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """获取数据集配置"""
        return self.current_config["dataset"]

    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        return {
            "current_device": device_manager.current_device,
            "available_devices": device_manager.get_device_choices(),
            "device_descriptions": device_manager.get_device_descriptions(),
            "gpu_available": device_manager.is_gpu_available(),
            "device_status": device_manager.get_device_status()
        }

    def update_device(self, device_id: str) -> bool:
        """更新设备设置"""
        if device_manager.set_device(device_id):
            self.current_config["training"]["device"] = device_id

            # 自动调整相关参数
            recommended_batch = device_manager.get_optimal_batch_size(device_id)
            self.current_config["training"]["batch_size"] = recommended_batch

            # 根据设备类型调整workers数量
            if device_id == "cpu":
                self.current_config["training"]["workers"] = 2
            else:
                self.current_config["training"]["workers"] = 4

            self.save_config()
            return True
        return False
    
    def optimize_config_for_device(self, device_id: str):
        """根据设备类型优化配置"""
        recommendations = self.get_device_recommendations(device_id)
        
        # 更新训练配置
        training_updates = {}
        if "batch_size" in recommendations:
            training_updates["batch_size"] = recommendations["batch_size"]
        if "workers" in recommendations:
            training_updates["workers"] = recommendations["workers"]
        
        # 设备特定优化
        if device_id == "mps":
            # MPS设备优化
            training_updates.update({
                "amp": False,  # 禁用自动混合精度
                "cache": False,  # 禁用数据集缓存以节省内存
            })
        elif device_id.startswith("cuda"):
            # CUDA设备优化
            training_updates.update({
                "amp": True,   # 启用自动混合精度
                "cache": True,  # 启用数据集缓存
            })
        else:  # CPU
            # CPU设备优化
            training_updates.update({
                "amp": False,  # CPU不支持混合精度
                "cache": False,  # CPU内存有限，禁用缓存
            })
        
        # 应用优化配置
        if training_updates:
            self.update_training_config(**training_updates)
            print(f"✅ 已为设备 {device_id} 优化配置: {training_updates}")
        
        return training_updates

    def get_device_recommendations(self, device_id: str) -> Dict[str, Any]:
        """获取设备推荐配置"""
        from device_manager import get_recommended_settings
        recommendations = get_recommended_settings(device_id)
        
        # 添加MPS特定的优化建议
        if device_id == "mps":
            recommendations.update({
                "mixed_precision": False,  # MPS可能不完全支持混合精度
                "pin_memory": False,       # MPS使用共享内存
                "persistent_workers": True, # 提高数据加载效率
                "prefetch_factor": 2,      # 预取因子
            })
        
        return recommendations
    
    def reset_to_default(self):
        """重置为默认配置"""
        self.current_config = self._load_default_config()
        self.save_config()
    
    def export_config(self, file_path: str):
        """导出配置到指定文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.current_config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"导出配置失败: {e}")
            return False
    
    def import_config(self, file_path: str):
        """从指定文件导入配置"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)
            
            # 验证配置格式
            if self._validate_config(imported_config):
                self.current_config = imported_config
                self.save_config()
                return True
            else:
                return False
        except Exception as e:
            print(f"导入配置失败: {e}")
            return False
    
    def _validate_config(self, config: Dict) -> bool:
        """验证配置格式"""
        required_sections = ["training", "inference", "augmentation", "model", "dataset", "smart_training"]
        return all(section in config for section in required_sections)
    
    def get_config_summary(self) -> str:
        """获取配置摘要"""
        training = self.current_config["training"]
        inference = self.current_config["inference"]
        smart = self.current_config["smart_training"]
        
        summary = f"""
配置摘要:
========

训练配置:
- 训练轮数: {training['epochs']}
- 批次大小: {training['batch_size']}
- 学习率: {training['learning_rate']}
- 图片尺寸: {training['img_size']}
- 设备: {training['device']}

推理配置:
- 置信度阈值: {inference['conf_threshold']}
- IoU阈值: {inference['iou_threshold']}
- 最大检测数: {inference['max_det']}

智能训练配置:
- 目标mAP50: {smart['target_map50']}
- 目标Box Loss: {smart['target_box_loss']}
- 最大训练轮数: {smart['max_total_epochs']}
- 耐心值: {smart['patience']}
"""
        return summary


# 全局配置管理器实例
config_manager = ConfigManager()
