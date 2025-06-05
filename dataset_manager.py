"""
数据集管理模块
处理数据集目录选择和验证
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from data_converter import DataConverter
from config import YOLO_POINT_DIR, DATASETS_DIR


class DatasetDirectoryManager:
    """数据集目录管理器"""
    
    def __init__(self):
        self.current_source_dir = YOLO_POINT_DIR
        self.config_file = Path("dataset_config.json")
        self.load_config()
    
    def load_config(self):
        """加载数据集配置"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    saved_dir = config.get('source_directory')
                    if saved_dir and Path(saved_dir).exists():
                        self.current_source_dir = Path(saved_dir)
            except Exception as e:
                print(f"加载数据集配置失败: {e}")
    
    def save_config(self):
        """保存数据集配置"""
        try:
            config = {
                'source_directory': str(self.current_source_dir),
                'last_updated': str(Path().cwd())
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存数据集配置失败: {e}")
    
    def set_source_directory(self, directory_path: str) -> Dict:
        """设置源目录"""
        try:
            new_dir = Path(directory_path)
            
            # 验证目录
            validation_result = self.validate_directory(new_dir)
            
            if validation_result["valid"]:
                self.current_source_dir = new_dir
                self.save_config()
                return {
                    "success": True,
                    "message": f"✅ 数据集目录已设置为: {new_dir}",
                    "directory": str(new_dir),
                    "validation": validation_result
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ 目录验证失败: {validation_result['message']}",
                    "directory": str(new_dir),
                    "validation": validation_result
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ 设置目录失败: {e}",
                "directory": directory_path
            }
    
    def validate_directory(self, directory: Path) -> Dict:
        """验证目录是否适合作为数据集源目录"""
        if not directory.exists():
            return {
                "valid": False,
                "message": "目录不存在",
                "details": {
                    "exists": False,
                    "json_files": 0,
                    "image_files": 0
                }
            }
        
        if not directory.is_dir():
            return {
                "valid": False,
                "message": "路径不是目录",
                "details": {
                    "is_directory": False
                }
            }
        
        # 检查JSON文件
        json_files = list(directory.glob("*.json"))
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        details = {
            "exists": True,
            "is_directory": True,
            "json_files": len(json_files),
            "image_files": len(image_files),
            "json_file_names": [f.name for f in json_files[:5]],  # 显示前5个
            "image_file_names": [f.name for f in image_files[:5]]  # 显示前5个
        }
        
        if len(json_files) == 0:
            return {
                "valid": False,
                "message": "目录中没有找到JSON标注文件",
                "details": details
            }
        
        # 检查是否有对应的图片文件
        missing_images = []
        for json_file in json_files[:10]:  # 检查前10个JSON文件
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    image_name = data.get('imagePath', '')
                    if image_name:
                        image_path = directory / image_name
                        if not image_path.exists():
                            # 尝试其他可能的路径
                            found = False
                            for ext in image_extensions:
                                alt_path = directory / f"{json_file.stem}{ext}"
                                if alt_path.exists():
                                    found = True
                                    break
                            if not found:
                                missing_images.append(image_name)
            except:
                continue
        
        details["missing_images"] = missing_images[:5]  # 显示前5个缺失的图片
        
        if len(missing_images) > len(json_files) * 0.5:  # 如果超过50%的图片缺失
            return {
                "valid": False,
                "message": f"大量图片文件缺失 ({len(missing_images)} 个)",
                "details": details
            }
        
        return {
            "valid": True,
            "message": f"目录验证通过，找到 {len(json_files)} 个JSON文件和 {len(image_files)} 个图片文件",
            "details": details
        }
    
    def get_current_directory_info(self) -> Dict:
        """获取当前目录信息"""
        converter = DataConverter(self.current_source_dir)
        status = converter.check_source_directory()
        
        return {
            "current_directory": str(self.current_source_dir),
            "is_default": self.current_source_dir == YOLO_POINT_DIR,
            "status": status,
            "validation": self.validate_directory(self.current_source_dir)
        }
    
    def get_directory_suggestions(self) -> List[str]:
        """获取目录建议"""
        suggestions = []
        
        # 默认目录
        if YOLO_POINT_DIR.exists():
            suggestions.append(str(YOLO_POINT_DIR))
        
        # 项目根目录下的常见目录
        project_root = Path.cwd()
        common_dirs = ['data', 'dataset', 'datasets', 'annotations', 'labels', 'labeling_data']
        
        for dir_name in common_dirs:
            dir_path = project_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                suggestions.append(str(dir_path))
        
        # 去重并排序
        suggestions = list(set(suggestions))
        suggestions.sort()
        
        return suggestions
    
    def convert_dataset(self) -> Dict:
        """转换当前目录的数据集"""
        try:
            converter = DataConverter(self.current_source_dir)
            result = converter.convert_all()
            
            if isinstance(result, dict) and result.get("status") != "ready":
                return {
                    "success": False,
                    "message": result.get("message", "转换失败"),
                    "details": result
                }
            
            return {
                "success": True,
                "message": "✅ 数据集转换完成",
                "source_directory": str(self.current_source_dir),
                "target_directory": str(DATASETS_DIR)
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ 数据集转换失败: {e}",
                "source_directory": str(self.current_source_dir)
            }
    
    def get_conversion_preview(self) -> Dict:
        """获取转换预览信息"""
        try:
            converter = DataConverter(self.current_source_dir)
            status = converter.check_source_directory()
            
            if status["status"] != "ready":
                return status
            
            # 获取文件统计
            json_files = list(self.current_source_dir.glob("*.json"))
            
            # 模拟数据集分割
            from config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO
            total_files = len(json_files)
            train_count = int(total_files * TRAIN_RATIO)
            val_count = int(total_files * VAL_RATIO)
            test_count = total_files - train_count - val_count
            
            return {
                "status": "ready",
                "total_files": total_files,
                "split_preview": {
                    "train": train_count,
                    "val": val_count,
                    "test": test_count
                },
                "source_directory": str(self.current_source_dir),
                "target_directory": str(DATASETS_DIR),
                "sample_files": [f.name for f in json_files[:5]]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"获取预览信息失败: {e}"
            }


# 全局实例
dataset_directory_manager = DatasetDirectoryManager()
