"""
数据转换模块
将yolo_point目录中的JSON标注文件转换为YOLO训练格式
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import random
from PIL import Image
import numpy as np

from config import (
    YOLO_POINT_DIR, DATASETS_DIR, IMAGES_DIR, LABELS_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, CLASS_NAMES, ensure_directories
)


class DataConverter:
    """数据转换器类"""
    
    def __init__(self):
        self.class_to_id = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        ensure_directories()
    
    def convert_json_to_yolo(self, json_file_path: Path) -> List[str]:
        """
        将单个JSON文件转换为YOLO格式标注
        
        Args:
            json_file_path: JSON文件路径
            
        Returns:
            YOLO格式的标注行列表
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            image_width = data.get('imageWidth', 1)
            image_height = data.get('imageHeight', 1)
            shapes = data.get('shapes', [])
            
            yolo_lines = []
            
            for shape in shapes:
                shape_type = shape.get('shape_type', '')
                label = shape.get('label', '')

                if label not in self.class_to_id:
                    continue  # 跳过未知类别

                class_id = self.class_to_id[label]
                points = shape.get('points', [])

                if shape_type == 'rectangle':
                    # 处理矩形标注
                    if len(points) != 4:
                        continue  # 矩形应该有4个点

                    # 提取边界框坐标
                    x_coords = [point[0] for point in points]
                    y_coords = [point[1] for point in points]

                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                elif shape_type == 'point':
                    # 处理点标注，转换为小的边界框
                    if len(points) != 1:
                        continue  # 点标注应该只有1个点

                    point_x, point_y = points[0]

                    # 创建一个小的边界框（例如5x5像素）
                    box_size = 5
                    x_min = max(0, point_x - box_size/2)
                    x_max = min(image_width, point_x + box_size/2)
                    y_min = max(0, point_y - box_size/2)
                    y_max = min(image_height, point_y + box_size/2)

                else:
                    continue  # 跳过其他类型的标注
                
                # 转换为YOLO格式 (归一化的中心点坐标和宽高)
                x_center = (x_min + x_max) / 2.0 / image_width
                y_center = (y_min + y_max) / 2.0 / image_height
                width = (x_max - x_min) / image_width
                height = (y_max - y_min) / image_height
                
                # 确保坐标在[0,1]范围内
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                yolo_lines.append(yolo_line)
            
            return yolo_lines
            
        except Exception as e:
            print(f"转换文件 {json_file_path} 时出错: {e}")
            return []
    
    def get_image_path(self, json_file_path: Path) -> Path:
        """
        根据JSON文件路径获取对应的图片路径
        
        Args:
            json_file_path: JSON文件路径
            
        Returns:
            图片文件路径
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            image_name = data.get('imagePath', '')
            if image_name:
                # 假设图片在yolo_point目录的同级目录或子目录中
                possible_paths = [
                    YOLO_POINT_DIR / image_name,
                    YOLO_POINT_DIR.parent / image_name,
                    YOLO_POINT_DIR.parent / "images" / image_name,
                ]
                
                for path in possible_paths:
                    if path.exists():
                        return path
            
            # 如果找不到，尝试根据JSON文件名推断图片名
            base_name = json_file_path.stem
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_path = YOLO_POINT_DIR.parent / f"{base_name}{ext}"
                if image_path.exists():
                    return image_path
                    
        except Exception as e:
            print(f"获取图片路径时出错: {e}")
        
        return None
    
    def split_dataset(self, file_list: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        将数据集分割为训练集、验证集和测试集
        
        Args:
            file_list: 文件路径列表
            
        Returns:
            (训练集, 验证集, 测试集) 文件路径列表
        """
        random.shuffle(file_list)
        
        total_files = len(file_list)
        train_end = int(total_files * TRAIN_RATIO)
        val_end = int(total_files * (TRAIN_RATIO + VAL_RATIO))
        
        train_files = file_list[:train_end]
        val_files = file_list[train_end:val_end]
        test_files = file_list[val_end:]
        
        return train_files, val_files, test_files
    
    def copy_files_and_convert(self, json_files: List[Path], split_name: str):
        """
        复制图片文件并转换标注文件
        
        Args:
            json_files: JSON文件路径列表
            split_name: 数据集分割名称 ('train', 'val', 'test')
        """
        images_split_dir = IMAGES_DIR / split_name
        labels_split_dir = LABELS_DIR / split_name
        
        for json_file in json_files:
            # 获取对应的图片文件
            image_path = self.get_image_path(json_file)
            if image_path is None or not image_path.exists():
                print(f"找不到图片文件: {json_file}")
                continue
            
            # 复制图片文件
            image_dest = images_split_dir / image_path.name
            try:
                shutil.copy2(image_path, image_dest)
            except Exception as e:
                print(f"复制图片文件失败 {image_path}: {e}")
                continue
            
            # 转换并保存标注文件
            yolo_lines = self.convert_json_to_yolo(json_file)
            label_dest = labels_split_dir / f"{json_file.stem}.txt"
            
            try:
                with open(label_dest, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(yolo_lines))
                print(f"转换完成: {json_file.name} -> {split_name}")
            except Exception as e:
                print(f"保存标注文件失败 {label_dest}: {e}")
    
    def convert_all(self):
        """转换所有数据"""
        print("开始转换数据...")
        
        # 获取所有JSON文件
        json_files = list(YOLO_POINT_DIR.glob("*.json"))
        if not json_files:
            print(f"在 {YOLO_POINT_DIR} 中没有找到JSON文件")
            return
        
        print(f"找到 {len(json_files)} 个JSON文件")
        
        # 分割数据集
        train_files, val_files, test_files = self.split_dataset(json_files)
        
        print(f"数据集分割:")
        print(f"  训练集: {len(train_files)} 个文件")
        print(f"  验证集: {len(val_files)} 个文件") 
        print(f"  测试集: {len(test_files)} 个文件")
        
        # 转换各个数据集
        self.copy_files_and_convert(train_files, "train")
        self.copy_files_and_convert(val_files, "val")
        self.copy_files_and_convert(test_files, "test")
        
        print("数据转换完成!")
        
        # 生成数据集配置文件
        self.generate_dataset_yaml()
    
    def generate_dataset_yaml(self):
        """生成YOLO数据集配置文件"""
        yaml_content = f"""# YOLOv8 数据集配置文件
# 自动生成于数据转换过程

# 数据集路径 (相对于此文件的路径)
path: {DATASETS_DIR.absolute()}
train: images/train
val: images/val
test: images/test

# 类别数量
nc: {len(CLASS_NAMES)}

# 类别名称
names:
"""
        
        for i, name in enumerate(CLASS_NAMES):
            yaml_content += f"  {i}: {name}\n"
        
        yaml_file = DATASETS_DIR.parent / "dataset.yaml"
        with open(yaml_file, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        print(f"数据集配置文件已生成: {yaml_file}")


if __name__ == "__main__":
    converter = DataConverter()
    converter.convert_all()
