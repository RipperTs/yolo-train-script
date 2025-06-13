"""
数据转换模块
将yolo_point目录中的JSON标注文件转换为YOLO训练格式
支持点标注和矩形标注的自动检测和转换
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Set
import random
from PIL import Image
import numpy as np

from config import (
    YOLO_POINT_DIR, DATASETS_DIR, IMAGES_DIR, LABELS_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, ensure_directories
)
from class_manager import class_manager


class DataConverter:
    """数据转换器类"""

    def __init__(self, source_dir=None):
        self.source_dir = Path(source_dir) if source_dir else YOLO_POINT_DIR
        self.class_to_id = {}  # 将从类别管理器获取
        self.auto_detected_classes = set()
        ensure_directories()



    def check_source_directory(self):
        """检查源目录状态"""
        if not self.source_dir.exists():
            return {
                "status": "not_exists",
                "message": f"源目录不存在: {self.source_dir}",
                "path": str(self.source_dir)
            }

        json_files = list(self.source_dir.glob("*.json"))
        if not json_files:
            return {
                "status": "empty",
                "message": f"源目录中没有JSON文件: {self.source_dir}",
                "path": str(self.source_dir),
                "file_count": 0
            }

        return {
            "status": "ready",
            "message": f"找到 {len(json_files)} 个JSON文件",
            "path": str(self.source_dir),
            "file_count": len(json_files),
            "files": [f.name for f in json_files[:5]]
        }
    
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
            skipped_labels = set()
            
            for shape in shapes:
                shape_type = shape.get('shape_type', '')
                label = shape.get('label', '').strip()

                if not label:
                    continue  # 跳过空标签
                
                if label not in self.class_to_id:
                    skipped_labels.add(label)
                    continue  # 跳过未知类别

                class_id = self.class_to_id[label]
                points = shape.get('points', [])

                if shape_type == 'rectangle':
                    # 处理矩形标注
                    if len(points) < 2:  # 至少需要2个点来定义矩形
                        print(f"⚠️ 矩形标注点数不足: {len(points)}")
                        continue

                    # 提取边界框坐标（支持不同的矩形表示方式）
                    if len(points) == 4:
                        # 4个角点的情况
                        x_coords = [point[0] for point in points]
                        y_coords = [point[1] for point in points]
                    elif len(points) == 2:
                        # 2个对角点的情况
                        x_coords = [points[0][0], points[1][0]]
                        y_coords = [points[0][1], points[1][1]]
                    else:
                        print(f"⚠️ 不支持的矩形点数: {len(points)}")
                        continue

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
                    print(f"⚠️ 跳过不支持的标注类型: {shape_type}")
                    continue  # 跳过其他类型的标注
                
                # 检查边界框有效性
                if x_max <= x_min or y_max <= y_min:
                    print(f"⚠️ 无效的边界框: ({x_min}, {y_min}, {x_max}, {y_max})")
                    continue
                
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
                
                # 检查转换后的数值有效性
                if width > 0 and height > 0:
                    yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    yolo_lines.append(yolo_line)
                else:
                    print(f"⚠️ 转换后的边界框无效: width={width}, height={height}")
            
            # 报告跳过的标签
            if skipped_labels:
                print(f"⚠️ 文件 {json_file_path.name} 中跳过了未知类别: {skipped_labels}")
            
            return yolo_lines
            
        except Exception as e:
            print(f"❌ 转换文件 {json_file_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
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
                # 假设图片在源目录的同级目录或子目录中
                possible_paths = [
                    self.source_dir / image_name,
                    self.source_dir.parent / image_name,
                    self.source_dir.parent / "images" / image_name,
                ]
                
                for path in possible_paths:
                    if path.exists():
                        return path
            
            # 如果找不到，尝试根据JSON文件名推断图片名
            base_name = json_file_path.stem
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_path = self.source_dir / f"{base_name}{ext}"
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
        
        successful_conversions = 0
        failed_conversions = 0
        empty_label_files = 0
        total_annotations = 0
        
        print(f"  🔄 正在处理 {split_name} 集: {len(json_files)} 个文件")
        
        for json_file in json_files:
            # 获取对应的图片文件
            image_path = self.get_image_path(json_file)
            if image_path is None or not image_path.exists():
                print(f"  ❌ 找不到图片文件: {json_file}")
                failed_conversions += 1
                continue
            
            # 复制图片文件
            image_dest = images_split_dir / image_path.name
            try:
                shutil.copy2(image_path, image_dest)
            except Exception as e:
                print(f"  ❌ 复制图片文件失败 {image_path}: {e}")
                failed_conversions += 1
                continue
            
            # 转换并保存标注文件
            yolo_lines = self.convert_json_to_yolo(json_file)
            label_dest = labels_split_dir / f"{json_file.stem}.txt"
            
            try:
                with open(label_dest, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(yolo_lines))
                
                if len(yolo_lines) == 0:
                    empty_label_files += 1
                    print(f"  ⚠️ 空标签文件: {json_file.name} -> {split_name}")
                else:
                    total_annotations += len(yolo_lines)
                    successful_conversions += 1
                
            except Exception as e:
                print(f"  ❌ 保存标注文件失败 {label_dest}: {e}")
                failed_conversions += 1
        
        # 输出转换统计
        print(f"  ✅ {split_name} 集转换完成:")
        print(f"     成功: {successful_conversions} 个文件")
        if failed_conversions > 0:
            print(f"     失败: {failed_conversions} 个文件")
        if empty_label_files > 0:
            print(f"     空标签: {empty_label_files} 个文件")
        print(f"     标注总数: {total_annotations} 个")
    
    def convert_all(self):
        """转换所有数据"""
        print(f"🚀 开始转换数据，源目录: {self.source_dir}")

        # 检查源目录状态
        status = self.check_source_directory()
        if status["status"] != "ready":
            print(status["message"])
            return status

        # 获取所有JSON文件
        json_files = list(self.source_dir.glob("*.json"))
        if not json_files:
            print(f"在 {self.source_dir} 中没有找到JSON文件")
            return status
        
        print(f"找到 {len(json_files)} 个JSON文件")
        
        # 使用类别管理器同步类别信息
        print("\n🔍 使用智能类别管理器检测类别...")
        class_manager.sync_with_annotation_data(self.source_dir)
        
        # 获取类别映射
        self.class_to_id = class_manager.export_class_mapping()
        class_names = class_manager.get_class_names()
        
        if not class_names:
            print("❌ 没有发现任何有效的类别")
            return {"status": "no_classes", "message": "没有发现任何有效的类别"}
        
        print(f"\n📊 检测到的类别 ({len(class_names)} 个):")
        for class_name, class_id in self.class_to_id.items():
            class_info = class_manager.get_class_info(class_name)
            count = class_info.count if class_info else 0
            print(f"  {class_id}: {class_name} ({count} 个标注)")
        
        # 分割数据集
        train_files, val_files, test_files = self.split_dataset(json_files)
        
        print(f"\n📁 数据集分割:")
        print(f"  训练集: {len(train_files)} 个文件")
        print(f"  验证集: {len(val_files)} 个文件") 
        print(f"  测试集: {len(test_files)} 个文件")
        
        # 转换各个数据集
        print(f"\n🔄 开始转换数据...")
        self.copy_files_and_convert(train_files, "train")
        self.copy_files_and_convert(val_files, "val")
        self.copy_files_and_convert(test_files, "test")
        
        print("\n✅ 数据转换完成!")
        
        # 使用类别管理器生成数据集配置文件
        class_manager.generate_dataset_yaml(DATASETS_DIR)
        
        # 显示转换统计
        stats = class_manager.get_class_statistics()
        print(f"\n📊 转换统计:")
        print(f"  总类别数: {stats['total_classes']}")
        print(f"  总标注数: {stats['total_annotations']}")
        
        return {"status": "success", "class_count": len(class_names), "classes": class_names}
    
    def generate_dataset_yaml(self, class_names=None):
        """生成YOLO数据集配置文件"""
        if class_names is None:
            # 使用类别管理器获取类别名称
            class_names = class_manager.get_class_names()
            
        yaml_content = f"""# YOLOv8 数据集配置文件
# 自动生成于数据转换过程

# 数据集路径 (相对于此文件的路径)
path: {DATASETS_DIR.absolute()}
train: images/train
val: images/val
test: images/test

# 类别数量
nc: {len(class_names)}

# 类别名称
names:
"""
        
        for i, name in enumerate(class_names):
            yaml_content += f"  {i}: {name}\n"
        
        yaml_file = DATASETS_DIR.parent / "dataset.yaml"
        with open(yaml_file, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        print(f"📄 数据集配置文件已生成: {yaml_file}")
        print(f"📊 包含 {len(class_names)} 个类别: {class_names}")


if __name__ == "__main__":
    converter = DataConverter()
    converter.convert_all()
