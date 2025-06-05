"""
YOLOv8 工具模块
提供数据处理、可视化和评估等工具函数
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import pandas as pd

from config import DATASETS_DIR, CLASS_NAMES


def visualize_dataset_distribution():
    """可视化数据集分布"""
    splits = ['train', 'val', 'test']
    class_counts = {split: {cls: 0 for cls in CLASS_NAMES} for split in splits}
    total_images = {split: 0 for split in splits}
    
    for split in splits:
        labels_dir = DATASETS_DIR / "labels" / split
        if not labels_dir.exists():
            continue
            
        label_files = list(labels_dir.glob("*.txt"))
        total_images[split] = len(label_files)
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    if line.strip():
                        class_id = int(line.split()[0])
                        if class_id < len(CLASS_NAMES):
                            class_counts[split][CLASS_NAMES[class_id]] += 1
            except Exception as e:
                print(f"读取标签文件出错 {label_file}: {e}")
    
    # 绘制分布图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 图片数量分布
    splits_list = list(total_images.keys())
    counts_list = list(total_images.values())
    ax1.bar(splits_list, counts_list)
    ax1.set_title('数据集图片数量分布')
    ax1.set_ylabel('图片数量')
    for i, count in enumerate(counts_list):
        ax1.text(i, count + 0.5, str(count), ha='center')
    
    # 类别分布
    class_data = []
    for split in splits:
        for cls in CLASS_NAMES:
            class_data.append({
                'split': split,
                'class': cls,
                'count': class_counts[split][cls]
            })
    
    df = pd.DataFrame(class_data)
    pivot_df = df.pivot(index='class', columns='split', values='count')
    pivot_df.plot(kind='bar', ax=ax2)
    ax2.set_title('类别分布')
    ax2.set_ylabel('标注数量')
    ax2.legend(title='数据集')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(DATASETS_DIR.parent / 'dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print("\n数据集统计信息:")
    print("-" * 50)
    for split in splits:
        print(f"{split.upper()}:")
        print(f"  图片数量: {total_images[split]}")
        total_annotations = sum(class_counts[split].values())
        print(f"  标注数量: {total_annotations}")
        for cls in CLASS_NAMES:
            count = class_counts[split][cls]
            if total_annotations > 0:
                percentage = count / total_annotations * 100
                print(f"    {cls}: {count} ({percentage:.1f}%)")
        print()


def check_dataset_integrity():
    """检查数据集完整性"""
    print("检查数据集完整性...")
    
    issues = []
    splits = ['train', 'val', 'test']
    
    for split in splits:
        images_dir = DATASETS_DIR / "images" / split
        labels_dir = DATASETS_DIR / "labels" / split
        
        if not images_dir.exists():
            issues.append(f"图片目录不存在: {images_dir}")
            continue
            
        if not labels_dir.exists():
            issues.append(f"标签目录不存在: {labels_dir}")
            continue
        
        # 获取图片和标签文件
        image_files = set()
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.update(f.stem for f in images_dir.glob(ext))
        
        label_files = set(f.stem for f in labels_dir.glob("*.txt"))
        
        # 检查匹配性
        missing_labels = image_files - label_files
        missing_images = label_files - image_files
        
        if missing_labels:
            issues.append(f"{split}: {len(missing_labels)} 个图片缺少标签文件")
            
        if missing_images:
            issues.append(f"{split}: {len(missing_images)} 个标签文件缺少对应图片")
        
        # 检查标签文件格式
        for label_file in labels_dir.glob("*.txt"):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        issues.append(f"{label_file}:{line_num} - 格式错误，应为5个值")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])
                        
                        if class_id < 0 or class_id >= len(CLASS_NAMES):
                            issues.append(f"{label_file}:{line_num} - 类别ID超出范围: {class_id}")
                        
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                               0 <= width <= 1 and 0 <= height <= 1):
                            issues.append(f"{label_file}:{line_num} - 坐标超出[0,1]范围")
                            
                    except ValueError:
                        issues.append(f"{label_file}:{line_num} - 数值格式错误")
                        
            except Exception as e:
                issues.append(f"读取标签文件出错 {label_file}: {e}")
    
    if issues:
        print(f"发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("数据集完整性检查通过!")
    
    return len(issues) == 0


def visualize_annotations(image_path: str, label_path: str = None, output_path: str = None):
    """
    可视化图片和对应的标注
    
    Args:
        image_path: 图片路径
        label_path: 标签文件路径，如果为None则自动推断
        output_path: 输出路径
    """
    image_path = Path(image_path)
    
    if label_path is None:
        # 自动查找对应的标签文件
        possible_label_paths = [
            image_path.with_suffix('.txt'),
            image_path.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
        ]
        
        label_path = None
        for path in possible_label_paths:
            if path.exists():
                label_path = path
                break
        
        if label_path is None:
            print(f"找不到对应的标签文件: {image_path}")
            return
    
    # 读取图片
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"无法读取图片: {image_path}")
        return
    
    height, width = image.shape[:2]
    
    # 读取标签
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 5:
                continue
            
            class_id = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:])
            
            # 转换为像素坐标
            x_center *= width
            y_center *= height
            w *= width
            h *= height
            
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)
            
            # 绘制边界框
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制类别标签
            if class_id < len(CLASS_NAMES):
                label = CLASS_NAMES[class_id]
            else:
                label = f"class_{class_id}"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    except Exception as e:
        print(f"读取标签文件出错: {e}")
        return
    
    # 保存或显示结果
    if output_path:
        cv2.imwrite(str(output_path), image)
        print(f"可视化结果已保存: {output_path}")
    else:
        cv2.imshow('Annotations', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def create_data_subset(source_split: str, target_split: str, num_samples: int):
    """
    从现有数据集创建子集
    
    Args:
        source_split: 源数据集分割 ('train', 'val', 'test')
        target_split: 目标数据集分割名称
        num_samples: 样本数量
    """
    source_images_dir = DATASETS_DIR / "images" / source_split
    source_labels_dir = DATASETS_DIR / "labels" / source_split
    
    target_images_dir = DATASETS_DIR / "images" / target_split
    target_labels_dir = DATASETS_DIR / "labels" / target_split
    
    # 创建目标目录
    target_images_dir.mkdir(parents=True, exist_ok=True)
    target_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取源文件列表
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(source_images_dir.glob(ext))
    
    if len(image_files) < num_samples:
        print(f"源数据集只有 {len(image_files)} 个样本，少于请求的 {num_samples} 个")
        num_samples = len(image_files)
    
    # 随机选择样本
    import random
    selected_files = random.sample(image_files, num_samples)
    
    # 复制文件
    for image_file in selected_files:
        # 复制图片
        target_image = target_images_dir / image_file.name
        shutil.copy2(image_file, target_image)
        
        # 复制标签
        label_file = source_labels_dir / f"{image_file.stem}.txt"
        if label_file.exists():
            target_label = target_labels_dir / f"{image_file.stem}.txt"
            shutil.copy2(label_file, target_label)
    
    print(f"已创建包含 {num_samples} 个样本的 {target_split} 数据集")


def calculate_anchor_boxes(num_anchors: int = 9):
    """
    基于数据集计算锚框
    
    Args:
        num_anchors: 锚框数量
        
    Returns:
        锚框列表
    """
    print("计算锚框...")
    
    all_boxes = []
    
    for split in ['train', 'val']:
        labels_dir = DATASETS_DIR / "labels" / split
        if not labels_dir.exists():
            continue
        
        for label_file in labels_dir.glob("*.txt"):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    
                    _, _, _, width, height = map(float, parts)
                    all_boxes.append([width, height])
                    
            except Exception as e:
                print(f"读取标签文件出错 {label_file}: {e}")
    
    if not all_boxes:
        print("没有找到有效的边界框数据")
        return []
    
    # 使用K-means聚类计算锚框
    from sklearn.cluster import KMeans
    
    boxes_array = np.array(all_boxes)
    kmeans = KMeans(n_clusters=num_anchors, random_state=42)
    kmeans.fit(boxes_array)
    
    anchors = kmeans.cluster_centers_
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]  # 按面积排序
    
    print(f"计算得到的锚框 (width, height):")
    for i, (w, h) in enumerate(anchors):
        print(f"  {i+1}: ({w:.4f}, {h:.4f})")
    
    return anchors.tolist()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv8工具脚本")
    parser.add_argument("--action", choices=["visualize", "check", "distribution", "subset", "anchors"],
                       required=True, help="执行的操作")
    parser.add_argument("--image", type=str, help="图片路径")
    parser.add_argument("--label", type=str, help="标签路径")
    parser.add_argument("--output", type=str, help="输出路径")
    parser.add_argument("--source", type=str, help="源数据集分割")
    parser.add_argument("--target", type=str, help="目标数据集分割")
    parser.add_argument("--num", type=int, help="样本数量")
    
    args = parser.parse_args()
    
    if args.action == "visualize":
        if not args.image:
            print("可视化需要指定 --image 参数")
        else:
            visualize_annotations(args.image, args.label, args.output)
    
    elif args.action == "check":
        check_dataset_integrity()
    
    elif args.action == "distribution":
        visualize_dataset_distribution()
    
    elif args.action == "subset":
        if not all([args.source, args.target, args.num]):
            print("创建子集需要指定 --source, --target, --num 参数")
        else:
            create_data_subset(args.source, args.target, args.num)
    
    elif args.action == "anchors":
        calculate_anchor_boxes()
