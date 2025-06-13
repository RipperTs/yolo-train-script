#!/usr/bin/env python3
"""
类别管理模块
自动检测、管理和同步标注数据中的类别信息
完全基于数据驱动，不依赖硬编码的类别配置
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ClassInfo:
    """类别信息"""
    name: str
    id: int
    count: int = 0  # 该类别的标注数量
    first_seen: Optional[str] = None  # 首次发现时间
    last_seen: Optional[str] = None   # 最后见到时间


class ClassManager:
    """类别管理器 - 完全基于数据驱动"""
    
    def __init__(self, project_root: Path = None):
        """
        初始化类别管理器
        
        Args:
            project_root: 项目根目录
        """
        self.project_root = project_root or Path(__file__).parent
        self.class_registry_file = self.project_root / "class_registry.json"
        self.dataset_yaml_file = self.project_root / "dataset.yaml"
        
        # 内存中的类别信息
        self.classes: Dict[str, ClassInfo] = {}
        self.class_to_id: Dict[str, int] = {}
        self.id_to_class: Dict[int, str] = {}
        
        # 加载已有的类别注册表
        self.load_class_registry()
    
    def scan_annotation_directory(self, annotation_dir: Path) -> Set[str]:
        """
        扫描标注目录，发现所有类别
        
        Args:
            annotation_dir: 标注文件目录
            
        Returns:
            发现的类别集合
        """
        discovered_classes = set()
        annotation_counts = {}
        
        if not annotation_dir.exists():
            print(f"⚠️ 标注目录不存在: {annotation_dir}")
            return discovered_classes
        
        # 扫描JSON文件
        json_files = list(annotation_dir.glob("*.json"))
        print(f"🔍 扫描 {len(json_files)} 个标注文件...")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                shapes = data.get('shapes', [])
                for shape in shapes:
                    label = shape.get('label', '').strip()
                    if label:  # 只处理非空标签
                        discovered_classes.add(label)
                        annotation_counts[label] = annotation_counts.get(label, 0) + 1
                        
            except Exception as e:
                print(f"⚠️ 扫描文件 {json_file} 时出错: {e}")
                continue
        
        # 更新类别信息
        current_time = datetime.now().isoformat()
        for class_name in discovered_classes:
            if class_name not in self.classes:
                # 新发现的类别
                new_id = self._get_next_class_id()
                self.classes[class_name] = ClassInfo(
                    name=class_name,
                    id=new_id,
                    count=annotation_counts.get(class_name, 0),
                    first_seen=current_time,
                    last_seen=current_time
                )
                print(f"📋 发现新类别: {class_name} (ID: {new_id})")
            else:
                # 更新已有类别
                self.classes[class_name].count = annotation_counts.get(class_name, 0)
                self.classes[class_name].last_seen = current_time
        
        # 重建映射
        self._rebuild_mappings()
        
        print(f"✅ 类别扫描完成，共发现 {len(discovered_classes)} 个类别")
        return discovered_classes
    
    def _get_next_class_id(self) -> int:
        """获取下一个可用的类别ID"""
        if not self.classes:
            return 0
        return max(cls_info.id for cls_info in self.classes.values()) + 1
    
    def _rebuild_mappings(self):
        """重建类别映射"""
        # 按类别名称排序以保证一致性
        sorted_classes = sorted(self.classes.items(), key=lambda x: x[1].id)
        
        self.class_to_id = {name: info.id for name, info in sorted_classes}
        self.id_to_class = {info.id: name for name, info in sorted_classes}
    
    def get_class_names(self) -> List[str]:
        """
        获取所有类别名称（按ID排序）
        
        Returns:
            类别名称列表
        """
        if not self.classes:
            return []
        
        # 按ID排序
        sorted_items = sorted(self.class_to_id.items(), key=lambda x: x[1])
        return [name for name, _ in sorted_items]
    
    def get_class_count(self) -> int:
        """获取类别数量"""
        return len(self.classes)
    
    def get_class_id(self, class_name: str) -> Optional[int]:
        """获取类别ID"""
        return self.class_to_id.get(class_name)
    
    def get_class_name(self, class_id: int) -> Optional[str]:
        """获取类别名称"""
        return self.id_to_class.get(class_id)
    
    def get_class_info(self, class_name: str) -> Optional[ClassInfo]:
        """获取类别详细信息"""
        return self.classes.get(class_name)
    
    def save_class_registry(self):
        """保存类别注册表到文件"""
        registry_data = {
            "last_updated": datetime.now().isoformat(),
            "total_classes": len(self.classes),
            "classes": {
                name: {
                    "id": info.id,
                    "count": info.count,
                    "first_seen": info.first_seen,
                    "last_seen": info.last_seen
                }
                for name, info in self.classes.items()
            }
        }
        
        try:
            with open(self.class_registry_file, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)
            print(f"💾 类别注册表已保存: {self.class_registry_file}")
        except Exception as e:
            print(f"❌ 保存类别注册表失败: {e}")
    
    def load_class_registry(self):
        """从文件加载类别注册表"""
        if not self.class_registry_file.exists():
            print("📋 类别注册表不存在，将创建新的注册表")
            return
        
        try:
            with open(self.class_registry_file, 'r', encoding='utf-8') as f:
                registry_data = json.load(f)
            
            classes_data = registry_data.get('classes', {})
            
            for name, data in classes_data.items():
                self.classes[name] = ClassInfo(
                    name=name,
                    id=data['id'],
                    count=data.get('count', 0),
                    first_seen=data.get('first_seen'),
                    last_seen=data.get('last_seen')
                )
            
            self._rebuild_mappings()
            
            print(f"📋 从注册表加载了 {len(self.classes)} 个类别")
            
        except Exception as e:
            print(f"⚠️ 加载类别注册表失败: {e}")
            self.classes = {}
    
    def generate_dataset_yaml(self, dataset_dir: Path, class_names: List[str] = None):
        """
        生成dataset.yaml文件
        
        Args:
            dataset_dir: 数据集目录
            class_names: 类别名称列表，如果为None则使用当前管理的类别
        """
        if class_names is None:
            class_names = self.get_class_names()
        
        if not class_names:
            raise ValueError("没有可用的类别信息")
        
        yaml_content = {
            "path": str(dataset_dir.absolute()),
            "train": "images/train",
            "val": "images/val", 
            "test": "images/test",
            "nc": len(class_names),
            "names": {i: name for i, name in enumerate(class_names)}
        }
        
        try:
            with open(self.dataset_yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_content, f, default_flow_style=False, 
                         allow_unicode=True, sort_keys=False)
            
            print(f"📄 Dataset.yaml已生成: {self.dataset_yaml_file}")
            print(f"📊 包含 {len(class_names)} 个类别: {class_names}")
            
        except Exception as e:
            print(f"❌ 生成dataset.yaml失败: {e}")
            raise
    
    def load_classes_from_yaml(self) -> List[str]:
        """
        从dataset.yaml文件加载类别
        
        Returns:
            类别名称列表
        """
        if not self.dataset_yaml_file.exists():
            return []
        
        try:
            with open(self.dataset_yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            names = data.get('names', {})
            if isinstance(names, dict):
                # 按ID排序
                class_names = [names[i] for i in sorted(names.keys()) if isinstance(i, int)]
                return class_names
            elif isinstance(names, list):
                return names
            
        except Exception as e:
            print(f"⚠️ 从dataset.yaml加载类别失败: {e}")
        
        return []
    
    def sync_with_annotation_data(self, annotation_dir: Path) -> bool:
        """
        与标注数据同步类别信息
        
        Args:
            annotation_dir: 标注目录
            
        Returns:
            是否有类别更新
        """
        print("🔄 同步类别信息...")
        
        # 记录原有类别
        old_classes = set(self.classes.keys())
        
        # 扫描新的类别
        discovered_classes = self.scan_annotation_directory(annotation_dir)
        
        # 检查是否有变化
        has_changes = old_classes != discovered_classes
        
        if has_changes:
            print(f"📊 类别信息已更新:")
            
            # 新增的类别
            new_classes = discovered_classes - old_classes
            if new_classes:
                print(f"  ➕ 新增类别: {sorted(new_classes)}")
            
            # 移除的类别（在注册表中但不在数据中）
            removed_classes = old_classes - discovered_classes
            if removed_classes:
                print(f"  ➖ 不再使用的类别: {sorted(removed_classes)}")
                # 注意：我们不删除注册表中的类别，只是标记它们不活跃
            
            # 保存更新后的注册表
            self.save_class_registry()
        else:
            print("✅ 类别信息无变化")
        
        return has_changes
    
    def get_class_statistics(self) -> Dict:
        """获取类别统计信息"""
        if not self.classes:
            return {"total_classes": 0, "classes": []}
        
        class_stats = []
        for name, info in self.classes.items():
            class_stats.append({
                "name": name,
                "id": info.id,
                "count": info.count,
                "first_seen": info.first_seen,
                "last_seen": info.last_seen
            })
        
        # 按标注数量排序
        class_stats.sort(key=lambda x: x["count"], reverse=True)
        
        return {
            "total_classes": len(self.classes),
            "total_annotations": sum(info.count for info in self.classes.values()),
            "classes": class_stats
        }
    
    def export_class_mapping(self) -> Dict[str, int]:
        """导出类别映射（用于训练）"""
        return self.class_to_id.copy()


# 全局类别管理器实例
class_manager = ClassManager()


def get_class_names_from_data(annotation_dir: Path) -> List[str]:
    """
    便捷函数：从标注数据获取类别名称
    
    Args:
        annotation_dir: 标注目录
        
    Returns:
        类别名称列表
    """
    class_manager.sync_with_annotation_data(annotation_dir)
    return class_manager.get_class_names()


def get_current_class_mapping() -> Dict[str, int]:
    """
    便捷函数：获取当前的类别映射
    
    Returns:
        类别映射字典
    """
    return class_manager.export_class_mapping()


if __name__ == "__main__":
    # 测试类别管理器
    from config import YOLO_POINT_DIR, DATASETS_DIR
    
    print("🧪 测试类别管理器")
    print("=" * 50)
    
    # 同步标注数据
    if YOLO_POINT_DIR.exists():
        class_manager.sync_with_annotation_data(YOLO_POINT_DIR)
        
        # 显示统计信息
        stats = class_manager.get_class_statistics()
        print(f"\n📊 类别统计:")
        print(f"总类别数: {stats['total_classes']}")
        print(f"总标注数: {stats['total_annotations']}")
        
        for cls_info in stats['classes']:
            print(f"  {cls_info['id']}: {cls_info['name']} ({cls_info['count']} 个标注)")
        
        # 生成dataset.yaml
        try:
            class_manager.generate_dataset_yaml(DATASETS_DIR)
        except Exception as e:
            print(f"❌ 生成dataset.yaml失败: {e}")
    
    else:
        print(f"⚠️ 标注目录不存在: {YOLO_POINT_DIR}") 