"""
YOLOv8 推理模块
提供模型推理和预测功能
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict, Union, Tuple
import json
from PIL import Image

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from config import INFERENCE_CONFIG, MODELS_DIR, get_default_device
from config_manager import config_manager
from device_manager import device_manager
from class_manager import class_manager


class YOLOv8Inference:
    """YOLOv8推理器类"""
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        初始化推理器
        
        Args:
            model_path: 模型路径，如果为None则尝试加载最新的训练模型
            device: 推理设备，如果为None则使用配置中的设备或自动选择
        """
        self.model = None
        self.model_path = model_path
        self.class_names = self._load_class_names()
        self.device = device
        
        if model_path is None:
            self.model_path = self._find_latest_model()
        
        # 设置推理设备
        self._setup_device()
        
        self.load_model()
    
    def _load_class_names(self) -> List[str]:
        """
        从类别管理器加载类别名称
        
        Returns:
            类别名称列表
        """
        try:
            # 优先从类别管理器获取最新的类别信息
            class_names = class_manager.get_class_names()
            
            if class_names:
                print(f"📋 从类别管理器加载类别: {class_names}")
                return class_names
            
            # 备用方案：尝试从dataset.yaml加载
            class_names = class_manager.load_classes_from_yaml()
            if class_names:
                print(f"📋 从dataset.yaml加载类别: {class_names}")
                return class_names
            
            # 最后备用：返回通用类别名称
            print("⚠️ 无法加载类别信息，使用通用类别名称")
            return [f"class_{i}" for i in range(10)]  # 返回10个通用类别
            
        except Exception as e:
            print(f"⚠️ 加载类别名称失败: {e}，使用通用类别名称")
            return [f"class_{i}" for i in range(10)]
    
    def _setup_device(self):
        """设置推理设备"""
        if self.device is None:
            # 从配置管理器获取推理设备设置
            inference_config = config_manager.get_inference_config()
            self.device = inference_config.get("device")
            
            # 如果配置中也没有设备设置，使用训练配置的设备
            if self.device is None:
                training_config = config_manager.get_training_config()
                self.device = training_config.get("device") or get_default_device()
        
        # 验证并设置设备
        if not device_manager.set_device(self.device):
            print(f"⚠️ 无法使用推理设备 {self.device}，使用降级设备")
        
        self.device = device_manager.current_device
        print(f"🎯 使用推理设备: {self.device}")
    
    def _find_latest_model(self) -> str:
        """查找最新的训练模型"""
        model_files = []
        
        # 查找.pt文件
        for pattern in ["**/*.pt", "**/*best.pt", "**/best.pt"]:
            model_files.extend(MODELS_DIR.glob(pattern))
        
        if not model_files:
            raise FileNotFoundError(
                f"在 {MODELS_DIR} 中没有找到训练好的模型文件\n"
                "请先运行训练或指定模型路径"
            )
        
        # 按修改时间排序，返回最新的
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        print(f"自动选择最新模型: {latest_model}")
        
        return str(latest_model)
    
    def load_model(self):
        """加载模型"""
        try:
            from ultralytics import YOLO
            
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            self.model = YOLO(self.model_path)
            print(f"模型加载成功: {self.model_path}")
            
        except ImportError:
            raise ImportError(
                "未安装ultralytics库，请运行: pip install ultralytics"
            )
        except Exception as e:
            raise Exception(f"加载模型失败: {e}")
    
    def reload_class_names(self):
        """重新加载类别名称"""
        self.class_names = self._load_class_names()
    
    def predict_image(self, image_path: str, save_result: bool = True) -> Dict:
        """
        对单张图片进行预测
        
        Args:
            image_path: 图片路径
            save_result: 是否保存结果图片
            
        Returns:
            预测结果字典
        """
        if self.model is None:
            raise ValueError("模型未加载")
        
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 获取当前推理配置
        inference_config = config_manager.get_inference_config()
        
        # 进行预测
        results = self.model(
            str(image_path),
            conf=inference_config.get("conf_threshold", INFERENCE_CONFIG["conf_threshold"]),
            iou=inference_config.get("iou_threshold", INFERENCE_CONFIG["iou_threshold"]),
            max_det=inference_config.get("max_det", INFERENCE_CONFIG["max_det"]),
            imgsz=inference_config.get("img_size", INFERENCE_CONFIG["img_size"]),
            device=self.device,  # 使用配置的设备
            save=save_result,
            verbose=False
        )
        
        # 解析结果
        result = results[0]
        predictions = self._parse_results(result)
        
        return {
            "image_path": str(image_path),
            "image_size": (result.orig_shape[1], result.orig_shape[0]),  # (width, height)
            "predictions": predictions,
            "num_detections": len(predictions)
        }
    
    def predict_batch(self, image_paths: List[str], save_results: bool = True) -> List[Dict]:
        """
        批量预测多张图片
        
        Args:
            image_paths: 图片路径列表
            save_results: 是否保存结果图片
            
        Returns:
            预测结果列表
        """
        if self.model is None:
            raise ValueError("模型未加载")
        
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path, save_result=save_results)
                results.append(result)
                print(f"预测完成: {image_path}")
            except Exception as e:
                print(f"预测失败 {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "predictions": [],
                    "num_detections": 0
                })
        
        return results
    
    def _parse_results(self, result) -> List[Dict]:
        """
        解析YOLO预测结果
        
        Args:
            result: YOLO结果对象
            
        Returns:
            解析后的预测结果列表
        """
        predictions = []
        
        if result.boxes is not None:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # 获取边界框坐标 (xyxy格式)
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = box
                
                # 获取置信度
                confidence = float(boxes.conf[i].cpu().numpy())
                
                # 获取类别
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                
                prediction = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1),
                        "center_x": float((x1 + x2) / 2),
                        "center_y": float((y1 + y2) / 2)
                    }
                }
                
                predictions.append(prediction)
        
        return predictions
    
    def predict_from_array(self, image_array: np.ndarray) -> Dict:
        """
        从numpy数组预测
        
        Args:
            image_array: 图片数组 (BGR格式)
            
        Returns:
            预测结果字典
        """
        if self.model is None:
            raise ValueError("模型未加载")
        
        # 进行预测
        results = self.model(
            image_array,
            conf=INFERENCE_CONFIG["conf_threshold"],
            iou=INFERENCE_CONFIG["iou_threshold"],
            max_det=INFERENCE_CONFIG["max_det"],
            imgsz=INFERENCE_CONFIG["img_size"],
            verbose=False
        )
        
        # 解析结果
        result = results[0]
        predictions = self._parse_results(result)
        
        return {
            "image_size": (result.orig_shape[1], result.orig_shape[0]),  # (width, height)
            "predictions": predictions,
            "num_detections": len(predictions)
        }
    
    def visualize_predictions(self, image_path: str, predictions: List[Dict], 
                            output_path: str = None) -> str:
        """
        可视化预测结果
        
        Args:
            image_path: 原始图片路径
            predictions: 预测结果列表
            output_path: 输出图片路径
            
        Returns:
            输出图片路径
        """
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # 绘制预测结果
        for pred in predictions:
            bbox = pred["bbox"]
            class_name = pred["class_name"]
            confidence = pred["confidence"]
            
            # 绘制边界框
            x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # 保存结果
        if output_path is None:
            image_path = Path(image_path)
            output_path = image_path.parent / f"{image_path.stem}_result{image_path.suffix}"
        
        cv2.imwrite(str(output_path), image)
        print(f"可视化结果已保存: {output_path}")
        
        return str(output_path)
    
    def save_predictions_json(self, predictions_list: List[Dict], output_path: str):
        """
        保存预测结果为JSON文件
        
        Args:
            predictions_list: 预测结果列表
            output_path: 输出JSON文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions_list, f, ensure_ascii=False, indent=2)
        
        print(f"预测结果已保存: {output_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv8推理脚本")
    parser.add_argument("--model", type=str, help="模型路径")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--images", type=str, help="图片目录路径")
    parser.add_argument("--output", type=str, help="输出目录")
    parser.add_argument("--save-json", action="store_true", help="保存JSON结果")
    
    args = parser.parse_args()
    
    # 初始化推理器
    inference = YOLOv8Inference(model_path=args.model)
    
    if args.image:
        # 单张图片预测
        result = inference.predict_image(args.image)
        print(f"检测到 {result['num_detections']} 个目标")
        
        if args.save_json:
            json_path = Path(args.image).with_suffix('.json')
            inference.save_predictions_json([result], str(json_path))
    
    elif args.images:
        # 批量预测
        image_dir = Path(args.images)
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(image_dir.glob(ext))
        
        if not image_files:
            print(f"在 {image_dir} 中没有找到图片文件")
            return
        
        results = inference.predict_batch([str(f) for f in image_files])
        
        total_detections = sum(r['num_detections'] for r in results)
        print(f"批量预测完成，共检测到 {total_detections} 个目标")
        
        if args.save_json:
            json_path = image_dir / "predictions.json"
            inference.save_predictions_json(results, str(json_path))


if __name__ == "__main__":
    main()
