#!/usr/bin/env python3
"""
YOLOv8 快速开始脚本
用于快速测试训练和推理功能
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from config import ensure_directories, DATASETS_DIR, MODELS_DIR
from trainer import YOLOv8Trainer
from inference import YOLOv8Inference


def quick_train():
    """快速训练测试"""
    print("开始快速训练测试...")
    
    # 确保目录存在
    ensure_directories()
    
    # 检查数据集是否存在
    dataset_yaml = DATASETS_DIR.parent / "dataset.yaml"
    if not dataset_yaml.exists():
        print("数据集配置文件不存在，请先运行数据转换:")
        print("python run.py --pipeline convert")
        return False
    
    try:
        # 初始化训练器
        trainer = YOLOv8Trainer()
        trainer.get_device_info()
        
        # 修改训练配置为快速测试
        from config import TRAINING_CONFIG
        quick_config = TRAINING_CONFIG.copy()
        quick_config.update({
            "epochs": 5,  # 只训练5个epoch用于测试
            "batch_size": 4,  # 减小batch size
            "img_size": 320,  # 减小图片尺寸
            "patience": 10,
            "save_period": 1,
        })
        
        print("快速训练配置:")
        for key, value in quick_config.items():
            print(f"  {key}: {value}")
        
        # 开始训练
        print("\n开始训练...")
        trainer.model = None  # 重置模型
        trainer.setup_model()
        
        # 训练参数
        train_args = {
            "data": str(dataset_yaml),
            "epochs": quick_config["epochs"],
            "batch": quick_config["batch_size"],
            "imgsz": quick_config["img_size"],
            "lr0": quick_config["learning_rate"],
            "patience": quick_config["patience"],
            "save_period": quick_config["save_period"],
            "workers": 2,  # 减少worker数量
            "device": "cpu",  # 强制使用CPU
            "project": str(MODELS_DIR),
            "name": "quick_test",
            "exist_ok": True,
            "verbose": True,
        }
        
        results = trainer.model.train(**train_args)
        
        print("快速训练完成!")
        print(f"模型保存在: {results.save_dir}")
        
        return str(results.save_dir / "weights" / "best.pt")
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_inference(model_path=None):
    """快速推理测试"""
    print("开始快速推理测试...")
    
    if model_path is None:
        # 查找最新的模型
        model_files = list(MODELS_DIR.glob("**/best.pt"))
        if not model_files:
            print("没有找到训练好的模型，请先运行训练")
            return False
        model_path = str(max(model_files, key=lambda x: x.stat().st_mtime))
    
    try:
        # 初始化推理器
        inference = YOLOv8Inference(model_path=model_path)
        
        # 找一张测试图片
        test_images_dir = DATASETS_DIR / "images" / "train"  # 使用训练集中的图片
        image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.jpeg"))
        
        if not image_files:
            print("没有找到测试图片")
            return False
        
        test_image = image_files[0]
        print(f"使用测试图片: {test_image}")
        
        # 进行推理
        result = inference.predict_image(str(test_image))
        
        print(f"推理结果:")
        print(f"  检测到 {result['num_detections']} 个目标")
        print(f"  图片尺寸: {result['image_size']}")
        
        if result['predictions']:
            for i, pred in enumerate(result['predictions']):
                print(f"  目标 {i+1}:")
                print(f"    类别: {pred['class_name']}")
                print(f"    置信度: {pred['confidence']:.3f}")
                print(f"    边界框: ({pred['bbox']['x1']:.1f}, {pred['bbox']['y1']:.1f}, {pred['bbox']['x2']:.1f}, {pred['bbox']['y2']:.1f})")
        
        # 可视化结果
        if result['predictions']:
            output_path = inference.visualize_predictions(
                str(test_image), result['predictions']
            )
            print(f"可视化结果已保存: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"推理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv8快速开始脚本")
    parser.add_argument("--action", choices=["train", "inference", "both"], 
                       default="both", help="执行的操作")
    parser.add_argument("--model", type=str, help="推理时使用的模型路径")
    
    args = parser.parse_args()
    
    if args.action in ["train", "both"]:
        print("=" * 50)
        print("快速训练测试")
        print("=" * 50)
        model_path = quick_train()
        
        if model_path and args.action == "both":
            print("\n" + "=" * 50)
            print("快速推理测试")
            print("=" * 50)
            quick_inference(model_path)
    
    elif args.action == "inference":
        print("=" * 50)
        print("快速推理测试")
        print("=" * 50)
        quick_inference(args.model)


if __name__ == "__main__":
    main()
