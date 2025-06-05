#!/usr/bin/env python3
"""
YOLOv8 主运行脚本
提供完整的训练和推理流程
"""

import sys
import argparse
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from data_converter import DataConverter
from trainer import YOLOv8Trainer
from inference import YOLOv8Inference
from utils import (
    check_dataset_integrity, 
    visualize_dataset_distribution,
    visualize_annotations
)
from config import ensure_directories


def setup_environment():
    """设置环境"""
    print("设置YOLOv8环境...")
    ensure_directories()
    print("环境设置完成")


def convert_data():
    """转换数据"""
    print("=" * 50)
    print("开始数据转换...")
    print("=" * 50)
    
    converter = DataConverter()
    converter.convert_all()
    
    print("\n数据转换完成，开始检查数据集完整性...")
    if check_dataset_integrity():
        print("数据集检查通过!")
        visualize_dataset_distribution()
    else:
        print("数据集存在问题，请检查后重新转换")
        return False
    
    return True


def train_model(resume=False, model_path=None):
    """训练模型"""
    print("=" * 50)
    print("开始模型训练...")
    print("=" * 50)
    
    trainer = YOLOv8Trainer()
    trainer.get_device_info()
    
    try:
        results = trainer.train(resume=resume, resume_path=model_path)
        print("训练完成!")
        
        # 验证模型
        print("\n开始验证模型...")
        trainer.validate()
        
        return True
        
    except Exception as e:
        print(f"训练失败: {e}")
        return False


def run_inference(model_path=None, image_path=None, images_dir=None):
    """运行推理"""
    print("=" * 50)
    print("开始模型推理...")
    print("=" * 50)
    
    try:
        inference = YOLOv8Inference(model_path=model_path)
        
        if image_path:
            # 单张图片推理
            result = inference.predict_image(image_path)
            print(f"检测结果: {result['num_detections']} 个目标")
            
            # 可视化结果
            if result['predictions']:
                output_path = inference.visualize_predictions(
                    image_path, result['predictions']
                )
                print(f"可视化结果已保存: {output_path}")
        
        elif images_dir:
            # 批量推理
            image_dir = Path(images_dir)
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(image_dir.glob(ext))
            
            if not image_files:
                print(f"在 {image_dir} 中没有找到图片文件")
                return False
            
            results = inference.predict_batch([str(f) for f in image_files])
            total_detections = sum(r['num_detections'] for r in results)
            print(f"批量推理完成，共检测到 {total_detections} 个目标")
            
            # 保存结果
            json_path = image_dir / "predictions.json"
            inference.save_predictions_json(results, str(json_path))
        
        else:
            print("请指定 --image 或 --images 参数")
            return False
        
        return True
        
    except Exception as e:
        print(f"推理失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="YOLOv8钢筋图纸分析完整流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 完整流程 (数据转换 + 训练 + 推理)
  python run.py --pipeline full
  
  # 只转换数据
  python run.py --pipeline convert
  
  # 只训练模型
  python run.py --pipeline train
  
  # 只运行推理
  python run.py --pipeline inference --image path/to/image.jpg
  
  # 恢复训练
  python run.py --pipeline train --resume
  
  # 使用指定模型推理
  python run.py --pipeline inference --model path/to/model.pt --images path/to/images/
        """
    )
    
    parser.add_argument(
        "--pipeline", 
        choices=["full", "convert", "train", "inference"],
        required=True,
        help="执行的流程"
    )
    
    parser.add_argument("--model", type=str, help="模型路径")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--images", type=str, help="图片目录路径")
    parser.add_argument("--resume", action="store_true", help="恢复训练")
    parser.add_argument("--skip-check", action="store_true", help="跳过数据集检查")
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    
    success = True
    
    if args.pipeline == "full":
        # 完整流程
        print("执行完整流程: 数据转换 -> 训练 -> 推理")
        
        # 1. 数据转换
        if not convert_data():
            print("数据转换失败，流程终止")
            return
        
        # 2. 训练模型
        if not train_model():
            print("模型训练失败，流程终止")
            return
        
        # 3. 推理测试 (使用测试集中的一张图片)
        from config import DATASETS_DIR
        test_images_dir = DATASETS_DIR / "images" / "test"
        if test_images_dir.exists():
            test_images = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
            if test_images:
                print(f"\n使用测试图片进行推理验证: {test_images[0]}")
                run_inference(image_path=str(test_images[0]))
        
    elif args.pipeline == "convert":
        # 只转换数据
        success = convert_data()
        
    elif args.pipeline == "train":
        # 只训练模型
        if not args.skip_check:
            print("检查数据集...")
            if not check_dataset_integrity():
                print("数据集存在问题，请先运行数据转换")
                return
        
        success = train_model(resume=args.resume, model_path=args.model)
        
    elif args.pipeline == "inference":
        # 只运行推理
        success = run_inference(
            model_path=args.model,
            image_path=args.image,
            images_dir=args.images
        )
    
    if success:
        print("\n" + "=" * 50)
        print("流程执行成功!")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("流程执行失败!")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()
