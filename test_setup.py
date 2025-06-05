#!/usr/bin/env python3
"""
YOLOv8系统设置测试脚本
验证环境配置和依赖是否正确安装
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    
    try:
        import config
        print("✓ config模块导入成功")
    except ImportError as e:
        print(f"✗ config模块导入失败: {e}")
        return False
    
    try:
        from data_converter import DataConverter
        print("✓ DataConverter导入成功")
    except ImportError as e:
        print(f"✗ DataConverter导入失败: {e}")
        return False
    
    try:
        from trainer import YOLOv8Trainer
        print("✓ YOLOv8Trainer导入成功")
    except ImportError as e:
        print(f"✗ YOLOv8Trainer导入失败: {e}")
        return False
    
    try:
        from inference import YOLOv8Inference
        print("✓ YOLOv8Inference导入成功")
    except ImportError as e:
        print(f"✗ YOLOv8Inference导入失败: {e}")
        return False
    
    try:
        import utils
        print("✓ utils模块导入成功")
    except ImportError as e:
        print(f"✗ utils模块导入失败: {e}")
        return False
    
    return True


def test_dependencies():
    """测试依赖包"""
    print("\n测试依赖包...")
    
    dependencies = [
        ("ultralytics", "YOLOv8核心库"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow图像处理"),
        ("matplotlib", "绘图库"),
        ("pandas", "数据处理"),
        ("sklearn", "机器学习库"),
        ("numpy", "数值计算"),
        ("torch", "PyTorch深度学习框架")
    ]
    
    missing_deps = []
    
    for dep_name, description in dependencies:
        try:
            if dep_name == "cv2":
                import cv2
            elif dep_name == "PIL":
                from PIL import Image
            elif dep_name == "sklearn":
                import sklearn
            else:
                __import__(dep_name)
            print(f"✓ {description} ({dep_name}) 可用")
        except ImportError:
            print(f"✗ {description} ({dep_name}) 缺失")
            missing_deps.append(dep_name)
    
    if missing_deps:
        print(f"\n缺失的依赖包: {', '.join(missing_deps)}")
        print("请运行以下命令安装:")
        if "ultralytics" in missing_deps:
            print("pip install ultralytics")
        if "cv2" in missing_deps:
            print("pip install opencv-python")
        if "PIL" in missing_deps:
            print("pip install pillow")
        if "matplotlib" in missing_deps:
            print("pip install matplotlib")
        if "pandas" in missing_deps:
            print("pip install pandas")
        if "sklearn" in missing_deps:
            print("pip install scikit-learn")
        if "numpy" in missing_deps:
            print("pip install numpy")
        if "torch" in missing_deps:
            print("pip install torch torchvision")
        return False
    
    return True


def test_directories():
    """测试目录结构"""
    print("\n测试目录结构...")
    
    from config import ensure_directories, DATASETS_DIR, MODELS_DIR, YOLO_POINT_DIR
    
    # 确保目录存在
    ensure_directories()
    
    required_dirs = [
        DATASETS_DIR,
        DATASETS_DIR / "images" / "train",
        DATASETS_DIR / "images" / "val",
        DATASETS_DIR / "images" / "test",
        DATASETS_DIR / "labels" / "train",
        DATASETS_DIR / "labels" / "val", 
        DATASETS_DIR / "labels" / "test",
        MODELS_DIR,
        YOLO_POINT_DIR
    ]
    
    all_exist = True
    for directory in required_dirs:
        if directory.exists():
            print(f"✓ {directory} 存在")
        else:
            print(f"✗ {directory} 不存在")
            all_exist = False
    
    return all_exist


def test_data_availability():
    """测试数据可用性"""
    print("\n测试数据可用性...")
    
    from config import YOLO_POINT_DIR
    
    json_files = list(YOLO_POINT_DIR.glob("*.json"))
    print(f"找到 {len(json_files)} 个JSON标注文件")
    
    if len(json_files) == 0:
        print("✗ 没有找到JSON标注文件")
        print(f"请确保 {YOLO_POINT_DIR} 目录中有标注文件")
        return False
    
    # 检查前几个文件的格式
    valid_files = 0
    for i, json_file in enumerate(json_files[:5]):  # 只检查前5个文件
        try:
            import json
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            required_keys = ['shapes', 'imageWidth', 'imageHeight', 'imagePath']
            if all(key in data for key in required_keys):
                valid_files += 1
                print(f"✓ {json_file.name} 格式正确")
            else:
                print(f"✗ {json_file.name} 缺少必要字段")
        except Exception as e:
            print(f"✗ {json_file.name} 读取失败: {e}")
    
    if valid_files > 0:
        print(f"✓ 检查了 {min(5, len(json_files))} 个文件，{valid_files} 个格式正确")
        return True
    else:
        print("✗ 没有找到格式正确的JSON文件")
        return False


def test_gpu_availability():
    """测试GPU可用性"""
    print("\n测试GPU可用性...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            print(f"✓ CUDA可用")
            print(f"  GPU数量: {gpu_count}")
            print(f"  当前设备: {current_device}")
            print(f"  设备名称: {device_name}")
            
            # 测试GPU内存
            memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
            memory_cached = torch.cuda.memory_reserved(current_device) / 1024**3
            print(f"  已分配内存: {memory_allocated:.2f} GB")
            print(f"  缓存内存: {memory_cached:.2f} GB")
            
            return True
        else:
            print("✗ CUDA不可用，将使用CPU训练")
            print("  如需GPU加速，请安装CUDA和对应版本的PyTorch")
            return False
            
    except ImportError:
        print("✗ PyTorch未安装，无法检测GPU")
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("YOLOv8钢筋图纸分析系统 - 环境测试")
    print("=" * 60)
    
    tests = [
        ("模块导入", test_imports),
        ("依赖包", test_dependencies),
        ("目录结构", test_directories),
        ("数据可用性", test_data_availability),
        ("GPU可用性", test_gpu_availability)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ 测试 {test_name} 时出错: {e}")
            results[test_name] = False
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:15} : {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！系统已准备就绪。")
        print("\n下一步:")
        print("1. 运行数据转换: python run.py --pipeline convert")
        print("2. 开始训练: python run.py --pipeline train")
        print("3. 或运行完整流程: python run.py --pipeline full")
    else:
        print(f"\n⚠️  有 {total - passed} 项测试失败，请解决相关问题后重新测试。")
        
        if not results.get("依赖包", True):
            print("\n建议先安装缺失的依赖包:")
            print("pip install -r ../requirements.txt")


if __name__ == "__main__":
    main()
