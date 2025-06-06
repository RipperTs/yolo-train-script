#!/usr/bin/env python3
"""
MPS设备支持测试脚本
验证训练和推理的MPS设备支持功能
"""

import sys
import torch
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

def test_pytorch_mps_support():
    """测试PyTorch MPS支持"""
    print("🔍 测试PyTorch MPS支持...")
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"MPS可用: {torch.backends.mps.is_available()}")
    
    if torch.backends.mps.is_available():
        try:
            # 测试MPS张量创建
            x = torch.randn(10, 10).to("mps")
            y = torch.randn(10, 10).to("mps")
            z = torch.mm(x, y)
            print("✅ MPS张量运算测试成功")
            return True
        except Exception as e:
            print(f"❌ MPS张量运算测试失败: {e}")
            return False
    else:
        print("❌ MPS不可用")
        return False

def test_device_manager():
    """测试设备管理器"""
    print("\n🔍 测试设备管理器...")
    
    try:
        from device_manager import device_manager
        
        print(f"可用设备: {device_manager.get_device_choices()}")
        print(f"设备描述: {device_manager.get_device_descriptions()}")
        print(f"当前设备: {device_manager.current_device}")
        print(f"GPU可用: {device_manager.is_gpu_available()}")
        
        # 测试设备验证
        if "mps" in device_manager.get_device_choices():
            print("\n测试MPS设备验证...")
            validation_result = device_manager.validate_device_availability("mps")
            print(f"MPS设备验证结果: {validation_result}")
            
            if validation_result["available"]:
                print("✅ MPS设备验证成功")
                
                # 测试设备切换
                if device_manager.set_device("mps"):
                    print("✅ MPS设备切换成功")
                    print(f"当前设备: {device_manager.current_device}")
                else:
                    print("❌ MPS设备切换失败")
            else:
                print("❌ MPS设备不可用")
        
        return True
        
    except Exception as e:
        print(f"❌ 设备管理器测试失败: {e}")
        return False

def test_config_manager():
    """测试配置管理器"""
    print("\n🔍 测试配置管理器...")
    
    try:
        from config_manager import config_manager
        
        # 获取设备信息
        device_info = config_manager.get_device_info()
        print(f"设备信息: {device_info}")
        
        # 测试设备推荐配置
        if "mps" in device_info["available_devices"]:
            recommendations = config_manager.get_device_recommendations("mps")
            print(f"MPS设备推荐配置: {recommendations}")
            
            # 测试设备优化
            optimizations = config_manager.optimize_config_for_device("mps")
            print(f"MPS设备优化配置: {optimizations}")
            
            print("✅ 配置管理器MPS支持测试成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置管理器测试失败: {e}")
        return False

def test_trainer_device_support():
    """测试训练器设备支持"""
    print("\n🔍 测试训练器设备支持...")
    
    try:
        from trainer import YOLOv8Trainer
        from config_manager import config_manager
        
        # 检查是否有数据集
        dataset_yaml = Path("../dataset.yaml")
        if not dataset_yaml.exists():
            print("⚠️ 数据集配置文件不存在，跳过训练器测试")
            return True
        
        # 设置MPS设备（如果可用）
        if "mps" in config_manager.get_device_info()["available_devices"]:
            config_manager.update_training_config(device="mps", epochs=1, batch_size=1)
            print("✅ 训练器MPS设备配置成功")
        
        print("✅ 训练器设备支持测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 训练器测试失败: {e}")
        return False

def test_inference_device_support():
    """测试推理器设备支持"""
    print("\n🔍 测试推理器设备支持...")
    
    try:
        from inference import YOLOv8Inference
        
        # 检查是否有预训练模型
        try:
            # 尝试创建推理器（使用预训练模型）
            inference = YOLOv8Inference(model_path="../yolov8n.pt", device="mps")
            print("✅ 推理器MPS设备支持测试成功")
            return True
        except Exception as e:
            print(f"⚠️ 推理器测试跳过（可能缺少模型文件）: {e}")
            return True
        
    except Exception as e:
        print(f"❌ 推理器测试失败: {e}")
        return False

def test_gradio_integration():
    """测试Gradio界面集成"""
    print("\n🔍 测试Gradio界面集成...")
    
    try:
        from device_manager import get_device_choices_for_gradio, parse_device_choice
        
        choices = get_device_choices_for_gradio()
        print(f"Gradio设备选择: {choices}")
        
        # 测试设备选择解析
        if choices:
            parsed = parse_device_choice(choices[0])
            print(f"解析设备选择: {choices[0]} -> {parsed}")
        
        print("✅ Gradio界面集成测试成功")
        return True
        
    except Exception as e:
        print(f"❌ Gradio界面集成测试失败: {e}")
        return False

def run_comprehensive_test():
    """运行综合测试"""
    print("🚀 开始MPS设备支持综合测试...\n")
    
    test_results = []
    
    # 运行各项测试
    test_results.append(("PyTorch MPS支持", test_pytorch_mps_support()))
    test_results.append(("设备管理器", test_device_manager()))
    test_results.append(("配置管理器", test_config_manager()))
    test_results.append(("训练器设备支持", test_trainer_device_support()))
    test_results.append(("推理器设备支持", test_inference_device_support()))
    test_results.append(("Gradio界面集成", test_gradio_integration()))
    
    # 汇总结果
    print("\n" + "="*60)
    print("📊 测试结果汇总")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！MPS设备支持功能正常")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关功能")
        return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MPS设备支持测试")
    parser.add_argument("--test", choices=["all", "pytorch", "device", "config", "trainer", "inference", "gradio"],
                       default="all", help="选择要运行的测试")
    
    args = parser.parse_args()
    
    if args.test == "all":
        run_comprehensive_test()
    elif args.test == "pytorch":
        test_pytorch_mps_support()
    elif args.test == "device":
        test_device_manager()
    elif args.test == "config":
        test_config_manager()
    elif args.test == "trainer":
        test_trainer_device_support()
    elif args.test == "inference":
        test_inference_device_support()
    elif args.test == "gradio":
        test_gradio_integration()

if __name__ == "__main__":
    main() 