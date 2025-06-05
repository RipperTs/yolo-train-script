#!/usr/bin/env python3
"""
设备管理器测试脚本
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def test_device_detection():
    """测试设备检测"""
    print("🔍 测试设备检测...")
    
    from device_manager import device_manager
    
    print(f"当前设备: {device_manager.current_device}")
    print(f"GPU可用: {device_manager.is_gpu_available()}")
    print(f"可用设备数量: {len(device_manager.available_devices)}")
    
    print("\n📋 设备列表:")
    for i, device in enumerate(device_manager.available_devices):
        print(f"  {i+1}. {device['description']}")
        print(f"     ID: {device['id']}")
        print(f"     类型: {device['type']}")
        print(f"     内存: {device['memory']}")
        print()

def test_device_choices():
    """测试Gradio设备选择"""
    print("🎛️ 测试Gradio设备选择...")
    
    from device_manager import get_device_choices_for_gradio, parse_device_choice
    
    choices = get_device_choices_for_gradio()
    print(f"Gradio选择项: {choices}")
    
    for choice in choices:
        device_id = parse_device_choice(choice)
        print(f"  '{choice}' -> '{device_id}'")

def test_device_switching():
    """测试设备切换"""
    print("🔄 测试设备切换...")
    
    from device_manager import device_manager
    
    original_device = device_manager.current_device
    print(f"原始设备: {original_device}")
    
    # 尝试切换到每个可用设备
    for device in device_manager.available_devices:
        device_id = device['id']
        print(f"\n尝试切换到: {device_id}")
        
        success = device_manager.set_device(device_id)
        if success:
            print(f"✅ 成功切换到: {device_manager.current_device}")
            
            # 获取推荐配置
            batch_size = device_manager.get_optimal_batch_size(device_id)
            print(f"   推荐批次大小: {batch_size}")
            
            # 验证兼容性
            compatibility = device_manager.validate_device_compatibility(device_id)
            print(f"   兼容性: {compatibility['compatible']}")
            if compatibility['warnings']:
                print(f"   警告: {compatibility['warnings']}")
            if compatibility['recommendations']:
                print(f"   建议: {compatibility['recommendations']}")
        else:
            print(f"❌ 切换失败")
    
    # 恢复原始设备
    device_manager.set_device(original_device)
    print(f"\n🔙 恢复到原始设备: {device_manager.current_device}")

def test_config_integration():
    """测试配置管理器集成"""
    print("⚙️ 测试配置管理器集成...")
    
    from config_manager import config_manager
    
    # 获取设备信息
    device_info = config_manager.get_device_info()
    print("设备信息:")
    for key, value in device_info.items():
        if key != "device_status":  # 跳过复杂的状态信息
            print(f"  {key}: {value}")
    
    # 测试设备更新
    available_devices = device_info['available_devices']
    if len(available_devices) > 1:
        test_device = available_devices[1]  # 选择第二个设备
        print(f"\n测试切换到设备: {test_device}")
        
        success = config_manager.update_device(test_device)
        if success:
            print("✅ 配置管理器设备切换成功")
            
            # 获取推荐设置
            recommendations = config_manager.get_device_recommendations(test_device)
            print(f"推荐设置: {recommendations}")
        else:
            print("❌ 配置管理器设备切换失败")

def test_pytorch_integration():
    """测试PyTorch集成"""
    print("🔥 测试PyTorch集成...")
    
    import torch
    from device_manager import device_manager
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 测试MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ MPS (Apple Metal) 可用")
    else:
        print("❌ MPS (Apple Metal) 不可用")
    
    # 测试张量创建
    print("\n🧮 测试张量创建:")
    for device in device_manager.available_devices:
        device_id = device['id']
        try:
            if device_id == "cpu":
                tensor = torch.randn(10, 10)
            else:
                tensor = torch.randn(10, 10).to(device_id)
            
            print(f"✅ {device_id}: 张量创建成功 - {tensor.device}")
        except Exception as e:
            print(f"❌ {device_id}: 张量创建失败 - {e}")

def main():
    """主测试函数"""
    print("🚀 设备管理器测试")
    print("=" * 50)
    
    tests = [
        ("设备检测", test_device_detection),
        ("Gradio选择", test_device_choices),
        ("设备切换", test_device_switching),
        ("配置集成", test_config_integration),
        ("PyTorch集成", test_pytorch_integration)
    ]
    
    for name, test_func in tests:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            test_func()
        except Exception as e:
            print(f"❌ {name}测试失败: {e}")
            import traceback
            traceback.print_exc()
        print("=" * 50)
    
    print("\n🎉 测试完成！")

if __name__ == "__main__":
    main()
