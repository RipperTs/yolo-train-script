#!/usr/bin/env python3
"""
数据集目录选择功能测试脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataset_manager import dataset_directory_manager
from data_converter import DataConverter

def test_directory_validation():
    """测试目录验证功能"""
    print("🔍 测试目录验证功能...")
    
    # 测试默认目录
    default_dir = project_root / "labeling_data"
    print(f"测试默认目录: {default_dir}")
    
    validation_result = dataset_directory_manager.validate_directory(default_dir)
    print(f"验证结果: {validation_result['valid']}")
    print(f"消息: {validation_result['message']}")
    print(f"详情: {validation_result['details']}")
    
    # 测试不存在的目录
    print("\n测试不存在的目录...")
    fake_dir = project_root / "fake_directory"
    validation_result = dataset_directory_manager.validate_directory(fake_dir)
    print(f"验证结果: {validation_result['valid']}")
    print(f"消息: {validation_result['message']}")

def test_directory_suggestions():
    """测试目录建议功能"""
    print("\n📁 测试目录建议功能...")
    
    suggestions = dataset_directory_manager.get_directory_suggestions()
    print(f"找到 {len(suggestions)} 个建议目录:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")

def test_current_directory_info():
    """测试当前目录信息"""
    print("\n📊 测试当前目录信息...")
    
    info = dataset_directory_manager.get_current_directory_info()
    print(f"当前目录: {info['current_directory']}")
    print(f"是否为默认目录: {info['is_default']}")
    print(f"状态: {info['status']}")
    print(f"验证: {info['validation']}")

def test_set_directory():
    """测试设置目录功能"""
    print("\n⚙️ 测试设置目录功能...")
    
    # 获取原始目录
    original_dir = str(dataset_directory_manager.current_source_dir)
    print(f"原始目录: {original_dir}")
    
    # 测试设置为默认目录
    default_dir = str(project_root / "labeling_data")
    result = dataset_directory_manager.set_source_directory(default_dir)
    print(f"设置结果: {result['success']}")
    print(f"消息: {result['message']}")
    
    # 恢复原始目录
    dataset_directory_manager.set_source_directory(original_dir)
    print(f"已恢复到原始目录: {dataset_directory_manager.current_source_dir}")

def test_conversion_preview():
    """测试转换预览功能"""
    print("\n👁️ 测试转换预览功能...")
    
    preview = dataset_directory_manager.get_conversion_preview()
    print(f"预览状态: {preview.get('status', 'unknown')}")
    
    if preview.get('status') == 'ready':
        print(f"总文件数: {preview['total_files']}")
        print(f"数据分割预览: {preview['split_preview']}")
        print(f"源目录: {preview['source_directory']}")
        print(f"目标目录: {preview['target_directory']}")
        print(f"示例文件: {preview['sample_files']}")
    else:
        print(f"预览消息: {preview.get('message', 'No message')}")

def test_data_converter_with_custom_dir():
    """测试自定义目录的数据转换器"""
    print("\n🔄 测试自定义目录的数据转换器...")
    
    # 使用默认目录创建转换器
    default_dir = project_root / "labeling_data"
    converter = DataConverter(str(default_dir))
    
    # 检查源目录状态
    status = converter.check_source_directory()
    print(f"源目录状态: {status['status']}")
    print(f"消息: {status['message']}")
    print(f"路径: {status['path']}")
    
    if status['status'] == 'ready':
        print(f"文件数量: {status['file_count']}")
        print(f"示例文件: {status.get('files', [])}")

def test_config_persistence():
    """测试配置持久化"""
    print("\n💾 测试配置持久化...")
    
    # 获取原始配置
    original_dir = str(dataset_directory_manager.current_source_dir)
    
    # 设置新目录
    test_dir = str(project_root / "labeling_data")
    dataset_directory_manager.set_source_directory(test_dir)
    
    # 创建新的管理器实例来测试加载
    from dataset_manager import DatasetDirectoryManager
    new_manager = DatasetDirectoryManager()
    
    print(f"原始目录: {original_dir}")
    print(f"设置目录: {test_dir}")
    print(f"新实例加载的目录: {new_manager.current_source_dir}")
    
    # 验证配置是否正确保存和加载
    if str(new_manager.current_source_dir) == test_dir:
        print("✅ 配置持久化测试通过")
    else:
        print("❌ 配置持久化测试失败")

def test_empty_directory_handling():
    """测试空目录处理"""
    print("\n📂 测试空目录处理...")
    
    # 创建临时空目录
    temp_dir = project_root / "temp_empty_dir"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # 验证空目录
        validation_result = dataset_directory_manager.validate_directory(temp_dir)
        print(f"空目录验证结果: {validation_result['valid']}")
        print(f"消息: {validation_result['message']}")
        
        # 尝试设置空目录
        result = dataset_directory_manager.set_source_directory(str(temp_dir))
        print(f"设置空目录结果: {result['success']}")
        print(f"消息: {result['message']}")
        
    finally:
        # 清理临时目录
        if temp_dir.exists():
            temp_dir.rmdir()

def main():
    """主测试函数"""
    print("🚀 数据集目录选择功能测试")
    print("=" * 50)
    
    tests = [
        ("目录验证", test_directory_validation),
        ("目录建议", test_directory_suggestions),
        ("当前目录信息", test_current_directory_info),
        ("设置目录", test_set_directory),
        ("转换预览", test_conversion_preview),
        ("自定义目录转换器", test_data_converter_with_custom_dir),
        ("配置持久化", test_config_persistence),
        ("空目录处理", test_empty_directory_handling)
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
    
    # 显示最终状态
    print("\n📋 最终状态:")
    info = dataset_directory_manager.get_current_directory_info()
    print(f"当前数据源目录: {info['current_directory']}")
    print(f"目录状态: {info['status']['status']}")

if __name__ == "__main__":
    main()
