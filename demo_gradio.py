#!/usr/bin/env python3
"""
Gradio界面演示脚本
展示如何使用YOLOv8训练系统的各项功能
"""

import os
import sys
import time
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def demo_data_management():
    """演示数据管理功能"""
    print("🔄 演示数据管理功能...")
    
    from data_converter import DataConverter
    from utils import check_dataset_integrity, visualize_dataset_distribution
    
    try:
        # 1. 数据转换
        print("1. 执行数据转换...")
        converter = DataConverter()
        converter.convert_all()
        print("✅ 数据转换完成")
        
        # 2. 数据集检查
        print("2. 检查数据集完整性...")
        is_valid = check_dataset_integrity()
        if is_valid:
            print("✅ 数据集检查通过")
        else:
            print("⚠️ 数据集存在问题")
        
        # 3. 数据可视化
        print("3. 生成数据分布图...")
        visualize_dataset_distribution()
        print("✅ 数据分布图已生成")
        
    except Exception as e:
        print(f"❌ 数据管理演示失败: {e}")

def demo_training():
    """演示训练功能"""
    print("🎯 演示训练功能...")
    
    from trainer import YOLOv8Trainer
    from smart_trainer import SmartTrainer
    
    try:
        # 1. 快速训练测试（5个epoch）
        print("1. 执行快速训练测试...")
        trainer = YOLOv8Trainer()
        
        # 修改配置为快速测试
        from config import TRAINING_CONFIG
        test_config = TRAINING_CONFIG.copy()
        test_config['epochs'] = 5
        test_config['batch_size'] = 4  # 减小批次大小
        
        print("⚠️ 注意: 这是演示模式，只训练5个epoch")
        print("实际使用时请根据需要调整训练轮数")
        
        # 这里只是演示，不实际执行训练
        print("✅ 训练配置已准备就绪")
        
    except Exception as e:
        print(f"❌ 训练演示失败: {e}")

def demo_inference():
    """演示推理功能"""
    print("🔍 演示推理功能...")
    
    from inference import YOLOv8Inference
    from config import DATASETS_DIR
    
    try:
        # 查找测试图片
        test_images_dir = DATASETS_DIR / "images" / "test"
        if test_images_dir.exists():
            test_images = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
            
            if test_images:
                print(f"找到 {len(test_images)} 张测试图片")
                
                # 检查是否有训练好的模型
                from config import MODELS_DIR
                model_files = list(MODELS_DIR.glob("**/*.pt"))
                
                if model_files:
                    print(f"找到 {len(model_files)} 个模型文件")
                    print("✅ 推理环境准备就绪")
                else:
                    print("⚠️ 没有找到训练好的模型，请先进行训练")
            else:
                print("⚠️ 没有找到测试图片")
        else:
            print("⚠️ 测试图片目录不存在，请先进行数据转换")
            
    except Exception as e:
        print(f"❌ 推理演示失败: {e}")

def demo_monitoring():
    """演示监控功能"""
    print("📈 演示监控功能...")
    
    from gradio_utils import training_monitor, log_monitor
    
    try:
        # 1. 检查训练状态
        print("1. 检查训练状态...")
        status = training_monitor.get_training_status()
        print(f"训练状态: {status.get('status', 'unknown')}")
        
        # 2. 查找日志文件
        print("2. 查找日志文件...")
        log_file = log_monitor.find_latest_log()
        if log_file:
            print(f"找到日志文件: {log_file}")
        else:
            print("没有找到日志文件")
        
        # 3. 生成训练曲线
        print("3. 尝试生成训练曲线...")
        plot_data = training_monitor.generate_training_plot()
        if plot_data:
            print("✅ 训练曲线生成成功")
        else:
            print("⚠️ 没有训练数据，无法生成曲线")
            
    except Exception as e:
        print(f"❌ 监控演示失败: {e}")

def demo_config_management():
    """演示配置管理功能"""
    print("⚙️ 演示配置管理功能...")
    
    from config_manager import config_manager
    
    try:
        # 1. 显示当前配置
        print("1. 当前配置摘要:")
        summary = config_manager.get_config_summary()
        print(summary)
        
        # 2. 更新配置
        print("2. 更新训练配置...")
        config_manager.update_training_config(
            epochs=50,
            batch_size=8,
            learning_rate=0.005
        )
        print("✅ 配置更新完成")
        
        # 3. 导出配置
        print("3. 导出配置到文件...")
        export_path = "demo_config.json"
        if config_manager.export_config(export_path):
            print(f"✅ 配置已导出到: {export_path}")
        else:
            print("❌ 配置导出失败")
            
    except Exception as e:
        print(f"❌ 配置管理演示失败: {e}")

def demo_tools():
    """演示工具功能"""
    print("🛠️ 演示工具功能...")
    
    try:
        # 1. 系统信息
        print("1. 获取系统信息...")
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        print(f"CPU使用率: {cpu_percent}%")
        print(f"内存使用率: {memory.percent}%")
        print(f"可用内存: {memory.available / (1024**3):.2f} GB")
        
        # 2. 环境检查
        print("2. 检查Python环境...")
        import torch
        import ultralytics
        
        print(f"Python版本: {sys.version}")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"Ultralytics版本: {ultralytics.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        
        print("✅ 环境检查完成")
        
    except Exception as e:
        print(f"❌ 工具演示失败: {e}")

def main():
    """主演示函数"""
    print("🚀 YOLOv8训练系统Gradio界面演示")
    print("=" * 50)
    
    # 检查项目结构
    print("📁 检查项目结构...")
    required_files = [
        "config.py",
        "data_converter.py", 
        "trainer.py",
        "inference.py",
        "gradio_app.py",
        "gradio_utils.py",
        "config_manager.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少文件: {missing_files}")
        return
    
    print("✅ 项目结构检查通过")
    print()
    
    # 执行各项演示
    demos = [
        ("数据管理", demo_data_management),
        ("配置管理", demo_config_management),
        ("监控功能", demo_monitoring),
        ("推理功能", demo_inference),
        ("工具功能", demo_tools),
        ("训练功能", demo_training),  # 放在最后，因为可能耗时较长
    ]
    
    for name, demo_func in demos:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            demo_func()
        except KeyboardInterrupt:
            print("\n👋 用户中断演示")
            break
        except Exception as e:
            print(f"❌ {name}演示出错: {e}")
        
        print(f"{'='*50}")
        time.sleep(1)  # 短暂暂停
    
    print("\n🎉 演示完成！")
    print("\n💡 提示:")
    print("- 运行 'python start_gradio.py' 启动Web界面")
    print("- 访问 http://localhost:7860 使用图形界面")
    print("- 查看 GRADIO_README.md 了解详细使用说明")

if __name__ == "__main__":
    main()
