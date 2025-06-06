#!/usr/bin/env python3
"""
测试Gradio智能训练功能
专门用于测试和修复智能训练中的"nothing to resume"错误
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from smart_trainer import SmartTrainer


def test_smart_training():
    """测试智能训练功能"""
    print("🧪 测试智能训练功能")
    print("="*50)
    
    # 创建智能训练器
    smart_trainer = SmartTrainer(
        target_loss_thresholds={
            'box_loss': 0.05,
            'cls_loss': 1.0,
            'dfl_loss': 0.8,
            'map50': 0.7
        },
        patience=10,
        min_improvement=0.001
    )
    
    # 测试分析功能
    print("\n📊 测试训练状态分析...")
    latest_dir = smart_trainer.find_latest_training()
    if latest_dir:
        print(f"找到最新训练目录: {latest_dir}")
        
        results_file = latest_dir / "results.csv"
        if results_file.exists():
            analysis = smart_trainer.analyze_training_progress(results_file)
            if analysis:
                smart_trainer.print_training_status(analysis)
                
                should_continue, reason = smart_trainer.should_continue_training(analysis)
                print(f"\n决策: {reason}")
            else:
                print("❌ 无法分析训练进度")
        else:
            print("❌ 没有找到results.csv文件")
    else:
        print("❌ 没有找到训练结果")
    
    # 测试继续训练功能
    print("\n🚀 测试继续训练功能...")
    try:
        success = smart_trainer.continue_training(additional_epochs=10)
        if success:
            print("✅ 继续训练测试成功")
        else:
            print("❌ 继续训练测试失败")
    except Exception as e:
        print(f"❌ 继续训练测试出错: {e}")


def test_resume_error_handling():
    """专门测试恢复训练错误处理"""
    print("\n🔧 测试恢复训练错误处理")
    print("="*50)
    
    from trainer import YOLOv8Trainer
    from config import MODELS_DIR
    
    # 查找最新的训练结果
    train_dirs = list(MODELS_DIR.glob("train_*"))
    if not train_dirs:
        print("❌ 没有找到训练结果")
        return
    
    latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
    best_model = latest_dir / "weights" / "best.pt"
    last_model = latest_dir / "weights" / "last.pt"
    
    print(f"📁 最新训练目录: {latest_dir}")
    print(f"🏆 最佳模型存在: {best_model.exists()}")
    print(f"📄 最新模型存在: {last_model.exists()}")
    
    # 测试直接恢复训练
    trainer = YOLOv8Trainer()
    
    print("\n🔄 测试直接恢复训练...")
    try:
        # 尝试恢复训练
        results = trainer.train(resume=True)
        print("✅ 恢复训练成功")
    except Exception as e:
        error_msg = str(e)
        print(f"❌ 恢复训练失败: {e}")
        
        if "nothing to resume" in error_msg or "is finished" in error_msg:
            print("🎯 检测到'nothing to resume'错误，这是预期的")
            
            # 测试使用最佳模型开始新训练
            if best_model.exists():
                print("\n🆕 测试使用最佳模型开始新训练...")
                try:
                    from ultralytics import YOLO
                    trainer.model = YOLO(str(best_model))
                    
                    # 更新配置为较少的epochs用于测试
                    from config_manager import config_manager
                    config_manager.update_training_config(epochs=2)
                    
                    results = trainer.train(resume=False)
                    print("✅ 新训练会话成功")
                except Exception as new_error:
                    print(f"❌ 新训练会话失败: {new_error}")
            else:
                print("❌ 没有找到最佳模型文件")
        else:
            print("❌ 其他类型的错误")


def test_smart_training_loop():
    """测试智能训练循环"""
    print("\n🤖 测试智能训练循环")
    print("="*50)
    
    # 创建智能训练器，设置较低的目标用于测试
    smart_trainer = SmartTrainer(
        target_loss_thresholds={
            'box_loss': 0.1,    # 较宽松的目标
            'cls_loss': 2.0,
            'dfl_loss': 1.0,
            'map50': 0.5
        },
        patience=5,
        min_improvement=0.001
    )
    
    print("🎯 使用较宽松的目标阈值进行测试")
    print(f"目标阈值: {smart_trainer.target_loss_thresholds}")
    
    try:
        # 运行智能训练循环，但限制较少的epochs
        result_dir = smart_trainer.smart_training_loop(
            initial_epochs=5,    # 较少的初始epochs
            continue_epochs=3,   # 较少的继续epochs
            max_total_epochs=10  # 较少的最大epochs
        )
        
        if result_dir:
            print(f"✅ 智能训练循环完成，结果保存在: {result_dir}")
        else:
            print("❌ 智能训练循环失败")
            
    except Exception as e:
        print(f"❌ 智能训练循环出错: {e}")


def main():
    """主函数"""
    print("🧪 Gradio智能训练功能测试")
    print("="*60)
    
    # 测试1: 基本功能测试
    test_smart_training()
    
    # 测试2: 恢复训练错误处理
    test_resume_error_handling()
    
    # 测试3: 智能训练循环（可选，因为会实际训练）
    print("\n❓ 是否要测试智能训练循环？（这会实际进行训练）")
    choice = input("输入 'y' 继续，其他键跳过: ").strip().lower()
    if choice == 'y':
        test_smart_training_loop()
    else:
        print("⏭️ 跳过智能训练循环测试")
    
    print("\n🎉 测试完成！")


if __name__ == "__main__":
    main()
