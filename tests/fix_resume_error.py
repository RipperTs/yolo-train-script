#!/usr/bin/env python3
"""
解决 "nothing to resume" 错误的脚本
当YOLO训练完成后无法恢复训练时使用此脚本
"""

import sys
from pathlib import Path
import yaml

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from trainer import YOLOv8Trainer
from config import MODELS_DIR
from config_manager import config_manager


def find_latest_training():
    """查找最新的训练结果"""
    train_dirs = list(MODELS_DIR.glob("train_*"))
    if not train_dirs:
        return None
    
    return max(train_dirs, key=lambda x: x.stat().st_mtime)


def check_training_status(train_dir):
    """检查训练状态"""
    if not train_dir or not train_dir.exists():
        return None
    
    args_file = train_dir / "args.yaml"
    results_file = train_dir / "results.csv"
    
    if not args_file.exists() or not results_file.exists():
        return None
    
    # 读取训练参数
    with open(args_file, 'r') as f:
        args = yaml.safe_load(f)
    
    planned_epochs = args.get('epochs', 0)
    
    # 读取实际完成的epochs
    import pandas as pd
    df = pd.read_csv(results_file)
    actual_epochs = len(df) - 1  # 减去header行
    
    return {
        'train_dir': train_dir,
        'planned_epochs': planned_epochs,
        'actual_epochs': actual_epochs,
        'is_completed': actual_epochs >= planned_epochs,
        'best_model': train_dir / "weights" / "best.pt",
        'last_model': train_dir / "weights" / "last.pt"
    }


def continue_training_new_session(model_path, epochs=100):
    """开始新的训练会话（不使用resume）"""
    print(f"🚀 开始新的训练会话: {epochs} epochs")
    print(f"📂 使用模型: {model_path}")

    try:
        # 更新配置
        config_manager.update_training_config(epochs=epochs)

        # 初始化训练器
        trainer = YOLOv8Trainer()

        # 使用指定模型作为预训练模型
        from ultralytics import YOLO
        trainer.model = YOLO(str(model_path))

        # 开始训练（不使用resume）
        results = trainer.train(resume=False)

        print("✅ 训练完成！")
        print(f"📁 结果保存在: {results.save_dir}")

        return True

    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return False


def main():
    """主函数"""
    print("🔧 YOLO训练恢复错误修复工具")
    print("="*50)
    
    # 查找最新训练
    latest_dir = find_latest_training()
    if not latest_dir:
        print("❌ 没有找到训练结果")
        return
    
    # 检查训练状态
    status = check_training_status(latest_dir)
    if not status:
        print("❌ 无法读取训练状态")
        return
    
    print(f"📊 训练状态分析:")
    print(f"   训练目录: {status['train_dir']}")
    print(f"   计划epochs: {status['planned_epochs']}")
    print(f"   实际完成: {status['actual_epochs']}")
    print(f"   是否完成: {'是' if status['is_completed'] else '否'}")
    
    if status['is_completed']:
        print("\n✅ 训练已完成，这就是为什么无法恢复的原因")
        print("💡 解决方案: 开始新的训练会话")
        
        # 检查模型文件
        best_exists = status['best_model'].exists()
        last_exists = status['last_model'].exists()
        
        print(f"\n📁 可用模型:")
        print(f"   best.pt: {'✅' if best_exists else '❌'}")
        print(f"   last.pt: {'✅' if last_exists else '❌'}")
        
        if not (best_exists or last_exists):
            print("❌ 没有找到可用的模型文件")
            return
        
        # 选择模型
        if best_exists and last_exists:
            print("\n选择要使用的模型:")
            print("1. best.pt (推荐 - 验证性能最好)")
            print("2. last.pt (最新的检查点)")
            choice = input("选择 (1/2): ").strip()
            model_path = status['best_model'] if choice != '2' else status['last_model']
        elif best_exists:
            model_path = status['best_model']
            print(f"\n使用 best.pt 模型")
        else:
            model_path = status['last_model']
            print(f"\n使用 last.pt 模型")
        
        # 询问epochs数量
        epochs_input = input("\n输入新训练的epochs数量 (默认100): ").strip()
        if epochs_input:
            try:
                epochs = int(epochs_input)
                if epochs <= 0:
                    print("❌ Epochs必须大于0")
                    return
            except ValueError:
                print("❌ 无效输入")
                return
        else:
            epochs = 100
        
        # 确认开始训练
        print(f"\n🎯 准备开始新的训练:")
        print(f"   模型: {model_path.name}")
        print(f"   Epochs: {epochs}")
        
        confirm = input("确认开始？(y/n): ").strip().lower()
        if confirm == 'y':
            success = continue_training_new_session(model_path, epochs)
            if success:
                print("\n🎉 问题已解决！新的训练已完成")
            else:
                print("\n❌ 训练失败，请检查错误信息")
        else:
            print("取消训练")
    
    else:
        print("\n⚠️ 训练未完成，但仍然无法恢复")
        print("💡 建议: 尝试开始新的训练会话")
        
        # 提供选项
        print("\n选择操作:")
        print("1. 开始新的训练会话")
        print("2. 退出")
        
        choice = input("选择 (1/2): ").strip()
        if choice == '1':
            model_path = status['last_model'] if status['last_model'].exists() else status['best_model']
            if model_path.exists():
                epochs = 100
                success = continue_training_new_session(model_path, epochs)
                if success:
                    print("\n🎉 问题已解决！")
            else:
                print("❌ 没有找到可用的模型文件")


if __name__ == "__main__":
    main()
