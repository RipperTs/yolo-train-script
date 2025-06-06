#!/usr/bin/env python3
"""
简化的恢复训练脚本
用于在100个epoch后继续训练
"""

import sys
from pathlib import Path
import pandas as pd

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from trainer import YOLOv8Trainer
from config import MODELS_DIR


def find_latest_model():
    """查找最新的训练模型"""
    train_dirs = list(MODELS_DIR.glob("train_*"))
    if not train_dirs:
        print("❌ 没有找到训练结果")
        return None, None
    
    latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
    best_model = latest_dir / "weights" / "best.pt"
    last_model = latest_dir / "weights" / "last.pt"
    
    print(f"📁 最新训练目录: {latest_dir}")
    print(f"🏆 最佳模型: {best_model}")
    print(f"📄 最新模型: {last_model}")
    
    return latest_dir, best_model if best_model.exists() else last_model


def analyze_current_performance():
    """分析当前训练性能"""
    latest_dir, _ = find_latest_model()
    if not latest_dir:
        return None
    
    results_file = latest_dir / "results.csv"
    if not results_file.exists():
        print("❌ 找不到训练结果文件")
        return None
    
    df = pd.read_csv(results_file)
    if len(df) == 0:
        print("❌ 训练结果文件为空")
        return None
    
    latest = df.iloc[-1]
    
    print("\n📊 当前训练性能:")
    print("="*40)
    print(f"已训练轮数: {len(df)} epochs")
    
    # 显示主要指标
    metrics = {
        'Box Loss': latest.get('train/box_loss', 'N/A'),
        'Class Loss': latest.get('train/cls_loss', 'N/A'),
        'DFL Loss': latest.get('train/dfl_loss', 'N/A'),
        'mAP50': latest.get('metrics/mAP50(B)', 'N/A'),
        'mAP50-95': latest.get('metrics/mAP50-95(B)', 'N/A')
    }
    
    for name, value in metrics.items():
        if value != 'N/A':
            print(f"{name:12}: {value:.4f}")
        else:
            print(f"{name:12}: {value}")
    
    # 评估是否需要继续训练
    print("\n🎯 性能评估:")
    
    box_loss = latest.get('train/box_loss', float('inf'))
    cls_loss = latest.get('train/cls_loss', float('inf'))
    map50 = latest.get('metrics/mAP50(B)', 0)
    
    suggestions = []
    
    if box_loss > 0.1:
        suggestions.append("📍 Box Loss还有下降空间，建议继续训练")
    else:
        suggestions.append("✅ Box Loss已达到良好水平")
    
    if cls_loss > 2.0:
        suggestions.append("📚 Class Loss偏高，建议继续训练")
    else:
        suggestions.append("✅ Class Loss已达到良好水平")
    
    if map50 < 0.7:
        suggestions.append("🎯 mAP50还有提升空间，建议继续训练")
    else:
        suggestions.append("✅ mAP50已达到优秀水平")
    
    for suggestion in suggestions:
        print(f"  {suggestion}")
    
    # 检查最近的改善趋势
    if len(df) >= 10:
        recent_box_loss = df['train/box_loss'].tail(10).mean()
        earlier_box_loss = df['train/box_loss'].head(10).mean()
        improvement = earlier_box_loss - recent_box_loss
        
        print(f"\n📈 最近10轮改善情况:")
        print(f"  Box Loss改善: {improvement:.4f}")
        
        if improvement > 0.01:
            print("  ✅ 模型还在显著改善，建议继续训练")
        elif improvement > 0.001:
            print("  🟡 模型改善缓慢，可以继续训练")
        else:
            print("  ⚠️ 模型改善很少，可能接近收敛")
    
    return latest


def resume_training_with_more_epochs(additional_epochs=50, use_best_model=True):
    """恢复训练并增加更多epochs"""
    print(f"\n🚀 准备继续训练 {additional_epochs} 个epochs...")

    # 找到最新模型
    latest_dir, model_path = find_latest_model()
    if not model_path or not model_path.exists():
        print("❌ 找不到可用的模型文件")
        return False

    print(f"📂 使用模型: {model_path}")

    try:
        # 初始化训练器
        trainer = YOLOv8Trainer()

        # 检查训练是否已完成
        args_file = latest_dir / "args.yaml"
        if args_file.exists():
            import yaml
            with open(args_file, 'r') as f:
                args = yaml.safe_load(f)
            completed_epochs = args.get('epochs', 0)

            # 检查results.csv来确认实际完成的epochs
            results_file = latest_dir / "results.csv"
            if results_file.exists():
                import pandas as pd
                df = pd.read_csv(results_file)
                actual_epochs = len(df) - 1  # 减去header行

                print(f"📊 训练状态:")
                print(f"   - 设定epochs: {completed_epochs}")
                print(f"   - 实际完成: {actual_epochs}")

                if actual_epochs >= completed_epochs:
                    print("⚠️ 检测到训练已完成，将开始新的训练而不是恢复训练")
                    return start_new_training_from_model(model_path, additional_epochs)

        # 尝试恢复训练
        print("开始恢复训练...")
        success = trainer.train(resume=True, resume_path=str(model_path))

        if success:
            print("✅ 恢复训练完成！")

            # 分析新的结果
            print("\n分析新的训练结果...")
            analyze_current_performance()

            return True
        else:
            print("❌ 恢复训练失败")
            return False

    except Exception as e:
        error_msg = str(e)
        if "nothing to resume" in error_msg or "is finished" in error_msg:
            print(f"⚠️ 训练已完成，无法恢复。将开始新的训练: {e}")
            return start_new_training_from_model(model_path, additional_epochs)
        else:
            print(f"❌ 恢复训练时出错: {e}")
            return False


def start_new_training_from_model(model_path, epochs=50):
    """从已有模型开始新的训练"""
    print(f"\n🆕 从已有模型开始新的训练 ({epochs} epochs)...")

    try:
        from config_manager import config_manager

        # 更新配置
        config_manager.update_training_config(epochs=epochs)

        # 初始化训练器
        trainer = YOLOv8Trainer()

        # 使用已有模型作为预训练模型开始新训练
        from ultralytics import YOLO
        trainer.model = YOLO(str(model_path))
        print(f"📂 使用预训练模型: {model_path}")

        # 开始训练（不使用resume参数）
        success = trainer.train(resume=False)

        if success:
            print("✅ 新训练完成！")

            # 分析新的结果
            print("\n分析新的训练结果...")
            analyze_current_performance()

            return True
        else:
            print("❌ 新训练失败")
            return False

    except Exception as e:
        print(f"❌ 新训练时出错: {e}")
        return False


def interactive_resume():
    """交互式恢复训练"""
    print("🤖 交互式训练助手")
    print("="*50)

    # 分析当前状态
    current_metrics = analyze_current_performance()
    if not current_metrics:
        print("❌ 无法分析当前状态，请先完成初始训练")
        return

    # 询问用户选择操作
    print("\n❓ 请选择您要执行的操作:")
    print("1. 继续训练 (从上次停止的地方继续)")
    print("2. 开始新训练 (使用最佳模型作为起点)")
    print("3. 强制全新训练 (从预训练模型重新开始)")
    print("4. 退出")

    action_choice = input("选择操作 (1-4): ").strip()

    if action_choice == '4':
        print("👋 退出训练")
        return

    # 询问epochs数量
    epochs_input = input("输入训练轮数 (默认50): ").strip()
    if epochs_input:
        try:
            additional_epochs = int(epochs_input)
            if additional_epochs <= 0:
                print("❌ Epoch数量必须大于0")
                return
        except ValueError:
            print("❌ 无效输入")
            return
    else:
        additional_epochs = 50

    print(f"\n🎯 将训练 {additional_epochs} 个epochs")

    if action_choice == '1':
        # 继续训练
        print("\n选择恢复训练的模型:")
        print("1. best.pt (推荐 - 验证性能最好的模型)")
        print("2. last.pt (最新的模型)")

        model_choice = input("选择 (1/2): ").strip()
        use_best = model_choice != '2'

        success = resume_training_with_more_epochs(additional_epochs, use_best)

    elif action_choice == '2':
        # 从最佳模型开始新训练
        latest_dir, model_path = find_latest_model()
        if model_path and model_path.exists():
            success = start_new_training_from_model(model_path, additional_epochs)
        else:
            print("❌ 找不到可用的模型文件")
            return

    elif action_choice == '3':
        # 强制全新训练
        success = force_new_training(additional_epochs)

    else:
        print("❌ 无效选择")
        return

    if success:
        print("\n🎉 训练完成！")
        print("💡 提示: 您可以重复运行此脚本来继续训练，直到达到满意的效果")
    else:
        print("\n❌ 训练失败")


def force_new_training(epochs=100):
    """强制开始新的训练"""
    print(f"\n🔄 强制开始新的训练 ({epochs} epochs)...")

    try:
        from config_manager import config_manager

        # 更新配置
        config_manager.update_training_config(epochs=epochs)

        # 初始化训练器
        trainer = YOLOv8Trainer()

        # 开始全新训练
        success = trainer.train(resume=False)

        if success:
            print("✅ 新训练完成！")

            # 分析新的结果
            print("\n分析新的训练结果...")
            analyze_current_performance()

            return True
        else:
            print("❌ 新训练失败")
            return False

    except Exception as e:
        print(f"❌ 新训练时出错: {e}")
        return False


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="恢复训练脚本")
    parser.add_argument("--action", choices=["analyze", "resume", "interactive", "new"],
                       default="interactive", help="执行的操作")
    parser.add_argument("--epochs", type=int, default=50, help="继续训练的轮数")
    parser.add_argument("--auto", action="store_true", help="自动恢复训练，不询问")
    parser.add_argument("--force", action="store_true", help="强制开始新训练")

    args = parser.parse_args()

    if args.action == "analyze":
        # 只分析当前状态
        analyze_current_performance()

    elif args.action == "resume":
        # 直接恢复训练
        if args.auto:
            resume_training_with_more_epochs(args.epochs)
        else:
            print(f"准备继续训练 {args.epochs} 个epochs")
            choice = input("确认继续？(y/n): ").strip().lower()
            if choice == 'y':
                resume_training_with_more_epochs(args.epochs)
            else:
                print("取消训练")

    elif args.action == "new":
        # 强制开始新训练
        if args.auto:
            force_new_training(args.epochs)
        else:
            print(f"准备开始新的训练 {args.epochs} 个epochs")
            choice = input("确认开始？(y/n): ").strip().lower()
            if choice == 'y':
                force_new_training(args.epochs)
            else:
                print("取消训练")

    elif args.action == "interactive":
        # 交互式模式
        interactive_resume()


if __name__ == "__main__":
    main()
