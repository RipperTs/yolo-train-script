#!/usr/bin/env python3
"""
训练监控脚本
实时监控训练进度和loss变化
"""

import sys
import time
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from config import MODELS_DIR


def parse_training_log(log_file_path):
    """解析训练日志文件"""
    if not Path(log_file_path).exists():
        return None
    
    epochs = []
    box_losses = []
    cls_losses = []
    dfl_losses = []
    
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if '/5' in line and 'box_loss' in line:  # 匹配训练行
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.endswith('/5'):  # 找到epoch信息
                        epoch = int(part.split('/')[0])
                        epochs.append(epoch)
                        
                        # 查找loss值
                        for j in range(i, len(parts)):
                            try:
                                if '.' in parts[j] and len(parts[j]) < 10:
                                    loss_val = float(parts[j])
                                    if len(box_losses) == len(epochs) - 1:
                                        box_losses.append(loss_val)
                                    elif len(cls_losses) == len(epochs) - 1:
                                        cls_losses.append(loss_val)
                                    elif len(dfl_losses) == len(epochs) - 1:
                                        dfl_losses.append(loss_val)
                                        break
                            except ValueError:
                                continue
                        break
    except Exception as e:
        print(f"解析日志文件出错: {e}")
        return None
    
    if epochs:
        return {
            'epochs': epochs,
            'box_loss': box_losses,
            'cls_loss': cls_losses,
            'dfl_loss': dfl_losses
        }
    return None


def evaluate_model_performance(box_loss, cls_loss, dfl_loss):
    """评估模型性能"""
    print("\n" + "="*50)
    print("模型性能评估")
    print("="*50)
    
    # Box Loss评估
    if box_loss < 0.05:
        box_grade = "优秀"
        box_color = "🟢"
    elif box_loss < 0.15:
        box_grade = "良好"
        box_color = "🟡"
    elif box_loss < 0.5:
        box_grade = "一般"
        box_color = "🟠"
    else:
        box_grade = "需要改进"
        box_color = "🔴"
    
    # Class Loss评估
    if cls_loss < 0.8:
        cls_grade = "优秀"
        cls_color = "🟢"
    elif cls_loss < 2.0:
        cls_grade = "良好"
        cls_color = "🟡"
    elif cls_loss < 4.0:
        cls_grade = "一般"
        cls_color = "🟠"
    else:
        cls_grade = "需要改进"
        cls_color = "🔴"
    
    # DFL Loss评估
    if 0.6 <= dfl_loss <= 1.0:
        dfl_grade = "优秀"
        dfl_color = "🟢"
    elif 0.8 <= dfl_loss <= 1.2:
        dfl_grade = "良好"
        dfl_color = "🟡"
    elif dfl_loss < 2.0:
        dfl_grade = "一般"
        dfl_color = "🟠"
    else:
        dfl_grade = "需要改进"
        dfl_color = "🔴"
    
    print(f"Box Loss:   {box_loss:.4f} {box_color} {box_grade}")
    print(f"Class Loss: {cls_loss:.4f} {cls_color} {cls_grade}")
    print(f"DFL Loss:   {dfl_loss:.4f} {dfl_color} {dfl_grade}")
    
    # 总体评估
    scores = []
    if box_grade == "优秀": scores.append(4)
    elif box_grade == "良好": scores.append(3)
    elif box_grade == "一般": scores.append(2)
    else: scores.append(1)
    
    if cls_grade == "优秀": scores.append(4)
    elif cls_grade == "良好": scores.append(3)
    elif cls_grade == "一般": scores.append(2)
    else: scores.append(1)
    
    if dfl_grade == "优秀": scores.append(4)
    elif dfl_grade == "良好": scores.append(3)
    elif dfl_grade == "一般": scores.append(2)
    else: scores.append(1)
    
    avg_score = np.mean(scores)
    
    if avg_score >= 3.5:
        overall = "🟢 模型性能优秀！可以用于生产环境"
    elif avg_score >= 2.5:
        overall = "🟡 模型性能良好，建议继续训练优化"
    elif avg_score >= 1.5:
        overall = "🟠 模型性能一般，需要更多训练"
    else:
        overall = "🔴 模型性能较差，建议检查数据和参数"
    
    print(f"\n总体评估: {overall}")
    
    # 改进建议
    print("\n改进建议:")
    if cls_loss > 2.0:
        print("- Class Loss较高，建议增加训练轮数或降低学习率")
    if box_loss > 0.15:
        print("- Box Loss较高，建议检查标注质量或增加box loss权重")
    if dfl_loss > 1.5:
        print("- DFL Loss较高，建议调整模型架构或数据增强策略")
    
    return avg_score


def plot_loss_curves(data, save_path=None):
    """绘制loss曲线"""
    if not data:
        print("没有可用的训练数据")
        return
    
    epochs = data['epochs']
    box_loss = data['box_loss']
    cls_loss = data['cls_loss']
    dfl_loss = data['dfl_loss']
    
    plt.figure(figsize=(15, 5))
    
    # Box Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, box_loss, 'b-', linewidth=2, label='Box Loss')
    plt.axhline(y=0.15, color='g', linestyle='--', alpha=0.7, label='良好阈值 (0.15)')
    plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='优秀阈值 (0.05)')
    plt.xlabel('Epoch')
    plt.ylabel('Box Loss')
    plt.title('边界框回归损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Class Loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, cls_loss, 'r-', linewidth=2, label='Class Loss')
    plt.axhline(y=2.0, color='g', linestyle='--', alpha=0.7, label='良好阈值 (2.0)')
    plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='优秀阈值 (0.8)')
    plt.xlabel('Epoch')
    plt.ylabel('Class Loss')
    plt.title('分类损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # DFL Loss
    plt.subplot(1, 3, 3)
    plt.plot(epochs, dfl_loss, 'g-', linewidth=2, label='DFL Loss')
    plt.axhline(y=1.2, color='g', linestyle='--', alpha=0.7, label='良好阈值 (1.2)')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='优秀阈值 (1.0)')
    plt.xlabel('Epoch')
    plt.ylabel('DFL Loss')
    plt.title('分布焦点损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss曲线已保存到: {save_path}")
    
    plt.show()


def monitor_latest_training():
    """监控最新的训练"""
    # 查找最新的训练目录
    train_dirs = list(MODELS_DIR.glob("train_*"))
    if not train_dirs:
        print("没有找到训练目录")
        return
    
    latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
    print(f"监控训练目录: {latest_dir}")
    
    # 查找results.csv文件
    results_file = latest_dir / "results.csv"
    if results_file.exists():
        try:
            df = pd.read_csv(results_file)
            print("\n训练结果摘要:")
            print(df.tail())
            
            if len(df) > 0:
                last_row = df.iloc[-1]
                box_loss = last_row.get('train/box_loss', 0)
                cls_loss = last_row.get('train/cls_loss', 0)
                dfl_loss = last_row.get('train/dfl_loss', 0)
                
                evaluate_model_performance(box_loss, cls_loss, dfl_loss)
                
        except Exception as e:
            print(f"读取results.csv失败: {e}")
    
    # 查找权重文件
    weights_dir = latest_dir / "weights"
    if weights_dir.exists():
        best_pt = weights_dir / "best.pt"
        last_pt = weights_dir / "last.pt"
        
        print(f"\n模型文件:")
        if best_pt.exists():
            print(f"✅ 最佳模型: {best_pt}")
        if last_pt.exists():
            print(f"✅ 最新模型: {last_pt}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="训练监控脚本")
    parser.add_argument("--log", type=str, help="日志文件路径")
    parser.add_argument("--latest", action="store_true", help="监控最新训练")
    
    args = parser.parse_args()
    
    if args.latest:
        monitor_latest_training()
    elif args.log:
        data = parse_training_log(args.log)
        if data:
            plot_loss_curves(data)
            if data['box_loss'] and data['cls_loss'] and data['dfl_loss']:
                evaluate_model_performance(
                    data['box_loss'][-1],
                    data['cls_loss'][-1], 
                    data['dfl_loss'][-1]
                )
    else:
        monitor_latest_training()


if __name__ == "__main__":
    main()
