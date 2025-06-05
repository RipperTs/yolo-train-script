#!/usr/bin/env python3
"""
YOLO训练理论解释脚本
帮助新手理解训练过程和epoch选择
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_typical_training_curve():
    """绘制典型的训练曲线"""
    epochs = np.arange(1, 201)
    
    # 模拟典型的loss曲线
    train_loss = 8 * np.exp(-epochs/30) + 2 * np.exp(-epochs/80) + 0.5 + 0.1 * np.random.normal(0, 1, len(epochs))
    val_loss = 8.5 * np.exp(-epochs/35) + 2.2 * np.exp(-epochs/85) + 0.6 + 0.15 * np.random.normal(0, 1, len(epochs))
    
    # 模拟mAP曲线
    map50 = 0.8 * (1 - np.exp(-epochs/40)) + 0.05 * np.random.normal(0, 1, len(epochs))
    map50 = np.clip(map50, 0, 0.85)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss曲线
    ax1.plot(epochs, train_loss, 'b-', label='训练Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='验证Loss', linewidth=2)
    ax1.axvline(x=100, color='g', linestyle='--', alpha=0.7, label='建议停止点(100 epochs)')
    ax1.axvline(x=150, color='orange', linestyle='--', alpha=0.7, label='过拟合风险点')
    
    # 标注关键阶段
    ax1.annotate('快速学习期\n(0-30 epochs)', xy=(15, 6), xytext=(40, 7),
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                fontsize=10, ha='center')
    
    ax1.annotate('稳定收敛期\n(30-100 epochs)', xy=(65, 2), xytext=(90, 3.5),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                fontsize=10, ha='center')
    
    ax1.annotate('过拟合风险期\n(>150 epochs)', xy=(170, 1), xytext=(170, 2.5),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, ha='center')
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('典型的YOLO训练Loss曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 9)
    
    # mAP曲线
    ax2.plot(epochs, map50, 'g-', label='mAP50', linewidth=2)
    ax2.axvline(x=100, color='g', linestyle='--', alpha=0.7, label='建议停止点')
    ax2.axhline(y=0.7, color='orange', linestyle=':', alpha=0.7, label='良好性能阈值(70%)')
    ax2.axhline(y=0.5, color='yellow', linestyle=':', alpha=0.7, label='可接受阈值(50%)')
    
    ax2.annotate('性能快速提升期', xy=(30, 0.4), xytext=(60, 0.6),
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                fontsize=10, ha='center')
    
    ax2.annotate('性能趋于稳定', xy=(120, 0.75), xytext=(150, 0.6),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                fontsize=10, ha='center')
    
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('mAP50')
    ax2.set_title('典型的YOLO训练mAP曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.9)
    
    plt.tight_layout()
    plt.savefig('yolov8/training_theory_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


def explain_epoch_selection():
    """解释epoch选择的原理"""
    print("🎯 YOLO训练Epoch选择指南")
    print("=" * 50)
    
    print("\n📊 训练阶段分析:")
    stages = [
        ("第1-10轮", "模型初始化", "Loss快速下降", "mAP从0开始上升"),
        ("第10-30轮", "快速学习期", "Loss大幅下降", "mAP快速提升"),
        ("第30-80轮", "稳定学习期", "Loss缓慢下降", "mAP稳步提升"),
        ("第80-120轮", "精细调优期", "Loss微调", "mAP接近最优"),
        ("第120轮以上", "过拟合风险期", "验证Loss可能上升", "mAP可能下降")
    ]
    
    for stage, name, loss_trend, map_trend in stages:
        print(f"\n{stage:12} | {name:10} | {loss_trend:15} | {map_trend}")
    
    print("\n🎯 不同数据集大小的建议:")
    dataset_recommendations = [
        ("< 100张", "200-500 epochs", "数据少，需要更多重复学习"),
        ("100-500张", "100-300 epochs", "您的情况，中等训练量"),
        ("500-2000张", "50-150 epochs", "数据充足，训练效率高"),
        ("2000-10000张", "30-100 epochs", "大数据集，快速收敛"),
        ("> 10000张", "20-80 epochs", "超大数据集，避免过拟合")
    ]
    
    for dataset_size, epochs, reason in dataset_recommendations:
        print(f"{dataset_size:12} | {epochs:15} | {reason}")
    
    print("\n⚠️ 过拟合的信号:")
    overfitting_signs = [
        "训练Loss继续下降，但验证Loss开始上升",
        "训练mAP继续提升，但验证mAP开始下降",
        "模型在训练集上表现很好，但在新数据上表现差"
    ]
    
    for i, sign in enumerate(overfitting_signs, 1):
        print(f"{i}. {sign}")
    
    print("\n✅ 何时停止训练:")
    stop_criteria = [
        "验证Loss连续10-20个epoch不再下降",
        "mAP达到满意水平（如70%以上）",
        "训练和验证指标开始发散",
        "达到预设的最大epoch数"
    ]
    
    for i, criteria in enumerate(stop_criteria, 1):
        print(f"{i}. {criteria}")


def calculate_training_time():
    """计算训练时间估算"""
    print("\n⏱️ 训练时间估算")
    print("=" * 30)
    
    # 基于您的快速训练结果
    time_per_epoch = 0.023 * 60 / 5  # 5个epoch用了0.023小时，转换为分钟
    
    scenarios = [
        (50, "快速测试"),
        (100, "标准训练"),
        (200, "充分训练"),
        (300, "深度训练")
    ]
    
    print(f"基于您的系统性能（每个epoch约{time_per_epoch:.1f}分钟）:")
    print()
    
    for epochs, description in scenarios:
        total_minutes = epochs * time_per_epoch
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)
        
        if hours > 0:
            time_str = f"{hours}小时{minutes}分钟"
        else:
            time_str = f"{minutes}分钟"
        
        print(f"{epochs:3d} epochs ({description:8}) ≈ {time_str}")


def analyze_your_current_results():
    """分析您当前的训练结果"""
    print("\n📈 您的训练结果分析")
    print("=" * 30)
    
    current_results = {
        "epochs": 5,
        "map50": 0.461,
        "precision": 0.489,
        "recall": 0.495,
        "box_loss": 2.692,
        "cls_loss": 3.597
    }
    
    print(f"当前训练轮数: {current_results['epochs']} epochs")
    print(f"当前mAP50: {current_results['map50']:.3f} (46.1%)")
    print()
    
    # 预测100 epochs的性能
    improvement_factor = (100 / 5) ** 0.3  # 经验公式：性能提升随epoch数的0.3次方增长
    predicted_map50 = min(0.85, current_results['map50'] * improvement_factor)
    
    print("🔮 预测100 epochs后的性能:")
    print(f"预期mAP50: {predicted_map50:.3f} ({predicted_map50*100:.1f}%)")
    
    if predicted_map50 > 0.7:
        print("✅ 预期达到良好性能水平（>70%）")
    elif predicted_map50 > 0.5:
        print("🟡 预期达到可接受性能水平（>50%）")
    else:
        print("⚠️ 可能需要更多训练或数据优化")
    
    print("\n💡 建议:")
    if current_results['map50'] > 0.4:
        print("- 您的模型显示出良好潜力")
        print("- 建议进行100-150 epochs的完整训练")
        print("- 可以期待显著的性能提升")
    else:
        print("- 建议检查数据质量和标注准确性")
        print("- 考虑调整学习率或模型参数")


def main():
    """主函数"""
    print("🎓 YOLO训练新手指南：为什么选择100个Epochs？")
    print("=" * 60)
    
    # 解释基本概念
    explain_epoch_selection()
    
    # 计算训练时间
    calculate_training_time()
    
    # 分析当前结果
    analyze_your_current_results()
    
    # 绘制理论曲线
    print("\n📊 正在生成训练曲线图...")
    plot_typical_training_curve()
    print("✅ 训练曲线图已保存为 'training_theory_curves.png'")
    
    print("\n🎯 总结:")
    print("- 100 epochs是基于您的数据集大小(288张)的经验建议")
    print("- 这个数量能让模型充分学习而不过拟合")
    print("- 实际最优epoch数需要通过验证指标来确定")
    print("- 可以设置早停机制，在性能不再提升时自动停止")


if __name__ == "__main__":
    main()
