#!/usr/bin/env python3
"""
智能训练器
自动监控loss变化，决定是否继续训练
"""

import sys
import time
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from trainer import YOLOv8Trainer
from config import MODELS_DIR, ensure_directories


class SmartTrainer:
    """智能训练器类"""
    
    def __init__(self, target_loss_thresholds=None, patience=20, min_improvement=0.001):
        """
        初始化智能训练器
        
        Args:
            target_loss_thresholds: 目标loss阈值字典 {'box_loss': 0.05, 'cls_loss': 1.0, 'dfl_loss': 0.8}
            patience: 早停耐心值（多少个epoch没有改善就停止）
            min_improvement: 最小改善幅度
        """
        self.target_loss_thresholds = target_loss_thresholds or {
            'box_loss': 0.05,    # 优秀水平
            'cls_loss': 1.0,     # 良好水平  
            'dfl_loss': 0.8,     # 良好水平
            'map50': 0.7         # 目标mAP50
        }
        self.patience = patience
        self.min_improvement = min_improvement
        self.trainer = YOLOv8Trainer()
        ensure_directories()
    
    def find_latest_training(self):
        """查找最新的训练结果"""
        train_dirs = list(MODELS_DIR.glob("train_*"))
        if not train_dirs:
            return None
        
        latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
        return latest_dir
    
    def analyze_training_progress(self, results_csv_path):
        """分析训练进度"""
        if not Path(results_csv_path).exists():
            return None
        
        df = pd.read_csv(results_csv_path)
        if len(df) == 0:
            return None
        
        # 获取最新的指标
        latest = df.iloc[-1]
        
        analysis = {
            'total_epochs': len(df),
            'latest_metrics': {
                'box_loss': latest.get('train/box_loss', float('inf')),
                'cls_loss': latest.get('train/cls_loss', float('inf')),
                'dfl_loss': latest.get('train/dfl_loss', float('inf')),
                'map50': latest.get('metrics/mAP50(B)', 0),
                'map50_95': latest.get('metrics/mAP50-95(B)', 0)
            }
        }
        
        # 检查是否达到目标
        targets_met = {}
        for metric, threshold in self.target_loss_thresholds.items():
            current_value = analysis['latest_metrics'].get(metric, float('inf'))
            if metric == 'map50':
                targets_met[metric] = current_value >= threshold
            else:
                targets_met[metric] = current_value <= threshold
        
        analysis['targets_met'] = targets_met
        analysis['all_targets_met'] = all(targets_met.values())
        
        # 检查是否还在改善
        if len(df) >= self.patience:
            recent_data = df.tail(self.patience)
            
            # 检查loss是否还在下降
            box_loss_trend = self._check_improvement_trend(recent_data, 'train/box_loss', decreasing=True)
            cls_loss_trend = self._check_improvement_trend(recent_data, 'train/cls_loss', decreasing=True)
            map50_trend = self._check_improvement_trend(recent_data, 'metrics/mAP50(B)', decreasing=False)
            
            analysis['still_improving'] = any([box_loss_trend, cls_loss_trend, map50_trend])
        else:
            analysis['still_improving'] = True  # 训练轮数不够，继续训练
        
        return analysis
    
    def _check_improvement_trend(self, data, column, decreasing=True):
        """检查指标是否还在改善"""
        if column not in data.columns:
            return False
        
        values = data[column].values
        if len(values) < 5:
            return True
        
        # 计算最近的趋势
        recent_half = values[-len(values)//2:]
        earlier_half = values[:len(values)//2]
        
        if decreasing:
            # 对于loss，检查是否在下降
            improvement = np.mean(earlier_half) - np.mean(recent_half)
        else:
            # 对于mAP，检查是否在上升
            improvement = np.mean(recent_half) - np.mean(earlier_half)
        
        return improvement > self.min_improvement
    
    def should_continue_training(self, analysis):
        """判断是否应该继续训练"""
        if analysis is None:
            return True, "无法分析训练进度，建议开始训练"
        
        # 如果所有目标都达到了
        if analysis['all_targets_met']:
            return False, "🎉 所有目标都已达到！训练可以停止"
        
        # 如果还在改善
        if analysis['still_improving']:
            unmet_targets = [k for k, v in analysis['targets_met'].items() if not v]
            return True, f"📈 模型还在改善，继续训练。未达标指标: {', '.join(unmet_targets)}"
        
        # 如果不再改善
        return False, f"📉 模型已停止改善（连续{self.patience}个epoch无显著提升），建议停止训练"
    
    def print_training_status(self, analysis):
        """打印训练状态"""
        if analysis is None:
            print("❌ 无法获取训练状态")
            return
        
        print("\n" + "="*60)
        print("📊 训练状态分析")
        print("="*60)
        
        print(f"已训练轮数: {analysis['total_epochs']} epochs")
        print("\n当前指标:")
        
        metrics = analysis['latest_metrics']
        targets = self.target_loss_thresholds
        
        for metric, value in metrics.items():
            if metric in targets:
                target = targets[metric]
                if metric == 'map50':
                    status = "✅" if value >= target else "❌"
                    print(f"  {metric:12}: {value:.4f} (目标: ≥{target:.3f}) {status}")
                else:
                    status = "✅" if value <= target else "❌"
                    print(f"  {metric:12}: {value:.4f} (目标: ≤{target:.3f}) {status}")
        
        print(f"\n目标达成情况: {sum(analysis['targets_met'].values())}/{len(analysis['targets_met'])}")
        print(f"是否还在改善: {'是' if analysis['still_improving'] else '否'}")
    
    def plot_training_curves(self, results_csv_path, save_path=None):
        """绘制训练曲线"""
        if not Path(results_csv_path).exists():
            print("❌ 找不到训练结果文件")
            return
        
        df = pd.read_csv(results_csv_path)
        if len(df) == 0:
            print("❌ 训练结果文件为空")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = df.index + 1
        
        # Box Loss
        if 'train/box_loss' in df.columns:
            ax1.plot(epochs, df['train/box_loss'], 'b-', label='Box Loss', linewidth=2)
            ax1.axhline(y=self.target_loss_thresholds['box_loss'], color='r', 
                       linestyle='--', alpha=0.7, label=f'目标 ({self.target_loss_thresholds["box_loss"]})')
            ax1.set_title('Box Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Class Loss
        if 'train/cls_loss' in df.columns:
            ax2.plot(epochs, df['train/cls_loss'], 'r-', label='Class Loss', linewidth=2)
            ax2.axhline(y=self.target_loss_thresholds['cls_loss'], color='r', 
                       linestyle='--', alpha=0.7, label=f'目标 ({self.target_loss_thresholds["cls_loss"]})')
            ax2.set_title('Class Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # DFL Loss
        if 'train/dfl_loss' in df.columns:
            ax3.plot(epochs, df['train/dfl_loss'], 'g-', label='DFL Loss', linewidth=2)
            ax3.axhline(y=self.target_loss_thresholds['dfl_loss'], color='r', 
                       linestyle='--', alpha=0.7, label=f'目标 ({self.target_loss_thresholds["dfl_loss"]})')
            ax3.set_title('DFL Loss')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # mAP50
        if 'metrics/mAP50(B)' in df.columns:
            ax4.plot(epochs, df['metrics/mAP50(B)'], 'purple', label='mAP50', linewidth=2)
            ax4.axhline(y=self.target_loss_thresholds['map50'], color='r', 
                       linestyle='--', alpha=0.7, label=f'目标 ({self.target_loss_thresholds["map50"]})')
            ax4.set_title('mAP50')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('mAP50')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 训练曲线已保存: {save_path}")
        
        plt.show()
    
    def continue_training(self, additional_epochs=50, model_path=None):
        """继续训练"""
        print(f"🚀 继续训练 {additional_epochs} 个epochs...")
        
        try:
            # 修改配置以继续训练
            from config import TRAINING_CONFIG
            continue_config = TRAINING_CONFIG.copy()
            continue_config['epochs'] = additional_epochs
            
            # 使用trainer进行恢复训练
            success = self.trainer.train(resume=True, resume_path=model_path)
            
            if success:
                print("✅ 继续训练完成！")
                return True
            else:
                print("❌ 继续训练失败")
                return False
                
        except Exception as e:
            print(f"❌ 继续训练时出错: {e}")
            return False
    
    def smart_training_loop(self, initial_epochs=100, continue_epochs=50, max_total_epochs=500):
        """智能训练循环"""
        print("🤖 开始智能训练循环...")
        print(f"初始训练: {initial_epochs} epochs")
        print(f"每次继续: {continue_epochs} epochs")
        print(f"最大总轮数: {max_total_epochs} epochs")
        print(f"目标阈值: {self.target_loss_thresholds}")
        
        total_epochs = 0
        
        # 检查是否已有训练结果
        latest_dir = self.find_latest_training()
        if latest_dir:
            results_file = latest_dir / "results.csv"
            analysis = self.analyze_training_progress(results_file)
            
            if analysis:
                total_epochs = analysis['total_epochs']
                print(f"\n发现已有训练结果: {total_epochs} epochs")
                self.print_training_status(analysis)
                
                should_continue, reason = self.should_continue_training(analysis)
                print(f"\n决策: {reason}")
                
                if not should_continue:
                    print("🎉 训练已完成！")
                    return latest_dir
        
        # 训练循环
        while total_epochs < max_total_epochs:
            if total_epochs == 0:
                # 初始训练
                print(f"\n🚀 开始初始训练 ({initial_epochs} epochs)...")
                success = self.trainer.train()
            else:
                # 继续训练
                print(f"\n🔄 继续训练 ({continue_epochs} epochs)...")
                success = self.continue_training(continue_epochs)
            
            if not success:
                print("❌ 训练失败，停止循环")
                break
            
            # 分析新的训练结果
            latest_dir = self.find_latest_training()
            if latest_dir:
                results_file = latest_dir / "results.csv"
                analysis = self.analyze_training_progress(results_file)
                
                if analysis:
                    total_epochs = analysis['total_epochs']
                    self.print_training_status(analysis)
                    
                    # 绘制训练曲线
                    curve_path = latest_dir / "training_curves.png"
                    self.plot_training_curves(results_file, curve_path)
                    
                    should_continue, reason = self.should_continue_training(analysis)
                    print(f"\n决策: {reason}")
                    
                    if not should_continue:
                        print("🎉 训练目标达成或已收敛！")
                        break
            
            # 等待一下再继续
            time.sleep(2)
        
        if total_epochs >= max_total_epochs:
            print(f"⚠️ 达到最大训练轮数 ({max_total_epochs})，停止训练")
        
        return latest_dir


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="智能训练器")
    parser.add_argument("--action", choices=["analyze", "continue", "smart"], 
                       default="smart", help="执行的操作")
    parser.add_argument("--epochs", type=int, default=50, help="继续训练的轮数")
    parser.add_argument("--model", type=str, help="指定模型路径")
    
    # 目标阈值参数
    parser.add_argument("--box-loss", type=float, default=0.05, help="Box Loss目标")
    parser.add_argument("--cls-loss", type=float, default=1.0, help="Class Loss目标")
    parser.add_argument("--dfl-loss", type=float, default=0.8, help="DFL Loss目标")
    parser.add_argument("--map50", type=float, default=0.7, help="mAP50目标")
    
    args = parser.parse_args()
    
    # 设置目标阈值
    target_thresholds = {
        'box_loss': args.box_loss,
        'cls_loss': args.cls_loss,
        'dfl_loss': args.dfl_loss,
        'map50': args.map50
    }
    
    smart_trainer = SmartTrainer(target_thresholds)
    
    if args.action == "analyze":
        # 分析当前训练状态
        latest_dir = smart_trainer.find_latest_training()
        if latest_dir:
            results_file = latest_dir / "results.csv"
            analysis = smart_trainer.analyze_training_progress(results_file)
            smart_trainer.print_training_status(analysis)
            
            should_continue, reason = smart_trainer.should_continue_training(analysis)
            print(f"\n建议: {reason}")
            
            # 绘制训练曲线
            curve_path = latest_dir / "current_training_curves.png"
            smart_trainer.plot_training_curves(results_file, curve_path)
        else:
            print("❌ 没有找到训练结果")
    
    elif args.action == "continue":
        # 继续训练
        smart_trainer.continue_training(args.epochs, args.model)
    
    elif args.action == "smart":
        # 智能训练循环
        result_dir = smart_trainer.smart_training_loop()
        print(f"\n🎯 最终结果保存在: {result_dir}")


if __name__ == "__main__":
    main()
