#!/usr/bin/env python3
"""
内存监控工具
实时监控训练过程中的内存使用情况
"""

import time
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import json

from memory_optimizer import MemoryOptimizer


class MemoryMonitorTool:
    """内存监控工具"""
    
    def __init__(self, log_file=None, plot_interval=30):
        self.optimizer = MemoryOptimizer(enable_monitoring=True, cleanup_interval=5)
        self.log_file = log_file or f"memory_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.plot_interval = plot_interval
        self.memory_data = []
        
    def start_monitoring(self, duration=None):
        """开始监控"""
        print(f"🔍 开始内存监控...")
        print(f"📁 日志文件: {self.log_file}")
        if duration:
            print(f"⏱️ 监控时长: {duration} 秒")
        
        self.optimizer.start_monitoring()
        
        start_time = time.time()
        last_plot_time = start_time
        
        try:
            while True:
                current_time = time.time()
                
                # 获取内存信息
                memory_info = self.optimizer.get_memory_info()
                memory_info['elapsed_time'] = current_time - start_time
                self.memory_data.append(memory_info)
                
                # 保存到日志文件
                self._save_log()
                
                # 打印当前状态
                self._print_status(memory_info)
                
                # 定期绘制图表
                if current_time - last_plot_time >= self.plot_interval:
                    self._plot_memory_usage()
                    last_plot_time = current_time
                
                # 检查是否达到监控时长
                if duration and (current_time - start_time) >= duration:
                    break
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n⏹️ 监控已停止")
        finally:
            self.optimizer.stop_monitoring()
            self._save_final_report()
    
    def _print_status(self, memory_info):
        """打印当前状态"""
        elapsed = memory_info['elapsed_time']
        ram_percent = memory_info['ram_percent']
        ram_used = memory_info['ram_used_gb']
        ram_total = memory_info['ram_total_gb']
        
        status_line = f"⏱️ {elapsed:6.0f}s | 🖥️ RAM: {ram_used:.1f}/{ram_total:.1f}GB ({ram_percent:.1f}%)"
        
        if 'gpu_memory_percent' in memory_info:
            gpu_percent = memory_info['gpu_memory_percent']
            gpu_allocated = memory_info['gpu_allocated_gb']
            gpu_total = memory_info['gpu_total_gb']
            status_line += f" | 🎮 GPU: {gpu_allocated:.1f}/{gpu_total:.1f}GB ({gpu_percent:.1f}%)"
        
        print(f"\r{status_line}", end="", flush=True)
        
        # 如果内存使用率过高，换行显示警告
        if ram_percent > 85 or memory_info.get('gpu_memory_percent', 0) > 90:
            print()  # 换行
            if ram_percent > 85:
                print(f"⚠️ 系统内存使用率过高: {ram_percent:.1f}%")
            if memory_info.get('gpu_memory_percent', 0) > 90:
                print(f"⚠️ GPU内存使用率过高: {memory_info['gpu_memory_percent']:.1f}%")
    
    def _save_log(self):
        """保存日志"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory_data, f, indent=2)
        except Exception as e:
            print(f"\n❌ 保存日志失败: {e}")
    
    def _plot_memory_usage(self):
        """绘制内存使用图表"""
        if len(self.memory_data) < 2:
            return
        
        try:
            times = [d['elapsed_time'] / 60 for d in self.memory_data]  # 转换为分钟
            ram_usage = [d['ram_percent'] for d in self.memory_data]
            
            plt.figure(figsize=(12, 6))
            
            # RAM使用率
            plt.subplot(1, 2, 1)
            plt.plot(times, ram_usage, 'b-', linewidth=2, label='RAM使用率')
            plt.axhline(y=85, color='r', linestyle='--', alpha=0.7, label='警告线 (85%)')
            plt.xlabel('时间 (分钟)')
            plt.ylabel('使用率 (%)')
            plt.title('系统内存使用率')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # GPU使用率（如果有）
            plt.subplot(1, 2, 2)
            if any('gpu_memory_percent' in d for d in self.memory_data):
                gpu_usage = [d.get('gpu_memory_percent', 0) for d in self.memory_data]
                plt.plot(times, gpu_usage, 'g-', linewidth=2, label='GPU内存使用率')
                plt.axhline(y=90, color='r', linestyle='--', alpha=0.7, label='警告线 (90%)')
                plt.xlabel('时间 (分钟)')
                plt.ylabel('使用率 (%)')
                plt.title('GPU内存使用率')
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'GPU不可用', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('GPU内存使用率')
            
            plt.tight_layout()
            
            # 保存图表
            plot_file = f"memory_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"\n📊 内存使用图表已保存: {plot_file}")
            
        except Exception as e:
            print(f"\n❌ 绘制图表失败: {e}")
    
    def _save_final_report(self):
        """保存最终报告"""
        if not self.memory_data:
            return
        
        try:
            # 计算统计信息
            ram_usage = [d['ram_percent'] for d in self.memory_data]
            max_ram = max(ram_usage)
            avg_ram = sum(ram_usage) / len(ram_usage)
            
            report = {
                'monitoring_duration': self.memory_data[-1]['elapsed_time'],
                'total_samples': len(self.memory_data),
                'ram_stats': {
                    'max_usage_percent': max_ram,
                    'avg_usage_percent': avg_ram,
                    'max_usage_gb': max(d['ram_used_gb'] for d in self.memory_data),
                    'total_ram_gb': self.memory_data[0]['ram_total_gb']
                }
            }
            
            # GPU统计（如果有）
            if any('gpu_memory_percent' in d for d in self.memory_data):
                gpu_usage = [d.get('gpu_memory_percent', 0) for d in self.memory_data]
                report['gpu_stats'] = {
                    'max_usage_percent': max(gpu_usage),
                    'avg_usage_percent': sum(gpu_usage) / len(gpu_usage),
                    'max_usage_gb': max(d.get('gpu_allocated_gb', 0) for d in self.memory_data),
                    'total_gpu_gb': self.memory_data[0].get('gpu_total_gb', 0)
                }
            
            # 保存报告
            report_file = f"memory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            # 打印摘要
            print(f"\n📋 内存监控报告")
            print("=" * 50)
            print(f"监控时长: {report['monitoring_duration']:.0f} 秒")
            print(f"采样次数: {report['total_samples']}")
            print(f"RAM最大使用率: {max_ram:.1f}%")
            print(f"RAM平均使用率: {avg_ram:.1f}%")
            
            if 'gpu_stats' in report:
                print(f"GPU最大使用率: {report['gpu_stats']['max_usage_percent']:.1f}%")
                print(f"GPU平均使用率: {report['gpu_stats']['avg_usage_percent']:.1f}%")
            
            print(f"📁 详细报告已保存: {report_file}")
            
        except Exception as e:
            print(f"\n❌ 保存报告失败: {e}")
    
    def analyze_log(self, log_file):
        """分析已有的日志文件"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                print("❌ 日志文件为空")
                return
            
            self.memory_data = data
            print(f"📊 分析日志文件: {log_file}")
            print(f"数据点数量: {len(data)}")
            
            # 绘制图表
            self._plot_memory_usage()
            
            # 生成报告
            self._save_final_report()
            
        except Exception as e:
            print(f"❌ 分析日志文件失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="内存监控工具")
    parser.add_argument("--action", choices=["monitor", "analyze"], 
                       default="monitor", help="执行的操作")
    parser.add_argument("--duration", type=int, help="监控时长（秒）")
    parser.add_argument("--log-file", type=str, help="日志文件路径")
    parser.add_argument("--plot-interval", type=int, default=30, 
                       help="绘图间隔（秒）")
    
    args = parser.parse_args()
    
    monitor = MemoryMonitorTool(
        log_file=args.log_file,
        plot_interval=args.plot_interval
    )
    
    if args.action == "monitor":
        monitor.start_monitoring(duration=args.duration)
    elif args.action == "analyze":
        if not args.log_file:
            print("❌ 分析模式需要指定 --log-file 参数")
            return
        monitor.analyze_log(args.log_file)


if __name__ == "__main__":
    main()
