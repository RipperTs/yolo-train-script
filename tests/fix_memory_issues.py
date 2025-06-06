#!/usr/bin/env python3
"""
内存问题快速修复脚本
解决训练过程中内存占用持续增大的问题
"""

import sys
import gc
import torch
import psutil
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from memory_optimizer import memory_optimizer, training_memory_manager
from config_manager import config_manager


def check_current_memory():
    """检查当前内存使用情况"""
    print("🔍 检查当前内存使用情况...")
    
    memory_info = memory_optimizer.get_memory_info()
    
    print(f"💾 系统内存: {memory_info['ram_used_gb']:.1f}/{memory_info['ram_total_gb']:.1f}GB ({memory_info['ram_percent']:.1f}%)")
    
    if 'gpu_allocated_gb' in memory_info:
        print(f"🎮 GPU内存: {memory_info['gpu_allocated_gb']:.1f}/{memory_info['gpu_total_gb']:.1f}GB ({memory_info['gpu_memory_percent']:.1f}%)")
    
    # 内存使用评估
    if memory_info['ram_percent'] > 85:
        print("⚠️ 系统内存使用率过高！")
        return 'high'
    elif memory_info['ram_percent'] > 70:
        print("🟡 系统内存使用率较高")
        return 'medium'
    else:
        print("✅ 系统内存使用率正常")
        return 'normal'


def force_memory_cleanup():
    """强制内存清理"""
    print("🧹 执行强制内存清理...")
    
    # Python垃圾回收
    collected = gc.collect()
    print(f"   Python垃圾回收: 清理了 {collected} 个对象")
    
    # GPU内存清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("   CUDA内存缓存已清理")
    
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print("   MPS内存缓存已清理")
    
    # 再次垃圾回收
    gc.collect()
    
    print("✅ 内存清理完成")


def optimize_training_config():
    """优化训练配置以减少内存使用"""
    print("⚙️ 优化训练配置...")
    
    current_config = config_manager.get_training_config()
    memory_info = memory_optimizer.get_memory_info()
    
    optimizations = {}
    
    # 根据内存使用情况调整配置
    if memory_info['ram_percent'] > 80:
        # 内存紧张，激进优化
        optimizations.update({
            'batch_size': max(1, current_config.get('batch_size', 16) // 2),
            'workers': min(2, current_config.get('workers', 8)),
            'cache': False,
            'save_period': max(20, current_config.get('save_period', 10)),
        })
        print("   应用激进内存优化配置")
    elif memory_info['ram_percent'] > 70:
        # 内存较高，温和优化
        optimizations.update({
            'batch_size': max(4, current_config.get('batch_size', 16) * 3 // 4),
            'workers': min(4, current_config.get('workers', 8)),
            'cache': False,
        })
        print("   应用温和内存优化配置")
    
    # 设备特定优化
    device = current_config.get('device', 'cpu')
    if device == 'mps':
        optimizations.update({
            'amp': False,
            'pin_memory': False,
            'persistent_workers': True,
        })
        print("   应用MPS设备优化")
    elif device.startswith('cuda'):
        # 检查GPU内存
        if memory_info.get('gpu_memory_percent', 0) > 80:
            optimizations.update({
                'batch_size': max(1, current_config.get('batch_size', 16) // 2),
                'amp': True,  # 启用混合精度节省显存
            })
            print("   应用CUDA设备内存优化")
    else:  # CPU
        optimizations.update({
            'amp': False,
            'pin_memory': False,
            'workers': min(2, current_config.get('workers', 8)),
        })
        print("   应用CPU设备优化")
    
    # 应用优化配置
    if optimizations:
        config_manager.update_training_config(**optimizations)
        print(f"✅ 配置优化完成: {optimizations}")
    else:
        print("✅ 当前配置已是最优")
    
    return optimizations


def kill_zombie_processes():
    """清理可能的僵尸进程"""
    print("🔍 检查并清理僵尸进程...")
    
    try:
        current_pid = psutil.Process().pid
        python_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    if proc.info['pid'] != current_pid:  # 不包括当前进程
                        cmdline = ' '.join(proc.info['cmdline'] or [])
                        if any(keyword in cmdline for keyword in ['trainer.py', 'smart_trainer.py', 'gradio_app.py']):
                            python_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if python_processes:
            print(f"   发现 {len(python_processes)} 个相关Python进程")
            for proc in python_processes:
                try:
                    print(f"   进程 {proc.pid}: {' '.join(proc.cmdline())}")
                except:
                    print(f"   进程 {proc.pid}: 无法获取命令行")
            
            response = input("是否终止这些进程？(y/N): ")
            if response.lower() == 'y':
                for proc in python_processes:
                    try:
                        proc.terminate()
                        print(f"   已终止进程 {proc.pid}")
                    except:
                        print(f"   无法终止进程 {proc.pid}")
        else:
            print("   未发现相关进程")
    
    except Exception as e:
        print(f"   检查进程时出错: {e}")


def apply_emergency_fixes():
    """应用紧急修复措施"""
    print("🚨 应用紧急内存修复措施...")
    
    # 1. 强制垃圾回收
    for i in range(3):
        collected = gc.collect()
        print(f"   第{i+1}次垃圾回收: 清理了 {collected} 个对象")
    
    # 2. 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # 重置CUDA上下文（如果可能）
        try:
            torch.cuda.reset_peak_memory_stats()
            print("   CUDA内存统计已重置")
        except:
            pass
    
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    # 3. 设置最保守的配置
    emergency_config = {
        'batch_size': 1,
        'workers': 0,
        'cache': False,
        'amp': False,
        'pin_memory': False,
        'save_period': 50,
        'patience': 10,
    }
    
    config_manager.update_training_config(**emergency_config)
    print(f"   应用紧急配置: {emergency_config}")
    
    print("✅ 紧急修复措施已应用")


def main():
    """主函数"""
    print("🔧 内存问题快速修复工具")
    print("=" * 50)
    
    # 1. 检查当前内存状态
    memory_status = check_current_memory()
    print()
    
    # 2. 强制内存清理
    force_memory_cleanup()
    print()
    
    # 3. 检查清理后的内存状态
    print("🔍 清理后内存状态:")
    memory_status_after = check_current_memory()
    print()
    
    # 4. 优化训练配置
    optimizations = optimize_training_config()
    print()
    
    # 5. 检查僵尸进程
    kill_zombie_processes()
    print()
    
    # 6. 如果内存仍然很高，应用紧急修复
    if memory_status_after == 'high':
        apply_emergency_fixes()
        print()
    
    # 7. 最终检查
    print("🔍 最终内存状态:")
    final_status = check_current_memory()
    print()
    
    # 8. 给出建议
    print("💡 建议:")
    if final_status == 'normal':
        print("✅ 内存状态已恢复正常，可以继续训练")
        print("📝 建议启用内存监控: python memory_monitor.py --action monitor")
    elif final_status == 'medium':
        print("🟡 内存状态有所改善，建议:")
        print("   - 使用较小的批次大小")
        print("   - 启用内存监控")
        print("   - 考虑分批训练")
    else:
        print("🔴 内存状态仍然较高，建议:")
        print("   - 重启系统")
        print("   - 关闭其他应用程序")
        print("   - 使用更小的模型")
        print("   - 考虑使用云端训练")
    
    print("\n📚 更多信息请查看: docs/内存优化指南.md")


if __name__ == "__main__":
    main()
