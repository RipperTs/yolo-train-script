#!/usr/bin/env python3
"""
测试日志监控功能
"""

import sys
import time
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from gradio_utils import log_monitor

def test_log_monitoring():
    """测试日志监控功能"""
    print("🧪 测试日志监控功能...")
    
    # 1. 测试查找日志文件
    print("\n1. 查找最新日志文件...")
    log_file = log_monitor.find_latest_log()
    if log_file:
        print(f"✅ 找到日志文件: {log_file}")
    else:
        print("❌ 没有找到日志文件")
        
        # 创建一个测试日志文件
        log_dir = Path("../logs")
        log_dir.mkdir(exist_ok=True)
        test_log = log_dir / "test_training.log"
        
        with open(test_log, 'w', encoding='utf-8') as f:
            f.write("Test log entry 1\n")
            f.write("Epoch 1/100: box_loss: 0.5, cls_loss: 1.2, dfl_loss: 0.8\n")
            f.write("Test log entry 2\n")
            f.write("Epoch 2/100: box_loss: 0.4, cls_loss: 1.1, dfl_loss: 0.7\n")
        
        print(f"✅ 创建测试日志文件: {test_log}")
        log_file = str(test_log)
    
    # 2. 测试启动监控
    print("\n2. 启动日志监控...")
    if log_monitor.start_monitoring(log_file):
        print("✅ 日志监控启动成功")
    else:
        print("❌ 日志监控启动失败")
        return
    
    # 3. 测试读取日志
    print("\n3. 测试读取日志...")
    time.sleep(1)  # 等待一下
    
    recent_logs = log_monitor.get_recent_logs_as_string(10)
    print(f"最近日志内容:\n{recent_logs}")
    
    # 4. 测试监控状态
    print(f"\n4. 监控状态: {log_monitor.is_monitoring}")
    print(f"当前日志文件: {log_monitor.current_log_file}")
    
    # 5. 停止监控
    print("\n5. 停止日志监控...")
    log_monitor.stop_monitoring()
    print("✅ 日志监控已停止")

if __name__ == "__main__":
    test_log_monitoring()
