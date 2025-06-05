#!/usr/bin/env python3
"""
测试训练日志显示功能
"""

import sys
import time
import threading
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from gradio_utils import log_monitor

def simulate_training_with_logs():
    """模拟训练过程并生成日志"""
    print("🎯 模拟训练过程...")
    
    # 创建日志目录
    log_dir = Path("../logs")
    log_dir.mkdir(exist_ok=True)
    
    # 创建模拟训练日志文件
    log_file = log_dir / "simulated_training.log"
    
    # 模拟训练日志内容
    training_logs = [
        "2024-01-01 10:00:00 - INFO - 开始YOLO训练...",
        "2024-01-01 10:00:01 - INFO - 训练参数: {'epochs': 10, 'batch': 16, 'lr0': 0.01}",
        "2024-01-01 10:00:02 - INFO - 加载数据集...",
        "2024-01-01 10:00:05 - INFO - Epoch 1/10: box_loss: 4.205, cls_loss: 12.398, dfl_loss: 1.513",
        "2024-01-01 10:00:10 - INFO - Epoch 1/10: mAP50: 0.005, mAP50-95: 0.002, precision: 0.006, recall: 0.721",
        "2024-01-01 10:00:15 - INFO - Epoch 2/10: box_loss: 2.710, cls_loss: 3.929, dfl_loss: 1.025",
        "2024-01-01 10:00:20 - INFO - Epoch 2/10: mAP50: 0.003, mAP50-95: 0.001, precision: 0.003, recall: 0.587",
        "2024-01-01 10:00:25 - INFO - Epoch 3/10: box_loss: 2.315, cls_loss: 3.054, dfl_loss: 0.921",
        "2024-01-01 10:00:30 - INFO - Epoch 3/10: mAP50: 0.008, mAP50-95: 0.003, precision: 0.005, recall: 0.808",
        "2024-01-01 10:00:35 - INFO - 训练进度: 30% 完成",
        "2024-01-01 10:00:40 - INFO - Epoch 4/10: box_loss: 2.142, cls_loss: 2.844, dfl_loss: 0.899",
        "2024-01-01 10:00:45 - INFO - Epoch 4/10: mAP50: 0.679, mAP50-95: 0.188, precision: 0.903, recall: 0.380",
        "2024-01-01 10:00:50 - INFO - 检测到性能提升，保存最佳模型...",
        "2024-01-01 10:00:55 - INFO - Epoch 5/10: box_loss: 1.985, cls_loss: 2.654, dfl_loss: 0.845",
        "2024-01-01 10:01:00 - INFO - Epoch 5/10: mAP50: 0.712, mAP50-95: 0.205, precision: 0.915, recall: 0.425",
    ]
    
    # 写入初始日志
    with open(log_file, 'w', encoding='utf-8') as f:
        for log in training_logs[:5]:
            f.write(log + '\n')
    
    print(f"✅ 创建模拟日志文件: {log_file}")
    
    # 启动日志监控
    print("📡 启动日志监控...")
    if log_monitor.start_monitoring(str(log_file)):
        print("✅ 日志监控启动成功")
    else:
        print("❌ 日志监控启动失败")
        return
    
    # 模拟实时写入日志
    def write_logs_gradually():
        """逐步写入日志，模拟实时训练"""
        time.sleep(2)  # 等待监控启动
        
        with open(log_file, 'a', encoding='utf-8') as f:
            for log in training_logs[5:]:
                f.write(log + '\n')
                f.flush()  # 确保立即写入
                print(f"📝 写入日志: {log}")
                time.sleep(2)  # 每2秒写入一行
    
    # 在后台线程中写入日志
    log_thread = threading.Thread(target=write_logs_gradually)
    log_thread.daemon = True
    log_thread.start()
    
    # 监控日志变化
    print("\n🔍 监控日志变化...")
    for i in range(15):  # 监控30秒
        time.sleep(2)
        
        # 获取最新日志
        recent_logs = log_monitor.get_recent_logs_as_string(5)
        print(f"\n--- 第{i+1}次检查 (最近5行) ---")
        print(recent_logs)
        print("--- 检查结束 ---")
    
    # 停止监控
    print("\n⏹️ 停止日志监控...")
    log_monitor.stop_monitoring()
    print("✅ 测试完成")

def test_log_update_function():
    """测试日志更新函数"""
    print("\n🧪 测试日志更新函数...")
    
    # 模拟训练状态
    class MockApp:
        def __init__(self):
            self.is_training = True
    
    app = MockApp()
    
    # 模拟定时器更新函数
    def update_training_log():
        """定时更新训练日志"""
        if app.is_training and log_monitor.is_monitoring:
            # 获取最新的日志内容
            log_content = log_monitor.get_recent_logs_as_string(10)  # 获取最近10行
            if log_content and log_content != "没有找到日志文件":
                return log_content
        return "没有新的日志内容"
    
    # 测试更新函数
    for i in range(5):
        result = update_training_log()
        print(f"\n更新 {i+1}: {result[:100]}..." if len(result) > 100 else f"\n更新 {i+1}: {result}")
        time.sleep(3)

if __name__ == "__main__":
    try:
        simulate_training_with_logs()
        test_log_update_function()
    except KeyboardInterrupt:
        print("\n👋 用户中断测试")
        log_monitor.stop_monitoring()
