#!/usr/bin/env python3
"""
测试集成的训练日志功能
"""

import sys
import time
import threading
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from training_logger import training_log_manager


def simulate_yolo_training():
    """模拟YOLO训练过程"""
    print("🎯 模拟YOLO训练过程...")
    
    # 启动日志捕获
    log_file = training_log_manager.start_training_logging()
    print(f"📁 日志文件: {log_file}")
    
    try:
        # 模拟YOLO训练输出
        print("Ultralytics YOLOv8.0.0 🚀 Python-3.9.7 torch-1.13.0 CPU")
        print("Model summary: 225 layers, 3157200 parameters, 0 gradients, 8.9 GFLOPs")
        print("Optimizer: SGD(lr=0.01) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 64 bias")
        print("train: Scanning /path/to/train/labels... 100 images, 0 backgrounds, 0 corrupt: 100%")
        print("val: Scanning /path/to/val/labels... 20 images, 0 backgrounds, 0 corrupt: 100%")
        
        # 模拟训练epochs
        for epoch in range(1, 6):
            time.sleep(1)
            print(f"Epoch {epoch}/5")
            
            # 模拟训练指标
            box_loss = 4.5 - epoch * 0.5
            cls_loss = 12.0 - epoch * 2.0
            dfl_loss = 1.5 - epoch * 0.2
            
            print(f"      Class     Images  Instances          P          R      mAP50   mAP50-95")
            print(f"        all        100        150      0.{epoch:03d}      0.{epoch+2:03d}      0.{epoch:03d}      0.{epoch:03d}")
            print(f"train/box_loss: {box_loss:.3f}, train/cls_loss: {cls_loss:.3f}, train/dfl_loss: {dfl_loss:.3f}")
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch}: saving model to best.pt")
                
        print("Training complete (5 epochs)")
        print("Results saved to runs/detect/train")
        print("Predict:         yolo predict model=best.pt")
        print("Validate:        yolo val model=best.pt")
        print("Export:          yolo export model=best.pt")
        
    finally:
        # 停止日志捕获
        training_log_manager.stop_training_logging()


def test_real_time_log_access():
    """测试实时日志访问"""
    print("\n🔍 测试实时日志访问...")
    
    def training_simulation():
        """在后台运行训练模拟"""
        time.sleep(1)  # 等待一下
        simulate_yolo_training()
    
    # 在后台线程中运行训练
    training_thread = threading.Thread(target=training_simulation)
    training_thread.daemon = True
    training_thread.start()
    
    # 模拟前端定时获取日志
    for i in range(10):
        time.sleep(2)
        
        if training_log_manager.is_logging():
            logs = training_log_manager.get_current_logs(10)
            print(f"\n--- 前端日志更新 {i+1} ---")
            print(logs[-200:] if len(logs) > 200 else logs)  # 显示最后200个字符
            print("--- 更新结束 ---")
        else:
            print(f"\n--- 前端日志更新 {i+1} ---")
            print("训练未在进行或日志捕获未启动")
            print("--- 更新结束 ---")
    
    # 等待训练线程完成
    training_thread.join(timeout=10)


def test_gradio_integration():
    """测试Gradio集成"""
    print("\n🎨 测试Gradio集成...")
    
    # 模拟Gradio应用的训练状态
    class MockGradioApp:
        def __init__(self):
            self.is_training = False
            
        def start_training(self):
            """模拟开始训练"""
            self.is_training = True
            
            # 在后台线程中运行训练
            training_thread = threading.Thread(target=simulate_yolo_training)
            training_thread.daemon = True
            training_thread.start()
            
            return "🚀 训练已启动", "📡 实时日志捕获已开始..."
            
        def update_training_log(self):
            """模拟Gradio的定时日志更新"""
            if self.is_training and training_log_manager.is_logging():
                log_content = training_log_manager.get_current_logs(20)
                if log_content and log_content != "暂无训练日志":
                    return log_content
            return "暂无日志更新"
    
    # 测试模拟应用
    app = MockGradioApp()
    
    # 启动训练
    status, initial_log = app.start_training()
    print(f"训练状态: {status}")
    print(f"初始日志: {initial_log}")
    
    # 模拟定时更新
    for i in range(8):
        time.sleep(2)
        log_update = app.update_training_log()
        print(f"\n=== 定时更新 {i+1} ===")
        print(log_update[-300:] if len(log_update) > 300 else log_update)
        print("=== 更新结束 ===")


if __name__ == "__main__":
    try:
        print("🧪 开始集成测试...")
        
        # 测试1: 基本训练模拟
        simulate_yolo_training()
        
        # 测试2: 实时日志访问
        test_real_time_log_access()
        
        # 测试3: Gradio集成
        test_gradio_integration()
        
        print("\n✅ 所有测试完成!")
        
    except KeyboardInterrupt:
        print("\n👋 用户中断测试")
        training_log_manager.stop_training_logging()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        training_log_manager.stop_training_logging()
