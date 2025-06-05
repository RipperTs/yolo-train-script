#!/usr/bin/env python3
"""
测试真实YOLO训练的日志捕获
"""

import sys
import time
import threading
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from trainer import YOLOv8Trainer
from training_logger import training_log_manager


def test_real_yolo_training():
    """测试真实的YOLO训练日志捕获"""
    print("🎯 测试真实YOLO训练日志捕获...")
    
    try:
        # 检查数据集是否存在
        dataset_yaml = Path("../dataset.yaml")
        if not dataset_yaml.exists():
            print("❌ 数据集配置文件不存在，跳过真实训练测试")
            print("请先运行 data_converter.py 转换数据")
            return
        
        # 创建训练器
        trainer = YOLOv8Trainer()
        
        # 在后台线程中运行训练
        def run_training():
            try:
                # 修改训练配置为快速测试
                from config import config_manager
                config_manager.update_training_config(
                    epochs=2,  # 只训练2个epoch用于测试
                    batch_size=1,  # 小批次
                    learning_rate=0.01
                )
                
                print("🚀 开始真实YOLO训练...")
                trainer.train()
                print("✅ 训练完成")
                
            except Exception as e:
                print(f"❌ 训练失败: {e}")
        
        # 启动训练线程
        training_thread = threading.Thread(target=run_training)
        training_thread.daemon = True
        training_thread.start()
        
        # 监控日志输出
        print("📡 开始监控训练日志...")
        for i in range(30):  # 监控60秒
            time.sleep(2)
            
            if training_log_manager.is_logging():
                logs = training_log_manager.get_current_logs(10)
                print(f"\n--- 日志更新 {i+1} ---")
                # 只显示最后几行，避免输出过多
                log_lines = logs.split('\n')
                recent_lines = log_lines[-5:] if len(log_lines) > 5 else log_lines
                for line in recent_lines:
                    if line.strip():
                        print(line)
                print("--- 更新结束 ---")
            else:
                print(f"第{i+1}次检查: 训练日志捕获未启动")
                
            # 检查训练线程是否还在运行
            if not training_thread.is_alive():
                print("训练线程已结束")
                break
        
        # 等待训练完成
        training_thread.join(timeout=10)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")


def test_gradio_app_integration():
    """测试Gradio应用集成"""
    print("\n🎨 测试Gradio应用集成...")
    
    try:
        # 导入Gradio应用
        from gradio_app import GradioApp
        
        # 创建应用实例
        app = GradioApp()
        
        # 模拟训练启动
        print("模拟启动普通训练...")
        
        # 这里我们不实际启动训练，只测试日志更新逻辑
        def mock_update_training_log():
            """模拟定时日志更新"""
            if training_log_manager.is_logging():
                log_content = training_log_manager.get_current_logs(20)
                if log_content and log_content != "暂无训练日志":
                    return log_content
            return "暂无日志更新"
        
        # 启动模拟训练日志
        training_log_manager.start_training_logging()
        
        # 模拟一些训练输出
        print("模拟训练输出...")
        print("Epoch 1/5: box_loss: 4.205, cls_loss: 12.398")
        print("Epoch 2/5: box_loss: 2.710, cls_loss: 3.929")
        
        # 测试日志更新
        time.sleep(1)
        log_update = mock_update_training_log()
        print(f"\n前端日志更新结果:\n{log_update}")
        
        # 停止日志捕获
        training_log_manager.stop_training_logging()
        
        print("✅ Gradio集成测试完成")
        
    except Exception as e:
        print(f"❌ Gradio集成测试失败: {e}")


if __name__ == "__main__":
    try:
        print("🧪 开始真实训练日志测试...")
        
        # 测试1: 真实YOLO训练（如果数据集存在）
        test_real_yolo_training()
        
        # 测试2: Gradio应用集成
        test_gradio_app_integration()
        
        print("\n✅ 所有测试完成!")
        
    except KeyboardInterrupt:
        print("\n👋 用户中断测试")
        training_log_manager.stop_training_logging()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        training_log_manager.stop_training_logging()
