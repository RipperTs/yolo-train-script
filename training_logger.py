#!/usr/bin/env python3
"""
训练日志捕获器
实时捕获YOLO训练过程中的控制台输出
"""

import sys
import io
import threading
import queue
import time
from typing import List, Optional
from datetime import datetime
from pathlib import Path


class TrainingOutputCapture:
    """训练输出捕获器"""
    
    def __init__(self, max_lines: int = 1000):
        self.max_lines = max_lines
        self.output_queue = queue.Queue()
        self.log_buffer = []
        self.is_capturing = False
        self.original_stdout = None
        self.original_stderr = None
        self.capture_stream = None
        
    def start_capture(self):
        """开始捕获输出"""
        if self.is_capturing:
            return
            
        self.is_capturing = True
        self.log_buffer.clear()
        
        # 保存原始的stdout和stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # 创建自定义的输出流
        self.capture_stream = CaptureStream(self)
        
        # 重定向stdout和stderr
        sys.stdout = self.capture_stream
        sys.stderr = self.capture_stream
        
        print("📡 训练输出捕获已启动")
        
    def stop_capture(self):
        """停止捕获输出"""
        if not self.is_capturing:
            return
            
        self.is_capturing = False
        
        # 恢复原始的stdout和stderr
        if self.original_stdout:
            sys.stdout = self.original_stdout
        if self.original_stderr:
            sys.stderr = self.original_stderr
            
        print("⏹️ 训练输出捕获已停止")
        
    def add_log_line(self, line: str):
        """添加日志行"""
        if not line.strip():
            return
            
        # 添加时间戳
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_line = f"[{timestamp}] {line.strip()}"
        
        # 添加到缓冲区
        self.log_buffer.append(formatted_line)
        if len(self.log_buffer) > self.max_lines:
            self.log_buffer.pop(0)
            
        # 添加到队列
        self.output_queue.put(formatted_line)
        
    def get_recent_logs(self, num_lines: int = 50) -> List[str]:
        """获取最近的日志"""
        if num_lines <= 0:
            return self.log_buffer.copy()
        return self.log_buffer[-num_lines:] if len(self.log_buffer) > num_lines else self.log_buffer.copy()
        
    def get_recent_logs_as_string(self, num_lines: int = 50) -> str:
        """获取最近的日志作为字符串"""
        logs = self.get_recent_logs(num_lines)
        return '\n'.join(logs) if logs else "暂无训练日志"
        
    def get_new_logs(self) -> List[str]:
        """获取新的日志行"""
        logs = []
        while not self.output_queue.empty():
            try:
                logs.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return logs
        
    def get_all_logs(self) -> str:
        """获取所有日志"""
        return '\n'.join(self.log_buffer)


class CaptureStream:
    """自定义输出流，用于捕获print输出"""
    
    def __init__(self, capture_instance: TrainingOutputCapture):
        self.capture = capture_instance
        self.buffer = ""
        
    def write(self, text: str):
        """写入文本"""
        # 同时输出到原始stdout（保持控制台输出）
        if self.capture.original_stdout:
            self.capture.original_stdout.write(text)
            self.capture.original_stdout.flush()
            
        # 捕获输出到日志
        self.buffer += text
        
        # 处理完整的行
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if line.strip():  # 只处理非空行
                self.capture.add_log_line(line)
                
    def flush(self):
        """刷新缓冲区"""
        if self.capture.original_stdout:
            self.capture.original_stdout.flush()
            
        # 处理缓冲区中剩余的内容
        if self.buffer.strip():
            self.capture.add_log_line(self.buffer)
            self.buffer = ""


class TrainingLogManager:
    """训练日志管理器"""
    
    def __init__(self):
        self.output_capture = TrainingOutputCapture()
        self.log_file_path = None
        
    def start_training_logging(self) -> str:
        """开始训练日志记录"""
        # 创建日志文件
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        self.log_file_path = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # 启动输出捕获
        self.output_capture.start_capture()
        
        return str(self.log_file_path)
        
    def stop_training_logging(self):
        """停止训练日志记录"""
        self.output_capture.stop_capture()
        
        # 将日志保存到文件
        if self.log_file_path and self.output_capture.log_buffer:
            try:
                with open(self.log_file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(self.output_capture.log_buffer))
                print(f"📁 训练日志已保存到: {self.log_file_path}")
            except Exception as e:
                print(f"❌ 保存日志文件失败: {e}")
                
    def get_current_logs(self, num_lines: int = 50) -> str:
        """获取当前日志"""
        return self.output_capture.get_recent_logs_as_string(num_lines)
        
    def is_logging(self) -> bool:
        """检查是否正在记录日志"""
        return self.output_capture.is_capturing


# 全局实例
training_log_manager = TrainingLogManager()


def test_capture():
    """测试输出捕获功能"""
    print("🧪 测试训练输出捕获...")
    
    # 启动捕获
    training_log_manager.start_training_logging()
    
    # 模拟训练输出
    print("开始训练...")
    print("Epoch 1/10: box_loss: 4.205, cls_loss: 12.398")
    time.sleep(1)
    print("Epoch 2/10: box_loss: 2.710, cls_loss: 3.929")
    time.sleep(1)
    print("Epoch 3/10: box_loss: 2.315, cls_loss: 3.054")
    
    # 获取日志
    logs = training_log_manager.get_current_logs(10)
    print(f"\n捕获的日志:\n{logs}")
    
    # 停止捕获
    training_log_manager.stop_training_logging()
    print("✅ 测试完成")


if __name__ == "__main__":
    test_capture()
