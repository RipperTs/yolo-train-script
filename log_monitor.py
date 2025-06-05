"""
实时日志监控模块
提供训练过程的实时日志监控功能
"""

import os
import time
import threading
import queue
from pathlib import Path
from typing import Optional, List, Dict
import re
from datetime import datetime


class RealTimeLogMonitor:
    """实时日志监控器"""
    
    def __init__(self, max_lines: int = 1000):
        self.max_lines = max_lines
        self.log_queue = queue.Queue()
        self.is_monitoring = False
        self.monitor_thread = None
        self.current_log_file = None
        self.log_buffer = []
        
    def start_monitoring(self, log_file_path: Optional[str] = None) -> bool:
        """开始监控日志文件"""
        if self.is_monitoring:
            self.stop_monitoring()
        
        if log_file_path is None:
            log_file_path = self._find_latest_log()
        
        if log_file_path and Path(log_file_path).exists():
            self.current_log_file = log_file_path
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_log_file)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            return True
        return False
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
    
    def _find_latest_log(self) -> Optional[str]:
        """查找最新的日志文件"""
        # 查找可能的日志位置
        possible_dirs = [
            Path("logs"),
            Path("runs"),
            Path("."),
            Path("models")
        ]
        
        log_files = []
        for log_dir in possible_dirs:
            if log_dir.exists():
                # 查找各种日志文件
                patterns = ["*.log", "**/*.log", "**/train.log", "**/results.csv"]
                for pattern in patterns:
                    log_files.extend(log_dir.glob(pattern))
        
        if not log_files:
            return None
        
        # 返回最新修改的日志文件
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        return str(latest_log)
    
    def _monitor_log_file(self):
        """监控日志文件的变化"""
        try:
            with open(self.current_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                # 移动到文件末尾
                f.seek(0, 2)
                
                while self.is_monitoring:
                    line = f.readline()
                    if line:
                        self._process_log_line(line.strip())
                    else:
                        time.sleep(0.1)
        except Exception as e:
            self.log_queue.put(f"[ERROR] 日志监控错误: {e}")
    
    def _process_log_line(self, line: str):
        """处理日志行"""
        if not line:
            return
        
        # 添加时间戳
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_line = f"[{timestamp}] {line}"
        
        # 添加到缓冲区
        self.log_buffer.append(formatted_line)
        if len(self.log_buffer) > self.max_lines:
            self.log_buffer.pop(0)
        
        # 添加到队列
        self.log_queue.put(formatted_line)
    
    def get_new_logs(self) -> List[str]:
        """获取新的日志行"""
        logs = []
        while not self.log_queue.empty():
            try:
                logs.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        return logs
    
    def get_all_logs(self) -> str:
        """获取所有缓存的日志"""
        return '\n'.join(self.log_buffer)
    
    def get_recent_logs(self, num_lines: int = 100) -> str:
        """获取最近的日志内容"""
        if not self.current_log_file or not Path(self.current_log_file).exists():
            return "没有找到日志文件"
        
        try:
            with open(self.current_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                recent_lines = lines[-num_lines:] if len(lines) > num_lines else lines
                return ''.join(recent_lines)
        except Exception as e:
            return f"读取日志文件错误: {e}"
    
    def parse_training_metrics(self, log_content: str) -> Dict:
        """从日志中解析训练指标"""
        metrics = {
            'current_epoch': 0,
            'total_epochs': 0,
            'box_loss': 0.0,
            'cls_loss': 0.0,
            'dfl_loss': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'map50': 0.0,
            'map50_95': 0.0
        }
        
        try:
            # 解析epoch信息
            epoch_pattern = r'Epoch\s+(\d+)/(\d+)'
            epoch_matches = re.findall(epoch_pattern, log_content)
            if epoch_matches:
                current, total = epoch_matches[-1]
                metrics['current_epoch'] = int(current)
                metrics['total_epochs'] = int(total)
            
            # 解析loss信息
            loss_patterns = {
                'box_loss': r'box_loss[:\s]+([0-9.]+)',
                'cls_loss': r'cls_loss[:\s]+([0-9.]+)',
                'dfl_loss': r'dfl_loss[:\s]+([0-9.]+)'
            }
            
            for metric, pattern in loss_patterns.items():
                matches = re.findall(pattern, log_content)
                if matches:
                    metrics[metric] = float(matches[-1])
            
            # 解析mAP信息
            map_patterns = {
                'map50': r'mAP50[:\s]+([0-9.]+)',
                'map50_95': r'mAP50-95[:\s]+([0-9.]+)',
                'precision': r'precision[:\s]+([0-9.]+)',
                'recall': r'recall[:\s]+([0-9.]+)'
            }
            
            for metric, pattern in map_patterns.items():
                matches = re.findall(pattern, log_content)
                if matches:
                    metrics[metric] = float(matches[-1])
        
        except Exception as e:
            print(f"解析训练指标错误: {e}")
        
        return metrics
    
    def get_training_status(self) -> str:
        """获取训练状态摘要"""
        if not self.log_buffer:
            return "暂无日志信息"
        
        recent_logs = '\n'.join(self.log_buffer[-50:])  # 最近50行
        metrics = self.parse_training_metrics(recent_logs)
        
        status = f"""
训练状态摘要:
============
当前轮数: {metrics['current_epoch']}/{metrics['total_epochs']}
Box Loss: {metrics['box_loss']:.4f}
Class Loss: {metrics['cls_loss']:.4f}
DFL Loss: {metrics['dfl_loss']:.4f}
mAP50: {metrics['map50']:.4f}
mAP50-95: {metrics['map50_95']:.4f}
精确度: {metrics['precision']:.4f}
召回率: {metrics['recall']:.4f}
"""
        return status


class TrainingProgressTracker:
    """训练进度跟踪器"""
    
    def __init__(self):
        self.metrics_history = []
        self.start_time = None
        self.last_update = None
    
    def update_metrics(self, metrics: Dict):
        """更新训练指标"""
        current_time = datetime.now()
        
        if self.start_time is None:
            self.start_time = current_time
        
        metrics['timestamp'] = current_time
        metrics['elapsed_time'] = (current_time - self.start_time).total_seconds()
        
        self.metrics_history.append(metrics)
        self.last_update = current_time
    
    def get_progress_summary(self) -> Dict:
        """获取进度摘要"""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        latest = self.metrics_history[-1]
        
        # 计算训练速度
        if len(self.metrics_history) > 1:
            time_diff = (latest['timestamp'] - self.metrics_history[-2]['timestamp']).total_seconds()
            epoch_speed = 1 / time_diff if time_diff > 0 else 0
        else:
            epoch_speed = 0
        
        # 估算剩余时间
        if latest['current_epoch'] > 0 and latest['total_epochs'] > latest['current_epoch']:
            remaining_epochs = latest['total_epochs'] - latest['current_epoch']
            estimated_remaining = remaining_epochs / epoch_speed if epoch_speed > 0 else 0
        else:
            estimated_remaining = 0
        
        return {
            "status": "training",
            "current_epoch": latest['current_epoch'],
            "total_epochs": latest['total_epochs'],
            "progress_percent": (latest['current_epoch'] / latest['total_epochs'] * 100) if latest['total_epochs'] > 0 else 0,
            "elapsed_time": latest['elapsed_time'],
            "estimated_remaining": estimated_remaining,
            "epoch_speed": epoch_speed,
            "latest_metrics": latest
        }
    
    def get_metrics_trend(self, metric_name: str, window_size: int = 10) -> List[float]:
        """获取指标趋势"""
        if len(self.metrics_history) < window_size:
            return [m.get(metric_name, 0) for m in self.metrics_history]
        
        return [m.get(metric_name, 0) for m in self.metrics_history[-window_size:]]


# 全局实例
real_time_monitor = RealTimeLogMonitor()
progress_tracker = TrainingProgressTracker()
