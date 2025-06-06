"""
内存优化模块
解决训练过程中内存占用持续增大的问题
"""

import gc
import torch
import psutil
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self, enable_monitoring=True, cleanup_interval=10):
        """
        初始化内存优化器
        
        Args:
            enable_monitoring: 是否启用内存监控
            cleanup_interval: 清理间隔（秒）
        """
        self.enable_monitoring = enable_monitoring
        self.cleanup_interval = cleanup_interval
        self.monitoring_thread = None
        self.is_monitoring = False
        self.memory_history = []
        self.max_history_size = 100
        
    def start_monitoring(self):
        """开始内存监控"""
        if self.enable_monitoring and not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitor_memory)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("🔍 内存监控已启动")
    
    def stop_monitoring(self):
        """停止内存监控"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("⏹️ 内存监控已停止")
    
    def _monitor_memory(self):
        """内存监控线程"""
        while self.is_monitoring:
            try:
                memory_info = self.get_memory_info()
                self.memory_history.append(memory_info)
                
                # 限制历史记录大小
                if len(self.memory_history) > self.max_history_size:
                    self.memory_history.pop(0)
                
                # 检查内存使用情况
                if memory_info['ram_percent'] > 85:
                    logger.warning(f"⚠️ 内存使用率过高: {memory_info['ram_percent']:.1f}%")
                    self.force_cleanup()
                
                # 检查GPU内存（如果可用）
                if memory_info.get('gpu_memory_percent', 0) > 90:
                    logger.warning(f"⚠️ GPU内存使用率过高: {memory_info['gpu_memory_percent']:.1f}%")
                    self.cleanup_gpu_memory()
                
                time.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"内存监控出错: {e}")
                time.sleep(self.cleanup_interval)
    
    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存信息"""
        # 系统内存
        memory = psutil.virtual_memory()
        info = {
            'ram_used_gb': memory.used / (1024**3),
            'ram_total_gb': memory.total / (1024**3),
            'ram_percent': memory.percent,
            'timestamp': time.time()
        }
        
        # GPU内存（CUDA）
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_stats()
                allocated = gpu_memory.get('allocated_bytes.all.current', 0)
                reserved = gpu_memory.get('reserved_bytes.all.current', 0)
                total = torch.cuda.get_device_properties(0).total_memory
                
                info.update({
                    'gpu_allocated_gb': allocated / (1024**3),
                    'gpu_reserved_gb': reserved / (1024**3),
                    'gpu_total_gb': total / (1024**3),
                    'gpu_memory_percent': (reserved / total) * 100 if total > 0 else 0
                })
            except Exception as e:
                logger.debug(f"获取GPU内存信息失败: {e}")
        
        # MPS内存（Apple Silicon）
        if torch.backends.mps.is_available():
            try:
                # MPS内存信息较难获取，使用系统内存作为近似
                info['mps_available'] = True
            except Exception as e:
                logger.debug(f"获取MPS内存信息失败: {e}")
        
        return info
    
    def cleanup_python_memory(self):
        """清理Python内存"""
        try:
            # 强制垃圾回收
            collected = gc.collect()
            logger.debug(f"🧹 Python垃圾回收: 清理了 {collected} 个对象")
            
            # 清理未引用的循环
            gc.collect()
            
        except Exception as e:
            logger.error(f"Python内存清理失败: {e}")
    
    def cleanup_gpu_memory(self):
        """清理GPU内存"""
        try:
            if torch.cuda.is_available():
                # 清空CUDA缓存
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("🧹 CUDA内存缓存已清理")
            
            if torch.backends.mps.is_available():
                # MPS内存清理
                torch.mps.empty_cache()
                logger.debug("🧹 MPS内存缓存已清理")
                
        except Exception as e:
            logger.error(f"GPU内存清理失败: {e}")
    
    def force_cleanup(self):
        """强制内存清理"""
        logger.info("🧹 执行强制内存清理...")
        
        # 清理Python内存
        self.cleanup_python_memory()
        
        # 清理GPU内存
        self.cleanup_gpu_memory()
        
        # 再次垃圾回收
        gc.collect()
        
        logger.info("✅ 强制内存清理完成")
    
    def optimize_training_config(self, device: str, current_config: Dict) -> Dict:
        """根据内存情况优化训练配置"""
        memory_info = self.get_memory_info()
        optimized_config = current_config.copy()
        
        # 根据内存使用情况调整批次大小
        if memory_info['ram_percent'] > 80:
            # 内存紧张，减小批次大小
            current_batch = optimized_config.get('batch_size', 16)
            new_batch = max(1, current_batch // 2)
            optimized_config['batch_size'] = new_batch
            logger.warning(f"⚠️ 内存紧张，批次大小从 {current_batch} 调整为 {new_batch}")
        
        # 根据设备类型优化
        if device == "mps":
            # MPS设备优化
            optimized_config.update({
                'workers': min(4, optimized_config.get('workers', 8)),  # 限制worker数量
                'pin_memory': False,  # MPS不需要pin_memory
                'persistent_workers': True,  # 使用持久化worker
                'cache': False,  # 禁用数据集缓存以节省内存
            })
        elif device.startswith("cuda"):
            # CUDA设备优化
            gpu_memory_percent = memory_info.get('gpu_memory_percent', 0)
            if gpu_memory_percent > 80:
                # GPU内存紧张
                current_batch = optimized_config.get('batch_size', 16)
                new_batch = max(1, current_batch // 2)
                optimized_config['batch_size'] = new_batch
                optimized_config['cache'] = False  # 禁用缓存
                logger.warning(f"⚠️ GPU内存紧张，批次大小调整为 {new_batch}")
        else:  # CPU
            # CPU设备优化
            optimized_config.update({
                'workers': min(2, optimized_config.get('workers', 8)),  # CPU限制worker数量
                'cache': False,  # CPU内存有限，禁用缓存
                'pin_memory': False,  # CPU不需要pin_memory
            })
        
        return optimized_config
    
    def get_memory_report(self) -> str:
        """生成内存使用报告"""
        if not self.memory_history:
            return "📊 暂无内存监控数据"
        
        current = self.memory_history[-1]
        report = [
            "📊 内存使用报告",
            "=" * 40,
            f"🖥️  系统内存: {current['ram_used_gb']:.1f}GB / {current['ram_total_gb']:.1f}GB ({current['ram_percent']:.1f}%)"
        ]
        
        if 'gpu_allocated_gb' in current:
            report.append(f"🎮 GPU内存: {current['gpu_allocated_gb']:.1f}GB / {current['gpu_total_gb']:.1f}GB ({current['gpu_memory_percent']:.1f}%)")
        
        if 'mps_available' in current:
            report.append("🍎 MPS设备: 可用")
        
        # 内存趋势分析
        if len(self.memory_history) >= 2:
            prev = self.memory_history[-2]
            ram_trend = current['ram_percent'] - prev['ram_percent']
            trend_symbol = "📈" if ram_trend > 0 else "📉" if ram_trend < 0 else "➡️"
            report.append(f"{trend_symbol} 内存趋势: {ram_trend:+.1f}%")
        
        return "\n".join(report)


class TrainingMemoryManager:
    """训练内存管理器"""
    
    def __init__(self, optimizer: MemoryOptimizer):
        self.optimizer = optimizer
        self.epoch_cleanup_interval = 5  # 每5个epoch清理一次
        self.last_cleanup_epoch = 0
    
    def pre_training_setup(self, config: Dict) -> Dict:
        """训练前设置"""
        logger.info("🚀 训练前内存优化设置...")
        
        # 启动内存监控
        self.optimizer.start_monitoring()
        
        # 初始内存清理
        self.optimizer.force_cleanup()
        
        # 优化配置
        device = config.get('device', 'cpu')
        optimized_config = self.optimizer.optimize_training_config(device, config)
        
        logger.info("✅ 训练前内存优化完成")
        return optimized_config
    
    def post_training_cleanup(self):
        """训练后清理"""
        logger.info("🧹 训练后内存清理...")
        
        # 停止监控
        self.optimizer.stop_monitoring()
        
        # 强制清理
        self.optimizer.force_cleanup()
        
        # 生成报告
        report = self.optimizer.get_memory_report()
        logger.info(f"\n{report}")
        
        logger.info("✅ 训练后内存清理完成")
    
    def epoch_cleanup(self, current_epoch: int):
        """每个epoch的内存清理"""
        if current_epoch - self.last_cleanup_epoch >= self.epoch_cleanup_interval:
            logger.debug(f"🧹 Epoch {current_epoch} 内存清理...")
            self.optimizer.cleanup_python_memory()
            self.optimizer.cleanup_gpu_memory()
            self.last_cleanup_epoch = current_epoch


# 全局内存优化器实例
memory_optimizer = MemoryOptimizer()
training_memory_manager = TrainingMemoryManager(memory_optimizer)


def optimize_ultralytics_training_args(train_args: Dict, device: str) -> Dict:
    """优化Ultralytics训练参数以减少内存使用"""
    optimized_args = train_args.copy()
    
    # 基础内存优化参数
    memory_optimizations = {
        'cache': False,  # 禁用数据集缓存
        'save_period': max(10, train_args.get('save_period', 10)),  # 减少保存频率
        'plots': False,  # 禁用训练图表生成
        'val': True,  # 保持验证但可能减少频率
    }
    
    # 设备特定优化
    if device == "mps":
        memory_optimizations.update({
            'amp': False,  # MPS可能不完全支持混合精度
            'workers': min(4, train_args.get('workers', 8)),
        })
    elif device.startswith("cuda"):
        memory_optimizations.update({
            'amp': True,  # CUDA支持混合精度，可以节省内存
        })
    else:  # CPU
        memory_optimizations.update({
            'amp': False,  # CPU不支持混合精度
            'workers': min(2, train_args.get('workers', 8)),
        })
    
    optimized_args.update(memory_optimizations)
    return optimized_args


if __name__ == "__main__":
    # 测试内存优化器
    optimizer = MemoryOptimizer()
    
    print("🔍 开始内存监控测试...")
    optimizer.start_monitoring()
    
    # 模拟一些内存使用
    time.sleep(5)
    
    # 生成报告
    print(optimizer.get_memory_report())
    
    # 停止监控
    optimizer.stop_monitoring()
    
    print("✅ 内存优化器测试完成")
