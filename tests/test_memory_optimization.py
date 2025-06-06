#!/usr/bin/env python3
"""
内存优化功能测试
验证内存优化模块的各项功能
"""

import sys
import time
import gc
import torch
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from memory_optimizer import MemoryOptimizer, TrainingMemoryManager, optimize_ultralytics_training_args
from config import TRAINING_CONFIG


class MemoryOptimizationTester:
    """内存优化测试器"""
    
    def __init__(self):
        self.optimizer = MemoryOptimizer()
        self.manager = TrainingMemoryManager(self.optimizer)
        self.test_results = {}
    
    def test_memory_info(self):
        """测试内存信息获取"""
        print("🔍 测试内存信息获取...")
        
        try:
            memory_info = self.optimizer.get_memory_info()
            
            # 检查必需字段
            required_fields = ['ram_used_gb', 'ram_total_gb', 'ram_percent', 'timestamp']
            for field in required_fields:
                assert field in memory_info, f"缺少字段: {field}"
            
            # 检查数值合理性
            assert 0 <= memory_info['ram_percent'] <= 100, "RAM使用率超出合理范围"
            assert memory_info['ram_used_gb'] >= 0, "RAM使用量不能为负"
            assert memory_info['ram_total_gb'] > 0, "RAM总量必须大于0"
            
            print(f"✅ 内存信息获取正常: RAM {memory_info['ram_used_gb']:.1f}/{memory_info['ram_total_gb']:.1f}GB ({memory_info['ram_percent']:.1f}%)")
            
            # 检查GPU信息（如果可用）
            if torch.cuda.is_available():
                assert 'gpu_allocated_gb' in memory_info, "CUDA可用但缺少GPU内存信息"
                print(f"✅ GPU内存信息正常: {memory_info['gpu_allocated_gb']:.1f}GB")
            
            if torch.backends.mps.is_available():
                print("✅ MPS设备可用")
            
            self.test_results['memory_info'] = True
            return True
            
        except Exception as e:
            print(f"❌ 内存信息获取失败: {e}")
            self.test_results['memory_info'] = False
            return False
    
    def test_memory_cleanup(self):
        """测试内存清理功能"""
        print("🧹 测试内存清理功能...")
        
        try:
            # 获取清理前的内存信息
            before_info = self.optimizer.get_memory_info()
            
            # 创建一些内存占用
            test_data = []
            for i in range(1000):
                test_data.append([0] * 1000)  # 创建一些内存占用
            
            # 执行清理
            self.optimizer.cleanup_python_memory()
            self.optimizer.cleanup_gpu_memory()
            
            # 删除测试数据
            del test_data
            gc.collect()
            
            # 获取清理后的内存信息
            after_info = self.optimizer.get_memory_info()
            
            print(f"✅ 内存清理完成")
            print(f"   清理前: {before_info['ram_percent']:.1f}%")
            print(f"   清理后: {after_info['ram_percent']:.1f}%")
            
            self.test_results['memory_cleanup'] = True
            return True
            
        except Exception as e:
            print(f"❌ 内存清理测试失败: {e}")
            self.test_results['memory_cleanup'] = False
            return False
    
    def test_memory_monitoring(self):
        """测试内存监控功能"""
        print("📊 测试内存监控功能...")
        
        try:
            # 启动监控
            self.optimizer.start_monitoring()
            
            # 等待一段时间收集数据
            time.sleep(10)
            
            # 检查是否有监控数据
            assert len(self.optimizer.memory_history) > 0, "没有收集到监控数据"
            
            # 停止监控
            self.optimizer.stop_monitoring()
            
            # 生成报告
            report = self.optimizer.get_memory_report()
            assert "内存使用报告" in report, "内存报告格式不正确"
            
            print(f"✅ 内存监控正常，收集了 {len(self.optimizer.memory_history)} 个数据点")
            print(f"📋 报告预览:\n{report}")
            
            self.test_results['memory_monitoring'] = True
            return True
            
        except Exception as e:
            print(f"❌ 内存监控测试失败: {e}")
            self.optimizer.stop_monitoring()  # 确保停止监控
            self.test_results['memory_monitoring'] = False
            return False
    
    def test_config_optimization(self):
        """测试配置优化功能"""
        print("⚙️ 测试配置优化功能...")
        
        try:
            # 测试不同设备的配置优化
            devices = ['cpu']
            if torch.cuda.is_available():
                devices.append('cuda:0')
            if torch.backends.mps.is_available():
                devices.append('mps')
            
            for device in devices:
                print(f"   测试设备: {device}")
                
                # 获取优化配置
                optimized_config = self.optimizer.optimize_training_config(device, TRAINING_CONFIG)
                
                # 检查配置合理性
                assert 'batch_size' in optimized_config, "缺少batch_size配置"
                assert optimized_config['batch_size'] > 0, "batch_size必须大于0"
                
                print(f"   ✅ {device} 配置优化正常: batch_size={optimized_config['batch_size']}")
            
            self.test_results['config_optimization'] = True
            return True
            
        except Exception as e:
            print(f"❌ 配置优化测试失败: {e}")
            self.test_results['config_optimization'] = False
            return False
    
    def test_ultralytics_optimization(self):
        """测试Ultralytics参数优化"""
        print("🎯 测试Ultralytics参数优化...")
        
        try:
            # 测试参数优化
            test_args = {
                'epochs': 100,
                'batch': 16,
                'workers': 8,
                'cache': True,
                'save_period': 5
            }
            
            devices = ['cpu']
            if torch.cuda.is_available():
                devices.append('cuda:0')
            if torch.backends.mps.is_available():
                devices.append('mps')
            
            for device in devices:
                optimized_args = optimize_ultralytics_training_args(test_args, device)
                
                # 检查优化结果
                assert 'cache' in optimized_args, "缺少cache配置"
                assert 'workers' in optimized_args, "缺少workers配置"
                
                print(f"   ✅ {device} Ultralytics参数优化正常")
                print(f"      cache: {optimized_args['cache']}")
                print(f"      workers: {optimized_args['workers']}")
            
            self.test_results['ultralytics_optimization'] = True
            return True
            
        except Exception as e:
            print(f"❌ Ultralytics参数优化测试失败: {e}")
            self.test_results['ultralytics_optimization'] = False
            return False
    
    def test_training_memory_manager(self):
        """测试训练内存管理器"""
        print("🎓 测试训练内存管理器...")
        
        try:
            # 测试训练前设置
            optimized_config = self.manager.pre_training_setup(TRAINING_CONFIG)
            
            # 检查配置
            assert isinstance(optimized_config, dict), "优化配置必须是字典"
            assert 'batch_size' in optimized_config, "缺少batch_size配置"
            
            # 模拟epoch清理
            self.manager.epoch_cleanup(5)
            self.manager.epoch_cleanup(10)  # 应该触发清理
            
            # 测试训练后清理
            self.manager.post_training_cleanup()
            
            print("✅ 训练内存管理器测试正常")
            
            self.test_results['training_memory_manager'] = True
            return True
            
        except Exception as e:
            print(f"❌ 训练内存管理器测试失败: {e}")
            self.test_results['training_memory_manager'] = False
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始内存优化功能测试")
        print("=" * 60)
        
        tests = [
            self.test_memory_info,
            self.test_memory_cleanup,
            self.test_memory_monitoring,
            self.test_config_optimization,
            self.test_ultralytics_optimization,
            self.test_training_memory_manager
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
                print()  # 空行分隔
            except Exception as e:
                print(f"❌ 测试异常: {e}")
                print()
        
        # 输出测试结果
        print("=" * 60)
        print("📋 测试结果汇总")
        print("=" * 60)
        
        for test_name, result in self.test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{test_name:25}: {status}")
        
        print(f"\n总体结果: {passed}/{total} 测试通过")
        
        if passed == total:
            print("🎉 所有内存优化功能测试通过！")
            return True
        else:
            print("⚠️ 部分测试失败，请检查相关功能")
            return False


def main():
    """主函数"""
    tester = MemoryOptimizationTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ 内存优化功能正常，可以安全使用")
        sys.exit(0)
    else:
        print("\n❌ 内存优化功能存在问题，请检查")
        sys.exit(1)


if __name__ == "__main__":
    main()
