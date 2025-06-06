"""
设备管理模块
智能检测和管理CPU/GPU设备
"""

import torch
import platform
from typing import List, Dict, Optional


class DeviceManager:
    """设备管理器"""
    
    def __init__(self):
        self.available_devices = self._detect_devices()
        self.current_device = self._get_default_device()
    
    def _detect_devices(self) -> List[Dict]:
        """检测可用设备"""
        devices = []
        
        # 添加CPU设备
        devices.append({
            "name": "CPU",
            "type": "cpu",
            "id": "cpu",
            "memory": self._get_cpu_memory(),
            "available": True,
            "description": f"CPU ({platform.processor() or 'Unknown'})"
        })
        
        # 检测CUDA设备
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    device_name = torch.cuda.get_device_name(i)
                    device_memory = torch.cuda.get_device_properties(i).total_memory
                    devices.append({
                        "name": f"GPU {i}",
                        "type": "cuda",
                        "id": f"cuda:{i}",
                        "memory": device_memory,
                        "available": True,
                        "description": f"GPU {i}: {device_name} ({device_memory // (1024**3)}GB)"
                    })
                except Exception as e:
                    print(f"检测GPU {i}时出错: {e}")
        
        # 检测MPS设备 (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append({
                "name": "MPS",
                "type": "mps", 
                "id": "mps",
                "memory": "共享内存",
                "available": True,
                "description": "Apple Metal Performance Shaders (MPS)"
            })
        
        return devices
    
    def _get_cpu_memory(self) -> str:
        """获取CPU内存信息"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return f"{memory.total // (1024**3)}GB"
        except:
            return "Unknown"
    
    def _get_default_device(self) -> str:
        """获取默认设备"""
        # 优先级: CUDA > MPS > CPU
        for device in self.available_devices:
            if device["type"] == "cuda" and device["available"]:
                return device["id"]
        
        for device in self.available_devices:
            if device["type"] == "mps" and device["available"]:
                return device["id"]
        
        return "cpu"
    
    def get_device_choices(self) -> List[str]:
        """获取设备选择列表"""
        return [device["id"] for device in self.available_devices if device["available"]]
    
    def get_device_descriptions(self) -> List[str]:
        """获取设备描述列表"""
        return [device["description"] for device in self.available_devices if device["available"]]
    
    def get_device_info(self, device_id: str) -> Optional[Dict]:
        """获取指定设备信息"""
        for device in self.available_devices:
            if device["id"] == device_id:
                return device
        return None
    
    def is_gpu_available(self) -> bool:
        """检查是否有GPU可用"""
        return any(device["type"] in ["cuda", "mps"] for device in self.available_devices)
    
    def get_device_status(self) -> Dict:
        """获取设备状态信息"""
        status = {
            "current_device": self.current_device,
            "available_devices": len(self.available_devices),
            "gpu_available": self.is_gpu_available(),
            "devices": []
        }
        
        for device in self.available_devices:
            device_status = device.copy()
            
            # 添加实时状态信息
            if device["type"] == "cuda":
                try:
                    torch.cuda.set_device(device["id"])
                    device_status["memory_used"] = torch.cuda.memory_allocated()
                    device_status["memory_cached"] = torch.cuda.memory_reserved()
                    device_status["temperature"] = self._get_gpu_temperature(device["id"])
                except:
                    device_status["memory_used"] = "Unknown"
                    device_status["memory_cached"] = "Unknown"
                    device_status["temperature"] = "Unknown"
            
            status["devices"].append(device_status)
        
        return status
    
    def _get_gpu_temperature(self, device_id: str) -> str:
        """获取GPU温度（如果支持）"""
        try:
            # 这里可以添加GPU温度检测逻辑
            # 不同的GPU厂商有不同的API
            return "Unknown"
        except:
            return "Unknown"
    
    def set_device(self, device_id: str, auto_fallback: bool = True) -> bool:
        """设置当前设备，支持自动降级"""
        # 首先验证设备是否真实可用
        validation_result = self.validate_device_availability(device_id)
        
        if validation_result["available"]:
            # 设备可用，直接设置
            self.current_device = device_id
            
            # 设置PyTorch默认设备
            try:
                if device_id.startswith("cuda"):
                    torch.cuda.set_device(device_id)
                elif device_id == "mps":
                    # MPS设备设置
                    pass
                
                print(f"✅ 成功设置设备: {device_id}")
                return True
            except Exception as e:
                print(f"❌ 设置设备失败: {e}")
                if auto_fallback:
                    fallback_device = validation_result.get("fallback_device") or self.get_fallback_device(device_id)
                    print(f"🔄 尝试降级到设备: {fallback_device}")
                    return self.set_device(fallback_device, auto_fallback=False)
                return False
        else:
            # 设备不可用
            if validation_result["warnings"]:
                for warning in validation_result["warnings"]:
                    print(f"⚠️ {warning}")
            
            if auto_fallback and validation_result["fallback_device"]:
                fallback_device = validation_result["fallback_device"]
                print(f"🔄 自动降级到设备: {fallback_device}")
                return self.set_device(fallback_device, auto_fallback=False)
            
            return False
    
    def get_best_available_device(self) -> str:
        """获取当前最佳可用设备"""
        # 按优先级检查设备可用性
        priority_devices = []
        
        # 添加CUDA设备
        for device in self.available_devices:
            if device["type"] == "cuda":
                priority_devices.append(device["id"])
        
        # 添加MPS设备
        for device in self.available_devices:
            if device["type"] == "mps":
                priority_devices.append(device["id"])
        
        # 添加CPU设备
        priority_devices.append("cpu")
        
        # 逐个验证设备可用性
        for device_id in priority_devices:
            validation_result = self.validate_device_availability(device_id)
            if validation_result["available"]:
                return device_id
        
        # 如果所有设备都不可用，返回CPU作为最后的选择
        return "cpu"
    
    def get_optimal_batch_size(self, device_id: str, image_size: int = 640) -> int:
        """根据设备推荐批次大小"""
        device_info = self.get_device_info(device_id)
        if not device_info:
            return 16
        
        if device_info["type"] == "cpu":
            return 4  # CPU通常使用较小的批次
        elif device_info["type"] == "cuda":
            # 根据GPU内存推荐批次大小
            try:
                memory_gb = device_info["memory"] // (1024**3)
                if memory_gb >= 24:
                    return 32
                elif memory_gb >= 12:
                    return 16
                elif memory_gb >= 8:
                    return 8
                else:
                    return 4
            except:
                return 8
        elif device_info["type"] == "mps":
            return 8  # MPS设备的推荐批次
        
        return 16
    
    def validate_device_availability(self, device_id: str) -> Dict:
        """验证设备是否真实可用（运行时检查）"""
        result = {
            "available": False,
            "device_id": device_id,
            "warnings": [],
            "fallback_device": None
        }
        
        device_info = self.get_device_info(device_id)
        if not device_info:
            result["warnings"].append(f"设备 {device_id} 不存在")
            result["fallback_device"] = self.get_fallback_device(device_id)
            return result
        
        try:
            # 尝试创建测试张量来验证设备真实可用性
            if device_id == "cpu":
                test_tensor = torch.randn(2, 2)
                result["available"] = True
                
            elif device_id.startswith("cuda"):
                if torch.cuda.is_available():
                    test_tensor = torch.randn(2, 2).to(device_id)
                    result["available"] = True
                else:
                    result["warnings"].append("CUDA运行时不可用")
                    result["fallback_device"] = self.get_fallback_device(device_id)
                    
            elif device_id == "mps":
                if torch.backends.mps.is_available():
                    # 尝试创建MPS张量
                    test_tensor = torch.randn(2, 2).to("mps")
                    result["available"] = True
                else:
                    result["warnings"].append("MPS运行时不可用")
                    result["fallback_device"] = self.get_fallback_device(device_id)
            
        except Exception as e:
            result["warnings"].append(f"设备测试失败: {str(e)}")
            result["fallback_device"] = self.get_fallback_device(device_id)
        
        return result
    
    def get_fallback_device(self, failed_device_id: str) -> str:
        """获取降级设备"""
        # 设备降级优先级策略
        if failed_device_id.startswith("cuda"):
            # CUDA失败 -> MPS -> CPU
            for device in self.available_devices:
                if device["type"] == "mps" and device["available"]:
                    # 验证MPS是否真实可用
                    mps_check = self.validate_device_availability("mps")
                    if mps_check["available"]:
                        print(f"⚠️ CUDA设备不可用，降级到MPS设备")
                        return "mps"
            print(f"⚠️ CUDA和MPS设备都不可用，降级到CPU")
            return "cpu"
            
        elif failed_device_id == "mps":
            # MPS失败 -> CUDA -> CPU
            for device in self.available_devices:
                if device["type"] == "cuda" and device["available"]:
                    cuda_check = self.validate_device_availability(device["id"])
                    if cuda_check["available"]:
                        print(f"⚠️ MPS设备不可用，降级到CUDA设备: {device['id']}")
                        return device["id"]
            print(f"⚠️ MPS和CUDA设备都不可用，降级到CPU")
            return "cpu"
        
        # 其他情况或CPU失败，返回CPU（CPU应该总是可用的）
        return "cpu"
    
    def validate_device_compatibility(self, device_id: str) -> Dict:
        """验证设备兼容性"""
        result = {
            "compatible": False,
            "warnings": [],
            "recommendations": []
        }
        
        device_info = self.get_device_info(device_id)
        if not device_info:
            result["warnings"].append("设备不存在")
            return result
        
        if device_info["type"] == "cuda":
            # 检查CUDA版本兼容性
            try:
                cuda_version = torch.version.cuda
                if cuda_version:
                    result["compatible"] = True
                    result["recommendations"].append(f"CUDA版本: {cuda_version}")
                else:
                    result["warnings"].append("CUDA不可用")
            except:
                result["warnings"].append("无法检测CUDA版本")
        
        elif device_info["type"] == "mps":
            # 检查MPS兼容性
            try:
                if torch.backends.mps.is_available():
                    result["compatible"] = True
                    result["recommendations"].append("MPS加速可用")
                    
                    # 检查PyTorch版本是否支持MPS
                    pytorch_version = torch.__version__
                    result["recommendations"].append(f"PyTorch版本: {pytorch_version}")
                    
                    # 检查macOS版本（MPS需要macOS 12.3+）
                    try:
                        import platform
                        macos_version = platform.mac_ver()[0]
                        if macos_version:
                            result["recommendations"].append(f"macOS版本: {macos_version}")
                    except:
                        pass
                else:
                    result["warnings"].append("MPS不可用")
            except:
                result["warnings"].append("无法检测MPS支持")
        
        else:  # CPU
            result["compatible"] = True
            result["recommendations"].append("CPU训练稳定但速度较慢")
        
        return result


# 全局设备管理器实例
device_manager = DeviceManager()


def get_device_choices_for_gradio():
    """为Gradio获取设备选择"""
    choices = device_manager.get_device_choices()
    descriptions = device_manager.get_device_descriptions()
    
    # 创建选择项，格式为 "device_id - description"
    formatted_choices = []
    for choice, desc in zip(choices, descriptions):
        formatted_choices.append(f"{choice} - {desc}")
    
    return formatted_choices


def parse_device_choice(choice_str: str) -> str:
    """解析Gradio设备选择"""
    if " - " in choice_str:
        return choice_str.split(" - ")[0]
    return choice_str


def get_recommended_settings(device_id: str, image_size: int = 640) -> Dict:
    """获取设备推荐设置"""
    return {
        "batch_size": device_manager.get_optimal_batch_size(device_id, image_size),
        "workers": 4 if device_id == "cpu" else 8,
        "mixed_precision": device_id.startswith("cuda"),
        "pin_memory": device_id != "cpu"
    }
