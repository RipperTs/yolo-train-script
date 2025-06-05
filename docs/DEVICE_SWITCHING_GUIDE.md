# 设备切换功能使用指南

## 🎯 功能概述

YOLOv8训练系统现在支持智能的CPU/GPU设备切换功能，能够自动检测可用的计算设备并提供友好的切换界面。

## 🔍 支持的设备类型

### 1. CPU设备
- **标识**: `cpu`
- **适用场景**: 兼容性最好，适合小规模训练和测试
- **推荐批次大小**: 4
- **特点**: 稳定可靠，但训练速度较慢

### 2. CUDA设备 (NVIDIA GPU)
- **标识**: `cuda:0`, `cuda:1`, ...
- **适用场景**: 大规模训练，高性能计算
- **推荐批次大小**: 根据显存自动调整 (4-32)
- **特点**: 训练速度快，需要NVIDIA GPU和CUDA支持

### 3. MPS设备 (Apple Silicon)
- **标识**: `mps`
- **适用场景**: Apple M1/M2芯片的GPU加速
- **推荐批次大小**: 8
- **特点**: Apple设备的GPU加速，性能优于CPU

## 🚀 使用方法

### 在Gradio界面中切换设备

#### 1. 训练配置页面
1. 进入"模型训练"标签页
2. 在"训练配置"部分找到"训练设备"下拉菜单
3. 选择想要使用的设备
4. 系统会自动调整推荐的批次大小
5. 查看"设备信息"了解设备状态

#### 2. 配置管理页面
1. 进入"配置管理"标签页
2. 在"快速配置"部分选择"训练设备"
3. 点击"保存快速配置"应用设置
4. 在"设备信息"部分查看详细状态

### 设备自动检测

系统启动时会自动检测：
- ✅ CPU设备（始终可用）
- ✅ CUDA设备（如果有NVIDIA GPU）
- ✅ MPS设备（如果是Apple Silicon）

### 智能推荐

切换设备时，系统会自动：
- 🔄 调整推荐的批次大小
- ⚙️ 优化训练参数
- 📊 显示设备性能信息
- ⚠️ 提供兼容性警告

## 📊 设备性能对比

| 设备类型 | 训练速度 | 内存使用 | 兼容性 | 推荐场景 |
|---------|---------|---------|--------|----------|
| CPU | 慢 | 低 | 最好 | 测试、小数据集 |
| CUDA | 快 | 高 | 需要NVIDIA | 大规模训练 |
| MPS | 中等 | 中等 | Apple设备 | Apple用户训练 |

## 🔧 技术实现

### 设备检测机制
```python
# 自动检测可用设备
devices = []
devices.append({"name": "CPU", "type": "cpu"})

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        devices.append({"name": f"GPU {i}", "type": "cuda"})

if torch.backends.mps.is_available():
    devices.append({"name": "MPS", "type": "mps"})
```

### 智能批次大小推荐
```python
def get_optimal_batch_size(device_id, image_size=640):
    if device_id == "cpu":
        return 4
    elif device_id.startswith("cuda"):
        # 根据GPU内存推荐
        memory_gb = get_gpu_memory() // (1024**3)
        if memory_gb >= 24: return 32
        elif memory_gb >= 12: return 16
        elif memory_gb >= 8: return 8
        else: return 4
    elif device_id == "mps":
        return 8
```

## 🎮 测试功能

### 设备测试界面
运行以下命令启动设备测试界面：
```bash
python gradio_device_test.py
```

访问: http://127.0.0.1:7863

### 测试功能包括：
- 🔄 设备切换测试
- 📊 设备信息显示
- 🧪 性能基准测试
- ⚙️ 配置验证

### 命令行测试
```bash
python test_device_manager.py
```

## 📋 使用示例

### 示例1: 从CPU切换到GPU
1. 默认使用CPU设备
2. 在设备下拉菜单中选择"cuda:0 - NVIDIA GeForce RTX 3080"
3. 系统自动将批次大小从4调整到16
4. 显示GPU内存和温度信息

### 示例2: Apple Silicon用户
1. 系统自动检测到MPS设备
2. 默认选择"mps - Apple Metal Performance Shaders"
3. 推荐批次大小为8
4. 享受GPU加速训练

### 示例3: 兼容性检查
1. 选择设备后自动验证兼容性
2. 显示CUDA版本信息
3. 提供优化建议
4. 警告潜在问题

## ⚠️ 注意事项

### 设备兼容性
- **CUDA设备**: 需要安装CUDA和对应的PyTorch版本
- **MPS设备**: 需要macOS 12.3+和Apple Silicon芯片
- **CPU设备**: 无特殊要求，始终可用

### 内存管理
- GPU训练时注意显存使用
- 批次大小过大可能导致内存不足
- 系统会提供内存使用监控

### 性能优化
- 首次使用GPU可能需要预热
- 混合精度训练可以节省内存
- 数据加载器的worker数量会自动调整

## 🔮 未来功能

### 计划中的增强功能
- 🔄 多GPU并行训练支持
- 📊 实时性能监控图表
- 🎯 自动设备选择算法
- 🔧 设备特定的优化配置
- 📈 训练速度对比分析

### 高级功能
- 分布式训练支持
- 云端GPU集成
- 设备负载均衡
- 智能资源调度

## 🆘 故障排除

### 常见问题

#### 1. CUDA设备不可用
**问题**: 显示"CUDA不可用"
**解决**: 
- 检查NVIDIA驱动安装
- 验证CUDA版本兼容性
- 重新安装PyTorch CUDA版本

#### 2. MPS设备不可用
**问题**: Apple设备上MPS不可用
**解决**:
- 确认macOS版本 >= 12.3
- 检查是否为Apple Silicon芯片
- 更新PyTorch到最新版本

#### 3. 内存不足错误
**问题**: GPU内存不足
**解决**:
- 减小批次大小
- 降低图片分辨率
- 启用梯度累积

#### 4. 设备切换失败
**问题**: 无法切换到指定设备
**解决**:
- 检查设备是否真实存在
- 验证驱动程序状态
- 重启应用程序

### 调试命令
```bash
# 检查PyTorch设备支持
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 检查MPS支持
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# 运行设备测试
python test_device_manager.py
```

## 📞 技术支持

如果遇到设备相关问题：
1. 查看本指南的故障排除部分
2. 运行设备测试脚本诊断问题
3. 检查系统日志和错误信息
4. 确认硬件和驱动程序状态

---

🎉 享受智能设备切换带来的便利！现在您可以轻松在不同计算设备间切换，获得最佳的训练性能。
