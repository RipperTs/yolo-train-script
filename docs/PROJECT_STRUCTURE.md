# YOLOv8 训练系统项目结构

## 📁 项目目录结构

```
yolo-train/
├── README.md                    # 项目主要说明文档
├── requirements.txt             # Python依赖列表
├── dataset.yaml                 # YOLO数据集配置文件
├── 
├── 🔧 核心模块
├── config.py                    # 项目配置文件
├── data_converter.py            # 数据格式转换器
├── trainer.py                   # 标准训练器
├── smart_trainer.py             # 智能训练器
├── inference.py                 # 模型推理器
├── utils.py                     # 工具函数
├── 
├── 🎨 前端界面
├── gradio_app.py               # 主要Gradio应用
├── gradio_utils.py             # Gradio工具函数
├── start_gradio.py             # Gradio启动脚本
├── 
├── ⚙️ 管理模块
├── config_manager.py           # 配置管理器
├── device_manager.py           # 设备管理器
├── dataset_manager.py          # 数据集管理器
├── log_monitor.py              # 日志监控器
├── 
├── 📊 数据目录
├── labeling_data/              # 原始标注数据（JSON格式）
├── datasets/                   # 转换后的YOLO数据集
│   ├── images/                 # 图片文件
│   │   ├── train/             # 训练集图片
│   │   ├── val/               # 验证集图片
│   │   └── test/              # 测试集图片
│   └── labels/                # 标签文件
│       ├── train/             # 训练集标签
│       ├── val/               # 验证集标签
│       └── test/              # 测试集标签
├── 
├── 🤖 模型目录
├── models/                     # 训练输出的模型
├── YOLOv8/                    # 预训练模型
├── logs/                      # 训练日志
├── 
├── 📚 文档目录
├── docs/                      # 项目文档
│   ├── GRADIO_README.md       # Gradio界面使用说明
│   ├── GRADIO_SUMMARY.md      # Gradio功能总结
│   ├── DEVICE_SWITCHING_GUIDE.md # 设备切换指南
│   ├── PROJECT_STRUCTURE.md   # 项目结构说明（本文件）
│   ├── 使用说明.md             # 中文使用说明
│   └── 智能训练快速参考.md      # 智能训练参考
├── 
└── 🧪 测试目录
    └── tests/                  # 测试脚本
        ├── test_setup.py       # 环境测试
        ├── test_device_manager.py # 设备管理测试
        ├── test_dataset_directory.py # 数据集目录测试
        ├── test_directory_selection.py # 目录选择界面测试
        ├── demo_gradio.py      # Gradio功能演示
        ├── gradio_simple.py    # 简化版Gradio应用
        ├── gradio_minimal.py   # 最小化Gradio应用
        ├── gradio_device_test.py # 设备测试界面
        └── launch_gradio_stable.py # 稳定版启动脚本
```

## 🔧 核心模块说明

### 配置和数据处理
- **config.py**: 项目全局配置，包括路径、训练参数等
- **data_converter.py**: JSON标注转YOLO格式，支持动态源目录
- **utils.py**: 通用工具函数，数据集检查、可视化等

### 训练和推理
- **trainer.py**: 标准YOLO训练流程
- **smart_trainer.py**: 智能训练，自动监控指标达标停止
- **inference.py**: 模型推理，支持单图和批量处理

### 前端界面
- **gradio_app.py**: 主要的Web界面，包含所有功能模块
- **gradio_utils.py**: Gradio相关的工具类和监控器
- **start_gradio.py**: 标准启动脚本

### 管理模块
- **config_manager.py**: 动态配置管理，支持保存和加载
- **device_manager.py**: CPU/GPU设备检测和切换
- **dataset_manager.py**: 数据集目录管理和验证
- **log_monitor.py**: 实时日志监控和解析

## 🎯 主要功能模块

### 1. 数据管理
- 📁 **动态目录选择**: 支持自定义数据源目录
- 🔄 **格式转换**: JSON标注转YOLO格式
- ✅ **数据验证**: 检查数据集完整性
- 📊 **数据可视化**: 数据分布和样本预览

### 2. 模型训练
- 🎯 **普通训练**: 传统的固定轮数训练
- 🤖 **智能训练**: 自动监控指标，达标停止
- 🔄 **恢复训练**: 从中断点继续训练
- ⚙️ **设备切换**: CPU/GPU/MPS智能切换

### 3. 模型推理
- 🔍 **单图推理**: 上传图片进行检测
- 📁 **批量推理**: 批量处理多张图片
- 📊 **结果可视化**: 检测结果可视化展示
- 💾 **结果导出**: JSON格式结果导出

### 4. 训练监控
- 📈 **实时日志**: 训练过程实时日志显示
- 📊 **训练曲线**: 动态生成loss和mAP曲线
- 🖥️ **系统监控**: CPU、内存、GPU使用率
- 📋 **状态分析**: 训练状态和进度分析

### 5. 配置管理
- ⚙️ **参数配置**: 图形化配置训练参数
- 💾 **配置持久化**: 自动保存和加载配置
- 📤 **配置导入导出**: 配置文件分享和备份
- 🎛️ **快速配置**: 预设的常用配置模板

## 🚀 快速开始

### 1. 环境准备
```bash
# 激活conda环境
conda activate yolo-train

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动界面
```bash
# 启动主界面
python start_gradio.py

# 访问界面
# http://localhost:7860
```

### 3. 基本流程
1. **数据准备**: 在"数据管理"中选择数据源目录
2. **数据转换**: 将JSON标注转换为YOLO格式
3. **配置训练**: 在"模型训练"中设置参数和设备
4. **开始训练**: 选择普通训练或智能训练
5. **监控进度**: 在"训练监控"中查看实时状态
6. **模型推理**: 在"模型推理"中测试训练好的模型

## 🧪 测试和开发

### 运行测试
```bash
# 环境测试
python tests/test_setup.py

# 设备管理测试
python tests/test_device_manager.py

# 数据集目录测试
python tests/test_dataset_directory.py

# 功能演示
python tests/demo_gradio.py
```

### 开发调试
```bash
# 简化版界面（调试用）
python tests/gradio_simple.py

# 设备测试界面
python tests/gradio_device_test.py

# 目录选择测试界面
python tests/test_directory_selection.py
```

## 📋 配置文件

### 自动生成的配置文件
- `gradio_config.json` - Gradio界面配置
- `dataset_config.json` - 数据集目录配置
- `demo_config.json` - 演示配置

### 数据集配置
- `dataset.yaml` - YOLO数据集配置文件

## 🔧 扩展开发

### 添加新功能模块
1. 在项目根目录创建模块文件
2. 在`gradio_app.py`中添加对应的标签页
3. 在`tests/`中创建测试脚本
4. 更新文档说明

### 自定义配置
1. 修改`config.py`中的默认配置
2. 使用`config_manager.py`进行动态配置
3. 通过Gradio界面进行实时调整

## 📞 技术支持

### 文档资源
- `docs/GRADIO_README.md` - 详细使用说明
- `docs/DEVICE_SWITCHING_GUIDE.md` - 设备切换指南
- `docs/使用说明.md` - 中文使用指南

### 故障排除
1. 查看相关文档的故障排除部分
2. 运行对应的测试脚本诊断问题
3. 检查日志文件和错误信息

---

📝 **注意**: 本项目采用模块化设计，各个功能模块相对独立，便于维护和扩展。所有的用户界面都通过Gradio提供，支持本地和远程访问。
