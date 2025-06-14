# YOLOv8 训练系统 Gradio 前端界面

## 🌟 功能概述

本项目为YOLOv8训练系统提供了一个功能完整的Web前端界面，基于Gradio构建，提供直观的图形化操作界面。

### 主要功能模块

#### 📊 数据管理
- **数据转换**: 将JSON标注文件转换为YOLO格式
- **数据集检查**: 验证数据集完整性和格式正确性
- **数据可视化**: 展示数据集分布和样本预览
- **数据统计**: 实时显示训练/验证/测试集的图片和标签数量

#### 🎯 模型训练
- **普通训练**: 传统的YOLO训练模式，支持自定义参数
- **智能训练**: 自动监控训练进度，达到目标后自动停止
- **恢复训练**: 从中断点继续训练
- **实时配置**: 动态调整训练参数（学习率、批次大小等）

#### 🔍 模型推理
- **单图推理**: 上传单张图片进行目标检测
- **批量推理**: 批量处理多张图片
- **实时预览**: 可视化检测结果
- **结果导出**: 支持JSON格式结果导出

#### 📈 训练监控
- **实时日志**: 显示训练过程的实时日志输出
- **训练曲线**: 动态绘制loss和mAP曲线
- **状态监控**: 实时显示训练状态和进度
- **性能指标**: 展示详细的训练指标

#### ⚙️ 配置管理
- **参数配置**: 图形化配置训练和推理参数
- **配置保存**: 保存和加载自定义配置
- **快速配置**: 预设的常用配置模板
- **配置导入导出**: 支持配置文件的导入导出

#### 🛠️ 工具集
- **数据分布可视化**: 生成数据集分布图表
- **锚框计算**: 基于数据集计算最优锚框
- **环境检查**: 检查Python环境和依赖
- **系统监控**: 显示CPU、内存、磁盘使用情况

## 🚀 快速开始

### 1. 环境准备

确保已安装所需依赖：

```bash
# 激活conda环境
conda activate yolo-train

# 安装Gradio相关依赖
pip install gradio psutil
```

### 2. 启动界面

```bash
# 方法1: 使用启动脚本（推荐）
python start_gradio.py

# 方法2: 直接启动
python gradio_app.py
```

### 3. 访问界面

启动后在浏览器中访问：
- 本地访问: http://localhost:7860
- 网络访问: http://0.0.0.0:7860

## 📋 使用指南

### 数据管理流程

1. **数据转换**
   - 点击"数据管理"标签页
   - 确保`labeling_data`目录中有JSON标注文件
   - 点击"转换JSON数据到YOLO格式"按钮
   - 等待转换完成

2. **数据检查**
   - 点击"检查数据集完整性"按钮
   - 查看检查结果，确保没有错误

3. **数据预览**
   - 点击"刷新信息"查看数据集统计
   - 点击"刷新样本"预览样本图片

### 模型训练流程

1. **配置训练参数**
   - 在"模型训练"标签页设置训练参数
   - 训练轮数、批次大小、学习率等

2. **选择训练模式**
   - **普通训练**: 按设定轮数训练
   - **智能训练**: 自动监控，达到目标自动停止
   - **恢复训练**: 从上次中断点继续

3. **监控训练过程**
   - 切换到"训练监控"标签页
   - 查看实时日志和训练曲线
   - 监控训练状态和进度

### 模型推理流程

1. **选择模型**
   - 在"模型推理"标签页选择训练好的模型
   - 点击"刷新模型列表"获取最新模型

2. **配置推理参数**
   - 设置置信度阈值、IoU阈值等

3. **执行推理**
   - **单图推理**: 上传图片，点击"单图推理"
   - **批量推理**: 上传多张图片，点击"批量推理"

4. **查看结果**
   - 查看可视化的检测结果
   - 下载JSON格式的检测数据

## 🔧 高级功能

### 智能训练配置

智能训练模式支持以下自定义目标：

- **目标mAP50**: 期望达到的mAP50值
- **目标Box Loss**: 期望的边界框损失值
- **目标Class Loss**: 期望的分类损失值
- **最大训练轮数**: 防止过度训练的安全限制

### 实时监控功能

- **日志监控**: 实时显示训练日志，支持关键词高亮
- **训练曲线**: 动态更新的loss和mAP曲线图
- **系统监控**: CPU、内存、磁盘使用率监控
- **训练状态**: 当前epoch、剩余时间估算

### 配置管理

- **配置持久化**: 自动保存用户配置
- **配置模板**: 预设的训练配置模板
- **配置导入导出**: 支持配置文件的分享和备份

## 🎨 界面特性

- **响应式设计**: 适配不同屏幕尺寸
- **实时更新**: 数据和状态实时刷新
- **友好提示**: 详细的操作提示和错误信息
- **主题支持**: 支持明暗主题切换
- **多语言**: 中文界面，易于理解

## 🔍 故障排除

### 常见问题

1. **界面无法启动**
   - 检查Gradio是否正确安装
   - 确认端口7860未被占用
   - 查看终端错误信息

2. **训练无法开始**
   - 确认数据集已正确转换
   - 检查配置参数是否合理
   - 查看训练日志中的错误信息

3. **推理失败**
   - 确认模型文件存在
   - 检查图片格式是否支持
   - 验证推理参数设置

### 日志查看

- 应用日志: 查看终端输出
- 训练日志: 在"训练监控"标签页查看
- 错误日志: 检查`logs`目录下的日志文件

## 📞 技术支持

如遇到问题，请：

1. 查看本文档的故障排除部分
2. 检查终端输出的错误信息
3. 确认环境配置是否正确
4. 查看项目的其他文档文件

## 🎯 最佳实践

1. **数据准备**: 确保标注数据质量，定期检查数据集完整性
2. **训练监控**: 使用智能训练模式，避免过拟合
3. **参数调优**: 根据数据集特点调整训练参数
4. **模型验证**: 训练完成后及时进行推理测试
5. **配置备份**: 定期导出和备份训练配置

---

🎉 享受使用YOLOv8训练系统的Gradio界面！
