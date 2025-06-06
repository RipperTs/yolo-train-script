# YOLOv8 一站式训练程序


## 系统截图
![iShot_2025-06-05_16.34.20.png](docs/img/iShot_2025-06-05_16.34.20.png)

![iShot_2025-06-05_16.34.44.png](docs/img/iShot_2025-06-05_16.34.44.png)

![iShot_2025-06-05_16.36.38.png](docs/img/iShot_2025-06-05_16.36.38.png)

## 安装依赖

首先安装必要的Python包：

```bash
pip install ultralytics opencv-python pillow matplotlib pandas scikit-learn gradio psutil
```

## 快速开始

### 1. Gradio Web界面（推荐）

启动可视化Web界面，提供完整的训练、推理和监控功能：

```bash
python start_gradio.py
```

启动后访问：http://localhost:7860

**Gradio界面功能**：
- 📊 **数据管理**: 数据转换、数据集检查、样本预览
- 🎯 **模型训练**: 普通训练、智能训练、恢复训练
- 🔍 **模型推理**: 单图推理、批量推理、实时预览
- 📈 **训练监控**: 实时日志、训练曲线、状态监控
- ⚙️ **配置管理**: 参数配置、快速设置、配置导入导出
- 🛠️ **工具集**: 数据分析、模型导出、环境检查

### 2. 完整流程（命令行）

运行完整的数据转换、训练和推理流程：

```bash
cd yolov8
python run.py --pipeline full
```

### 2. 分步执行

#### 步骤1: 数据转换

将`yolo_point`目录中的JSON标注文件转换为YOLO格式：

```bash
python run.py --pipeline convert
```

或者直接运行：

```bash
python data_converter.py
```

#### 步骤2: 训练模型

```bash
python run.py --pipeline train
```

或者直接运行：

```bash
python trainer.py --action train
```

#### 步骤3: 模型推理

单张图片推理：

```bash
python run.py --pipeline inference --image path/to/image.jpg
```

批量推理：

```bash
python run.py --pipeline inference --images path/to/images/
```

## 详细使用说明

### 数据转换

数据转换模块会：
1. 读取`yolo_point`目录中的JSON标注文件
2. 将标注转换为YOLO格式（归一化的边界框坐标）
3. 按照7:2:1的比例分割数据集为训练集、验证集和测试集
4. 复制图片文件到对应目录
5. 生成YOLO数据集配置文件

### 设备支持

系统支持多种计算设备，自动选择最佳可用设备：

- **🖥️ CPU**: 兼容性最好，适合小规模训练和测试
- **🚀 CUDA GPU**: 高性能训练，支持NVIDIA显卡
- **🍎 Apple MPS**: 苹果M芯片GPU加速，支持M1/M2/M3等芯片

**设备自动选择优先级**: CUDA > MPS > CPU

**MPS设备要求**:
- macOS 12.3+ 
- Apple Silicon芯片 (M1/M2/M3等)
- PyTorch 1.12+ 且支持MPS

### 训练配置

可以在`config.py`中修改训练参数：

```python
TRAINING_CONFIG = {
    "epochs": 100,           # 训练轮数
    "batch_size": 16,        # 批次大小
    "img_size": 640,         # 图片尺寸
    "learning_rate": 0.01,   # 学习率
    "patience": 50,          # 早停耐心值
    "device": None,          # 设备选择（None=自动选择）
}
```

**设备配置说明**:
- `None`: 自动选择最佳可用设备
- `"cpu"`: 强制使用CPU
- `"cuda:0"`: 使用指定的CUDA设备
- `"mps"`: 使用Apple MPS设备

### 推理配置

推理参数也可以在`config.py`中调整：

```python
INFERENCE_CONFIG = {
    "conf_threshold": 0.25,  # 置信度阈值
    "iou_threshold": 0.45,   # NMS IoU阈值
    "max_det": 1000,         # 最大检测数量
}
```

## 高级功能

### 1. 智能训练系统（推荐）

自动监控loss变化，智能决策是否继续训练：

```bash
# 智能训练循环 - 自动训练直到满意效果
python smart_trainer.py --action smart
```

**智能训练特点**：
- 🤖 **自动监控**: 实时分析loss和mAP变化趋势
- 🎯 **目标导向**: 根据预设阈值自动决策
- 🔄 **自动继续**: 性能未达标时自动继续训练
- ⏹️ **智能停止**: 达到目标或收敛时自动停止
- 📈 **可视化**: 自动生成训练曲线和分析报告

### 2. 交互式恢复训练

用户友好的训练继续界面：

```bash
python resume_training.py --action interactive
```

### 3. 自定义训练目标

根据应用需求设置性能标准：

```bash
# 高精度要求（生产环境推荐）
python smart_trainer.py --action smart \
  --box-loss 0.03 \      # 更严格的Box Loss要求
  --cls-loss 0.8 \       # 更严格的Class Loss要求
  --map50 0.8            # 更高的mAP50要求（80%）
```

### 4. 传统恢复训练

如果训练中断，可以恢复训练：

```bash
python run.py --pipeline train --resume
```

或指定特定的检查点：

```bash
python trainer.py --action train --resume --model path/to/checkpoint.pt
```

### 2. 模型验证

验证训练好的模型：

```bash
python trainer.py --action val --model path/to/model.pt
```

### 3. 模型导出

将模型导出为其他格式（如ONNX）：

```bash
python trainer.py --action export --model path/to/model.pt --format onnx
```

### 4. 设备测试

测试MPS设备支持（苹果M芯片用户）：

```bash
# 运行完整的MPS支持测试
python test_mps_support.py

# 测试特定功能
python test_mps_support.py --test pytorch    # 测试PyTorch MPS支持
python test_mps_support.py --test device     # 测试设备管理器
python test_mps_support.py --test config     # 测试配置管理器
python test_mps_support.py --test trainer    # 测试训练器
python test_mps_support.py --test inference  # 测试推理器
```

### 5. 数据集工具

检查数据集完整性：

```bash
python utils.py --action check
```

可视化数据集分布：

```bash
python utils.py --action distribution
```

可视化单张图片的标注：

```bash
python utils.py --action visualize --image path/to/image.jpg
```

计算锚框：

```bash
python utils.py --action anchors
```

### 6. 创建数据子集

从现有数据集创建小的子集用于快速测试：

```bash
python utils.py --action subset --source train --target debug --num 100
```

## 配置说明

### 类别配置

当前系统配置为检测一个类别"p"，如需添加更多类别，请修改`config.py`中的`CLASS_NAMES`：

```python
CLASS_NAMES = ["p", "other_class"]  # 添加新类别
```

### 路径配置

所有路径都在`config.py`中定义，可根据需要调整：

```python
YOLO_POINT_DIR = PROJECT_ROOT / "yolo_point"  # JSON标注目录
DATASETS_DIR = YOLO_ROOT / "datasets"         # 数据集目录
MODELS_DIR = YOLO_ROOT / "models"             # 模型目录
```

## 输出文件

### 训练输出

- 训练好的模型：`models/train_YYYYMMDD_HHMMSS/weights/best.pt`
- 训练日志和图表：`models/train_YYYYMMDD_HHMMSS/`
- TensorBoard日志：`logs/`

### 推理输出

- 预测结果图片：与输入图片同目录，文件名添加`_result`后缀
- JSON格式结果：包含检测框坐标、类别和置信度信息

## 故障排除

### 常见问题

1. **找不到图片文件**
   - 确保JSON文件中的`imagePath`字段正确
   - 检查图片文件是否存在于预期位置

2. **CUDA内存不足**
   - 减小`batch_size`
   - 减小`img_size`
   - 使用CPU训练（设置`device: "cpu"`）

3. **数据集检查失败**
   - 运行`python utils.py --action check`查看具体问题
   - 检查JSON标注文件格式是否正确

4. **MPS设备问题**
   - 确认macOS版本 >= 12.3
   - 确认使用Apple Silicon芯片
   - 更新PyTorch到最新版本：`pip install --upgrade torch`
   - 运行设备测试：`python test_mps_support.py`

5. **训练不收敛**
   - 检查数据质量和标注准确性
   - 调整学习率
   - 增加训练轮数

### 性能优化

1. **训练速度优化**
   - 使用GPU训练（CUDA或MPS）
   - 苹果M芯片用户：确保启用MPS加速
   - 增加`workers`数量（MPS推荐4-8个）
   - 使用更小的模型（yolov8n vs yolov8x）
   - MPS设备推荐批次大小：8

2. **推理速度优化**
   - 导出为ONNX格式
   - 使用TensorRT（NVIDIA GPU）
   - 减小输入图片尺寸

## 扩展开发

### 添加新功能

1. **自定义数据增强**：修改`config.py`中的`AUGMENTATION_CONFIG`
2. **新的评估指标**：在`trainer.py`中添加自定义验证逻辑
3. **后处理优化**：在`inference.py`中添加自定义后处理

### 集成到现有系统

可以将推理模块集成到现有的Flask应用中：

```python
from yolov8.inference import YOLOv8Inference

# 初始化推理器
inference = YOLOv8Inference("path/to/model.pt")

# 在Flask路由中使用
@app.route('/detect', methods=['POST'])
def detect():
    # 获取图片
    image = request.files['image']
    
    # 进行推理
    result = inference.predict_image(image)
    
    return jsonify(result)
```

## 许可证

本项目基于MIT许可证开源。
