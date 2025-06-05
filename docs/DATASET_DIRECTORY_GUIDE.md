# 数据集目录选择功能使用指南

## 🎯 功能概述

YOLOv8训练系统现在支持灵活的数据集目录选择功能，用户可以动态选择数据源目录，而不再局限于固定的`labeling_data`目录。

## ✨ 主要特性

### 1. 动态目录选择
- 🔄 支持运行时切换数据源目录
- 📁 自动保存和恢复目录设置
- 💾 配置持久化存储

### 2. 智能目录验证
- ✅ 自动检查目录是否存在
- 📊 验证JSON标注文件数量
- 🖼️ 检查对应图片文件
- ⚠️ 提供详细的验证报告

### 3. 用户友好提示
- 📋 目录建议列表
- 🔍 实时状态显示
- 📈 转换预览信息
- ❌ 空目录或无效目录提示

## 🚀 使用方法

### 在Gradio界面中使用

#### 1. 访问数据管理页面
1. 启动Gradio界面：`python start_gradio.py`
2. 访问：http://localhost:7860
3. 点击"📊 数据管理"标签页

#### 2. 选择数据源目录
有三种方式选择目录：

**方式一：手动输入路径**
```
1. 在"数据源目录路径"输入框中输入完整路径
2. 例如：/Users/username/my_dataset
3. 点击"📁 设置目录"按钮
```

**方式二：使用目录建议**
```
1. 从"目录建议"下拉菜单中选择
2. 系统会自动填入路径
3. 点击"📁 设置目录"按钮
```

**方式三：使用默认目录**
```
默认使用项目根目录下的 labeling_data 文件夹
如果该目录存在且包含JSON文件，会自动设置
```

#### 3. 验证目录状态
```
1. 点击"✅ 验证目录"查看详细验证信息
2. 查看"目录状态"显示的验证结果
3. 确认JSON文件和图片文件数量
```

#### 4. 预览转换信息
```
1. 点击"👁️ 预览转换"查看转换预览
2. 查看数据集分割信息（训练/验证/测试）
3. 确认源目录和目标目录路径
```

#### 5. 执行数据转换
```
1. 确认目录和预览信息无误后
2. 点击"🔄 转换数据到YOLO格式"
3. 查看转换结果和状态信息
```

## 📋 目录要求

### 必需条件
- ✅ 目录必须存在
- ✅ 包含至少一个JSON标注文件（*.json）
- ✅ JSON文件格式正确（labelme格式）

### 推荐条件
- 🖼️ 包含对应的图片文件
- 📁 图片和JSON文件在同一目录
- 🏷️ 文件命名一致（如：image1.jpg 对应 image1.json）

### 支持的文件格式
**图片格式：**
- .jpg / .jpeg
- .png
- .bmp
- .tiff

**标注格式：**
- .json（labelme格式）

## 🔍 状态说明

### 目录状态类型

#### ✅ ready（就绪）
```
- 目录存在且包含有效的JSON文件
- 可以进行数据转换
- 显示文件数量和示例文件名
```

#### ❌ not_exists（不存在）
```
- 指定的目录路径不存在
- 需要检查路径是否正确
- 或创建对应的目录
```

#### ⚠️ empty（空目录）
```
- 目录存在但没有JSON文件
- 需要添加标注文件
- 或选择其他目录
```

### 验证结果说明

#### 🟢 valid: true
```
目录验证通过，包含：
- exists: 目录是否存在
- is_directory: 是否为目录
- json_files: JSON文件数量
- image_files: 图片文件数量
- missing_images: 缺失的图片列表
```

#### 🔴 valid: false
```
目录验证失败，可能原因：
- 目录不存在
- 不是有效目录
- 没有JSON文件
- 大量图片文件缺失
```

## 📊 转换预览信息

### 预览内容
```json
{
  "status": "ready",
  "total_files": 311,
  "split_preview": {
    "train": 217,    // 训练集文件数
    "val": 62,       // 验证集文件数  
    "test": 32       // 测试集文件数
  },
  "source_directory": "/path/to/source",
  "target_directory": "/path/to/datasets",
  "sample_files": ["file1.json", "file2.json", ...]
}
```

### 数据分割比例
- 🎯 训练集：70%
- 📊 验证集：20%
- 🧪 测试集：10%

## 🛠️ 高级功能

### 1. 配置持久化
```
- 目录设置自动保存到 dataset_config.json
- 重启应用后自动恢复上次设置
- 支持多个项目的独立配置
```

### 2. 目录建议
```
系统自动扫描常见目录：
- 项目根目录/labeling_data
- 项目根目录/data
- 项目根目录/dataset
- 项目根目录/datasets
- 项目根目录/annotations
```

### 3. 批量验证
```
- 自动检查前10个JSON文件的图片对应关系
- 统计缺失图片数量和比例
- 提供详细的验证报告
```

## 🚨 常见问题

### Q1: 设置目录后提示"目录验证失败"
**解决方案：**
1. 检查目录路径是否正确
2. 确认目录中包含JSON文件
3. 检查JSON文件格式是否正确
4. 查看详细的验证报告

### Q2: 转换时提示"找不到图片文件"
**解决方案：**
1. 确保图片文件与JSON文件在同一目录
2. 检查文件命名是否一致
3. 确认图片格式是否支持
4. 查看JSON文件中的imagePath字段

### Q3: 目录建议中没有我的目录
**解决方案：**
1. 手动输入完整的目录路径
2. 确保目录名称符合常见命名规范
3. 将数据移动到建议的目录中

### Q4: 转换后的数据集在哪里？
**答案：**
```
转换后的数据集保存在：
- 图片：datasets/images/train|val|test/
- 标签：datasets/labels/train|val|test/
- 配置：dataset.yaml
```

## 🔧 命令行使用

### 测试目录功能
```bash
# 运行目录选择测试
python tests/test_dataset_directory.py

# 启动目录选择界面
python tests/test_directory_selection.py
```

### 编程接口
```python
from dataset_manager import dataset_directory_manager

# 设置目录
result = dataset_directory_manager.set_source_directory("/path/to/data")

# 验证目录
validation = dataset_directory_manager.validate_directory(Path("/path/to/data"))

# 获取预览
preview = dataset_directory_manager.get_conversion_preview()

# 执行转换
result = dataset_directory_manager.convert_dataset()
```

## 📈 最佳实践

### 1. 目录组织建议
```
推荐的目录结构：
my_dataset/
├── image1.jpg
├── image1.json
├── image2.jpg
├── image2.json
└── ...
```

### 2. 文件命名规范
```
- 使用一致的命名规范
- 避免特殊字符和空格
- 保持图片和JSON文件名一致
```

### 3. 数据质量检查
```
转换前建议：
1. 检查JSON文件格式
2. 验证标注质量
3. 确认类别标签正确
4. 检查图片完整性
```

### 4. 备份重要数据
```
转换前建议备份：
- 原始标注文件
- 原始图片文件
- 项目配置文件
```

---

🎉 现在您可以灵活地选择任意目录作为数据源，不再受限于固定的`labeling_data`目录！系统会智能验证目录内容并提供友好的提示信息。
