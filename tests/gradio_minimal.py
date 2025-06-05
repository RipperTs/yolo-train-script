#!/usr/bin/env python3
"""
最小化Gradio应用
用于测试基本功能
"""

import gradio as gr
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def hello_world():
    return "✅ YOLOv8训练系统正常运行！"

def get_project_info():
    """获取项目信息"""
    project_root = Path(__file__).parent
    
    info = {
        "项目目录": str(project_root),
        "Python版本": sys.version.split()[0],
        "工作目录": os.getcwd()
    }
    
    # 检查关键文件
    key_files = [
        "config.py",
        "data_converter.py", 
        "trainer.py",
        "inference.py"
    ]
    
    for file in key_files:
        info[f"文件_{file}"] = "存在" if (project_root / file).exists() else "缺失"
    
    return info

def test_data_conversion():
    """测试数据转换功能"""
    try:
        from config import YOLO_POINT_DIR, DATASETS_DIR
        
        # 检查输入目录
        if not YOLO_POINT_DIR.exists():
            return f"❌ 输入目录不存在: {YOLO_POINT_DIR}"
        
        # 检查JSON文件
        json_files = list(YOLO_POINT_DIR.glob("*.json"))
        if not json_files:
            return f"❌ 在 {YOLO_POINT_DIR} 中没有找到JSON文件"
        
        return f"✅ 找到 {len(json_files)} 个JSON文件，可以进行数据转换"
        
    except Exception as e:
        return f"❌ 检查失败: {e}"

def run_data_conversion():
    """执行数据转换"""
    try:
        from data_converter import DataConverter
        converter = DataConverter()
        converter.convert_all()
        return "✅ 数据转换完成！"
    except Exception as e:
        return f"❌ 数据转换失败: {e}"

def create_minimal_interface():
    """创建最小化界面"""
    with gr.Blocks(title="YOLOv8 训练系统 - 测试版") as app:
        gr.Markdown("# 🚀 YOLOv8 训练系统 - 测试版")
        gr.Markdown("基本功能测试界面")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 系统测试")
                test_btn = gr.Button("🧪 系统测试", variant="primary")
                test_result = gr.Textbox(label="测试结果", lines=2)
                
                gr.Markdown("### 项目信息")
                info_btn = gr.Button("📋 获取项目信息")
                project_info = gr.JSON(label="项目信息")
                
            with gr.Column():
                gr.Markdown("### 数据转换")
                check_data_btn = gr.Button("🔍 检查数据")
                convert_data_btn = gr.Button("🔄 转换数据", variant="secondary")
                data_result = gr.Textbox(label="数据操作结果", lines=5)
        
        # 绑定事件
        test_btn.click(hello_world, outputs=test_result)
        info_btn.click(get_project_info, outputs=project_info)
        check_data_btn.click(test_data_conversion, outputs=data_result)
        convert_data_btn.click(run_data_conversion, outputs=data_result)
        
        # 初始化
        app.load(get_project_info, outputs=project_info)
    
    return app

def main():
    """主函数"""
    print("🚀 启动YOLOv8训练系统 - 测试版...")
    
    try:
        app = create_minimal_interface()
        print("✅ 界面创建成功")
        
        app.launch(
            server_name="127.0.0.1",
            server_port=7862,
            share=False,
            debug=False,
            show_error=True,
            quiet=False,
            inbrowser=True
        )
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
