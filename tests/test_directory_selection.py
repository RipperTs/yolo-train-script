#!/usr/bin/env python3
"""
数据集目录选择功能的Gradio测试界面
"""

import gradio as gr
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataset_manager import dataset_directory_manager

def get_current_directory_info():
    """获取当前目录信息"""
    return dataset_directory_manager.get_current_directory_info()

def set_source_directory(directory_path):
    """设置数据源目录"""
    try:
        result = dataset_directory_manager.set_source_directory(directory_path)
        current_info = dataset_directory_manager.get_current_directory_info()
        preview_info = dataset_directory_manager.get_conversion_preview()
        
        status_message = result["message"]
        
        return current_info, status_message, preview_info
    except Exception as e:
        error_message = f"❌ 设置目录失败: {e}"
        return gr.update(), error_message, gr.update()

def validate_directory(directory_path):
    """验证目录"""
    try:
        validation_result = dataset_directory_manager.validate_directory(Path(directory_path))
        return {
            "directory": directory_path,
            "validation": validation_result
        }
    except Exception as e:
        return {
            "directory": directory_path,
            "error": f"验证失败: {e}"
        }

def get_conversion_preview():
    """获取转换预览"""
    return dataset_directory_manager.get_conversion_preview()

def convert_dataset():
    """转换数据集"""
    try:
        result = dataset_directory_manager.convert_dataset()
        return result["message"]
    except Exception as e:
        return f"❌ 转换失败: {e}"

def create_directory_selection_interface():
    """创建目录选择测试界面"""
    with gr.Blocks(title="数据集目录选择测试", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📁 数据集目录选择功能测试")
        gr.Markdown("测试动态选择数据集源目录的功能")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 目录选择")
                
                # 当前目录信息
                current_dir_display = gr.JSON(
                    label="当前数据源目录",
                    value=get_current_directory_info()
                )
                
                # 目录输入
                with gr.Row():
                    directory_input = gr.Textbox(
                        label="数据源目录路径",
                        placeholder="输入完整的目录路径",
                        value=str(dataset_directory_manager.current_source_dir)
                    )
                    set_dir_btn = gr.Button("📁 设置目录", variant="primary")
                
                # 目录建议
                suggestions = dataset_directory_manager.get_directory_suggestions()
                if suggestions:
                    gr.Markdown("### 目录建议")
                    dir_suggestions = gr.Dropdown(
                        choices=suggestions,
                        label="常用目录",
                        info="选择常见的数据目录"
                    )
                    
                    # 当选择建议时自动填入输入框
                    dir_suggestions.change(
                        lambda x: x if x else "",
                        inputs=[dir_suggestions],
                        outputs=[directory_input]
                    )
                
                # 状态显示
                status_message = gr.Textbox(
                    label="操作状态",
                    lines=3,
                    interactive=False
                )
                
            with gr.Column():
                gr.Markdown("### 目录验证")
                validate_btn = gr.Button("✅ 验证目录")
                validation_result = gr.JSON(label="验证结果")
                
                gr.Markdown("### 转换预览")
                preview_btn = gr.Button("👁️ 预览转换")
                conversion_preview = gr.JSON(label="转换预览")
                
                gr.Markdown("### 数据转换")
                convert_btn = gr.Button("🔄 执行转换", variant="secondary")
                convert_result = gr.Textbox(label="转换结果", lines=3)
        
        with gr.Row():
            gr.Markdown("### 使用说明")
            gr.Markdown("""
            **功能说明:**
            1. **设置目录**: 输入目录路径并点击"设置目录"按钮
            2. **目录建议**: 从下拉菜单选择常用目录
            3. **验证目录**: 检查目录是否包含有效的JSON标注文件
            4. **预览转换**: 查看数据集分割预览信息
            5. **执行转换**: 将JSON数据转换为YOLO格式
            
            **目录要求:**
            - 目录必须存在
            - 包含JSON标注文件（*.json）
            - 最好有对应的图片文件
            
            **默认目录:** `labeling_data`（如果存在）
            """)
        
        # 事件绑定
        set_dir_btn.click(
            set_source_directory,
            inputs=[directory_input],
            outputs=[current_dir_display, status_message, conversion_preview]
        )
        
        validate_btn.click(
            validate_directory,
            inputs=[directory_input],
            outputs=[validation_result]
        )
        
        preview_btn.click(
            get_conversion_preview,
            outputs=[conversion_preview]
        )
        
        convert_btn.click(
            convert_dataset,
            outputs=[convert_result]
        )
        
        # 刷新按钮
        refresh_btn = gr.Button("🔄 刷新当前信息")
        refresh_btn.click(
            get_current_directory_info,
            outputs=[current_dir_display]
        )
    
    return app

def main():
    """主函数"""
    print("🚀 启动数据集目录选择测试界面...")
    
    # 显示当前状态
    current_info = get_current_directory_info()
    print(f"当前数据源目录: {current_info['current_directory']}")
    print(f"目录状态: {current_info['status']['status']}")
    
    try:
        app = create_directory_selection_interface()
        app.launch(
            server_name="127.0.0.1",
            server_port=7864,
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
