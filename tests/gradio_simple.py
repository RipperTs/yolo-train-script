#!/usr/bin/env python3
"""
简化版Gradio应用
用于测试和快速启动
"""

import gradio as gr
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def test_data_conversion():
    """测试数据转换"""
    try:
        from data_converter import DataConverter
        converter = DataConverter()
        converter.convert_all()
        return "✅ 数据转换完成！"
    except Exception as e:
        return f"❌ 数据转换失败: {e}"

def test_dataset_check():
    """测试数据集检查"""
    try:
        from utils import check_dataset_integrity
        is_valid = check_dataset_integrity()
        if is_valid:
            return "✅ 数据集检查通过！"
        else:
            return "⚠️ 数据集存在问题"
    except Exception as e:
        return f"❌ 数据集检查失败: {e}"

def get_dataset_info():
    """获取数据集信息"""
    try:
        from gradio_utils import dataset_manager
        info = dataset_manager.get_dataset_info()
        return info
    except Exception as e:
        return {"error": f"获取数据集信息失败: {e}"}

def get_system_info():
    """获取系统信息"""
    try:
        import psutil
        import torch
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        info = {
            "Python版本": sys.version.split()[0],
            "PyTorch版本": torch.__version__,
            "CUDA可用": torch.cuda.is_available(),
            "CPU使用率": f"{cpu_percent}%",
            "内存使用率": f"{memory.percent}%",
            "可用内存": f"{memory.available / (1024**3):.2f} GB"
        }
        return info
    except Exception as e:
        return {"error": f"获取系统信息失败: {e}"}

def create_simple_interface():
    """创建简化界面"""
    with gr.Blocks(title="YOLOv8 训练系统 - 简化版", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🚀 YOLOv8 训练系统 - 简化版")
        gr.Markdown("快速测试和基本功能")
        
        with gr.Tabs():
            # 数据管理
            with gr.TabItem("📊 数据管理"):
                gr.Markdown("## 数据管理功能")
                
                with gr.Row():
                    with gr.Column():
                        convert_btn = gr.Button("🔄 转换数据", variant="primary")
                        check_btn = gr.Button("✅ 检查数据集")
                        refresh_btn = gr.Button("🔄 刷新信息")
                        
                    with gr.Column():
                        result_text = gr.Textbox(label="操作结果", lines=5)
                        dataset_info = gr.JSON(label="数据集信息")
                
                # 绑定事件
                convert_btn.click(test_data_conversion, outputs=result_text)
                check_btn.click(test_dataset_check, outputs=result_text)
                refresh_btn.click(get_dataset_info, outputs=dataset_info)
            
            # 系统信息
            with gr.TabItem("🔧 系统信息"):
                gr.Markdown("## 系统状态")
                
                with gr.Row():
                    with gr.Column():
                        sys_refresh_btn = gr.Button("🔄 刷新系统信息", variant="primary")
                        
                    with gr.Column():
                        system_info = gr.JSON(label="系统信息")
                
                # 绑定事件
                sys_refresh_btn.click(get_system_info, outputs=system_info)
            
            # 配置信息
            with gr.TabItem("⚙️ 配置"):
                gr.Markdown("## 配置信息")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 训练配置")
                        epochs = gr.Slider(1, 1000, value=100, label="训练轮数")
                        batch_size = gr.Slider(1, 64, value=16, label="批次大小")
                        learning_rate = gr.Slider(0.0001, 0.1, value=0.01, label="学习率")
                        
                    with gr.Column():
                        gr.Markdown("### 推理配置")
                        conf_threshold = gr.Slider(0.1, 1.0, value=0.25, label="置信度阈值")
                        iou_threshold = gr.Slider(0.1, 1.0, value=0.45, label="IoU阈值")
                        
                        save_config_btn = gr.Button("💾 保存配置")
                        config_status = gr.Textbox(label="配置状态", lines=2)
                
                def save_config(epochs, batch_size, lr, conf, iou):
                    try:
                        from config_manager import config_manager
                        config_manager.update_training_config(
                            epochs=epochs,
                            batch_size=batch_size,
                            learning_rate=lr
                        )
                        config_manager.update_inference_config(
                            conf_threshold=conf,
                            iou_threshold=iou
                        )
                        return "✅ 配置保存成功"
                    except Exception as e:
                        return f"❌ 配置保存失败: {e}"
                
                save_config_btn.click(
                    save_config,
                    inputs=[epochs, batch_size, learning_rate, conf_threshold, iou_threshold],
                    outputs=config_status
                )
        
        # 初始化数据
        app.load(get_dataset_info, outputs=dataset_info)
        app.load(get_system_info, outputs=system_info)
    
    return app

def main():
    """主函数"""
    print("🚀 启动YOLOv8训练系统 - 简化版...")
    
    try:
        app = create_simple_interface()
        app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True,
            quiet=False,
            inbrowser=True,
            prevent_thread_lock=False
        )
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
