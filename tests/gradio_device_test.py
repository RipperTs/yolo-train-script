#!/usr/bin/env python3
"""
设备切换功能测试的Gradio应用
"""

import gradio as gr
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from device_manager import device_manager, get_device_choices_for_gradio, parse_device_choice
from config_manager import config_manager

def get_device_info():
    """获取设备信息"""
    return config_manager.get_device_info()

def switch_device(device_choice):
    """切换设备"""
    try:
        device_id = parse_device_choice(device_choice)
        success = config_manager.update_device(device_id)
        
        if success:
            # 获取推荐配置
            recommended_batch = device_manager.get_optimal_batch_size(device_id)
            device_info = config_manager.get_device_info()
            
            status = f"✅ 成功切换到设备: {device_id}\n推荐批次大小: {recommended_batch}"
            
            return status, recommended_batch, device_info
        else:
            return "❌ 设备切换失败", gr.update(), gr.update()
            
    except Exception as e:
        return f"❌ 设备切换出错: {e}", gr.update(), gr.update()

def test_device_performance(device_choice, batch_size):
    """测试设备性能"""
    try:
        import torch
        import time
        
        device_id = parse_device_choice(device_choice)
        
        # 创建测试张量
        if device_id == "cpu":
            device_torch = torch.device("cpu")
        else:
            device_torch = torch.device(device_id)
        
        # 性能测试
        start_time = time.time()
        
        # 创建随机张量模拟训练数据
        x = torch.randn(batch_size, 3, 640, 640).to(device_torch)
        y = torch.randn(batch_size, 1000).to(device_torch)
        
        # 简单的矩阵运算
        for _ in range(10):
            z = torch.mm(x.view(batch_size, -1), torch.randn(3*640*640, 1000).to(device_torch))
            loss = torch.nn.functional.mse_loss(z, y)
            loss.backward()
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        result = f"""
设备性能测试结果:
==================
设备: {device_id}
批次大小: {batch_size}
测试时间: {elapsed:.2f} 秒
张量形状: {x.shape}
内存使用: {torch.cuda.memory_allocated() // (1024**2) if device_id.startswith('cuda') else 'N/A'} MB

性能评估: {'优秀' if elapsed < 1 else '良好' if elapsed < 3 else '一般'}
"""
        return result
        
    except Exception as e:
        return f"❌ 性能测试失败: {e}"

def create_device_test_interface():
    """创建设备测试界面"""
    with gr.Blocks(title="设备管理测试", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🔧 设备管理和切换测试")
        gr.Markdown("测试CPU/GPU设备切换功能")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 设备选择")
                device_choices = get_device_choices_for_gradio()
                device_dropdown = gr.Dropdown(
                    choices=device_choices,
                    value=device_choices[0] if device_choices else "cpu",
                    label="选择设备",
                    info="选择要使用的计算设备"
                )
                
                switch_btn = gr.Button("🔄 切换设备", variant="primary")
                device_status = gr.Textbox(label="切换状态", lines=3)
                
                gr.Markdown("### 训练配置")
                batch_size = gr.Slider(1, 32, value=4, label="批次大小")
                epochs = gr.Slider(1, 100, value=10, label="训练轮数")
                learning_rate = gr.Slider(0.0001, 0.1, value=0.01, label="学习率")
                
                test_performance_btn = gr.Button("🧪 测试设备性能")
                
            with gr.Column():
                gr.Markdown("### 设备信息")
                device_info_display = gr.JSON(label="设备详情")
                refresh_info_btn = gr.Button("🔄 刷新信息")
                
                gr.Markdown("### 性能测试结果")
                performance_result = gr.Textbox(label="测试结果", lines=15)
        
        with gr.Row():
            gr.Markdown("### 配置摘要")
            config_summary = gr.Textbox(label="当前配置", lines=10)
            refresh_config_btn = gr.Button("🔄 刷新配置")
        
        # 事件绑定
        switch_btn.click(
            switch_device,
            inputs=[device_dropdown],
            outputs=[device_status, batch_size, device_info_display]
        )
        
        device_dropdown.change(
            switch_device,
            inputs=[device_dropdown],
            outputs=[device_status, batch_size, device_info_display]
        )
        
        refresh_info_btn.click(get_device_info, outputs=device_info_display)
        
        test_performance_btn.click(
            test_device_performance,
            inputs=[device_dropdown, batch_size],
            outputs=performance_result
        )
        
        refresh_config_btn.click(
            lambda: config_manager.get_config_summary(),
            outputs=config_summary
        )
        
        # 初始化
        app.load(get_device_info, outputs=device_info_display)
        app.load(lambda: config_manager.get_config_summary(), outputs=config_summary)
    
    return app

def main():
    """主函数"""
    print("🚀 启动设备管理测试界面...")
    
    # 显示当前设备信息
    print(f"当前设备: {device_manager.current_device}")
    print(f"GPU可用: {device_manager.is_gpu_available()}")
    print(f"可用设备: {device_manager.get_device_choices()}")
    
    try:
        app = create_device_test_interface()
        app.launch(
            server_name="127.0.0.1",
            server_port=7863,
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
