#!/usr/bin/env python3
"""
测试Gradio前端修复
验证模型列表刷新和其他Gradio API修复是否正常工作
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import gradio as gr


def test_gradio_version():
    """测试Gradio版本"""
    print(f"🔍 Gradio版本: {gr.__version__}")
    
    # 检查是否支持新的API
    has_skip = hasattr(gr, 'skip')
    has_dropdown_update = hasattr(gr.Dropdown, 'update')
    
    print(f"支持 gr.skip(): {has_skip}")
    print(f"支持 gr.Dropdown.update(): {has_dropdown_update}")
    
    if not has_skip:
        print("⚠️ 当前Gradio版本不支持gr.skip()，可能需要升级")
    
    if has_dropdown_update:
        print("⚠️ 当前Gradio版本仍支持gr.Dropdown.update()，但建议使用新API")


def test_model_refresh_function():
    """测试模型刷新功能"""
    print("\n🧪 测试模型刷新功能")
    
    try:
        # 导入模型管理器
        from gradio_utils import model_manager
        
        # 获取可用模型
        models = model_manager.get_available_models()
        print(f"找到 {len(models)} 个模型: {models}")
        
        # 测试新的Dropdown创建方式
        if models:
            dropdown = gr.Dropdown(choices=models, value=models[0])
            print("✅ 新的Dropdown创建方式测试成功")
        else:
            dropdown = gr.Dropdown(choices=[], value=None)
            print("✅ 空模型列表的Dropdown创建成功")
            
        return True
        
    except Exception as e:
        print(f"❌ 模型刷新功能测试失败: {e}")
        return False


def test_gradio_app_import():
    """测试Gradio应用导入"""
    print("\n🧪 测试Gradio应用导入")
    
    try:
        from gradio_app import GradioApp
        print("✅ GradioApp导入成功")
        
        # 测试创建应用实例
        app_instance = GradioApp()
        print("✅ GradioApp实例创建成功")
        
        # 测试模型刷新方法
        result = app_instance._refresh_models()
        print(f"✅ 模型刷新方法调用成功，返回类型: {type(result)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Gradio应用导入测试失败: {e}")
        return False


def test_gradio_components():
    """测试Gradio组件创建"""
    print("\n🧪 测试Gradio组件创建")
    
    try:
        # 测试各种组件的创建
        components = {
            "Dropdown": gr.Dropdown(choices=["选项1", "选项2"], value="选项1"),
            "Slider": gr.Slider(0, 100, value=50),
            "Button": gr.Button("测试按钮"),
            "Textbox": gr.Textbox(label="测试文本框"),
            "Image": gr.Image(label="测试图片"),
            "JSON": gr.JSON(label="测试JSON"),
            "File": gr.File(label="测试文件")
        }
        
        for name, component in components.items():
            print(f"✅ {name} 组件创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ Gradio组件创建测试失败: {e}")
        return False


def test_gradio_interface_creation():
    """测试Gradio界面创建"""
    print("\n🧪 测试Gradio界面创建")
    
    try:
        def dummy_function(x):
            return f"输入: {x}"
        
        # 创建简单的界面
        interface = gr.Interface(
            fn=dummy_function,
            inputs=gr.Textbox(label="输入"),
            outputs=gr.Textbox(label="输出"),
            title="测试界面"
        )
        
        print("✅ Gradio界面创建成功")
        return True
        
    except Exception as e:
        print(f"❌ Gradio界面创建测试失败: {e}")
        return False


def create_test_interface():
    """创建测试界面"""
    print("\n🧪 创建测试界面")
    
    def test_refresh():
        """测试刷新功能"""
        choices = ["模型1", "模型2", "模型3"]
        return gr.Dropdown(choices=choices, value=choices[0] if choices else None)
    
    def test_skip():
        """测试skip功能"""
        return gr.skip()
    
    with gr.Blocks(title="Gradio修复测试") as demo:
        gr.Markdown("# Gradio修复测试界面")
        
        with gr.Row():
            with gr.Column():
                dropdown = gr.Dropdown(label="模型选择", choices=[])
                refresh_btn = gr.Button("🔄 刷新模型列表")
                skip_btn = gr.Button("测试Skip功能")
                
            with gr.Column():
                output = gr.Textbox(label="输出", lines=5)
        
        # 绑定事件
        refresh_btn.click(test_refresh, outputs=dropdown)
        skip_btn.click(test_skip, outputs=output)
    
    return demo


def main():
    """主函数"""
    print("🧪 Gradio前端修复测试")
    print("="*50)
    
    # 测试1: Gradio版本检查
    test_gradio_version()
    
    # 测试2: 模型刷新功能
    success1 = test_model_refresh_function()
    
    # 测试3: Gradio应用导入
    success2 = test_gradio_app_import()
    
    # 测试4: Gradio组件创建
    success3 = test_gradio_components()
    
    # 测试5: Gradio界面创建
    success4 = test_gradio_interface_creation()
    
    # 汇总结果
    print("\n📊 测试结果汇总:")
    print(f"模型刷新功能: {'✅' if success1 else '❌'}")
    print(f"Gradio应用导入: {'✅' if success2 else '❌'}")
    print(f"Gradio组件创建: {'✅' if success3 else '❌'}")
    print(f"Gradio界面创建: {'✅' if success4 else '❌'}")
    
    all_success = all([success1, success2, success3, success4])
    print(f"\n总体结果: {'🎉 所有测试通过' if all_success else '❌ 部分测试失败'}")
    
    # 询问是否启动测试界面
    if all_success:
        print("\n❓ 是否启动测试界面？")
        choice = input("输入 'y' 启动测试界面，其他键退出: ").strip().lower()
        if choice == 'y':
            print("🚀 启动测试界面...")
            demo = create_test_interface()
            demo.launch(
                server_name="127.0.0.1",
                server_port=7861,  # 使用不同的端口避免冲突
                share=False,
                debug=True
            )
        else:
            print("👋 测试完成")
    else:
        print("❌ 由于测试失败，不启动测试界面")


if __name__ == "__main__":
    main()
