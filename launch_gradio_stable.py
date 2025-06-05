#!/usr/bin/env python3
"""
稳定版Gradio启动脚本
解决网络连接和启动问题
"""

import os
import sys
import time
from pathlib import Path

# 确保在正确的目录
project_root = Path(__file__).parent
os.chdir(project_root)

# 添加项目路径到Python路径
sys.path.insert(0, str(project_root))

def check_port_available(port=7860):
    """检查端口是否可用"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', port))
            return True
    except OSError:
        return False

def find_available_port(start_port=7860):
    """查找可用端口"""
    for port in range(start_port, start_port + 10):
        if check_port_available(port):
            return port
    return None

def main():
    """主函数"""
    print("🚀 启动YOLOv8训练系统Gradio界面（稳定版）...")
    print(f"📁 项目目录: {project_root}")
    
    # 检查端口
    port = find_available_port()
    if port is None:
        print("❌ 无法找到可用端口，请检查网络设置")
        return
    
    print(f"🌐 使用端口: {port}")
    print(f"🌐 界面地址: http://127.0.0.1:{port}")
    print("=" * 50)
    
    # 设置环境变量
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    os.environ['GRADIO_TEMP_DIR'] = str(project_root / 'temp')
    
    try:
        # 导入Gradio相关模块
        import gradio as gr
        print("✅ Gradio模块加载成功")
        
        # 导入应用模块
        from gradio_app import GradioApp
        print("✅ 应用模块加载成功")
        
        # 创建应用实例
        app_instance = GradioApp()
        print("✅ 应用实例创建成功")
        
        # 创建界面
        app = app_instance.create_interface()
        print("✅ 界面创建成功")
        
        # 启动应用
        print("🚀 正在启动服务器...")
        app.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            debug=False,
            show_error=True,
            quiet=True,
            inbrowser=False,
            prevent_thread_lock=False
        )
        
    except KeyboardInterrupt:
        print("\n👋 用户中断，正在关闭...")
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        print("💡 请确保已安装所有依赖：pip install gradio psutil")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("\n🔧 故障排除建议:")
        print("1. 检查Python环境是否正确")
        print("2. 确认所有依赖已安装")
        print("3. 检查端口是否被占用")
        print("4. 尝试重启终端")
        
        import traceback
        print("\n📋 详细错误信息:")
        traceback.print_exc()
    finally:
        print("🔚 应用已关闭")

if __name__ == "__main__":
    main()
