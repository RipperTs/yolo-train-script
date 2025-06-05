#!/usr/bin/env python3
"""
启动Gradio前端界面的脚本
"""

import os
import sys
from pathlib import Path

# 确保在正确的目录
project_root = Path(__file__).parent
os.chdir(project_root)

# 添加项目路径到Python路径
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['GRADIO_SERVER_NAME'] = '0.0.0.0'
os.environ['GRADIO_SERVER_PORT'] = '7860'

def main():
    """主函数"""
    print("🚀 启动YOLOv8训练系统Gradio界面...")
    print(f"📁 项目目录: {project_root}")
    print("🌐 界面地址: http://localhost:7860")
    print("=" * 50)
    
    try:
        # 导入并启动应用
        from gradio_app import launch_app
        launch_app()
    except KeyboardInterrupt:
        print("\n👋 用户中断，正在关闭...")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔚 应用已关闭")

if __name__ == "__main__":
    main()
