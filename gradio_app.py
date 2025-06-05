#!/usr/bin/env python3
"""
YOLOv8 训练系统 Gradio 前端界面
提供完整的数据管理、模型训练、推理和监控功能
"""

import gradio as gr
import os
import sys
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import json

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from gradio_utils import log_monitor, training_monitor, dataset_manager, model_manager
from config_manager import config_manager
from data_converter import DataConverter
from trainer import YOLOv8Trainer
from inference import YOLOv8Inference
from smart_trainer import SmartTrainer
from utils import check_dataset_integrity, visualize_dataset_distribution, calculate_anchor_boxes


class GradioApp:
    """Gradio应用主类"""
    
    def __init__(self):
        self.current_process = None
        self.training_thread = None
        self.is_training = False
        
    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks(title="YOLO 训练系统", theme=gr.themes.Soft()) as app:
            gr.Markdown("# 🚀 YOLO 训练系统")
            gr.Markdown("一站式YOLO训练、推理和监控平台")
            
            with gr.Tabs():
                # 数据管理标签页
                with gr.TabItem("📊 数据管理"):
                    self._create_data_tab()
                
                # 模型训练标签页
                with gr.TabItem("🎯 模型训练"):
                    self._create_training_tab()
                
                # 模型推理标签页
                with gr.TabItem("🔍 模型推理"):
                    self._create_inference_tab()
                
                # 训练监控标签页
                with gr.TabItem("📈 训练监控"):
                    self._create_monitoring_tab()
                
                # 配置管理标签页
                with gr.TabItem("⚙️ 配置管理"):
                    self._create_config_tab()
                
                # 工具集标签页
                with gr.TabItem("🛠️ 工具集"):
                    self._create_tools_tab()
        
        return app
    
    def _create_data_tab(self):
        """创建数据管理标签页"""
        gr.Markdown("## 数据集管理")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 数据转换")
                convert_btn = gr.Button("🔄 转换JSON数据到YOLO格式", variant="primary")
                convert_output = gr.Textbox(label="转换结果", lines=5)
                
                gr.Markdown("### 数据集检查")
                check_btn = gr.Button("✅ 检查数据集完整性")
                check_output = gr.Textbox(label="检查结果", lines=5)
                
            with gr.Column():
                gr.Markdown("### 数据集信息")
                dataset_info = gr.JSON(label="数据集统计")
                refresh_info_btn = gr.Button("🔄 刷新信息")
                
                gr.Markdown("### 样本预览")
                sample_gallery = gr.Gallery(label="样本图片", columns=3, rows=2)
                refresh_samples_btn = gr.Button("🔄 刷新样本")
        
        # 绑定事件
        convert_btn.click(self._convert_data, outputs=convert_output)
        check_btn.click(self._check_dataset, outputs=check_output)
        refresh_info_btn.click(self._get_dataset_info, outputs=dataset_info)
        refresh_samples_btn.click(self._get_sample_images, outputs=sample_gallery)
        
        # 初始化数据
        # refresh_info_btn.click(self._get_dataset_info, outputs=dataset_info)
    
    def _create_training_tab(self):
        """创建模型训练标签页"""
        gr.Markdown("## 模型训练")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 训练配置")
                epochs = gr.Slider(1, 1000, value=100, label="训练轮数")
                batch_size = gr.Slider(1, 64, value=16, label="批次大小")
                learning_rate = gr.Slider(0.0001, 0.1, value=0.01, label="学习率")
                img_size = gr.Dropdown([320, 416, 512, 640, 832], value=640, label="图片尺寸")
                device = gr.Dropdown(["auto", "cpu", "cuda"], value="cpu", label="训练设备")
                
                gr.Markdown("### 训练模式")
                with gr.Row():
                    normal_train_btn = gr.Button("🎯 普通训练", variant="primary")
                    smart_train_btn = gr.Button("🤖 智能训练", variant="secondary")
                    resume_train_btn = gr.Button("🔄 恢复训练")
                
                training_status = gr.Textbox(label="训练状态", lines=3)
                
            with gr.Column():
                gr.Markdown("### 智能训练配置")
                target_map50 = gr.Slider(0.1, 1.0, value=0.7, label="目标mAP50")
                target_box_loss = gr.Slider(0.01, 1.0, value=0.05, label="目标Box Loss")
                target_cls_loss = gr.Slider(0.1, 5.0, value=1.0, label="目标Class Loss")
                max_epochs = gr.Slider(100, 2000, value=500, label="最大训练轮数")
                
                gr.Markdown("### 训练控制")
                stop_train_btn = gr.Button("⏹️ 停止训练", variant="stop")
                
                training_log = gr.Textbox(label="训练日志", lines=10, max_lines=20)
        
        # 绑定事件
        normal_train_btn.click(
            self._start_normal_training,
            inputs=[epochs, batch_size, learning_rate, img_size, device],
            outputs=training_status
        )
        smart_train_btn.click(
            self._start_smart_training,
            inputs=[target_map50, target_box_loss, target_cls_loss, max_epochs],
            outputs=training_status
        )
        resume_train_btn.click(self._resume_training, outputs=training_status)
        stop_train_btn.click(self._stop_training, outputs=training_status)
    
    def _create_inference_tab(self):
        """创建模型推理标签页"""
        gr.Markdown("## 模型推理")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 模型选择")
                model_dropdown = gr.Dropdown(label="选择模型", choices=[], interactive=True)
                refresh_models_btn = gr.Button("🔄 刷新模型列表")
                
                gr.Markdown("### 推理配置")
                conf_threshold = gr.Slider(0.1, 1.0, value=0.25, label="置信度阈值")
                iou_threshold = gr.Slider(0.1, 1.0, value=0.45, label="IoU阈值")
                max_det = gr.Slider(1, 1000, value=1000, label="最大检测数")
                
                gr.Markdown("### 单图推理")
                input_image = gr.Image(label="上传图片", type="filepath")
                single_inference_btn = gr.Button("🔍 单图推理", variant="primary")
                
            with gr.Column():
                gr.Markdown("### 推理结果")
                output_image = gr.Image(label="检测结果")
                detection_info = gr.JSON(label="检测信息")
                
                gr.Markdown("### 批量推理")
                batch_images = gr.File(label="上传图片文件夹", file_count="multiple")
                batch_inference_btn = gr.Button("📁 批量推理")
                batch_results = gr.Textbox(label="批量推理结果", lines=5)
        
        # 绑定事件
        refresh_models_btn.click(self._refresh_models, outputs=model_dropdown)
        single_inference_btn.click(
            self._single_inference,
            inputs=[model_dropdown, input_image, conf_threshold, iou_threshold, max_det],
            outputs=[output_image, detection_info]
        )
        batch_inference_btn.click(
            self._batch_inference,
            inputs=[model_dropdown, batch_images, conf_threshold, iou_threshold, max_det],
            outputs=batch_results
        )
        
        # 初始化模型列表
        # refresh_models_btn.click(self._refresh_models, outputs=model_dropdown)
    
    def _create_monitoring_tab(self):
        """创建训练监控标签页"""
        gr.Markdown("## 训练监控")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 训练状态")
                training_info = gr.JSON(label="当前训练信息")
                refresh_status_btn = gr.Button("🔄 刷新状态")
                
                gr.Markdown("### 日志监控")
                start_log_btn = gr.Button("▶️ 开始日志监控", variant="primary")
                stop_log_btn = gr.Button("⏹️ 停止日志监控")
                log_display = gr.Textbox(label="实时日志", lines=15, max_lines=30)
                
            with gr.Column():
                gr.Markdown("### 训练曲线")
                training_plot = gr.Image(label="训练曲线图")
                refresh_plot_btn = gr.Button("🔄 刷新曲线")
                
                gr.Markdown("### 模型信息")
                model_info = gr.JSON(label="模型详情")
        
        # 绑定事件
        refresh_status_btn.click(self._get_training_status, outputs=training_info)
        start_log_btn.click(self._start_log_monitoring, outputs=log_display)
        stop_log_btn.click(self._stop_log_monitoring, outputs=log_display)
        refresh_plot_btn.click(self._refresh_training_plot, outputs=training_plot)
        
        # 定时刷新
        # refresh_status_btn.click(self._get_training_status, outputs=training_info)
    
    def _create_config_tab(self):
        """创建配置管理标签页"""
        gr.Markdown("## 配置管理")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 当前配置")
                config_display = gr.Textbox(label="配置摘要", lines=15)
                refresh_config_btn = gr.Button("🔄 刷新配置")
                
                gr.Markdown("### 配置操作")
                reset_config_btn = gr.Button("🔄 重置为默认配置")
                export_config_btn = gr.Button("📤 导出配置")
                import_config_file = gr.File(label="导入配置文件")
                import_config_btn = gr.Button("📥 导入配置")
                
            with gr.Column():
                gr.Markdown("### 快速配置")
                with gr.Accordion("训练配置", open=True):
                    quick_epochs = gr.Slider(1, 1000, value=100, label="训练轮数")
                    quick_batch = gr.Slider(1, 64, value=16, label="批次大小")
                    quick_lr = gr.Slider(0.0001, 0.1, value=0.01, label="学习率")
                
                with gr.Accordion("推理配置", open=False):
                    quick_conf = gr.Slider(0.1, 1.0, value=0.25, label="置信度阈值")
                    quick_iou = gr.Slider(0.1, 1.0, value=0.45, label="IoU阈值")
                
                save_quick_config_btn = gr.Button("💾 保存快速配置", variant="primary")
                config_status = gr.Textbox(label="配置状态", lines=3)
        
        # 绑定事件
        refresh_config_btn.click(self._get_config_summary, outputs=config_display)
        reset_config_btn.click(self._reset_config, outputs=[config_display, config_status])
        save_quick_config_btn.click(
            self._save_quick_config,
            inputs=[quick_epochs, quick_batch, quick_lr, quick_conf, quick_iou],
            outputs=config_status
        )
        
        # 初始化配置显示
        # refresh_config_btn.click(self._get_config_summary, outputs=config_display)
    
    def _create_tools_tab(self):
        """创建工具集标签页"""
        gr.Markdown("## 工具集")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 数据集工具")
                visualize_dist_btn = gr.Button("📊 可视化数据分布")
                calc_anchors_btn = gr.Button("⚓ 计算锚框")
                
                gr.Markdown("### 模型工具")
                export_model_btn = gr.Button("📤 导出模型")
                model_format = gr.Dropdown(["onnx", "torchscript", "tflite"], value="onnx", label="导出格式")
                
                gr.Markdown("### 系统工具")
                check_env_btn = gr.Button("🔧 检查环境")
                clean_cache_btn = gr.Button("🧹 清理缓存")
                
            with gr.Column():
                gr.Markdown("### 工具输出")
                tools_output = gr.Textbox(label="工具执行结果", lines=15)
                
                gr.Markdown("### 系统信息")
                system_info = gr.JSON(label="系统状态")
                refresh_system_btn = gr.Button("🔄 刷新系统信息")
        
        # 绑定事件
        visualize_dist_btn.click(self._visualize_distribution, outputs=tools_output)
        calc_anchors_btn.click(self._calculate_anchors, outputs=tools_output)
        check_env_btn.click(self._check_environment, outputs=tools_output)
        refresh_system_btn.click(self._get_system_info, outputs=system_info)
        
        # 初始化系统信息
        # refresh_system_btn.click(self._get_system_info, outputs=system_info)
    
    # 数据管理相关方法
    def _convert_data(self):
        """转换数据"""
        try:
            converter = DataConverter()
            converter.convert_all()
            return "✅ 数据转换完成！"
        except Exception as e:
            return f"❌ 数据转换失败: {e}"
    
    def _check_dataset(self):
        """检查数据集"""
        try:
            is_valid = check_dataset_integrity()
            if is_valid:
                return "✅ 数据集检查通过！"
            else:
                return "⚠️ 数据集存在问题，请查看详细日志"
        except Exception as e:
            return f"❌ 数据集检查失败: {e}"
    
    def _get_dataset_info(self):
        """获取数据集信息"""
        return dataset_manager.get_dataset_info()
    
    def _get_sample_images(self):
        """获取样本图片"""
        return dataset_manager.get_sample_images()

    # 训练相关方法
    def _start_normal_training(self, epochs, batch_size, learning_rate, img_size, device):
        """开始普通训练"""
        if self.is_training:
            return "⚠️ 训练正在进行中，请等待完成或停止当前训练"

        try:
            # 更新配置
            config_manager.update_training_config(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                img_size=img_size,
                device=device
            )

            # 启动训练线程
            self.training_thread = threading.Thread(target=self._run_training)
            self.training_thread.daemon = True
            self.is_training = True
            self.training_thread.start()

            return f"🚀 开始普通训练 - {epochs} epochs, batch_size={batch_size}"
        except Exception as e:
            return f"❌ 启动训练失败: {e}"

    def _start_smart_training(self, target_map50, target_box_loss, target_cls_loss, max_epochs):
        """开始智能训练"""
        if self.is_training:
            return "⚠️ 训练正在进行中，请等待完成或停止当前训练"

        try:
            # 更新智能训练配置
            config_manager.update_smart_training_config(
                target_map50=target_map50,
                target_box_loss=target_box_loss,
                target_cls_loss=target_cls_loss,
                max_total_epochs=max_epochs
            )

            # 启动智能训练线程
            self.training_thread = threading.Thread(target=self._run_smart_training)
            self.training_thread.daemon = True
            self.is_training = True
            self.training_thread.start()

            return f"🤖 开始智能训练 - 目标mAP50={target_map50}, 最大轮数={max_epochs}"
        except Exception as e:
            return f"❌ 启动智能训练失败: {e}"

    def _resume_training(self):
        """恢复训练"""
        if self.is_training:
            return "⚠️ 训练正在进行中，无法恢复训练"

        try:
            trainer = YOLOv8Trainer()
            trainer.train(resume=True)
            return "🔄 恢复训练完成"
        except Exception as e:
            return f"❌ 恢复训练失败: {e}"

    def _stop_training(self):
        """停止训练"""
        self.is_training = False
        if self.training_thread and self.training_thread.is_alive():
            # 这里可以添加更优雅的停止机制
            return "⏹️ 正在停止训练..."
        return "⏹️ 没有正在进行的训练"

    def _run_training(self):
        """运行普通训练"""
        try:
            trainer = YOLOv8Trainer()
            trainer.train()
        except Exception as e:
            print(f"训练过程出错: {e}")
        finally:
            self.is_training = False

    def _run_smart_training(self):
        """运行智能训练"""
        try:
            smart_config = config_manager.get_smart_training_config()
            target_thresholds = {
                'box_loss': smart_config['target_box_loss'],
                'cls_loss': smart_config['target_cls_loss'],
                'dfl_loss': smart_config['target_dfl_loss'],
                'map50': smart_config['target_map50']
            }

            smart_trainer = SmartTrainer(target_thresholds)
            smart_trainer.smart_training_loop(
                max_total_epochs=smart_config['max_total_epochs'],
                continue_epochs=smart_config['continue_epochs']
            )
        except Exception as e:
            print(f"智能训练过程出错: {e}")
        finally:
            self.is_training = False

    # 推理相关方法
    def _refresh_models(self):
        """刷新模型列表"""
        models = model_manager.get_available_models()
        return gr.Dropdown.update(choices=models, value=models[0] if models else None)

    def _single_inference(self, model_path, image_path, conf_threshold, iou_threshold, max_det):
        """单图推理"""
        if not model_path or not image_path:
            return None, {"error": "请选择模型和上传图片"}

        try:
            # 更新推理配置
            config_manager.update_inference_config(
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                max_det=max_det
            )

            inference = YOLOv8Inference(model_path)
            result = inference.predict_image(image_path)

            # 可视化结果
            output_path = inference.visualize_predictions(
                image_path, result['predictions']
            )

            return output_path, result
        except Exception as e:
            return None, {"error": f"推理失败: {e}"}

    def _batch_inference(self, model_path, image_files, conf_threshold, iou_threshold, max_det):
        """批量推理"""
        if not model_path or not image_files:
            return "请选择模型和上传图片文件"

        try:
            # 更新推理配置
            config_manager.update_inference_config(
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                max_det=max_det
            )

            inference = YOLOv8Inference(model_path)
            image_paths = [f.name for f in image_files]
            results = inference.predict_batch(image_paths)

            total_detections = sum(r['num_detections'] for r in results)
            return f"✅ 批量推理完成！处理了 {len(image_paths)} 张图片，共检测到 {total_detections} 个目标"
        except Exception as e:
            return f"❌ 批量推理失败: {e}"

    # 监控相关方法
    def _get_training_status(self):
        """获取训练状态"""
        return training_monitor.get_training_status()

    def _start_log_monitoring(self):
        """开始日志监控"""
        if log_monitor.start_monitoring():
            return "📡 日志监控已启动"
        return "❌ 启动日志监控失败"

    def _stop_log_monitoring(self):
        """停止日志监控"""
        log_monitor.stop_monitoring()
        return "⏹️ 日志监控已停止"

    def _refresh_training_plot(self):
        """刷新训练曲线"""
        plot_data = training_monitor.generate_training_plot()
        return plot_data

    # 配置相关方法
    def _get_config_summary(self):
        """获取配置摘要"""
        return config_manager.get_config_summary()

    def _reset_config(self):
        """重置配置"""
        config_manager.reset_to_default()
        return config_manager.get_config_summary(), "✅ 配置已重置为默认值"

    def _save_quick_config(self, epochs, batch_size, lr, conf_threshold, iou_threshold):
        """保存快速配置"""
        try:
            config_manager.update_training_config(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr
            )
            config_manager.update_inference_config(
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            return "✅ 快速配置已保存"
        except Exception as e:
            return f"❌ 保存配置失败: {e}"

    # 工具相关方法
    def _visualize_distribution(self):
        """可视化数据分布"""
        try:
            visualize_dataset_distribution()
            return "✅ 数据分布可视化完成，图片已保存"
        except Exception as e:
            return f"❌ 可视化失败: {e}"

    def _calculate_anchors(self):
        """计算锚框"""
        try:
            anchors = calculate_anchor_boxes()
            result = "计算得到的锚框:\n"
            for i, (w, h) in enumerate(anchors):
                result += f"  {i+1}: ({w:.4f}, {h:.4f})\n"
            return result
        except Exception as e:
            return f"❌ 计算锚框失败: {e}"

    def _check_environment(self):
        """检查环境"""
        try:
            import torch
            import ultralytics

            result = "环境检查结果:\n"
            result += f"Python版本: {sys.version}\n"
            result += f"PyTorch版本: {torch.__version__}\n"
            result += f"Ultralytics版本: {ultralytics.__version__}\n"
            result += f"CUDA可用: {torch.cuda.is_available()}\n"
            if torch.cuda.is_available():
                result += f"CUDA版本: {torch.version.cuda}\n"
                result += f"GPU数量: {torch.cuda.device_count()}\n"

            return result
        except Exception as e:
            return f"❌ 环境检查失败: {e}"

    def _get_system_info(self):
        """获取系统信息"""
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                "CPU使用率": f"{cpu_percent}%",
                "内存使用率": f"{memory.percent}%",
                "可用内存": f"{memory.available / (1024**3):.2f} GB",
                "磁盘使用率": f"{disk.percent}%",
                "可用磁盘": f"{disk.free / (1024**3):.2f} GB",
                "训练状态": "进行中" if self.is_training else "空闲"
            }
        except Exception as e:
            return {"错误": f"获取系统信息失败: {e}"}


# 创建应用实例
app_instance = GradioApp()


def launch_app():
    """启动应用"""
    app = app_instance.create_interface()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True,
        quiet=False,
        inbrowser=False,
        prevent_thread_lock=False
    )


if __name__ == "__main__":
    launch_app()
