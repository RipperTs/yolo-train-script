#### 创建配置文件

config.yaml在与模型相同的文件夹中创建 YAML 格式的模型配置文件。配置文件需要遵循以下格式：

- rtdetr_r50.yaml（目标检测）

```
type: rtdetr
name: rtdetr_r50-r20230520
display_name: RT-DETR (ResNet50) PaddleDetection
model_path: https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.1.0/rtdetr_r50vd_6x_coco.onnx
input_width: 640
input_height: 640
score_threshold: 0.45
classes:
  - person
  - bicycle
  - car
  ...
```

- yolov6lite_s_face.yaml（人脸及关键点检测）

```
type: yolov6_face
name: yolov6lite_s_face-r20230520
display_name: YOLOv6Lite_s-Face MeiTuan
model_path: https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.1.0/yolov6lite_s_face.onnx
input_width: 320
input_height: 320
stride: 64
nms_threshold: 0.45
confidence_threshold: 0.4
classes:
  - face
five_key_points_classes:
  - left_eye
  - right_eye
  - nost_tip
  - left_mouth_corner
  - right_mouth_corner
```

- yolov5s_resnet50.yaml（检测+分类级联）

```
type: yolov5_cls
name: yolov5s_resnet50-r20230520
display_name: YOLOv5s-ResNet50
det_model_path: https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.1.0/yolov5s.onnx
cls_model_path: https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.1.0/resnet50.onnx
det_input_width: 640
det_input_height: 640
cls_input_width: 224
cls_input_height: 224
cls_score_threshold: 0.5
stride: 32
nms_threshold: 0.45
confidence_threshold: 0.45
det_classes:
  - person
  - bicycle
  - car
  ...
cls_classes:
  0: tench
  1: goldfish
  2: great white shark
  3: tiger shark
```

需要注意的是，这里检测+分类仅提供样例模板给大家，模型`yolov5`和`resnet`中涉及到的类别分别是`coco`和`imagenet`上预训练得到的权重，大家需要根据自身任务重新训练新的模型进行替换。