import cv2
from ultralytics import YOLO
import os
import numpy as np

# 定义模型路径
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'YoloV8-Human-Emotion-Detection', 'best.pt')

print(f"尝试加载模型路径: {model_path}")

try:
    # 加载YOLOv8模型
    model = YOLO(model_path)
    print(f"成功加载模型: {model_path}")

    # 打印类别名称
    print("-" * 20)
    print("模型支持的表情类别:")
    print(model.names)
    print("-" * 20)

    # 创建一个纯黑色图像进行模拟预测
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

    print("开始进行模拟推理测试...")
    results = model(dummy_image)
    print("推理测试成功完成。")

except Exception as e:
    print(f"发生错误: {e}")
