import torch
import sys

print(f"Python 版本: {sys.version}")
print(f"PyTorch 版本: {torch.__version__}")

print("\n--- GPU 检查 ---")
if torch.cuda.is_available():
    print(f"CUDA 可用: 是")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
else:
    print(f"CUDA 可用: 否 (正在使用 CPU)")
    print("\n可能有以下原因:")
    print("1. 没有安装 NVIDIA 显卡驱动。")
    print("2. 安装了 CPU 版本的 PyTorch (默认 pip install torch 往往是 CPU 版)。")

