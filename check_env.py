import sys
import torch

print("=== 环境检测 ===")
print(f"Python版本: {sys.version.split()[0]}\n{sys.executable}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU数量: {torch.cuda.device_count()}")