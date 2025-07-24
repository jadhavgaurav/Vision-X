import torch

# Check if PyTorch is installed
print(f"PyTorch version: {torch.__version__}")

# Check if CUDA is available and PyTorch can use the GPU
is_available = torch.cuda.is_available()
print(f"CUDA available: {is_available}")

if is_available:
    # Print the name of the GPU
    print(f"GPU: {torch.cuda.get_device_name(0)}")