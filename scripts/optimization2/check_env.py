import torch

print("PyTorch version:", torch.__version__)
print("MKL available:", torch.backends.mkl.is_available())
print("OpenMP available:", torch.backends.openmp.is_available())
print("Threads:", torch.get_num_threads())
print("CPU Info:")
print(torch.__config__.show())
