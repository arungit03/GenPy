import torch

print("PyTorch:", torch.__version__)
print("MPS Available:", torch.backends.mps.is_available())
print("CUDA Available:", torch.cuda.is_available())

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using:", device)
