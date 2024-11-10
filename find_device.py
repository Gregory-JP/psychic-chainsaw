import torch

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)
else:
    print("No GPU available. Training will run on CPU.")
