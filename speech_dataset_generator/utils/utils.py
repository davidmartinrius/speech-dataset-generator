import torch

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA is available. Using GPU.")
    else:
        device = "cpu"
        print("CUDA is not available. Using CPU.")
        
    return device
    