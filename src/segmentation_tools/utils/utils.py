import torch

def is_gpu_available() -> bool:
    """
    Check if a GPU is available for PyTorch.
    
    Returns:
        bool: True if a GPU is available, False otherwise.
    """
    return torch.cuda.is_available() and torch.cuda.device_count() > 0