import torch

def get_device() -> torch.device:
    """
    Get the available device (CUDA or CPU) for tensor computations.

    Returns:
    torch.device: Device object representing CUDA or CPU.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)