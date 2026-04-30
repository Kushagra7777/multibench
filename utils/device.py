"""Device detection utilities for MultiBench."""
import torch


def get_device() -> torch.device:
    """Return the best available device: CUDA → MPS → CPU.

    Priority order:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon GPU via Metal Performance Shaders, PyTorch ≥ 1.12)
    3. CPU (fallback)

    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")
