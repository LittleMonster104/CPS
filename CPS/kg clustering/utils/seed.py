"""Reproducibility utilities for seeding all random number generators."""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int):
    """Set random seed for reproducibility.

    Sets seeds for:
    - Python `random` module
    - NumPy random generator
    - PyTorch CPU and CUDA

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Additional PyTorch determinism settings (torch >= 1.8)
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True)
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')


def get_torch_device(device_str: str = 'auto') -> str:
    """Get the best available device.

    Args:
        device_str: Device specifier. 'auto' uses CUDA if available.

    Returns:
        Device string ('cuda', 'cuda:0', or 'cpu').
    """
    if device_str == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_str


def ensure_cuda_available(min_memory_gb: float = 16.0) -> bool:
    """Check if CUDA is available with sufficient memory.

    Args:
        min_memory_gb: Minimum GPU memory required in GB.

    Returns:
        True if CUDA is available with sufficient memory.
    """
    if not torch.cuda.is_available():
        return False

    # Check available GPU memory
    for i in range(torch.cuda.device_count()):
        memory_total = torch.cuda.get_device_properties(i).total_mem / (1024 ** 3)
        if memory_total >= min_memory_gb:
            print(f"  [GPU {i}] {torch.cuda.get_device_name(i)} | "
                  f"{memory_total:.1f}GB total")
            return True

    return False
