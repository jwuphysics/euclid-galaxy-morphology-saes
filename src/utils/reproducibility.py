"""
Utilities for ensuring reproducible results across all experiments.

This module provides functions to set random seeds for all libraries used
in the project, ensuring deterministic behavior for scientific reproducibility.
"""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int, deterministic_cudnn: bool = True) -> None:
    """
    Set random seeds for all libraries to ensure reproducible results.
    
    Args:
        seed: Random seed value
        deterministic_cudnn: If True, forces deterministic cuDNN operations
                           (may impact performance but ensures reproducibility)
    """
    # Set Python built-in random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seeds
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if deterministic_cudnn:
            # Force deterministic behavior in cuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_worker_init_fn(base_seed: int):
    """
    Create a worker initialization function for DataLoader to ensure 
    reproducible results across multiple workers.
    
    Args:
        base_seed: Base seed value
        
    Returns:
        Worker initialization function for DataLoader
    """
    def worker_init_fn(worker_id):
        # Set unique seed for each worker
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    return worker_init_fn


def configure_torch_reproducibility(use_deterministic: bool = True) -> None:
    """
    Configure PyTorch for maximum reproducibility.
    
    Args:
        use_deterministic: If True, use deterministic algorithms where possible
                          (may impact performance)
    """
    if use_deterministic:
        # Use deterministic algorithms when available
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set multiprocessing sharing strategy to avoid issues with some operations
    torch.multiprocessing.set_sharing_strategy('file_system')