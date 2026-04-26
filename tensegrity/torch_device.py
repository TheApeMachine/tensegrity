"""
Pick inference dtype and placement for transformers models.

Preference order: CUDA (device_map auto) → Apple MPS → CPU.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple


def inference_load_settings() -> Tuple[Any, Optional[str], Optional[Any]]:
    """
    Returns (torch_dtype, device_map, move_to_device).

    - CUDA: float16, device_map=\"auto\", move_to_device=None
    - MPS: float16, device_map=None, move_to_device=torch.device(\"mps\")
    - CPU: float32, device_map=None, move_to_device=None
    """
    import torch

    if torch.cuda.is_available():
        return torch.float16, "auto", None
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.float16, None, torch.device("mps")
    return torch.float32, None, None
