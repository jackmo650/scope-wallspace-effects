"""Invert — RGB channel inversion."""

import torch


def invert(frames: torch.Tensor) -> torch.Tensor:
    """Invert RGB channels (1.0 - value).

    Args:
        frames: (T, H, W, C) in [0, 1]
    """
    return 1.0 - frames
