"""Threshold — binary black/white conversion."""

import torch


def threshold(
    frames: torch.Tensor,
    level: float = 128.0,
    invert: bool = False,
) -> torch.Tensor:
    """Convert to binary black/white based on luminance threshold.

    Args:
        frames: (T, H, W, C) in [0, 1]
        level: Threshold level (0-255 range, applied to [0,1] after conversion)
        invert: If True, swap black and white
    """
    lum_weights = torch.tensor([0.299, 0.587, 0.114], device=frames.device, dtype=frames.dtype)
    lum = (frames * lum_weights).sum(dim=-1)  # (T, H, W)

    thresh_01 = level / 255.0
    binary = (lum >= thresh_01).float()

    if invert:
        binary = 1.0 - binary

    return binary.unsqueeze(-1).expand_as(frames)
