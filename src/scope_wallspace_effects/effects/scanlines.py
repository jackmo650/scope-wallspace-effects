"""Scanlines — CRT-style horizontal line overlay."""

import torch


def scanlines(
    frames: torch.Tensor,
    spacing: int = 2,
    opacity: float = 50.0,
    invert: bool = False,
) -> torch.Tensor:
    """Apply CRT-style horizontal scanlines.

    Args:
        frames: (T, H, W, C) in [0, 1]
        spacing: Pixel spacing between lines (1-20)
        opacity: Line darkness (0-100, 100 = fully black)
        invert: If True, invert result
    """
    if opacity <= 0 or spacing < 1:
        return frames

    _T, H, _W, _C = frames.shape
    rows = torch.arange(H, device=frames.device)
    is_scanline = (rows % spacing == 0).float()
    mask = 1.0 - (opacity / 100.0) * is_scanline
    result = frames * mask.view(1, H, 1, 1)

    if invert:
        result = 1.0 - result

    return result.clamp(0, 1)
