"""Glow effect — blurred additive blend for bloom."""

import torch

from .blur import gaussian_blur


def glow(
    frames: torch.Tensor,
    radius: float = 10.0,
    intensity: float = 100.0,
    invert: bool = False,
) -> torch.Tensor:
    """Apply glow by adding a blurred copy of the frame (additive blend).

    Args:
        frames: (T, H, W, C) in [0, 1]
        radius: Glow blur radius (1-50)
        intensity: Glow brightness (0-300, 100 = normal)
        invert: If True, invert result after glow
    """
    if intensity <= 0 or radius < 1:
        return frames

    blurred = gaussian_blur(frames, radius=radius)

    # Additive blend: add scaled blurred version
    alpha = (intensity / 100.0) * 0.4
    result = frames + blurred * alpha

    if invert:
        result = 1.0 - result

    return result.clamp(0, 1)
