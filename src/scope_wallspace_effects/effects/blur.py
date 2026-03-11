"""Gaussian blur — separable 2-pass for performance."""

import torch
import torch.nn.functional as F


def gaussian_blur(
    frames: torch.Tensor,
    radius: float = 5.0,
    invert: bool = False,
) -> torch.Tensor:
    """Apply Gaussian blur using separable 1D convolutions.

    Args:
        frames: (T, H, W, C) in [0, 1]
        radius: Blur radius in pixels (1-50)
        invert: If True, invert result after blurring
    """
    if radius < 1.0:
        return frames

    # Build 1D Gaussian kernel
    kernel_size = int(radius * 2) | 1  # Ensure odd
    if kernel_size < 3:
        kernel_size = 3

    sigma = radius / 3.0
    x = torch.arange(kernel_size, device=frames.device, dtype=frames.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Reshape for conv: (T, H, W, C) → (T*C, 1, H, W)
    T, H, W, C = frames.shape
    result = frames.permute(0, 3, 1, 2).reshape(T * C, 1, H, W)

    pad_h = kernel_size // 2
    pad_w = kernel_size // 2

    # Horizontal pass
    k_h = kernel_1d.view(1, 1, 1, kernel_size)
    result = F.pad(result, [pad_w, pad_w, 0, 0], mode="reflect")
    result = F.conv2d(result, k_h)

    # Vertical pass
    k_v = kernel_1d.view(1, 1, kernel_size, 1)
    result = F.pad(result, [0, 0, pad_h, pad_h], mode="reflect")
    result = F.conv2d(result, k_v)

    # Reshape back: (T*C, 1, H, W) → (T, H, W, C)
    result = result.reshape(T, C, H, W).permute(0, 2, 3, 1)

    if invert:
        result = 1.0 - result

    return result.clamp(0, 1)
