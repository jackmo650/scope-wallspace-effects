"""Color adjustment effects — brightness, contrast, saturation, hue shift,
gamma, black/white point, color temperature, RGB gains."""

import math

import torch


def apply_color_adjustments(
    frames: torch.Tensor,
    brightness: float = 0.0,
    contrast: float = 100.0,
    saturation: float = 100.0,
    hue_shift: float = 0.0,
) -> torch.Tensor:
    """Apply simple color adjustments matching CSS filter behavior.

    Args:
        frames: (T, H, W, C) in [0, 1]
        brightness: -100 to +100 (0 = unchanged)
        contrast: 0-200 (100 = normal)
        saturation: 0-200 (100 = normal)
        hue_shift: 0-360 degrees
    """
    result = frames

    # Brightness: shift pixel values
    if brightness != 0.0:
        result = result + (brightness / 100.0)

    # Contrast: scale around midpoint
    if contrast != 100.0:
        factor = contrast / 100.0
        result = (result - 0.5) * factor + 0.5

    # Saturation: lerp toward luminance
    if saturation != 100.0:
        lum_weights = torch.tensor([0.299, 0.587, 0.114], device=frames.device)
        lum = (result * lum_weights).sum(dim=-1, keepdim=True)
        factor = saturation / 100.0
        result = lum + factor * (result - lum)

    # Hue shift: rotate in HSV space
    if hue_shift != 0.0:
        result = _hue_rotate(result, hue_shift)

    return result.clamp(0, 1)


def apply_advanced_color(
    frames: torch.Tensor,
    gamma: float = 1.0,
    black_point: float = 0.0,
    white_point: float = 255.0,
    color_temp: float = 0.0,
) -> torch.Tensor:
    """Apply advanced color adjustments — gamma, levels, temperature.

    Args:
        frames: (T, H, W, C) in [0, 1]
        gamma: 0.1-5.0 (1.0 = linear)
        black_point: 0-255
        white_point: 0-255
        color_temp: -100 to +100 (cool to warm)
    """
    result = frames

    # Black/white point (levels)
    bp = black_point / 255.0
    wp = white_point / 255.0
    rng = max(wp - bp, 0.001)
    if bp != 0.0 or wp != 1.0:
        result = (result - bp) / rng

    # Gamma
    if gamma != 1.0:
        result = torch.pow(result.clamp(min=0.0), 1.0 / gamma)

    # Color temperature (warm/cool shift)
    if color_temp != 0.0:
        t = color_temp / 100.0
        # Warm: boost red, reduce blue. Cool: boost blue, reduce red.
        temp_gains = torch.tensor([1.0 + t * 0.15, 1.0, 1.0 - t * 0.15], device=frames.device)
        result = result * temp_gains

    return result.clamp(0, 1)


def apply_rgb_gains(
    frames: torch.Tensor,
    red_gain: float = 100.0,
    green_gain: float = 100.0,
    blue_gain: float = 100.0,
) -> torch.Tensor:
    """Per-channel RGB gain (0-200, 100 = normal)."""
    if red_gain == 100.0 and green_gain == 100.0 and blue_gain == 100.0:
        return frames

    gains = torch.tensor(
        [red_gain / 100.0, green_gain / 100.0, blue_gain / 100.0],
        device=frames.device,
    )
    return (frames * gains).clamp(0, 1)


def _hue_rotate(frames: torch.Tensor, degrees: float) -> torch.Tensor:
    """Rotate hue via a 3x3 matrix in RGB space (avoids RGB→HSV conversion)."""
    rad = math.radians(degrees)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    # Hue rotation matrix (Pregibon formula)
    sqrt3 = math.sqrt(1.0 / 3.0)
    matrix = torch.tensor([
        [cos_a + (1.0 - cos_a) / 3.0,
         (1.0 - cos_a) / 3.0 - sqrt3 * sin_a,
         (1.0 - cos_a) / 3.0 + sqrt3 * sin_a],
        [(1.0 - cos_a) / 3.0 + sqrt3 * sin_a,
         cos_a + (1.0 - cos_a) / 3.0,
         (1.0 - cos_a) / 3.0 - sqrt3 * sin_a],
        [(1.0 - cos_a) / 3.0 - sqrt3 * sin_a,
         (1.0 - cos_a) / 3.0 + sqrt3 * sin_a,
         cos_a + (1.0 - cos_a) / 3.0],
    ], device=frames.device, dtype=frames.dtype)

    # (T, H, W, 3) @ (3, 3)^T → (T, H, W, 3)
    return torch.matmul(frames, matrix.T)
