"""ControlNet preprocessors — GPU-accelerated via PyTorch.
Mirrors the browser-based preprocessors in controlnetPreprocessor.ts."""

import torch
import torch.nn.functional as F


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_luminance(frames: torch.Tensor) -> torch.Tensor:
    """(T, H, W, C) → (T, H, W) luminance."""
    weights = torch.tensor([0.299, 0.587, 0.114], device=frames.device, dtype=frames.dtype)
    return (frames[..., :3] * weights).sum(dim=-1)


def _sobel_magnitude(lum: torch.Tensor) -> torch.Tensor:
    """Compute Sobel gradient magnitude. Input: (T, H, W), Output: (T, H, W)."""
    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           device=lum.device, dtype=lum.dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           device=lum.device, dtype=lum.dtype).view(1, 1, 3, 3)

    # (T, H, W) → (T, 1, H, W)
    x = lum.unsqueeze(1)
    x = F.pad(x, [1, 1, 1, 1], mode="reflect")
    gx = F.conv2d(x, sobel_x)
    gy = F.conv2d(x, sobel_y)
    mag = torch.sqrt(gx * gx + gy * gy).squeeze(1)
    return mag


def _gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply separable Gaussian blur. Input: (T, 1, H, W)."""
    kernel_size = int(sigma * 6) | 1
    if kernel_size < 3:
        kernel_size = 3
    half = kernel_size // 2
    t = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - half
    k1d = torch.exp(-0.5 * (t / max(sigma, 0.01)) ** 2)
    k1d = k1d / k1d.sum()

    # Horizontal
    kh = k1d.view(1, 1, 1, kernel_size)
    out = F.pad(x, [half, half, 0, 0], mode="reflect")
    out = F.conv2d(out, kh)
    # Vertical
    kv = k1d.view(1, 1, kernel_size, 1)
    out = F.pad(out, [0, 0, half, half], mode="reflect")
    out = F.conv2d(out, kv)
    return out


def _to_greyscale_rgb(single_channel: torch.Tensor, frames: torch.Tensor, invert: bool) -> torch.Tensor:
    """Convert single-channel (T, H, W) to RGB (T, H, W, C), optionally inverted."""
    if invert:
        single_channel = 1.0 - single_channel
    return single_channel.clamp(0, 1).unsqueeze(-1).expand_as(frames)


# ── Canny Edge Detection ────────────────────────────────────────────────────

def canny(
    frames: torch.Tensor,
    low_threshold: float = 100.0,
    high_threshold: float = 200.0,
    invert: bool = False,
) -> torch.Tensor:
    """Canny edge detection with non-maximum suppression and hysteresis.

    Args:
        frames: (T, H, W, C) in [0, 1]
        low_threshold: Weak edge threshold (0-255)
        high_threshold: Strong edge threshold (0-255)
        invert: If True, invert result
    """
    lum = _to_luminance(frames)

    # Pre-blur to reduce noise
    blurred = _gaussian_blur_2d(lum.unsqueeze(1), sigma=1.4).squeeze(1)

    # Sobel gradients for magnitude and direction
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           device=frames.device, dtype=frames.dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           device=frames.device, dtype=frames.dtype).view(1, 1, 3, 3)

    x = blurred.unsqueeze(1)
    x_padded = F.pad(x, [1, 1, 1, 1], mode="reflect")
    gx = F.conv2d(x_padded, sobel_x).squeeze(1)
    gy = F.conv2d(x_padded, sobel_y).squeeze(1)
    mag = torch.sqrt(gx * gx + gy * gy)

    # Non-maximum suppression (simplified: quantize angle to 4 directions)
    angle = torch.atan2(gy, gx)
    # Quantize to 0, 45, 90, 135 degrees
    angle_deg = (angle * 180.0 / 3.14159 + 180.0) % 180.0

    # Pad magnitude for neighbor comparison
    mag_pad = F.pad(mag.unsqueeze(1), [1, 1, 1, 1], mode="reflect").squeeze(1)
    T, H, W = mag.shape

    # Compare with neighbors along gradient direction
    suppressed = torch.zeros_like(mag)
    for t_idx in range(T):
        m = mag[t_idx]
        mp = mag_pad[t_idx]
        a = angle_deg[t_idx]

        # Horizontal edges (angle ~0 or ~180)
        h_mask = ((a < 22.5) | (a >= 157.5))
        # Diagonal 45 edges
        d45_mask = ((a >= 22.5) & (a < 67.5))
        # Vertical edges (angle ~90)
        v_mask = ((a >= 67.5) & (a < 112.5))
        # Diagonal 135 edges
        d135_mask = ((a >= 112.5) & (a < 157.5))

        nms = torch.zeros_like(m)
        # For each direction, keep pixel only if it's a local maximum
        nms += h_mask.float() * m * (m >= mp[1:H+1, 2:W+2]) * (m >= mp[1:H+1, :W])
        nms += d45_mask.float() * m * (m >= mp[:H, 2:W+2]) * (m >= mp[2:H+2, :W])
        nms += v_mask.float() * m * (m >= mp[:H, 1:W+1]) * (m >= mp[2:H+2, 1:W+1])
        nms += d135_mask.float() * m * (m >= mp[:H, :W]) * (m >= mp[2:H+2, 2:W+2])
        suppressed[t_idx] = nms

    # Double thresholding
    low_t = low_threshold / 255.0
    high_t = high_threshold / 255.0
    strong = (suppressed >= high_t).float()
    weak = ((suppressed >= low_t) & (suppressed < high_t)).float()

    # Simple hysteresis: dilate strong edges and keep weak edges that touch
    kernel = torch.ones(1, 1, 3, 3, device=frames.device)
    dilated = F.conv2d(
        F.pad(strong.unsqueeze(1), [1, 1, 1, 1], mode="constant", value=0),
        kernel
    ).squeeze(1)
    edges = (strong + weak * (dilated > 0).float()).clamp(0, 1)

    return _to_greyscale_rgb(edges, frames, invert)


# ── Lineart Extraction ───────────────────────────────────────────────────────

def lineart(
    frames: torch.Tensor,
    line_weight: float = 1.0,
    invert: bool = False,
) -> torch.Tensor:
    """Extract line art using Sobel gradient + threshold.

    Args:
        frames: (T, H, W, C) in [0, 1]
        line_weight: Edge thickness multiplier (0.5-5.0)
        invert: If True, dark lines on white background
    """
    lum = _to_luminance(frames)
    mag = _sobel_magnitude(lum)

    # Scale by line weight and threshold
    mag = (mag * line_weight * 2.0).clamp(0, 1)

    # Default: white lines on black background
    return _to_greyscale_rgb(mag, frames, invert)


def lineart_anime(
    frames: torch.Tensor,
    line_weight: float = 1.0,
    invert: bool = False,
) -> torch.Tensor:
    """Anime-style lineart — thicker strokes with higher threshold.

    Args:
        frames: (T, H, W, C) in [0, 1]
        line_weight: Edge thickness multiplier (0.5-5.0)
        invert: If True, dark lines on white background
    """
    lum = _to_luminance(frames)

    # Pre-blur for smoother, thicker lines
    blurred = _gaussian_blur_2d(lum.unsqueeze(1), sigma=1.5).squeeze(1)
    mag = _sobel_magnitude(blurred)

    # Stronger scaling + binary threshold for anime-clean look
    mag = (mag * line_weight * 3.0)
    binary = (mag > 0.15).float()

    return _to_greyscale_rgb(binary, frames, invert)


# ── Soft Edge Detection ──────────────────────────────────────────────────────

def softedge(
    frames: torch.Tensor,
    blur_radius: float = 3.0,
    invert: bool = False,
) -> torch.Tensor:
    """HED-like soft edge detection — Sobel magnitude with Gaussian smoothing.

    Args:
        frames: (T, H, W, C) in [0, 1]
        blur_radius: Gaussian sigma for smoothing (1-10)
        invert: If True, invert result
    """
    lum = _to_luminance(frames)
    mag = _sobel_magnitude(lum)

    # Smooth the edge map
    if blur_radius > 0:
        mag = _gaussian_blur_2d(mag.unsqueeze(1), sigma=blur_radius).squeeze(1)

    mag = (mag * 2.0).clamp(0, 1)
    return _to_greyscale_rgb(mag, frames, invert)


# ── Scribble Edge ────────────────────────────────────────────────────────────

def scribble(
    frames: torch.Tensor,
    threshold_val: float = 50.0,
    invert: bool = False,
) -> torch.Tensor:
    """Sketch-like edge extraction with hard threshold.

    Args:
        frames: (T, H, W, C) in [0, 1]
        threshold_val: Edge threshold (0-255)
        invert: If True, invert result
    """
    lum = _to_luminance(frames)
    mag = _sobel_magnitude(lum)

    thresh = threshold_val / 255.0
    binary = (mag > thresh).float()

    return _to_greyscale_rgb(binary, frames, invert)


# ── Depth Approximation ─────────────────────────────────────────────────────

def depth_approx(
    frames: torch.Tensor,
    invert: bool = False,
) -> torch.Tensor:
    """Approximate depth from luminance (not ML-based).

    Args:
        frames: (T, H, W, C) in [0, 1]
        invert: If True, invert depth map (near = dark, far = bright)
    """
    depth = _to_luminance(frames)
    return _to_greyscale_rgb(depth, frames, invert)


# ── Color Quantization ──────────────────────────────────────────────────────

def color_quantize(
    frames: torch.Tensor,
    invert: bool = False,
) -> torch.Tensor:
    """Downsample colors to reduced palette (8 levels per channel).

    Args:
        frames: (T, H, W, C) in [0, 1]
        invert: If True, invert result
    """
    levels = 8.0
    quantized = torch.round(frames * levels) / levels

    if invert:
        quantized = 1.0 - quantized

    return quantized.clamp(0, 1)
