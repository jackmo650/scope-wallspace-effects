"""Chromakey (green/blue screen removal) — GPU-accelerated."""

import torch


def chromakey(
    frames: torch.Tensor,
    key_r: int = 0,
    key_g: int = 255,
    key_b: int = 0,
    similarity: float = 40.0,
    smoothness: float = 10.0,
    invert: bool = False,
) -> torch.Tensor:
    """Remove pixels matching a key color, setting alpha via distance.

    Since Scope works in RGB (no alpha channel), keyed pixels are set to black.
    In invert mode, non-keyed pixels are set to black instead.

    Args:
        frames: (T, H, W, C) in [0, 1]
        key_r/g/b: Key color components (0-255)
        similarity: Color distance threshold (0-100)
        smoothness: Edge smoothness (0-100)
        invert: If True, keep keyed area, remove everything else
    """
    # Convert key to [0, 1] range
    key = torch.tensor(
        [key_r / 255.0, key_g / 255.0, key_b / 255.0],
        device=frames.device, dtype=frames.dtype,
    )

    threshold = similarity * 2.55 / 255.0  # Convert to [0, 1] space
    smooth = smoothness * 2.55 / 255.0

    # Compute color distance from key
    diff = frames[..., :3] - key
    dist = torch.sqrt((diff * diff).sum(dim=-1))

    # Compute alpha: 0 = fully keyed, 1 = fully visible
    outer = threshold + smooth
    alpha = torch.where(
        dist < threshold,
        torch.zeros_like(dist),
        torch.where(
            dist < outer,
            (dist - threshold) / max(smooth, 1e-6),
            torch.ones_like(dist),
        ),
    )

    if invert:
        alpha = 1.0 - alpha

    # Apply alpha as mask (multiply RGB by alpha)
    return (frames * alpha.unsqueeze(-1)).clamp(0, 1)
