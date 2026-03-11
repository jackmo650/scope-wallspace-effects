"""Edge detection — simplified gradient magnitude to greyscale."""

import torch


def edge_detect(
    frames: torch.Tensor,
    strength: float = 100.0,
    invert: bool = False,
) -> torch.Tensor:
    """Detect edges using gradient magnitude (right + bottom neighbor difference).

    Args:
        frames: (T, H, W, C) in [0, 1]
        strength: Edge intensity multiplier (0-200, 100 = normal)
        invert: If True, invert result (dark edges on white)
    """
    # Convert to luminance
    lum_weights = torch.tensor([0.299, 0.587, 0.114], device=frames.device, dtype=frames.dtype)
    lum = (frames * lum_weights).sum(dim=-1)  # (T, H, W)

    # Compute gradients via shifted subtraction
    # Right neighbor difference
    grad_x = torch.zeros_like(lum)
    grad_x[:, :, :-1] = torch.abs(lum[:, :, :-1] - lum[:, :, 1:])

    # Bottom neighbor difference
    grad_y = torch.zeros_like(lum)
    grad_y[:, :-1, :] = torch.abs(lum[:, :-1, :] - lum[:, 1:, :])

    factor = (strength / 100.0) * 2.0
    edge = ((grad_x + grad_y) * factor).clamp(0, 1)

    if invert:
        edge = 1.0 - edge

    # Expand back to 3-channel greyscale
    return edge.unsqueeze(-1).expand_as(frames)
