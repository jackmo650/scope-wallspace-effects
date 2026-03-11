"""WallSpace Effects pipeline — GPU-accelerated effects chain.

Registers as both preprocessor (WS Effects Pre) and postprocessor (WS Effects Post).
Both share the same effect chain; users choose where to apply them in the Scope pipeline.
"""

import logging
from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline, Requirements

from .effects import (
    apply_color_adjustments,
    apply_advanced_color,
    apply_rgb_gains,
    chromakey,
    edge_detect,
    threshold,
    gaussian_blur,
    glow,
    scanlines,
    invert,
    canny,
    lineart,
    lineart_anime,
    softedge,
    scribble,
    depth_approx,
    color_quantize,
)

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

from .schema import WallspaceEffectsPreConfig, WallspaceEffectsPostConfig

logger = logging.getLogger(__name__)


class _WallspaceEffectsBase(Pipeline):
    """Shared effect chain logic for pre and post variants."""

    def __init__(self, device: torch.device | None = None, **kwargs):
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info("WS Effects using device: %s", self.device)

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("WallspaceEffectsPipeline requires video input")

        # Stack and normalize: list of (1,H,W,C) → (T,H,W,C) in [0,1]
        frames = torch.stack([frame.squeeze(0) for frame in video], dim=0)
        frames = frames.to(device=self.device, dtype=torch.float32) / 255.0

        # ── 1. Simple color adjustments ──────────────────────────────────
        brightness = kwargs.get("brightness", 0.0)
        contrast = kwargs.get("contrast", 100.0)
        saturation = kwargs.get("saturation", 100.0)
        hue_shift = kwargs.get("hue_shift", 0.0)

        if brightness != 0.0 or contrast != 100.0 or saturation != 100.0 or hue_shift != 0.0:
            frames = apply_color_adjustments(
                frames,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue_shift=hue_shift,
            )

        # ── 2. Advanced color adjustments ────────────────────────────────
        gamma = kwargs.get("gamma", 1.0)
        black_point = kwargs.get("black_point", 0.0)
        white_point = kwargs.get("white_point", 255.0)
        color_temp = kwargs.get("color_temp", 0.0)

        if gamma != 1.0 or black_point != 0.0 or white_point != 255.0 or color_temp != 0.0:
            frames = apply_advanced_color(
                frames,
                gamma=gamma,
                black_point=black_point,
                white_point=white_point,
                color_temp=color_temp,
            )

        # ── 3. RGB gains ────────────────────────────────────────────────
        red_gain = kwargs.get("red_gain", 100.0)
        green_gain = kwargs.get("green_gain", 100.0)
        blue_gain = kwargs.get("blue_gain", 100.0)

        if red_gain != 100.0 or green_gain != 100.0 or blue_gain != 100.0:
            frames = apply_rgb_gains(
                frames,
                red_gain=red_gain,
                green_gain=green_gain,
                blue_gain=blue_gain,
            )

        # ── 4. Effect chain (order matches app effects pipeline) ─────────

        # Chromakey
        if kwargs.get("chromakey_enabled", False):
            frames = chromakey(
                frames,
                key_r=kwargs.get("chromakey_color_r", 0),
                key_g=kwargs.get("chromakey_color_g", 255),
                key_b=kwargs.get("chromakey_color_b", 0),
                similarity=kwargs.get("chromakey_similarity", 40.0),
                smoothness=kwargs.get("chromakey_smoothness", 10.0),
                invert=kwargs.get("chromakey_invert", False),
            )

        # Edge detect
        if kwargs.get("edge_detect_enabled", False):
            frames = edge_detect(
                frames,
                strength=kwargs.get("edge_detect_strength", 100.0),
                invert=kwargs.get("edge_detect_invert", False),
            )

        # Threshold
        if kwargs.get("threshold_enabled", False):
            frames = threshold(
                frames,
                level=kwargs.get("threshold_level", 128.0),
                invert=kwargs.get("threshold_invert", False),
            )

        # Blur
        if kwargs.get("blur_enabled", False):
            frames = gaussian_blur(
                frames,
                radius=kwargs.get("blur_radius", 5.0),
                invert=kwargs.get("blur_invert", False),
            )

        # Glow
        if kwargs.get("glow_enabled", False):
            frames = glow(
                frames,
                radius=kwargs.get("glow_radius", 10.0),
                intensity=kwargs.get("glow_intensity", 100.0),
                invert=kwargs.get("glow_invert", False),
            )

        # Scanlines
        if kwargs.get("scanlines_enabled", False):
            frames = scanlines(
                frames,
                spacing=kwargs.get("scanlines_spacing", 2),
                opacity=kwargs.get("scanlines_opacity", 50.0),
                invert=kwargs.get("scanlines_invert", False),
            )

        # Full invert
        if kwargs.get("invert_enabled", False):
            frames = invert(frames)

        # ── 5. ControlNet preprocessors ──────────────────────────────────

        # Canny
        if kwargs.get("canny_enabled", False):
            frames = canny(
                frames,
                low_threshold=kwargs.get("canny_low_threshold", 100.0),
                high_threshold=kwargs.get("canny_high_threshold", 200.0),
                invert=kwargs.get("canny_invert", False),
            )

        # Lineart
        if kwargs.get("lineart_enabled", False):
            frames = lineart(
                frames,
                line_weight=kwargs.get("lineart_line_weight", 1.0),
                invert=kwargs.get("lineart_invert", False),
            )

        # Lineart Anime
        if kwargs.get("lineart_anime_enabled", False):
            frames = lineart_anime(
                frames,
                line_weight=kwargs.get("lineart_anime_line_weight", 1.0),
                invert=kwargs.get("lineart_anime_invert", False),
            )

        # Softedge
        if kwargs.get("softedge_enabled", False):
            frames = softedge(
                frames,
                blur_radius=kwargs.get("softedge_blur_radius", 3.0),
                invert=kwargs.get("softedge_invert", False),
            )

        # Scribble
        if kwargs.get("scribble_enabled", False):
            frames = scribble(
                frames,
                threshold_val=kwargs.get("scribble_threshold", 50.0),
                invert=kwargs.get("scribble_invert", False),
            )

        # Depth
        if kwargs.get("depth_enabled", False):
            frames = depth_approx(
                frames,
                invert=kwargs.get("depth_invert", False),
            )

        # Color Quantize
        if kwargs.get("color_quantize_enabled", False):
            frames = color_quantize(
                frames,
                invert=kwargs.get("color_quantize_invert", False),
            )

        # Move to CPU for Scope frame encoding
        result = frames.clamp(0, 1)
        if result.device.type != "cpu":
            result = result.cpu()

        return {"video": result}


class WallspaceEffectsPrePipeline(_WallspaceEffectsBase):
    """Preprocessor variant — runs before the main diffusion pipeline."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return WallspaceEffectsPreConfig


class WallspaceEffectsPostPipeline(_WallspaceEffectsBase):
    """Postprocessor/standalone variant — runs after pipeline or standalone."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return WallspaceEffectsPostConfig
