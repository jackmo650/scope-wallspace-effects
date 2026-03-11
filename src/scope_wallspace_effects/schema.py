from pydantic import Field
from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    UsageType,
    ui_field_config,
)


class _WallspaceEffectsBaseConfig(BasePipelineConfig):
    """Shared config for pre and post effects pipelines."""

    pipeline_description = (
        "GPU-accelerated effects chain: chromakey, edge detect, threshold, blur, "
        "glow, scanlines, invert, RGB gains, color adjustments, and ControlNet "
        "preprocessors (canny, lineart, softedge, scribble, depth, color quantize)"
    )
    pipeline_version = "0.1.0"
    estimated_vram_gb = 1.0
    supports_prompts = False
    supports_lora = False
    supports_vace = False
    supports_cache_management = False
    supports_quantization = False

    modes = {"video": ModeDefaults(default=True)}

    # ── Color Adjustments (Simple) ────────────────────────────────────────

    brightness: float = Field(
        default=0.0, ge=-100.0, le=100.0,
        description="Brightness adjustment (-100 to +100, 0 = unchanged)",
        json_schema_extra=ui_field_config(order=1, label="Brightness"),
    )
    contrast: float = Field(
        default=100.0, ge=0.0, le=200.0,
        description="Contrast (0-200, 100 = normal)",
        json_schema_extra=ui_field_config(order=2, label="Contrast"),
    )
    saturation: float = Field(
        default=100.0, ge=0.0, le=200.0,
        description="Saturation (0-200, 100 = normal)",
        json_schema_extra=ui_field_config(order=3, label="Saturation"),
    )
    hue_shift: float = Field(
        default=0.0, ge=0.0, le=360.0,
        description="Hue rotation in degrees (0-360)",
        json_schema_extra=ui_field_config(order=4, label="Hue Shift"),
    )

    # ── Color Adjustments (Advanced) ──────────────────────────────────────

    gamma: float = Field(
        default=1.0, ge=0.1, le=5.0,
        description="Gamma correction (1.0 = linear, <1 = brighter, >1 = darker)",
        json_schema_extra=ui_field_config(order=10, label="Gamma"),
    )
    black_point: float = Field(
        default=0.0, ge=0.0, le=255.0,
        description="Black point level (0-255, pixels below become black)",
        json_schema_extra=ui_field_config(order=11, label="Black Point"),
    )
    white_point: float = Field(
        default=255.0, ge=0.0, le=255.0,
        description="White point level (0-255, pixels above become white)",
        json_schema_extra=ui_field_config(order=12, label="White Point"),
    )
    color_temp: float = Field(
        default=0.0, ge=-100.0, le=100.0,
        description="Color temperature shift (-100 cool blue to +100 warm orange)",
        json_schema_extra=ui_field_config(order=13, label="Color Temperature"),
    )

    # ── RGB Gains ─────────────────────────────────────────────────────────

    red_gain: float = Field(
        default=100.0, ge=0.0, le=200.0,
        description="Red channel gain (0-200, 100 = normal)",
        json_schema_extra=ui_field_config(order=20, label="Red Gain"),
    )
    green_gain: float = Field(
        default=100.0, ge=0.0, le=200.0,
        description="Green channel gain (0-200, 100 = normal)",
        json_schema_extra=ui_field_config(order=21, label="Green Gain"),
    )
    blue_gain: float = Field(
        default=100.0, ge=0.0, le=200.0,
        description="Blue channel gain (0-200, 100 = normal)",
        json_schema_extra=ui_field_config(order=22, label="Blue Gain"),
    )

    # ── Chromakey ─────────────────────────────────────────────────────────

    chromakey_enabled: bool = Field(
        default=False,
        description="Enable chroma key (green/blue screen removal)",
        json_schema_extra=ui_field_config(order=30, label="Chromakey"),
    )
    chromakey_color_r: int = Field(
        default=0, ge=0, le=255,
        description="Key color red component (0-255)",
        json_schema_extra=ui_field_config(order=31, label="Key Red"),
    )
    chromakey_color_g: int = Field(
        default=255, ge=0, le=255,
        description="Key color green component (0-255)",
        json_schema_extra=ui_field_config(order=32, label="Key Green"),
    )
    chromakey_color_b: int = Field(
        default=0, ge=0, le=255,
        description="Key color blue component (0-255)",
        json_schema_extra=ui_field_config(order=33, label="Key Blue"),
    )
    chromakey_similarity: float = Field(
        default=40.0, ge=0.0, le=100.0,
        description="Color distance threshold for key removal (0-100)",
        json_schema_extra=ui_field_config(order=34, label="Similarity"),
    )
    chromakey_smoothness: float = Field(
        default=10.0, ge=0.0, le=100.0,
        description="Edge smoothness for gradual alpha falloff (0-100)",
        json_schema_extra=ui_field_config(order=35, label="Smoothness"),
    )
    chromakey_invert: bool = Field(
        default=False,
        description="Invert chromakey result",
        json_schema_extra=ui_field_config(order=36, label="Invert Chromakey"),
    )

    # ── Edge Detect ───────────────────────────────────────────────────────

    edge_detect_enabled: bool = Field(
        default=False,
        description="Enable edge detection (gradient magnitude to greyscale)",
        json_schema_extra=ui_field_config(order=40, label="Edge Detect"),
    )
    edge_detect_strength: float = Field(
        default=100.0, ge=0.0, le=200.0,
        description="Edge detection strength (0-200, 100 = normal)",
        json_schema_extra=ui_field_config(order=41, label="Edge Strength"),
    )
    edge_detect_invert: bool = Field(
        default=False,
        description="Invert edge detection result (white edges on black)",
        json_schema_extra=ui_field_config(order=42, label="Invert Edges"),
    )

    # ── Threshold ─────────────────────────────────────────────────────────

    threshold_enabled: bool = Field(
        default=False,
        description="Enable threshold (binary black/white)",
        json_schema_extra=ui_field_config(order=50, label="Threshold"),
    )
    threshold_level: float = Field(
        default=128.0, ge=0.0, le=255.0,
        description="Luminance threshold (0-255, pixels above = white)",
        json_schema_extra=ui_field_config(order=51, label="Threshold Level"),
    )
    threshold_invert: bool = Field(
        default=False,
        description="Invert threshold result",
        json_schema_extra=ui_field_config(order=52, label="Invert Threshold"),
    )

    # ── Blur ──────────────────────────────────────────────────────────────

    blur_enabled: bool = Field(
        default=False,
        description="Enable Gaussian blur",
        json_schema_extra=ui_field_config(order=60, label="Blur"),
    )
    blur_radius: float = Field(
        default=5.0, ge=1.0, le=50.0,
        description="Blur radius in pixels (1-50)",
        json_schema_extra=ui_field_config(order=61, label="Blur Radius"),
    )
    blur_invert: bool = Field(
        default=False,
        description="Invert blurred result",
        json_schema_extra=ui_field_config(order=62, label="Invert Blur"),
    )

    # ── Glow ──────────────────────────────────────────────────────────────

    glow_enabled: bool = Field(
        default=False,
        description="Enable glow (blur + additive blend for bloom effect)",
        json_schema_extra=ui_field_config(order=70, label="Glow"),
    )
    glow_radius: float = Field(
        default=10.0, ge=1.0, le=50.0,
        description="Glow blur radius (1-50)",
        json_schema_extra=ui_field_config(order=71, label="Glow Radius"),
    )
    glow_intensity: float = Field(
        default=100.0, ge=0.0, le=300.0,
        description="Glow intensity (0-300, 100 = normal)",
        json_schema_extra=ui_field_config(order=72, label="Glow Intensity"),
    )
    glow_invert: bool = Field(
        default=False,
        description="Invert glow result",
        json_schema_extra=ui_field_config(order=73, label="Invert Glow"),
    )

    # ── Scanlines ─────────────────────────────────────────────────────────

    scanlines_enabled: bool = Field(
        default=False,
        description="Enable CRT-style scanline overlay",
        json_schema_extra=ui_field_config(order=80, label="Scanlines"),
    )
    scanlines_spacing: int = Field(
        default=2, ge=1, le=20,
        description="Spacing between scan lines in pixels (1-20)",
        json_schema_extra=ui_field_config(order=81, label="Line Spacing"),
    )
    scanlines_opacity: float = Field(
        default=50.0, ge=0.0, le=100.0,
        description="Scanline darkness (0-100, 100 = fully black lines)",
        json_schema_extra=ui_field_config(order=82, label="Line Opacity"),
    )
    scanlines_invert: bool = Field(
        default=False,
        description="Invert scanlines result",
        json_schema_extra=ui_field_config(order=83, label="Invert Scanlines"),
    )

    # ── Invert ────────────────────────────────────────────────────────────

    invert_enabled: bool = Field(
        default=False,
        description="Enable full RGB inversion",
        json_schema_extra=ui_field_config(order=90, label="Invert"),
    )

    # ── ControlNet: Canny ─────────────────────────────────────────────────

    canny_enabled: bool = Field(
        default=False,
        description="Enable Canny edge detection (ControlNet preprocessor)",
        json_schema_extra=ui_field_config(order=100, label="Canny"),
    )
    canny_low_threshold: float = Field(
        default=100.0, ge=0.0, le=255.0,
        description="Canny low threshold for weak edges (0-255)",
        json_schema_extra=ui_field_config(order=101, label="Canny Low"),
    )
    canny_high_threshold: float = Field(
        default=200.0, ge=0.0, le=255.0,
        description="Canny high threshold for strong edges (0-255)",
        json_schema_extra=ui_field_config(order=102, label="Canny High"),
    )
    canny_invert: bool = Field(
        default=False,
        description="Invert canny result (white edges on black)",
        json_schema_extra=ui_field_config(order=103, label="Invert Canny"),
    )

    # ── ControlNet: Lineart ───────────────────────────────────────────────

    lineart_enabled: bool = Field(
        default=False,
        description="Enable lineart extraction (ControlNet preprocessor)",
        json_schema_extra=ui_field_config(order=110, label="Lineart"),
    )
    lineart_line_weight: float = Field(
        default=1.0, ge=0.5, le=5.0,
        description="Line thickness (0.5-5.0, 1.0 = normal)",
        json_schema_extra=ui_field_config(order=111, label="Line Weight"),
    )
    lineart_invert: bool = Field(
        default=False,
        description="Invert lineart result",
        json_schema_extra=ui_field_config(order=112, label="Invert Lineart"),
    )

    # ── ControlNet: Lineart Anime ─────────────────────────────────────────

    lineart_anime_enabled: bool = Field(
        default=False,
        description="Enable anime-style lineart extraction (ControlNet preprocessor)",
        json_schema_extra=ui_field_config(order=120, label="Lineart Anime"),
    )
    lineart_anime_line_weight: float = Field(
        default=1.0, ge=0.5, le=5.0,
        description="Anime line thickness (0.5-5.0, 1.0 = normal)",
        json_schema_extra=ui_field_config(order=121, label="Anime Line Weight"),
    )
    lineart_anime_invert: bool = Field(
        default=False,
        description="Invert anime lineart result",
        json_schema_extra=ui_field_config(order=122, label="Invert Anime Lineart"),
    )

    # ── ControlNet: Softedge ──────────────────────────────────────────────

    softedge_enabled: bool = Field(
        default=False,
        description="Enable soft edge detection (HED-like, ControlNet preprocessor)",
        json_schema_extra=ui_field_config(order=130, label="Softedge"),
    )
    softedge_blur_radius: float = Field(
        default=3.0, ge=1.0, le=10.0,
        description="Softedge blur sigma (1-10)",
        json_schema_extra=ui_field_config(order=131, label="Softedge Blur"),
    )
    softedge_invert: bool = Field(
        default=False,
        description="Invert softedge result",
        json_schema_extra=ui_field_config(order=132, label="Invert Softedge"),
    )

    # ── ControlNet: Scribble ──────────────────────────────────────────────

    scribble_enabled: bool = Field(
        default=False,
        description="Enable scribble edge extraction (sketch-like, ControlNet preprocessor)",
        json_schema_extra=ui_field_config(order=140, label="Scribble"),
    )
    scribble_threshold: float = Field(
        default=50.0, ge=0.0, le=255.0,
        description="Scribble edge threshold (0-255)",
        json_schema_extra=ui_field_config(order=141, label="Scribble Threshold"),
    )
    scribble_invert: bool = Field(
        default=False,
        description="Invert scribble result",
        json_schema_extra=ui_field_config(order=142, label="Invert Scribble"),
    )

    # ── ControlNet: Depth ─────────────────────────────────────────────────

    depth_enabled: bool = Field(
        default=False,
        description="Enable depth approximation (luminance-based, ControlNet preprocessor)",
        json_schema_extra=ui_field_config(order=150, label="Depth"),
    )
    depth_invert: bool = Field(
        default=False,
        description="Invert depth map",
        json_schema_extra=ui_field_config(order=151, label="Invert Depth"),
    )

    # ── ControlNet: Color Quantize ────────────────────────────────────────

    color_quantize_enabled: bool = Field(
        default=False,
        description="Enable color quantization / downsampling (ControlNet preprocessor)",
        json_schema_extra=ui_field_config(order=160, label="Color Quantize"),
    )
    color_quantize_invert: bool = Field(
        default=False,
        description="Invert color quantized result",
        json_schema_extra=ui_field_config(order=161, label="Invert Color Quantize"),
    )


class WallspaceEffectsPreConfig(_WallspaceEffectsBaseConfig):
    """Preprocessor variant — runs before the main diffusion pipeline."""

    pipeline_id = "wallspace-effects-pre"
    pipeline_name = "WS Effects (Pre)"

    usage = [UsageType.PREPROCESSOR]


class WallspaceEffectsPostConfig(_WallspaceEffectsBaseConfig):
    """Postprocessor/standalone variant — runs after pipeline or standalone."""

    pipeline_id = "wallspace-effects-post"
    pipeline_name = "WS Effects (Post)"
