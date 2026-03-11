# scope-wallspace-effects

GPU-accelerated effects chain for [Daydream Scope](https://docs.daydream.live/scope). Deployable as **preprocessor** or **postprocessor** in the Scope pipeline.

## Effects

| Category | Effects |
|----------|---------|
| **Color** | Brightness, Contrast, Saturation, Hue Shift |
| **Advanced Color** | Gamma, Black Point, White Point, Color Temperature |
| **RGB Gains** | Red, Green, Blue channel gain |
| **Chromakey** | Color key removal with similarity/smoothness controls |
| **Edge Detect** | Gradient magnitude edge detection |
| **Threshold** | Binary black/white threshold |
| **Blur** | Separable Gaussian blur |
| **Glow** | Blur + additive bloom |
| **Scanlines** | CRT-style horizontal line overlay |
| **Invert** | Full RGB inversion |
| **Canny** | Canny edge detection with NMS + hysteresis |
| **Lineart** | Sobel-based line extraction |
| **Lineart Anime** | Thick anime-style line extraction |
| **Softedge** | HED-like soft edge detection |
| **Scribble** | Sketch-like binary edges |
| **Depth** | Luminance-based depth approximation |
| **Color Quantize** | 8-level per-channel color reduction |

All effects have enable toggles, parameter sliders, and invert options.

## Pipelines

- `wallspace-effects-pre` — Preprocessor (runs before diffusion)
- `wallspace-effects-post` — Postprocessor (runs after diffusion)

Both share the same effect chain. Choose where to apply in the Scope pipeline.

## Install

```bash
pip install -e .
```

## Requirements

- Python 3.12+
- PyTorch (ships with Scope)
- No additional model downloads required
