"""Fixed-width integer quantization of RT-density vectors (proposal Section 3).

A normalized density in [0, 1] becomes integers ``000..scale`` (default 0..100),
each rendered as a zero-padded fixed-width token so the Gemma tokenizer emits a
predictable number of tokens per bin. Strict, lossless, easy to parse.
"""

from __future__ import annotations

import numpy as np

from elutediff.config import TargetConfig


def quantize(density: np.ndarray, cfg: TargetConfig) -> np.ndarray:
    """Quantize a normalized density (values in [0, 1]) to integers ``0..scale``."""
    density = np.asarray(density, dtype=float)
    levels = np.rint(np.clip(density, 0.0, 1.0) * cfg.scale)
    return levels.astype(int)


def dequantize(levels: np.ndarray, cfg: TargetConfig) -> np.ndarray:
    """Inverse of :func:`quantize`: integers ``0..scale`` back to [0, 1]."""
    return np.asarray(levels, dtype=float) / cfg.scale


def density_to_emitted(levels: np.ndarray, cfg: TargetConfig) -> np.ndarray:
    """Map density levels to the *emitted* per-bin levels per ``cfg.encoding``.

    ``"density"`` emits the PDF unchanged. ``"cdf"`` emits the cumulative density
    rescaled to ``0..scale`` -- a monotonic thermometer ramp where every bin is
    informative (so plain token-CE becomes sensitive to peak *location*).
    """
    levels = np.asarray(levels, dtype=int)
    if getattr(cfg, "encoding", "density") != "cdf":
        return levels
    cum = np.cumsum(levels)
    total = int(cum[-1]) if cum.size and cum[-1] > 0 else 1
    return np.rint(cum.astype(float) / total * cfg.scale).astype(int)


def emitted_to_density(emitted: np.ndarray, cfg: TargetConfig) -> np.ndarray:
    """Inverse of :func:`density_to_emitted`: recover a (non-negative) PDF.

    For ``"cdf"`` this first-differences the monotonic ramp; the result preserves
    the peak location (argmax) even though the rescale+round is not exactly
    invertible.
    """
    emitted = np.asarray(emitted, dtype=int)
    if getattr(cfg, "encoding", "density") != "cdf":
        return emitted
    diff = np.diff(emitted, prepend=0)
    return np.clip(diff, 0, None)


def vector_to_tokens(levels: np.ndarray, cfg: TargetConfig) -> list[str]:
    """Render density levels as fixed-width tokens, applying ``cfg.encoding``."""
    emitted = np.asarray(density_to_emitted(levels, cfg), dtype=int)
    if emitted.min(initial=0) < 0 or emitted.max(initial=0) > cfg.scale:
        raise ValueError(f"emitted levels out of range [0, {cfg.scale}]")
    width = cfg.token_width
    return [str(int(v)).zfill(width) for v in emitted]


def tokens_to_vector(tokens: list[str], cfg: TargetConfig) -> np.ndarray:
    """Parse fixed-width tokens back to density levels (raises on malformed).

    Decodes ``cfg.encoding`` so callers always receive a PDF regardless of how it
    was serialized.
    """
    emitted = []
    for tok in tokens:
        if len(tok) != cfg.token_width or not tok.isdigit():
            raise ValueError(f"malformed token: {tok!r}")
        v = int(tok)
        if v > cfg.scale:
            raise ValueError(f"token {tok!r} exceeds scale {cfg.scale}")
        emitted.append(v)
    return emitted_to_density(np.asarray(emitted, dtype=int), cfg)
