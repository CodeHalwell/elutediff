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


def vector_to_tokens(levels: np.ndarray, cfg: TargetConfig) -> list[str]:
    """Render integer levels as fixed-width zero-padded tokens (e.g. ``"037"``)."""
    levels = np.asarray(levels, dtype=int)
    if levels.min(initial=0) < 0 or levels.max(initial=0) > cfg.scale:
        raise ValueError(f"levels out of range [0, {cfg.scale}]")
    width = cfg.token_width
    return [str(int(v)).zfill(width) for v in levels]


def tokens_to_vector(tokens: list[str], cfg: TargetConfig) -> np.ndarray:
    """Parse fixed-width tokens back to integer levels (raises on malformed)."""
    levels = []
    for tok in tokens:
        if len(tok) != cfg.token_width or not tok.isdigit():
            raise ValueError(f"malformed token: {tok!r}")
        v = int(tok)
        if v > cfg.scale:
            raise ValueError(f"token {tok!r} exceeds scale {cfg.scale}")
        levels.append(v)
    return np.asarray(levels, dtype=int)
