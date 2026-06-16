"""Weak baseline augmentation for RT-density vectors (proposal Section 3, B7).

This is *stress-test augmentation only*. It must never be presented as a learned
chromatographic signal -- it exists to probe robustness of structured generation.
Applied to the normalized density *before* quantization.
"""

from __future__ import annotations

import numpy as np

from elutediff.config import NoiseConfig


def apply_noise(density: np.ndarray, cfg: NoiseConfig) -> np.ndarray:
    """Add optional floor / linear drift / sparse spikes, then re-normalize.

    Returns a new array; the input is not modified. If ``cfg.enabled`` is False
    the (copied) input is returned unchanged -- callers should pass an already
    max-normalized density, as :func:`elutediff.targets.density.gaussian_density`
    produces.
    """
    density = np.asarray(density, dtype=float).copy()
    if not cfg.enabled:
        return density

    rng = np.random.default_rng(cfg.seed)
    n = density.size

    if cfg.floor:
        density = density + cfg.floor
    if cfg.drift:
        density = density + cfg.drift * np.linspace(0.0, 1.0, n)
    if cfg.spike_prob and cfg.spike_scale:
        mask = rng.random(n) < cfg.spike_prob
        density = density + mask * rng.uniform(0.0, cfg.spike_scale, size=n)

    density = np.clip(density, 0.0, None)
    peak = density.max()
    if peak > 0:
        density = density / peak
    return density
