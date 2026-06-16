"""Gaussian RT-density target construction (proposal Section 3).

A scalar retention time ``r_i`` becomes a fixed-length 1-D density over a time
grid::

    g_i(t_j) = exp(-0.5 * ((t_j - r_i) / sigma) ** 2)

The vector is max-normalized to 1.0; quantization to integer tokens happens in
:mod:`elutediff.targets.quantize`.
"""

from __future__ import annotations

import numpy as np

from elutediff.config import TargetConfig


def time_grid(cfg: TargetConfig) -> np.ndarray:
    """Return the bin-center time grid (seconds) implied by ``cfg``.

    Bins are centered at ``rt_min + (j + 0.5) * bin_width`` so that a peak at the
    middle of a bin maps cleanly to that bin's center.
    """
    n = cfg.n_bins
    edges = cfg.rt_min + np.arange(n + 1) * cfg.bin_width
    return 0.5 * (edges[:-1] + edges[1:])


def gaussian_density(rt: float, cfg: TargetConfig, normalize: bool = True) -> np.ndarray:
    """Build a Gaussian RT-density vector centered at ``rt`` (seconds).

    Args:
        rt: observed retention time in seconds.
        cfg: target configuration (grid range, bin width, sigma).
        normalize: if True, scale so the peak equals 1.0.

    Returns:
        Float array of length ``cfg.n_bins`` in [0, 1].
    """
    grid = time_grid(cfg)
    z = (grid - rt) / cfg.sigma
    g = np.exp(-0.5 * z * z)
    if normalize:
        peak = g.max()
        if peak > 0:
            g = g / peak
    return g


def clipped_fraction(rt_values: np.ndarray, cfg: TargetConfig) -> float:
    """Fraction of RTs falling outside the grid range (report this, Section 3)."""
    rt_values = np.asarray(rt_values, dtype=float)
    if rt_values.size == 0:
        return 0.0
    outside = (rt_values < cfg.rt_min) | (rt_values > cfg.rt_max)
    return float(outside.mean())
