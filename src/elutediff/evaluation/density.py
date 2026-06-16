"""Density-quality metrics: JS divergence, earth-mover distance, CRPS, window prob.

Measures distributional target quality beyond the argmax (proposal Section 9).
Inputs are intensity vectors over the shared time grid; they are normalized to
probability mass internally where a distribution is required.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _as_prob(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    v = np.clip(v, 0.0, None)
    total = v.sum()
    if total <= 0:
        return np.full_like(v, 1.0 / v.size)
    return v / total


def kl_divergence(p, q) -> float:
    """KL(p || q) over the bin axis (nats)."""
    p = _as_prob(p)
    q = _as_prob(q)
    return float(np.sum(p * np.log((p + _EPS) / (q + _EPS))))


def js_divergence(p, q) -> float:
    """Jensen-Shannon divergence (symmetric, bounded by ln 2)."""
    p = _as_prob(p)
    q = _as_prob(q)
    m = 0.5 * (p + q)
    return float(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m))


def earth_mover(p, q, grid=None) -> float:
    """1-D earth-mover (Wasserstein-1) distance between two densities.

    With a ``grid`` (bin centers, e.g. seconds) the distance is in those units;
    otherwise it is in bin indices.
    """
    p = _as_prob(p)
    q = _as_prob(q)
    cdf_diff = np.cumsum(p - q)
    if grid is None:
        return float(np.sum(np.abs(cdf_diff)))
    widths = np.diff(np.asarray(grid, dtype=float))
    widths = np.append(widths, widths[-1] if widths.size else 1.0)
    return float(np.sum(np.abs(cdf_diff) * widths))


def crps_1d(forecast_density, true_value, grid) -> float:
    """Continuous ranked probability score for a 1-D predictive density.

    ``forecast_density`` is an intensity vector over ``grid`` (bin centers);
    ``true_value`` is the observed RT. Lower is better.
    """
    p = _as_prob(forecast_density)
    grid = np.asarray(grid, dtype=float)
    cdf = np.cumsum(p)
    heaviside = (grid >= true_value).astype(float)
    widths = np.diff(grid)
    widths = np.append(widths, widths[-1] if widths.size else 1.0)
    return float(np.sum((cdf - heaviside) ** 2 * widths))


def window_probability(density, grid, center, half_width) -> float:
    """Integrated probability mass within +/- ``half_width`` of ``center`` (seconds).

    This is the quantity that downstream metabolite annotation actually uses:
    probability mass over an RT tolerance window (proposal Sections 9-10).
    """
    p = _as_prob(density)
    grid = np.asarray(grid, dtype=float)
    in_window = (grid >= center - half_width) & (grid <= center + half_width)
    return float(p[in_window].sum())
