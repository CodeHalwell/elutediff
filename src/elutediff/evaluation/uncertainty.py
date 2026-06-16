"""Uncertainty metrics: interval coverage, width, calibration error (B9).

Metric of record (proposal Section 8): 90% interval-coverage error at a
comparable median width; CRPS secondary. Compare diffusion-sample intervals
against MC dropout, ensembles, quantile regression, and conformal baselines --
and report a clean negative if diffusion is not better calibrated.
"""

from __future__ import annotations

import numpy as np


def interval_coverage(y_true, lower, upper) -> float:
    """Fraction of true values inside the predicted [lower, upper] interval."""
    yt = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return float(((yt >= lo) & (yt <= hi)).mean())


def median_interval_width(lower, upper) -> float:
    """Median width of the predicted intervals (for coverage-width tradeoffs)."""
    return float(np.median(np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)))


def coverage_error(y_true, lower, upper, nominal: float) -> float:
    """Signed coverage error: empirical coverage minus the nominal level."""
    return interval_coverage(y_true, lower, upper) - nominal


def expected_calibration_error(y_true, samples, levels) -> float:
    """ECE from per-molecule prediction samples.

    ``samples`` is shape ``(n_molecules, n_samples)``. For each nominal central
    level we form the empirical interval from sample quantiles and measure the
    absolute gap between nominal and realized coverage, averaged over levels.
    """
    yt = np.asarray(y_true, dtype=float)
    s = np.asarray(samples, dtype=float)
    gaps = []
    for level in levels:
        alpha = (1.0 - level) / 2.0
        lo = np.quantile(s, alpha, axis=1)
        hi = np.quantile(s, 1.0 - alpha, axis=1)
        gaps.append(abs(interval_coverage(yt, lo, hi) - level))
    return float(np.mean(gaps)) if gaps else float("nan")
