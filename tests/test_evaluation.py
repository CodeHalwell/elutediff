import numpy as np

from elutediff.config import TargetConfig
from elutediff.evaluation.density import (
    crps_1d,
    earth_mover,
    js_divergence,
    kl_divergence,
    window_probability,
)
from elutediff.evaluation.point_rt import point_rt_metrics, tolerance_hit_rate
from elutediff.evaluation.ranking import mean_reciprocal_rank, top_k_accuracy
from elutediff.evaluation.uncertainty import expected_calibration_error, interval_coverage
from elutediff.targets.density import gaussian_density, time_grid


def test_point_rt_perfect():
    y = [100.0, 200.0, 300.0]
    m = point_rt_metrics(y, y)
    assert m["mae"] == 0.0 and m["rmse"] == 0.0
    assert np.isclose(m["r2"], 1.0)


def test_tolerance_hit_rate():
    yt = [100.0, 200.0]
    yp = [110.0, 250.0]
    hits = tolerance_hit_rate(yt, yp, [15.0, 60.0])
    assert hits["within_15s"] == 0.5
    assert hits["within_60s"] == 1.0


def test_js_zero_for_identical():
    p = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    assert js_divergence(p, p) < 1e-9
    assert kl_divergence(p, p) < 1e-9


def test_earth_mover_shift_in_seconds():
    cfg = TargetConfig(bin_width=10.0, sigma=20.0)
    grid = time_grid(cfg)
    a = gaussian_density(600.0, cfg)
    b = gaussian_density(700.0, cfg)
    emd = earth_mover(a, b, grid)
    # EMD between two equal-mass Gaussians ~ distance between their centers.
    assert 80.0 <= emd <= 120.0


def test_crps_lower_when_centered():
    cfg = TargetConfig(bin_width=10.0, sigma=20.0)
    grid = time_grid(cfg)
    good = gaussian_density(600.0, cfg)
    bad = gaussian_density(900.0, cfg)
    assert crps_1d(good, 600.0, grid) < crps_1d(bad, 600.0, grid)


def test_window_probability_mass():
    cfg = TargetConfig(bin_width=10.0, sigma=20.0)
    grid = time_grid(cfg)
    dens = gaussian_density(600.0, cfg)
    near = window_probability(dens, grid, 600.0, 60.0)
    far = window_probability(dens, grid, 200.0, 60.0)
    assert near > 0.9 and far < 0.05


def test_uncertainty_coverage_and_ece():
    yt = np.array([1.0, 2.0, 3.0, 4.0])
    lo = yt - 1.0
    hi = yt + 1.0
    assert interval_coverage(yt, lo, hi) == 1.0
    rng = np.random.default_rng(0)
    samples = yt[:, None] + rng.normal(0, 1.0, size=(4, 500))
    ece = expected_calibration_error(yt, samples, [0.5, 0.9])
    assert 0.0 <= ece <= 1.0


def test_ranking():
    ranks = [1, 2, 5, 1]
    assert top_k_accuracy(ranks, 1) == 0.5
    assert top_k_accuracy(ranks, 5) == 1.0
    assert mean_reciprocal_rank([1, 2]) == 0.75
