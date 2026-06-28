import numpy as np

from elutediff.config import NoiseConfig, TargetConfig
from elutediff.targets.density import clipped_fraction, gaussian_density, time_grid
from elutediff.targets.noise import apply_noise
from elutediff.targets.quantize import (
    dequantize,
    quantize,
    tokens_to_vector,
    vector_to_tokens,
)


def test_time_grid_shape_and_centers():
    cfg = TargetConfig(rt_min=0.0, rt_max=1200.0, bin_width=10.0)
    grid = time_grid(cfg)
    assert grid.shape == (cfg.n_bins,) == (120,)
    assert np.isclose(grid[0], 5.0)      # first bin center
    assert np.isclose(grid[-1], 1195.0)  # last bin center


def test_gaussian_peaks_at_center_and_normalized():
    cfg = TargetConfig(bin_width=10.0, sigma=20.0)
    g = gaussian_density(600.0, cfg)
    assert np.isclose(g.max(), 1.0)
    grid = time_grid(cfg)
    assert abs(grid[g.argmax()] - 600.0) <= cfg.bin_width


def test_gaussian_spans_multiple_bins():
    # With sigma >= 2-3 bins the peak should be resolved, not a single spike.
    cfg = TargetConfig(bin_width=10.0, sigma=20.0)
    g = gaussian_density(600.0, cfg)
    assert int((g > 0.1).sum()) >= 3


def test_quantize_roundtrip_levels_and_tokens():
    # density encoding: exact level<->token round-trip (CDF would be lossy)
    cfg = TargetConfig(bin_width=10.0, sigma=20.0, scale=100, token_width=3,
                       encoding="density")
    g = gaussian_density(600.0, cfg)
    levels = quantize(g, cfg)
    assert levels.max() == 100 and levels.min() >= 0
    tokens = vector_to_tokens(levels, cfg)
    assert all(len(t) == 3 and t.isdigit() for t in tokens)
    assert np.array_equal(tokens_to_vector(tokens, cfg), levels)


def test_dequantize_inverse():
    cfg = TargetConfig(scale=100, token_width=3)
    levels = np.array([0, 50, 100])
    assert np.allclose(dequantize(levels, cfg), [0.0, 0.5, 1.0])


def test_clipped_fraction():
    cfg = TargetConfig(rt_min=0.0, rt_max=1200.0)
    rts = np.array([100.0, 600.0, 1300.0, -5.0])
    assert np.isclose(clipped_fraction(rts, cfg), 0.5)


def test_noise_disabled_is_passthrough():
    cfg = TargetConfig()
    g = gaussian_density(600.0, cfg)
    out = apply_noise(g, NoiseConfig(enabled=False))
    assert np.allclose(out, g)


def test_noise_renormalizes_to_peak_one():
    cfg = TargetConfig()
    g = gaussian_density(600.0, cfg)
    out = apply_noise(g, NoiseConfig(enabled=True, floor=0.05, spike_prob=0.1,
                                     spike_scale=0.2, seed=0))
    assert np.isclose(out.max(), 1.0)
    assert out.min() >= 0.0


def test_noise_per_molecule_seed_is_independent_and_reproducible():
    cfg = TargetConfig()
    nc = NoiseConfig(enabled=True, floor=0.02, spike_prob=0.2, spike_scale=0.1, seed=0)
    g = gaussian_density(600.0, cfg)
    a = apply_noise(g, nc, seed=(nc.seed, 1))
    b = apply_noise(g, nc, seed=(nc.seed, 2))
    assert not np.allclose(a, b)                                   # different rows differ
    assert np.allclose(a, apply_noise(g, nc, seed=(nc.seed, 1)))   # but reproducible
