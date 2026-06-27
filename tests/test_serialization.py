import numpy as np

from elutediff.config import ConditioningConfig, TargetConfig
from elutediff.serialization.parser import (
    decoded_rt,
    parse_rt_vector,
    validity_report,
)
from elutediff.serialization.prompts import build_prompt, format_rt_vector, target_string
from elutediff.targets.density import gaussian_density
from elutediff.targets.quantize import quantize


def _levels(rt=600.0):
    cfg = TargetConfig(bin_width=10.0, sigma=20.0)
    return cfg, quantize(gaussian_density(rt, cfg), cfg)


def test_roundtrip_target_string_parses():
    cfg, levels = _levels()
    text = target_string(levels, cfg)
    res = parse_rt_vector(text, cfg)
    assert res.ok
    assert np.array_equal(res.levels, levels)


def test_parse_strips_rt_vector_wrapper():
    cfg, levels = _levels()
    wrapped = format_rt_vector(levels, cfg)
    res = parse_rt_vector(wrapped, cfg)
    assert res.ok
    assert np.array_equal(res.levels, levels)


def test_parse_rejects_wrong_length():
    cfg, levels = _levels()
    text = target_string(levels[:-5], cfg)  # too few bins
    res = parse_rt_vector(text, cfg)
    assert not res.ok and "expected" in res.reason


def test_parse_rejects_malformed_token_strict():
    cfg, levels = _levels()
    toks = target_string(levels, cfg).split()
    toks[0] = "55"  # wrong width (two digits where the canvas uses one)
    res = parse_rt_vector(" ".join(toks), cfg)
    assert not res.ok


def test_validity_single_peak():
    cfg, levels = _levels()
    rep = validity_report(levels, cfg)
    assert rep["length_ok"] and rep["range_ok"]
    assert rep["single_dominant_peak"]
    assert rep["n_local_maxima"] == 1


def test_flat_vector_is_not_a_peak():
    cfg, _ = _levels()
    flat = np.zeros(cfg.n_bins, dtype=int)
    rep = validity_report(flat, cfg)
    assert rep["n_local_maxima"] == 0
    assert not rep["single_dominant_peak"]
    # A constant non-zero vector is equally degenerate.
    const = np.full(cfg.n_bins, 50, dtype=int)
    assert validity_report(const, cfg)["n_local_maxima"] == 0


def test_monotonic_ramp_has_no_interior_peak():
    cfg, _ = _levels()
    ramp = np.linspace(0, cfg.scale, cfg.n_bins).astype(int)
    assert validity_report(ramp, cfg)["n_local_maxima"] == 0


def test_two_peaks_counted():
    cfg, _ = _levels()
    v = np.zeros(cfg.n_bins, dtype=int)
    v[30] = 100  # two well-separated spikes
    v[80] = 100
    assert validity_report(v, cfg)["n_local_maxima"] == 2


def test_prompt_header_reflects_config():
    from elutediff.serialization.prompts import _header

    cfg = TargetConfig(bin_width=10.0, sigma=20.0, scale=255, token_width=3)
    h = _header(cfg)
    assert "3-digit" in h
    assert "000-255" in h
    assert str(cfg.n_bins) in h


def test_decoded_rt_argmax_near_truth():
    cfg, levels = _levels(rt=730.0)
    rt_hat = decoded_rt(levels, cfg, mode="argmax")
    assert abs(rt_hat - 730.0) <= cfg.bin_width
    centroid = decoded_rt(levels, cfg, mode="centroid")
    assert abs(centroid - 730.0) <= 2 * cfg.bin_width


def test_cdf_encoding_is_monotone_and_preserves_peak():
    from elutediff.targets.quantize import density_to_emitted

    cfg = TargetConfig(bin_width=10.0, sigma=20.0, encoding="cdf")
    levels = quantize(gaussian_density(600.0, cfg), cfg)
    emitted = density_to_emitted(levels, cfg)
    # Thermometer ramp: monotone non-decreasing, within [0, scale].
    assert np.all(np.diff(emitted) >= 0)
    assert emitted.min() >= 0 and emitted.max() <= cfg.scale
    # Same token budget as the density encoding (one digit per bin).
    assert len(target_string(levels, cfg).split()) == cfg.n_bins
    # Round-trip through tokens recovers a PDF whose peak matches (within 1 bin).
    res = parse_rt_vector(target_string(levels, cfg), cfg)
    assert res.ok
    assert abs(int(np.argmax(res.levels)) - int(np.argmax(levels))) <= 1


def test_build_prompt_levels():
    cfg = TargetConfig()
    p1 = build_prompt(smiles="CCO", target_cfg=cfg, cond_cfg=ConditioningConfig(level=1))
    assert "smiles=CCO" in p1 and "descriptors" not in p1
    p2 = build_prompt(
        smiles="CCO", target_cfg=cfg, cond_cfg=ConditioningConfig(level=2),
        descriptors={"MolWt": 46.07, "LogP": -0.14},
    )
    assert "descriptors:" in p2 and "MolWt=46.07" in p2
