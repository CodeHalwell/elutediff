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
    toks[0] = "5"  # not fixed-width
    res = parse_rt_vector(" ".join(toks), cfg)
    assert not res.ok


def test_validity_single_peak():
    cfg, levels = _levels()
    rep = validity_report(levels, cfg)
    assert rep["length_ok"] and rep["range_ok"]
    assert rep["single_dominant_peak"]
    assert rep["n_local_maxima"] == 1


def test_decoded_rt_argmax_near_truth():
    cfg, levels = _levels(rt=730.0)
    rt_hat = decoded_rt(levels, cfg, mode="argmax")
    assert abs(rt_hat - 730.0) <= cfg.bin_width
    centroid = decoded_rt(levels, cfg, mode="centroid")
    assert abs(centroid - 730.0) <= 2 * cfg.bin_width


def test_build_prompt_levels():
    cfg = TargetConfig()
    p1 = build_prompt(smiles="CCO", target_cfg=cfg, cond_cfg=ConditioningConfig(level=1))
    assert "smiles=CCO" in p1 and "descriptors" not in p1
    p2 = build_prompt(
        smiles="CCO", target_cfg=cfg, cond_cfg=ConditioningConfig(level=2),
        descriptors={"MolWt": 46.07, "LogP": -0.14},
    )
    assert "descriptors:" in p2 and "MolWt=46.07" in p2
