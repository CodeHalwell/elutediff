from pathlib import Path

import numpy as np

from elutediff.audit import audit_target, sweep_bin_widths
from elutediff.config import Config, TargetConfig, load_config
from elutediff.serialization.prompts import target_string
from elutediff.targets.density import gaussian_density
from elutediff.targets.quantize import quantize


def test_default_target_fits_canvas_under_real_tokenization():
    """Regression for the canvas overflow: the Gemma tokenizer emits ~one token
    per character (each digit and each separating space), so the real target
    length is ``len(target_string) + 1`` (eos), *not* the optimistic one-token-
    per-bin estimate. The default single-digit config must fit the 256 canvas;
    the old 3-digit default produced ~479 tokens and silently dropped every row.
    """
    cfg = TargetConfig()  # defaults: 120 bins, single-digit levels
    levels = quantize(gaussian_density(600.0, cfg), cfg)
    text = target_string(levels, cfg)
    assert max(len(t) for t in text.split()) == 1  # one digit per bin
    assert len(text) + 1 <= Config().model.canvas_length


def test_default_target_fits_canvas():
    cfg = TargetConfig(rt_min=0.0, rt_max=1200.0, bin_width=10.0)
    res = audit_target(cfg, canvas_length=256)
    assert res.n_bins == 120
    assert res.est_target_tokens == 121  # bins + eos
    assert res.fits_canvas


def test_five_second_bins_stress_canvas():
    # 1200s / 5s = 240 bins (+eos) = 241 -> still under 256 but tight.
    cfg = TargetConfig(rt_min=0.0, rt_max=1200.0, bin_width=5.0)
    res = audit_target(cfg, canvas_length=256)
    assert res.n_bins == 240
    assert res.fits_canvas
    # A wider range at 5s would overflow:
    over = audit_target(TargetConfig(rt_max=1500.0, bin_width=5.0), canvas_length=256)
    assert not over.fits_canvas


def test_sweep_returns_one_per_width():
    res = sweep_bin_widths(TargetConfig(), bin_widths=(10.0, 5.0))
    assert len(res) == 2


def test_load_config_roundtrip(tmp_path: Path):
    yaml_text = """
experiment: my_exp
target:
  bin_width: 5.0
  sigma: 15.0
conditioning:
  level: 4
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    cfg = load_config(p)
    assert isinstance(cfg, Config)
    assert cfg.experiment == "my_exp"
    assert cfg.target.bin_width == 5.0
    assert cfg.conditioning.level == 4
    # Untouched fields keep defaults:
    assert cfg.model.canvas_length == 256
