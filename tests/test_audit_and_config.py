from pathlib import Path

import pytest

from elutediff.audit import audit_target, sweep_bin_widths
from elutediff.config import Config, TargetConfig, load_config
from elutediff.serialization.prompts import target_string
from elutediff.targets.density import gaussian_density
from elutediff.targets.quantize import quantize


def test_default_target_fits_canvas_under_char_upper_bound():
    """Regression for the canvas overflow. The Gemma tokenizer emits at most one
    token per character (each digit and each separating space), so
    ``len(target_string) + 1`` (eos) is a conservative upper bound on the real
    token length -- enough to guarantee the canvas fits without loading the
    tokenizer. The default single-digit config must stay under the 256 canvas;
    the old 3-digit default produced ~479 characters and silently dropped every
    row.
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
    # Single-digit levels: 120 digits + 119 spaces + eos = 240 (the realistic
    # per-character cost), comfortably under the 256-token canvas.
    assert res.est_target_tokens == 240
    assert res.fits_canvas


def test_five_second_bins_overflow_canvas():
    # 1200s / 5s = 240 bins. Even single-digit, the spaced target costs
    # 240 digits + 239 spaces + eos = 480 tokens -> overflows 256. The old
    # one-token-per-bin estimate wrongly passed this; the realistic one catches it.
    cfg = TargetConfig(rt_min=0.0, rt_max=1200.0, bin_width=5.0)
    res = audit_target(cfg, canvas_length=256)
    assert res.n_bins == 240
    assert res.est_target_tokens == 480
    assert not res.fits_canvas


def test_three_digit_levels_overflow_is_caught():
    # Regression for the original bug: 3-digit levels at 120 bins are ~480
    # tokens, far over the canvas. The audit must now flag this as OVER.
    cfg = TargetConfig(rt_min=0.0, rt_max=1200.0, bin_width=10.0,
                       scale=100, token_width=3)
    res = audit_target(cfg, canvas_length=256)
    assert res.n_bins == 120
    assert res.est_target_tokens > 256
    assert not res.fits_canvas


def test_target_config_validation():
    # A level that overflows its fixed width is the bug this guards against.
    with pytest.raises(ValueError, match="cannot be represented with token_width"):
        TargetConfig(scale=10, token_width=1)
    with pytest.raises(ValueError, match="must be greater than rt_min"):
        TargetConfig(rt_min=100.0, rt_max=50.0)
    with pytest.raises(ValueError, match="bin_width.*must be positive"):
        TargetConfig(bin_width=0.0)
    with pytest.raises(ValueError, match="sigma.*must be positive"):
        TargetConfig(sigma=-5.0)
    with pytest.raises(ValueError, match="token_width.*must be at least 1"):
        TargetConfig(token_width=0)


def test_default_encoding_is_cdf():
    # The ablation showed the sparse "density" target collapses; CDF is the
    # validated default that lets the model actually learn RT.
    assert TargetConfig().encoding == "cdf"
    assert Config().target.encoding == "cdf"


def test_encoding_and_peak_loss_validation():
    from elutediff.config import TrainConfig

    # Arm 1: target encoding
    TargetConfig(encoding="cdf")  # ok
    with pytest.raises(ValueError, match="encoding must be"):
        TargetConfig(encoding="bogus")
    # Arm 2: peak-aware loss
    TrainConfig(peak_loss="emd", peak_lambda=0.5)  # ok
    with pytest.raises(ValueError, match="peak_loss must be"):
        TrainConfig(peak_loss="bogus")
    with pytest.raises(ValueError, match="peak_lambda"):
        TrainConfig(peak_lambda=-1.0)


def test_data_config_defaults_and_validation():
    from elutediff.config import DataConfig

    # Default is the 5-minute void cutoff; wired into the top-level Config.
    assert DataConfig().min_retention_s == 300.0
    assert Config().data.min_retention_s == 300.0
    with pytest.raises(ValueError, match="min_retention_s"):
        DataConfig(min_retention_s=-1.0)


def test_load_config_accepts_data_block(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("data:\n  min_retention_s: 0.0\n")
    cfg = load_config(p)
    assert cfg.data.min_retention_s == 0.0


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
