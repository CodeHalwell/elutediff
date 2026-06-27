"""Typed configuration for the RT-density pipeline.

Configs are plain dataclasses with sane defaults drawn from the proposal
(``docs/density-first-revision.md``). They can be loaded from / merged with the
YAML files under ``configs/`` via :func:`load_config`, so experiments are
reproducible and ablations are a one-line override.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class TargetConfig:
    """RT-density target construction (Section 3 of the proposal).

    ``bin_width`` and ``sigma`` are *not* independent: to make the density
    meaningfully different from a center-bin label, the Gaussian must span at
    least ~2-3 bins. With 10 s bins the first setting is sigma 20-30 s.
    """

    rt_min: float = 0.0          # seconds; lower edge of the time grid
    rt_max: float = 1200.0       # seconds; 99-99.5th percentile in practice
    bin_width: float = 10.0      # seconds per bin (try 5 s if token budget allows)
    sigma: float = 20.0          # Gaussian width in seconds (>= 2-3 bins)
    # One token per bin is a hard constraint of the block-diffusion canvas: the
    # Gemma tokenizer splits each digit into its own token, so a multi-digit
    # level (e.g. "037" -> 3 tokens) blows the 256-token canvas at 120 bins.
    # Single-digit levels (0..9) keep each bin to one token so the target fits.
    scale: int = 9               # quantization levels: integers 0..scale (single digit)
    token_width: int = 1         # single-digit tokens ("0".."9"), one token per bin

    @property
    def n_bins(self) -> int:
        """Number of equal-width bins covering [rt_min, rt_max)."""
        span = self.rt_max - self.rt_min
        return int(round(span / self.bin_width))


@dataclass
class NoiseConfig:
    """Optional weak baseline augmentation (stress test only, never a claim)."""

    enabled: bool = False
    floor: float = 0.0           # constant baseline added before normalization
    drift: float = 0.0           # linear baseline slope across the axis
    spike_prob: float = 0.0      # per-bin probability of a sparse spike
    spike_scale: float = 0.0     # max amplitude of injected spikes
    seed: int | None = None


@dataclass
class ConditioningConfig:
    """Molecular input representation progression (Section 6).

    ``level`` selects how much structure is serialized into the prompt:
      1 = SMILES only, 2 = + descriptors, 3 = + atom/bond table,
      4 = + Laplacian eigenvectors (LapPE), 5 = + graph-transformer embedding.
    """

    level: int = 1
    descriptors: list[str] = field(
        default_factory=lambda: ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds"]
    )
    lappe_k: int = 8             # number of Laplacian eigenvectors
    lappe_round: int = 2         # decimal places for LapPE values
    lappe_sign_flip: bool = True  # random sign flips during training


@dataclass
class ModelConfig:
    """DiffusionGemma + Unsloth/LoRA settings (Section 7)."""

    model_name: str = "unsloth/diffusiongemma-26B-A4B-it"
    load_in_4bit: bool = False   # 4bit cannot shrink the ~46GB MoE experts
    canvas_length: int = 256     # generation canvas (a first-order constraint)
    lora_r: int = 64
    lora_alpha: int = 128        # alpha = 2 * r
    use_gradient_checkpointing: bool = False


@dataclass
class TrainConfig:
    """Block-diffusion fine-tuning loop (Sections 7-8)."""

    steps: int = 500             # full run in the reference report: 4000
    grad_accum: int = 4
    lr: float = 1e-4
    t_lo: float = 0.1            # lower bound of the corruption noise level
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    warmup_pct: float = 0.03
    seed: int = 0
    output_dir: str = "diffusiongemma_lora"


@dataclass
class EvalConfig:
    """Evaluation / sampling settings (Section 9)."""

    denoising_steps: list[int] = field(default_factory=lambda: [1, 16, 64])
    rt_tolerances_s: list[float] = field(default_factory=lambda: [15.0, 30.0, 60.0])
    coverage_levels: list[float] = field(default_factory=lambda: [0.5, 0.8, 0.9])
    n_eval: int = 200
    n_samples: int = 8           # diffusion samples per molecule for uncertainty


@dataclass
class SplitConfig:
    """Reproducible random / scaffold / Tanimoto-cluster splits (Section 8)."""

    strategy: str = "scaffold"   # one of: random, scaffold, cluster
    val_frac: float = 0.1
    test_frac: float = 0.1
    cluster_cutoff: float = 0.6  # Tanimoto cutoff for cluster splits
    seed: int = 0


@dataclass
class Config:
    """Top-level experiment config."""

    experiment: str = "b6_clean_vector"
    metlin_path: str = "data/raw/metlin_smrt"
    target: TargetConfig = field(default_factory=TargetConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    split: SplitConfig = field(default_factory=SplitConfig)


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a (possibly nested) dataclass from a plain dict."""
    if not is_dataclass(cls):
        return data
    kwargs: dict[str, Any] = {}
    # get_type_hints resolves the stringized annotations from
    # ``from __future__ import annotations`` back into real types.
    type_by_name = get_type_hints(cls)
    for key, value in data.items():
        if key not in type_by_name:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        field_type = type_by_name[key]
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[key] = _from_dict(field_type, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load a :class:`Config` from a YAML file, falling back to defaults."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return _from_dict(Config, data)


def to_dict(config: Config) -> dict[str, Any]:
    """Serialize a config back to a plain dict (e.g. to log with a run)."""
    return asdict(config)
