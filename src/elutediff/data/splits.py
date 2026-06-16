"""Reproducible data splits: random, Bemis-Murcko scaffold, Tanimoto cluster.

Splits are a scientific control, not a convenience: the headline point metric is
*scaffold-split* MAE, with random-split and Tanimoto-cluster MAE reported
alongside it, to show the model learned chemistry rather than format/memorization
(proposal Sections 8, 12).

NOTE: scaffold. Implement in roadmap step 1.
"""

from __future__ import annotations

from dataclasses import dataclass

from elutediff.config import SplitConfig


@dataclass
class Split:
    train_idx: list[int]
    val_idx: list[int]
    test_idx: list[int]


def make_split(smiles: list[str], cfg: SplitConfig) -> Split:
    """Build a train/val/test split according to ``cfg.strategy``.

    - ``random``: shuffled index split.
    - ``scaffold``: group by Bemis-Murcko scaffold; whole scaffolds go to one fold.
    - ``cluster``: Taylor-Butina Tanimoto clustering at ``cfg.cluster_cutoff``;
      whole clusters go to one fold.
    """
    raise NotImplementedError("make_split: random/scaffold/cluster splits (roadmap step 1).")
