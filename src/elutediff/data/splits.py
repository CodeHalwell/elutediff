"""Reproducible data splits: random, Bemis-Murcko scaffold, Tanimoto cluster.

Splits are a scientific control, not a convenience: the headline point metric is
*scaffold-split* MAE, with random-split and Tanimoto-cluster MAE reported
alongside it, to show the model learned chemistry rather than format/memorization
(proposal Sections 8, 12).

Group-based splits (scaffold, cluster) keep every member of a group in a single
fold so that no scaffold/cluster leaks across train/val/test.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

from elutediff.config import SplitConfig


@dataclass
class Split:
    train_idx: list[int]
    val_idx: list[int]
    test_idx: list[int]

    def sizes(self) -> tuple[int, int, int]:
        return len(self.train_idx), len(self.val_idx), len(self.test_idx)


def make_split(smiles: list[str], cfg: SplitConfig) -> Split:
    """Build a train/val/test split according to ``cfg.strategy``.

    - ``random``: shuffled index split.
    - ``scaffold``: group by Bemis-Murcko scaffold; whole scaffolds per fold.
    - ``cluster``: Taylor-Butina Tanimoto clustering at ``cfg.cluster_cutoff``;
      whole clusters per fold.
    """
    if cfg.strategy == "random":
        return _random_split(len(smiles), cfg)
    if cfg.strategy == "scaffold":
        return _grouped_split(_scaffold_groups(smiles), len(smiles), cfg)
    if cfg.strategy == "cluster":
        return _grouped_split(_cluster_groups(smiles, cfg.cluster_cutoff), len(smiles), cfg)
    raise ValueError(f"unknown split strategy: {cfg.strategy!r}")


def _random_split(n: int, cfg: SplitConfig) -> Split:
    idx = list(range(n))
    random.Random(cfg.seed).shuffle(idx)
    n_test = int(round(n * cfg.test_frac))
    n_val = int(round(n * cfg.val_frac))
    test = idx[:n_test]
    val = idx[n_test : n_test + n_val]
    train = idx[n_test + n_val :]
    return Split(sorted(train), sorted(val), sorted(test))


def _scaffold_groups(smiles: list[str]) -> list[list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffold = f"__invalid_{i}__"  # unparseable -> its own singleton group
        else:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffold = scaffold or f"__empty_{i}__"
        groups[scaffold].append(i)
    return list(groups.values())


def _cluster_groups(smiles: list[str], cutoff: float) -> list[list[int]]:
    fps = []
    valid_idx = []
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(gen.GetFingerprint(mol))
        valid_idx.append(i)

    groups: list[list[int]] = []
    n = len(fps)
    if n:
        # Condensed lower-triangle distance matrix for Butina.
        dists = []
        for i in range(1, n):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend(1.0 - s for s in sims)
        clusters = Butina.ClusterData(dists, n, 1.0 - cutoff, isDistData=True)
        groups = [[valid_idx[j] for j in cluster] for cluster in clusters]

    # Unparseable molecules become singleton groups so indices are not lost.
    covered = {i for g in groups for i in g}
    for i in range(len(smiles)):
        if i not in covered:
            groups.append([i])
    return groups


def _grouped_split(groups: list[list[int]], n: int, cfg: SplitConfig) -> Split:
    """Greedily assign whole groups to test, then val, then train by target size.

    Groups are shuffled (seeded) and sorted largest-first so big scaffolds don't
    overshoot a small fold; each group fills test until its quota, then val, then
    everything else lands in train.
    """
    rng = random.Random(cfg.seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    n_test_target = int(round(n * cfg.test_frac))
    n_val_target = int(round(n * cfg.val_frac))

    train, val, test = [], [], []
    for group in groups:
        if len(test) + len(group) <= n_test_target:
            test.extend(group)
        elif len(val) + len(group) <= n_val_target:
            val.extend(group)
        else:
            train.extend(group)
    return Split(sorted(train), sorted(val), sorted(test))
