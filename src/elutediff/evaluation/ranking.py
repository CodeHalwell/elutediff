"""Annotation-utility metrics: candidate ranking and filtering (proposal Section 9).

Connects RT density to metabolite annotation rather than pure regression. Given a
query with a true RT and a candidate set scored by the model (e.g. by integrated
window probability around each candidate's expected RT, or by RT agreement),
measure whether the correct candidate ranks highly.
"""

from __future__ import annotations

import numpy as np


def top_k_accuracy(ranks, k: int) -> float:
    """Fraction of queries whose correct candidate has 1-based rank <= ``k``."""
    ranks = np.asarray(ranks, dtype=int)
    return float((ranks <= k).mean())


def mean_reciprocal_rank(ranks) -> float:
    """Mean of 1 / rank over queries (1-based ranks)."""
    ranks = np.asarray(ranks, dtype=float)
    return float(np.mean(1.0 / ranks))


def retained_fraction_at_recall(scores, is_correct, recall: float) -> float:
    """Fraction of candidates retained while keeping ``recall`` of true positives.

    Lower is better: a good RT filter discards more decoys at a fixed recall.
    ``scores`` rank candidates (higher = more plausible); ``is_correct`` flags the
    true positives.
    """
    scores = np.asarray(scores, dtype=float)
    is_correct = np.asarray(is_correct, dtype=bool)
    n_pos = int(is_correct.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores)
    correct_sorted = is_correct[order]
    need = int(np.ceil(recall * n_pos))
    cumulative_pos = np.cumsum(correct_sorted)
    idx = int(np.searchsorted(cumulative_pos, need))
    return (idx + 1) / scores.size
