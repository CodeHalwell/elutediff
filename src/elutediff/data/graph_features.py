"""Laplacian positional encodings (LapPE) and optional graph-transformer features.

Conditioning levels 4-5 (proposal Section 6). LapPE adds per-atom graph
positional information to the prompt. Sign ambiguity and eigenspace instability
*must* be handled explicitly: eigenvectors are sign-canonicalized (and can be
randomly sign-flipped during training), and values are rounded to keep the
prompt short.

References: Dwivedi et al. (LSPE, 2022); Rampasek et al. (GraphGPS, 2022).

This module needs only RDKit + numpy (the ``chem`` extra).
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from elutediff.config import ConditioningConfig


def _adjacency(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles!r}")
    return Chem.GetAdjacencyMatrix(mol).astype(float)


def _canonicalize_signs(vecs: np.ndarray) -> np.ndarray:
    """Fix each eigenvector's arbitrary sign deterministically.

    Convention: make the entry with the largest magnitude positive (ties broken
    by the first such entry). Removes the +/- ambiguity of eigendecomposition so
    the same molecule always serializes identically.
    """
    vecs = vecs.copy()
    for j in range(vecs.shape[1]):
        col = vecs[:, j]
        idx = int(np.argmax(np.abs(col)))
        if col[idx] < 0:
            vecs[:, j] = -col
    return vecs


def laplacian_eigenvectors(smiles: str, k: int) -> np.ndarray:
    """Return the first ``k`` non-trivial Laplacian eigenvectors, sign-canonicalized.

    Uses the symmetric normalized Laplacian ``L = I - D^-1/2 A D^-1/2``. The
    trivial (smallest-eigenvalue) component is skipped. The result has shape
    ``(n_atoms, k)``; if the molecule has fewer than ``k+1`` atoms the missing
    columns are zero-padded.
    """
    adj = _adjacency(smiles)
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    with np.errstate(divide="ignore"):
        dinv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    lap = np.eye(n) - (dinv_sqrt[:, None] * adj * dinv_sqrt[None, :])

    # Symmetric -> real eigenvalues; eigh returns them ascending.
    eigvals, eigvecs = np.linalg.eigh(lap)
    order = np.argsort(eigvals)
    eigvecs = eigvecs[:, order]

    # Skip the first (trivial) eigenvector; take the next k available.
    available = eigvecs[:, 1 : 1 + k]
    out = np.zeros((n, k), dtype=float)
    out[:, : available.shape[1]] = available
    return _canonicalize_signs(out)


def serialize_lappe(smiles: str, cfg: ConditioningConfig, training: bool = False) -> str:
    """Render per-atom LapPE coordinates to a compact, rounded text block.

    When ``training`` is True, apply random per-eigenvector sign flips
    (``cfg.lappe_sign_flip``) so the model cannot latch onto an arbitrary sign.
    For precomputed datasets keep ``training=False`` (deterministic, canonical
    signs); apply flips on the fly in the training loop instead.
    """
    vecs = laplacian_eigenvectors(smiles, cfg.lappe_k)
    if training and cfg.lappe_sign_flip:
        flips = np.random.choice([-1.0, 1.0], size=vecs.shape[1])
        vecs = vecs * flips[None, :]
    vecs = np.round(vecs, cfg.lappe_round)
    fmt = f"%.{cfg.lappe_round}f"
    lines = [f"{i} " + " ".join(fmt % v for v in row) for i, row in enumerate(vecs)]
    return "\n".join(lines)
