import numpy as np
import pytest

from elutediff.config import ConditioningConfig
from elutediff.data.graph_features import laplacian_eigenvectors, serialize_lappe


def test_eigenvector_shape_and_padding():
    # Ethanol has 3 heavy atoms; ask for more components than available.
    vecs = laplacian_eigenvectors("CCO", k=8)
    assert vecs.shape == (3, 8)
    # Only n_atoms - 1 = 2 non-trivial components exist; the rest are zero-padded.
    assert np.allclose(vecs[:, 2:], 0.0)
    assert not np.allclose(vecs[:, :2], 0.0)


def test_sign_canonicalization_is_deterministic():
    a = laplacian_eigenvectors("c1ccccc1O", k=4)
    b = laplacian_eigenvectors("Oc1ccccc1", k=4)  # same molecule, different SMILES order
    # Canonical SMILES + canonical signs => identical encodings up to atom order;
    # at minimum, repeated calls on the same input are identical.
    c = laplacian_eigenvectors("c1ccccc1O", k=4)
    assert np.array_equal(a, c)
    # Largest-magnitude entry of each column is non-negative by convention.
    for j in range(a.shape[1]):
        col = a[:, j]
        assert col[np.argmax(np.abs(col))] >= 0
    assert b.shape == a.shape


def test_serialize_lappe_format():
    cfg = ConditioningConfig(level=4, lappe_k=4, lappe_round=2)
    text = serialize_lappe("CCO", cfg, training=False)
    lines = text.splitlines()
    assert len(lines) == 3  # one line per atom
    idx, *vals = lines[0].split()
    assert idx == "0"
    assert len(vals) == 4
    # Two decimal places.
    assert all(len(v.split(".")[1]) == 2 for v in vals)


def test_sign_flip_augmentation_changes_signs_only():
    cfg = ConditioningConfig(level=4, lappe_k=4, lappe_round=3, lappe_sign_flip=True)
    base = laplacian_eigenvectors("c1ccccc1O", cfg.lappe_k)
    np.random.seed(0)
    flipped_text = serialize_lappe("c1ccccc1O", cfg, training=True)
    flipped = np.array([[float(x) for x in line.split()[1:]] for line in flipped_text.splitlines()])
    # Magnitudes are preserved; only column signs may differ.
    assert np.allclose(np.abs(flipped), np.round(np.abs(base), cfg.lappe_round))


def test_invalid_smiles_raises():
    with pytest.raises(ValueError):
        laplacian_eigenvectors("not_a_molecule", k=4)
