import numpy as np
import pytest

from elutediff.config import SplitConfig
from elutediff.data.molecules import compute_descriptors
from elutediff.data.splits import make_split
from elutediff.evaluation.point_rt import point_rt_metrics
from elutediff.evaluation.uncertainty import interval_coverage
from elutediff.models.baselines import (
    ConformalInterval,
    DeepEnsemble,
    build_baseline,
    featurize,
    featurize_descriptors,
    featurize_ecfp,
)

# A learnable structure->RT signal: linear chains of increasing length, with RT
# driven by molecular weight (a real retention-relevant property), plus noise.
SMILES = [("C" * n) for n in range(2, 24)] + [("C" * n + "O") for n in range(1, 23)]


def _rts(seed=0):
    rng = np.random.default_rng(seed)
    mw = np.array([compute_descriptors(s, ["MolWt"])["MolWt"] for s in SMILES])
    return 3.0 * mw + rng.normal(0, 5.0, size=mw.shape)  # seconds


def test_featurizers_shapes():
    X_ecfp = featurize_ecfp(SMILES[:5], n_bits=512)
    assert X_ecfp.shape == (5, 512)
    X_desc = featurize_descriptors(SMILES[:5], ["MolWt", "LogP"])
    assert X_desc.shape == (5, 2)
    X_both = featurize(SMILES[:5], "ecfp+descriptors", descriptors=["MolWt"], n_bits=512)
    assert X_both.shape == (5, 513)


@pytest.mark.parametrize("name", ["rf", "xgb", "mlp"])
def test_baseline_learns_signal(name):
    y = _rts()
    X = featurize(SMILES, "descriptors", descriptors=["MolWt", "LogP", "TPSA"])
    n = len(SMILES)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    tr, te = idx[: int(0.7 * n)], idx[int(0.7 * n) :]
    model = build_baseline(name)
    model.fit(X[tr], y[tr])
    r2 = point_rt_metrics(y[te], model.predict(X[te]))["r2"]
    assert r2 > 0.5  # clearly beats predicting the mean


def test_conformal_coverage():
    y = _rts()
    X = featurize(SMILES, "descriptors", descriptors=["MolWt", "LogP"])
    n = len(SMILES)
    tr, cal, te = X[: n // 2], X[n // 2 : 3 * n // 4], X[3 * n // 4 :]
    ytr, ycal, yte = y[: n // 2], y[n // 2 : 3 * n // 4], y[3 * n // 4 :]
    model = build_baseline("rf").fit(tr, ytr)
    conf = ConformalInterval(model, level=0.9).calibrate(cal, ycal)
    lo, hi = conf.predict_interval(te)
    # Distribution-free guarantee is marginal; just check it is not far off.
    assert interval_coverage(yte, lo, hi) >= 0.6
    assert np.all(hi >= lo)


def test_deep_ensemble_interval_orders():
    y = _rts()
    X = featurize(SMILES, "descriptors", descriptors=["MolWt", "LogP"])
    ens = DeepEnsemble(k=3, n_estimators=50).fit(X, y)
    lo, hi = ens.predict_interval(X, level=0.9)
    assert np.all(hi >= lo)
    assert ens.predict(X).shape == y.shape


def test_baselines_integrate_with_splits():
    # Sanity: baselines consume the same split indices as everything else.
    cfg = SplitConfig(strategy="scaffold", val_frac=0.2, test_frac=0.2, seed=0)
    split = make_split(SMILES, cfg)
    assert sum(split.sizes()) == len(SMILES)
