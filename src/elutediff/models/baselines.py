"""Scalar RT baselines -- the *known bar* (proposal Section 8).

These are required controls, not optional comparisons:
  B1  ECFP + XGBoost / random forest    -> scalar RT
  B2  ECFP / descriptor MLP             -> scalar RT
  B3  sparse graph transformer / GNN    -> scalar RT (see models.gnn)

plus the cheap uncertainty baselines (B9) the diffusion sampler must beat:
deep ensembles, quantile regression, and split-conformal intervals.

B1/B2 and the uncertainty helpers use scikit-learn / XGBoost (the ``baselines``
extra) and run on CPU. The GNN (B3) lives in :mod:`elutediff.models.gnn` and
needs the ``graph`` extra (torch + torch-geometric).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from elutediff.data.molecules import compute_descriptors, ecfp_fingerprint


class RTRegressor(Protocol):
    """Common interface so baselines and the diffusion decoder are comparable."""

    def fit(self, X, y) -> "RTRegressor": ...
    def predict(self, X) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
# Featurization
# --------------------------------------------------------------------------- #
def featurize_ecfp(smiles: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Stack ECFP (Morgan) fingerprints into an ``(N, n_bits)`` float matrix."""
    return np.vstack([ecfp_fingerprint(s, radius=radius, n_bits=n_bits) for s in smiles]).astype(
        np.float32
    )


def featurize_descriptors(smiles: list[str], names: list[str]) -> np.ndarray:
    """Stack RDKit descriptors into an ``(N, D)`` float matrix."""
    rows = [list(compute_descriptors(s, names).values()) for s in smiles]
    return np.asarray(rows, dtype=np.float32)


def featurize(smiles: list[str], mode: str, *, descriptors=None, radius=2, n_bits=2048) -> np.ndarray:
    """Featurize by ``mode``: ``ecfp``, ``descriptors``, or ``ecfp+descriptors``."""
    if mode == "ecfp":
        return featurize_ecfp(smiles, radius, n_bits)
    if mode == "descriptors":
        return featurize_descriptors(smiles, descriptors or [])
    if mode == "ecfp+descriptors":
        return np.hstack(
            [featurize_ecfp(smiles, radius, n_bits), featurize_descriptors(smiles, descriptors or [])]
        )
    raise ValueError(f"unknown featurization mode: {mode!r}")


# --------------------------------------------------------------------------- #
# B1 / B2 point regressors
# --------------------------------------------------------------------------- #
def ecfp_random_forest(n_estimators: int = 500, n_jobs: int = -1, random_state: int = 0, **kw):
    """B1: random-forest regressor on scalar RT (trees need no input scaling)."""
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state, **kw
    )


def ecfp_xgboost(n_estimators: int = 600, max_depth: int = 6, learning_rate: float = 0.05,
                 subsample: float = 0.8, random_state: int = 0, **kw):
    """B1: XGBoost regressor on scalar RT."""
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=subsample, random_state=random_state, n_jobs=-1, **kw
    )


def descriptor_mlp(hidden=(256, 128), alpha: float = 1e-3, max_iter: int = 500,
                   random_state: int = 0, **kw):
    """B2: standardized MLP regressor (works for descriptor or ECFP features)."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(with_mean=True),
        MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, max_iter=max_iter,
                     random_state=random_state, **kw),
    )


_REGISTRY = {
    "rf": ecfp_random_forest,
    "xgb": ecfp_xgboost,
    "mlp": descriptor_mlp,
}


def build_baseline(name: str, **kwargs) -> RTRegressor:
    """Construct a baseline regressor by short name (``rf``/``xgb``/``mlp``)."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown baseline {name!r}; choose from {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


# --------------------------------------------------------------------------- #
# B9 uncertainty baselines
# --------------------------------------------------------------------------- #
class DeepEnsemble:
    """Variance-from-disagreement intervals across ``k`` differently-seeded models."""

    def __init__(self, factory=ecfp_random_forest, k: int = 5, **kwargs):
        self.models = [factory(random_state=s, **kwargs) for s in range(k)]

    def fit(self, X, y) -> "DeepEnsemble":
        for m in self.models:
            m.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return np.mean([m.predict(X) for m in self.models], axis=0)

    def predict_interval(self, X, level: float = 0.9):
        """Gaussian interval from the ensemble mean +/- z * std."""
        from scipy.stats import norm

        preds = np.stack([m.predict(X) for m in self.models])
        mean, std = preds.mean(0), preds.std(0)
        z = norm.ppf(0.5 + level / 2.0)
        return mean - z * std, mean + z * std


def quantile_regressors(lower: float = 0.05, upper: float = 0.95, **kw):
    """B9 quantile regression: return ``(lo_model, hi_model)`` GBMs."""
    from sklearn.ensemble import GradientBoostingRegressor

    lo = GradientBoostingRegressor(loss="quantile", alpha=lower, **kw)
    hi = GradientBoostingRegressor(loss="quantile", alpha=upper, **kw)
    return lo, hi


@dataclass
class ConformalInterval:
    """Split-conformal intervals around any point regressor (distribution-free).

    Fit the base model on the training fold, then :meth:`calibrate` on a held-out
    calibration fold to size a symmetric interval with finite-sample coverage.
    """

    model: RTRegressor
    level: float = 0.9
    _q: float = 0.0

    def calibrate(self, X_cal, y_cal) -> "ConformalInterval":
        residuals = np.abs(np.asarray(y_cal) - self.model.predict(X_cal))
        n = len(residuals)
        # Finite-sample-adjusted quantile of the calibration residuals.
        rank = min(n, int(np.ceil((n + 1) * self.level)))
        self._q = float(np.sort(residuals)[rank - 1])
        return self

    def predict_interval(self, X):
        center = self.model.predict(X)
        return center - self._q, center + self._q
