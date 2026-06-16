"""Scalar / density RT baselines -- the *known bar* (proposal Section 8).

These are required controls, not optional comparisons:
  B1  ECFP + XGBoost / random forest    -> scalar RT
  B2  ECFP / descriptor MLP             -> scalar RT
  B3  sparse graph transformer / GNN    -> scalar RT or density (high bar)
plus uncertainty baselines (B9): MC dropout, ensembles, quantile regression,
conformal intervals.

NOTE: scaffold. Implement in roadmap step 4. Requires the ``baselines`` /
``graph`` extras.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class RTRegressor(Protocol):
    """Common interface so baselines and the diffusion decoder are comparable."""

    def fit(self, X, y) -> "RTRegressor": ...
    def predict(self, X) -> np.ndarray: ...


def ecfp_xgboost(**kwargs):
    """B1: ECFP fingerprints -> XGBoost regressor on scalar RT."""
    raise NotImplementedError("ecfp_xgboost: classical baseline (roadmap step 4).")


def ecfp_random_forest(**kwargs):
    """B1: ECFP fingerprints -> random-forest regressor on scalar RT."""
    raise NotImplementedError("ecfp_random_forest: classical baseline (roadmap step 4).")


def descriptor_mlp(**kwargs):
    """B2: descriptor/ECFP MLP on scalar RT."""
    raise NotImplementedError("descriptor_mlp: neural baseline (roadmap step 4).")


def graph_transformer(**kwargs):
    """B3: sparse graph transformer / GNN -- the high bar for point MAE."""
    raise NotImplementedError("graph_transformer: GNN baseline (roadmap step 4).")
