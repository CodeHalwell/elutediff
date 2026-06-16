"""Point RT metrics: MAE, median AE, RMSE, R2, tolerance-window hit rates.

Keeps the result grounded against conventional RT prediction. These apply to any
scalar RT estimate -- baseline regressors or RT decoded from a density vector.
"""

from __future__ import annotations

import numpy as np


def point_rt_metrics(y_true, y_pred) -> dict[str, float]:
    """Standard point-regression metrics in seconds."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    err = yp - yt
    abs_err = np.abs(err)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return {
        "mae": float(abs_err.mean()),
        "median_ae": float(np.median(abs_err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def tolerance_hit_rate(y_true, y_pred, tolerances_s) -> dict[str, float]:
    """Fraction of predictions within +/- each tolerance (seconds)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(yp - yt)
    return {f"within_{int(t)}s": float((abs_err <= t).mean()) for t in tolerances_s}
