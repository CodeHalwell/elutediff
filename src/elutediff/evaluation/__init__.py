"""Evaluation metric families (proposal Section 9).

point_rt    -- MAE, median AE, RMSE, R2, tolerance-window hit rates
density     -- KL/JS divergence, earth-mover distance, CRPS, window probability
validity    -- see elutediff.serialization.parser.validity_report
uncertainty -- interval coverage, calibration error
ranking     -- top-k candidate ranking / filtering at fixed recall
"""

from elutediff.evaluation.density import crps_1d, earth_mover, js_divergence, window_probability
from elutediff.evaluation.point_rt import point_rt_metrics, tolerance_hit_rate

__all__ = [
    "point_rt_metrics",
    "tolerance_hit_rate",
    "js_divergence",
    "earth_mover",
    "crps_1d",
    "window_probability",
]
