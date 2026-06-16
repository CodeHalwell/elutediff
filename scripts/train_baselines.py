#!/usr/bin/env python3
"""Train and evaluate scalar RT baselines (B1/B2) -- the known bar (roadmap step 4).

Consumes the JSONL produced by scripts/build_targets.py (needs smiles, rt, split)
and reports point-RT metrics + tolerance hit rates per fold. Optionally adds a
split-conformal interval (B9) calibrated on val and scored on test.

    python scripts/train_baselines.py --data data/processed/targets.jsonl --model rf --features ecfp
    python scripts/train_baselines.py --data data/processed/targets.jsonl --model xgb --features ecfp+descriptors --conformal
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from elutediff.config import ConditioningConfig
from elutediff.evaluation.point_rt import point_rt_metrics, tolerance_hit_rate
from elutediff.evaluation.uncertainty import interval_coverage, median_interval_width
from elutediff.models.baselines import ConformalInterval, build_baseline, featurize


def _fold(rows, name):
    sub = [r for r in rows if r.get("split") == name]
    return [r["smiles"] for r in sub], np.array([r["rt"] for r in sub], dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="JSONL from scripts/build_targets.py")
    ap.add_argument("--model", default="rf", choices=["rf", "xgb", "mlp"])
    ap.add_argument("--features", default="ecfp",
                    choices=["ecfp", "descriptors", "ecfp+descriptors"])
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--conformal", action="store_true", help="Add split-conformal intervals.")
    ap.add_argument("--tolerances", type=float, nargs="+", default=[15.0, 30.0, 60.0])
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    descriptors = ConditioningConfig().descriptors

    def feats(smiles):
        return featurize(smiles, args.features, descriptors=descriptors, n_bits=args.n_bits)

    tr_s, tr_y = _fold(rows, "train")
    va_s, va_y = _fold(rows, "val")
    te_s, te_y = _fold(rows, "test")
    print(f"folds: train={len(tr_s)} val={len(va_s)} test={len(te_s)}")

    model = build_baseline(args.model)
    model.fit(feats(tr_s), tr_y)

    for name, smi, y in (("val", va_s, va_y), ("test", te_s, te_y)):
        if not smi:
            continue
        pred = model.predict(feats(smi))
        m = point_rt_metrics(y, pred)
        hits = tolerance_hit_rate(y, pred, args.tolerances)
        print(
            f"[{name}] MAE {m['mae']:.1f}s medAE {m['median_ae']:.1f}s "
            f"RMSE {m['rmse']:.1f}s R2 {m['r2']:.3f} | "
            + " ".join(f"{k} {v*100:.0f}%" for k, v in hits.items())
        )

    if args.conformal and va_s and te_s:
        conf = ConformalInterval(model, level=0.9).calibrate(feats(va_s), va_y)
        lo, hi = conf.predict_interval(feats(te_s))
        print(
            f"[test] conformal 90%: coverage {interval_coverage(te_y, lo, hi)*100:.1f}% "
            f"| median width {median_interval_width(lo, hi):.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
