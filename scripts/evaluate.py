#!/usr/bin/env python3
"""Evaluate a trained adapter: sweep denoising steps and score (roadmap step 8).

Generates RT vectors at each denoising-step count, parses them strictly, decodes
a point RT, and reports point/density/validity metrics. "Refinement is the
point": expect 1-step << multi-step.

    python scripts/evaluate.py -c configs/b6_clean_vector.yaml --data data/processed/targets.jsonl --adapter diffusiongemma_lora
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from elutediff.config import Config, load_config
from elutediff.evaluation.density import window_probability
from elutediff.evaluation.point_rt import point_rt_metrics, tolerance_hit_rate
from elutediff.serialization.parser import decoded_rt, parse_rt_vector, validity_report
from elutediff.targets.density import time_grid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--data", required=True, help="Held-out JSONL rows.")
    ap.add_argument("--adapter", required=True, help="Saved LoRA adapter dir.")
    args = ap.parse_args()

    cfg: Config = load_config(args.config)
    # Evaluate only held-out rows: keep the test split (or all rows if the file
    # carries no split field, i.e. it is already a held-out-only set).
    all_rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    rows = [r for r in all_rows if r.get("split", "test") == "test"][: cfg.eval.n_eval]
    print(f"evaluating on {len(rows)} held-out rows (of {len(all_rows)} total)")
    grid = time_grid(cfg.target)

    # Lazy heavy imports so --help works without torch/unsloth installed.
    from elutediff.models.diffusion import load_model
    from elutediff.training.sampling import generate

    model, processor = load_model(cfg.model)
    model.load_adapter(args.adapter)

    for steps in cfg.eval.denoising_steps:
        y_true, y_pred, valid, win_prob = [], [], 0, []
        for r in rows:
            text = generate(model, processor, r["prompt"], steps, cfg.model.canvas_length)
            res = parse_rt_vector(text, cfg.target)
            if not res.ok:
                continue
            valid += 1
            rt_hat = decoded_rt(res.levels, cfg.target, mode=cfg.eval.decode_mode)
            y_true.append(r["rt"])
            y_pred.append(rt_hat)
            win_prob.append(window_probability(res.levels, grid, r["rt"], cfg.target.sigma))
            _ = validity_report(res.levels, cfg.target)

        if not y_pred:
            print(f"{steps:>3}-step: no valid vectors parsed")
            continue
        m = point_rt_metrics(y_true, y_pred)
        hits = tolerance_hit_rate(y_true, y_pred, cfg.eval.rt_tolerances_s)
        print(
            f"{steps:>3}-step | valid {valid}/{len(rows)} | "
            f"MAE {m['mae']:.1f}s R2 {m['r2']:.3f} | "
            + " ".join(f"{k} {v*100:.0f}%" for k, v in hits.items())
            + f" | mean window-prob {np.mean(win_prob):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
