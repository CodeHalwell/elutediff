#!/usr/bin/env python3
"""Tokenizer / canvas-budget audit (roadmap step 3, MANDATORY before training).

Reports whether the RT-vector target fits the 256-token canvas at 10 s and 5 s
bins. Optionally pass --model to measure real token lengths with the
DiffusionGemma tokenizer (requires the 'train' extra).

    python scripts/tokenizer_audit.py -c configs/b6_clean_vector.yaml
    python scripts/tokenizer_audit.py -c configs/b6_clean_vector.yaml --model unsloth/diffusiongemma-26B-A4B-it
"""

from __future__ import annotations

import argparse

from elutediff.audit import sweep_bin_widths
from elutediff.config import Config, load_config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", help="YAML config; defaults to built-in defaults.")
    ap.add_argument("--model", help="Optional HF model/tokenizer name for measured lengths.")
    args = ap.parse_args()

    cfg: Config = load_config(args.config) if args.config else Config()

    tokenizer = None
    if args.model:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Canvas length: {cfg.model.canvas_length}")
    print(f"RT range: {cfg.target.rt_min}-{cfg.target.rt_max}s  sigma={cfg.target.sigma}s")
    for res in sweep_bin_widths(
        cfg.target, bin_widths=(10.0, 5.0), canvas_length=cfg.model.canvas_length,
        tokenizer=tokenizer,
    ):
        flag = "OK  " if res.fits_canvas else "OVER"
        print(f"  [{flag}] {res.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
