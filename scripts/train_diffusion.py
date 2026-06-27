#!/usr/bin/env python3
"""Fine-tune DiffusionGemma on RT-density targets (roadmap step 5).

Requires the 'train' extra and a >= ~50GB GPU (A100 80GB / H100). Mirrors the
Unsloth DiffusionGemma Sudoku notebook, swapping the Sudoku grid for RT vectors.

    python scripts/train_diffusion.py -c configs/b6_clean_vector.yaml --data data/processed/targets.jsonl
"""

from __future__ import annotations

import argparse
import json

from elutediff.config import Config, load_config
from elutediff.models.diffusion import add_lora, load_model, model_dimensions
from elutediff.training.block_diffusion import build_examples, train


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--data", required=True, help="JSONL from scripts/build_targets.py")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    cfg: Config = load_config(args.config)
    all_rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    # Train only on the train split -- never on val/test, which would leak
    # held-out labels and invalidate the reported experiments.
    rows = [r for r in all_rows if r.get("split", "train") == "train"]
    print(f"training on {len(rows)} train rows (of {len(all_rows)} total)")

    model, processor = load_model(cfg.model, hf_token=args.hf_token)
    print("model dims:", model_dimensions(model))
    model = add_lora(model, cfg.model)

    examples = build_examples(rows, processor, cfg.model)
    print(f"usable examples: {len(examples)} / {len(rows)} (rest overflow the canvas)")

    model = train(model, examples, cfg.model, cfg.train,
                  processor=processor, target_cfg=cfg.target)
    model.save_pretrained(cfg.train.output_dir)
    processor.save_pretrained(cfg.train.output_dir)
    print(f"saved LoRA adapter -> {cfg.train.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
