#!/usr/bin/env python3
"""Build (prompt, RT-density target) training rows from METLIN (roadmap step 2).

Pipeline per molecule:
  RT (s)  -> gaussian_density -> [apply_noise] -> quantize -> target_string
  SMILES (+descriptors/+graph) -> build_prompt

Writes a JSONL of {"prompt", "target", "rt", "smiles", "split"} rows ready for
elutediff.training.block_diffusion.build_examples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from elutediff.config import Config, load_config
from elutediff.data.metlin import load_metlin
from elutediff.serialization.prompts import build_prompt, target_string
from elutediff.targets.density import gaussian_density
from elutediff.targets.noise import apply_noise
from elutediff.targets.quantize import quantize


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", required=True, help="YAML config.")
    ap.add_argument("--out", default="data/processed/targets.jsonl")
    args = ap.parse_args()

    cfg: Config = load_config(args.config)
    molecules = load_metlin(cfg.metlin_path)  # NotImplementedError until step 1 is done

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for mol in molecules:
            density = gaussian_density(mol.rt_seconds, cfg.target)
            density = apply_noise(density, cfg.noise)
            levels = quantize(density, cfg.target)
            row = {
                "smiles": mol.smiles,
                "rt": mol.rt_seconds,
                "prompt": build_prompt(
                    smiles=mol.smiles, target_cfg=cfg.target, cond_cfg=cfg.conditioning
                ),
                "target": target_string(levels, cfg.target),
            }
            fh.write(json.dumps(row) + "\n")
    print(f"wrote targets -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
