#!/usr/bin/env python3
"""Build (prompt, RT-density target) training rows from METLIN (roadmap step 2).

Pipeline per molecule:
  RT (s)  -> gaussian_density -> [apply_noise] -> quantize -> target_string
  SMILES (+descriptors/+atom-bond/+LapPE) -> build_prompt

Applies the configured split (random/scaffold/cluster) and writes a JSONL of
{"smiles", "rt", "pubchem", "split", "prompt", "target"} rows ready for
elutediff.training.block_diffusion.build_examples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from elutediff.config import Config, load_config
from elutediff.data.metlin import load_metlin
from elutediff.data.molecules import atom_bond_table, compute_descriptors
from elutediff.data.splits import make_split
from elutediff.serialization.prompts import build_prompt, target_string
from elutediff.targets.density import clipped_fraction, gaussian_density
from elutediff.targets.noise import apply_noise
from elutediff.targets.quantize import quantize


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", required=True, help="YAML config.")
    ap.add_argument("--out", default="data/processed/targets.jsonl")
    args = ap.parse_args()

    cfg: Config = load_config(args.config)
    molecules, stats = load_metlin(cfg.metlin_path, return_stats=True)
    print(stats)

    smiles = [m.smiles for m in molecules]
    rts = [m.rt_seconds for m in molecules]
    print(f"clipped fraction (RT outside grid): {clipped_fraction(rts, cfg.target):.4f}")

    split = make_split(smiles, cfg.split)
    fold_of = {}
    for name, idxs in (("train", split.train_idx), ("val", split.val_idx), ("test", split.test_idx)):
        for i in idxs:
            fold_of[i] = name
    print(f"split sizes (train/val/test): {split.sizes()}")

    cond = cfg.conditioning
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for i, mol in enumerate(molecules):
            density = apply_noise(gaussian_density(mol.rt_seconds, cfg.target), cfg.noise)
            levels = quantize(density, cfg.target)

            descriptors = compute_descriptors(mol.smiles, cond.descriptors) if cond.level >= 2 else None
            atoms_bonds = atom_bond_table(mol.smiles) if cond.level >= 3 else None
            # LapPE (level >= 4) is wired in data/graph_features.py at roadmap step 7.

            row = {
                "smiles": mol.smiles,
                "rt": mol.rt_seconds,
                "pubchem": mol.pubchem_id,
                "split": fold_of.get(i, "train"),
                "prompt": build_prompt(
                    smiles=mol.smiles,
                    target_cfg=cfg.target,
                    cond_cfg=cond,
                    descriptors=descriptors,
                    atom_bond_table=atoms_bonds,
                ),
                "target": target_string(levels, cfg.target),
            }
            fh.write(json.dumps(row) + "\n")
    print(f"wrote {len(molecules)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
