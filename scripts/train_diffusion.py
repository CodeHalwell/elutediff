#!/usr/bin/env python3
"""Fine-tune DiffusionGemma on RT-density targets, with resume + eval curve.

Self-contained run harness (roadmap step 5). Builds the void-filtered targets and
the scaffold split from ``cfg.metlin_path`` in-process (no separate build step),
trains the block-diffusion objective, and every ``--checkpoint-every`` steps
saves the LoRA adapter, optionally pushes it to the HF Hub, and runs a held-out
eval -- appending an MAE/R2/window-prob curve to ``<out>/curve.json`` so a
preempted run resumes cleanly and the run is reproducible from the repo (previous
runs' logic lived only in disposable notebooks, and their adapters were lost).

Requires the 'train'+'chem' extras and a >= ~50 GB GPU (A100 80GB / H100 / G4).

    # fresh run
    python scripts/train_diffusion.py -c configs/cdf_8000.yaml \
        --hf-repo HallD/elutediff-cdf-8000
    # resume a preempted run from the last saved adapter
    python scripts/train_diffusion.py -c configs/cdf_8000.yaml \
        --resume-adapter runs/cdf_8000/ckpt --start-step 2500 \
        --hf-repo HallD/elutediff-cdf-8000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from elutediff.config import Config, load_config
from elutediff.data.metlin import load_metlin
from elutediff.data.splits import make_split
from elutediff.evaluation.density import window_probability
from elutediff.evaluation.point_rt import point_rt_metrics, tolerance_hit_rate
from elutediff.serialization.parser import decoded_rt, parse_rt_vector
from elutediff.serialization.prompts import build_prompt, target_string
from elutediff.targets.density import gaussian_density, time_grid
from elutediff.targets.noise import apply_noise
from elutediff.targets.quantize import quantize


def _build_rows(cfg: Config):
    """Load METLIN (void-filtered) and build train (prompt+target) / test (prompt+rt) rows."""
    mols, stats = load_metlin(
        cfg.metlin_path, return_stats=True, min_retention_s=cfg.data.min_retention_s
    )
    print(stats, flush=True)
    split = make_split([m.smiles for m in mols], cfg.split)
    fold = {i: name for name, idx in (("train", split.train_idx), ("val", split.val_idx),
                                      ("test", split.test_idx)) for i in idx}
    train_rows, eval_rows = [], []
    for i, m in enumerate(mols):
        prompt = build_prompt(smiles=m.smiles, target_cfg=cfg.target, cond_cfg=cfg.conditioning)
        if fold.get(i, "train") == "train":
            density = apply_noise(gaussian_density(m.rt_seconds, cfg.target),
                                  cfg.noise, seed=(cfg.noise.seed, i))
            levels = quantize(density, cfg.target)
            train_rows.append({"prompt": prompt, "target": target_string(levels, cfg.target)})
        elif fold.get(i) == "test":
            eval_rows.append({"prompt": prompt, "rt": m.rt_seconds})
    print(f"train rows {len(train_rows)} | test rows {len(eval_rows)}", flush=True)
    return train_rows, eval_rows


def make_eval_fn(cfg: Config, eval_rows, gen):
    """Build ``do_eval(n, steps_list, tag) -> dict`` from a ``gen(prompt, steps)`` callable."""
    grid = time_grid(cfg.target)

    def do_eval(n, steps_list, tag):
        out = {}
        for steps in steps_list:
            yt, yp, valid, wp = [], [], 0, []
            for j, r in enumerate(eval_rows[:n]):
                if j % 25 == 0:
                    print(f"    eval {steps}-step {j}/{n}", flush=True)
                pr = parse_rt_vector(gen(r["prompt"], steps), cfg.target)
                if not pr.ok:
                    continue
                valid += 1
                yt.append(r["rt"])
                yp.append(decoded_rt(pr.levels, cfg.target, mode=cfg.eval.decode_mode))
                wp.append(window_probability(pr.levels, grid, r["rt"], cfg.target.sigma))
            if yp:
                m = point_rt_metrics(yt, yp)
                h = tolerance_hit_rate(yt, yp, cfg.eval.rt_tolerances_s)
                out[str(steps)] = {"valid": valid, "n": n, "mae": m["mae"], "r2": m["r2"],
                                   "tolerance_hits": h, "window_prob": float(np.mean(wp))}
                print(f"  [{tag}] {steps:>3}-step valid {valid}/{n} "
                      f"MAE {m['mae']:.1f} R2 {m['r2']:.3f} wp {np.mean(wp):.3f}", flush=True)
            else:
                out[str(steps)] = {"valid": 0, "n": n}
                print(f"  [{tag}] {steps}-step no valid", flush=True)
        return out
    return do_eval


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--out", default=None, help="output dir (default: cfg.train.output_dir)")
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--eval-n-intermediate", type=int, default=40,
                    help="held-out mols for periodic eval (full n_eval at the final step)")
    ap.add_argument("--resume-adapter", default=None, help="adapter dir to inject before training")
    ap.add_argument("--start-step", type=int, default=0)
    ap.add_argument("--hf-repo", default=None, help="push each checkpoint here (e.g. HallD/elutediff-cdf-8000)")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    cfg: Config = load_config(args.config)
    out = Path(args.out or cfg.train.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from elutediff.models.diffusion import add_lora, load_model, model_dimensions
    from elutediff.training.block_diffusion import build_examples, train
    from elutediff.training.sampling import generate as _generate

    train_rows, eval_rows = _build_rows(cfg)
    model, processor = load_model(cfg.model, hf_token=args.hf_token)
    print("model dims:", model_dimensions(model), flush=True)
    model = add_lora(model, cfg.model)
    if args.resume_adapter:
        model.load_adapter(args.resume_adapter, adapter_name="default")
        print(f"injected {args.resume_adapter} @ start_step {args.start_step}", flush=True)

    examples = build_examples(train_rows, processor, cfg.model)
    print(f"usable examples: {len(examples)} / {len(train_rows)}", flush=True)

    def gen(prompt, steps):
        return _generate(model, processor, prompt, steps, cfg.model.canvas_length)
    do_eval = make_eval_fn(cfg, eval_rows, gen)

    curve_path = out / "curve.json"
    curve = json.loads(curve_path.read_text()) if curve_path.exists() else []

    def on_checkpoint(step, model):
        ck = out / "ckpt"
        model.save_pretrained(ck)
        processor.save_pretrained(ck)
        final = step >= cfg.train.steps
        n = cfg.eval.n_eval if final else args.eval_n_intermediate
        steps_list = cfg.eval.denoising_steps if final else cfg.eval.denoising_steps[:2]
        curve.append({"step": step, "eval": do_eval(n, steps_list, f"step{step}")})
        curve_path.write_text(json.dumps(curve, indent=2))
        (out / "step.txt").write_text(str(step))
        if args.hf_repo:
            model.push_to_hub(args.hf_repo, token=args.hf_token, commit_message=f"step {step}")
            print(f"  pushed adapter -> {args.hf_repo} (step {step})", flush=True)

    t0 = time.time()
    train(model, examples, cfg.model, cfg.train, processor=processor, target_cfg=cfg.target,
          start_step=args.start_step, on_checkpoint=on_checkpoint,
          checkpoint_every=args.checkpoint_every)
    print(f"done in {time.time() - t0:.0f}s -> {out}/ckpt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
