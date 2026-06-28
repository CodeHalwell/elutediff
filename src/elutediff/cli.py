"""Command-line entry point: ``elutediff <command>``.

Thin dispatcher over the pipeline stages so experiments are reproducible from a
config file. Heavy stages (train/eval) import torch/unsloth lazily inside their
handlers, so ``elutediff audit`` runs on a CPU-only box.
"""

from __future__ import annotations

import argparse
import sys

from elutediff.audit import sweep_bin_widths
from elutediff.config import Config, load_config


def _load(args) -> Config:
    return load_config(args.config) if args.config else Config()


def cmd_audit(args) -> int:
    cfg = _load(args)
    print(f"Canvas budget audit (canvas_length={cfg.model.canvas_length}):")
    for res in sweep_bin_widths(
        cfg.target, bin_widths=(10.0, 5.0), canvas_length=cfg.model.canvas_length
    ):
        flag = "OK " if res.fits_canvas else "OVER"
        print(f"  [{flag}] bin_width varies -> {res.summary()}")
    return 0


def _not_wired(name: str, script: str, extra: str = "") -> int:
    """A pipeline stage that lives in a script, not the CLI. Returns non-zero so
    automation does not mistake the pointer message for a successful run."""
    print(f"`elutediff {name}` is not wired into the CLI; run {script}{extra}",
          file=sys.stderr)
    return 2


def cmd_build_targets(args) -> int:
    return _not_wired("build-targets", "scripts/build_targets.py (roadmap step 2)")


def cmd_train(args) -> int:
    return _not_wired("train", "scripts/train_diffusion.py (roadmap step 5)",
                      " -- requires the 'train' extra + GPU")


def cmd_eval(args) -> int:
    return _not_wired("eval", "scripts/evaluate.py (roadmap step 8)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="elutediff", description=__doc__)
    p.add_argument("-c", "--config", help="Path to a YAML config (configs/*.yaml).")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("audit", help="Tokenizer/canvas budget audit (roadmap step 3).").set_defaults(
        func=cmd_audit
    )
    sub.add_parser("build-targets", help="Build RT-density targets (roadmap step 2).").set_defaults(
        func=cmd_build_targets
    )
    sub.add_parser("train", help="Fine-tune DiffusionGemma (roadmap step 5).").set_defaults(
        func=cmd_train
    )
    sub.add_parser("eval", help="Evaluate a trained adapter (roadmap step 8).").set_defaults(
        func=cmd_eval
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
