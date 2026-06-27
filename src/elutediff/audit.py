"""Tokenizer-length / canvas-budget audit (proposal Section 11, roadmap step 3).

A tokenizer-length audit is a *mandatory* first-order requirement, not an
implementation detail: the 256-token canvas constrains bin width, RT range, and
serialization format. This module renders the target string for a given
:class:`TargetConfig` and reports how many tokens it consumes.

Two modes:
  * realistic estimate (no model): the Gemma tokenizer emits roughly one token
    per *character* -- every digit of a level and every separating space -- so
    the canvas cost is ``len(target_string) + eos``. A naive one-token-per-bin
    count hides the truth: 120 three-digit bins are ~480 tokens, not 121, and
    silently overflow the 256-token canvas.
  * measured (optional tokenizer): pass a HuggingFace tokenizer to count the
    real token length and flag any bin that is not exactly one token.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from elutediff.config import TargetConfig
from elutediff.serialization.prompts import target_string
from elutediff.targets.density import gaussian_density
from elutediff.targets.quantize import quantize


@dataclass
class AuditResult:
    n_bins: int
    canvas_length: int
    est_target_tokens: int      # realistic per-character estimate len(text)+eos
    measured_tokens: int | None  # real tokenizer length, if provided
    fits_canvas: bool
    one_token_per_bin: bool | None  # measured: is every bin a single token?

    def summary(self) -> str:
        meas = "-" if self.measured_tokens is None else str(self.measured_tokens)
        return (
            f"bins={self.n_bins} canvas={self.canvas_length} "
            f"est_tokens={self.est_target_tokens} measured={meas} "
            f"fits={self.fits_canvas} one_tok_per_bin={self.one_token_per_bin}"
        )


def audit_target(
    cfg: TargetConfig,
    canvas_length: int = 256,
    tokenizer=None,
    example_rt: float | None = None,
) -> AuditResult:
    """Audit one RT-vector target against the canvas budget.

    ``example_rt`` defaults to the middle of the grid (worst case for a centered
    peak has no effect on length, since the vector is fixed-width regardless).
    """
    if example_rt is None:
        example_rt = 0.5 * (cfg.rt_min + cfg.rt_max)

    levels = quantize(gaussian_density(example_rt, cfg), cfg)
    text = target_string(levels, cfg)

    # Realistic estimate: the Gemma tokenizer emits one token per character
    # (each digit of a level AND each separating space), so the real canvas cost
    # is len(text) + eos -- NOT the naive one-token-per-bin count. Multi-digit
    # levels therefore overflow the budget (120 three-digit bins ~= 480 tokens).
    est = len(text) + 1  # + eos

    measured = None
    one_per_bin = None
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        measured = len(ids) + 1  # + eos
        # Each bin token, when encoded alone, should be exactly one token.
        one_per_bin = all(
            len(tokenizer.encode(t, add_special_tokens=False)) == 1
            for t in target_string(levels, cfg).split()
        )

    effective = measured if measured is not None else est
    return AuditResult(
        n_bins=cfg.n_bins,
        canvas_length=canvas_length,
        est_target_tokens=est,
        measured_tokens=measured,
        fits_canvas=effective <= canvas_length,
        one_token_per_bin=one_per_bin,
    )


def sweep_bin_widths(
    base: TargetConfig, bin_widths=(10.0, 5.0), canvas_length: int = 256, tokenizer=None
) -> list[AuditResult]:
    """Audit several bin widths (proposal recommends auditing 10 s and 5 s)."""
    from dataclasses import replace

    results = []
    for bw in bin_widths:
        cfg = replace(base, bin_width=bw)
        results.append(audit_target(cfg, canvas_length, tokenizer))
    return results
