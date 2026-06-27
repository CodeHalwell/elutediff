"""Strict parser + validity checks for generated RT-density vectors (Section 7).

A strict parser is a first-order requirement: it turns "did the model produce a
well-formed density?" into a measurable yes/no and underpins the *vector
validity* metric family (length, parse success, range, single dominant peak,
smoothness, local maxima count).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from elutediff.config import TargetConfig
from elutediff.targets.quantize import dequantize, emitted_to_density

# A token is exactly `token_width` digits; we extract them from possibly noisy text.
_TOKEN_RE = re.compile(r"\d+")


@dataclass
class ParseResult:
    ok: bool
    levels: np.ndarray | None       # integer levels 0..scale, length n_bins
    reason: str = ""


def parse_rt_vector(text: str, cfg: TargetConfig, strict: bool = True) -> ParseResult:
    """Parse generated text into integer levels, enforcing the fixed format.

    In ``strict`` mode every token must be exactly ``token_width`` digits and the
    count must equal ``n_bins``. In lenient mode we take the first ``n_bins``
    digit-runs, clamp to range, and report success if at least ``n_bins`` were
    found (useful for diagnosing near-misses).
    """
    # Strip a wrapping <RT_VECTOR ...> ... </RT_VECTOR> if present.
    body = text
    m = re.search(r"<RT_VECTOR[^>]*>(.*?)</RT_VECTOR>", text, flags=re.DOTALL)
    if m:
        body = m.group(1)

    raw_tokens = _TOKEN_RE.findall(body)

    if strict:
        bad = [t for t in raw_tokens if len(t) != cfg.token_width]
        if bad:
            return ParseResult(False, None, f"malformed token(s): {bad[:3]}")
        if len(raw_tokens) != cfg.n_bins:
            return ParseResult(
                False, None, f"expected {cfg.n_bins} tokens, got {len(raw_tokens)}"
            )

    levels = [int(t) for t in raw_tokens[: cfg.n_bins]]
    if len(levels) < cfg.n_bins:
        return ParseResult(False, None, f"too few tokens: {len(levels)} < {cfg.n_bins}")

    arr = np.asarray(levels, dtype=int)  # emitted levels (PDF or CDF ramp)
    if arr.max(initial=0) > cfg.scale:
        if strict:
            return ParseResult(False, None, f"level exceeds scale {cfg.scale}")
        arr = np.clip(arr, 0, cfg.scale)

    # Decode the configured encoding so callers always receive a PDF.
    return ParseResult(True, emitted_to_density(arr, cfg))


def validity_report(levels: np.ndarray, cfg: TargetConfig) -> dict:
    """Compute the vector-validity metrics for a parsed level array.

    Returns a dict with: ``length_ok``, ``range_ok``, ``n_local_maxima``,
    ``single_dominant_peak``, ``smoothness`` (mean abs first difference, lower is
    smoother), and ``peak_bin`` (argmax).
    """
    arr = np.asarray(levels, dtype=int)
    n = arr.size

    local_max = _count_local_maxima(arr)
    smoothness = float(np.abs(np.diff(arr)).mean()) if n > 1 else 0.0

    return {
        "length_ok": n == cfg.n_bins,
        "range_ok": bool(arr.min(initial=0) >= 0 and arr.max(initial=0) <= cfg.scale),
        "n_local_maxima": local_max,
        "single_dominant_peak": local_max == 1,
        "smoothness": smoothness,
        "peak_bin": int(arr.argmax()) if n else -1,
    }


def decoded_rt(levels: np.ndarray, cfg: TargetConfig, mode: str = "argmax") -> float:
    """Decode a single scalar RT (seconds) from a density vector.

    ``argmax`` returns the center of the peak bin; ``centroid`` returns the
    intensity-weighted mean over the time grid (a softer estimate).
    """
    from elutediff.targets.density import time_grid

    grid = time_grid(cfg)
    arr = np.asarray(levels, dtype=float)
    if mode == "argmax":
        return float(grid[int(arr.argmax())])
    if mode == "centroid":
        w = dequantize(arr, cfg)
        total = w.sum()
        return float((grid * w).sum() / total) if total > 0 else float("nan")
    raise ValueError(f"unknown decode mode: {mode}")


def _count_local_maxima(arr: np.ndarray) -> int:
    """Count strict *interior* local maxima, treating a flat top as one peak.

    A maximal plateau ``[i, j]`` is a peak only when it has real neighbours on
    both sides that are strictly lower. Requiring interior neighbours excludes
    degenerate outputs -- an all-constant (e.g. all-zero) vector, a monotonic
    ramp, or any boundary plateau -- which must not satisfy the
    single-dominant-peak validity criterion.
    """
    n = arr.size
    if n < 3:
        return 0
    count = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[j + 1] == arr[i]:
            j += 1  # walk across a plateau [i, j]
        if i > 0 and j < n - 1 and arr[i - 1] < arr[i] and arr[j + 1] < arr[j]:
            count += 1
        i = j + 1
    return count
