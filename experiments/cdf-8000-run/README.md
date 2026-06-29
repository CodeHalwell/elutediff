# Extended CDF run — 8000 steps (does more training help?)

A follow-up to [`cdf-real-run`](../cdf-real-run/) (4000 steps): the 4000-step curve
was **still improving** at the end (not plateaued), so we doubled the budget to
8000 steps. Same setup — real METLIN SMRT, scaffold split, `encoding="cdf"`,
`grad_accum=4`, OneCycle cosine — sized to 8000 steps. Final eval on the full
**200-molecule** scaffold-test set across 1/16/64 denoising steps.

Mechanically this was a **warm restart**: the step-3500 adapter from the 4000-step
run was injected under a fresh 8000-total cosine (so the LR re-warmed from ~0 back
to ~6e-5), then trained to 8000. (Run executed in 500-step chunks across many
Colab G4 VMs with resume-from-checkpoint; raw per-point data in
[`curve.json`](curve.json).)

## Final eval @ step 8000 (200 molecules)

| denoising | MAE (s) | R² | window-prob |
|-----------|---------|-----|-------------|
| **1-step** | **112.3** | **+0.527** | 0.172 |
| 16-step | 113.3 | +0.517 | 0.158 |
| 64-step | 112.4 | +0.521 | 0.164 |

## Headline progression

| | MAE (s) | R² |
|---|---------|-----|
| `density_none` control (sparse target) | 668.8 | −5.71 |
| CDF @4000 (prior run) | 126.8 | +0.498 |
| **CDF @8000** | **112.3** | **+0.527** |

Doubling the steps cut MAE another **~11%** (126.8 → 112.3 s) and lifted R²
(+0.498 → +0.527). So yes — the extra training helped, materially.

## Trajectory (1-step, intermediate evals on 40 mols — noisy)

| step | MAE (s) | R² | note |
|------|---------|-----|------|
| 4000 | 158.9 | +0.382 | warm-restart **dip** (LR re-warmed → perturbs the converged minimum) |
| 4500 | 152.5 | +0.404 | |
| 5000 | 155.8 | +0.374 | |
| 5500 | 136.1 | +0.505 | **recovering** |
| 6000 | 128.9 | +0.552 | back to ~4000-step level |
| 6500 | 120.0 | +0.607 | **surpassing** |
| 7000 | 134.6 | +0.477 | (40-mol eval noise) |
| 7500 | 112.8 | +0.644 | |
| **8000** | **112.3** | **+0.527** | *(200-mol final — the reliable number)* |

## Findings

1. **More steps help — the 4000-step result was not the ceiling.** MAE 126.8 →
   112.3 s, R² +0.498 → +0.527 on the full 200-mol test set.

2. **Warm restart goes dip → recover → surpass.** Re-warming the LR on a
   converged adapter first *hurts* (MAE jumps to ~159 at step 4000) as it's
   knocked off its minimum, then re-converges and exceeds it as the new cosine
   anneals. An early mid-run read called it a "net negative" — that was premature;
   it just needed the full schedule to climb back out and past.

3. **Multi-step refinement is roughly neutral at convergence.** At 8000, 1/16/64-step
   are within noise of each other on the 200-mol eval (MAE 112–113, R² 0.52).
   The diffusion-refinement edge that appeared on the noisy 40-mol intermediates
   (e.g. 16-step R² +0.689 @7500) washes out on the reliable larger eval — so the
   converged model is close to deterministic-per-molecule.

4. **40-molecule intermediate evals are noisy (±15 MAE, ±0.1 R²).** Read the
   trajectory shape, not individual points; the 200-mol finals (@4000, @8000) are
   the trustworthy numbers.

## Caveat
The step-8000 LoRA adapter was not recovered (the VM was preempted during the
final download; the metrics are captured from the run log). The latest saved
adapter is step-7500. Re-running would recover an 8000 adapter if needed for
inference.

## Files
- [`curve.json`](curve.json) — full 500→8000 eval curve.
