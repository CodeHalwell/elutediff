# Ablation sweep: target encoding × peak-aware loss

A single-base-model sweep over the two arms (`TargetConfig.encoding`,
`TrainConfig.peak_loss`/`peak_lambda`) on real METLIN SMRT (scaffold split),
run on a Colab G4 via [`run_sweep.py`](run_sweep.py).

## Method

- One frozen DiffusionGemma 26B base, a **fresh LoRA per config** (`reinit_lora`),
  so configs share the base and the exact eval subset/seed.
- **grad_accum=1, 1000 steps/config**, eval at 500/1000 on the first 40
  scaffold-test molecules (1-step + 16-step denoising), OneCycle cosine LR.
- Metrics: point **MAE/R²** (argmax-decoded RT), **window-probability** (density
  mass within ±σ of the true RT — the density-quality metric), tolerance-hit
  rate, and parse validity.
- Resumable: results written + downloaded after each config; the harness skips
  configs already in `sweep_results.json` (the run survived several Colab VM
  preemptions this way).

> **Caveat on the numbers.** These are short `grad_accum=1` runs chosen so the
> whole grid fits within a (frequently-preempted) VM lifetime. They are valid for
> **ranking**, but absolute values differ from the longer `grad_accum=4` /
> 1500-step reference runs (control ~655 s, CDF ~199 s). See the EMD note below.

## Results (@1000, 1-step)

| config | encoding | peak loss (λ) | MAE (s) | R² | window-prob | ±60 s | valid |
|--------|----------|---------------|---------|-----|-------------|-------|-------|
| `density_none` (control) | density | — | 668.8 | −5.71 | 0.034 | 0% | 40/40 |
| `cdf_none` | cdf | — | 197.8 | +0.01 | 0.075 | 18% | 40/40 |
| `cdf_emd_0.01` | cdf | emd (0.01) | 202.1 | −0.08 | 0.075 | 22% | 40/40 |
| `cdf_emd_0.05` | cdf | emd (0.05) | 203.3 | −0.13 | 0.100 | 28% | 40/40 |
| `cdf_softargmax_0.05` | cdf | softargmax (0.05) | — | — | — | 0% | **0/40** |
| `density_emd_0.05` | density | emd (0.05) | **668.8** | **−5.71** | **0.034** | **0%** | 40/40 |

## Findings

1. **CDF encoding is the decisive fix.** The sparse-density control collapses to a
   constant, molecule-agnostic output (MAE 669, R² −5.7, 0% within 60 s). *Every*
   CDF variant jumps to MAE ~200, R² ≈ 0, window-prob ~0.075–0.10, ±60 s 18–28%.
   The dense (thermometer/CDF) target is what lets token-CE learn RT at all —
   because a misplaced peak now costs a whole run of wrong `0→9` transition tokens
   instead of a handful of bins in a sea of zeros.

2. **EMD peak loss: a little helps; a lot is a long-training liability.** In these
   short runs, more EMD monotonically improved tolerance (±60 s 18→22→28%) and
   window-prob (0.075→0.100). **But** a separate longer `grad_accum=4` / 1500-step
   run showed `cdf_emd_0.05` *cratering* window-prob (0.125→0.025) by gaming the
   loss into brittle single-bin spikes that nail the soft peak-mean but carry no
   proper mass. So short-run ranking flatters high λ; the **safe pick is
   `cdf_none` or `cdf+emd@0.01`** (keeps density quality, modest tolerance gain).

3. **`softargmax` is broken** — `0/40 valid`: it destroys the model's ability to
   emit parseable density vectors. Discard.

4. **EMD alone does *not* rescue the sparse target — CDF is necessary, not just
   helpful.** `density_emd_0.05` (sparse PDF + EMD) collapses to **exactly** the
   control's numbers (MAE 668.835, R² −5.71, wp 0.034, identical to the decimal).
   So adding the peak-aware loss to the *sparse* encoding does nothing: the
   all-zeros collapse basin is too strong for EMD to escape (a degenerate
   all-zeros prediction gives the EMD term no useful gradient). This cleanly
   disentangles the two arms: **CDF encoding is necessary *and* sufficient to
   break the collapse; EMD is only a secondary refinement that requires CDF to
   have any effect.**

## Recommendation

Default to **CDF encoding**. Add **gentle EMD (λ≈0.01)** only for a marginal
tolerance gain, and **monitor window-probability** on longer runs for the
spike-gaming collapse. Avoid high-λ EMD for long training, and avoid softargmax.

## Files

- [`run_sweep.py`](run_sweep.py) — the Colab harness (idempotent, one config per
  invocation; reload-base-once + reinit-LoRA-per-config).
- [`sweep_results.json`](sweep_results.json) — raw per-config eval results.
