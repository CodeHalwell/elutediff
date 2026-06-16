# DiffusionGemma + METLIN SMRT RT Density Proposal (Density-first revision)

Prepared June 16, 2026.

## Abstract

This project reframes retention-time (RT) prediction as **molecule-conditioned RT-density generation** instead of scalar-only regression. Each scalar RT label is converted into a fixed-length 1D RT-density vector using a Gaussian peak centered at observed RT, with optional weak baseline noise for robustness studies.

The core hypothesis is that DiffusionGemma's bidirectional, whole-canvas denoising can generate and refine valid RT-density signals over the full chromatographic axis, potentially improving density validity, uncertainty behavior, tolerance-window scoring, and candidate ranking.

## Primary Claims

- The dense vector is a probabilistic RT-density target, not cosmetic scalar reformatting.
- A symmetric Gaussian is a defensible first prior when only scalar RT centers are available (no raw traces).
- DiffusionGemma's whole-canvas denoising is structurally aligned with fixed-axis RT-density generation.
- Strong scalar baselines (ECFP ML, GNN, sparse graph transformer) are required controls, not optional comparisons.

## Research Question

Can bidirectional discrete diffusion learn chemically conditioned RT-density generation in ways that improve:

1. vector validity,
2. uncertainty handling,
3. RT-window scoring,
4. candidate ranking,

beyond scalar and center-bin controls?

## Data and Target Definition

METLIN SMRT provides scalar RT (seconds), structural files, descriptors, fingerprints, and PubChem IDs, but not raw chromatograms. Therefore, targets are synthetic RT-density vectors.

For molecule _i_ with RT \(r_i\), over time grid \(t_1,\ldots,t_T\):

\[
 g_i(t_j)=\exp\left(-\frac{1}{2}\left(\frac{t_j-r_i}{\sigma}\right)^2\right)
\]

Use fixed-width integer quantization (e.g., `000..100`) after max-normalization.

### Recommended initial target settings

- **RT range:** empirical 99–99.5 percentile (or 0–1200 s), report clipped fraction.
- **Bin width:** start 10 s; test 5 s if token budget permits.
- **Sigma:** at least 2–3 bins (e.g., 20–30 s for 10 s bins).
- **Quantization:** integer `000..100`.
- **Noise:** optional weak floor/drift/spikes as stress augmentation only.

## Why Gaussian First

The Gaussian is a narrow first approximation given only center labels and smoothness/locality priors. It does **not** claim real LC peaks are typically perfectly symmetric. Future upgrades include variable sigma and asymmetric/EMG-like peaks if shape supervision becomes available.

## Why DiffusionGemma

For RT-density vectors, whole-canvas denoising can jointly revise peak center, tails, and baseline with global context, unlike left-to-right generation that commits token prefixes before global placement is stable.

## Molecular Conditioning Progression

1. SMILES only
2. SMILES + descriptors
3. Atom/bond table serialization
4. Laplacian eigenvectors (LapPE) with sign handling + value rounding
5. Optional sparse graph transformer embeddings/prior features

## Training Design

- Base: DiffusionGemma 26B-A4B instruction model
- PEFT: LoRA first
- Output: fixed-width integer RT vector
- Loss: diffusion token cross-entropy (padding ignored)
- Sampling: compare 1-step vs multi-step denoising
- Parsing: strict validity checks (length/range/malformed tokens/peak validity)

## Required Baselines and Ablations

- Classical scalar: ECFP + XGBoost/random forest; descriptor/ECFP MLP
- Strong graph baselines: sparse graph transformer / GNN
- Diffusion controls:
  - center-bin output
  - Gaussian-parameter output (center+sigma)
  - clean dense vector
  - noisy dense vector
  - LapPE-conditioned vector
- Uncertainty baselines: MC dropout, ensembles, quantile regression, conformal intervals

## Evaluation Metrics

- **Point RT:** MAE, median AE, RMSE, R², tolerance-window hit rates
- **Density quality:** KL/JS, EMD, CRPS, window probability around true RT
- **Validity:** parse success, range checks, smoothness, local maxima count
- **Diffusion benefit:** step-count ablations, sample variance/error relationships
- **Uncertainty:** coverage-width tradeoffs, calibration curves
- **Annotation utility:** top-k candidate ranking / filtering at fixed recall
- **Compute:** GPU hours, peak VRAM, trainable params, inference latency

## Practical Constraints

- 256-token canvas constraints are first-order design parameters.
- Run tokenization audits early for 10 s and 5 s bins.
- Keep serialization minimal and parser strict.

## Success Criteria (Density-first)

Primary success is not necessarily lowest global MAE.

A strong success case is that clean vector models (B6/B8):

- generate valid RT-density vectors,
- improve integrated mass near true RT and candidate ranking,
- remain competitive on decoded point RT versus strong graph baselines.

## Risks and Mitigations

- Vector dismissed as scalar encoding -> keep center-bin/parameter controls central.
- Model learns format not chemistry -> scaffold + Tanimoto cluster splits.
- Overclaiming synthetic noise -> treat as augmentation only.
- Weak uncertainty from fixed sigma -> compare against explicit uncertainty baselines.
- Canvas overflow/token brittleness -> fixed-width integers + tokenizer audits + strict parser.

## Implementation Roadmap

1. Build parser, canonicalization, descriptors, graph features, and reproducible splits.
2. Implement clean and noisy RT-density target generators.
3. Run tokenizer-length audits and lock target format.
4. Train scalar and graph baselines.
5. Fine-tune DiffusionGemma on center-bin, parameterized, and clean-vector targets.
6. Add noisy-vector and LapPE ablations.
7. Run uncertainty and candidate-ranking studies.
8. Report outcomes as disciplined ablations (positive, mixed, or negative).

## Conclusion

This proposal treats RT-density generation as the primary scientific target and scalar MAE as a necessary control. It frames DiffusionGemma as a test of molecule-conditioned, whole-axis scientific signal generation under strict ablation and calibration standards.
