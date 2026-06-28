# elutediff

**Molecule-conditioned retention-time *density* generation with discrete text diffusion.**

`elutediff` reframes retention-time (RT) prediction on the **METLIN SMRT** dataset
as conditioned RT-*density* generation instead of scalar regression. Each scalar
RT label becomes a fixed-length 1-D Gaussian density over a quantized time axis,
and **DiffusionGemma 26B-A4B** (a block-diffusion MoE, fine-tuned via
[Unsloth](https://github.com/unslothai/unsloth) + LoRA) is trained to generate
that density across the whole chromatographic canvas at once.

The hypothesis: bidirectional, whole-canvas denoising is structurally aligned
with a fixed-axis density and may improve **density validity, uncertainty,
tolerance-window scoring, and candidate ranking** beyond scalar and center-bin
controls — while strong graph/ECFP regressors remain the known bar for point MAE.

## Status

Initial project scaffold. The CPU-only core (RT-density target construction,
fixed-width tokenization, strict parsing, and the evaluation metric families) is
implemented and unit-tested. METLIN/RDKit featurization and the classical/GNN
baselines are typed scaffolds; the DiffusionGemma training and sampling code is
ported from the reference Unsloth Sudoku notebook and runs once the `train` extra
and a GPU are available.

```bash
pip install -e ".[dev]"            # core + pytest/ruff
pytest                              # baseline tests auto-skip without the extra
pip install -e ".[dev,baselines]"  # add scikit-learn/xgboost for the full suite
elutediff -c configs/b6_clean_vector.yaml audit   # canvas budget check (CPU)
```

## Notebooks (Google Colab)

Runnable Colab notebooks live in [`notebooks/`](notebooks/): data + RT-density
targets and the RT baselines run on a free CPU runtime; the DiffusionGemma
fine-tune needs an A100/H100. Each one installs `elutediff` from GitHub and runs
on synthetic demo data out of the box.

## Documentation

- [`docs/project-structure.md`](docs/project-structure.md) — package layout, the
  roadmap → code map, dependency extras, and quickstart.
- [`docs/density-first-revision.md`](docs/density-first-revision.md) — the
  density-first scientific proposal (target design, baselines, ablation grid,
  evaluation, risks).

## License

Apache-2.0.
