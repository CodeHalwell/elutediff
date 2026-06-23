# CLAUDE.md

Guidance for AI assistants working in this repository.

## What this project is

`elutediff` reframes **retention-time (RT) prediction** on the METLIN SMRT
dataset as conditioned RT-**density** generation instead of scalar regression.
Each scalar RT label is turned into a fixed-length 1-D Gaussian density over a
quantized time axis, serialized as space-separated integer tokens, and
**DiffusionGemma 26B-A4B** (a block-diffusion MoE fine-tuned via Unsloth + LoRA)
is trained to generate that whole-canvas density at once.

The scientific proposal is `docs/density-first-revision.md`; the proposal →
code map is `docs/project-structure.md`. Read those for the "why"; this file is
the operational "how".

## Repository layout

```
src/elutediff/        # the installable package (all reusable logic lives here)
  config.py           # typed dataclasses + YAML loader (single source of config truth)
  cli.py              # `elutediff audit|build-targets|train|eval` dispatcher
  audit.py            # tokenizer/canvas budget audit
  targets/            # scalar RT -> Gaussian density -> quantized integer vector
  serialization/      # prompt assembly + strict canvas parser / RT decode
  data/               # METLIN loading, RDKit descriptors/ECFP, splits, LapPE
  models/             # baselines (B1/B2), gnn (B3), diffusion (Unsloth loaders)
  training/           # block-diffusion corrupt-canvas objective + sampling
  evaluation/         # point_rt, density, uncertainty, ranking metric families
scripts/              # thin runnable pipeline stages (argparse, call into package)
configs/              # one YAML per ablation; keys mirror config.py dataclasses
notebooks/            # Colab drivers (thin wrappers over the package)
tests/                # pytest; cover the CPU-only core
data/                 # raw/ + processed/, git-ignored (only .gitkeep tracked)
docs/                 # scientific proposal + structure map
```

## Implementation status (don't assume code is a stub — check)

- **Implemented + unit-tested on CPU**: `config`, `audit`, `targets/`,
  `serialization/`, `evaluation/`, plus `data/` (needs the `chem` extra) and
  `models/baselines.py`, `models/gnn.py`, `data/graph_features.py`.
- **Wired** (real Unsloth/DiffusionGemma code ported from the reference Sudoku
  notebook, runnable only with the `train` extra + a GPU): `models/diffusion.py`,
  `training/block_diffusion.py`, `training/sampling.py`.

## Environment & commands

```bash
pip install -e ".[dev]"      # core (numpy/scipy/pandas/pyyaml) + pytest + ruff
pytest                       # run the unit suite (configured in pyproject.toml)
ruff check src tests scripts # lint (line-length 100, target py310)
elutediff -c configs/b6_clean_vector.yaml audit   # canvas budget check (CPU)
```

Optional extras are intentionally split so the CPU core de-risks the design
before any GPU spend:

- `chem` — RDKit (molecule parsing, descriptors, ECFP, graph features, splits)
- `baselines` — scikit-learn + xgboost (B1/B2 regressors)
- `graph` — torch + torch-geometric (B3 GNN)
- `train` — Unsloth + transformers/peft/accelerate (DiffusionGemma fine-tune)

Heavy stages import torch/unsloth **lazily inside their handlers**, so
`elutediff audit` and the target/eval code must keep running on a CPU-only box.
Preserve that — do not add top-level torch/rdkit imports to the CPU core.

### Full pipeline (needs a ≥ ~50 GB GPU for train)

```bash
pip install -e ".[chem,baselines,train]"
python scripts/download_metlin.py                                   # figshare DOI 10.6084/m9.figshare.8038913
python scripts/build_targets.py  -c configs/b6_clean_vector.yaml --out data/processed/targets.jsonl
python scripts/train_diffusion.py -c configs/b6_clean_vector.yaml --data data/processed/targets.jsonl
python scripts/evaluate.py        -c configs/b6_clean_vector.yaml --data data/processed/targets.jsonl --adapter runs/b6_clean_vector
```

DiffusionGemma 26B-A4B is ~52 GB in bf16; the 128 MoE experts (~46 GB) stay bf16,
so `load_in_4bit` cannot shrink them. A 40 GB GPU offloads and is impractically
slow — use A100 80 GB / H100 / B200.

## Conventions

- **Config is centralized and typed.** All knobs live in `src/elutediff/config.py`
  as nested `@dataclass`es with proposal-backed defaults. `configs/*.yaml` override
  only changed fields; YAML keys must mirror the dataclass field names exactly
  (`load_config` raises `KeyError` on unknown keys). When adding a tunable, add it
  to the dataclass first, then to `base.yaml`.
- **One YAML per ablation.** `base.yaml` holds defaults; `b4_center_bin`,
  `b6_clean_vector`, `b7_noisy_vector`, `b8_lappe` are the experiment variants.
- **Scripts are thin.** Real logic lives in the package so it is importable,
  testable, and shared with the notebooks; `scripts/` and `notebooks/` only parse
  args / drive. Keep new logic in `src/elutediff/`, not in scripts.
- **The canvas target is header-free.** `target_string()` emits only
  space-separated fixed-width tokens (e.g. `"050 075 100 ..."`) to conserve the
  256-token budget; the `<RT_VECTOR ...>` wrapper (`format_rt_vector`) is for
  logging/inspection only.
- **Conditioning levels are additive** (Section 6): 1 SMILES, 2 +descriptors,
  3 +atom/bond table, 4 +LapPE, 5 +graph embedding. Each level keeps everything below.
- **LapPE**: deterministic canonical-sign LapPE is baked into the JSONL; sign-flip
  augmentation happens on the fly in the training loop, not at build time.
- `from __future__ import annotations` everywhere; type hints throughout;
  module/function docstrings reference the relevant proposal section.
- **`numpy` core, no torch in the data path.** Targets/serialization/evaluation
  use numpy only.

## Testing

- `pytest` runs everything under `tests/` (see `[tool.pytest.ini_options]`).
- Invoke as `python -m pytest` — the bare `pytest` binary may resolve to a
  different interpreter without the deps installed.
- The full suite needs the heavier extras: `test_data.py`, `test_baselines.py`,
  and `test_graph_features.py` import **rdkit** (`chem`/`baselines`), and
  `test_gnn.py` needs **torch** (`graph`). A missing extra fails *collection* and
  aborts the whole run. To run the pure-numpy core only, target the files
  explicitly:

  ```bash
  python -m pytest tests/test_targets.py tests/test_serialization.py \
                   tests/test_evaluation.py tests/test_audit_and_config.py -q
  ```

  Run the full suite with `pip install -e ".[dev,chem,baselines,graph]"`.
- Core tests assert numeric properties (grid shape, peak normalization, quantize
  round-trips, clipped fraction, noise renormalization).
- When you change `targets/`, `serialization/`, `evaluation/`, `audit`, or
  `config`, add/extend the matching `tests/test_*.py` and keep the suite green.

## Data

`data/raw/` and `data/processed/` are git-ignored (only `.gitkeep` is tracked).
METLIN SMRT provides no raw chromatograms — RT-density targets are synthetic
(Gaussian) by construction. Never commit datasets, checkpoints, adapters, or
notebook outputs (`.gitignore` already excludes `*.csv/*.parquet/*.safetensors`,
`runs/`, `checkpoints/`, `diffusiongemma_lora/`, `wandb/`).

## Git workflow

- Active development branch: `claude/claude-md-docs-bkoibx` (do not push to `main`
  without explicit permission).
- Use `git push -u origin <branch>`; do not open a PR unless explicitly asked.
- Notebook URLs are pinned to a feature branch — update them when `main` changes.
