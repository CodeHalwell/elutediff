# Project structure

`elutediff` fine-tunes **DiffusionGemma 26B-A4B** (block-diffusion MoE, via
**Unsloth** + LoRA) to generate **molecule-conditioned RT-density vectors** from
METLIN SMRT. The scientific rationale lives in
[`density-first-revision.md`](density-first-revision.md); this document maps the
proposal onto the code.

## Layout

```
elutediff/
├── pyproject.toml            # package + extras: chem / baselines / graph / train / dev
├── configs/                  # one YAML per ablation (base, b4, b6, b7, b8)
├── scripts/                  # runnable pipeline stages (CLI-friendly)
│   ├── download_metlin.py    # step 1: fetch METLIN SMRT
│   ├── build_targets.py      # step 2: (prompt, RT-density target) JSONL
│   ├── tokenizer_audit.py    # step 3: MANDATORY canvas-budget check
│   ├── train_baselines.py    # step 4: B1/B2 known-bar regressors + metrics
│   ├── train_gnn.py          # step 4: B3 GINE/graph-transformer (graph extra)
│   ├── train_diffusion.py    # step 5: Unsloth/LoRA block-diffusion fine-tune
│   └── evaluate.py           # step 8: denoising-step sweep + metrics
├── src/elutediff/
│   ├── config.py             # typed dataclasses + YAML loader
│   ├── cli.py                # `elutediff audit|build-targets|train|eval`
│   ├── audit.py              # tokenizer/canvas budget audit
│   ├── targets/              # scalar RT -> fixed-width integer density vector
│   │   ├── density.py        #   Gaussian density + time grid          [implemented]
│   │   ├── quantize.py       #   000..100 integer tokenization         [implemented]
│   │   └── noise.py          #   weak baseline augmentation (B7)        [implemented]
│   ├── serialization/        # canvas I/O
│   │   ├── prompts.py        #   conditioning prompts + target render   [implemented]
│   │   └── parser.py         #   strict parser + validity + RT decode   [implemented]
│   ├── data/                 # METLIN + RDKit featurization
│   │   ├── metlin.py         #   load/canonicalize records (SDF/CSV)     [implemented]
│   │   ├── molecules.py      #   descriptors, ECFP, atom/bond table      [implemented]
│   │   ├── splits.py         #   random / scaffold / Tanimoto cluster    [implemented]
│   │   └── graph_features.py #   LapPE (sign-canonicalized)              [implemented]
│   ├── models/
│   │   ├── diffusion.py      #   Unsloth FastModel + LoRA loaders       [wired]
│   │   ├── baselines.py      #   B1/B2 ECFP/MLP + conformal/ensemble     [implemented]
│   │   └── gnn.py            #   B3 GINE/graph-transformer (graph extra) [wired]
│   ├── training/
│   │   ├── block_diffusion.py#   corrupt-canvas objective + train loop  [wired]
│   │   └── sampling.py        #   multi-step denoising generation        [wired]
│   └── evaluation/           # metric families (Section 9)
│       ├── point_rt.py       #   MAE/RMSE/R2/tolerance hits             [implemented]
│       ├── density.py        #   KL/JS/EMD/CRPS/window-prob             [implemented]
│       ├── uncertainty.py    #   coverage/width/ECE (B9)                [implemented]
│       └── ranking.py        #   top-k / MRR / filter@recall            [implemented]
└── tests/                    # cover the implemented CPU-only core
```

**Implemented** = full + unit-tested on CPU (numpy core; the `data/` modules use
RDKit from the `chem` extra). **Wired** = real
DiffusionGemma/Unsloth code ported from the reference Sudoku notebook, runnable
once the `train` extra and a GPU are present. **Scaffold** = typed interface +
docstring; needs RDKit / GNN deps (roadmap steps 1, 4, 7).

## Why these dependencies are split

The tokenization-audit and target-construction stages (roadmap steps 1-3) must
de-risk the design *before* any expensive fine-tuning, so the core package only
needs numpy/scipy/pandas/pyyaml. RDKit (`chem`), classical/GNN baselines
(`baselines`/`graph`), and Unsloth (`train`) are opt-in extras.

## Roadmap → code

| Step | Proposal item | Where |
|------|---------------|-------|
| 1 | METLIN loader, canonicalization, descriptors, ECFP, splits | `data/` (done; LapPE pending) |
| 2 | Clean + noisy RT-density target generators | `targets/`, `scripts/build_targets.py` |
| 3 | Tokenizer-length audit; lock target format | `audit.py`, `scripts/tokenizer_audit.py` |
| 4 | Scalar + graph baselines | `models/baselines.py` (B1/B2), `models/gnn.py` + `scripts/train_gnn.py` (B3), `scripts/train_baselines.py` |
| 5 | Fine-tune center-bin / parameter / clean-vector | `models/diffusion.py`, `training/`, `scripts/train_diffusion.py` |
| 6 | Noisy-vector augmentation | `configs/b7_noisy_vector.yaml` |
| 7 | LapPE conditioning + split evaluation | `data/graph_features.py` (done), `configs/b8_lappe.yaml` |
| 8 | Uncertainty + candidate-ranking studies | `evaluation/uncertainty.py`, `evaluation/ranking.py` |

## Quickstart

```bash
pip install -e ".[dev]"          # core + tests
pytest                           # run the unit suite
elutediff -c configs/b6_clean_vector.yaml audit   # canvas budget check

# Later, on a >= ~50GB GPU:
pip install -e ".[chem,baselines,train]"
python scripts/build_targets.py -c configs/b6_clean_vector.yaml
python scripts/train_diffusion.py -c configs/b6_clean_vector.yaml --data data/processed/targets.jsonl
python scripts/evaluate.py -c configs/b6_clean_vector.yaml --data data/processed/targets.jsonl --adapter runs/b6_clean_vector
```

## Hardware

DiffusionGemma 26B-A4B is ~52 GB in bf16 (the 128 MoE experts alone are ~46 GB
and stay bf16, so 4-bit cannot shrink them). Use a ≥ ~50 GB GPU (A100 80 GB /
H100 / B200); a 40 GB GPU offloads weights and becomes impractically slow. CPU is
fine for everything in `targets/`, `serialization/`, `evaluation/`, and `audit.py`.
