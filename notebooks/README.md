# notebooks/

Google Colab–ready notebooks. Each one `pip install`s `elutediff` straight from
GitHub, so they run standalone — no local checkout needed. The data/baseline
notebooks run on a **free CPU runtime**; the fine-tune notebook needs an **A100
80GB / H100**.

Open in Colab (branch `claude/project-structure-unsloth-bnwe57`):

| Notebook | Runtime | What it does |
|----------|---------|--------------|
| [`01_data_and_targets.ipynb`](https://colab.research.google.com/github/CodeHalwell/elutediff/blob/claude/project-structure-unsloth-bnwe57/notebooks/01_data_and_targets.ipynb) | CPU | Load METLIN (or a synthetic stand-in), audit the 256-token canvas, build & visualize the Gaussian RT-density targets, write the `(prompt, target)` JSONL + splits. |
| [`02_baselines.ipynb`](https://colab.research.google.com/github/CodeHalwell/elutediff/blob/claude/project-structure-unsloth-bnwe57/notebooks/02_baselines.ipynb) | CPU | Train the B1/B2 known-bar regressors (RF / XGBoost / MLP over ECFP + descriptors) and split-conformal intervals (B9). |
| [`03_diffusiongemma_finetune.ipynb`](https://colab.research.google.com/github/CodeHalwell/elutediff/blob/claude/project-structure-unsloth-bnwe57/notebooks/03_diffusiongemma_finetune.ipynb) | A100/H100 | Load DiffusionGemma 26B-A4B via Unsloth, attach LoRA, run the block-diffusion fine-tune on RT vectors, and sweep denoising steps at eval. |

> When `main` is updated, swap the branch in the URLs (or use the badge from the
> repo root) so Colab pulls the merged version.

The synthetic demo data lets every notebook run out of the box; set the
`METLIN_PATH` / `DATA` variables to your real dataset to use METLIN SMRT. For the
real download see `scripts/download_metlin.py` (figshare DOI
`10.6084/m9.figshare.8038913`).

## Relationship to the package

The notebooks are thin drivers over `elutediff`; the heavy lifting lives in the
package so the same code is unit-tested and reusable. The fine-tune notebook
follows Unsloth's reference **DiffusionGemma (26B-A4B) Sudoku** notebook
([unslothai/notebooks](https://github.com/unslothai/notebooks/blob/main/nb/DiffusionGemma_(26B-A4B)-Sudoku.ipynb)),
swapping the Sudoku grid for the RT-density vector; that pattern lives in
`src/elutediff/models/diffusion.py` and `src/elutediff/training/`.

Keep heavy outputs out of version control (`.ipynb_checkpoints/` and notebook
outputs are git-ignored).
