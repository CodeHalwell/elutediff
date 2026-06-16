# notebooks/

Exploratory notebooks (tokenizer audits, target visualization, qualitative
generation review).

The training mechanics follow the reference **Unsloth DiffusionGemma (26B-A4B)
Sudoku** notebook
([unslothai/notebooks](https://github.com/unslothai/notebooks/blob/main/nb/DiffusionGemma_(26B-A4B)-Sudoku.ipynb)):
`FastModel.from_pretrained` → `FastModel.get_peft_model` (LoRA) → the
block-diffusion corrupt-and-denoise loop → multi-step `generate`. In this project
that pattern lives in `src/elutediff/models/diffusion.py` and
`src/elutediff/training/`, with the Sudoku grid replaced by the RT-density vector.

Keep heavy outputs out of version control (`.ipynb_checkpoints/` and notebook
outputs are git-ignored).
