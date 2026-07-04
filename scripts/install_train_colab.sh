#!/usr/bin/env bash
# Exact DiffusionGemma dependency set from the reference Unsloth notebook
# (DiffusionGemma_(26B-A4B)-Sudoku.ipynb, Colab branch). The `--no-deps`
# everywhere is load-bearing: it stops pip from resolving into an incompatible
# unsloth / trl / transformers / datasets combo (a plain `pip install .[train]`
# pulls a PyPI unsloth too old for this brand-new gemma4-diffusion model, plus
# trl>=1 that dropped ConstantLengthDataset and datasets 5.x -- every failure we
# hit). unsloth + unsloth-zoo come from git main because no PyPI release yet
# supports the model. transformers is pinned to 5.11.0 (ships the DiffusionGemma
# classes) and installed --no-deps so it doesn't drag the rest out of alignment.
#
# Assumes torch is already present (Colab images ship it); reads its version to
# choose the matching xformers wheel. Run AFTER `pip install -e '.[chem]'`.
set -euo pipefail

V=$(python -c "import torch,re; print(re.match(r'[0-9]+\.[0-9]+', torch.__version__).group(0))")
case "$V" in
  2.10) XF="xformers==0.0.34" ;;
  2.9)  XF="xformers==0.0.33.post1" ;;
  2.8)  XF="xformers==0.0.32.post2" ;;
  *)    XF="xformers==0.0.34" ;;
esac
echo "[install_train_colab] torch $V -> $XF"

pip install -q sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
pip install -q --no-deps bitsandbytes accelerate "$XF" peft trl triton
pip install -q --no-deps --upgrade \
    git+https://github.com/unslothai/unsloth-zoo.git \
    git+https://github.com/unslothai/unsloth.git
pip install -q --no-deps --upgrade "torchao>=0.16.0"
pip install -q --no-deps transformers==5.11.0 "tokenizers>=0.22.0,<=0.23.0"

python -c "import torch; torch._dynamo.config.recompile_limit = 64" 2>/dev/null || true
echo "[install_train_colab] done"
