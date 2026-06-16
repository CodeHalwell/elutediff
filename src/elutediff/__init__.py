"""elutediff: molecule-conditioned retention-time density generation.

Reframes RT prediction on METLIN SMRT as conditioned RT-*density* generation
(a fixed-length Gaussian peak over a quantized time axis) and fine-tunes the
DiffusionGemma 26B-A4B block-diffusion model (via Unsloth + LoRA) to generate it.

See ``docs/density-first-revision.md`` for the scientific proposal and
``docs/project-structure.md`` for how this package is organized.
"""

__version__ = "0.0.1"
