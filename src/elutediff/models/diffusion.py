"""Load DiffusionGemma 26B-A4B via Unsloth and attach LoRA adapters.

Follows the reference Unsloth DiffusionGemma (26B-A4B) Sudoku notebook. The
heavy imports (``unsloth``, ``torch``) are deferred into the functions so this
module is importable on a CPU-only box for the tokenization-audit stages.

Hardware note: the model is ~52GB in bf16 (the 128 MoE experts alone are ~46GB
and stay bf16, so 4-bit cannot shrink them). Use a >= ~50GB GPU (A100 80GB /
H100 / B200); a 40GB GPU offloads weights and becomes impractically slow.
"""

from __future__ import annotations

from elutediff.config import ModelConfig


def load_model(cfg: ModelConfig, hf_token: str | None = None):
    """Load the DiffusionGemma model + processor via Unsloth ``FastModel``.

    Returns ``(model, processor)``. ``FastModel`` auto-detects the diffusion
    ``model_type`` and routes to the transformers-only ``FastDiffusionModel``
    slow path. The processor bundles the chat template and tokenizer.
    """
    import torch
    from unsloth import FastModel

    if torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        if free_gb < 50:
            print(
                f"[warn] {free_gb:.0f}GB free < ~52GB needed: weights will offload "
                "(meta/CPU) and run very slowly. Use an 80GB GPU (A100 80GB / H100)."
            )

    model, processor = FastModel.from_pretrained(
        model_name=cfg.model_name,
        dtype=torch.bfloat16,
        load_in_4bit=cfg.load_in_4bit,
        token=hf_token,
    )
    return model, processor


def add_lora(model, cfg: ModelConfig):
    """Attach LoRA adapters to the shared Gemma-4 backbone (MoE experts stay frozen).

    Targets attention + dense MLP; trains ~0.5% of parameters at r=64.
    """
    from unsloth import FastModel

    return FastModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
    )


def model_dimensions(model) -> dict:
    """Read vocab size and canvas length off the loaded model config."""
    return {
        "vocab": model.config.text_config.vocab_size,
        "canvas_length": model.config.canvas_length,
    }
