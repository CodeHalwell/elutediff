"""Multi-step denoising generation (proposal Section 9: "refinement is the point").

Sweeping the number of denoising steps is a core experiment: one-shot generation
is weak, but letting the model revise the canvas over steps is where bidirectional
diffusion earns its place. The same sweep also feeds the diffusion-benefit metric
(1-step vs 8/16/48-step) and sampling-based uncertainty.
"""

from __future__ import annotations

import copy


def generate(model, processor, prompt: str, steps: int, canvas_len: int, seed: int = 0) -> str:
    """Generate one RT-vector completion with ``steps`` denoising iterations.

    Returns the decoded completion text (prompt stripped); parse it with
    :func:`elutediff.serialization.parser.parse_rt_vector`.
    """
    import torch

    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    dev = next(
        (p.device for p in model.parameters() if p.device.type != "meta"),
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    ids = processor.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(dev)

    gc = copy.deepcopy(model.generation_config)
    gc.max_denoising_steps = steps
    gc.max_new_tokens = canvas_len
    torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        out = model.generate(input_ids=ids, generation_config=gc)
    # Some transformers/unsloth versions return a bare tensor instead of an
    # object with `.sequences`; handle both.
    seq = out.sequences[0, ids.shape[1]:] if hasattr(out, "sequences") else out[0, ids.shape[1]:]
    return tok.decode(seq.tolist(), skip_special_tokens=True)


def sample_many(model, processor, prompt: str, steps: int, canvas_len: int, n: int) -> list[str]:
    """Draw ``n`` samples (varying seed) for sampling-based uncertainty (B9)."""
    return [generate(model, processor, prompt, steps, canvas_len, seed=s) for s in range(n)]
