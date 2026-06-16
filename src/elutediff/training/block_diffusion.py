"""Block-diffusion fine-tuning for DiffusionGemma (proposal Section 7).

DiffusionGemma is *not* trained autoregressively (no SFTTrainer). Instead we use
its own block-diffusion objective, ported from the Unsloth Sudoku notebook:

  1. Pad the target RT-vector string to the 256-token canvas.
  2. Corrupt the canvas: replace each token with prob ``t`` by a random token,
     where ``t ~ U(t_lo, 1.0)``.
  3. Ask the model to predict the clean canvas; cross-entropy on the target
     tokens (+ eos), ignoring the padding tail.

Heavy imports are deferred so this module is importable without torch.
"""

from __future__ import annotations

import random
import time

from elutediff.config import ModelConfig, TrainConfig


def build_examples(rows, processor, model_cfg: ModelConfig):
    """Tokenize ``(prompt, target_string)`` rows into canvas-padded examples.

    Each row is a dict with ``prompt`` (molecule conditioning text) and
    ``target`` (bare space-separated RT-vector tokens). Returns a list of
    ``(prompt_ids, x0, loss_mask)`` tuples; examples whose target overflows the
    canvas are skipped (and should be surfaced by the tokenizer audit first).
    """
    import torch

    canvas_len = model_cfg.canvas_length
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    eos = _eos_id(processor)
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos

    out = []
    for r in rows:
        prompt_ids = processor.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )[0]
        ids = tok.encode(r["target"], add_special_tokens=False)
        content = ids + [eos]
        n = len(content)
        if n > canvas_len:
            continue
        x0 = torch.tensor(content + [pad] * (canvas_len - n), dtype=torch.long)
        mask = torch.zeros(canvas_len, dtype=torch.bool)
        mask[:n] = True
        out.append((prompt_ids, x0, mask))
    return out


def train(model, examples, model_cfg: ModelConfig, train_cfg: TrainConfig):
    """Run the block-diffusion fine-tuning loop. Returns the trained ``model``."""
    import torch

    canvas_len = model_cfg.canvas_length
    vocab = model.config.text_config.vocab_size
    dev = next(
        (p.device for p in model.parameters() if p.device.type != "meta"),
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Seed both RNGs used by the loop (Python `random` for noise level + shuffle,
    # torch for the corruption mask/token sampling) for reproducible runs.
    random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)

    model.config.use_cache = True
    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=train_cfg.weight_decay,
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=train_cfg.lr, total_steps=train_cfg.steps,
        pct_start=train_cfg.warmup_pct, anneal_strategy="cos",
    )

    def corrupt(x0):
        t = random.uniform(train_cfg.t_lo, 1.0)
        xt = x0.to(dev).clone()
        m = torch.rand(canvas_len, device=dev) < t
        xt[m] = torch.randint(0, vocab, (canvas_len,), device=dev)[m]
        return xt.unsqueeze(0)

    order = list(range(len(examples)))
    ptr = 0
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    for step in range(1, train_cfg.steps + 1):
        step_loss = 0.0
        for _ in range(train_cfg.grad_accum):
            if ptr >= len(order):
                random.shuffle(order)
                ptr = 0
            prompt_ids, x0, lm = examples[order[ptr]]
            ptr += 1
            out = model(
                input_ids=prompt_ids.unsqueeze(0).to(dev),
                canvas_ids=corrupt(x0),
                self_conditioning_logits=None,
            )
            logits = out.logits[0].float()
            m = lm.to(dev)
            loss = torch.nn.functional.cross_entropy(logits[m], x0.to(dev)[m])
            (loss / train_cfg.grad_accum).backward()
            step_loss += loss.item() / train_cfg.grad_accum
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], train_cfg.grad_clip
        )
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 20 == 0:
            print(f"step {step:4d}/{train_cfg.steps} | loss {step_loss:.4f} "
                  f"| {time.time() - t0:.0f}s", flush=True)
    return model


def _eos_id(processor) -> int:
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    eos = getattr(tok, "eos_token_id", None) or 1
    if isinstance(eos, (list, tuple, set)):
        return next(iter(eos)) if eos else 1
    return eos
