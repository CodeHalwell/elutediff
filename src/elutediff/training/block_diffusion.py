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


def _digit_id_value_tensors(processor, scale, dev):
    """Token ids for the single-digit levels ``0..scale`` and their float values."""
    import torch

    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(scale + 1)]
    return (torch.tensor(ids, device=dev),
            torch.arange(scale + 1, device=dev, dtype=torch.float32))


def _peak_loss(logits, x0, digit_ids, values, encoding, mode):
    """Differentiable peak-aware loss on the soft density decoded from logits.

    Restricts to the bin positions (where the clean target is a digit token),
    forms the expected emitted level per bin from a softmax over the digit tokens,
    decodes to a PDF (first-difference for the CDF encoding), then scores the
    predicted vs true PDF by 1-D Wasserstein (``emd``) or soft-argmax MSE.
    """
    import torch

    is_bin = torch.isin(x0, digit_ids)
    if int(is_bin.sum()) < 2:
        return logits.new_zeros(())
    lb = logits[is_bin][:, digit_ids]          # [B, scale+1]
    p = torch.softmax(lb, dim=-1)
    pred_emit = (p * values).sum(-1)           # expected emitted level per bin
    onehot = (x0[is_bin][:, None] == digit_ids[None, :]).to(values.dtype)
    true_emit = (onehot * values).sum(-1)
    if encoding == "cdf":
        pred_d = torch.relu(pred_emit[1:] - pred_emit[:-1])
        true_d = torch.relu(true_emit[1:] - true_emit[:-1])
    else:
        pred_d, true_d = pred_emit, true_emit
    eps = 1e-8
    pn = pred_d / (pred_d.sum() + eps)
    tn = true_d / (true_d.sum() + eps)
    if mode == "emd":
        return torch.abs(torch.cumsum(pn, 0) - torch.cumsum(tn, 0)).sum()
    idx = torch.arange(pn.numel(), device=pn.device, dtype=pn.dtype)
    return ((pn * idx).sum() - (tn * idx).sum()) ** 2


def train(model, examples, model_cfg: ModelConfig, train_cfg: TrainConfig,
          processor=None, target_cfg=None, start_step=0, on_checkpoint=None,
          checkpoint_every=0):
    """Run the block-diffusion fine-tuning loop. Returns the trained ``model``.

    When ``train_cfg.peak_loss`` is enabled, ``processor`` and ``target_cfg`` are
    required (to map digit tokens to levels and read the encoding); the auxiliary
    ``peak_lambda * peak_loss`` term is added to the denoising cross-entropy.

    Resumable / observable run support (used by ``scripts/train_diffusion.py`` to
    survive preemption and log a checkpoint/eval curve):

    * ``start_step`` -- resume from this optimizer step. The OneCycle schedule is
      built for the full ``train_cfg.steps`` and fast-forwarded ``start_step``
      times so the LR continues from where a killed run left off (inject the
      adapter weights before calling; optimizer moments restart, as in a warm
      resume). The loop then runs ``start_step+1 .. steps``.
    * ``on_checkpoint(step, model)`` -- called every ``checkpoint_every`` steps
      (and once at the final step) so the caller can save/push the adapter and
      run a held-out eval. No-op when either argument is unset.
    """
    import torch

    canvas_len = model_cfg.canvas_length
    vocab = model.config.text_config.vocab_size
    dev = next(
        (p.device for p in model.parameters() if p.device.type != "meta"),
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    peak_mode = getattr(train_cfg, "peak_loss", "none")
    digit_ids = values = encoding = None
    if peak_mode != "none":
        if processor is None or target_cfg is None:
            raise ValueError("peak_loss requires processor and target_cfg")
        digit_ids, values = _digit_id_value_tensors(processor, target_cfg.scale, dev)
        encoding = getattr(target_cfg, "encoding", "density")

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
    # Resume: fast-forward the LR schedule to `start_step` so a preempted run
    # continues from the same point on the cosine (not a fresh warmup). Stepping
    # the scheduler before the optimizer is intentional here (we are only
    # advancing the LR), so silence PyTorch's order warning for this block.
    if start_step > 0:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Detected call of")
            for _ in range(min(start_step, train_cfg.steps)):
                sched.step()

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
    for step in range(start_step + 1, train_cfg.steps + 1):
        step_loss = 0.0
        step_peak = 0.0
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
            x0d = x0.to(dev)
            m = lm.to(dev)
            ce = torch.nn.functional.cross_entropy(logits[m], x0d[m])
            loss = ce
            if peak_mode != "none":
                pk = _peak_loss(logits, x0d, digit_ids, values, encoding, peak_mode)
                loss = ce + train_cfg.peak_lambda * pk
                step_peak += float(pk) / train_cfg.grad_accum
            (loss / train_cfg.grad_accum).backward()
            step_loss += loss.item() / train_cfg.grad_accum
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], train_cfg.grad_clip
        )
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 20 == 0:
            extra = f" | peak {step_peak:.3f}" if peak_mode != "none" else ""
            print(f"step {step:4d}/{train_cfg.steps} | loss {step_loss:.4f}{extra} "
                  f"| lr {sched.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s", flush=True)
        if (on_checkpoint is not None and checkpoint_every
                and step % checkpoint_every == 0 and step != train_cfg.steps):
            on_checkpoint(step, model)
    if on_checkpoint is not None:
        on_checkpoint(train_cfg.steps, model)   # always checkpoint/eval the final step
    return model


def _eos_id(processor) -> int:
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    eos = getattr(tok, "eos_token_id", None) or 1
    if isinstance(eos, (list, tuple, set)):
        return next(iter(eos)) if eos else 1
    return eos
