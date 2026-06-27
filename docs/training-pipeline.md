# How training works, end to end

A precise, code-grounded walk-through of what happens to a single METLIN molecule
as it becomes a training signal — from SMILES string to a gradient update — and
how generation (inference) runs the same machinery in reverse. Function and file
names point at the real implementation so this doubles as a code map.

> **Mental model.** DiffusionGemma is **not** an autoregressive next-token
> predictor here. It is a *block-diffusion* (discrete text diffusion) model: it
> sees a **corrupted** copy of the whole answer and learns to **denoise it in one
> shot**, conditioned on a prompt. Training = "given the molecule and a noisy
> density canvas, reconstruct the clean density." Inference = "given the molecule
> and a fully-noised canvas, denoise it over N steps." There is no left-to-right
> generation; every canvas position is predicted simultaneously and
> bidirectionally.

---

## 1. The two halves of one training example

Every example is a pair: a **condition** (what the model is told) and a **target
canvas** (what it must produce).

```
   condition  =  prompt text  ->  chat-templated token ids   (input_ids)
   target     =  RT density   ->  fixed-length token canvas   (x0)
```

`build_examples()` (`training/block_diffusion.py`) produces a tuple
`(prompt_ids, x0, loss_mask)` per molecule. Everything below is how those three
tensors are built and used.

---

## 2. From a retention time to a target canvas

The label in METLIN is a single scalar: a retention time `r` in seconds. We turn
that scalar into a fixed-length 1-D **density** over a quantized time axis. Four
stages (`targets/density.py`, `targets/quantize.py`, `serialization/prompts.py`):

### 2a. Scalar RT -> Gaussian density  (`gaussian_density`)
A time grid of `n_bins` bin-centers spans `[rt_min, rt_max)` (default 0–1200 s,
10 s bins -> **120 bins**). The RT becomes a Gaussian bump on that grid:

```
g(t_j) = exp(-0.5 * ((t_j - r) / sigma)^2)        # sigma = 20 s ~ 2 bins
```

then **max-normalized so the peak = 1.0**. So `g` is a length-120 float vector,
mostly ~0 with a small bump (~10–15 non-zero bins) centered on the true RT.

### 2b. Density -> integer levels  (`quantize`)
Each bin is rounded to an integer level `0..scale`:

```
levels = round(clip(g, 0, 1) * scale)             # scale = 9  ->  digits 0..9
```

With `scale=9` every bin is a single digit `0..9`. (Why single-digit: the Gemma
tokenizer splits each digit into its own token, so multi-digit levels would blow
the 256-token canvas — see `docs`/the `audit` command. One digit = one bin = one
token.)

### 2c. Levels -> emitted tokens  (`density_to_emitted`, `vector_to_tokens`)
This is where the **target encoding** (experiment arm) lives:

- **`encoding="density"`** (baseline): emit the PDF levels directly. ~88% of the
  tokens are `0` (the flat background).
- **`encoding="cdf"`**: emit the *cumulative* density rescaled to `0..scale` — a
  monotone thermometer ramp `0 0 0 1 3 6 8 9 9 9 …`. Every bin is informative and
  the peak is the steepest part of the ramp. (Recovered by first-differencing on
  parse.)

### 2d. Tokens -> string  (`target_string`)
The levels render as a space-separated string of fixed-width digits:

```
target_string  =  "0 0 0 0 2 5 8 9 7 4 1 0 0 …"      # 120 fields
```

No header, no wrapper — just the bare canvas, to conserve the 256-token budget.

---

## 3. From a SMILES to the conditioning prompt  (`build_prompt`)

The molecule is serialized into a text prompt at the configured *conditioning
level* (1 = SMILES only … 5 = + graph embedding):

```
Generate the retention-time density vector for this molecule. Reply with 120
space-separated 1-digit intensity tokens (0-9).
smiles=O=C(O)c1ccccc1
descriptors: MolWt=122.12 LogP=1.87 …        # only at level >= 2
```

The instruction header is generated from the target config (`_header`) so it
always matches the expected token count and range. **The SMILES itself is just
text** — the model reads it through its normal tokenizer; there is no graph
encoder in the base path. Whatever chemistry the frozen 26B backbone knows about
that string is what conditions the density.

---

## 4. Tokenizing into a training example  (`build_examples`)

```python
prompt_ids = processor.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=True, add_generation_prompt=True, return_tensors="pt")[0]

ids   = tok.encode(target_string, add_special_tokens=False)   # the digit/space canvas
content = ids + [eos]
x0    = content + [pad] * (canvas_len - len(content))          # pad to 256
mask  = [True]*len(content) + [False]*(canvas_len - len(content))
```

Key shapes and facts:

- **`prompt_ids`** — the chat-templated condition (`<start_of_turn>user … <end_of_turn>`
  + generation prompt). Variable length; lives on the input side.
- **`x0`** — the **clean target canvas**, length `canvas_len = 256`. Note the
  content is **per character**: each bin contributes a *digit token* and (between
  bins) a *space token*, so 120 bins ≈ **240 content tokens** (digits + spaces) +
  1 `eos`, then ~16 `pad` tokens to fill 256.
- **`loss_mask`** — `True` over the 240+eos content tokens, `False` over the pad
  tail. Loss is only computed where the mask is `True`.
- Examples whose target would exceed 256 tokens are **dropped** (the canvas-budget
  guarantee; with single-digit levels this never triggers at 120 bins).

So one example = `(prompt_ids [Lp], x0 [256], mask [256])`.

---

## 5. One training step, in full  (`train`)

A "step" is one optimizer update, made of `grad_accum` (=4) micro-batches. Each
micro-batch processes **one** example like this:

### 5a. Corrupt the canvas  (`corrupt`)
Pick a noise level `t ~ Uniform(t_lo, 1.0)` (`t_lo = 0.1`). Then independently for
every one of the 256 canvas positions, with probability `t`, replace the clean
token with a **uniformly random token from the whole 262 144 vocabulary**:

```
xt = x0.clone()
flip = rand(256) < t
xt[flip] = randint(0, vocab)[flip]               # uniform-corruption diffusion
```

At `t≈0.1` only ~10% of positions are scrambled (easy denoising); at `t≈1.0`
almost the entire canvas is random noise (hard). Sampling `t` per step is what
makes the model robust across the whole noise schedule.

### 5b. Forward pass
The model is given the **condition** and the **noisy canvas** together:

```python
out = model(input_ids=prompt_ids,        # the SMILES prompt (condition)
            canvas_ids=xt,               # the corrupted 256-token canvas
            self_conditioning_logits=None)
logits = out.logits[0]                    # [256, 262144] — a distribution per position
```

`DiffusionGemmaForBlockDiffusion` attends **bidirectionally** over the whole
canvas (every position sees every other) and predicts, for each canvas position,
a distribution over the vocabulary for what the *clean* token should be.

### 5c. Loss — "predict the clean canvas"
Cross-entropy between the predicted distribution and the true clean tokens, **only
over the masked (content) positions** (padding is ignored):

```python
ce = cross_entropy(logits[mask], x0[mask])
```

This is the standard discrete-diffusion denoising objective: reconstruct `x0` from
`xt`. Because ~88% of the (density-encoded) content tokens are `0`/space, this CE
is dominated by the trivial background — which is exactly why the **peak-aware
loss** and the **CDF encoding** exist (see §7).

### 5d. Backward + accumulate
```python
(ce / grad_accum).backward()              # accumulate grads over 4 micro-batches
```
Only the LoRA adapter params require grad (~0.58% of the model, ~150 M params);
the 26 B base weights are frozen.

### 5e. Optimizer update (once per step, after grad_accum micro-batches)
```python
clip_grad_norm_(lora_params, grad_clip=1.0)
opt.step()                                 # AdamW, betas (0.9, 0.95)
sched.step()                               # OneCycleLR, cosine anneal
opt.zero_grad()
```

The LR follows a **one-cycle cosine** schedule: warm up over the first ~3% of
steps to `max_lr=1e-4`, then cosine-anneal toward ~0 across the rest. The
scheduler is sized to the full `steps`, so it steps exactly once per optimizer
update.

That is one complete step. The printed line is e.g.
`step 20/4000 | loss 0.94 | lr 1.05e-05 | 39s`.

---

## 6. Many steps — the training loop

```
for step in 1..steps:
    for _ in range(grad_accum):           # 4 micro-batches
        take next example (reshuffle when the epoch pointer wraps)
        corrupt -> forward -> CE -> backward
    clip -> opt.step -> sched.step -> zero_grad
```

- Examples are consumed in a shuffled order; when the pointer passes the end of
  the list it reshuffles (a fresh "epoch").
- Each step therefore sees 4 molecules at 4 *independently sampled* noise levels.
- With 63 950 training molecules and 4000 steps × 4 = 16 000 example-views, the
  model sees only ~25% of the data once — this is few-shot adaptation of a frozen
  backbone, not full training.

**What the model is actually learning:** a LoRA steering of the frozen backbone
such that, conditioned on the SMILES prompt, the denoiser reconstructs the
molecule's specific density canvas — i.e. places the peak at the right bin. The
hard part is that the peak is a tiny fraction of the canvas (§7).

---

## 7. How the two experiment arms change the step

Both are config-gated and default off (baseline = `encoding="density"`,
`peak_loss="none"`).

### Arm 1 — CDF dense target (`TargetConfig.encoding="cdf"`)
Changes **§2c only**. The target becomes the cumulative ramp, so the per-position
CE in §5c now rewards getting the *transition point* (the peak location) right: a
misplaced peak flips a whole run of tokens from `0` to `9` and costs a lot of CE.
This is the "make it dense like Sudoku" fix — same loss, better-aligned target.
Nothing else in the loop changes; the parser differences the ramp back to a PDF.

### Arm 2 — peak-aware auxiliary loss (`TrainConfig.peak_loss`, `peak_lambda`)
Adds a term to **§5c**. From `logits`, the soft expected level at each *bin*
position (positions where `x0` is a digit token) is decoded differentiably:

```
p_b      = softmax(logits[bin_b, digit_token_ids])      # over the 10 digits
pred_b   = sum_d d * p_b[d]                              # expected level (soft)
pred_pdf = pred (density) or diff(pred) (cdf)
```

Then either:
- **`emd`** — 1-D Wasserstein `sum |CDF(pred_pdf) - CDF(true_pdf)|` (penalizes mass
  by *distance* from the true location), or
- **`softargmax`** — MSE between the expected peak bin `sum_b b * pred_pdf_b` and
  the true peak.

The total loss is `CE + peak_lambda * peak_loss`. CE is always kept so the model
still learns to emit *valid* densities; the peak term forces it to spend capacity
on the ~12% of tokens that actually set the RT. The two arms compose.

---

## 8. Inference — multi-step denoising  (`generate`)

At eval/inference there is no clean canvas. The model starts from noise and
**iteratively denoises** for `steps` iterations:

```python
ids = chat_template(prompt)                    # same conditioning as training
gc.max_denoising_steps = steps                 # 1, 16, 64, …
out = model.generate(input_ids=ids, generation_config=gc)
text = decode(out.sequences[0, len(ids):])     # the generated canvas, prompt stripped
```

The denoising loop lives inside `DiffusionGemmaForBlockDiffusion.generate`. Each
of the `steps` iterations re-runs the bidirectional forward pass and refines the
canvas (remasking/replacing the least-confident positions and re-predicting):

- **1 step** = a single shot: the model denoises the fully-noised canvas once. Weak.
- **N steps** = the model revises its own draft N times, each pass sharpening the
  density. "Refinement is the point" — multi-step revision is where bidirectional
  diffusion should beat one-shot, and the **1-step vs 16/64-step gap is itself a
  reported metric** (the diffusion-benefit signal).

A telling failure mode: if 1-step and N-step give *identical* metrics, the model
is emitting the same canvas regardless of input/refinement — i.e. it collapsed to
a constant, input-independent output (what the token-CE control does early on).

---

## 9. Decoding a generation back to a retention time  (`parser`, `evaluation`)

The generated text is turned back into a number and scored:

1. **`parse_rt_vector`** — strictly extract the digit tokens, check there are
   exactly `n_bins` of them in range, then **decode the encoding** back to a PDF
   (`emitted_to_density`: identity for `density`, first-difference for `cdf`).
   Returns `levels` (the density) or a parse failure.
2. **`decoded_rt`** — collapse the density to one scalar RT: `argmax` mode returns
   the center time of the peak bin; `centroid` mode returns the
   intensity-weighted mean.
3. **Metrics** (`evaluation/`):
   - **MAE / R²** of `decoded_rt` vs the true RT (point accuracy).
   - **tolerance-hit rate** — fraction within ±15/30/60 s.
   - **window-probability** — integrated density mass within a window around the
     true RT (rewards a well-placed, well-shaped density, not just the argmax).
   - **vector validity** (`validity_report`) — length, range, single-dominant-peak,
     smoothness.

---

## 10. One worked example (single-digit density encoding)

```
RT label:          730.0 s
gaussian_density:  [0, …, 0, 0.21, 0.61, 0.95, 0.88, 0.47, 0.15, …, 0]   (len 120, peak=1 at bin 73)
quantize (×9):     [0, …, 0, 2, 5, 9, 8, 4, 1, …, 0]                      (len 120, digits)
target_string:     "0 0 … 0 2 5 9 8 4 1 0 … 0"                           (120 fields)
tok.encode:        [id0, idsp, id0, idsp, …]                              (~239 tokens)
x0 (canvas):       [ …239 content ids…, eos, pad, pad, … ]               (len 256)
loss_mask:         [ True ×240, False ×16 ]

prompt:            "Generate the retention-time density vector … smiles=…"
prompt_ids:        chat-templated condition

--- one training micro-step ---
t ~ U(0.1,1)  ->  corrupt ~ t·256 random positions of x0  ->  xt
logits = model(input_ids=prompt_ids, canvas_ids=xt)        # [256, 262144]
loss   = cross_entropy(logits[mask], x0[mask])             # reconstruct the clean canvas
loss.backward(); (every 4) opt.step(); sched.step()

--- inference ---
generate(prompt, steps=16)  ->  "0 0 … 0 2 5 9 8 4 1 0 … 0"
parse -> density -> argmax bin 73 -> grid[73] ≈ 735 s  ->  |735 - 730| = 5 s error
```

---

## File map

| Stage | Code |
|-------|------|
| Gaussian density, time grid | `targets/density.py` |
| Quantize, encode (density/cdf), tokens | `targets/quantize.py` |
| Prompt + target string | `serialization/prompts.py` |
| Tokenize into examples | `training/block_diffusion.py::build_examples` |
| Corruption, forward, loss, optimizer | `training/block_diffusion.py::train` |
| Peak-aware loss (arm 2) | `training/block_diffusion.py::_peak_loss` |
| Multi-step denoising (inference) | `training/sampling.py::generate` |
| Parse + decode RT | `serialization/parser.py` |
| Metrics | `evaluation/point_rt.py`, `evaluation/density.py` |
| Model load + LoRA | `models/diffusion.py` |
| Config knobs | `config.py` (`TargetConfig`, `TrainConfig`) |
