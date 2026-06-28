"""Ablation sweep, idempotent + resumable. One base model, fresh LoRA per config.
Each invocation does setup-if-needed, then trains+evals the NEXT not-yet-done
config and appends to /content/sweep_results.json. Driver calls it once per
config; skips configs already in the results file (survives kernel restarts).
"""
import json, math, os, random, time
import numpy as np
import torch
from elutediff.config import Config
from elutediff.models.diffusion import load_model, add_lora, model_dimensions
from elutediff.data.metlin import load_metlin
from elutediff.data.splits import make_split
from elutediff.targets.density import gaussian_density, time_grid
from elutediff.targets.quantize import quantize
from elutediff.serialization.prompts import build_prompt, target_string
from elutediff.serialization.parser import parse_rt_vector, decoded_rt
from elutediff.training.block_diffusion import build_examples, _peak_loss, _digit_id_value_tensors
from elutediff.training.sampling import generate
from elutediff.evaluation.point_rt import point_rt_metrics, tolerance_hit_rate
from elutediff.evaluation.density import window_probability

CSV = "/content/metlin_smrt/SMRT_dataset.csv"
RESULTS = "/content/sweep_results.json"
STEPS = 1000
EVAL_AT = [500, 1000]
N_EVAL = 40
GRID = [
    ("density_none",        "density", "none",       0.0),
    ("cdf_none",            "cdf",     "none",       0.0),
    ("cdf_emd_0.01",        "cdf",     "emd",        0.01),
    ("cdf_emd_0.05",        "cdf",     "emd",        0.05),
    ("cdf_softargmax_0.05", "cdf",     "softargmax", 0.05),
    ("density_emd_0.05",    "density", "emd",        0.05),
]

if "SWEEP" not in globals():
    cfg = Config(); cfg.split.strategy = "scaffold"; cfg.train.steps = STEPS
    cfg.train.grad_accum = 1  # ~4x faster: fit whole sweep in one VM lifetime
    print("[sweep] loading base + LoRA template ...", flush=True)
    model, processor = load_model(cfg.model)
    model = add_lora(model, cfg.model)
    print("[sweep] dims", model_dimensions(model), flush=True)
    mols, stats = load_metlin(CSV, return_stats=True)
    print("[sweep]", stats, flush=True)
    split = make_split([m.smiles for m in mols], cfg.split)
    fold = {i: n for n, idx in (("train", split.train_idx), ("val", split.val_idx),
                                ("test", split.test_idx)) for i in idx}

    def build_examples_for(enc):
        cfg.target.encoding = enc
        tr, ev = [], []
        for i, m in enumerate(mols):
            lv = quantize(gaussian_density(m.rt_seconds, cfg.target), cfg.target)
            row = {"rt": m.rt_seconds,
                   "prompt": build_prompt(smiles=m.smiles, target_cfg=cfg.target, cond_cfg=cfg.conditioning),
                   "target": target_string(lv, cfg.target)}
            (tr if fold.get(i, "train") == "train" else ev).append(row) if fold.get(i, "train") in ("train", "test") else None
            if fold.get(i, "train") == "test":
                pass
        # simpler: split explicitly
        tr = [{"rt": m.rt_seconds, "prompt": build_prompt(smiles=m.smiles, target_cfg=cfg.target, cond_cfg=cfg.conditioning),
               "target": target_string(quantize(gaussian_density(m.rt_seconds, cfg.target), cfg.target), cfg.target)}
              for i, m in enumerate(mols) if fold.get(i, "train") == "train"]
        ev = [{"rt": m.rt_seconds, "prompt": build_prompt(smiles=m.smiles, target_cfg=cfg.target, cond_cfg=cfg.conditioning)}
              for i, m in enumerate(mols) if fold.get(i, "train") == "test"]
        return build_examples(tr, processor, cfg.model), ev

    print("[sweep] building density + cdf example sets ...", flush=True)
    dex, eval_rows = build_examples_for("density")
    cex, _ = build_examples_for("cdf")
    print(f"[sweep] density {len(dex)} | cdf {len(cex)} | test {len(eval_rows)}", flush=True)
    vocab = model.config.text_config.vocab_size
    dev = next((p.device for p in model.parameters() if p.device.type != "meta"),
               torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    digit_ids, values = _digit_id_value_tensors(processor, cfg.target.scale, dev)
    grid_t = time_grid(cfg.target)
    SWEEP = dict(cfg=cfg, model=model, processor=processor, dex=dex, cex=cex,
                 eval_rows=eval_rows, vocab=vocab, dev=dev, digit_ids=digit_ids,
                 values=values, grid_t=grid_t)
    print("SWEEP_SETUP_DONE", flush=True)

S = SWEEP; cfg = S["cfg"]; model = S["model"]; processor = S["processor"]
dev = S["dev"]; vocab = S["vocab"]; canvas = cfg.model.canvas_length


def reinit_lora():
    for _, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            for a in list(mod.lora_A.keys()):
                torch.nn.init.kaiming_uniform_(mod.lora_A[a].weight, a=math.sqrt(5))
                torch.nn.init.zeros_(mod.lora_B[a].weight)


def corrupt(x0):
    t = random.uniform(cfg.train.t_lo, 1.0); xt = x0.to(dev).clone()
    m = torch.rand(canvas, device=dev) < t
    xt[m] = torch.randint(0, vocab, (canvas,), device=dev)[m]; return xt.unsqueeze(0)


def do_eval(enc, n):
    cfg.target.encoding = enc; model.eval(); out = {}
    for steps in [1, 16]:
        yt, yp, valid, wp = [], [], 0, []
        for j, r in enumerate(S["eval_rows"][:n]):
            if j % 10 == 0:
                print(f"    eval {enc} {steps}-step {j}/{n}", flush=True)
            txt = generate(model, processor, r["prompt"], steps, canvas)
            pr = parse_rt_vector(txt, cfg.target)
            if not pr.ok:
                continue
            valid += 1; yt.append(r["rt"]); yp.append(decoded_rt(pr.levels, cfg.target))
            wp.append(window_probability(pr.levels, S["grid_t"], r["rt"], cfg.target.sigma))
        if yp:
            mm = point_rt_metrics(yt, yp); h = tolerance_hit_rate(yt, yp, cfg.eval.rt_tolerances_s)
            out[str(steps)] = {"valid": valid, "n": n, "mae": mm["mae"], "r2": mm["r2"],
                               "tolerance_hits": h, "window_prob": float(np.mean(wp))}
        else:
            out[str(steps)] = {"valid": 0, "n": n}
    model.train(); return out


results = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
todo = [g for g in GRID if g[0] not in results]
if not todo:
    print("SWEEP_ALL_DONE", flush=True)
else:
    name, enc, peak, lam = todo[0]
    print(f"=== [sweep] CONFIG {name} (enc={enc} peak={peak} lam={lam}) ===", flush=True)
    reinit_lora(); cfg.target.encoding = enc
    examples = S["cex"] if enc == "cdf" else S["dex"]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.train.lr, betas=(0.9, 0.95), weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.train.lr, total_steps=STEPS,
                                                pct_start=cfg.train.warmup_pct, anneal_strategy="cos")
    order = list(range(len(examples))); ptr = 0; step = 0; t0 = time.time()
    random.seed(cfg.train.seed); torch.manual_seed(cfg.train.seed); model.train()
    rec = {"encoding": enc, "peak": peak, "lambda": lam, "evals": {}}
    for tgt in EVAL_AT:
        opt.zero_grad(set_to_none=True)
        while step < tgt:
            step += 1
            for _ in range(cfg.train.grad_accum):
                if ptr >= len(order):
                    random.shuffle(order); ptr = 0
                pid, x0, lm = examples[order[ptr]]; ptr += 1
                o = model(input_ids=pid.unsqueeze(0).to(dev), canvas_ids=corrupt(x0), self_conditioning_logits=None)
                lg = o.logits[0].float(); x0d = x0.to(dev); mk = lm.to(dev)
                ce = torch.nn.functional.cross_entropy(lg[mk], x0d[mk])
                loss = ce if peak == "none" else ce + lam * _peak_loss(lg, x0d, S["digit_ids"], S["values"], enc, peak)
                (loss / cfg.train.grad_accum).backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.train.grad_clip)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if step % 20 == 0:
                print(f"  [{name}] step {step}/{STEPS} {time.time()-t0:.0f}s", flush=True)
        res = do_eval(enc, N_EVAL); rec["evals"][str(tgt)] = res
        one = res.get("1", {})
        print(f"  [{name}] @{tgt} 1-step valid {one.get('valid')}/{N_EVAL} MAE {one.get('mae')} "
              f"R2 {one.get('r2')} wp {one.get('window_prob')}", flush=True)
    results[name] = rec
    json.dump(results, open(RESULTS, "w"), indent=2)
    print(f"CONFIG_DONE {name} ({len(results)}/{len(GRID)})", flush=True)
