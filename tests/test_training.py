"""CPU tests for the resumable/observable training loop mechanics.

The GPU denoising can't run on CPU, but the loop's *control flow* -- which steps
fire ``on_checkpoint``, and that ``start_step`` resumes rather than restarts -- is
pure Python and is exactly the logic that must be correct for a preempted run to
resume without redoing or skipping work. A tiny fake model exercises it.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from elutediff.config import ModelConfig, TrainConfig  # noqa: E402
from elutediff.training.block_diffusion import train  # noqa: E402

CANVAS, VOCAB = 8, 20


class _FakeModel(nn.Module):
    """Minimal stand-in: its logits are a trainable parameter (inputs ignored)."""

    def __init__(self):
        super().__init__()
        self.logit_param = nn.Parameter(torch.randn(CANVAS, VOCAB))
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(vocab_size=VOCAB), use_cache=False
        )

    def forward(self, input_ids=None, canvas_ids=None, self_conditioning_logits=None):
        return SimpleNamespace(logits=self.logit_param.unsqueeze(0))


def _one_example():
    return [(
        torch.zeros(3, dtype=torch.long),                     # prompt ids (ignored)
        torch.randint(0, VOCAB, (CANVAS,), dtype=torch.long),  # x0 target
        torch.ones(CANVAS, dtype=torch.bool),                  # loss mask
    )]


def _cfgs(steps):
    # OneCycleLR needs a non-trivial total_steps (tiny values hit an internal
    # zero-width-phase division); the real runs use thousands of steps.
    return ModelConfig(canvas_length=CANVAS), TrainConfig(
        steps=steps, grad_accum=1, lr=1e-3, warmup_pct=0.1
    )


def test_checkpoint_fires_on_interval_and_final():
    mcfg, tcfg = _cfgs(50)
    seen = []
    train(_FakeModel(), _one_example(), mcfg, tcfg,
          on_checkpoint=lambda s, m: seen.append(s), checkpoint_every=20)
    # every 20th step, then always the final step (50 is not a multiple of 20).
    assert seen == [20, 40, 50]


def test_start_step_resumes_not_restarts():
    mcfg, tcfg = _cfgs(50)
    seen = []
    train(_FakeModel(), _one_example(), mcfg, tcfg, start_step=30,
          on_checkpoint=lambda s, m: seen.append(s), checkpoint_every=20)
    # loop runs 31..50 only -> interval hit at 40, plus the final 50. No re-run of 20.
    assert seen == [40, 50]


def test_no_hook_is_a_noop_and_trains():
    mcfg, tcfg = _cfgs(30)
    model = _FakeModel()
    before = model.logit_param.detach().clone()
    out = train(model, _one_example(), mcfg, tcfg)   # no checkpoint args
    assert out is model
    assert not torch.allclose(before, model.logit_param)  # weights actually moved
