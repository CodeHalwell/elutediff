"""Build user prompts (molecule conditioning) and render the RT-vector target.

The *target* placed on the denoising canvas is kept header-free (just the
space-separated fixed-width tokens) to conserve the 256-token budget. The
human-readable ``<RT_VECTOR ...>`` wrapper from the proposal is used only for
logging / inspection via :func:`format_rt_vector`.
"""

from __future__ import annotations

from elutediff.config import ConditioningConfig, TargetConfig
from elutediff.targets.quantize import vector_to_tokens

PROMPT_HEADER = (
    "Generate the retention-time density vector for this molecule. "
    "Reply with {n_bins} space-separated 3-digit intensity tokens (000-100)."
)


def format_rt_vector(levels, cfg: TargetConfig) -> str:
    """Render a human-readable ``<RT_VECTOR ...>`` block for logs/inspection."""
    body = " ".join(vector_to_tokens(levels, cfg))
    return (
        f"<RT_VECTOR bw={cfg.bin_width:g}s max_rt={cfg.rt_max:g} "
        f"sigma={cfg.sigma:g} scale={cfg.scale}>\n{body}\n</RT_VECTOR>"
    )


def target_string(levels, cfg: TargetConfig) -> str:
    """The bare canvas target: space-separated fixed-width tokens, no header."""
    return " ".join(vector_to_tokens(levels, cfg))


def build_prompt(
    *,
    smiles: str,
    target_cfg: TargetConfig,
    cond_cfg: ConditioningConfig,
    descriptors: dict[str, float] | None = None,
    atom_bond_table: str | None = None,
    lappe: str | None = None,
) -> str:
    """Assemble a conditioning prompt at the configured representation level.

    Levels (proposal Section 6) are additive: each higher level keeps everything
    from the levels below it. Optional inputs are only appended when the level
    requests them *and* the data is supplied.
    """
    lines = [PROMPT_HEADER.format(n_bins=target_cfg.n_bins), f"smiles={smiles}"]

    if cond_cfg.level >= 2 and descriptors:
        desc = " ".join(f"{k}={descriptors[k]:g}" for k in cond_cfg.descriptors if k in descriptors)
        if desc:
            lines.append(f"descriptors: {desc}")
    if cond_cfg.level >= 3 and atom_bond_table:
        lines.append(f"atoms_bonds:\n{atom_bond_table}")
    if cond_cfg.level >= 4 and lappe:
        lines.append(f"lappe:\n{lappe}")

    return "\n".join(lines)
