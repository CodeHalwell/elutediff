"""RT-density target construction: scalar RT -> fixed-width integer token vector."""

from elutediff.targets.density import gaussian_density, time_grid
from elutediff.targets.noise import apply_noise
from elutediff.targets.quantize import dequantize, quantize, vector_to_tokens, tokens_to_vector

__all__ = [
    "gaussian_density",
    "time_grid",
    "apply_noise",
    "quantize",
    "dequantize",
    "vector_to_tokens",
    "tokens_to_vector",
]
