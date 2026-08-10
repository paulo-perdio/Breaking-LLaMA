"""
Structured weight corruption for LLaMA-3.2-1B-Instruct.

Implements the perturbation method from:
  Chawuthai, R., Thongsawaeng, A., Perdio, J.P.L., Zaw, K.K., Kraichoke, P.,
  Nwe, H.M., & Kertkeidkachorn, N. (2025). "Assessing the Effects of Corrupted
  Parameters in a Large Language Model: A Case Study of LLAMA 3.2 1B."
  ASSE 2025. https://doi.org/10.1145/3775030.3775040

One function drives all four paper experiments (global sweep, per-layer sweep,
self-attention-only, feed-forward-only) via the `layers` / `matrices` args —
see run_experiment.py for the CLI that maps each experiment to a config.
"""

from typing import Iterable, Literal
import torch
from transformers import AutoModelForCausalLM

MatrixName = Literal["q", "k", "v", "gate", "up", "down"]

ALL_MATRICES: tuple[MatrixName, ...] = ("q", "k", "v", "gate", "up", "down")
SELF_ATTENTION: tuple[MatrixName, ...] = ("q", "k", "v")
FEED_FORWARD: tuple[MatrixName, ...] = ("gate", "up", "down")

_MATRIX_PATH = {
    "q": ("self_attn", "q_proj"),
    "k": ("self_attn", "k_proj"),
    "v": ("self_attn", "v_proj"),
    "gate": ("mlp", "gate_proj"),
    "up": ("mlp", "up_proj"),
    "down": ("mlp", "down_proj"),
}


def dropout_mask(weight: torch.Tensor, p: float) -> torch.Tensor:
    """Zero out each element of `weight` independently with probability p.

    Matches the paper's `corrupt()` function exactly (Section 2.4):
        mask = torch.rand_like(input) > p
        return input * mask
    """
    mask = torch.rand_like(weight) > p
    return weight * mask


def corrupt_model(
    model: AutoModelForCausalLM,
    p: float,
    layers: Iterable[int],
    matrices: Iterable[MatrixName] = ALL_MATRICES,
) -> None:
    """Corrupt `model` in place.

    Args:
        model: a loaded causal LM (tested against Llama-3.2-1B-Instruct's
            16-layer architecture; layer indices are 0-based here).
        p: corruption probability, e.g. 0.15 for 15%.
        layers: which transformer layer indices to corrupt.
        matrices: which of the 6 matrix types to corrupt in each targeted
            layer. Defaults to all six (the paper's "global" experiment).

    Reproduces each paper experiment by choice of `layers` / `matrices`:
        Experiment 1 (global sweep):     layers=range(16), matrices=ALL_MATRICES
        Experiment 2 (per-layer sweep):  layers=[i],        matrices=ALL_MATRICES
        Experiment 3 (self-attention):   layers=range(16) or [15], matrices=SELF_ATTENTION
        Experiment 4 (feed-forward):     layers=range(16) or [15], matrices=FEED_FORWARD
    """
    layer_modules = model.model.layers
    for layer_idx in layers:
        layer = layer_modules[layer_idx]
        for matrix_name in matrices:
            submodule_attr, proj_attr = _MATRIX_PATH[matrix_name]
            submodule = getattr(layer, submodule_attr)
            proj = getattr(submodule, proj_attr)
            proj.weight.data.copy_(dropout_mask(proj.weight, p))
