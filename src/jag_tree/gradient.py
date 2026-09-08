"""Registered-parameter per-example score sketches with lazy Torch import."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class GradientSpec:
    parameter_names: tuple[str, ...]
    dimension: int = 256
    seed: int = 0
    keep_exact: bool = False

    def __post_init__(self) -> None:
        if not self.parameter_names or self.dimension < 1:
            raise ValueError("gradient spec needs registered parameters and positive dimension")


@dataclass(frozen=True)
class ScoreSketch:
    sketch: np.ndarray
    projection: np.ndarray
    exact: np.ndarray | None


@dataclass(frozen=True)
class ScoreExample:
    """One sampled response with an explicit causal-token score mask."""

    model_inputs: Mapping[str, Any]
    edge_start: int
    edge_length: int


def causal_edge_logprob(logits: Any, input_ids: Any, edge_start: int, edge_length: int) -> Any:
    """Sum log p(x[t] | x[:t]) over one edge using one shared shift."""

    import torch

    if logits.ndim != 3 or input_ids.ndim != 2 or logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input IDs must align on batch/token axes")
    if edge_start < 1 or edge_length < 1 or edge_start + edge_length > input_ids.shape[1]:
        raise ValueError("edge span must have causal context and lie inside input IDs")
    shifted = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return shifted[:, edge_start - 1 : edge_start - 1 + edge_length].sum()


def score_sketch(model: Any, batch: Any, spec: GradientSpec) -> ScoreSketch:
    """Differentiate sampled conditional log-probabilities over registered blocks."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("score sketches require the optional torch extra") from exc
    registered = dict(model.named_parameters())
    missing = sorted(set(spec.parameter_names) - set(registered))
    if missing:
        raise ValueError(f"unregistered parameter names: {missing}")
    parameters = tuple(registered[name] for name in spec.parameter_names)
    rows: list[np.ndarray] = []
    sketches: list[np.ndarray] = []
    projection_chunks: list[np.ndarray] = []
    for example in batch:
        if not isinstance(example, ScoreExample):
            raise TypeError("score batch rows must be ScoreExample instances")
        model.zero_grad(set_to_none=True)
        output = model(**dict(example.model_inputs))
        if hasattr(output, "logits"):
            output = output.logits
        input_ids = example.model_inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("score examples require input_ids")
        score = causal_edge_logprob(output, input_ids, example.edge_start, example.edge_length)
        gradients = torch.autograd.grad(score, parameters, retain_graph=False, create_graph=False)
        rng = np.random.default_rng(spec.seed)
        sketch = np.zeros(spec.dimension, dtype=np.float64)
        exact_parts: list[np.ndarray] = []
        current_projection: list[np.ndarray] = []
        for gradient in gradients:
            flattened = gradient.detach().cpu().float().numpy().reshape(-1)
            if spec.keep_exact:
                exact_parts.append(flattened)
            for start in range(0, flattened.size, 65_536):
                chunk = flattened[start : start + 65_536]
                signs = rng.choice(np.array([-1.0, 1.0]), size=(chunk.size, spec.dimension)) / np.sqrt(spec.dimension)
                sketch += chunk @ signs
                if spec.keep_exact and not sketches:
                    current_projection.append(signs)
        if spec.keep_exact:
            rows.append(np.concatenate(exact_parts))
            if not sketches:
                projection_chunks = current_projection
        sketches.append(sketch)
    exact = np.stack(rows) if spec.keep_exact else None
    projection = np.concatenate(projection_chunks, axis=0) if spec.keep_exact else np.empty((0, spec.dimension), dtype=np.float64)
    return ScoreSketch(np.stack(sketches), projection, exact)
