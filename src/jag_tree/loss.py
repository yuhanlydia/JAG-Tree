"""Unique-edge JAG policy-gradient objective."""

from __future__ import annotations

from typing import Any

import numpy as np


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in value.shape)


def jag_loss(logprobs: Any, edge_mask: Any, credit: Any, weights: Any) -> Any:
    """Return ``-sum_i w_i c_i sum_e mask[i,e] log p_e``.

    Each unique edge may be assigned to one credit row only; this prevents
    shared prefixes from silently receiving duplicated leaf weighting.
    """

    if len(_shape(logprobs)) != 1 or len(_shape(edge_mask)) != 2 or _shape(edge_mask)[1] != _shape(logprobs)[0]:
        raise ValueError("edge mask shape must be [credit_rows, unique_edges]")
    if _shape(credit) != (_shape(edge_mask)[0],) or _shape(weights) != (_shape(edge_mask)[0],):
        raise ValueError("credit and weights shape must match edge mask rows")
    if hasattr(edge_mask, "detach"):
        mask_values = edge_mask.detach().cpu().numpy()
    else:
        mask_values = np.asarray(edge_mask)
    if not np.all((mask_values == 0) | (mask_values == 1)):
        raise ValueError("edge mask must be binary")
    if np.any(mask_values.sum(axis=0) > 1):
        raise ValueError("repeated unique edges are forbidden")
    selected = edge_mask.to(dtype=logprobs.dtype) if hasattr(edge_mask, "to") else np.asarray(edge_mask, dtype=np.asarray(logprobs).dtype)
    return -((selected * logprobs).sum(axis=1) * credit * weights).sum()
