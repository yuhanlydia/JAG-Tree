"""Stable labelled NumPy random streams."""

from __future__ import annotations

from hashlib import sha256

import numpy as np


def named_rng(seed: int, label: str) -> np.random.Generator:
    digest = sha256(f"{int(seed)}:{label}".encode()).digest()
    return np.random.default_rng(np.frombuffer(digest[:16], dtype=np.uint32))
