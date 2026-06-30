"""Cosine matching of a speaker embedding against enrolled references."""
from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray | None:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        return None
    return v / n


def match(
    vec: np.ndarray, refs: dict[str, np.ndarray], threshold: float
) -> tuple[str | None, float]:
    """Return (best_name, score) or (None, score) when below threshold.

    Both ``vec`` and each ref are L2-normalized, so cosine == dot product.
    """
    u = _unit(vec)
    if u is None or not refs:
        return None, 0.0
    best_name, best_score = None, -1.0
    for name, ref in refs.items():
        r = _unit(ref)
        if r is None or r.shape != u.shape:
            continue
        score = float(np.dot(u, r))
        if score > best_score:
            best_name, best_score = name, score
    if best_name is None:
        return None, 0.0
    if best_score < threshold:
        return None, best_score
    return best_name, best_score
