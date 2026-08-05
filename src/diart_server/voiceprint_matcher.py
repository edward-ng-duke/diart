"""Cosine matching of a speaker embedding against enrolled references.

Two scoring modes are supported:

* **Raw cosine** (``mean_embedding is None``): the historical behaviour —
  each vector is L2-normalized and the score is the plain cosine (dot product)
  in ``[-1, 1]``. Threshold semantics are applied directly to that cosine.

* **Mean-centered cosine** (``mean_embedding`` provided): every vector (both the
  query and each reference) is first projected away from a shared *cohort/impostor
  mean direction* and re-normalized — ``c = unit(unit(v) - mean_embedding)`` — which
  removes the large non-discriminative component that compresses raw cosines in
  wespeaker/pyannote embedding spaces (see
  ``docs/bugs/2026-06-30-voiceprint-score-compression.md``). The centered cosine is
  then remapped to ``[0, 1]`` via ``(c + 1) / 2`` so the returned score sits on a
  usable absolute scale, and the threshold is applied to that remapped score.

A ``mean_embedding`` whose shape does not match the query vector (wrong cohort
for this embedding space) is treated as absent: the whole call runs the raw
mode with NO ``(c + 1) / 2`` remap. Remapping a raw cosine would inflate it
onto the centered scale and turn impostors into false accepts
(``docs/bugs/2026-08-05-cohort-missing-threshold.md`` §2.2).

In both modes the function returns ``(best_name | None, score)`` with the same
threshold semantics: a name is returned only when its score is ``>= threshold``.
"""
from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray | None:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        return None
    return v / n


def _centered_unit(v: np.ndarray, mean: np.ndarray) -> np.ndarray | None:
    """Return ``unit(unit(v) - mean)`` or ``None`` if it collapses to zero.

    The cohort ``mean`` is expected to already be L2-normalized, but its exact
    norm is irrelevant: subtracting it removes the shared direction and the
    result is re-normalized regardless.
    """
    u = _unit(v)
    if u is None:
        return None
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    if mean.shape != u.shape:
        # Shape mismatch (wrong cohort for this model) -> fall back to raw unit.
        return u
    return _unit(u - mean)


def match(
    vec: np.ndarray,
    refs: dict[str, np.ndarray],
    threshold: float,
    mean_embedding: np.ndarray | None = None,
) -> tuple[str | None, float]:
    """Return (best_name, score) or (None, score) when below threshold.

    Parameters
    ----------
    vec:
        Query embedding (live centroid or utterance embedding).
    refs:
        Mapping of enrolled name -> reference embedding.
    threshold:
        Minimum score for a positive match. Applied to the raw cosine when
        ``mean_embedding is None``, or to the remapped centered score otherwise.
    mean_embedding:
        Optional cohort/impostor mean (L2-normalized). When provided, both the
        query and every reference are mean-centered the same way before scoring,
        and the centered cosine is remapped to ``[0, 1]`` via ``(c + 1) / 2``.
        When ``None`` the historical raw-cosine behaviour is preserved exactly.
        A mean whose shape differs from the query (wrong cohort for this
        model's space) is discarded up front: the call scores in pure raw mode
        with no remap, exactly as if ``mean_embedding=None`` had been passed.
    """
    if mean_embedding is not None:
        mean = np.asarray(mean_embedding, dtype=np.float32).reshape(-1)
        q = np.asarray(vec, dtype=np.float32).reshape(-1)
        if mean.shape != q.shape:
            mean_embedding = None  # wrong cohort for this space -> raw mode, no remap
    if mean_embedding is None:
        u = _unit(vec)
    else:
        u = _centered_unit(vec, mean_embedding)
    if u is None or not refs:
        return None, 0.0

    best_name, best_score = None, -1.0
    for name, ref in refs.items():
        if mean_embedding is None:
            r = _unit(ref)
        else:
            r = _centered_unit(ref, mean_embedding)
        if r is None or r.shape != u.shape:
            continue
        score = float(np.dot(u, r))
        if mean_embedding is not None:
            # Remap centered cosine [-1, 1] -> [0, 1] for a usable scale.
            score = (score + 1.0) / 2.0
        if score > best_score:
            best_name, best_score = name, score

    if best_name is None:
        return None, 0.0
    if best_score < threshold:
        return None, best_score
    return best_name, best_score
