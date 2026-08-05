"""Cohort/impostor-mean assets for voiceprint mean-centering.

Per-embedding-model cohort means live in ``assets/cohort/<slug>.npy`` (see
``scripts/compute_cohort_mean.py`` and
``docs/bugs/2026-06-30-voiceprint-score-compression.md``). This module owns
locating and loading them; :func:`has_cohort` lets callers (e.g. ``/v1/info``)
report which embedding models are voiceprint-ready, because
``settings.match_threshold`` is calibrated for the mean-centered score only —
a model without a cohort must not silently pretend to do identification
(``docs/bugs/2026-08-05-cohort-missing-threshold.md``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Directory holding per-embedding-model cohort/impostor means used to
# mean-center voiceprint embeddings before cosine matching (see
# scripts/compute_cohort_mean.py and
# docs/bugs/2026-06-30-voiceprint-score-compression.md).
_COHORT_DIR = Path(__file__).resolve().parents[2] / "assets" / "cohort"


def _cohort_slug(model_name: str) -> str:
    """Map an embedding-model name to its cohort-mean filename slug.

    Must stay consistent with ``scripts/compute_cohort_mean.py:model_slug``.
    """
    return model_name.replace("/", "__")


def _load_cohort_mean(model_name: str) -> Optional[np.ndarray]:
    """Load the cohort/impostor mean for ``model_name``, or ``None`` if absent.

    Missing or unreadable cohort files are non-fatal: the matcher falls back to
    raw cosine behaviour when the mean is ``None``.
    """
    path = _COHORT_DIR / f"{_cohort_slug(model_name)}.npy"
    if not path.exists():
        logger.info("no cohort mean for %s at %s; using raw cosine", model_name, path)
        return None
    try:
        mean = np.load(path).astype(np.float32).reshape(-1)
    except Exception:  # noqa: BLE001
        logger.warning("failed to load cohort mean %s; using raw cosine", path)
        return None
    if mean.size == 0:
        return None
    return mean


def has_cohort(model_name: str) -> bool:
    """True iff a cohort-mean file exists for this embedding model."""
    return (_COHORT_DIR / f"{_cohort_slug(model_name)}.npy").exists()


__all__ = ["has_cohort", "_load_cohort_mean", "_cohort_slug", "_COHORT_DIR"]
