"""Lazy model registry for diart_server.

Loads a JSON manifest describing available segmentation/embedding models
and constructs them on demand. Constructed models are cached so repeated
lookups return the same instance.

See STANDALONE_SERVER.md section 5.5 for the full design rationale.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from diart.models import EmbeddingModel, SegmentationModel
from diart_server import metrics
from diart_server.errors import ModelNotFoundError


ModelKind = Literal["segmentation", "embedding"]
ModelBackend = Literal["pyannote", "onnx", "speechbrain", "nemo", "wespeaker_onnx"]


@dataclass
class ModelSpec:
    """A single entry in the models manifest.

    ``match_threshold`` optionally overrides ``settings.match_threshold`` for
    voiceprint identification with this embedding model: the centered-score
    scale differs per embedding space, so a single global threshold cannot fit
    every model (e.g. cnceleb-resnet34 needs ~0.85 where the deployed
    voxceleb-resnet34 needs 0.82). ``None`` -> use the global setting.
    """

    hf_name: str
    local_path: str
    kind: ModelKind
    backend: ModelBackend
    match_threshold: float | None = None


class ModelsRegistry:
    """Lazy, cached registry of segmentation and embedding models.

    The registry is constructed from a JSON manifest enumerating every
    model the server is willing to serve. Models are not instantiated
    until first requested via :meth:`get_segmentation` or
    :meth:`get_embedding`; subsequent calls return the cached instance.

    Backend dispatch:
        * ``onnx`` backends are loaded via ``from_onnx``.
        * Every other backend (``pyannote``, ``speechbrain``, ``nemo``)
          goes through ``from_pyannote``, which internally falls back
          to :class:`pyannote.audio.pipelines.speaker_verification.PretrainedSpeakerEmbedding`
          for non-pyannote checkpoints (see ``diart.models.PyannoteLoader``).
    """

    def __init__(self, models_dir: Path, manifest_path: Path) -> None:
        self.models_dir = Path(models_dir)
        self.specs: dict[str, ModelSpec] = self._load_manifest(Path(manifest_path))
        self._segmentations: dict[str, SegmentationModel] = {}
        self._embeddings: dict[str, EmbeddingModel] = {}

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, ModelSpec]:
        data = json.loads(path.read_text())
        # ``ModelSpec`` doesn't carry every JSON field (e.g. ``filename``
        # used by the ONNX downloader). Restrict to the dataclass fields
        # so optional manifest extensions don't break construction.
        keep = {"hf_name", "local_path", "kind", "backend", "match_threshold"}
        return {
            entry["hf_name"]: ModelSpec(**{k: v for k, v in entry.items() if k in keep})
            for entry in data
        }

    def _resolve_path(self, spec: ModelSpec) -> str:
        """Return the path the underlying loader expects.

        - ``onnx`` / ``speechbrain`` / ``nemo``: the ``local_path`` as-is
          (loader accepts a file or directory).
        - ``pyannote``: ``Model.from_pretrained`` insists on either a HF
          repo id or a path to the *checkpoint file* (not the snapshot
          directory). We always point it at ``pytorch_model.bin`` inside
          the snapshot, falling back to any sibling ``*.bin``.
        """
        base = self.models_dir / spec.local_path
        if spec.backend != "pyannote":
            return str(base)
        candidate = base / "pytorch_model.bin"
        if candidate.exists():
            return str(candidate)
        bins = sorted(base.glob("*.bin")) if base.is_dir() else []
        if bins:
            return str(bins[0])
        # Nothing found yet — return the directory so the loader can raise
        # a clear "not found" instead of us guessing wrong.
        return str(base)

    def warm_up(
        self,
        device: torch.device,
        default_seg: str,
        default_emb: str,
    ) -> None:
        """Synchronously load the default segmentation+embedding models.

        Intended to be invoked once during server startup so that the
        first session does not pay model-load latency. Each model is
        moved to ``device`` and put into eval mode.
        """
        seg = self.get_segmentation(default_seg)
        seg.to(device).eval()
        emb = self.get_embedding(default_emb)
        emb.to(device).eval()

    def get_segmentation(self, name: str) -> SegmentationModel:
        """Return the segmentation model for ``name``, loading it on first access.

        Raises
        ------
        ModelNotFoundError
            If ``name`` is not in the manifest, or its kind is not
            ``segmentation``.
        """
        cached = self._segmentations.get(name)
        if cached is not None:
            return cached
        spec = self.specs.get(name)
        if spec is None or spec.kind != "segmentation":
            raise ModelNotFoundError(f"segmentation model not found: {name}")
        local_path = self._resolve_path(spec)
        # Time the actual load so dashboards can spot slow disks/network.
        start = time.monotonic()
        if spec.backend == "onnx":
            model = SegmentationModel.from_onnx(local_path)
        else:
            model = SegmentationModel.from_pyannote(local_path, use_hf_token=False)
        metrics.model_load_seconds.labels(model=name).set(
            time.monotonic() - start
        )
        self._segmentations[name] = model
        return model

    def get_embedding(self, name: str) -> EmbeddingModel:
        """Return the embedding model for ``name``, loading it on first access.

        Raises
        ------
        ModelNotFoundError
            If ``name`` is not in the manifest, or its kind is not
            ``embedding``.
        """
        cached = self._embeddings.get(name)
        if cached is not None:
            return cached
        spec = self.specs.get(name)
        if spec is None or spec.kind != "embedding":
            raise ModelNotFoundError(f"embedding model not found: {name}")
        local_path = self._resolve_path(spec)
        # Time the actual load so dashboards can spot slow disks/network.
        start = time.monotonic()
        if spec.backend == "onnx":
            model = EmbeddingModel.from_onnx(local_path)
        elif spec.backend == "wespeaker_onnx":
            # wespeaker model-zoo ONNX exports take FBANK features, not raw
            # waveforms, so diart's own ONNXLoader (waveform input) cannot run
            # them. pyannote.audio ships a wrapper that computes fbank in
            # Python and feeds the ORT session; wrap it in a lazy loader so
            # the EmbeddingModel contract (load-on-first-use) is preserved.
            from pyannote.audio.pipelines.speaker_verification import (
                ONNXWeSpeakerPretrainedSpeakerEmbedding,
            )

            model = EmbeddingModel(
                lambda: ONNXWeSpeakerPretrainedSpeakerEmbedding(local_path)
            )
        else:
            model = EmbeddingModel.from_pyannote(local_path, use_hf_token=False)
        metrics.model_load_seconds.labels(model=name).set(
            time.monotonic() - start
        )
        self._embeddings[name] = model
        return model


__all__ = ["ModelSpec", "ModelsRegistry"]
