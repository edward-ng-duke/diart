"""Persistent, per-embedding-model voiceprint store.

Layout under ``root``:
  manifest.json  -> {model: {name: [vec_id, ...]}}
  vectors.npz    -> {vec_id: np.ndarray}
Vectors are stored raw (one per enrolled clip). ``vectors(model)`` returns
the per-name mean, L2-normalized, ready for cosine matching.
"""
from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

import numpy as np


class VoiceprintStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._manifest: dict[str, dict[str, list[str]]] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    # ----- mutation -----
    def enroll(self, model: str, name: str, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        with self._lock:
            vid = uuid.uuid4().hex
            self._vectors[vid] = vec
            self._manifest.setdefault(model, {}).setdefault(name, []).append(vid)
        self.save()

    def remove(self, model: str, name: str) -> None:
        with self._lock:
            ids = self._manifest.get(model, {}).pop(name, [])
            for vid in ids:
                self._vectors.pop(vid, None)
        self.save()

    # ----- query -----
    def vectors(self, model: str) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        with self._lock:
            for name, ids in self._manifest.get(model, {}).items():
                arrs = [self._vectors[i] for i in ids if i in self._vectors]
                if not arrs:
                    continue
                mean = np.mean(np.stack(arrs, axis=0), axis=0)
                norm = float(np.linalg.norm(mean))
                out[name] = mean / norm if norm > 0 else mean
        return out

    def list(self, model: str) -> list[dict]:
        with self._lock:
            return [
                {"name": name, "n_clips": len(ids)}
                for name, ids in self._manifest.get(model, {}).items()
            ]

    # ----- persistence -----
    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock:
            (self.root / "manifest.json").write_text(
                json.dumps(self._manifest, ensure_ascii=False)
            )
            if self._vectors:
                np.savez(self.root / "vectors.npz", **self._vectors)

    def load(self) -> None:
        man = self.root / "manifest.json"
        vecs = self.root / "vectors.npz"
        with self._lock:
            self._manifest = json.loads(man.read_text()) if man.exists() else {}
            if vecs.exists():
                with np.load(vecs) as data:
                    self._vectors = {k: data[k] for k in data.files}
            else:
                self._vectors = {}
