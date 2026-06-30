# ASR + Voiceprint Near-Real-Time Demo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `diart_server` with pre-enrolled voiceprint identification and per-utterance Qwen3-ASR transcription, exposed through a switchable-model `/demo/live` browser page that does near-real-time mic capture.

**Architecture:** Approach A — all orchestration in-process. The WS session taps decoded PCM into a timestamped ring buffer, matches each speaker's centroid embedding against a per-embedding-model voiceprint store, aggregates predictions into utterances, and on each finalized turn slices the audio and calls Qwen3-ASR asynchronously, streaming back a new `transcript` message.

**Tech Stack:** Python 3.10–3.12, FastAPI + Starlette WebSockets, pydantic v2, numpy<2, torch, diart, httpx (async ASR), onnxruntime/speechbrain/pyannote embedding backends, vanilla-JS browser page (AudioWorklet).

## Global Constraints

- Python 3.10–3.12; `numpy>=1.20,<2.0`; pydantic v2 (`ConfigDict(extra="forbid")` on wire models).
- **Protocol changes are additive only** — new fields default to `None`; never remove/rename existing fields.
- Internal pipeline sample rate is **16000 Hz mono**; the server already resamples inbound frames (`audio_decoder.to_pcm_16k_mono`).
- **Voiceprints are namespaced by `embedding_model`** — never compare vectors across models.
- Models live in repo-root `./models` (layout per `docker/models.json` `local_path`); `./models` and `./voiceprints` are git-ignored.
- ASR contract (verified 2026-06-30): `POST {asr_base_url}/audio/transcriptions`, multipart `model`/`file`(+optional `language`), header `Authorization: Bearer <key>`; response `{"text":"language <Lang><asr_text><文本>", ...}`. Parse text after `<asr_text>`.
- TDD: failing test → minimal impl → green → commit. One logical change per commit.
- Test runner: `pytest` from repo root (`pyproject.toml` sets `pythonpath=["src","tests"]`, `--import-mode=importlib`).

---

## File Structure

**New (`src/diart_server/`):** `pcm_ring_buffer.py`, `voiceprint_matcher.py`, `voiceprint_store.py`, `turn_aggregator.py`, `asr_client.py`, `embedding_extractor.py`, `voiceprint_routes.py`, `static/live.html`
**Modified:** `protocol.py`, `config.py`, `session.py`, `app.py`, `../../setup.cfg`, `../../.gitignore`
**New tests (`tests/diart_server/`):** one per new module + `test_session_transcript.py`
**New docs:** repo-root `MODELS.md`

---

## Task 1: Localize models to `./models` + MODELS.md

**Files:**
- Create: `models/` (downloaded, git-ignored), `MODELS.md`
- Modify: `.gitignore`

**Interfaces:**
- Produces: a populated `./models` dir usable as `DIART_MODELS_DIR`; `MODELS.md` documenting model choices.

- [ ] **Step 1: Ignore artifacts.** Append to `.gitignore`:

```
# local model + voiceprint artifacts
/models/
/voiceprints/
```

- [ ] **Step 2: Download all models from HF cache into `./models`.**

```bash
HF_TOKEN=$(sed -n 's/^huggingface-key=//p' .env) python docker/download_models.py ./models
```
Expected: ends with `[done] all 10 models present under models`. (Most are copied from the local HF cache; titanet/nemo files download but need `nemo_toolkit` to load — that's expected.)

- [ ] **Step 3: Verify layout.**

Run: `ls models && ls models/onnx models/speechbrain models/pyannote 2>/dev/null`
Expected: `onnx/speaker-embedding.onnx`, `pyannote/{embedding,segmentation,segmentation-3.0,wespeaker}`, `speechbrain/{ecapa,ecapa-mel,resnet,xvect}`, `nvidia/titanet`.

- [ ] **Step 4: Write `MODELS.md`** at repo root with this content:

````markdown
# 模型清单（声纹 embedding / 分割 segmentation）

所有模型下载在 `./models`，运行时 `DIART_MODELS_DIR=./models`。WebSocket 连接用
`?embedding_model=<name>&segmentation_model=<name>` 选择；前端 `/demo/live` 提供下拉切换。

> ⚠️ 切换 embedding 模型会切换声纹库命名空间——不同模型的向量不可互比，
> 注册过的人只在“注册时所用的模型”下能被识别。

## Embedding（声纹）模型

| name (传给 `embedding_model`) | 后端 | 维度 | 速度 | 质量 | 备注 |
|---|---|---|---|---|---|
| `onnx/speaker-embedding.onnx`* | onnx | 256 | ★★★ | ★★☆ | WeSpeaker，纯离线最快 |
| `pyannote/wespeaker` | pyannote | 256 | ★★☆ | ★★★ | WeSpeaker fp32，第一梯队 |
| `speechbrain/ecapa` | speechbrain | 192 | ★★☆ | ★★★ | ECAPA-TDNN，SV 最强之一 |
| `speechbrain/ecapa-mel` | speechbrain | 192 | ★★☆ | ★★★ | ECAPA mel-spec 变体 |
| `speechbrain/resnet` | speechbrain | 256 | ★★☆ | ★★☆ | ResNet 声纹 |
| `speechbrain/xvect` | speechbrain | 512 | ★★☆ | ★★☆ | 经典 x-vector |
| `pyannote/embedding` | pyannote | 512 | ★★☆ | ★★☆ | diart 默认，精度偏弱 |
| `nvidia/titanet` | nemo | 192 | ★☆☆ | ★★★ | 需 `pip install nemo_toolkit[asr]`，默认隐藏 |

\* `name` 是相对 `DIART_MODELS_DIR` 的 `local_path`（见 `docker/models.json`）。

## Segmentation 模型
| name | 后端 | 备注 |
|---|---|---|
| `pyannote/segmentation-3.0` | pyannote | 默认 |
| `pyannote/segmentation` | pyannote | 旧版 |

## 切换 / 注册注意
- 推荐默认 `speechbrain/ecapa` 或 `pyannote/wespeaker`。
- 注册（`POST /v1/voiceprints`）必须带当前 `embedding_model`。
- titanet 默认在前端隐藏；要用先装 nemo 并从前端代码取消隐藏。
````

- [ ] **Step 5: Commit.**

```bash
git add .gitignore MODELS.md
git commit -m "chore: localize models to ./models and document them in MODELS.md"
```

---

## Task 2: Config settings

**Files:**
- Modify: `src/diart_server/config.py`
- Test: `tests/diart_server/test_config_new_settings.py`

**Interfaces:**
- Produces: `settings.{voiceprints_dir, asr_base_url, asr_model, asr_api_key, asr_timeout, match_threshold, silence_gap, max_turn_seconds, ring_buffer_seconds, max_asr_concurrency}`.

- [ ] **Step 1: Write the failing test** — `tests/diart_server/test_config_new_settings.py`:

```python
from diart_server.config import Settings


def test_new_settings_have_expected_defaults():
    s = Settings()
    assert str(s.asr_base_url) == "http://10.0.0.32:6001/v1"
    assert s.asr_model == "models/Qwen3-ASR-1.7B"
    assert s.asr_api_key == "EMPTY"
    assert s.asr_timeout == 60
    assert s.match_threshold == 0.5
    assert s.silence_gap == 0.7
    assert s.max_turn_seconds == 20.0
    assert s.ring_buffer_seconds == 120.0
    assert s.max_asr_concurrency == 2
    assert str(s.voiceprints_dir) == "voiceprints"
```

- [ ] **Step 2: Run test, verify it fails.** Run: `pytest tests/diart_server/test_config_new_settings.py -v` → FAIL (`AttributeError`).

- [ ] **Step 3: Add fields** to `Settings` in `src/diart_server/config.py` (after `log_level`):

```python
    voiceprints_dir: Path = Field(
        default=Path("voiceprints"), validation_alias="DIART_VOICEPRINTS_DIR"
    )
    asr_base_url: str = Field(
        default="http://10.0.0.32:6001/v1", validation_alias="DIART_ASR_BASE_URL"
    )
    asr_model: str = Field(
        default="models/Qwen3-ASR-1.7B", validation_alias="DIART_ASR_MODEL"
    )
    asr_api_key: str = Field(default="EMPTY", validation_alias="DIART_ASR_API_KEY")
    asr_timeout: float = Field(default=60, validation_alias="DIART_ASR_TIMEOUT")
    match_threshold: float = Field(
        default=0.5, validation_alias="DIART_MATCH_THRESHOLD"
    )
    silence_gap: float = Field(default=0.7, validation_alias="DIART_SILENCE_GAP")
    max_turn_seconds: float = Field(
        default=20.0, validation_alias="DIART_MAX_TURN_SECONDS"
    )
    ring_buffer_seconds: float = Field(
        default=120.0, validation_alias="DIART_RING_BUFFER_SECONDS"
    )
    max_asr_concurrency: int = Field(
        default=2, validation_alias="DIART_MAX_ASR_CONCURRENCY"
    )
```

- [ ] **Step 4: Run test, verify it passes.** Run: `pytest tests/diart_server/test_config_new_settings.py -v` → PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/diart_server/config.py tests/diart_server/test_config_new_settings.py
git commit -m "feat(diart_server): add voiceprint/ASR/turn settings to config"
```

---

## Task 3: Protocol messages

**Files:**
- Modify: `src/diart_server/protocol.py`
- Test: `tests/diart_server/test_protocol_new.py`

**Interfaces:**
- Produces: `PredictionMessage` gains `name: str | None = None`, `score: float | None = None`; new `TranscriptMessage`, `EnrollResponse`, `VoiceprintInfo`.

- [ ] **Step 1: Write the failing test** — `tests/diart_server/test_protocol_new.py`:

```python
from diart_server.protocol import (
    PredictionMessage,
    TranscriptMessage,
    EnrollResponse,
    VoiceprintInfo,
)


def test_prediction_message_optional_identity_defaults_none():
    m = PredictionMessage(
        start=0.0, duration=1.0, speaker="speaker0",
        embedding=[0.1, 0.2], samples_count=3,
    )
    assert m.name is None and m.score is None


def test_transcript_message_roundtrip():
    m = TranscriptMessage(
        speaker="speaker0", name="张三", score=0.81,
        start=1.0, end=3.0, text="你好世界", lang="Chinese",
    )
    d = m.model_dump()
    assert d["type"] == "transcript" and d["name"] == "张三"


def test_enroll_response_and_info():
    r = EnrollResponse(name="张三", embedding_model="speechbrain/ecapa", n_clips=2)
    assert r.type == "enrolled"
    info = VoiceprintInfo(name="张三", n_clips=2)
    assert info.n_clips == 2
```

- [ ] **Step 2: Run test, verify it fails.** Run: `pytest tests/diart_server/test_protocol_new.py -v` → FAIL (ImportError).

- [ ] **Step 3: Implement.** In `src/diart_server/protocol.py` add `name`/`score` to `PredictionMessage` (after `embedding`):

```python
    name: str | None = None
    score: float | None = None
```

Add new models (before the `QueryParams` section) and extend `__all__`:

```python
class TranscriptMessage(BaseModel):
    """Per-utterance ASR result, sent when a speaker turn is finalized."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["transcript"] = "transcript"
    speaker: str
    name: str | None = None
    score: float | None = None
    start: float
    end: float
    text: str
    lang: str = ""


class VoiceprintInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    n_clips: int


class EnrollResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["enrolled"] = "enrolled"
    name: str
    embedding_model: str
    n_clips: int
```

Append the three names to `__all__`.

- [ ] **Step 4: Run test, verify it passes.** Run: `pytest tests/diart_server/test_protocol_new.py -v` → PASS.

- [ ] **Step 5: Run existing protocol tests to confirm no regression.** Run: `pytest tests/diart_server/test_protocol.py -v` → PASS.

- [ ] **Step 6: Commit.**

```bash
git add src/diart_server/protocol.py tests/diart_server/test_protocol_new.py
git commit -m "feat(diart_server): add transcript + identity fields to protocol"
```

---

## Task 4: PCM ring buffer

**Files:**
- Create: `src/diart_server/pcm_ring_buffer.py`
- Test: `tests/diart_server/test_pcm_ring_buffer.py`

**Interfaces:**
- Produces: `PcmRingBuffer(max_seconds=120.0, sample_rate=16000)` with `append(pcm: np.ndarray) -> None`, `slice(t0: float, t1: float) -> np.ndarray | None`, property `total_seconds -> float`.

- [ ] **Step 1: Write the failing test** — `tests/diart_server/test_pcm_ring_buffer.py`:

```python
import numpy as np
from diart_server.pcm_ring_buffer import PcmRingBuffer


def test_slice_returns_exact_window():
    rb = PcmRingBuffer(max_seconds=10, sample_rate=100)  # 100 Hz for easy math
    rb.append(np.arange(0, 100, dtype=np.float32))   # t=0..1s
    rb.append(np.arange(100, 300, dtype=np.float32))  # t=1..3s
    assert abs(rb.total_seconds - 3.0) < 1e-9
    seg = rb.slice(1.0, 2.0)
    assert seg is not None
    np.testing.assert_array_equal(seg, np.arange(100, 200, dtype=np.float32))


def test_slice_out_of_window_returns_none():
    rb = PcmRingBuffer(max_seconds=1, sample_rate=100)
    rb.append(np.zeros(100, dtype=np.float32))   # t=0..1
    rb.append(np.ones(100, dtype=np.float32))    # t=1..2, evicts t=0..1
    assert rb.slice(0.0, 0.5) is None             # evicted
    np.testing.assert_array_equal(rb.slice(1.0, 2.0), np.ones(100, dtype=np.float32))


def test_empty_slice_returns_none():
    rb = PcmRingBuffer(max_seconds=5, sample_rate=100)
    rb.append(np.zeros(100, dtype=np.float32))
    assert rb.slice(0.5, 0.5) is None
```

- [ ] **Step 2: Run test, verify it fails.** Run: `pytest tests/diart_server/test_pcm_ring_buffer.py -v` → FAIL (ImportError).

- [ ] **Step 3: Implement** `src/diart_server/pcm_ring_buffer.py`:

```python
"""Timestamped rolling buffer of 16 kHz mono float32 PCM.

Indexed by absolute stream time (cumulative samples / sample_rate), so a
finalized turn's [start, end] in diart's timeline can be sliced back out
of the raw audio to send to ASR.
"""
from __future__ import annotations

import threading

import numpy as np


class PcmRingBuffer:
    def __init__(self, max_seconds: float = 120.0, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._capacity = max(1, int(round(max_seconds * sample_rate)))
        self._buf = np.zeros(self._capacity, dtype=np.float32)
        self._total = 0  # total samples ever appended
        self._lock = threading.Lock()

    @property
    def total_seconds(self) -> float:
        return self._total / float(self.sample_rate)

    def append(self, pcm: np.ndarray) -> None:
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        with self._lock:
            n = pcm.size
            if n >= self._capacity:
                self._buf[:] = pcm[-self._capacity:]
            else:
                start = self._total % self._capacity
                end = start + n
                if end <= self._capacity:
                    self._buf[start:end] = pcm
                else:
                    first = self._capacity - start
                    self._buf[start:] = pcm[:first]
                    self._buf[: n - first] = pcm[first:]
            self._total += n

    def slice(self, t0: float, t1: float) -> np.ndarray | None:
        with self._lock:
            s0 = int(round(t0 * self.sample_rate))
            s1 = int(round(t1 * self.sample_rate))
            oldest = max(0, self._total - self._capacity)
            s0 = max(s0, oldest)
            s1 = min(s1, self._total)
            if s1 <= s0:
                return None
            out = np.empty(s1 - s0, dtype=np.float32)
            for i, s in enumerate(range(s0, s1)):
                out[i] = self._buf[s % self._capacity]
            return out
```

- [ ] **Step 4: Run test, verify it passes.** Run: `pytest tests/diart_server/test_pcm_ring_buffer.py -v` → PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/diart_server/pcm_ring_buffer.py tests/diart_server/test_pcm_ring_buffer.py
git commit -m "feat(diart_server): add timestamped PCM ring buffer"
```

---

## Task 5: Voiceprint matcher

**Files:**
- Create: `src/diart_server/voiceprint_matcher.py`
- Test: `tests/diart_server/test_voiceprint_matcher.py`

**Interfaces:**
- Produces: `match(vec: np.ndarray, refs: dict[str, np.ndarray], threshold: float) -> tuple[str | None, float]`.

- [ ] **Step 1: Write the failing test** — `tests/diart_server/test_voiceprint_matcher.py`:

```python
import numpy as np
from diart_server.voiceprint_matcher import match


def test_matches_closest_above_threshold():
    refs = {"张三": np.array([1.0, 0.0]), "李四": np.array([0.0, 1.0])}
    name, score = match(np.array([0.9, 0.1]), refs, threshold=0.5)
    assert name == "张三" and score > 0.5


def test_below_threshold_is_unknown():
    refs = {"张三": np.array([1.0, 0.0])}
    name, score = match(np.array([0.0, 1.0]), refs, threshold=0.5)
    assert name is None and score < 0.5


def test_empty_refs_or_vec_is_unknown():
    assert match(np.array([1.0, 0.0]), {}, 0.5) == (None, 0.0)
    assert match(np.array([0.0, 0.0]), {"a": np.array([1.0, 0.0])}, 0.5)[0] is None
```

- [ ] **Step 2: Run test, verify it fails.** Run: `pytest tests/diart_server/test_voiceprint_matcher.py -v` → FAIL.

- [ ] **Step 3: Implement** `src/diart_server/voiceprint_matcher.py`:

```python
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
```

- [ ] **Step 4: Run test, verify it passes.** Run: `pytest tests/diart_server/test_voiceprint_matcher.py -v` → PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/diart_server/voiceprint_matcher.py tests/diart_server/test_voiceprint_matcher.py
git commit -m "feat(diart_server): add cosine voiceprint matcher"
```

---

## Task 6: Voiceprint store

**Files:**
- Create: `src/diart_server/voiceprint_store.py`
- Test: `tests/diart_server/test_voiceprint_store.py`

**Interfaces:**
- Produces: `VoiceprintStore(root: Path)` with `enroll(model, name, vec)`, `vectors(model) -> dict[str, np.ndarray]` (per-name mean, L2-normalized), `list(model) -> list[dict]`, `remove(model, name)`, `save()`, `load()`. Namespaced by `model`.

- [ ] **Step 1: Write the failing test** — `tests/diart_server/test_voiceprint_store.py`:

```python
import numpy as np
from diart_server.voiceprint_store import VoiceprintStore


def test_enroll_mean_and_namespace(tmp_path):
    s = VoiceprintStore(tmp_path)
    s.enroll("ecapa", "张三", np.array([2.0, 0.0]))
    s.enroll("ecapa", "张三", np.array([0.0, 0.0]))  # mean -> [1,0] before norm
    s.enroll("wespeaker", "张三", np.array([0.0, 1.0]))
    v = s.vectors("ecapa")["张三"]
    np.testing.assert_allclose(v, np.array([1.0, 0.0]), atol=1e-6)  # normalized
    assert "张三" not in s.vectors("wespeaker") or \
        not np.allclose(s.vectors("wespeaker")["张三"], v)  # different namespace
    assert s.list("ecapa") == [{"name": "张三", "n_clips": 2}]


def test_persistence_roundtrip(tmp_path):
    s = VoiceprintStore(tmp_path)
    s.enroll("ecapa", "李四", np.array([1.0, 2.0, 3.0]))
    s.save()
    s2 = VoiceprintStore(tmp_path)
    s2.load()
    assert s2.list("ecapa") == [{"name": "李四", "n_clips": 1}]
    assert s2.vectors("ecapa")["李四"].shape == (3,)


def test_remove(tmp_path):
    s = VoiceprintStore(tmp_path)
    s.enroll("ecapa", "王五", np.array([1.0, 0.0]))
    s.remove("ecapa", "王五")
    assert s.list("ecapa") == []
```

- [ ] **Step 2: Run test, verify it fails.** Run: `pytest tests/diart_server/test_voiceprint_store.py -v` → FAIL.

- [ ] **Step 3: Implement** `src/diart_server/voiceprint_store.py`:

```python
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
```

- [ ] **Step 4: Run test, verify it passes.** Run: `pytest tests/diart_server/test_voiceprint_store.py -v` → PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/diart_server/voiceprint_store.py tests/diart_server/test_voiceprint_store.py
git commit -m "feat(diart_server): add per-model persistent voiceprint store"
```

---

## Task 7: Turn aggregator

**Files:**
- Create: `src/diart_server/turn_aggregator.py`
- Test: `tests/diart_server/test_turn_aggregator.py`

**Interfaces:**
- Produces: `FinalizedTurn(speaker: str, start: float, end: float)` (dataclass); `TurnAggregator(silence_gap=0.7, max_turn_seconds=20.0)` with `feed(speaker, start, end, now) -> list[FinalizedTurn]` and `flush(now=None) -> list[FinalizedTurn]`.

- [ ] **Step 1: Write the failing test** — `tests/diart_server/test_turn_aggregator.py`:

```python
from diart_server.turn_aggregator import TurnAggregator, FinalizedTurn


def test_same_speaker_extends_without_finalizing():
    agg = TurnAggregator(silence_gap=0.7, max_turn_seconds=20)
    assert agg.feed("s0", 0.0, 0.5, now=0.5) == []
    assert agg.feed("s0", 0.5, 1.0, now=1.0) == []
    out = agg.flush(now=1.0)
    assert out == [FinalizedTurn("s0", 0.0, 1.0)]


def test_silence_gap_finalizes():
    agg = TurnAggregator(silence_gap=0.7, max_turn_seconds=20)
    agg.feed("s0", 0.0, 1.0, now=1.0)
    out = agg.feed("s0", 2.0, 2.5, now=2.0)  # 1.0s gap > 0.7 -> finalize old
    assert FinalizedTurn("s0", 0.0, 1.0) in out


def test_speaker_switch_finalizes_previous():
    agg = TurnAggregator(silence_gap=0.7, max_turn_seconds=20)
    agg.feed("s0", 0.0, 1.0, now=1.0)
    out = agg.feed("s1", 1.0, 1.5, now=1.5)  # s1 starts at/after s0 end
    assert FinalizedTurn("s0", 0.0, 1.0) in out


def test_max_turn_finalizes():
    agg = TurnAggregator(silence_gap=5.0, max_turn_seconds=2.0)
    agg.feed("s0", 0.0, 1.0, now=1.0)
    out = agg.feed("s0", 1.0, 2.5, now=2.5)  # length 2.5 > 2.0
    assert FinalizedTurn("s0", 0.0, 2.5) in out
```

- [ ] **Step 2: Run test, verify it fails.** Run: `pytest tests/diart_server/test_turn_aggregator.py -v` → FAIL.

- [ ] **Step 3: Implement** `src/diart_server/turn_aggregator.py`:

```python
"""Aggregate per-window diart predictions into finalized utterances.

diart re-emits a growing turn each step; this collapses them into one
``FinalizedTurn`` per utterance and decides when an utterance is done.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FinalizedTurn:
    speaker: str
    start: float
    end: float


class _Open:
    __slots__ = ("start", "last_end")

    def __init__(self, start: float, last_end: float) -> None:
        self.start = start
        self.last_end = last_end


class TurnAggregator:
    def __init__(self, silence_gap: float = 0.7, max_turn_seconds: float = 20.0) -> None:
        self.silence_gap = silence_gap
        self.max_turn_seconds = max_turn_seconds
        self._open: dict[str, _Open] = {}

    def feed(self, speaker: str, start: float, end: float, now: float) -> list[FinalizedTurn]:
        finalized: list[FinalizedTurn] = []

        # 1) speaker switch: finalize other speakers whose turn ended before this starts
        for other in list(self._open):
            if other != speaker and self._open[other].last_end <= start:
                t = self._open.pop(other)
                finalized.append(FinalizedTurn(other, t.start, t.last_end))

        # 2) extend or open the current speaker
        cur = self._open.get(speaker)
        if cur is None:
            cur = _Open(start, end)
            self._open[speaker] = cur
        else:
            cur.last_end = max(cur.last_end, end)

        # 3) silence / max-length finalization across all open turns
        for s in list(self._open):
            t = self._open[s]
            if (now - t.last_end) > self.silence_gap or (
                t.last_end - t.start
            ) > self.max_turn_seconds:
                self._open.pop(s)
                finalized.append(FinalizedTurn(s, t.start, t.last_end))

        finalized.sort(key=lambda f: f.start)
        return finalized

    def flush(self, now: float | None = None) -> list[FinalizedTurn]:
        out = [FinalizedTurn(s, t.start, t.last_end) for s, t in self._open.items()]
        self._open.clear()
        out.sort(key=lambda f: f.start)
        return out
```

- [ ] **Step 4: Run test, verify it passes.** Run: `pytest tests/diart_server/test_turn_aggregator.py -v` → PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/diart_server/turn_aggregator.py tests/diart_server/test_turn_aggregator.py
git commit -m "feat(diart_server): add turn aggregator for utterance finalization"
```

---

## Task 8: ASR client

**Files:**
- Create: `src/diart_server/asr_client.py`
- Modify: `../../setup.cfg` (add `httpx` to `[options.extras_require] server`)
- Test: `tests/diart_server/test_asr_client.py`

**Interfaces:**
- Produces: `parse_asr_output(raw: str) -> tuple[str, str]`; `encode_wav(pcm_16k: np.ndarray, sample_rate=16000) -> bytes`; `AsrClient(base_url, model, api_key, timeout)` with `async transcribe(pcm_16k, language=None) -> tuple[str, str]` (returns `(lang, text)`).

- [ ] **Step 1: Add httpx dependency.** In `setup.cfg`, under `[options.extras_require]` `server=`, add a line: `    httpx>=0.27.0,<1.0.0`. Then `pip install -e '.[server]'`.

- [ ] **Step 2: Write the failing test** — `tests/diart_server/test_asr_client.py`:

```python
import asyncio
import numpy as np
import pytest
from diart_server.asr_client import parse_asr_output, encode_wav, AsrClient


def test_parse_with_marker():
    assert parse_asr_output("language Chinese<asr_text>你好世界") == ("Chinese", "你好世界")


def test_parse_without_marker():
    assert parse_asr_output("just text") == ("", "just text")


def test_parse_language_none():
    assert parse_asr_output("language None<asr_text>hi") == ("", "hi")


def test_encode_wav_is_riff_pcm16():
    wav = encode_wav(np.zeros(160, dtype=np.float32), 16000)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"


def test_transcribe_parses_response(monkeypatch):
    client = AsrClient("http://x/v1", "m", "EMPTY", timeout=5)

    async def fake_post(self, wav_bytes, language):
        return "language Chinese<asr_text>山东"

    monkeypatch.setattr(AsrClient, "_post_wav", fake_post)
    lang, text = asyncio.run(client.transcribe(np.zeros(160, dtype=np.float32)))
    assert lang == "Chinese" and text == "山东"
```

- [ ] **Step 3: Run test, verify it fails.** Run: `pytest tests/diart_server/test_asr_client.py -v` → FAIL.

- [ ] **Step 4: Implement** `src/diart_server/asr_client.py`:

```python
"""Async client for the Qwen3-ASR OpenAI-compatible transcription API."""
from __future__ import annotations

import io
import re
import wave

import httpx
import numpy as np


def parse_asr_output(raw: str) -> tuple[str, str]:
    if "<asr_text>" not in raw:
        return "", raw.strip()
    meta, text = raw.split("<asr_text>", 1)
    m = re.match(r"language\s+(\S+)", meta.strip(), re.IGNORECASE)
    lang = m.group(1) if m and m.group(1).lower() != "none" else ""
    return lang, text.strip()


def encode_wav(pcm_16k: np.ndarray, sample_rate: int = 16000) -> bytes:
    pcm = np.asarray(pcm_16k, dtype=np.float32).reshape(-1)
    pcm = np.clip(pcm, -1.0, 1.0)
    int16 = (pcm * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(int16.tobytes())
    return buf.getvalue()


class AsrClient:
    def __init__(self, base_url: str, model: str, api_key: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    async def _post_wav(self, wav_bytes: bytes, language: str | None) -> str:
        data = {"model": self.model}
        if language:
            data["language"] = language
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/audio/transcriptions",
                data=data, files=files, headers=headers,
            )
            resp.raise_for_status()
            return resp.json().get("text", "")

    async def transcribe(
        self, pcm_16k: np.ndarray, language: str | None = None
    ) -> tuple[str, str]:
        wav = encode_wav(pcm_16k)
        raw = await self._post_wav(wav, language)
        return parse_asr_output(raw)
```

- [ ] **Step 5: Run test, verify it passes.** Run: `pytest tests/diart_server/test_asr_client.py -v` → PASS.

- [ ] **Step 6 (optional live check):** with the server's ASR reachable, run a quick real call from a Python REPL using a wav from `tests/data` to confirm end-to-end. Skip in CI.

- [ ] **Step 7: Commit.**

```bash
git add src/diart_server/asr_client.py tests/diart_server/test_asr_client.py setup.cfg
git commit -m "feat(diart_server): add async Qwen3-ASR client"
```

---

## Task 9: Embedding extractor (enrollment)

**Files:**
- Create: `src/diart_server/embedding_extractor.py`
- Test: `tests/diart_server/test_embedding_extractor.py`

**Interfaces:**
- Consumes: `ModelsRegistry.get_embedding(name) -> diart.models.EmbeddingModel`.
- Produces: `make_embedder(model_name, registry, device=None) -> SpeakerEmbedding`; `extract(pcm_16k: np.ndarray, embedder) -> np.ndarray` (1-D, L2-normalized).

- [ ] **Step 1: Write the failing test** — `tests/diart_server/test_embedding_extractor.py`:

```python
import numpy as np
import torch
from diart_server.embedding_extractor import extract


class FakeEmbedder:
    """Mimics diart SpeakerEmbedding: takes (samples, channels) -> (dim,)."""
    def __call__(self, waveform):
        return torch.tensor([3.0, 4.0])  # norm 5 -> expect [0.6, 0.8]


def test_extract_normalizes():
    v = extract(np.zeros(16000, dtype=np.float32), FakeEmbedder())
    np.testing.assert_allclose(v, np.array([0.6, 0.8], dtype=np.float32), atol=1e-6)
    assert v.dtype == np.float32 and v.ndim == 1
```

- [ ] **Step 2: Run test, verify it fails.** Run: `pytest tests/diart_server/test_embedding_extractor.py -v` → FAIL.

- [ ] **Step 3: Implement** `src/diart_server/embedding_extractor.py`:

```python
"""Extract an utterance-level voiceprint from an enrollment clip.

Uses the same diart embedding model as live sessions so enrollment vectors
live in the same space as the streamed centroid embeddings.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from diart.blocks.embedding import SpeakerEmbedding

MAX_ENROLL_SECONDS = 30  # cap clip length fed to the model


def make_embedder(model_name: str, registry, device: Optional[torch.device] = None) -> SpeakerEmbedding:
    emb_model = registry.get_embedding(model_name)
    return SpeakerEmbedding(emb_model, device)


def extract(pcm_16k: np.ndarray, embedder) -> np.ndarray:
    pcm = np.asarray(pcm_16k, dtype=np.float32).reshape(-1)
    pcm = pcm[: MAX_ENROLL_SECONDS * 16000]
    waveform = torch.from_numpy(np.ascontiguousarray(pcm)).reshape(-1, 1)  # (samples, channels)
    vec = embedder(waveform)
    v = vec.detach().cpu().numpy().astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v
```

- [ ] **Step 4: Run test, verify it passes.** Run: `pytest tests/diart_server/test_embedding_extractor.py -v` → PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/diart_server/embedding_extractor.py tests/diart_server/test_embedding_extractor.py
git commit -m "feat(diart_server): add enrollment embedding extractor"
```

---

## Task 10: Voiceprint routes + app wiring

**Files:**
- Create: `src/diart_server/voiceprint_routes.py`
- Modify: `src/diart_server/app.py` (create store/asr_client on startup, mount router)
- Test: `tests/diart_server/test_voiceprint_routes.py`

**Interfaces:**
- Consumes: `VoiceprintStore`, `embedding_extractor.make_embedder/extract`, `audio_decoder`, `ModelsRegistry`.
- Produces: `POST /v1/voiceprints` (multipart `name`,`embedding_model`,`file`) → `EnrollResponse`; `GET /v1/voiceprints?embedding_model=` → `{voiceprints:[VoiceprintInfo]}`; `DELETE /v1/voiceprints/{name}?embedding_model=`. `app.state.store` and `app.state.asr_client` available to sessions.

- [ ] **Step 1: Write the failing test** — `tests/diart_server/test_voiceprint_routes.py`:

```python
import io
import wave
import numpy as np
from fastapi.testclient import TestClient


def _wav_bytes(seconds=1.0, sr=16000):
    int16 = (np.zeros(int(seconds * sr)) ).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(int16.tobytes())
    return buf.getvalue()


def test_enroll_list_delete(monkeypatch, tmp_path):
    # Patch the embedder so no real model is needed.
    import diart_server.voiceprint_routes as vr

    monkeypatch.setattr(vr, "make_embedder", lambda *a, **k: object())
    monkeypatch.setattr(vr, "extract", lambda pcm, emb: np.array([1.0, 0.0], dtype=np.float32))

    from diart_server.voiceprint_store import VoiceprintStore
    from diart_server.app import create_app

    app = create_app()
    app.state.store = VoiceprintStore(tmp_path)
    client = TestClient(app)

    r = client.post(
        "/v1/voiceprints",
        data={"name": "张三", "embedding_model": "speechbrain/ecapa"},
        files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
    )
    assert r.status_code == 200 and r.json()["name"] == "张三"

    r = client.get("/v1/voiceprints", params={"embedding_model": "speechbrain/ecapa"})
    assert r.json()["voiceprints"] == [{"name": "张三", "n_clips": 1}]

    r = client.delete("/v1/voiceprints/张三", params={"embedding_model": "speechbrain/ecapa"})
    assert r.status_code == 200
    assert client.get("/v1/voiceprints", params={"embedding_model": "speechbrain/ecapa"}).json()["voiceprints"] == []
```

> Note: this assumes `app.py` exposes a `create_app()` factory and the router reads `make_embedder`/`extract` at module scope (so they can be monkeypatched). If `app.py` currently builds the app at import, add a thin `create_app()` that returns the configured app and have module-level `app = create_app()`.

- [ ] **Step 2: Run test, verify it fails.** Run: `pytest tests/diart_server/test_voiceprint_routes.py -v` → FAIL.

- [ ] **Step 3: Implement** `src/diart_server/voiceprint_routes.py`:

```python
"""HTTP endpoints for enrolling/listing/removing voiceprints."""
from __future__ import annotations

from fastapi import APIRouter, Form, Request, UploadFile, HTTPException

from diart_server import audio_decoder
from diart_server.embedding_extractor import make_embedder, extract
from diart_server.protocol import EnrollResponse

voiceprint_router = APIRouter()


@voiceprint_router.post("/v1/voiceprints", response_model=EnrollResponse)
async def enroll(
    request: Request,
    name: str = Form(...),
    embedding_model: str = Form(...),
    file: UploadFile = Form(...),
) -> EnrollResponse:
    raw = await file.read()
    try:
        pcm = audio_decoder.to_pcm_16k_mono(raw, 16000)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad audio: {exc}")
    if pcm.size < 16000 // 2:  # < 0.5s
        raise HTTPException(status_code=400, detail="clip too short (need >=0.5s)")
    registry = request.app.state.registry
    embedder = make_embedder(embedding_model, registry)
    vec = extract(pcm, embedder)
    store = request.app.state.store
    store.enroll(embedding_model, name, vec)
    n = next((i["n_clips"] for i in store.list(embedding_model) if i["name"] == name), 1)
    return EnrollResponse(name=name, embedding_model=embedding_model, n_clips=n)


@voiceprint_router.get("/v1/voiceprints")
async def list_voiceprints(request: Request, embedding_model: str) -> dict:
    return {"voiceprints": request.app.state.store.list(embedding_model)}


@voiceprint_router.delete("/v1/voiceprints/{name}")
async def delete_voiceprint(request: Request, name: str, embedding_model: str) -> dict:
    request.app.state.store.remove(embedding_model, name)
    return {"status": "deleted", "name": name}
```

- [ ] **Step 4: Wire into `app.py`.** Ensure a `create_app()` factory exists; inside it (a) mount the router, (b) in the lifespan/startup create the store and ASR client:

```python
# imports
from diart_server.voiceprint_routes import voiceprint_router
from diart_server.voiceprint_store import VoiceprintStore
from diart_server.asr_client import AsrClient
from diart_server.config import settings

# inside create_app(), alongside other router includes:
app.include_router(voiceprint_router)

# inside the lifespan startup (after registry warm-up):
store = VoiceprintStore(settings.voiceprints_dir)
store.load()
app.state.store = store
app.state.asr_client = AsrClient(
    settings.asr_base_url, settings.asr_model, settings.asr_api_key, settings.asr_timeout
)
```

If `app.py` builds `app` at import time, refactor to:
```python
def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    # ... existing mounts ...
    app.include_router(voiceprint_router)
    return app

app = create_app()
```

- [ ] **Step 5: Run test, verify it passes.** Run: `pytest tests/diart_server/test_voiceprint_routes.py -v` → PASS.

- [ ] **Step 6: Run existing app/probe tests.** Run: `pytest tests/diart_server/test_probes.py -v` → PASS (no regression).

- [ ] **Step 7: Commit.**

```bash
git add src/diart_server/voiceprint_routes.py src/diart_server/app.py tests/diart_server/test_voiceprint_routes.py
git commit -m "feat(diart_server): add voiceprint enrollment endpoints"
```

---

## Task 11: Session wiring — matching, aggregation, async ASR, transcript

**Files:**
- Modify: `src/diart_server/session.py`, `src/diart_server/ws_routes.py`
- Test: `tests/diart_server/test_session_transcript.py`

**Interfaces:**
- Consumes: `PcmRingBuffer`, `TurnAggregator`, `voiceprint_matcher.match`, `VoiceprintStore`, `AsrClient`, `settings`.
- Produces: `DiarizationSession(params, registry, on_message, loop, store=None, asr_client=None)`; emits `prediction` (with `name`/`score`) and `transcript` messages.

- [ ] **Step 1: Write the failing test** — `tests/diart_server/test_session_transcript.py`. Mirror the dummy-model construction in `tests/diart_server/test_session.py` (open that file and reuse its fixture/helpers for building a `DiarizationSession` with dummy diart models and feeding PCM). The new assertions:

```python
import asyncio
import numpy as np
import pytest

# Reuse helpers from the existing session test for dummy-model session setup.
from tests.diart_server.test_session import make_session, feed_speech  # adjust to real names

@pytest.mark.asyncio
async def test_finalized_turn_emits_transcript(monkeypatch):
    messages = []

    async def on_message(msg):
        messages.append(msg)

    # Fake ASR so no network is needed.
    from diart_server.asr_client import AsrClient
    async def fake_transcribe(self, pcm, language=None):
        return ("Chinese", "你好世界")
    monkeypatch.setattr(AsrClient, "transcribe", fake_transcribe)

    session = make_session(on_message=on_message, with_asr=True)  # builds dummy-model session
    session.start()
    feed_speech(session)            # push synthetic PCM that yields >=1 speaker turn
    await asyncio.sleep(0.2)
    final = await session.close()   # flush + ASR pending

    types = [m["type"] for m in messages]
    assert "transcript" in types
    t = next(m for m in messages if m["type"] == "transcript")
    assert t["text"] == "你好世界" and t["lang"] == "Chinese"
```

> If `test_session.py` has no reusable `make_session`/`feed_speech`, factor the dummy-model setup it uses into small local helpers in this test file (copy, not import). The point: a real `DiarizationSession` with dummy models, ASR monkeypatched.

- [ ] **Step 2: Run test, verify it fails.** Run: `pytest tests/diart_server/test_session_transcript.py -v` → FAIL.

- [ ] **Step 3: Extend `DiarizationSession.__init__`** (additive params + new collaborators). After existing init:

```python
from diart_server.pcm_ring_buffer import PcmRingBuffer
from diart_server.turn_aggregator import TurnAggregator
from diart_server.voiceprint_matcher import match as vp_match

    def __init__(self, params, registry, on_message, loop, store=None, asr_client=None):
        ...  # existing body unchanged
        self.store = store
        self.asr_client = asr_client
        self.ring = PcmRingBuffer(settings.ring_buffer_seconds, sample_rate=16000)
        self.aggregator = TurnAggregator(settings.silence_gap, settings.max_turn_seconds)
        self._last_match: dict[str, tuple[str | None, float | None]] = {}
        self._asr_sem: asyncio.Semaphore | None = None
        self._asr_tasks: set = set()
```

- [ ] **Step 4: Create the semaphore in `start()`** (it must bind to the loop). Add at end of `start()`:

```python
        self._asr_sem = asyncio.Semaphore(settings.max_asr_concurrency)
```

- [ ] **Step 5: Tap PCM in `push_pcm`:**

```python
    def push_pcm(self, pcm) -> None:
        self.source.push_pcm(pcm)
        self.ring.append(pcm)
```

- [ ] **Step 6: Match + aggregate in `_on_prediction`.** Inside the `for turn, _, speaker in annotation.itertracks(...)` loop, after computing `embedding = self._extract_embedding(speaker)`:

```python
            name, score = None, None
            if self.store is not None and embedding is not None:
                refs = self.store.vectors(self.params.embedding_model)
                name, score = vp_match(embedding, refs, settings.match_threshold)
                self._last_match[speaker] = (name, score)
            msg = PredictionMessage(
                type="prediction",
                start=turn.start,
                duration=turn.duration,
                speaker=speaker,
                embedding=embedding.tolist() if embedding is not None else [],
                samples_count=self._speaker_counts[speaker],
                name=name,
                score=score,
            ).model_dump()
            asyncio.run_coroutine_threadsafe(self.on_message(msg), self.loop)
            # ... existing metrics.predictions_total inc ...

            # aggregate -> dispatch ASR for finalized turns
            now = self.ring.total_seconds
            for ft in self.aggregator.feed(speaker, turn.start, turn.start + turn.duration, now):
                asyncio.run_coroutine_threadsafe(self._handle_turn(ft), self.loop)
```

- [ ] **Step 7: Add `_handle_turn`** (runs on the event loop):

```python
    async def _handle_turn(self, ft) -> None:
        if self.asr_client is None or self._asr_sem is None:
            return
        task = asyncio.current_task()
        self._asr_tasks.add(task)
        try:
            async with self._asr_sem:
                pcm = self.ring.slice(ft.start, ft.end)
                if pcm is None or pcm.size < 16000 // 4:  # <0.25s -> skip
                    return
                name, score = self._last_match.get(ft.speaker, (None, None))
                try:
                    lang, text = await self.asr_client.transcribe(pcm)
                except Exception:  # noqa: BLE001
                    lang, text = "", "[ASR失败]"
                from diart_server.protocol import TranscriptMessage
                msg = TranscriptMessage(
                    speaker=ft.speaker, name=name, score=score,
                    start=ft.start, end=ft.end, text=text, lang=lang,
                ).model_dump()
                await self.on_message(msg)
        finally:
            self._asr_tasks.discard(task)
```

- [ ] **Step 8: Flush in `close()`.** Before building/sending the final summary, flush remaining turns and await pending ASR:

```python
        for ft in self.aggregator.flush(self.ring.total_seconds):
            self._asr_tasks.add(asyncio.ensure_future(self._handle_turn(ft)))
        if self._asr_tasks:
            await asyncio.wait(self._asr_tasks, timeout=settings.asr_timeout + 5)
```

- [ ] **Step 9: Pass collaborators from `ws_routes.py`.** Where `DiarizationSession(...)` is constructed, add:

```python
            store=getattr(websocket.app.state, "store", None),
            asr_client=getattr(websocket.app.state, "asr_client", None),
```

- [ ] **Step 10: Run new + existing session tests.** Run: `pytest tests/diart_server/test_session_transcript.py tests/diart_server/test_session.py tests/diart_server/test_e2e_websocket.py -v` → all PASS (update any strict `prediction` key-set assertion in the existing e2e test to allow the new optional `name`/`score`).

- [ ] **Step 11: Commit.**

```bash
git add src/diart_server/session.py src/diart_server/ws_routes.py tests/diart_server/test_session_transcript.py
git commit -m "feat(diart_server): voiceprint match + per-turn ASR transcript in session"
```

---

## Task 12: Browser `/demo/live` page

**Files:**
- Create: `src/diart_server/static/live.html`
- Modify: `src/diart_server/app.py` (serve `/demo/live`), `setup.cfg` (`package_data` already globs `static/*`)
- Test: `tests/diart_server/test_live_route.py`

**Interfaces:**
- Consumes: `GET /v1/info` (model list), `POST /v1/voiceprints`, WS `/v1/diarize/stream`.
- Produces: a route serving `live.html`.

- [ ] **Step 1: Write the failing test** — `tests/diart_server/test_live_route.py`:

```python
from fastapi.testclient import TestClient
from diart_server.app import create_app


def test_live_page_served():
    client = TestClient(create_app())
    r = client.get("/demo/live")
    assert r.status_code == 200 and "diarize/stream" in r.text
```

- [ ] **Step 2: Run test, verify it fails.** Run: `pytest tests/diart_server/test_live_route.py -v` → FAIL.

- [ ] **Step 3: Add the route** in `app.py` (near the existing `/` redirect / `/demo` static mount):

```python
from pathlib import Path as _P
from fastapi.responses import HTMLResponse

@app.get("/demo/live", response_class=HTMLResponse)
async def demo_live() -> str:
    return (_P(__file__).parent / "static" / "live.html").read_text(encoding="utf-8")
```

(If routes are registered inside `create_app()`, add it there instead.)

- [ ] **Step 4: Create `src/diart_server/static/live.html`** with a minimal working client:

```html
<!DOCTYPE html>
<html lang="zh">
<head><meta charset="UTF-8"><title>Diart · 实时声纹 + ASR</title>
<style>
 body{font:14px/1.5 system-ui;margin:0;background:#0b0d10;color:#e6e8eb}
 .wrap{max-width:900px;margin:0 auto;padding:20px}
 button{background:#6366f1;color:#fff;border:0;border-radius:6px;padding:8px 14px;cursor:pointer}
 button:disabled{opacity:.5}
 select,input{background:#1a1e25;color:#e6e8eb;border:1px solid #262b33;border-radius:6px;padding:6px}
 .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0}
 .line{padding:6px 10px;border-left:3px solid #6366f1;margin:4px 0;background:#14171c;border-radius:4px}
 .name{color:#10b981;font-weight:600} .unk{color:#8b929c} .t{color:#8b929c;font-size:12px}
</style></head>
<body><div class="wrap">
 <h2>实时声纹 + ASR</h2>
 <div class="row">
   embedding: <select id="emb"></select>
   segmentation: <select id="seg"></select>
 </div>
 <div class="row">
   <input id="enrollName" placeholder="注册姓名">
   <button id="enrollBtn">录 4 秒并注册</button>
   <span id="enrollStatus" class="t"></span>
 </div>
 <div class="row">
   <button id="startBtn">开始</button>
   <button id="stopBtn" disabled>停止</button>
   <span id="status" class="t">idle</span>
 </div>
 <div id="timeline"></div>
</div>
<script>
const $=s=>document.querySelector(s);
let ws, audioCtx, node, stream, running=false;

async function loadModels(){
  const info = await (await fetch('/v1/info')).json();
  const emb=$('#emb'), seg=$('#seg');
  for(const [name,kind] of Object.entries(info.models)){
    if(name.includes('titanet')) continue; // hidden: needs nemo
    const o=document.createElement('option'); o.value=name; o.textContent=name;
    (kind==='embedding'?emb:seg).appendChild(o);
  }
  emb.value = [...emb.options].find(o=>o.value.includes('ecapa'))?.value || emb.value;
  seg.value = [...seg.options].find(o=>o.value.includes('segmentation-3.0'))?.value || seg.value;
}

function downsampleSend(buf, inRate){
  // send float32 LE mono at native rate; server resamples to 16k
  if(ws && ws.readyState===1) ws.send(buf.buffer);
}

async function capture(onFrame){
  stream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1}});
  audioCtx = new AudioContext();
  const src = audioCtx.createMediaStreamSource(stream);
  const proc = audioCtx.createScriptProcessor(8192,1,1); // ~0.18s @48k
  proc.onaudioprocess = e => onFrame(new Float32Array(e.inputBuffer.getChannelData(0)));
  src.connect(proc); proc.connect(audioCtx.destination);
  node = proc;
  return audioCtx.sampleRate;
}

function addLine(m){
  const div=document.createElement('div'); div.className='line';
  const who = m.name ? `<span class="name">${m.name}</span>` : `<span class="unk">${m.speaker}</span>`;
  const sc = (m.score!=null)?` (${m.score.toFixed(2)})`:'';
  div.innerHTML = `<span class="t">${m.start.toFixed(1)}–${m.end.toFixed(1)}s</span> ${who}${sc}: ${m.text}`;
  $('#timeline').prepend(div);
}

$('#startBtn').onclick = async ()=>{
  $('#startBtn').disabled=true; $('#stopBtn').disabled=false; running=true;
  const sr0 = await capture(f=>downsampleSend(f, audioCtx.sampleRate));
  const q = new URLSearchParams({sample_rate:String(sr0), embedding_model:$('#emb').value, segmentation_model:$('#seg').value});
  ws = new WebSocket(`ws://${location.host}/v1/diarize/stream?${q}`);
  ws.binaryType='arraybuffer';
  ws.onmessage = ev=>{
    const m=JSON.parse(ev.data);
    if(m.type==='ready') $('#status').textContent='listening · '+m.session_id.slice(0,8);
    else if(m.type==='transcript') addLine(m);
    else if(m.type==='error') $('#status').textContent='error: '+m.message;
  };
};

$('#stopBtn').onclick = ()=>{
  running=false; $('#startBtn').disabled=false; $('#stopBtn').disabled=true;
  if(ws && ws.readyState===1) ws.send(JSON.stringify({type:'end'}));
  if(node) node.disconnect(); if(audioCtx) audioCtx.close();
  if(stream) stream.getTracks().forEach(t=>t.stop());
  $('#status').textContent='stopped';
};

$('#enrollBtn').onclick = async ()=>{
  const name=$('#enrollName').value.trim(); if(!name){$('#enrollStatus').textContent='填姓名';return;}
  $('#enrollStatus').textContent='录音中…';
  const s = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1}});
  const rec = new MediaRecorder(s); const chunks=[];
  rec.ondataavailable=e=>chunks.push(e.data);
  rec.onstop = async ()=>{
    s.getTracks().forEach(t=>t.stop());
    // decode webm -> 16k mono wav via OfflineAudioContext
    const buf = await new Blob(chunks).arrayBuffer();
    const ac = new AudioContext(); const decoded = await ac.decodeAudioData(buf); ac.close();
    const off = new OfflineAudioContext(1, Math.ceil(decoded.duration*16000), 16000);
    const bs = off.createBufferSource(); bs.buffer=decoded; bs.connect(off.destination); bs.start();
    const rendered = await off.startRendering();
    const wav = encodeWav(rendered.getChannelData(0), 16000);
    const fd = new FormData();
    fd.append('name', name); fd.append('embedding_model', $('#emb').value);
    fd.append('file', new Blob([wav],{type:'audio/wav'}), 'enroll.wav');
    const r = await fetch('/v1/voiceprints', {method:'POST', body:fd});
    $('#enrollStatus').textContent = r.ok ? ('已注册 '+name) : ('注册失败 '+r.status);
  };
  rec.start(); setTimeout(()=>rec.stop(), 4000);
};

function encodeWav(f32, sr){
  const buf=new ArrayBuffer(44+f32.length*2), v=new DataView(buf);
  const w=(o,s)=>{for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i))};
  w(0,'RIFF'); v.setUint32(4,36+f32.length*2,true); w(8,'WAVE'); w(12,'fmt ');
  v.setUint32(16,16,true); v.setUint16(20,1,true); v.setUint16(22,1,true);
  v.setUint32(24,sr,true); v.setUint32(28,sr*2,true); v.setUint16(32,2,true); v.setUint16(34,16,true);
  w(36,'data'); v.setUint32(40,f32.length*2,true);
  let o=44; for(let i=0;i<f32.length;i++,o+=2){const s=Math.max(-1,Math.min(1,f32[i]));v.setInt16(o,s*0x7fff,true);}
  return buf;
}

loadModels();
</script>
</body></html>
```

- [ ] **Step 5: Run test, verify it passes.** Run: `pytest tests/diart_server/test_live_route.py -v` → PASS.

- [ ] **Step 6: Manual end-to-end** (real models + real ASR):

```bash
DIART_MODELS_DIR=./models DIART_DEVICE=cuda uvicorn diart_server.app:app --host 0.0.0.0 --port 7007
# open http://localhost:7007/demo/live  → 选 ecapa → 注册自己 → 开始 → 说话
```
Expected: timeline shows `[你的名字 t0–t1] <转写文本>`; unenrolled speakers show `speakerN`. Tune `DIART_MATCH_THRESHOLD` using the printed score.

- [ ] **Step 7: Commit.**

```bash
git add src/diart_server/static/live.html src/diart_server/app.py tests/diart_server/test_live_route.py
git commit -m "feat(diart_server): add /demo/live mic + voiceprint + ASR page"
```

---

## Self-Review (done while writing)

- **Spec coverage:** §3 modules → Tasks 4–9; §4.1 protocol → Task 3; §4.2 session → Task 11; §4.3 routes → Task 10; §4.4 config → Task 2; §4.5 frontend → Task 12; §6 per-model namespacing → Tasks 6/10/11; §8 model localization + MODELS.md → Task 1; §9 error handling → Task 11 step 7 (`[ASR失败]`) + Task 10 (400s); §10 testing → each task's tests. All sections covered.
- **Type consistency:** `match() -> (str|None, float)` used identically in Tasks 5 & 11; `extract(pcm, embedder)` signature consistent Tasks 9 & 10; `PcmRingBuffer.total_seconds`/`slice` consistent Tasks 4 & 11; `TranscriptMessage` fields consistent Tasks 3 & 11.
- **Known follow-ups for the implementer:** (a) confirm `app.py` exposes `create_app()` — Task 10 step 4 adds it if missing; (b) reuse the real dummy-model helper names from `tests/diart_server/test_session.py` in Task 11 step 1; (c) relax any strict key-set assertion on `prediction` in the existing e2e test (Task 11 step 10).
