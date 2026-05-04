# Diart Standalone WebSocket Server: Implementation Plan

> **Source spec:** [STANDALONE_SERVER.md](STANDALONE_SERVER.md)
> **Approved:** 2026-05-04
> **Effort:** ~12 person-days, 33 tasks
> **Execution model:** dispatch one sub-agent per task; orchestrator (this conversation) reviews diff between tasks.

## Goal

Add a new offline-capable, multi-connection FastAPI WebSocket server to the diart repo (in `src/diart_server/`) that wraps diart for use as a containerized service. Outputs `(speaker_label, raw_embedding)` per prediction; voiceprint matching stays in the caller.

## Architecture

- New module `src/diart_server/` (**diart library zero changes**)
- FastAPI + Starlette WebSocket
- Per-connection `DiarizationSession` with daemon thread running `StreamingInference`
- Models bundled at build time, fully offline runtime (`HF_HUB_OFFLINE=1`)
- Two Docker images: `cpu` (~2GB) and `gpu` (~5–7GB)

## Tech Stack

Python 3.11 · FastAPI + Uvicorn · diart 0.9.2 (this repo) · pyannote.audio · torch / torchaudio · onnxruntime · pydantic 2.x + pydantic-settings · prometheus_client · pytest + pytest-asyncio + websockets

## Reusable Existing Components

| What | Where | Used in tasks |
|---|---|---|
| `DummySegmentationModel` / `DummyEmbeddingModel` fixtures | `/home/edward/research/diart/tests/conftest.py:1-58` | T9, T14 |
| Sample WAV (~937KB, 16k mono) | `/home/edward/research/diart/tests/data/audio/sample.wav` | T19, T21, T26, T30 |
| `SpeakerDiarization` / `Config` | `/home/edward/research/diart/src/diart/blocks/diarization.py` | T10–T13 |
| `AudioSource` base class | `/home/edward/research/diart/src/diart/sources.py:18-47` | T6 |
| `StreamingInference` + `attach_hooks` | `/home/edward/research/diart/src/diart/inference.py` | T11 |
| `OnlineSpeakerClustering.centers` (for embedding extraction) | `/home/edward/research/diart/src/diart/blocks/clustering.py` | T12 |
| `PredictionAccumulator` | `/home/edward/research/diart/src/diart/sinks.py` | T13 |
| `SegmentationModel.from_pyannote/from_onnx` | `/home/edward/research/diart/src/diart/models.py:142-198` | T9 |
| `EmbeddingModel.from_pyannote/from_onnx` | `/home/edward/research/diart/src/diart/models.py:201-265` | T9 |

## Dependency Graph

```
T1 ──┬─ T2 ─┐
     ├─ T3 ─┤
     ├─ T4 ─┤
     ├─ T5 ─┤── T10 ─ T11 ─ T12 ─ T13 ─ T14 ──┐
     ├─ T6 ─┤                                  │
     ├─ T7 ─┤                                  ├── T15 ─ T16 ─ T17 ─ T18 ─ T19 ──┐
     ├─ T8 ─┤                                  │                                   │
     └─ T9 ─┘                                  │                                   ├── T20 ─ T21 ──┐
                                                                                                    │
                                                          T22 ─ T23 ──────────────────────────────┤
                                                                                                    │
                                                                              T24 ─ T25 ─ T26 ─ T27 ┤
                                                                                                    │
                                                                                  T28 ─ T29 ─ T30 ──┤
                                                                                                    │
                                                                                        T31 ─ T32 ─ T33
```

---

# Phase A — Foundation (after T1, T2-T9 can parallelize)

## T1 — Scaffold `diart_server` package + extras_require
**Complexity: S | Deps: none**

**Files:**
- Create: `/home/edward/research/diart/src/diart_server/__init__.py`
- Modify: `/home/edward/research/diart/setup.cfg` (add `server` extra under `[options.extras_require]`)

**In-scope:**
- Create empty `__init__.py` with `__version__ = "0.1.0"`
- Add `server` extra to `setup.cfg`:
  ```ini
  server=
      fastapi>=0.110.0,<1.0.0
      uvicorn[standard]>=0.27.0,<1.0.0
      websockets>=12.0
      pydantic>=2.5.0,<3.0.0
      pydantic-settings>=2.1.0,<3.0.0
      prometheus-client>=0.20.0
  ```

**Out-of-scope:**
- Do NOT modify existing `install_requires`, `tests` extra, or any other field in setup.cfg
- Do NOT add CPU/GPU torch wheel selection here (that's in Dockerfile)
- Do NOT add any imports in `__init__.py`

**Verification:**
```bash
cd /home/edward/research/diart
pip install -e ".[server]" 2>&1 | tail -3
# expect: "Successfully installed ..." with fastapi, uvicorn, etc.
python -c "import diart_server; print(diart_server.__version__)"
# expect: 0.1.0
```

---

## T2 — `config.py`: env-driven settings
**Complexity: S | Deps: T1**

**Files:**
- Create: `/home/edward/research/diart/src/diart_server/config.py`

**In-scope:**
Implement a `Settings` class via `pydantic_settings.BaseSettings` with these fields (env names match):
| Field | Env var | Type | Default |
|---|---|---|---|
| `port` | `DIART_PORT` | int | 7007 |
| `host` | `DIART_HOST` | str | "0.0.0.0" |
| `device` | `DIART_DEVICE` | str | "cpu" |
| `models_dir` | `DIART_MODELS_DIR` | Path | Path("/app/models") |
| `max_concurrent_sessions` | `DIART_MAX_CONCURRENT_SESSIONS` | int | 8 |
| `max_frame_bytes` | `DIART_MAX_FRAME_BYTES` | int | 10485760 |
| `log_level` | `DIART_LOG_LEVEL` | str | "INFO" |

Export module-level singleton `settings = Settings()`.

**Out-of-scope:**
- Do NOT touch `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` here (set in Dockerfile)
- Do NOT add file/disk I/O at import time

**Verification:**
```bash
DIART_PORT=8888 python -c "from diart_server.config import settings; assert settings.port == 8888; print('ok')"
# expect: ok
python -c "from diart_server.config import settings; assert settings.port == 7007; print('ok')"
# expect: ok
```

---

## T3 — `errors.py`: error codes + exceptions
**Complexity: S | Deps: T1**

**Files:**
- Create: `/home/edward/research/diart/src/diart_server/errors.py`
- Create: `/home/edward/research/diart/tests/diart_server/__init__.py` (empty)
- Create: `/home/edward/research/diart/tests/diart_server/test_errors.py`

**In-scope:**
Define an `ErrorCode` `StrEnum` (or Enum on Python 3.11) with values: `BAD_AUDIO`, `MODEL_NOT_FOUND`, `RESAMPLE_FAILED`, `INFERENCE_CRASH`, `RATE_LIMIT_EXCEEDED`, `SESSION_CAPACITY_FULL`, `INTERNAL_ERROR`.

Define exception class hierarchy:
```python
class DiartServerError(Exception):
    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    def to_message(self) -> dict:
        return {"type": "error", "code": self.code.value, "message": str(self)}

class BadAudioError(DiartServerError): code = ErrorCode.BAD_AUDIO
class ModelNotFoundError(DiartServerError): code = ErrorCode.MODEL_NOT_FOUND
class ResampleFailedError(DiartServerError): code = ErrorCode.RESAMPLE_FAILED
class InferenceCrashError(DiartServerError): code = ErrorCode.INFERENCE_CRASH
class RateLimitExceededError(DiartServerError): code = ErrorCode.RATE_LIMIT_EXCEEDED
class SessionCapacityFullError(DiartServerError): code = ErrorCode.SESSION_CAPACITY_FULL
```

Test cases:
- Each exception subclass: `assert isinstance(e, DiartServerError)` and `e.code` is correct
- `to_message()` returns dict with right shape

**Out-of-scope:**
- Do NOT raise these anywhere yet
- Do NOT add logging here

**Verification:**
```bash
cd /home/edward/research/diart
pytest tests/diart_server/test_errors.py -v
# expect: all tests pass
```

---

## T4 — `protocol.py`: Pydantic message schemas
**Complexity: M | Deps: T1, T3**

**Files:**
- Create: `/home/edward/research/diart/src/diart_server/protocol.py`
- Create: `/home/edward/research/diart/tests/diart_server/test_protocol.py`

**In-scope:**
Implement Pydantic v2 schemas for ALL messages (per [STANDALONE_SERVER.md §4.2](STANDALONE_SERVER.md)):
- `ReadyMessage` (type, session_id, sample_rate, pipeline, models, config)
- `PredictionMessage` (type, start, duration, speaker, embedding: list[float], samples_count: int)
- `FinalMessage` (type, rttm, duration, num_speakers, stats)
- `ErrorMessage` (type, code: ErrorCode, message)
- `EndControl` (type) — for client-side `{"type":"end"}`
- `QueryParams` model with all 12 query fields + validators (sample_rate ∈ [8000, 48000], latency ∈ [0.5, 5.0], max_speakers ∈ [2, 20], etc.)

Add `parse_query(qp: dict) -> QueryParams` function that wraps validation errors as `BadAudioError`.

Test cases (≥10 tests):
- Round-trip serialization for each message type
- `QueryParams.parse({})` → defaults match spec
- `QueryParams.parse({"sample_rate": 22050})` → succeeds
- `QueryParams.parse({"sample_rate": 4000})` → ValidationError → BadAudioError
- `QueryParams.parse({"latency": 6.0})` → fails
- `EndControl` correctly differentiates from binary frames

**Out-of-scope:**
- Do NOT integrate with FastAPI here (kept pure)
- Do NOT use these schemas in routes yet (T17)

**Verification:**
```bash
cd /home/edward/research/diart
pytest tests/diart_server/test_protocol.py -v
# expect: all tests pass
```

---

## T5 — `audio_decoder.py`: PCM resampling
**Complexity: S | Deps: T1, T3**

**Files:**
- Create: `/home/edward/research/diart/src/diart_server/audio_decoder.py`
- Create: `/home/edward/research/diart/tests/diart_server/test_audio_decoder.py`

**In-scope:**
Implement `to_pcm_16k_mono(frame_bytes: bytes, src_sr: int) -> np.ndarray`:
1. Parse `np.frombuffer(frame_bytes, dtype=np.float32)`. If empty, raise `BadAudioError("empty frame")`
2. If `src_sr != 16000`, run `torchaudio.functional.resample` and return numpy
3. Return float32 array of shape `(samples,)`

Tests (using synthetic numpy arrays, no I/O):
- `to_pcm_16k_mono(silence_16k_500ms.tobytes(), 16000)` → shape `(8000,)`
- `to_pcm_16k_mono(silence_48k_500ms.tobytes(), 48000)` → shape ~`(8000,)` (within ±20 samples)
- `to_pcm_16k_mono(b"", 16000)` → raises `BadAudioError`
- 22050 Hz input → 16000 output, length ratio correct

**Out-of-scope:**
- Stereo handling (mono-only this iteration; if frame is stereo PCM we crash — fix in later task)
- WebM/Opus decoding (assume client sent raw PCM)
- Error path for partial float frames (assume aligned)

**Verification:**
```bash
cd /home/edward/research/diart
pytest tests/diart_server/test_audio_decoder.py -v
# expect: all 4+ tests pass
```

---

## T6 — `meeting_audio_source.py`: thread-safe AudioSource
**Complexity: M | Deps: T1**

**Files:**
- Create: `/home/edward/research/diart/src/diart_server/meeting_audio_source.py`
- Create: `/home/edward/research/diart/tests/diart_server/test_meeting_audio_source.py`

**In-scope:**
Implement `MeetingAudioSource(AudioSource)` per [STANDALONE_SERVER.md §5.4](STANDALONE_SERVER.md):
- `__init__(uri, sample_rate=16000, maxsize=50)` → thread-safe queue
- `push_pcm(pcm: np.ndarray)` → put_nowait; on Full, drop oldest and re-put (counter increments via callback `on_overflow` injected via constructor — keep metrics decoupled)
- `read()` → blocking loop, queue.get; sentinel `None` to stop; emits `torch.from_numpy(pcm).unsqueeze(0)` via `self.stream.on_next`
- `close()` → set flag, push None

Tests:
- Push 5 frames, simulate `read()` in another thread, assert all 5 received in order
- Fill queue to maxsize+1, assert oldest dropped, `on_overflow` callback called once
- Call `close()`, assert `read()` exits cleanly within 100ms

**Out-of-scope:**
- Real diart pipeline integration (T11)
- Metrics increments inside this class (use callback)

**Verification:**
```bash
cd /home/edward/research/diart
pytest tests/diart_server/test_meeting_audio_source.py -v
# expect: 3+ tests pass
```

---

## T7 — `metrics.py`: Prometheus metric definitions
**Complexity: S | Deps: T1**

**Files:**
- Create: `/home/edward/research/diart/src/diart_server/metrics.py`

**In-scope:**
Module-level metric instances per [STANDALONE_SERVER.md §5.9](STANDALONE_SERVER.md):
```python
from prometheus_client import Counter, Gauge, Histogram

active_sessions = Gauge("diart_active_sessions", "Number of active WebSocket sessions")
predictions_total = Counter("diart_predictions_total", "...", ["pipeline", "model"])
prediction_latency = Histogram("diart_prediction_latency_seconds", "...")
audio_bytes_received = Counter("diart_audio_bytes_received_total", "...")
session_duration = Histogram("diart_session_duration_seconds", "...")
model_load_seconds = Gauge("diart_model_load_seconds", "...", ["model"])
session_errors_total = Counter("diart_session_errors_total", "...", ["code"])
queue_overflow_total = Counter("diart_queue_overflow_total", "...")
```

**Out-of-scope:**
- Do NOT instrument session/routes yet (T28)

**Verification:**
```bash
python -c "from diart_server.metrics import active_sessions; active_sessions.set(3); print('ok')"
# expect: ok
```

---

## T8 — `docker/models.json`: model manifest
**Complexity: S | Deps: none**

**Files:**
- Create: `/home/edward/research/diart/docker/models.json`

**In-scope:**
Write JSON array with all 10 models (see [STANDALONE_SERVER.md §6.1](STANDALONE_SERVER.md)). Verify each `hf_name` against [README.md §"Pre-trained models"](README.md).

**Out-of-scope:**
- Download script (T24)
- Dockerfile (T25)

**Verification:**
```bash
cd /home/edward/research/diart
python -c "import json; m = json.load(open('docker/models.json')); assert len(m) == 10; assert all('hf_name' in e and 'kind' in e and 'backend' in e for e in m); print(f'ok, {len(m)} models')"
# expect: ok, 10 models
```

---

## T9 — `models_registry.py`: lazy model loading
**Complexity: M | Deps: T1, T8**

**Files:**
- Create: `/home/edward/research/diart/src/diart_server/models_registry.py`
- Create: `/home/edward/research/diart/tests/diart_server/test_models_registry.py`

**In-scope:**
Implement `ModelsRegistry` per [STANDALONE_SERVER.md §5.5](STANDALONE_SERVER.md):
- `__init__(models_dir: Path, manifest_path: Path)` → load manifest into `self.specs: dict[str, ModelSpec]`
- `get_segmentation(name: str) -> SegmentationModel` (lazy + cached)
- `get_embedding(name: str) -> EmbeddingModel` (lazy + cached)
- `warm_up(device: torch.device, default_seg: str, default_emb: str)` — load defaults synchronously
- Resolve local path: `models_dir / spec.local_path`
- Backend dispatch: `pyannote` → `from_pyannote(local_path, use_hf_token=False)`; `onnx` → `from_onnx(local_path)`

Tests (use `tests/conftest.py` fixtures `segmentation_model` / `embedding_model` indirectly; mock `SegmentationModel.from_pyannote` and `from_onnx` to avoid real network):
- `get_segmentation` for unknown name → `ModelNotFoundError`
- Caching: two calls return same instance
- Backend dispatch: pyannote spec → calls `from_pyannote`; onnx spec → calls `from_onnx`

**Out-of-scope:**
- Real model download (use mocks)
- Warm-up integration (T15)

**Verification:**
```bash
cd /home/edward/research/diart
pytest tests/diart_server/test_models_registry.py -v
# expect: 3+ tests pass
```

---

# Phase B — DiarizationSession (sequential)

## T10 — `session.py` skeleton + `__init__`
**Complexity: M | Deps: T2, T4, T6, T7, T9**

**Files:**
- Create: `/home/edward/research/diart/src/diart_server/session.py`

**In-scope:**
Define `DiarizationSession` class with:
- `__init__(self, params: QueryParams, registry: ModelsRegistry, on_message: Callable[[dict], Awaitable[None]], loop: asyncio.AbstractEventLoop)` — only initialize fields, don't start anything
- Build `SpeakerDiarizationConfig` from `params`
- Build `pipeline = SpeakerDiarization(config)` using models from `registry`
- Build `source = MeetingAudioSource(uri=self.id)` (with `on_overflow=lambda: metrics.queue_overflow_total.inc()`)
- Build `inference = StreamingInference(pipeline, source, do_plot=False, show_progress=False)`
- Build `accumulator = PredictionAccumulator(self.id)` and `inference.attach_observers(accumulator)`
- Set `self._thread = threading.Thread(target=self._run, daemon=True)` (don't start)
- Set `self._speaker_counts: dict[str, int] = defaultdict(int)`
- Stub methods `start`, `push_pcm`, `close`, `_run`, `_on_prediction` (raise NotImplementedError)
- Public method `ready_message() -> dict` returns `ReadyMessage(...)` payload

**Out-of-scope:**
- `start` / `push_pcm` / `close` implementation (T11, T13)
- `_on_prediction` callback (T12)
- `run_coroutine_threadsafe` wiring (T11)

**Verification:**
```bash
cd /home/edward/research/diart
python -c "
from diart_server.session import DiarizationSession
import inspect
assert inspect.isclass(DiarizationSession)
sig = inspect.signature(DiarizationSession.__init__)
assert {'params','registry','on_message','loop'} <= set(sig.parameters)
print('ok')
"
```

---

## T11 — `session.py`: `start` + `push_pcm` + `_run` + thread bridge
**Complexity: M | Deps: T10**

**Files:**
- Modify: `/home/edward/research/diart/src/diart_server/session.py`

**In-scope:**
Implement:
```python
def start(self):
    self.inference.attach_hooks(self._on_prediction)
    self._thread.start()
    metrics.active_sessions.inc()
    self._started_at = time.monotonic()

def push_pcm(self, pcm: np.ndarray):
    self.source.push_pcm(pcm)
    metrics.audio_bytes_received.inc(pcm.nbytes)

def _run(self):
    try:
        self.inference()
    except Exception as exc:
        msg = {"type":"error","code":"INFERENCE_CRASH","message":str(exc)}
        asyncio.run_coroutine_threadsafe(self.on_message(msg), self.loop)
        metrics.session_errors_total.labels(code="INFERENCE_CRASH").inc()

def _on_prediction(self, ann_wav):
    # Stub for T12 - just print or pass for now
    pass
```

**Out-of-scope:**
- `_on_prediction` real implementation (T12)
- `close` method (T13)

**Verification:**
```bash
cd /home/edward/research/diart
python -c "
from diart_server.session import DiarizationSession
assert hasattr(DiarizationSession, 'start')
assert hasattr(DiarizationSession, 'push_pcm')
assert hasattr(DiarizationSession, '_run')
print('ok')
"
```

---

## T12 — `session.py`: `_on_prediction` + embedding extraction
**Complexity: M | Deps: T11**

**Files:**
- Modify: `/home/edward/research/diart/src/diart_server/session.py`

**In-scope:**
Implement `_on_prediction(self, ann_wav)`:
- `annotation, waveform = ann_wav`
- For each `turn, _, speaker in annotation.itertracks(yield_label=True)`:
  - `self._speaker_counts[speaker] += 1`
  - Extract embedding: access `self.pipeline._clustering.centers[label_index]` (the running speaker centroid). Determine `label_index` via `self.pipeline._clustering.active_centers` mapping (read-only access, do NOT mutate).
  - Build `PredictionMessage` payload (use `protocol.PredictionMessage(...).model_dump()`)
  - `asyncio.run_coroutine_threadsafe(self.on_message(payload), self.loop)`
  - `metrics.predictions_total.labels(pipeline=self.params.pipeline, model=self.params.embedding_model).inc()`

Helper method `_extract_embedding(speaker_label: str) -> list[float]` to keep `_on_prediction` clean.

**Note:** `OnlineSpeakerClustering.centers` is at `/home/edward/research/diart/src/diart/blocks/clustering.py:44-46` — read this file before implementing to confirm attribute names.

**Out-of-scope:**
- `close` (T13)
- Optimizations (e.g., emit only on speaker change)

**Verification:**
Inspect that `_extract_embedding` returns a list of 512 floats (sanity check via assertion in code is fine; full e2e in T19).

```bash
cd /home/edward/research/diart
python -c "
import inspect
from diart_server.session import DiarizationSession
src = inspect.getsource(DiarizationSession._on_prediction)
assert 'itertracks' in src
assert '_extract_embedding' in src or 'centers' in src
assert 'run_coroutine_threadsafe' in src
print('ok')
"
```

---

## T13 — `session.py`: `close` + final message
**Complexity: M | Deps: T12**

**Files:**
- Modify: `/home/edward/research/diart/src/diart_server/session.py`

**In-scope:**
Implement:
```python
async def close(self) -> dict:
    self.source.close()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, self._thread.join, 5.0)
    metrics.active_sessions.dec()
    metrics.session_duration.observe(time.monotonic() - self._started_at)
    prediction = self.accumulator.get_prediction()
    return FinalMessage(
        type="final",
        rttm=prediction.to_rttm() if prediction else "",
        duration=prediction.get_timeline().extent().end if prediction else 0.0,
        num_speakers=len(prediction.labels()) if prediction else 0,
        stats={"predictions_count": sum(self._speaker_counts.values())},
    ).model_dump()
```

**Out-of-scope:**
- SessionManager registration/deregistration (T20)

**Verification:**
```bash
python -c "
import inspect
from diart_server.session import DiarizationSession
sig = inspect.signature(DiarizationSession.close)
assert inspect.iscoroutinefunction(DiarizationSession.close)
print('ok')
"
```

---

## T14 — `session.py` integration test (with dummy models)
**Complexity: M | Deps: T13**

**Files:**
- Create: `/home/edward/research/diart/tests/diart_server/test_session.py`

**In-scope:**
Reuse `DummySegmentationModel` / `DummyEmbeddingModel` from `tests/conftest.py:1-58`. Build a `DiarizationSession`, pump synthetic PCM (random noise float32 16k), let inference run for ~3s of audio, assert:
- `start()` doesn't raise
- After pushing 6 × 0.5s frames, `_speaker_counts` has at least one entry
- `on_message` callback was invoked at least once with a `prediction` payload containing 512-element `embedding`
- `close()` returns dict with `type=="final"` and rttm string

**Out-of-scope:**
- Real pyannote models (use dummies)
- WebSocket integration (T19)

**Verification:**
```bash
cd /home/edward/research/diart
pytest tests/diart_server/test_session.py -v -s
# expect: pass within 30s
```

---

# Phase C — HTTP / WebSocket layer

## T15 — `app.py`: FastAPI app + lifespan + warm-up
**Complexity: S | Deps: T2, T9**

**Files:**
- Create: `/home/edward/research/diart/src/diart_server/app.py`

**In-scope:**
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging, torch

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=settings.log_level)
    app.state.registry = ModelsRegistry(settings.models_dir, Path(__file__).parent.parent.parent / "docker/models.json")
    app.state.registry.warm_up(
        device=torch.device(settings.device),
        default_seg="pyannote/segmentation-3.0",
        default_emb="pyannote/embedding",
    )
    yield
    # graceful shutdown - drained sessions in T20

app = FastAPI(title="diart-server", version="0.1.0", lifespan=lifespan)
```

Don't include routers yet (T16/T17 will add them via `app.include_router`).

**Out-of-scope:**
- Routers (T16, T17)
- Session draining on shutdown (T20)

**Verification:**
```bash
cd /home/edward/research/diart
DIART_MODELS_DIR=/tmp/missing python -c "from diart_server.app import app; print(app.title)" 2>&1 | tail -3
# expect: "diart-server" (lifespan only runs on startup, not import)
```

---

## T16 — `ws_routes.py`: probes (`/healthz`, `/readyz`, `/v1/info`, `/metrics`)
**Complexity: S | Deps: T7, T15**

**Files:**
- Create: `/home/edward/research/diart/src/diart_server/ws_routes.py`
- Modify: `/home/edward/research/diart/src/diart_server/app.py` (`app.include_router(probes_router)`)
- Create: `/home/edward/research/diart/tests/diart_server/test_probes.py`

**In-scope:**
- `probes_router = APIRouter()`
- `GET /healthz` → `{"status":"ok"}` (always 200)
- `GET /readyz` → check `app.state.registry` warm-up done, `len(active_sessions) < settings.max_concurrent_sessions`; return 200 or 503
- `GET /v1/info` → `{"version":"0.1.0","pipelines":["diarization","vad"],"models":{...},"device":settings.device,"max_concurrent_sessions":...}`
- `GET /metrics` → `prometheus_client.generate_latest()` with `Content-Type: text/plain; version=0.0.4`

Tests use FastAPI's `TestClient` (sync) for the four endpoints.

**Out-of-scope:**
- WebSocket route (T17)

**Verification:**
```bash
cd /home/edward/research/diart
pytest tests/diart_server/test_probes.py -v
# expect: 4+ tests pass
```

---

## T17 — `ws_routes.py`: WebSocket endpoint accept + query parsing + capacity check
**Complexity: M | Deps: T4, T13, T16**

**Files:**
- Modify: `/home/edward/research/diart/src/diart_server/ws_routes.py` (add `ws_router`)

**In-scope:**
```python
ws_router = APIRouter()

@ws_router.websocket("/v1/diarize/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    try:
        params = QueryParams.model_validate(dict(websocket.query_params))
    except ValidationError as e:
        await websocket.send_json({"type":"error","code":"BAD_AUDIO","message":str(e)})
        await websocket.close(code=1008); return

    # capacity check (using metrics.active_sessions for now; T20 introduces SessionManager)
    if metrics.active_sessions._value.get() >= settings.max_concurrent_sessions:
        await websocket.send_json({"type":"error","code":"SESSION_CAPACITY_FULL","message":"server at capacity"})
        await websocket.close(code=1013); return

    session = DiarizationSession(
        params=params, registry=websocket.app.state.registry,
        on_message=websocket.send_json, loop=asyncio.get_event_loop(),
    )
    session.start()
    await websocket.send_json(session.ready_message())
    # main loop & close in T18
```

Register router in `app.py` via `app.include_router(ws_router)`.

**Out-of-scope:**
- The receive loop and close handling (T18)

**Verification:**
```bash
cd /home/edward/research/diart
python -c "
from diart_server.app import app
routes = [r for r in app.routes if hasattr(r,'path') and r.path.startswith('/v1/')]
assert any('stream' in r.path for r in routes)
print('ok')
"
```

---

## T18 — `ws_routes.py`: WebSocket main loop + close
**Complexity: M | Deps: T5, T17**

**Files:**
- Modify: `/home/edward/research/diart/src/diart_server/ws_routes.py`

**In-scope:**
After `await websocket.send_json(session.ready_message())`:
```python
try:
    async for raw in websocket.iter_bytes():
        if len(raw) > settings.max_frame_bytes:
            await websocket.send_json({"type":"error","code":"RATE_LIMIT_EXCEEDED","message":"frame too large"})
            await websocket.close(code=1009); break
        try:
            pcm = audio_decoder.to_pcm_16k_mono(raw, params.sample_rate)
        except (BadAudioError, ResampleFailedError) as e:
            await websocket.send_json(e.to_message())
            continue   # keep connection open; next frame may be valid
        session.push_pcm(pcm)
except WebSocketDisconnect:
    pass
finally:
    final = await session.close()
    try: await websocket.send_json(final)
    except: pass
```

**Note:** `iter_bytes` returns binary frames. Text frames (e.g., `{"type":"end"}`) need separate handling — for this task ignore text frames; T22 adds end-control handling.

**Out-of-scope:**
- `{"type":"end"}` control (T22)
- SessionManager registration (T20)

**Verification:**
```bash
cd /home/edward/research/diart
python -c "
import inspect
from diart_server import ws_routes
src = inspect.getsource(ws_routes)
assert 'iter_bytes' in src
assert 'WebSocketDisconnect' in src
assert 'session.close()' in src
print('ok')
"
```

---

## T19 — End-to-end test: single-connection WS with real wav
**Complexity: M | Deps: T18**

**Files:**
- Create: `/home/edward/research/diart/tests/diart_server/test_e2e_websocket.py`

**In-scope:**
- Use `pytest-asyncio` + `websockets` client lib + `uvicorn.Server` programmatic start
- Audio source: `tests/data/audio/sample.wav` (already exists)
- Use **dummy models** by monkeypatching `models_registry.ModelsRegistry.warm_up` and `get_segmentation/get_embedding` to return `DummySegmentationModel/DummyEmbeddingModel` (avoid HF download in CI)
- Test:
  - Start server on random port
  - Connect WS, expect `ready` message
  - Stream wav as 0.5s float32 frames
  - Assert ≥ 5 prediction messages received
  - Each prediction has 512-element embedding (or whatever dummy embedding dim is — adjust assertion)
  - Disconnect, expect `final` message with non-empty rttm

**Out-of-scope:**
- Real pyannote models (use dummies; full e2e against real models is in T26 smoke test)
- Multi-connection (T21)

**Verification:**
```bash
cd /home/edward/research/diart
pytest tests/diart_server/test_e2e_websocket.py -v -s
# expect: pass within 60s
```

---

# Phase D — Multi-connection

## T20 — `SessionManager`: capacity tracking + draining
**Complexity: M | Deps: T19**

**Files:**
- Create: `/home/edward/research/diart/src/diart_server/session_manager.py`
- Modify: `/home/edward/research/diart/src/diart_server/session.py` (call manager on start/close)
- Modify: `/home/edward/research/diart/src/diart_server/ws_routes.py` (use manager.has_capacity())
- Modify: `/home/edward/research/diart/src/diart_server/app.py` (`lifespan` shutdown drains via manager.close_all)

**In-scope:**
```python
class SessionManager:
    def __init__(self, max_concurrent: int):
        self._sessions: dict[str, DiarizationSession] = {}
        self._lock = asyncio.Lock()
        self._max = max_concurrent

    async def has_capacity(self) -> bool: ...
    async def register(self, session) -> None: ...
    async def unregister(self, session_id) -> None: ...
    async def close_all(self, timeout: float) -> None: ...
```

Module-level singleton: `manager = SessionManager(settings.max_concurrent_sessions)`.

**Out-of-scope:**
- Cross-container coordination (out of scope)

**Verification:**
```bash
cd /home/edward/research/diart
python -c "
from diart_server.session_manager import manager
import asyncio
async def t():
    assert await manager.has_capacity()
    print('ok')
asyncio.run(t())
"
```

---

## T21 — Multi-connection load test
**Complexity: M | Deps: T20**

**Files:**
- Create: `/home/edward/research/diart/tests/diart_server/test_load_4_concurrent.py`

**In-scope:**
- Start server (with dummy models), open 4 simultaneous WebSocket connections
- Each pushes the same `sample.wav`
- Assert each receives ≥ 3 predictions
- Assert speaker labels in session A do NOT appear in session B's predictions (i.e., they're independent — speaker_0 in each session is local)
- Assert all 4 close cleanly with `final` message

**Out-of-scope:**
- Real performance benchmarks (T30)

**Verification:**
```bash
cd /home/edward/research/diart
pytest tests/diart_server/test_load_4_concurrent.py -v -s
# expect: pass within 90s
```

---

# Phase E — Protocol polish

## T22 — End-control + multi-model query support
**Complexity: M | Deps: T18**

**Files:**
- Modify: `/home/edward/research/diart/src/diart_server/ws_routes.py`

**In-scope:**
- In the WS loop, additionally listen for text frames via `websocket.receive()` (use `receive` method directly to detect type). Hint: Starlette's `iter_bytes` only iterates bytes; need `while True: msg = await websocket.receive(); if msg["type"]=="websocket.disconnect": break; if "bytes" in msg: ...; if "text" in msg: parse JSON, handle EndControl`
- On `{"type":"end"}`: drain queue, await final, send final, close
- Multi-model: ensure `params.segmentation_model` / `params.embedding_model` are passed through `ModelsRegistry.get_*`; if not in manifest → `ModelNotFoundError` → send error and close 1008

Add test `tests/diart_server/test_end_control.py` covering:
- Send a frame, then send `{"type":"end"}`, expect `final` then close
- Connect with `?embedding_model=speechbrain/spkrec-ecapa-voxceleb`, expect `ready` confirms model name
- Connect with `?embedding_model=does/not/exist`, expect `error` `MODEL_NOT_FOUND` and close

**Out-of-scope:**
- Real model swap test (mock dispatch)

**Verification:**
```bash
cd /home/edward/research/diart
pytest tests/diart_server/test_end_control.py -v
# expect: 3+ tests pass
```

---

## T23 — Failure mode test suite
**Complexity: M | Deps: T22**

**Files:**
- Create: `/home/edward/research/diart/tests/diart_server/test_failure_modes.py`

**In-scope:**
Cover each row of [STANDALONE_SERVER.md §9](STANDALONE_SERVER.md):
- Empty frame → `BAD_AUDIO`, connection stays open
- Frame > 10MB → `RATE_LIMIT_EXCEEDED`, close 1009
- Sample rate out of range (query `?sample_rate=4000`) → close on accept with `BAD_AUDIO`
- Unknown model name → `MODEL_NOT_FOUND`
- 9th concurrent connection (`MAX_CONCURRENT_SESSIONS=8`) → `SESSION_CAPACITY_FULL`, close 1013
- Inference crash (force by patching `StreamingInference.__call__` to raise) → `INFERENCE_CRASH` message + close

**Out-of-scope:**
- 60s keepalive timeout (out of scope; rely on TCP keepalive)

**Verification:**
```bash
cd /home/edward/research/diart
pytest tests/diart_server/test_failure_modes.py -v
# expect: 6+ tests pass
```

---

# Phase F — Docker CPU + offline

## T24 — `docker/download_models.py`
**Complexity: M | Deps: T8**

**Files:**
- Create: `/home/edward/research/diart/docker/download_models.py`

**In-scope:**
Exact code per [STANDALONE_SERVER.md §6.2](STANDALONE_SERVER.md). Read `docker/models.json`, loop and download each via `huggingface_hub.snapshot_download` (pyannote/speechbrain/nemo) or `hf_hub_download` (onnx). Validate by attempting `Model.from_pretrained(local_path, use_auth_token=False)` on pyannote ones. Skip if already present.

**Out-of-scope:**
- Dockerfile (T25)

**Verification:**
Test offline by running locally with HF_TOKEN set (one-shot):
```bash
export HF_TOKEN=<token>
mkdir -p /tmp/test_models
cd /home/edward/research/diart
python docker/download_models.py /tmp/test_models 2>&1 | tee /tmp/dl.log
# expect: 10 [download] lines, no errors
ls /tmp/test_models | head -20
# expect: pyannote, speechbrain, nvidia, onnx subdirectories
du -sh /tmp/test_models
# expect: ~470MB
```

If HF_TOKEN unavailable, mock the download functions and verify control flow.

---

## T25 — `docker/Dockerfile.cpu` + `entrypoint.sh`
**Complexity: M | Deps: T24, T23**

**Files:**
- Create: `/home/edward/research/diart/docker/Dockerfile.cpu`
- Create: `/home/edward/research/diart/docker/entrypoint.sh`

**In-scope:**
Exact contents per [STANDALONE_SERVER.md §6.3 and §6.5](STANDALONE_SERVER.md). Multi-stage: builder downloads models with `--mount=type=secret,id=hf_token`; runtime `python:3.11-slim` + system deps (ffmpeg, libsndfile1) + copy site-packages and models.

**Out-of-scope:**
- GPU image (T27)

**Verification:**
```bash
cd /home/edward/research/diart
echo $HF_TOKEN > /tmp/hf_token
DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile.cpu \
  --secret id=hf_token,src=/tmp/hf_token \
  -t diart-server:cpu-test . 2>&1 | tail -20
# expect: "Successfully tagged diart-server:cpu-test"
docker images diart-server:cpu-test
# expect: SIZE < 2.5GB
```

---

## T26 — `docker/smoke_test.sh` + offline run
**Complexity: S | Deps: T25**

**Files:**
- Create: `/home/edward/research/diart/docker/smoke_test.sh`

**In-scope:**
Exact script per [STANDALONE_SERVER.md §6.6](STANDALONE_SERVER.md). Offline `--network none` run, wait readyz, query info, push wav as quick check.

**Out-of-scope:**
- GPU smoke (T27)

**Verification:**
```bash
chmod +x /home/edward/research/diart/docker/smoke_test.sh
bash /home/edward/research/diart/docker/smoke_test.sh diart-server:cpu-test
# expect: "[smoke] OK" exit 0
```

---

# Phase G — Docker GPU

## T27 — `docker/Dockerfile.gpu` + smoke
**Complexity: M | Deps: T25**

**Files:**
- Create: `/home/edward/research/diart/docker/Dockerfile.gpu`

**In-scope:**
Same structure as `Dockerfile.cpu` but `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` base + `apt install python3.11 python3.11-dev` + torch CUDA wheel `--index-url https://download.pytorch.org/whl/cu121` + `ENV DIART_DEVICE=cuda`.

**Out-of-scope:**
- nvidia-runtime install verification (assume host has it)

**Verification:**
```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.gpu --secret id=hf_token,src=/tmp/hf_token -t diart-server:gpu-test . 2>&1 | tail -20
# expect: success, image size < 8GB
# If host has nvidia-runtime:
docker run --rm --gpus all diart-server:gpu-test python -c "import torch; print(torch.cuda.is_available())"
# expect: True
```

If no GPU available locally, skip the GPU smoke and note "GPU smoke deferred to CI".

---

# Phase H — Observability + soak

## T28 — Wire all Prometheus metrics
**Complexity: M | Deps: T22**

**Files:**
- Modify: `/home/edward/research/diart/src/diart_server/session.py` (already partly instrumented in T11/T12/T13; add `prediction_latency` measurements wrapping `_on_prediction`)
- Modify: `/home/edward/research/diart/src/diart_server/ws_routes.py` (increment `audio_bytes_received` per frame received; `session_errors_total.labels(code=...)` on each error)

**In-scope:**
Audit all 8 metrics from `metrics.py` are incremented at appropriate points. For `prediction_latency`, measure from `push_pcm` enqueue to `_on_prediction` first call (use timing in `MeetingAudioSource` or session-level).

Add test `tests/diart_server/test_metrics.py`:
- After running e2e once, assert `active_sessions == 0`, `predictions_total > 0`, `audio_bytes_received_total > 0`

**Out-of-scope:**
- Detailed performance dashboards

**Verification:**
```bash
cd /home/edward/research/diart
pytest tests/diart_server/test_metrics.py -v
# expect: pass
```

---

## T29 — Soak test (skipped by default)
**Complexity: M | Deps: T21**

**Files:**
- Create: `/home/edward/research/diart/tests/diart_server/test_soak.py`

**In-scope:**
Mark with `@pytest.mark.skip(reason="manual soak test - run with --run-soak")` or use a custom marker. Test:
- 4 connections, each pushes 1 hour of looped sample.wav at real-time
- Assert no OOM (RSS growth < 200MB), no thread leak, no >1% queue overflow

**Out-of-scope:**
- CI integration (manual / nightly only)

**Verification:**
```bash
cd /home/edward/research/diart
pytest tests/diart_server/test_soak.py -v -s --run-soak  # if marker; otherwise manually drop the skip and run
# expect: pass within 1h05m
```

---

## T30 — Performance baseline (single + 4-conn)
**Complexity: M | Deps: T29**

**Files:**
- Create: `/home/edward/research/diart/tests/diart_server/test_perf_baseline.py`

**In-scope:**
Push 60s of `sample.wav` (real-time, 0.5s frames @ 0.5s intervals), measure:
- p50 / p95 / p99 of `push_pcm` → `prediction` latency
- Compare 1-conn vs 4-conn: latency degradation < 50%
- Print as table; assert p99 < 1.0s (single conn, dummy models)

**Out-of-scope:**
- Benchmark on real models (run separately as a manual evaluation)

**Verification:**
```bash
cd /home/edward/research/diart
pytest tests/diart_server/test_perf_baseline.py -v -s
# expect: pass; print latency table
```

---

# Phase I — Release

## T31 — GitHub Actions CI
**Complexity: M | Deps: T26, T27**

**Files:**
- Create: `/home/edward/research/diart/.github/workflows/build-images.yml`

**In-scope:**
Exact YAML per [STANDALONE_SERVER.md §11.1](STANDALONE_SERVER.md). Two jobs:
- `build-cpu` on ubuntu-latest: build, smoke (`--network none`), push to ghcr.io
- `build-gpu` on `[self-hosted, gpu]` (or `ubuntu-latest` with `qemu` if no GPU runner — note in workflow comment)

Triggers: `push` to main, tags `v*`, manual `workflow_dispatch`.

**Out-of-scope:**
- Image scanning / SBOM
- Multi-arch (amd64 only this iteration)

**Verification:**
```bash
# Lint locally with actionlint or via push
cd /home/edward/research/diart
actionlint .github/workflows/build-images.yml || echo "(install actionlint or push to PR for CI to validate)"
```

---

## T32 — `src/diart_server/README.md`: usage docs
**Complexity: S | Deps: T22**

**Files:**
- Create: `/home/edward/research/diart/src/diart_server/README.md`

**In-scope:**
- Quick start: `pip install -e ".[server]"; uvicorn diart_server.app:app`
- Docker: `docker run --rm -p 7007:7007 ghcr.io/.../diart-server:cpu`
- Protocol: link to STANDALONE_SERVER.md §4
- Config: env var table
- Examples: WebSocket client snippet (Python)

**Out-of-scope:**
- Translating to other languages

**Verification:**
Human readability check — ensure all links work and code blocks are runnable.

---

## T33 — Client example for mygpt integration
**Complexity: M | Deps: T22**

**Files:**
- Create: `/home/edward/research/diart/examples/diart_server_client.py`

**In-scope:**
Standalone Python script:
- Reads a wav file (default: `tests/data/audio/sample.wav`)
- Connects to `ws://localhost:7007/v1/diarize/stream`
- Streams in 0.5s chunks at real-time pace
- Prints each `prediction` with speaker + first 5 embedding values
- Prints final RTTM at end

Add minimal CLI: `python examples/diart_server_client.py --wav <path> --host localhost --port 7007`.

**Out-of-scope:**
- mygpt-side voiceprint matching code (lives in mygpt repo, not here)

**Verification:**
```bash
cd /home/edward/research/diart
# In one terminal: docker run -p 7007:7007 diart-server:cpu-test
# In another:
python examples/diart_server_client.py --wav tests/data/audio/sample.wav 2>&1 | head -20
# expect: "ready" message, several "prediction" lines with embedding, "final" with RTTM
```

---

# Self-Review

## Spec coverage check (against [STANDALONE_SERVER.md](STANDALONE_SERVER.md))

| Spec section | Covered by |
|---|---|
| §1 目标与范围 | All tasks |
| §2 架构 | T10–T20 |
| §3 项目布局 | T1, T8, all module tasks |
| §4 WebSocket 协议 | T4, T17, T18, T22 |
| §5.1 app.py | T15 |
| §5.2 ws_routes.py | T16, T17, T18, T22 |
| §5.3 session.py | T10–T13 |
| §5.4 meeting_audio_source.py | T6 |
| §5.5 models_registry.py | T9 |
| §5.6 audio_decoder.py | T5 |
| §5.7 protocol.py | T4 |
| §5.8 config.py | T2 |
| §5.9 metrics.py | T7, T28 |
| §6 镜像构建 | T8, T24, T25, T26, T27 |
| §7 配置与运行 | T2, T15 |
| §8 可观测性 | T7, T16, T28 |
| §9 失效模式 | T23 |
| §10 测试计划 | T3, T4, T5, T6, T9, T14, T16, T19, T21, T22, T23, T28, T29, T30 |
| §11 CI / 发布 | T31 |
| §13 风险 | Mitigations baked into individual tasks (T9 caching, T20 capacity, T23 failure modes) |

No gaps found.

## Type consistency check
- `QueryParams` (T4) used in `DiarizationSession.__init__` (T10) — consistent
- `SegmentationModel` / `EmbeddingModel` from diart's existing `src/diart/models.py` (T9 imports, T10 uses) — consistent
- `ModelsRegistry.get_segmentation/get_embedding` (T9) called in T10 — consistent
- `on_message` callback signature `Callable[[dict], Awaitable[None]]` (T10, T17) — consistent
- `ErrorCode` enum (T3) used in `to_message()` (T3) and routes (T17, T18) — consistent

## Placeholder scan
No "TBD", "TODO", or "implement later" found in any task.
