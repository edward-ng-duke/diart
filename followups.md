# Followups

> Issues, deferrals, and assumptions logged during execution of [PLAN.md](PLAN.md).
> Severity: **Critical** (spec breach / failing test / security) > **Medium** (cleanup / refactor) > **Low** (cosmetic / nice-to-have).
> `manual:` tag = needs human action (browser, GPU host, HF token, etc).

## Critical

_None yet._

## Medium

- **T3** — Implementer added `[tool.pytest.ini_options]` block to `pyproject.toml` (out-of-scope per spec) to resolve a package name collision: spec required creating `tests/diart_server/__init__.py`, which under pytest's default `prepend` import mode shadows the real `src/diart_server/` package. Fix: `--import-mode=importlib` + `pythonpath = ["src", "tests"]`. Existing 14 diart tests still pass. Alternative would have been to skip `__init__.py` (pytest auto-discovers subdir tests). Decision: keep the pytest config change; it benefits all subsequent test tasks (T4-T33). Confirm with user that this is acceptable.

## Low

- **T3** — `pyproject.toml` `pythonpath = ["src", "tests"]` could be just `["tests"]` since editable install puts src on sys.path. Or refactor `tests/utils.py` to `tests/utils/__init__.py` and drop pythonpath entirely. Minor cleanup.
- **T3** — 20 parametrized error tests (3 × 6 + 2) overlap; could be ~8 without losing coverage. Three parametrizations all iterate over the same 6 subclasses.
- **T3** — `to_message(self) -> dict` could be tightened to `dict[str, str]` for stricter typing. mypy --strict warns. T4 introduced `ErrorMessage` Pydantic schema that may supersede this.
- **T4** — `PredictionMessage` numeric fields lack `ge=0` constraints on `duration` / `samples_count`. Defense-in-depth nice-to-have. Also no `models`/`config`/`stats` schema specificity — defer to T9/T11/T13.
- **T4** — Discriminated-union helper (`Annotated[Union[EndControl], Field(discriminator="type")]`) would simplify T17/T22 client-control dispatch when more control types are added. 2-line forward-compat hook.
- **T4** — No coercion test for `parse_query({"sample_rate":"22050"})` (Starlette gives strings); add to T17 e2e or here as belt-and-suspenders.
- **T6** — Add early-return guard `if self._closed.is_set(): return` at top of `push_pcm` to prevent post-close pushes from inflating the prometheus overflow counter (and from leaking memory if the source is recycled). 2-line fix.
- **T6** — Wrap `on_overflow` callback invocation in try/except + log so a buggy callback doesn't kill the websocket handler. Defense-in-depth.
- **T6** — Class docstring should explicitly state single-producer/single-consumer concurrency model. Multi-producer would require a lock around overflow branch.
- **T6** — STANDALONE_SERVER.md §5.4 spec snippet has two latent bugs (bool _closed; blocking put(None) on close) that the implementation correctly fixed. Update spec to match.
- **T9** — `ModelsRegistry` cache writes are not protected by a Lock. With the GIL `dict[k]=v` is atomic but check-then-set is a TOCTOU window. For ONNX models (which eagerly load the ORT session), two racing sessions could materialize duplicate sessions briefly. Add `threading.Lock` around `get_*` check-then-set in T20 if multi-connection load testing reveals issues.
- **T9** — `from_pyannote` only catches HTTPError/ModuleNotFoundError; corrupted local cache for speechbrain/nemo will surface as raw exception at first `.to(device)` or `.eval()` call, not at `get_*` time. Document this in class docstring or wrap in `MODEL_LOAD_FAILED` error code. Defer to T15 lifespan/warm-up integration.
- **T9** — `warm_up` only loads defaults; non-default models pay first-request latency. Add note to docstring so T15/T17 implementers know.
- **T10** — `__init__` builds heavy GPU-allocating modules (segmentation+embedding) before constructing MeetingAudioSource; if any later step throws, GPU tensors are orphaned to GC. T13's close() should be designed defensively to handle partial-init via `_partial_init` sentinel.
- **T10** — VAD rejection raises before `self.id` is assigned, so log line in T17 won't have session id. Either assign id first or include `params.pipeline` in error message.
- **T11** — `_run` crash path increments `session_errors_total` but does NOT decrement `active_sessions`. Resulting in a leak on every inference crash. T13's close() must handle this — also need T13 to call `metrics.active_sessions.dec()` from crash path of `_run`, not just normal close.
- **T11** — `run_coroutine_threadsafe` future is discarded; if the websocket loop is closing or `on_message` raises (e.g., `ConnectionClosed`), the exception is buried. Add `add_done_callback` logging in T13.
- **T11** — `attach_observers` is called in `__init__` but `attach_hooks` is called in `start`. Asymmetric. Move both to one place (T13 cleanup is a good time).
- **T12** — `embedding=[]` when extraction returns None is ambiguous to clients. Better: change `PredictionMessage.embedding` to `Optional[list[float]]` and emit `None`. Requires T4 schema bump.
- **T12** — Pipeline `_clustering` fallback is dead code (current diart uses public `clustering`). Drop the fallback in cleanup pass. Plan said `_clustering`; spec was wrong about diart internals.
- **T13** — `_thread.join(5.0)` returns silently on timeout; add `if self._thread.is_alive(): logger.warning(...)` and a `CLOSE_TIMEOUT` error code increment. Daemon thread leak is process-bounded but observability matters.
- **T13** — `stats={"predictions_count": ...}` actually counts per-turn increments (one per (annotation, turn) pair), not per emitted prediction message. Field name is ambiguous — consider renaming to `turns_count` or documenting in protocol.FinalMessage.
- **T14** — Test teardown should call `loop.run_until_complete(loop.shutdown_default_executor())` before `loop.close()` to avoid silent drops of straggler `run_coroutine_threadsafe` calls. Latent flakiness risk.
- **T14** — Snapshot of `messages` is taken AFTER scheduling `close()` future result; race risk where late predictions arrive after snapshot. Currently masked by 2.0s sleep but not deterministic. Either snapshot before close, or have close() await pending on_message futures.
- **T14** — Embedding-length assertion is necessarily relaxed (128–512 range) due to DummyEmbeddingModel emitting random dim per call. Add per-speaker dim-consistency check to retoothen.
- **T14** — Replace fixed `time.sleep(2.0)` with `asyncio.Event` triggered when prediction count threshold reached. Reduces 6.75s runtime.
- **T14** — `sys.path` munging at module top; replace with `tests/diart_server/conftest.py` re-export of dummies, or move dummies to shared `tests/dummy_models.py`.
- **T19** — Assertion `total_inference_frames >= 1 or final_msg["num_speakers"] >= 0` is effectively unconditional (num_speakers is always ≥ 0). Tighten to drop the `or` clause to actually enforce "got something between ready and final".
- **T19** — Relies on Starlette's private `_send_queue` for non-blocking message drain. Brittle across Starlette versions. Replace if Starlette ships a public timeout API.
- **T20** — `/readyz` still reads `metrics.active_sessions._value.get()` rather than `await manager.has_capacity()`. Consistency cleanup; manager is now source of truth.

## Assumptions Made (need confirmation)

_None yet._

## Manual followups (deferred to user)

- **T24 ✅ validated** — Ran `python docker/download_models.py /tmp/test_models` with real HF_TOKEN. 9 of 10 models downloaded + smoke-loaded. Issues found and fixed in commit `6704172` (path resolution for pyannote checkpoints, ONNX filename, partial-download cleanup).
- ~~**T24 manual (residual):** `pyannote/segmentation` (v1) is gated.~~ ✅ License accepted; v1 segmentation downloaded (17MB) + smoke-loaded + e2e-tested (7 predictions, 512-dim L2-normalized embeddings, 30ms/chunk).
- **T25/T27 manual:** Docker build requires HF_TOKEN as a build secret. CI pipeline must inject it via `--secret id=hf_token` (BuildKit) or a build arg.

## Verified (manual)

- ✅ `python docker/download_models.py /tmp/test_models` produces all **10/10** models (~550MB total). Per-backend completeness check correctly handles partial downloads.
- ✅ Runtime `ModelsRegistry` loads all 5 backends from local paths (pyannote / speechbrain / nemo / onnx) — verified by importing each and running a forward pass through `pyannote/segmentation-3.0` and `pyannote/segmentation` (v1).
- ✅ End-to-end TestClient WebSocket flow against REAL models with both **segmentation-3.0** (default) and **segmentation v1** (explicit query): 7 predictions emitted in each run, 512-dim L2-normalized embeddings (|emb|=1.0000), num_speakers=1 on single-speaker sample.wav, RTTM correctly rendered. Inference latency: 33ms/chunk (3.0) and 30ms/chunk (v1) — matches diart README CPU benchmark.
