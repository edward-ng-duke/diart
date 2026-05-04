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

## Assumptions Made (need confirmation)

_None yet._

## Manual followups (deferred to user)

_None yet._
