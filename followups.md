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

## Assumptions Made (need confirmation)

_None yet._

## Manual followups (deferred to user)

_None yet._
