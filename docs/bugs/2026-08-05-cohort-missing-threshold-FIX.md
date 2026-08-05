# FIX report: cohort-missing embedding models under the centered-score threshold

- **Date:** 2026-08-05
- **Fixer:** fixer agent (Reporter -> Fixer -> Reviewer loop)
- **Base:** a6128f1 — **Fix commits:** 73aeabc (fix), 9885cfd (tests)
- **Spec:** docs/bugs/2026-08-05-cohort-missing-threshold.md (design §4 implemented as written)

## 1. Changes

### src/diart_server/cohort.py (new)
- `_COHORT_DIR`, `_cohort_slug`, `_load_cohort_mean` moved verbatim from
  `session.py` (byte-identical logic and names; the
  `Path(__file__).resolve().parents[2] / "assets" / "cohort"` expression
  resolves to the same repo `assets/cohort/` directory from this module).
  Module-level `logger` kept.
- New `has_cohort(model_name)`: True iff `assets/cohort/<slug>.npy` exists.

### src/diart_server/session.py
- Local cohort definitions replaced by
  `from diart_server.cohort import _load_cohort_mean`; the test monkeypatch
  target `diart_server.session._load_cohort_mean` is unchanged in spelling.
  Now-unused `pathlib.Path` import dropped.
- Constructor: when `store is not None and self.cohort_mean is None`, ONE
  WARNING naming the session id, the embedding model, and
  `settings.match_threshold` — per-turn identification is DISABLED.
- `_transcribe_and_emit`: identify gate extended to
  `self.store is not None and self.cohort_mean is not None and dur >=
  settings.min_identify_seconds`. A cohort-less session skips
  `_identify_slice` entirely: `name=None`, `score=None`, no `_last_match`
  write, no embedder construction, no executor hop.
- `ready_message()`: additive `config` key
  `"voiceprint": "enabled" | "disabled_no_cohort"`, keyed on `cohort_mean`
  alone (`ReadyMessage.config` is `dict[str, Any]` — no schema change).

### src/diart_server/ws_routes.py
- `/v1/info` gains additive `voiceprint_ready`: sorted embedding-model names
  for which `has_cohort()` is true. Dev layout: exactly
  `["pyannote/embedding", "pyannote/wespeaker-voxceleb-resnet34-LM"]`.
  The `models` map shape is untouched.

### src/diart_server/static/live.html
- `loadModels()`: builds `vpReady` from `info.voiceprint_ready` (a Set, or
  `null` when the field is absent — older-server fallback keeps today's
  behaviour) and hides embedding options not in the set. Dropdown now lists 2
  embedding models instead of 7; both preferred defaults are in the ready set,
  so the default-selection block is unchanged. Segmentation options unchanged.

### src/diart_server/voiceprint_matcher.py
- `match()` validates the mean shape ONCE, up front: a `mean_embedding` whose
  flattened shape differs from the query's is set to `None` before any
  centering, so the whole call (query, every reference, threshold comparison)
  runs the exact historical raw path — no centering, no `(c+1)/2` remap.
  Pre-fix, `_centered_unit`'s per-vector fallback returned the raw unit while
  `match()` still applied the remap, presenting a raw cosine on the centered
  scale: an impostor at raw 0.66 scored 0.830 >= 0.82 (spec §3 fact 3).
  `_centered_unit` itself is unchanged; mixed-shape references keep the
  historical skip-don't-score behaviour (`r.shape != u.shape` guard).
  Module and `match()` docstrings document the mismatch rule.
- `mean_embedding=None` callers keep exact historical behaviour (same floats;
  pinned by test, see §2.1).

### Constants / calibration
- No new constants and no recalibration: `match_threshold` stays 0.82 and
  `min_identify_seconds` stays 2.0 (both calibrated for the centered score in
  `2026-07-06-voiceprint-false-accept.md`; this fix makes the code refuse to
  apply that threshold to any other scale). The only new values are the two
  literal strings `"enabled"` / `"disabled_no_cohort"` in the ready config
  and the `voiceprint_ready` field name.

### Tests
- `test_voiceprint_matcher.py`: +3 — wrong-shape mean equals raw mode
  float-for-float (`==`, no tolerance) at thresholds 0.0 and 0.82
  (fixed-seed rng, dim 32, 3 refs, mean `np.zeros(dim + 7)`); correct-shape
  mean: all 4 candidate scores in [0, 1] and the returned score equals the
  manual `(cos(unit(unit(v)-m), unit(unit(r)-m)) + 1) / 2` within 1e-6;
  regression pin — impostor ref at raw cosine 0.66 with a wrong-shape mean
  returns `name=None` at 0.82 with score 0.66 (pre-fix: `name=impostor` at
  0.830).
- `test_probes.py`: +1 — `voiceprint_ready` pins the dev-layout pair; the
  existing 10-model `models`-map assertions are untouched and passing.
- `test_session_cohort_disable.py` (new, patterned on
  `test_session_transcript.py`; fakes only): cohort-less `dummy/emb` session —
  transcript emitted with `name is None and score is None`, spies on
  `session._identify_slice` AND `diart_server.session
  .extract_utterance_embedding` never called, `_last_match == {}`, ready
  config `voiceprint == "disabled_no_cohort"`, exactly one construction
  WARNING containing `dummy/emb` (caplog, filtered to the
  `diart_server.session` logger); with `_load_cohort_mean` monkeypatched to a
  unit vector — `voiceprint == "enabled"`, stubbed `_identify_slice` called
  once, transcript `name == "alice"`, `score == 0.9`.
- `test_live_route.py`: +1 — `/demo/live` HTML references `voiceprint_ready`.
- **Pre-existing tests updated (spec'd behaviour change):**
  `test_session_transcript.py`'s four identity-path tests
  (`test_turn_identity_from_own_slice_not_cluster_cache`,
  `test_short_turn_gets_no_name_even_with_cached_match`,
  `test_identify_failure_fails_closed_but_emits_transcript`,
  `test_absorbed_fragment_reidentifies_merged_span`) construct `dummy/emb`
  sessions with a store; under the new gate identification would be skipped
  before the path under test is reached. Each now calls a new `_with_cohort`
  helper (monkeypatches `diart_server.session._load_cohort_mean` to a unit
  vector before construction). Two of them (`test_turn_identity...`,
  `test_absorbed_fragment...`) FAILED post-fix without this; the other two
  passed but only vacuously (the cohort gate, not the duration gate /
  fail-closed path, produced their expected `None`s). Assertions unchanged.

## 2. Verification results (all run post-fix, 2026-08-05, this machine)

### 2.1 Matcher scale laws (spec §5.1) — PASS
Covered by the 3 new matcher tests: wrong-shape == raw exact (`==` on the
tuple, thresholds 0.0 and 0.82); centered scores in [0, 1] and equal to the
manual computation within 1e-6; 0.66-impostor pin returns `(None, 0.66)` at
threshold 0.82.

### 2.2 /v1/info (spec §5.2) — PASS
`python -m pytest tests/diart_server/test_probes.py -q -p no:cacheprovider`
-> `8 passed in 0.28s`, including the new pin
`voiceprint_ready == ["pyannote/embedding", "pyannote/wespeaker-voxceleb-resnet34-LM"]`.

### 2.3 Session disable semantics + live.html (spec §5.3/§5.4) — PASS
Combined targeted run
(`test_voiceprint_matcher.py test_probes.py test_live_route.py
test_session_cohort_disable.py`): `22 passed in 0.44s`.

### 2.4 Full unit suite (the ONLY acceptance command) — PASS

```
python -m pytest tests/diart_server -q \
  --ignore=tests/diart_server/test_e2e_websocket.py \
  --ignore=tests/diart_server/test_end_control.py \
  --ignore=tests/diart_server/test_failure_modes.py \
  -p no:cacheprovider
-> 156 passed, 1 skipped in 73.76s
```

Baseline at a6128f1: 149 passed, 1 skipped (~74s). +7 new tests
(3 matcher + 1 probes + 2 session-cohort + 1 live-route), 0 regressions.

### 2.5 Cross-matrix regression gate (matching math touched) — PASS

`DIART_MODELS_DIR=./models DIART_DEVICE=cuda DIART_VOICEPRINTS_DIR=./voiceprints
python scripts/verify_speaker_cross_matrix.py`

```
FALSE_ACCEPTS=0 (of 15 impostor turns)
MISSED_GENUINE=0 (of 3 asserted genuine turns)
SESSION_ERRORS=0
RESULT: PASS      (exit 0; identical turn matrix to the a6128f1 baseline,
                   e.g. S4 genuine newsA 0.992 accepted, newsB 0.662 and
                   cross-channel 0.793 rejected)
```

Expected: the gate runs `pyannote/wespeaker-voxceleb-resnet34-LM`, whose
cohort exists and matches its 256-d space, so both the session gate and the
matcher shape check are no-ops there.

### 2.6 Invariant audit (spec §5.7) — PASS by inspection
The diff only ADDS a gating condition (`self.cohort_mean is not None`) to the
existing identify gate: identity still comes exclusively from a turn's own
overlap-excised, re-embedded audio via `_identify_slice`; the cohort-less
path emits `name=None` (no previous-turn or centroid fallback — `_last_match`
is not even written); the fail-closed `try/except` around identification and
the unconditional transcript emission are untouched.

## 3. Residual risk

- The 5 cohort-less models are now hidden/disabled rather than fixed;
  generating their cohorts (needs checkpoints + the calibration pool) is the
  spec'd follow-up, out of scope here (spec §4.6).
- Enrollment on a cohort-less model remains allowed by design: harmless now,
  useful the moment a cohort ships.
- An older client (cached live.html) against the new server keeps the old
  7-model dropdown, but such sessions now visibly report
  `voiceprint: disabled_no_cohort` and never pretend to identify.

## Review
pending
