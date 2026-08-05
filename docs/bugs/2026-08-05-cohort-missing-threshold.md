# Cohort-missing embedding models silently fall back to raw cosine under the centered-score threshold — matching is mathematically impossible

- **Date:** 2026-08-05
- **Reporter:** bug-reporter agent (Reporter -> Fixer -> Reviewer loop)
- **HEAD at investigation:** a6128f1
- **Component:** `diart_server` voiceprint matching (`/demo/live`), model selection
- **Severity:** High — 5 of the 7 embedding models offered in the demo dropdown can
  never identify anyone; a related scale bug can also *inflate* impostor scores.
- **Status:** Root cause confirmed in current code; scale math reproduced offline
  (pure numpy, deterministic — see §3).

## 1. Symptom (user-visible, in the `/demo/live` demo)

The embedding-model dropdown offers 7 models. Pick any of the 5 non-pyannote ones
(e.g. `speechbrain/spkrec-ecapa-voxceleb`), enroll your voice, speak: enrollment
succeeds, diarization and transcripts work, but **every turn shows the bare
`speakerN` label forever — the enrolled name never appears**, regardless of how
long or clearly you speak. There is no error, no UI hint, and the only log line is
an easily-missed INFO. The user's natural conclusion is "voiceprints are broken",
when in fact the selected model was never capable of matching under the current
threshold.

## 2. Root cause (file:line, current code at a6128f1)

### 2.1 Primary: raw-cosine fallback scored against a centered-scale threshold

1. **The dropdown offers models that have no cohort mean.**
   `docker/models.json` declares **8** `kind=="embedding"` models.
   `src/diart_server/static/live.html:44-57` (`loadModels()`) populates the
   dropdown from `/v1/info`'s `models` map and hides only titanet
   (live.html:48) -> **7 visible**. But `assets/cohort/*.npy` exists for exactly
   **2**: `pyannote/embedding` and `pyannote/wespeaker-voxceleb-resnet34-LM`.
   The other **5 visible** models have no cohort file:
   `hbredin/wespeaker-voxceleb-resnet34-LM`, `speechbrain/spkrec-xvect-voxceleb`,
   `speechbrain/spkrec-ecapa-voxceleb`, `speechbrain/spkrec-ecapa-voxceleb-mel-spec`,
   `speechbrain/spkrec-resnet-voxceleb`.
2. **Missing cohort -> silent raw-cosine mode.**
   `src/diart_server/session.py:84-101` (`_load_cohort_mean`) returns `None` when
   the `.npy` is absent, logging only at **INFO** level (session.py:92).
   `session.py:234-236` stores that `None` as `self.cohort_mean`;
   `session.py:710-715` (`_identify_slice`) passes it to `vp_match`.
3. **Raw mode scores live in a different numeric universe than the threshold.**
   `src/diart_server/voiceprint_matcher.py:74-100` (`match`): with
   `mean_embedding=None` the score is the plain cosine of L2-normed vectors in
   `[-1, 1]`. But `settings.match_threshold = 0.82`
   (`src/diart_server/config.py:55-57`) was calibrated **exclusively for the
   mean-centered score remapped to [0, 1]** (`(c+1)/2`, matcher lines 90-92) —
   see the calibration record in
   `docs/bugs/2026-07-06-voiceprint-false-accept.md` §3.3/§4.1(e).
   Measured raw same-speaker cosines in these embedding spaces are far below
   0.82: **0.327** whole-utterance (production wespeaker model, real audio,
   `docs/bugs/2026-06-30-voiceprint-score-compression.md` §b) and **0.51-0.62**
   live (same doc §a). The genuine speaker falls short of the threshold by
   ~**0.49** — a match can essentially never fire. Every turn emits `name=None`.

### 2.2 Related scale bug: shape-mismatched mean remaps a RAW cosine onto the centered scale

`voiceprint_matcher.py:44-47` (`_centered_unit`): when the cohort mean's shape
does not match the vector (wrong cohort for this model — e.g. a 256-d wespeaker
mean against a 192-d ecapa embedding), it silently **falls back to the raw unit
vector**. But `match()` (voiceprint_matcher.py:90-92) still applies the
`(c+1)/2` remap because `mean_embedding is not None`. The result is a raw cosine
presented as a centered score: it is **inflated** by the remap, so an impostor
with raw cosine 0.66 scores **0.83 >= 0.82** — the fallback path can turn
"cannot match" into "false accept" (measured in §3, fact 3). Nothing hits this
path today (each cohort file matches its own model's dimension), but it is one
misnamed `.npy` away.

## 3. Evidence (measured, deterministic, no GPU needed)

Repro script (pure numpy, drives the real `voiceprint_matcher.match`):
`/tmp/claude-1000/-home-edward-research-diart/6fa6a523-2deb-44f0-9b8b-14031b757d63/scratchpad/repro_b3.py`.
Output at HEAD a6128f1:

```
embedding models in manifest : 8
visible in /demo/live dropdown: 7
cohort .npy files             : 2  ['pyannote/embedding', 'pyannote/wespeaker-voxceleb-resnet34-LM']
visible WITHOUT cohort        : 5
    hbredin/wespeaker-voxceleb-resnet34-LM
    speechbrain/spkrec-ecapa-voxceleb
    speechbrain/spkrec-ecapa-voxceleb-mel-spec
    speechbrain/spkrec-resnet-voxceleb
    speechbrain/spkrec-xvect-voxceleb

raw same-speaker cosine (measured, wespeaker): 0.327
threshold applied to it                      : 0.82
genuine match fires: False   (shortfall 0.493)
match() raw mode -> name=None score=0.327  (None == never matches)

mean=None        -> name=None score=0.3270
mean=WRONG-SHAPE -> name=None score=0.6635  (should equal raw, got (raw+1)/2 = 0.6635)
impostor raw cos 0.66 + wrong-shape mean -> name=impostor score=0.830 (>= 0.82: FALSE ACCEPT)
```

- **Fact 1 (inventory):** 5 of 7 selectable models have no cohort file; each such
  session runs raw cosine (code path §2.1 steps 2-3).
- **Fact 2 (impossibility):** the measured raw same-speaker cosine (0.327,
  `2026-06-30-voiceprint-score-compression.md`, production model + real audio)
  is 0.49 below the 0.82 threshold; `match()` reproduces `(None, 0.327)`.
  Even the *live* same-speaker raw range 0.51-0.62 never reaches 0.82.
- **Fact 3 (shape-mismatch remap):** with a wrong-shape mean, `match()` returns
  `(raw+1)/2` instead of the raw score — 0.6635 vs 0.3270 for the same vectors —
  and an impostor at raw cosine 0.66 crosses the threshold at 0.830.

Raw scores for the 5 cohort-less models cannot be measured here without their
checkpoints, but the code path is unconditional: `cohort_mean is None` for any
model without `assets/cohort/<slug>.npy` (session.py:90-93), and `match()` then
compares raw cosine to 0.82 (matcher lines 74-99). The compression itself is a
property of speaker-embedding spaces (shared "cosine cone" direction; measured
cohort-mean norm 0.666 in `2026-06-30` §b), so no model is exempt.

## 4. Design (final; a fixer needs no further decisions)

**Principle: a model without a calibrated cohort must not silently pretend to
do voiceprint identification. Disabled beats wrong.**

### 4.1 New module `src/diart_server/cohort.py`

Move `_COHORT_DIR`, `_cohort_slug`, `_load_cohort_mean` from
`session.py:62-101` into a new module `src/diart_server/cohort.py`,
**byte-identical logic and names** (the `Path(__file__).resolve().parents[2] /
"assets" / "cohort"` expression yields the same directory from
`src/diart_server/cohort.py`). Keep the module-level `logger`. Add one new
public helper:

```python
def has_cohort(model_name: str) -> bool:
    """True iff a cohort-mean file exists for this embedding model."""
    return (_COHORT_DIR / f"{_cohort_slug(model_name)}.npy").exists()
```

`session.py` replaces its local definitions with
`from diart_server.cohort import _load_cohort_mean` (behaviour identical; the
monkeypatch target for tests becomes `diart_server.session._load_cohort_mean`,
unchanged in spelling).

### 4.2 `/v1/info`: additive `voiceprint_ready` field (`src/diart_server/ws_routes.py:67-78`)

In `info()`, add one key to the returned dict — the existing `models` map shape
is NOT changed (tests pin it, `tests/diart_server/test_probes.py:94-108`):

```python
"voiceprint_ready": sorted(
    name
    for name, spec in registry.specs.items()
    if spec.kind == "embedding" and has_cohort(name)
),
```

with `from diart_server.cohort import has_cohort`. In the dev layout
(manifest `docker/models.json`, repo `assets/cohort/`) this is exactly
`["pyannote/embedding", "pyannote/wespeaker-voxceleb-resnet34-LM"]`.

### 4.3 `live.html`: dropdown lists only voiceprint-ready models (`src/diart_server/static/live.html:44-57`)

In `loadModels()`:

```js
const vpReady = Array.isArray(info.voiceprint_ready) ? new Set(info.voiceprint_ready) : null;
for(const [name,kind] of Object.entries(info.models)){
  if(name.includes('titanet')) continue; // hidden: needs nemo (belt-and-braces)
  if(kind==='embedding' && vpReady && !vpReady.has(name)) continue; // no cohort -> no voiceprint -> hide
  ...
}
```

Segmentation options unchanged. If `voiceprint_ready` is absent (older server),
`vpReady` is `null` and behaviour falls back to today's. The existing default
selection block (live.html:52-56) needs no change — both preferred defaults are
in the ready set.

### 4.4 `session.py`: cohort-less sessions disable per-turn identification, loudly

1. **Constructor** (after `self.cohort_mean = ...`, session.py:234-236): when
   `store is not None and self.cohort_mean is None`, emit ONE warning naming the
   model:

   ```python
   logger.warning(
       "session %s: no cohort mean for embedding model %s; per-turn "
       "voiceprint identification DISABLED (match_threshold %.2f is "
       "calibrated for the centered score only)",
       self.id, params.embedding_model, settings.match_threshold,
   )
   ```

2. **`_transcribe_and_emit`** (session.py:652): extend the identify gate to

   ```python
   if (
       self.store is not None
       and self.cohort_mean is not None
       and dur >= settings.min_identify_seconds
   ):
   ```

   so a cohort-less session **skips `_identify_slice` entirely**: `name=None`,
   `score=None`, no `_last_match` write, no embedder construction, no executor
   hop. This preserves the per-turn identity invariants
   (`2026-07-06-voiceprint-false-accept.md`): no fallback identity is ever
   stamped, and the transcript is never blocked.

3. **`ready_message()`** (session.py:284-292): add one additive key to `config`
   (`ReadyMessage.config` is `dict[str, Any]`, protocol.py:41 — no schema
   change):

   ```python
   "voiceprint": "enabled" if self.cohort_mean is not None else "disabled_no_cohort",
   ```

   Keyed on `cohort_mean` alone (not on `store`) so the field deterministically
   reports cohort availability for the session's embedding model.

### 4.5 `voiceprint_matcher.py`: validate the mean shape ONCE; mismatch = pure raw mode

In `match()` (voiceprint_matcher.py:74-77), validate up front, before any
centering:

```python
if mean_embedding is not None:
    mean = np.asarray(mean_embedding, dtype=np.float32).reshape(-1)
    q = np.asarray(vec, dtype=np.float32).reshape(-1)
    if mean.shape != q.shape:
        mean_embedding = None  # wrong cohort for this space -> raw mode, no remap
```

With `mean_embedding` cleared, the whole call — query, every reference, and the
threshold comparison — runs the exact historical raw path: **no centering
anywhere, no `(c+1)/2` remap** (fixes §2.2). `_centered_unit`'s own fallback
(lines 44-47) stays as written: after the up-front check it can only trigger for
an individual *reference* whose shape differs from the query, and that reference
is already skipped by the `r.shape != u.shape` guard (line 87) — behaviour for
mixed-shape refs is unchanged (skip, don't score). `mean_embedding=None` callers
keep exact historical behaviour (same floats). Update the module/function
docstrings to state the mismatch rule.

### 4.6 Out of scope

- Generating cohorts for the other 5 models (needs their checkpoints + the
  calibration pool; separate task — the demo's validated model remains
  `pyannote/wespeaker-voxceleb-resnet34-LM`).
- Per-model thresholds; `assets/cohort/*.npy` contents; enrollment routes
  (enrolling on a cohort-less model stays allowed — harmless, and the vectors
  become useful the moment a cohort ships).

### 4.7 New tests (deterministic; fakes only — never the flaky WS-path files)

1. **Matcher** (extend `tests/diart_server/test_voiceprint_matcher.py`):
   - fixed-seed rng vec + >= 2 refs; `match(vec, refs, thr, mean_embedding=None)`
     vs `mean_embedding=np.zeros(dim + 7)` (wrong shape): returned `(name,
     score)` **exactly equal** (`==` on the floats, no tolerance) for thresholds
     0.0 and 0.82;
   - correct-shape mean: every candidate score in `[0, 1]` and the returned
     score equals the manual `(cos(unit(unit(v)-m), unit(unit(r)-m)) + 1) / 2`
     computation to `abs diff <= 1e-6`;
   - regression pin from §3 fact 3: wrong-shape mean with an impostor ref at raw
     cosine 0.66 must return `name=None` at threshold 0.82.
2. **`/v1/info`** (extend `tests/diart_server/test_probes.py`): dev layout
   `body["voiceprint_ready"] == ["pyannote/embedding",
   "pyannote/wespeaker-voxceleb-resnet34-LM"]` (already sorted); existing
   10-model `models`-map assertions untouched and passing.
3. **Session** (new `tests/diart_server/test_session_cohort_disable.py`,
   patterned on `test_session_transcript.py`: dummy seg/emb models, FakeRegistry,
   fake store with `vectors()`, monkeypatched `AsrClient.transcribe`, driven via
   `_handle_turn` with a >= 2 s ring slice so `min_identify_seconds=2.0` is
   passed):
   - cohort-less model (`dummy/emb` has no cohort file): transcript emitted with
     `name is None and score is None`; a spy on
     `session._identify_slice` (and on
     `diart_server.session.extract_utterance_embedding`) is **never called**;
     `session._last_match == {}`;
     `ready_message()["config"]["voiceprint"] == "disabled_no_cohort"`;
     construction logs one WARNING containing the model name (caplog);
   - cohort present (monkeypatch `diart_server.session._load_cohort_mean` to
     return a unit vector before construction): `ready_message()["config"]
     ["voiceprint"] == "enabled"` and `_identify_slice` (stubbed to return
     `("alice", 0.9)`) IS called, transcript `name == "alice"`.
4. **live.html static assert** (extend `tests/diart_server/test_live_route.py`):
   `"voiceprint_ready" in r.text` for `/demo/live` (dropdown population
   references the new field).

## 5. Verification (Reviewer runs all; numeric criteria)

1. **Matcher scale laws:** the §4.7(1) tests pass — wrong-shape mean == raw-mode
   floats EXACTLY; correct-shape mean scores in `[0,1]` and equal to the manual
   centered+remapped computation within 1e-6; the 0.66-impostor pin returns
   `name=None`.
2. **`/v1/info`:**
   `python -m pytest tests/diart_server/test_probes.py -q -p no:cacheprovider`
   -> 0 failures, and the new assertion pins
   `voiceprint_ready == ["pyannote/embedding", "pyannote/wespeaker-voxceleb-resnet34-LM"]`.
3. **Session disable semantics:** §4.7(3) tests pass — cohort-less session:
   `name is None`, identify spy call count == 0, `_last_match == {}`, ready
   config `voiceprint == "disabled_no_cohort"`, exactly one construction
   WARNING; with-cohort session: `voiceprint == "enabled"`, identify called,
   `name == "alice"`.
4. **live.html:** §4.7(4) static assert passes; manual spot-check optional
   (dropdown must list exactly 2 embedding models against a dev-layout server).
5. **Full unit suite (the ONLY acceptance command):**

   ```bash
   python -m pytest tests/diart_server -q \
     --ignore=tests/diart_server/test_e2e_websocket.py \
     --ignore=tests/diart_server/test_end_control.py \
     --ignore=tests/diart_server/test_failure_modes.py \
     -p no:cacheprovider
   ```

   Baseline at a6128f1 (this machine, re-measured while writing this spec):
   **149 passed, 1 skipped in ~74 s** (the pre-workflow figure of 139 predates
   the tests added by today's earlier fixes in this loop). Post-fix:
   **0 failures**, all 149 still green, plus the new §4.7 tests collected and
   passing.
6. **Cross-matrix regression gate (matching math touched -> mandatory):**

   ```bash
   DIART_MODELS_DIR=./models DIART_DEVICE=cuda DIART_VOICEPRINTS_DIR=./voiceprints \
     python scripts/verify_speaker_cross_matrix.py
   ```

   PASS = `FALSE ACCEPTS = 0` and `MISSED GENUINE = 0` (exit 0; exit 3 SKIPPED
   counts as FAIL on this machine — models and the AISHELL cache are present).
   Expected unaffected: the gate runs `pyannote/wespeaker-voxceleb-resnet34-LM`,
   whose cohort file exists and matches its 256-d space, so §4.4's gate and
   §4.5's validation are both no-ops there. Pre-fix baseline recorded at
   a6128f1 (2026-08-05, this machine, cuda): `RESULT: PASS`,
   `FALSE_ACCEPTS=0` (of 15 impostor turns), `MISSED_GENUINE=0` (of 3 asserted
   genuine turns), `SESSION_ERRORS=0`.
7. **Invariant audit (per-turn identity, `2026-07-06-voiceprint-false-accept.md`):**
   confirm by inspection of the diff that identity still comes only from a
   turn's own re-embedded audio; the cohort-less path emits `name=None` (never a
   fallback); identification errors still fail closed without blocking the
   transcript.
