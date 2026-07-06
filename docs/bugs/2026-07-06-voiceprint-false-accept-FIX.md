# FIX report: voiceprint false accept (per-turn identity)

- **Date:** 2026-07-06
- **Fixer:** fixer agent (Reporter -> Fixer -> Reviewer loop, iteration 1)
- **Base:** c341aca — **Fix commits:** 877f25c (fix + calibration), dfc4857 (tests)
- **Spec:** docs/bugs/2026-07-06-voiceprint-false-accept.md (design §4 implemented as written)

## 1. Changes

### src/diart_server/session.py
- `_transcribe_and_emit` (both finals and partials, incl. BUG3 absorption
  re-emits): identity now computed from the turn's OWN overlap-excised pcm —
  `await loop.run_in_executor(EMBED_EXECUTOR, self._identify_slice, pcm)` —
  gated on `self.store is not None and (ft.end - ft.start) >=
  settings.min_identify_seconds`. Exceptions fail closed to `(None, None)`
  (warning logged, transcript still emitted). On success the result is written
  to `self._last_match[ft.speaker]` (both partial and final).
- New `_identify_slice(pcm)`: runs on the single-worker `EMBED_EXECUTOR`;
  lazily builds/caches `self._identity_embedder =
  make_embedder(params.embedding_model, registry, torch.device(settings.device))`,
  then `extract(pcm, embedder)` + `vp_match(vec, store.vectors(model),
  settings.match_threshold, mean_embedding=self.cohort_mean)` — the exact
  chain enrollment uses, so scoring is symmetric utterance-vs-utterance.
- `_on_prediction`: centroid `vp_match` + `_last_match` write DELETED.
  `PredictionMessage.name/score` are now a read-only echo of the last
  turn-verified identity (`None` before the first identified turn);
  `embedding` (centroid) kept on the wire for API compatibility.

### src/diart_server/embedding_extractor.py
- New module-level `EMBED_EXECUTOR = ThreadPoolExecutor(max_workers=1,
  thread_name_prefix="vp-embed")` — serializes every voiceprint forward
  process-wide (all sessions + enrollment).
- `make_embedder` device default `None` -> resolved from `settings.device`
  (fixes spec §3.2: `SpeakerEmbedding(model, None)` moved the SHARED registry
  model to CPU, crashing live cuda sessions with `INFERENCE_CRASH`).

### src/diart_server/voiceprint_routes.py
- Enrollment forward now runs on `EMBED_EXECUTOR` (was: default executor) and
  passes `torch.device(settings.device)` to `make_embedder` explicitly.

### src/diart_server/config.py
- New `min_identify_seconds: float = 2.0` (`DIART_MIN_IDENTIFY_SECONDS`).
  Turns/partials shorter than this emit `name=None` — no centroid fallback,
  no previous-turn fallback.
- `match_threshold` default `0.6 -> 0.82` (`DIART_MATCH_THRESHOLD`), one
  calibration with the cohort change below.

### scripts/compute_cohort_mean.py + assets/cohort/*.npy (regenerated, v2)
- Cohort = 14 x 4s 2news segments at starts 28..95s step 5 (0–27s dropped —
  contains the verification-target news speakers) + all 20 clips of five
  HELD-OUT AISHELL-3 speakers SSB0267/0407/0534/0609/0710 from the local HF
  cache (44.1 kHz clips resampled to 16 kHz; skipped with a warning if the
  cache is absent). Both `.npy` files regenerated and committed
  (`cohort clips: 14 x 2news + 20 x AISHELL`; pre-unit mean norms: wespeaker
  0.484, pyannote/embedding 0.454).

### scripts/verify_match_separation.py
- Criterion updated per spec §4.1(e): its SAME pair (A1 studio lead-in vs A2
  19–24s tail) is cross-channel, which under the new calibration deliberately
  lands below the accept threshold (unknown, never a wrong name). New PASS:
  `diff_name is None` AND `same_score >= diff_score + 0.10` AND
  `same_score >= 0.75`.

### src/diart_server/protocol.py
- `PredictionMessage` / `TranscriptMessage` docstrings document the new
  name/score semantics. No wire-schema change (protocol strictly additive:
  no fields added/removed).

### Tests
- `test_config_new_settings.py`: `match_threshold` pin 0.6 -> 0.82 + new
  `min_identify_seconds == 2.0` assert (config test; semantics legitimately
  changed by the calibration).
- `test_session_transcript.py`: +5 tests (own-slice identity w/ excised pcm
  + stale `_last_match` ignored/overwritten; short turn -> `None` w/o identify
  call; identify exception -> fail closed, transcript still emitted; absorbed
  fragment re-identifies the merged 2.6s span; `make_embedder` default device
  follows `settings.device`). `_make_session` gained an optional `store` arg.
- `test_voiceprint_routes.py`: +1 test (enroll route passes
  `torch.device(settings.device)` explicitly).
- `tests/diart_server/test_voiceprint_matcher.py`: byte-identical to c341aca
  (`git diff c341aca -- <file>` empty), matcher untouched.

## 2. Verification results (all run post-fix, RTX 4090, cuda, real models)

### 2.1 Cross-validation gate (the bug) — PASS

`DIART_MODELS_DIR=./models DIART_DEVICE=cuda python scripts/verify_speaker_cross_matrix.py`

```
FALSE_ACCEPTS=0 (of 14 impostor turns)
MISSED_GENUINE=0 (of 3 asserted genuine turns)
SESSION_ERRORS=0
RESULT: PASS        (exit 0; pre-fix at c341aca: FALSE_ACCEPTS=13/14, exit 1)
```

Wall time 10.5s (pre-fix run ~3 min budget; well within the 2x stability
guard). Full turn matrix (name/score = pipeline output, now identical to the
truth column by construction):

```
S1 enroll SSB0011:  genuine 0.61-5.01  SSB0011 0.934        (accepted)
                    impostors (SSB0341/SSB0590): None @ 0.681-0.753 (all rejected)
S2 enroll SSB0341:  genuine 0.79-6.01  SSB0341 0.906        (accepted)
                    the pre-fix 0.42s stale-match turn (was SSB0341@0.930):
                    now None (sub-2s -> no identify)
                    impostors (SSB0011/SSB0288): None @ 0.696-0.731
S3 enroll SSB0261:  genuine 0.73-3.25  SSB0261 0.844        (accepted)
                    near-twin SSB0590: None @ 0.703-0.727   (rejected)
S4 enroll newsA:    genuine 0.01-6.51  newsA   0.992        (accepted)
                    cross-gender newsB 11.3s: None @ 0.662  (rejected)
                    cross-channel genuine 20.14-25.09: None @ 0.793
                    (exempt by design: unknown, never a wrong name)
```

Threshold selection rule check: turn-level impostor max = 0.753, asserted
genuine min = 0.844 -> allowed band [0.773, 0.824]; default 0.82 is inside.
(S4's exempt cross-channel genuine at 0.793 also sits below 0.82 as designed.)

### 2.2 2news end-to-end regression (real ASR) — PASS

`DIART_MODELS_DIR=./models DIART_DEVICE=cuda python scripts/replay_2news.py`
-> exit 0, `REPLAY OK` (all 3 pinned baseline substrings present), 1m50s.

### 2.3 Scoring-chain sanity (regenerated v2 cohort) — PASS

`python scripts/verify_voiceprint_separation.py`:

```
RAW      : same=0.327 diff=0.146 gap=0.181
CENTERED : same=-0.019 diff=-0.642 gap=0.623   (> 0.25)
REMAPPED : same=0.491 diff=0.179 gap=0.312     (same > 0.40)
RESULT: PASS
```

`python scripts/verify_match_separation.py` (updated criterion):

```
SAME  (A2 vs A): name=None score=0.785
DIFF  (B  vs A): name=None score=0.681
diff_name None, same-diff = 0.104 >= 0.10, same 0.785 >= 0.75
RESULT: PASS
```

### 2.4 Test suite (non-flaky set) — PASS

`DIART_MODELS_DIR=./models python -m pytest tests/diart_server -q -p
no:cacheprovider --timeout=180 --ignore=...websocket/end_control/
failure_modes/load_4 --deselect test_metrics::test_e2e_metrics_after_websocket_session`

```
137 passed, 1 skipped, 1 deselected in 71.32s
(baseline at c341aca: 131 passed, 1 skipped; +6 new tests, >= 5 required)
```

Spec-mandated file list (test_session_transcript, test_session,
test_turn_aggregator{,_partials,_fragmentation}, test_overlap_excision,
test_voiceprint_matcher, test_protocol_new, test_live_route): 31 passed.

`git diff c341aca -- tests/diart_server/test_voiceprint_matcher.py` -> empty.

## 3. Residual risk (unchanged from spec §6)

- ~0.03 margin to the hardest measured near-twin impostor (0.79 vs 0.82);
  a larger multi-condition cohort / model fusion would widen it.
- `pyannote/embedding` sessions trade false accepts for more "unknown"
  (fail-closed); not tuned here by design.
- diart's `speakerN` labels still under-segment same-gender voices; after
  this fix that only affects the anonymous label, never the enrolled name.
