# FIX report: ring/diart clock desync (consumer-side ring feed)

- **Date:** 2026-08-05
- **Fixer:** fixer agent (Reporter -> Fixer -> Reviewer loop)
- **Base:** 856c9ad (spec) — **Fix commits:** cae79cd (fix), ed33a3f (gate-script
  unblock), e212fc7 (tests)
- **Spec:** docs/bugs/2026-08-05-ring-diart-clock-desync.md (design §4
  implemented as written)

## 1. Changes

### src/diart_server/meeting_audio_source.py

- New optional constructor parameter
  `on_frame: Optional[Callable[[np.ndarray], None]] = None`, stored as
  `self._on_frame`, documented alongside `on_overflow` (generic callback, so
  the class stays decoupled from the ring).
- `read()`: for every dequeued non-sentinel frame, `self._on_frame(pcm)` is
  invoked (when set) immediately BEFORE `self.stream.on_next(tensor)` —
  prediction-side code runs downstream of `on_next` on this same thread, so
  the ring already holds the frame when `_on_prediction` reads
  `ring.total_seconds` or slices the newest window. Deliberately NOT wrapped
  in try/except (spec §4.1): if `ring.append` ever raises, the existing
  `finally: stream.on_completed()` terminates the loop loudly instead of
  silently desyncing again.

### src/diart_server/session.py

- `self.ring = PcmRingBuffer(...)` construction moved ABOVE the
  `MeetingAudioSource` construction; the source is now built with
  `on_frame=self.ring.append` (append is lock-protected, safe from the
  inference thread; `slice`/`total_seconds` take the same lock). A frame
  enters the ring iff diart consumes it: `ring.total_seconds` is now the
  consumer clock, and frames dropped on overflow reach NEITHER diart NOR the
  ring — both timelines share the same splice by construction.
- `push_pcm`: `self.ring.append(pcm)` DELETED. `metrics.audio_bytes_received`
  and `_last_push_at` stay (they intentionally measure the producer side);
  docstring states why the ring is not fed here.
- Overflow visibility: the `on_overflow` lambda replaced by bound method
  `_on_queue_overflow`, which (a) still calls
  `metrics.queue_overflow_total.inc()`, (b) increments per-session
  `self._overflow_count`, and (c) emits
  `logger.warning("session %s: audio queue overflow, %d frames dropped so
  far (inference slower than real time)", ...)` rate-limited to once per
  `_OVERFLOW_LOG_INTERVAL` seconds via a `self._last_overflow_log` monotonic
  timestamp (initialized `None`, so the FIRST drop always logs).
- Consequence comments added at the two `ring.total_seconds`-as-"now" sites
  (`aggregator.feed` in `_on_prediction`; close-flush in `close()`): both now
  sit on the same consumed-samples timeline as the turn coordinates. Close
  stays complete: `close()` joins the inference thread first and `read()`
  drains every queued frame (each appended via `on_frame`) before the
  sentinel.
- No change to `PcmRingBuffer`, `turn_aggregator`, protocol, or frontend
  (spec §4.2). Identity logic untouched — only WHICH samples the ring holds
  changed, so the per-turn identity invariants of
  docs/bugs/2026-07-06-voiceprint-false-accept.md are preserved verbatim
  (identity from the turn's own re-embedded slice; no centroid, no
  previous-turn fallback; errors fail closed to `name=None`).

### Constants chosen

- `_OVERFLOW_LOG_INTERVAL = 10.0` (module-level, session.py), per spec §4.2:
  drops arrive per-frame (~6/s at 170 ms frames), so sustained RTF > 1 would
  otherwise log ~6 lines/s; once per 10 s bounds log volume at 0.1 lines/s
  while keeping every degraded session visible. The callback runs on the
  asyncio loop inside `push_pcm`, so it stays a counter bump + clock read +
  rare log line. No other constant, threshold, or config default changed.

### scripts/verify_speaker_cross_matrix.py (verification-infrastructure unblock)

- NOT part of the spec'd design; needed to RUN the mandated regression gate.
  `librosa.resample` now dies at import on this machine ("Numba needs NumPy
  2.3 or less. Got NumPy 2.5" — environment drift since the 2026-07-06 run).
  `load_wav` now falls back on ImportError to
  `soxr.resample(..., quality="HQ")`, which is what librosa 0.11's default
  `res_type="soxr_hq"` delegates to internally — numerically identical
  resampling, and the librosa path is preserved for healthy environments.

### Tests (e212fc7; 7 new, spec §4.3 demanded >= 6)

- `tests/diart_server/test_meeting_audio_source.py` (+3):
  `test_on_frame_called_once_per_consumed_frame_never_for_drops` (maxsize=2,
  5 pushed before the reader starts -> exactly frames 3,4 seen,
  `np.array_equal`, nothing on the producer side),
  `test_on_frame_invoked_before_stream_emission` (event interleaving pinned
  to frame-before-next per frame), `test_on_frame_default_none_keeps_behaviour`.
- `tests/diart_server/test_ring_clock_alignment.py` (new file, +4): drives a
  real `MeetingAudioSource(on_frame=ring.append)` + real `PcmRingBuffer`
  with the spec's repro geometry (FRAME=2730, maxsize=10, 40 pushed -> 30
  forced drops):
  - `test_ring_clock_equals_consumed_samples_exactly`:
    `ring.total_seconds == consumed_samples / 16000` with `==` (no
    tolerance).
  - `test_ring_slice_returns_the_consumed_frame_for_consumer_window`:
    ramp-valued frames; every consumer-timeline window `[k*F/SR,(k+1)*F/SR)`
    slices back `np.array_equal` to the k-th CONSUMED frame (first consumed
    sample value asserted `== drops * FRAME`, i.e. NOT the k-th pushed
    frame).
  - `test_session_push_pcm_no_longer_feeds_ring`: real `DiarizationSession`
    (fake-model registry, per test_session.py style); 55 pushes of 160
    samples with the reader NOT running -> `ring.total_seconds == 0.0` and
    `_overflow_count == 5`; after draining `source.read` on a thread ->
    exactly `50*160/16000 = 0.5` s (the 5 drops excluded).
  - `test_overflow_log_rate_limited_counter_not`: 7 calls to
    `_on_queue_overflow` within the interval -> exactly 1 caplog WARNING
    (carrying "1 frames dropped") while `queue_overflow_total` increments by
    7 and `_overflow_count == 7`.
- Pre-existing tests: none needed updating (`test_session_transcript.py`
  feeds `session.ring.append` directly, which remains valid;
  `test_session.py` end-to-end runs with the reader live, where the no-drop
  behaviour is unchanged).

## 2. Verification results (all run post-fix on this machine)

### 2.1 Repro flips (spec §5.1) — PASS

`scratchpad/repro_clock_desync_fixed.py` = the reporter's repro with its two
`session.push_pcm`-mirror lines replaced by the new wiring
(`MeetingAudioSource(on_frame=ring.append)`, no producer-side append), same
scenario (40 frames of 2730 samples into maxsize=10, stalled consumer, 30
drops):

```
frames pushed          : 40  (109200 samples, 6.8250 s)
frames dropped         : 30  (81900 samples, 5.1188 s)
frames consumed (diart): 10  (27300 samples)
ring.total_seconds     : 1.7063 s
diart clock            : 1.7063 s
clock desync           : 0.0000 s        (pre-fix: 5.1188 s)
diart-clock window [0.000, 0.171) s
  audio diart consumed there : samples 81900..84629
  audio ring.slice returns   : samples 81900..84629   (pre-fix: 0..2729)
all 10 consumer-timeline windows slice back sample-exact
```

Exit 0. Desync 5.1188 s -> 0.0000 s EXACT (`==`, both sides multiples of
1/16000); `ring.slice` now returns sample-for-sample the audio diart
consumed and labeled.

### 2.2 Unit suite (the ONLY acceptance test command) — PASS

`python -m pytest tests/diart_server -q --ignore=tests/diart_server/test_e2e_websocket.py --ignore=tests/diart_server/test_end_control.py --ignore=tests/diart_server/test_failure_modes.py -p no:cacheprovider`

```
146 passed, 1 skipped in 74.11s
(baseline immediately before this workflow: 139 passed, 1 skipped, ~75 s;
 +7 new tests, spec demanded >= 6; zero previously-passing tests changed)
```

### 2.3 Identity/matching regression gate — PASS

`DIART_MODELS_DIR=./models DIART_DEVICE=cuda DIART_VOICEPRINTS_DIR=./voiceprints python scripts/verify_speaker_cross_matrix.py`

```
FALSE_ACCEPTS=0 (of 15 impostor turns)
MISSED_GENUINE=0 (of 3 asserted genuine turns)
SESSION_ERRORS=0
RESULT: PASS       (exit 0)
```

Genuine turns: SSB0011 0.934, SSB0341 0.906, SSB0261 0.844, newsA 0.992 —
all accepted; every impostor turn `None` (max impostor score 0.731 on this
run); cross-channel genuine newsA tail 0.793 -> `None` (exempt by design).
On cuda with an empty queue nothing is dropped, so this pins "no behaviour
change in the aligned case". (15 impostor turns vs 14 in the 2026-07-06
report: diart segmentation nondeterminism produced one extra sub-threshold
turn; both PASS criteria are unconditional zeros either way.)

### 2.4 Invariant re-check (spec §5.4) — PASS (code review)

`git show cae79cd` touches only ring feeding, source plumbing, and overflow
logging. `_identify_slice`, `_transcribe_and_emit`'s identify gate
(`min_identify_seconds`, fail-closed except path), `_last_match` write
discipline, and `voiceprint_matcher` are byte-untouched — identity still
comes only from a turn's OWN re-embedded audio, now guaranteed to BE that
turn's audio even after queue overflow.

## 3. Residual risk (unchanged from spec §6)

- Sustained RTF > 1 still loses audio (bounded queue); the loss is now
  consistent across diart/ASR/identity and visible in logs (rate-limited
  warning + `queue_overflow_total`). Adaptive latency/step is a separate
  workstream.
- Spliced audio at a drop boundary can straddle a turn and degrade that one
  turn's ASR/identity; fail-closed identity bounds the damage.
- `prediction_latency` remains a producer-vs-prediction approximation by
  design.
- Environment: librosa (via numba) is import-broken under the machine's
  NumPy 2.5; the gate script self-heals via soxr, but other librosa users
  in scripts/ would hit the same wall until numba/numpy are reconciled.

## Review

pending
