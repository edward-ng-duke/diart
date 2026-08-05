# Ring/diart clock desync: queue-overflow drops make ASR and voiceprint identity run on the wrong audio

- **Date:** 2026-08-05
- **Reporter:** bug-reporter agent (Reporter -> Fixer -> Reviewer loop)
- **HEAD at investigation:** a6128f1
- **Status:** REPRODUCED offline (deterministic, no models, no GPU). Repro
  script: `scratchpad/repro_clock_desync.py` (numbers in §3; drives
  `MeetingAudioSource` + `PcmRingBuffer` exactly as `session.push_pcm` does).

## 1. Symptom (user-visible, /demo/live)

Whenever inference falls behind real time (`settings.device` defaults to
`cpu`, config.py:23; on cuda: load spikes, concurrent sessions, model-warmup
stalls), the audio queue overflows and drops frames. From that moment on,
**every transcript line in the live demo is produced from audio earlier than
the turn it is labeled with**, by the cumulative duration of all dropped
frames:

- ASR text belongs to an earlier utterance than the `[start-end]` shown on
  the line — text visibly "lags" the speaker turns, often attributing one
  speaker's words to another speaker's turn window.
- Per-turn voiceprint identification embeds that same wrong window, so
  `name/score` are computed from a different utterance than the one diart
  labeled — a genuine speaker's turn can be stamped with the previous
  speaker's name (or `None`), silently violating the per-turn identity
  invariant of docs/bugs/2026-07-06-voiceprint-false-accept.md ("identity
  comes from the turn's OWN audio") even though the session code does
  everything right — it is handed the wrong audio.
- The offset **accumulates monotonically for the rest of the session and
  never recovers** (each drop adds ~0.17 s; sustained RTF > 1 adds drift
  continuously). Only `queue_overflow_total` ticks; nothing is logged and
  nothing realigns the clocks.

## 2. Root cause (confirmed at HEAD a6128f1)

Two consumers of the pushed PCM keep **two different clocks**, and turn
coordinates are produced under one clock but resolved under the other.

1. `src/diart_server/session.py:301-302` (`push_pcm`): the same frame goes to
   BOTH
   - `self.source.push_pcm(pcm)` — `MeetingAudioSource`, a bounded queue
     (`maxsize=50` frames; live.html:62 sends 8192-sample @48 kHz frames
     ≈ 170.7 ms each, so the queue holds ≈ 8.5 s). On overflow the OLDEST
     frame is dropped (`src/diart_server/meeting_audio_source.py:70-91`) and
     only the `on_overflow` callback fires (wired at session.py:164-167 to
     `metrics.queue_overflow_total.inc()` — a counter, no log, no
     realignment).
   - `self.ring.append(pcm)` — `PcmRingBuffer` counts **every sample ever
     appended** (`src/diart_server/pcm_ring_buffer.py:19,42,52`
     `self._total += n`), dropped or not. `total_seconds`
     (pcm_ring_buffer.py:23-24) is therefore a **producer/wall-clock**.
2. diart's stream timeline is a **consumed-samples clock**:
   `src/diart/operators.py:44-90` (`rearrange_audio_stream`) advances
   `start_time += step` (operators.py:84) only when `step_samples` have been
   popped from frames actually emitted by `MeetingAudioSource.read()`
   (meeting_audio_source.py:93-109). A dropped frame never reaches `read()`,
   so it never advances diart's clock — every drop permanently compresses
   diart's clock relative to the ring's clock.
3. All turn coordinates are diart-clock: `session.py:445` feeds
   `turn.start/turn.start + turn.duration` (from the emitted annotation) into
   the aggregator, and the resulting `ft.start/ft.end` are used to slice the
   ring at `session.py:549` (`_handle_turn`) and `session.py:594`
   (`_transcribe_and_emit`). `PcmRingBuffer.slice`
   (pcm_ring_buffer.py:54-69) maps `t0/t1` to absolute sample indices of the
   *producer* clock. After `D` seconds of cumulative drops, the slice for a
   diart-labeled turn returns audio exactly `D` seconds EARLIER than what
   diart consumed and labeled.
4. The mixed-clock use of `ring.total_seconds` as "now" compounds it:
   `session.py:443` (`aggregator.feed(..., now)`) and `session.py:333`
   (close-flush) pass the producer clock into diart-clock turn bookkeeping,
   so silence-gap finalization (`silence_gap=0.7`, config.py:66) fires early
   by `D` seconds too; `session.py:477` prunes overlap intervals with the
   same skewed window.

Nothing in the codebase ever reconciles the two clocks.

## 3. Evidence (measured, deterministic, no models)

Repro script (run 2026-08-05 at HEAD a6128f1, python from the repo venv):
`/tmp/claude-1000/-home-edward-research-diart/6fa6a523-2deb-44f0-9b8b-14031b757d63/scratchpad/repro_clock_desync.py`.
It replays `session.push_pcm`'s exact two-consumer pattern
(session.py:301-302) with `maxsize=10` and a stalled consumer, using
ramp-valued samples (sample *i* has value *i*) so every sample is globally
identifiable, then drains the queue like the inference thread would:

```
frames pushed          : 40  (109200 samples, 6.8250 s)
frames dropped         : 30  (81900 samples, 5.1188 s)
frames consumed (diart): 10  (27300 samples)
ring.total_seconds     : 6.8250 s
diart clock            : 1.7063 s
clock desync           : 5.1188 s
desync == dropped_samples/SR (81900/16000 = 5.1188 s): EXACT

diart-clock window [0.000, 0.171) s
  audio diart consumed there : samples 81900..84629
  audio ring.slice returns   : samples 0..2729
  -> ASR/voiceprint would run on audio 5.1188 s EARLIER than the labeled turn
```

Cross-check of the "diart clock == consumed samples" claim with diart's own
operator: feeding the 10 consumed frames through
`rearrange_audio_stream(duration=1.0, step=0.5)` yields chunks
`[0.000, 1.000]` and `[0.500, 1.500]` — max timeline 1.500 s while the ring
clock reads 6.825 s. The desync equals `dropped_samples / 16000` to within
1e-9 s, i.e. **exactly** the dropped duration, and `ring.slice` returned
sample-for-sample the wrong frame (producer frame 0 instead of the consumed
frame 30).

Trigger plausibility in production: `queue_overflow_total` exists precisely
because this happens (session.py:163-167); `settings.device` defaults to
`cpu` (config.py:23) where wespeaker+segmentation inference is slower than
real time on this class of hardware, and the queue absorbs only ~8.5 s of
backlog before dropping.

## 4. Design (implement exactly this; constants justified inline)

**Principle: the ring must tick the SAME clock diart ticks — feed it from
the consumer side, so a frame enters the ring iff diart consumes it.**

### 4.1 `src/diart_server/meeting_audio_source.py`

- Add an optional constructor parameter
  `on_frame: Optional[Callable[[np.ndarray], None]] = None`, stored as
  `self._on_frame`, documented like `on_overflow` (callback kept generic so
  the class stays decoupled from the ring).
- In `read()` (meeting_audio_source.py:93-109), for every dequeued non-sentinel
  frame, invoke `self._on_frame(pcm)` (when set) **immediately before**
  `self.stream.on_next(tensor)`. Before, not after: `_on_prediction` runs
  downstream of `on_next` on this same thread, so the ring must already
  contain the frame when prediction-side code reads `ring.total_seconds` or
  slices the newest window. Do not wrap it in try/except: `ring.append` is
  pure numpy under a lock; if it ever raises, the existing
  `finally: stream.on_completed()` path terminates the loop loudly rather
  than silently desyncing again.

### 4.2 `src/diart_server/session.py`

- Move the `self.ring = PcmRingBuffer(...)` construction (currently
  session.py:215-217) ABOVE the `MeetingAudioSource` construction
  (session.py:164-167), and build the source with
  `on_frame=self.ring.append`. `PcmRingBuffer.append` is already
  lock-protected (pcm_ring_buffer.py:26-29), so appending from the inference
  thread is safe; `slice`/`total_seconds` take the same lock.
- In `push_pcm` (session.py:293-304): DELETE `self.ring.append(pcm)`
  (line 302). Keep `metrics.audio_bytes_received` and `_last_push_at` there —
  they intentionally measure the producer side.
- Overflow visibility: replace the `on_overflow` lambda (session.py:166) with
  a bound method `self._on_queue_overflow`, which (a) still calls
  `metrics.queue_overflow_total.inc()`, (b) increments a per-session
  `self._overflow_count`, and (c) emits a rate-limited
  `logger.warning("session %s: audio queue overflow, %d frames dropped so "
  "far (inference slower than real time)", self.id, self._overflow_count)`
  at most once per `_OVERFLOW_LOG_INTERVAL = 10.0` seconds (module-level
  constant; guard via `self._last_overflow_log` monotonic timestamp
  initialized to `None`). Justification for 10 s: drops arrive per-frame
  (~6/s at 170 ms frames), so sustained RTF > 1 would otherwise log ~6
  lines/s; once per 10 s bounds log volume at 0.1 lines/s while still making
  every degraded session visible. `push_pcm` runs on the asyncio loop, so
  the callback must stay this cheap (a counter, a clock read, rarely a log
  line).

Consequences that need no further code (state them in docstrings/comments):

- `ring.total_seconds` becomes the consumer clock, which is exactly the
  "now" that `aggregator.feed` (session.py:443) and the close-flush
  (session.py:333) want. Close remains correct: `close()` joins the
  inference thread first, and `read()` drains every queued frame before the
  sentinel, so the ring is complete when the flush runs.
- Frames dropped on overflow now reach NEITHER diart NOR the ring: both
  timelines share the same splice and stay aligned by construction. The
  audible discontinuity inside the (identically spliced) audio is inherent
  to dropping and out of scope.
- No change to `PcmRingBuffer`, `turn_aggregator`, protocol, or frontend.

### 4.3 New tests (deterministic, no models — extend the existing unit style)

`tests/diart_server/test_meeting_audio_source.py`:
1. `on_frame` is invoked once per CONSUMED frame, in consumption order, with
   the exact pushed arrays (`np.array_equal`), and never for dropped frames
   (small `maxsize`, push before starting the reader).
2. `on_frame=None` (default) keeps current behaviour (no raise).

`tests/diart_server/test_session.py` (or a new
`test_ring_clock_alignment.py` if the fake-session plumbing fits better),
driving a real `MeetingAudioSource(on_frame=ring.append)` + real
`PcmRingBuffer` with forced drops (`maxsize` small, reader started after
pushing) — the two acceptance assertions from §5:
3. `ring.total_seconds == consumed_samples / 16000` EXACTLY (`==`, no
   tolerance; both sides are exact multiples of 1/16000).
4. Ramp-valued frames (`np.arange` across frames): for a consumer-timeline
   window `[k*F/16000, (k+1)*F/16000)` (F = frame size), `ring.slice`
   returns values `np.array_equal` to the k-th CONSUMED frame (which, with
   drops forced, is NOT the k-th pushed frame).
5. Session-level wiring guard: constructing a `DiarizationSession` (existing
   fake-model fixtures) and calling `push_pcm` with the reader NOT running
   leaves `ring.total_seconds == 0.0`; after the source drains, it equals the
   consumed duration. This pins "push_pcm no longer feeds the ring".
6. Overflow log rate-limit: invoking the session's overflow callback N times
   within the interval emits exactly 1 warning (caplog) and increments the
   Prometheus counter N times.

## 5. Verification (Reviewer runs all; numeric criteria)

Pre-fix numbers are recorded in §3 so the fix is provably causal (desync
= 5.1188 s for 30 dropped frames of 2730 samples; `ring.slice` returns
sample-exact WRONG audio offset by that amount).

1. **Repro flips:** re-run
   `python /tmp/claude-1000/-home-edward-research-diart/6fa6a523-2deb-44f0-9b8b-14031b757d63/scratchpad/repro_clock_desync.py`
   with the script's two `session.push_pcm`-mirror lines replaced by the new
   wiring (`MeetingAudioSource(on_frame=ring.append)`, no producer-side
   append) — or equivalently run the new unit tests of §4.3, which encode
   the same assertions: with >= 30 forced drops,
   `ring.total_seconds == consumed_samples/16000` EXACTLY (was: exceeded it
   by dropped_samples/16000 = 5.1188 s), and the ramp-slice check passes
   `np.array_equal` (was: wrong by 81900 samples).
2. **Unit suite green (the ONLY acceptance test command; baseline measured
   on this machine immediately before this workflow: 139 passed, 1
   skipped, ~75 s):**
   `python -m pytest tests/diart_server -q --ignore=tests/diart_server/test_e2e_websocket.py --ignore=tests/diart_server/test_end_control.py --ignore=tests/diart_server/test_failure_modes.py -p no:cacheprovider`
   -> 0 failures, all previously-passing tests still green, and the >= 6 new
   tests of §4.3 collected and passing.
3. **Identity/matching regression gate (this change alters which audio the
   voiceprint path embeds):**
   `DIART_MODELS_DIR=./models DIART_DEVICE=cuda DIART_VOICEPRINTS_DIR=./voiceprints python scripts/verify_speaker_cross_matrix.py`
   -> `FALSE ACCEPTS = 0` and `MISSED GENUINE = 0` (PASS). On cuda with an
   empty queue nothing is dropped, so this guards "no behaviour change in
   the aligned case".
4. **Invariant re-check (docs/bugs/2026-07-06-voiceprint-false-accept.md):**
   identity still comes only from a turn's own re-embedded audio; no
   centroid or previous-turn fallback introduced; identification errors
   still fail closed to `name=None`. (Code review item — the fix touches
   only which samples the ring holds, not the identity logic.)

## 6. Out of scope / residual risk

- Sustained RTF > 1 still loses audio (drops are inherent to a bounded
  queue); after this fix the loss is at least *consistent* across diart,
  ASR, and identity, and is now visible in logs. Adaptive latency/step or
  server-side resampling to cheaper models is a separate workstream.
- The spliced audio at a drop boundary may straddle a turn and degrade that
  one turn's ASR/identity quality; fail-closed identity
  (`min_identify_seconds`, threshold 0.82) already bounds the damage.
- `prediction_latency` (session.py:412-415) still compares a producer-side
  timestamp with prediction-side now; it is an observability approximation
  by design (session.py:197-203) and is not made worse by this change.
