# Turn/ASR quality: latency, overlap contamination, fragmentation

Date: 2026-07-02
Author: Bug-Reporter agent (Reporter -> Fixer -> Reviewer loop)
HEAD at spec time: 058b47f
Scope: `src/diart_server/session.py`, `src/diart_server/turn_aggregator.py`,
`src/diart_server/protocol.py`, `src/diart_server/config.py`,
`src/diart_server/static/live.html`; tests under `tests/diart_server/`.

This spec is written to be implemented directly by the Fixer. Each bug section
gives (a) a code-grounded root cause, (b) the CHOSEN fix design (decisive; with
rejected alternatives), (c) deterministic verification with numeric criteria,
and (d) a do-not-break list.

## Shared facts (verified against the code)

- diart emits, every `step` (0.5s), only the newly-finalized ~0.5s region with
  per-speaker sub-tracks, NOT growing turns
  (`src/diart/blocks/aggregation.py:217`:
  `region = Segment(start, start + self.step)`).
- `session._on_prediction` (inference thread) iterates
  `annotation.itertracks(yield_label=True)` and feeds each track into
  `TurnAggregator.feed(speaker, turn.start, turn.start+turn.duration, now)` with
  `now = self.ring.total_seconds` (`session.py:392-425`). Finalized turns are
  dispatched via `asyncio.run_coroutine_threadsafe(self._handle_turn(ft), self.loop)`
  and tracked in `self._asr_futures` (`session.py:423-430`).
- `_handle_turn` slices the ring `[start,end]`, SKIPS if `pcm.size < 16000//4`
  (0.25s), runs ASR, emits `TranscriptMessage` (`session.py:432-454`).
- `TurnAggregator` (current, `turn_aggregator.py:32-64`): per-speaker open turns;
  triggers = (1) speaker-switch finalizes OTHER open speakers with
  `last_end <= start`; (2) same-speaker silence gap `now - last_end > 0.7`
  finalizes BEFORE extending; (3) `max_turn_seconds=20`; (4) `flush()` on close.
- `annotation` in `_on_prediction` is a `pyannote.core.Annotation`; it exposes
  `get_overlap() -> Timeline` giving exactly the multi-speaker intervals in the
  emitted region (`pyannote/core/annotation.py:621`). This is the data source
  for BUG2 — no manual pairwise interval math needed.
- Timeline skew (important, drives BUG3 design): `ring.total_seconds` is the
  INGEST clock; diart's emitted segment ends lag it by pipeline latency
  (~`latency`=0.5s). So `now(ingest) - last_end(diart)` INFLATES apparent
  silence by ~0.5s. A latency-free clock for silence decisions is
  `diart_now = max end ever emitted across all speakers` — compared within
  diart's own timeline.
- Protocol is pydantic v2 `extra="forbid"` — ALL changes must be additive
  (new OPTIONAL fields with defaults only) (`protocol.py:98`).

---

## BUG1 — Long-monologue latency (up to ~20s text lag)

### (a) Root cause
A `TranscriptMessage` is produced only inside `_handle_turn`, which only runs on
turn FINALIZATION (`session.py:423-430` dispatches on `aggregator.feed(...)`
output; `_handle_turn` at 432). For a continuous monologue no finalization
trigger fires until `max_turn_seconds` (`turn_aggregator.py:57-61`) caps the turn
at 20s. The screenshot's 31.0-51.5s turn is exactly this: the cap fired at 20.5s,
so the speaker saw NO text for that speech until t≈51.5 + ASR time (~1-3s). ASR
is strictly per-finalized-turn; there is no partial/streaming path.

### (b) CHOSEN fix design — incremental (partial) ASR for open turns + lower cap

Two coordinated changes:

**1. Partial ASR on open turns (primary).**
`TurnAggregator` gains a non-destructive "partials due" query. `session` runs
background ASR on the open-turn-so-far slice every
`DIART_PARTIAL_ASR_INTERVAL` seconds of accumulated open-turn audio and emits a
`TranscriptMessage` with additive fields `partial=true` + a stable `turn_id`.
On finalization it emits the final full-turn transcript with the SAME `turn_id`
and `partial=false`, so the frontend REPLACES the partial entry.

Concrete changes:

- `turn_aggregator.py`:
  - Add a stable id to each open turn. Give `_Open` a `turn_id: str`
    (`uuid4().hex`) and a `last_partial_end: float` (init = `start`).
  - `FinalizedTurn` gains `turn_id: str` (frozen dataclass; add field). Every
    construction site (`feed` finalize paths, `flush`) must pass the open turn's
    `turn_id`.
  - New method
    `due_partials(interval: float) -> list[PartialTurn]` where
    `PartialTurn(speaker, turn_id, start, end)` and `end = last_end`. A turn is
    "due" when `last_end - last_partial_end >= interval`. On yield, set
    `last_partial_end = last_end` (so partials fire every `interval` of NEW
    audio, not every step). Do NOT mutate `_open` membership. This method reads
    the aggregator's OWN open state; no wall-clock needed.
- `session.py`:
  - In `_on_prediction`, AFTER the `feed(...)` loop, call
    `self.aggregator.due_partials(settings.partial_asr_interval)` and for each
    dispatch `self._handle_turn(ft, partial=True)` the same way finalized turns
    are dispatched (track in `_asr_futures`). Guard behind
    `settings.partial_asr_interval > 0` (0 disables — feature flag).
  - `_handle_turn(self, ft, partial: bool = False)`: same slice+gate+ASR path;
    emit `TranscriptMessage(..., turn_id=ft.turn_id, partial=partial)`.
    Finalized turns pass `partial=False`. The 0.25s gate still applies (a partial
    with <0.25s audio simply produces nothing yet — fine).
  - `close()` flush turns carry their `turn_id` and emit `partial=False`.
- `protocol.py` — `TranscriptMessage` additive fields:
  `turn_id: str = ""` and `partial: bool = False`. (Empty default keeps every
  existing test/message valid.)
- `config.py`:
  `partial_asr_interval: float = Field(default=5.0, validation_alias="DIART_PARTIAL_ASR_INTERVAL")`.
- `live.html` `addLine`: key each timeline `<div>` by `m.turn_id` (store in a
  `Map`/`data-turn-id`). If a line with that `turn_id` exists, REPLACE its
  innerHTML; else prepend a new one. Render `m.partial===true` dimmed/italic
  (add a `partial` CSS class). Messages with empty/absent `turn_id` keep current
  prepend behaviour (back-compat).

**2. Lower `max_turn_seconds` default 20.0 -> 12.0 (defense-in-depth).**
Change `config.py` default and the corresponding assertion in
`test_config_new_settings.py`. Even if partials are disabled, worst-case final
latency drops from ~20s to ~12s. Keep the env override.

Rejected alternatives (one line each):
- Server-side true streaming ASR: Qwen3-ASR endpoint is request/response only; out of scope.
- Emit partials on a wall-clock timer thread: adds a thread + races the ring buffer clock; the ingest-driven `_on_prediction` cadence (every 0.5s) is already a natural tick.
- Only lower `max_turn_seconds` (no partials): still 12s lag; does not fix the felt latency.

### (c) Verification (deterministic, no live mic / no WS)
- Aggregator partial cadence — `tests/diart_server/test_turn_aggregator_partials.py`
  (NEW): feed a single speaker in 0.5s steps to 15.0s with `interval=5.0`; assert
  `due_partials` yields partials whose `end` first reaches `>= 5.0` no later than
  the step where `last_end==5.0`, then again near 10.0 and 15.0 -> assert
  `>= 2` partials before any finalization and all share ONE `turn_id`.
  Command:
  `python -m pytest tests/diart_server/test_turn_aggregator_partials.py -q -p no:cacheprovider --timeout=180`
  PASS: exit 0; test asserts `len(partials_seen) >= 2` and
  `len({p.turn_id for p in partials_seen}) == 1`.
- Session-level partial emission — extend
  `tests/diart_server/test_session_transcript.py` (NEW test
  `test_open_turn_emits_partial_then_final`): stub `AsrClient.transcribe` to an
  instant return; drive `_on_prediction`-style by feeding the aggregator +
  `due_partials` through the session with a ring populated to 15s of audio and
  `partial_asr_interval=5.0`; assert at least one emitted `transcript` message
  has `partial is True` with a non-empty `turn_id`, and the final message for the
  same `turn_id` has `partial is False`. PASS: both assertions true; exit 0.
- Config default — `test_config_new_settings.py`: assert
  `s.partial_asr_interval == 5.0` and `s.max_turn_seconds == 12.0`.

### (d) Do-not-break
- Additive-only protocol: `turn_id`/`partial` MUST have defaults.
- `test_config_new_settings.py` line 12 (`max_turn_seconds == 20.0`) legitimately
  CHANGES to 12.0 — UPDATE it (do not just delete).
- `test_session_transcript.py::test_handle_turn_emits_transcript` and
  `::test_handle_turn_skips_below_gate` PRESERVE: `_handle_turn` default
  `partial=False`, 0.25s gate unchanged, existing asserts must still hold.
- Voiceprint mean-centering (058b47f): untouched.

---

## BUG2 — Overlap contamination (mixed-speaker audio in both slices)

### (a) Root cause
ASR slices are pure time-cuts of MONO ring audio (`_handle_turn`:
`pcm = self.ring.slice(ft.start, ft.end)`, `session.py:437`). When two speakers
overlap in wall time, diart emits two tracks covering the same interval; each
becomes/extends a separate open turn, and each finalized turn's `[start,end]`
slice contains the SAME mixed audio for the overlap region -> both transcripts
polluted. There is zero overlap handling: `_on_prediction` never inspects
`annotation.get_overlap()`, and `_handle_turn` does no trimming/flagging.

### (b) CHOSEN fix design — track overlap intervals; excise if minority, flag if majority

**Data structure.** `session` maintains `self._overlap_intervals: list[tuple[float,float]]`
(diart-timeline seconds, merged, sorted). In `_on_prediction`, after building
messages, compute `annotation.get_overlap()` (a `Timeline`) and append each
segment `(seg.start, seg.end)` to `_overlap_intervals`; then PRUNE to the ring
window (drop intervals with `end < ring.total_seconds - settings.ring_buffer_seconds`)
and coalesce adjacent/overlapping intervals to keep the list small. Guard with
`self._overlap_lock` (threading.Lock) since `_on_prediction` runs on the
inference thread and `_handle_turn` reads on the asyncio loop.

**Decision rule at slice time (`_handle_turn`).** Compute
`ov = sum(overlap of each tracked interval with [ft.start, ft.end])` and
`ratio = ov / (ft.end - ft.start)`:
- `ratio <= DIART_OVERLAP_MAX_RATIO` (default 0.5) -> MINORITY: excise the
  overlapped samples from the slice before ASR (zero them out in-place on the
  copied slice — the ring's `slice()` already returns a `.copy()`/concatenation,
  so mutation is safe). Convert each overlap sub-interval to sample offsets
  relative to `ft.start` and set those samples to 0.0. Emit transcript with
  `overlap_ratio=ratio`, `overlapped=False`.
  Zeroing (not deletion) keeps timestamps aligned and is robust for ASR; deleting
  samples would desync `[start,end]` from text.
- `ratio > DIART_OVERLAP_MAX_RATIO` -> MAJORITY (mostly an interjection/crosstalk):
  leave audio INTACT (excising most of it would starve ASR), but mark
  `overlapped=True`, `overlap_ratio=ratio`. Frontend can flag/dim.
- `ratio == 0`: unchanged path, `overlapped=False, overlap_ratio=0.0`.

Concrete changes:
- `protocol.py` — `TranscriptMessage` additive fields:
  `overlapped: bool = False`, `overlap_ratio: float = 0.0`.
- `config.py`:
  `overlap_max_ratio: float = Field(default=0.5, validation_alias="DIART_OVERLAP_MAX_RATIO")`.
- `session.py`: add `_overlap_intervals`, `_overlap_lock`, init in `__init__`;
  populate/prune in `_on_prediction`; add a pure helper
  `_apply_overlap(pcm, ft) -> (pcm, overlapped, overlap_ratio)` used by
  `_handle_turn`. Keep it a small pure function so it can be unit-tested without
  a live pipeline.
- `live.html`: if `m.overlapped` render a small marker (e.g. a "⑂重叠" badge or
  dim); optional but include a minimal CSS hook.

Rejected alternatives (one line each):
- Source separation of the overlap region: heavy, no model wired; out of scope.
- Always drop overlapped turns: loses interjections entirely (BUG3-adjacent regression).
- Excise regardless of ratio: majority-overlap turns become empty -> ASR failure/garbage.

### (c) Verification (deterministic, pure function)
- Unit test `tests/diart_server/test_overlap_excision.py` (NEW), calling
  `DiarizationSession._apply_overlap` (or a module-level extracted helper) with a
  constructed 4.0s slice (16000*4 samples of constant 0.05) and:
  - overlap `[1.0,1.5]` (0.5s / 4.0s = 12.5% minority): assert returned
    `overlapped is False`, `abs(overlap_ratio-0.125) < 1e-6`, and samples in
    `[1.0,1.5]` are exactly 0.0 while samples outside are unchanged (0.05).
  - overlap covering `[0.0,3.0]` (75% majority): assert `overlapped is True`,
    `abs(overlap_ratio-0.75) < 1e-6`, and the pcm is byte-identical to input
    (no excision).
  Command:
  `python -m pytest tests/diart_server/test_overlap_excision.py -q -p no:cacheprovider --timeout=180`
  PASS: exit 0; both cases assert as above.
- Protocol additive test — `tests/diart_server/test_protocol_new.py` (extend or
  new): `TranscriptMessage(speaker="s0", start=0, end=1, text="x")` still
  validates AND defaults `overlapped is False`, `overlap_ratio == 0.0`.

### (d) Do-not-break
- Additive-only protocol (defaults required).
- `test_session_transcript.py` both tests: `_handle_turn` with NO overlap
  intervals must behave identically (ratio 0 -> no excision) — PRESERVE.
- The ring `slice()` return is already a fresh array; excision must NOT write
  into `self.ring._buf` (only into the returned copy).

---

## BUG3 — Fragmentation + short-audio loss (rapid A/B exchange)

### (a) Root cause
Two compounding faults:
1. Eager speaker-switch finalization: `feed` finalizes EVERY other open speaker
   whose `last_end <= start` on each incoming track (`turn_aggregator.py:36-39`).
   In a rapid A/B/A exchange, each 0.5s alternation immediately chops the other
   speaker's open turn, yielding 0.5s-granularity turns (screenshot:
   60.5-61.0 spk0 / 61.0-61.5 spk1 / 61.5-62.0 spk0).
2. Short-audio loss: `_handle_turn` silently drops turns whose slice is
   `< 16000//4` samples (`session.py:438`). The 0.5s fragments are near the gate;
   any that slice to <0.25s (edge/rounding, or partially outside the ring window)
   vanish entirely -> texts like "他还有。","然后，第。" or nothing at all.

### (b) CHOSEN fix design — silence-based finalization on a diart clock + fragment absorption

**1. Remove eager speaker-switch finalization; finalize ALL open speakers by
silence only, on a diart-timeline clock.**

- `TurnAggregator` tracks `self._diart_now: float` = max `end` ever seen across
  all `feed` calls (updated at top of `feed`). This is diart's own timeline
  (latency-free vs. the ingest `now`).
- DELETE trigger (1) (the speaker-switch loop, `turn_aggregator.py:36-39`).
- New silence finalization: at the top of `feed`, after updating `_diart_now`,
  finalize ANY open speaker (including others) whose
  `_diart_now - last_end > silence_gap`. This lets a brief interjection by B NOT
  chop A's open turn; A stays open and keeps extending. Turns MAY now overlap in
  time across speakers — that is correct and desired.
- Keep trigger (2) semantics but drive the same-speaker check off `_diart_now`
  instead of the passed `now` (the incoming ingest clock), so all silence
  decisions use one consistent latency-free clock. `feed`'s `now` parameter is
  retained for signature compatibility but is no longer used for gap logic
  (document this; callers unchanged).
- Keep trigger (3) `max_turn_seconds` and (4) `flush`.

New trigger rules (exact):
```
on feed(speaker, start, end, now):
  diart_now = max(diart_now, end)
  for s in open:                       # unified silence finalize (all speakers)
    if diart_now - open[s].last_end > silence_gap: finalize s
  cur = open.get(speaker)              # (already finalized above if it was silent)
  if cur is None: open[speaker] = Open(start, end)
  else: cur.last_end = max(cur.last_end, end)
  for s in open:                       # max-length
    if open[s].last_end - open[s].start > max_turn_seconds: finalize s
```
Defaults unchanged: `silence_gap=0.7`, `max_turn_seconds` -> 12.0 (BUG1).

**2. Absorb sub-0.25s finalized fragments instead of dropping them.**
`session` keeps `self._last_final: dict[speaker, FinalizedTurn]` = the last
transcript-emitting turn per speaker. In `_handle_turn`, when the slice is
`< 16000//4` (the current drop point, `session.py:438`): if the same speaker has
a `_last_final` whose `end` is within `settings.absorb_gap` (default 1.0s) of
`ft.start`, EXTEND that previous turn's end to `ft.end`, re-slice, re-ASR, and
re-emit with the SAME `turn_id` (partial=False) so the frontend replaces it
(reuses BUG1's turn_id/replace machinery). Otherwise (no adjacent turn) fall back
to the current behaviour (skip). This recovers the "然后，第。" fragments by
gluing them onto the neighbouring same-speaker utterance.

Concrete changes:
- `turn_aggregator.py`: as above (`_diart_now`, delete switch loop, unified
  silence finalize). Combined with BUG1's `turn_id` on `_Open`/`FinalizedTurn`.
- `config.py`: `absorb_gap: float = Field(default=1.0, validation_alias="DIART_ABSORB_GAP")`.
- `session.py`: `_last_final` map + absorption branch in `_handle_turn`.

Rejected alternatives (one line each):
- Post-hoc merge of adjacent same-speaker transcripts on the frontend: leaves back-end fragments/ASR-of-noise + can't re-ASR the merged audio.
- Raise the 0.25s gate: still drops speech, doesn't merge, worsens loss.
- Keep speaker-switch finalize but debounce it: half-measure; the diart-clock silence rule is simpler and removes the skew bug too.

### (c) Verification (deterministic, aggregator-driven)
- Rapid-alternation no-fragmentation — `tests/diart_server/test_turn_aggregator_fragmentation.py`
  (NEW). Reproduce the screenshot pattern: A speaks a long stretch while B
  interjects briefly. Feed (all in 0.5s steps, `now == end`):
  spk0 0.0-0.5, spk0 0.5-1.0, spk1 1.0-1.5 (interjection), spk0 1.5-2.0,
  spk0 2.0-2.5, ... spk0 up to 6.0. With the NEW rules, spk0's turn is NOT
  chopped by spk1's single 0.5s interjection (gap within `silence_gap`), so on
  flush we get exactly ONE spk0 turn (0.0-6.0) plus one spk1 turn (1.0-1.5).
  Assert: total finalized turns across feed+flush `<= 2` (vs. the old code's
  `>= 4`). Command:
  `python -m pytest tests/diart_server/test_turn_aggregator_fragmentation.py -q -p no:cacheprovider --timeout=180`
  PASS: exit 0; assert `len(all_turns) <= 2` and a spk0 turn with
  `end - start >= 5.0` exists.
- Fragment absorption — extend `test_session_transcript.py` (NEW
  `test_short_fragment_absorbed_into_previous`): stub instant ASR; emit a normal
  spk0 turn 0.0-1.0 (transcript T1 with turn_id X), then feed a 0.15s fragment
  spk0 1.0-1.15 within `absorb_gap`; assert the fragment does NOT vanish — a
  transcript is emitted with `turn_id == X` and `end == 1.15` (replacement),
  and NO orphan drop. PASS: exit 0.
- Existing aggregator semantics that CHANGE — `test_turn_aggregator.py`:
  `test_speaker_switch_finalizes_previous` (line 19-23) asserts the OLD eager
  behaviour and MUST be UPDATED: under the new rules a speaker switch alone does
  NOT finalize; finalization is silence-driven. Replace it with a test that
  feeds spk1 with a `silence_gap`+ worth of no spk0 activity and asserts spk0
  finalizes via silence (see do-not-break).
- End-to-end regression guard (real pipeline, real ASR, no WS):
  `scripts/replay_2news.py` (committed) replays `data/2news.mp3` through the real
  `SpeakerDiarization` pipeline + real `AsrClient`, collects emitted transcripts,
  and asserts the 3 known clean transcript lines still appear (substring match,
  order-independent). Run only when models + ASR endpoint reachable (skips
  cleanly otherwise). Command:
  `DIART_DEVICE=cuda python scripts/replay_2news.py`
  PASS: prints `REPLAY OK` and exits 0 when the 3 baseline lines are all present;
  the Fixer/Reviewer capture the baseline lines on first clean run and pin them
  in the script (see script header TODO).

### (d) Do-not-break
- `test_turn_aggregator.py::test_speaker_switch_finalizes_previous` semantics
  legitimately CHANGE — UPDATE (do not silently delete).
- `test_turn_aggregator.py::test_same_speaker_extends_without_finalizing`,
  `::test_silence_gap_finalizes`, `::test_max_turn_finalizes` must still PASS
  (single-speaker paths unchanged; silence gap still 0.7, max still fires).
  Note: `test_silence_gap_finalizes` currently passes `now=2.0` with the gap
  computed from `now`; under the new diart-clock rule the gap is computed from
  `_diart_now` (= max end = 2.5 after the second feed), and 2.5-1.0=1.5 > 0.7
  still finalizes — the assertion still holds, but the Fixer must confirm.
- `test_session_transcript.py` both existing tests: PRESERVE (no overlap, no
  absorption-eligible neighbour -> identical behaviour).
- Additive-only protocol; voiceprint mean-centering (058b47f) untouched.

---

## Global do-not-break checklist
- Protocol `TranscriptMessage`: only additive optional fields
  (`turn_id`, `partial`, `overlapped`, `overlap_ratio`), all with defaults.
- Voiceprint mean-centering fix (commit 058b47f) — do not regress.
- Never use the flaky WS/dummy-model tests for verification
  (`test_e2e_websocket.py`, `test_end_control.py`, `test_failure_modes.py`,
  `test_load_4_concurrent.py`, `test_metrics.py::test_e2e_metrics_after_websocket_session`).
- Tests whose semantics MUST be UPDATED: `test_config_new_settings.py`
  (`max_turn_seconds` 20->12; add new keys), `test_turn_aggregator.py::test_speaker_switch_finalizes_previous`.
- Tests that MUST be PRESERVED: the rest of `test_turn_aggregator.py`,
  both `test_session_transcript.py` tests, `test_protocol*.py` existing cases.
