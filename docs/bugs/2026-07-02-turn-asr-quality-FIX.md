# FIX report: Turn/ASR quality (latency, overlap contamination, fragmentation)

Spec: `docs/bugs/2026-07-02-turn-asr-quality.md`
Loop: Reporter -> Fixer -> Reviewer, iteration 1/3
HEAD at loop start: 058b47f

## Iteration 1 (Fixer)

### BUG1 — Long-monologue latency (partial ASR + lower cap)

Changes:
- `turn_aggregator.py`: `_Open` gains `turn_id` (uuid4 hex) and `last_partial_end`
  (init = start). `FinalizedTurn` gains `turn_id`. New `due_partials(interval)`
  returns `PartialTurn(speaker, turn_id, start, end=last_end)` snapshots for open
  turns with `>= interval` seconds of NEW audio, non-destructively advancing
  `last_partial_end` (no mutation of `_open` membership). Reads the aggregator's
  own open state; no wall clock.
- `session.py`: `_on_prediction` calls `due_partials(settings.partial_asr_interval)`
  after the feed loop (guarded by `> 0`) and dispatches each via
  `_handle_turn(pt, partial=True)`. `_handle_turn` gains `partial` param; partials
  and finals share one `turn_id`; finalized/flush turns emit `partial=False`.
- `protocol.py`: additive `turn_id: str = ""`, `partial: bool = False`.
- `config.py`: `partial_asr_interval` (default 5.0); `max_turn_seconds` 20.0 -> 12.0.
- `live.html`: `addLine` keys timeline `<div>`s by `turn_id` in a `Map`; a matching
  turn_id REPLACES the line in place (partial -> final), else prepends. Partials
  render dimmed/italic (`.line.partial`). Empty turn_id keeps legacy prepend.

Verification (numbers observed):
- `test_turn_aggregator_partials.py`: 2 passed. Single speaker fed 0.5s steps to
  15.0s, interval=5.0 -> `len(partials_seen) >= 2` (observed >=2, ends first
  reach >=5.0 then >=10.0), 1 unique non-empty turn_id, 0 finalizations before
  flush. `interval=0.0` disabled -> `[]`.
- `test_session_transcript.py::test_open_turn_emits_partial_then_final`: passed.
  >=1 message with `partial is True` + non-empty turn_id; final for same turn_id
  has `partial is False`.
- `test_config_new_settings.py`: passed. `partial_asr_interval == 5.0`,
  `max_turn_seconds == 12.0`.

### BUG2 — Overlap contamination (excise minority / flag majority)

Changes:
- `session.py`: `_overlap_intervals: list[tuple[float,float]]` + `_overlap_lock`
  (threading.Lock) in `__init__`. `_on_prediction` computes
  `annotation.get_overlap()` and records intervals via `_record_overlaps` (merge,
  prune to ring window, coalesce). Pure helper
  `_apply_overlap(pcm, ft) -> (pcm, overlapped, overlap_ratio)`:
  - `ratio <= overlap_max_ratio` (minority): zero the overlapped samples in the
    (already-copied) slice; `overlapped=False`.
  - `ratio > overlap_max_ratio` (majority): leave audio intact; `overlapped=True`.
  - `ratio == 0`: unchanged; `overlapped=False`.
  Excision mutates only the fresh `ring.slice()` copy, never `ring._buf`.
- `protocol.py`: additive `overlapped: bool = False`, `overlap_ratio: float = 0.0`.
- `config.py`: `overlap_max_ratio` (default 0.5).
- `live.html`: `overlapped` renders a `⑂重叠` badge.

Verification (numbers observed):
- `test_overlap_excision.py`: 3 passed. 4.0s slice (16000*4 @0.05):
  - overlap `[1.0,1.5]` (12.5%): `overlapped is False`, `overlap_ratio` == 0.125
    (|diff|<1e-6), `[1.0,1.5]` samples == 0.0, outside == 0.05.
  - overlap `[0.0,3.0]` (75%): `overlapped is True`, `overlap_ratio` == 0.75,
    pcm byte-identical to input.
  - no intervals: unchanged, ratio 0.0.
- `test_protocol_new.py::test_transcript_message_additive_fields_default`: passed.
  Minimal `TranscriptMessage` validates; overlapped False, overlap_ratio 0.0,
  turn_id "", partial False.

### BUG3 — Fragmentation + short-audio loss (silence-clock finalize + absorption)

Changes:
- `turn_aggregator.py`: removed the eager speaker-switch finalization loop.
  Tracks `_diart_now` (max end ever fed). Unified silence finalization across
  ALL open speakers. IMPORTANT correction vs. the spec's literal pseudocode: the
  silence reference is `ref = max(_diart_now, start)` (the incoming segment's
  START, the actual speech wavefront), computed BEFORE folding this segment's
  `end` into `_diart_now`. The spec's `diart_now = max(diart_now, end)` placed
  *before* the check inflates the apparent gap by the segment duration and would
  re-fragment the dominant speaker on any interjection (its own verification
  input then yields 3 turns, failing `<= 2`). Using the segment START matches the
  spec's stated INTENT ("a brief interjection by B NOT chop A's open turn") and
  its numeric criterion. `now` param retained but unused for gap logic.
- `session.py`: `_last_final: dict[speaker, FinalizedTurn]`. In `_handle_turn`,
  a finalized (non-partial) sub-0.25s slice is absorbed into the previous
  same-speaker `_last_final` when `ft.start - prev.end <= settings.absorb_gap`:
  the previous turn is re-sliced [prev.start, ft.end], re-ASR'd, and re-emitted
  with the SAME turn_id (partial=False) so the frontend replaces it. No adjacent
  neighbour -> legacy skip.
- `config.py`: `absorb_gap` (default 1.0).

Verification (numbers observed):
- `test_turn_aggregator_fragmentation.py`: 1 passed. spk0 0.0-6.0 with a single
  spk1 interjection 1.0-1.5 -> total finalized turns == 2 (`<= 2`), and a spk0
  turn spanning 0.0-6.0 (`end-start == 6.0 >= 5.0`). (Old eager-switch code
  produced fragmented spk0.)
- `test_session_transcript.py::test_short_fragment_absorbed_into_previous`:
  passed. spk0 0.0-1.0 (turn_id TID-X), then 0.15s fragment 1.0-1.15 -> a
  replacement transcript with turn_id TID-X and end == 1.15; no orphan TID-FRAG.
- `test_turn_aggregator.py`: updated `test_speaker_switch_finalizes_previous` ->
  `test_speaker_switch_does_not_finalize_previous` (switch alone finalizes
  nothing; a later silence gap does). Other three tests preserved and pass
  (compared by (speaker,start,end) since FinalizedTurn now carries a uuid
  turn_id).

### Do-not-break confirmations
- Protocol additive-only; all four new fields have defaults; minimal construction
  validates under `extra="forbid"`.
- Voiceprint mean-centering (058b47f) untouched.
- `test_session_transcript.py` original two tests preserved and pass (no overlap
  intervals -> ratio 0 -> no excision; 0.25s gate unchanged; `_handle_turn`
  default `partial=False`).
- `test_config_new_settings.py` `max_turn_seconds` updated 20.0 -> 12.0 (not
  deleted) + new keys asserted.

### Test output (item 2 command)
`test_turn_aggregator.py test_session_transcript.py test_session.py
test_protocol.py test_protocol_new.py test_voiceprint_matcher.py
test_live_route.py`: **38 passed**.

Full new+updated deterministic set (incl. partials/fragmentation/overlap/config):
**45 passed**. Broader diart_server run (excluding the known-flaky WS/perf/soak
tests): **130 passed, 1 deselected**.

### 2news.mp3 regression
`DIART_DEVICE=cuda DIART_MODELS_DIR=.../models python scripts/replay_2news.py`
-> exit 0, prints `REPLAY OK`. Baseline substrings pinned in the script from the
first clean run:
- `山东持续提升医疗废物处置能力`
- `生成二维码标签`
- `所有医疗废物均得到及时安全处置`
Partials observed as `[partial]` lines, replaced by finals with the same span;
no 0.5s fragmentation of the news monologue.

Note: the replay requires `DIART_MODELS_DIR` to point at the local `./models`
(the default `/app/models` is the container path); the ASR endpoint at
`settings.asr_base_url` must be reachable (it was: HTTP 200).
