# FIX report: identity continuity + overlap purity (B45)

- **Date:** 2026-08-05
- **Fixer:** fixer agent (Reporter -> Fixer -> Reviewer loop)
- **Base:** c3737c6 (spec commit; code HEAD e97677c) — **Fix commits:**
  66fd212 (identity purity + pure-seconds gate), a3ce8a0 (live.html
  continuity hint), 7c76ffc (tests)
- **Spec:** docs/bugs/2026-08-05-identity-continuity-overlap-purity.md
  (design §4 implemented as written; no deviations)

## 1. Changes

### src/diart_server/session.py
- `_apply_overlap` signature changed to return a 4-tuple
  `(asr_pcm, overlapped, overlap_ratio, identity_pcm)` (spec §4.1):
  - `ratio == 0` (incl. `duration <= 0`): `(pcm, False, 0.0, pcm)` —
    `identity_pcm` IS the intact slice (same array), so the non-overlap
    scoring chain is bit-identical to pre-fix (cross-matrix gate §2.2).
  - Minority (`0 < ratio <= overlap_max_ratio` = 0.5): `identity_pcm` is
    built FIRST from the still-unmutated slice — the complement spans of the
    intersected overlap sub-intervals `subs` within `[ft.start, ft.end]`
    (cursor walk over `subs`, which `_record_overlaps` keeps sorted and
    coalesced), each span mapped with the SAME sample convention as the
    zeroing (`int(round((t - ft.start) * sr))`, clamped to `[0, pcm.size]`,
    dropped if empty), then `np.concatenate` (empty list ->
    `np.empty(0, dtype=np.float32)`; `np.concatenate` allocates, so the
    subsequent in-place zeroing cannot alias it). THEN the overlap samples
    are zeroed in `pcm` exactly as before for ASR. Returns
    `(pcm, False, ratio, identity_pcm)`.
  - Majority (`ratio > 0.5`): audio left intact and flagged as before;
    `identity_pcm = None`. Returns `(pcm, True, ratio, None)`.
  - No new settings; docstring rewritten accordingly.
- `_transcribe_and_emit` (spec §4.2): unpacks the 4-tuple;
  `pure_seconds = identity_pcm.size / self.ring.sample_rate` (0.0 when
  `None`); the identification gate is now
  `store is not None and cohort_mean is not None and identity_pcm is not
  None and pure_seconds >= settings.min_identify_seconds` (was: total
  `ft.end - ft.start`); `_identify_slice` receives `identity_pcm` instead of
  the ASR pcm. `min_identify_seconds` stays 2.0 — it now measures pure
  speech, the quantity it was calibrated on (2026-07-06 §3.3). Majority
  overlap and sub-2.0s-pure turns skip identification entirely: `name=None`,
  **no `_last_match` write** (the write remains only in the `else:` of the
  existing try, reached only when identification ran). Preserved unchanged:
  below-threshold results (`name=None`) still update `_last_match`
  (2026-07-06 §4.1d); exception path fails closed to `(None, None)` with the
  transcript still emitted. Docstring updated.

### src/diart_server/protocol.py
- `TranscriptMessage` docstring: name/score come from the turn's own PURE
  (non-overlapped) speech; gate is on pure non-overlapped seconds;
  majority-overlap turns are never identified. No wire-schema change.

### src/diart_server/static/live.html (spec §4.3, display-only)
- CSS: `.hint{color:#8b929c;font-style:italic;opacity:.85;font-weight:400}`
  next to `.name`/`.unk`.
- `const lastVerified = new Map()` (diart label -> `{name, score}`), next to
  `turnLines`.
- `ws.onmessage`: new branch between `transcript` and `error`:
  `else if(m.type==='prediction'){ if(m.name!=null) lastVerified.set(m.speaker,{name:m.name,score:m.score}); }`.
- `lineHTML`: nameless lines whose label has a verified history render
  `<span class="unk">speakerN</span> <span class="hint">≈name?</span>`;
  lines with a real `m.name` and nameless labels without history are
  byte-identical to before. Computed at render time only — old lines are
  never rewritten. `TranscriptMessage.name` is untouched: the server stamps
  nothing it did not verify (per-turn identity invariants preserved).

### Constants / calibration
- **No constants changed, none added.** `min_identify_seconds=2.0`,
  `match_threshold=0.82`, `overlap_max_ratio=0.5` all unchanged; the fix
  only changes WHAT is measured (pure seconds) and WHAT is embedded (pure
  concatenation), per the spec's measured evidence (§3.2: concat_pure scores
  0.909 vs 0.903 clean; no windowing/cross-fade needed).

### Tests
- `tests/diart_server/test_overlap_excision.py`: the three 3-tuple unpacks
  mechanically adapted to the 4-tuple (existing assertions unchanged), plus
  new-contract assertions: minority -> 3.5s pure identity with no zeros;
  majority -> `identity is None`; zero overlap -> `identity is out` (same
  array).
- `tests/diart_server/test_session_transcript.py`:
  `test_turn_identity_from_own_slice_not_cluster_cache` **updated — it
  conflicted with the spec'd new behaviour** (it asserted the identify input
  was the full 2.5s slice with the overlap zeroed in place; under §4.1 the
  identify input is the 2.0s pure concatenation with no zeros, which still
  passes the 2.0s gate). Its stale-`_last_match` / write-back / transcript
  assertions are unchanged. All other tests in the file untouched.
- New `tests/diart_server/test_identity_overlap_purity.py` (spec §5.1 a-e):
  - a: ramp fixture, turn [0,5]s, overlap [2,3]s -> `identity_pcm.size ==
    64000` and `np.array_equal` to `concatenate([ramp[:32000],
    ramp[48000:80000]])`; ASR pcm zeroed exactly in [32000,48000) and equal
    to the ramp elsewhere.
  - b: majority overlap (ratio 0.8) -> `_identify_slice` spy NOT called,
    pre-seeded `_last_match={"speaker0": ("seed", 0.5)}` unchanged,
    transcript `name=None`, `overlapped=True`.
  - c: 3s turn with 1.5s pure (ratio 0.5 -> minority) -> spy NOT called,
    `_last_match` unchanged, `name=None`.
  - d: 4.0s pure -> spy called exactly once with the sample-exact 64000
    concatenation of (a); transcript carries the spy's identity.
  - e: no overlap, spy returns `(None, 0.5)` ->
    `_last_match["speaker0"] == (None, 0.5)` (below-threshold write-back
    preserved).
- `tests/diart_server/test_live_route.py`: new
  `test_live_page_identity_continuity_hint_static_asserts` — `/demo/live`
  contains `type==='prediction'`, `.hint{`, `≈`, `lastVerified` (spec §5.2).

## 2. Verification results (all run post-fix on this machine)

### 2.1 Unit suite (the ONLY acceptance command) — PASS

`python -m pytest tests/diart_server -q --ignore=tests/diart_server/test_e2e_websocket.py --ignore=tests/diart_server/test_end_control.py --ignore=tests/diart_server/test_failure_modes.py -p no:cacheprovider`

```
162 passed, 1 skipped in 74.12s
(baseline at e97677c: 156 passed, 1 skipped; +6 new tests: 5 in
test_identity_overlap_purity.py + 1 in test_live_route.py; every
previously-passing test green — the two adapted files pass with assertions
strengthened, not weakened)
```

### 2.2 Cross-matrix regression gate (cuda, real models) — PASS

`DIART_MODELS_DIR=./models DIART_DEVICE=cuda DIART_VOICEPRINTS_DIR=./voiceprints python scripts/verify_speaker_cross_matrix.py`

```
FALSE_ACCEPTS=0 (of 15 impostor turns)
MISSED_GENUINE=0 (of 3 asserted genuine turns)
SESSION_ERRORS=0
RESULT: PASS        (exit 0)
```

Identical to the pre-fix baseline (spec §3.3) — expected by construction:
its streams contain no overlapped speech, and the `ratio == 0` path returns
the intact slice as `identity_pcm` (same array), so the non-overlap identity
math is bit-identical. Spot checks match the pre-fix run (e.g. newsA genuine
0.992 accepted, newsB impostor 0.662 rejected, cross-channel genuine 0.793
exempt-unknown).

## 3. Per-turn identity invariants (2026-07-06) — status

- Identity only from the turn's OWN re-embedded audio: **strengthened** —
  now additionally restricted to the turn's own PURE samples.
- Never the diart cluster centroid: untouched (`_on_prediction` still a
  read-only `_last_match` echo).
- No server-side previous-turn fallback stamped: untouched —
  `TranscriptMessage.name` is only ever the §4.2 result; the ≈name? hint is
  frontend-only, rendered as explicitly uncertain, never enters transcript
  data.
- Fail closed to `name=None` without blocking the transcript: untouched
  (same try/except path).

## 4. Residual risk (unchanged from spec §6)

- Overlap intervals come from diart's `get_overlap()` at prediction
  granularity; missed/late overlap detection still pollutes identity.
- The ≈name? hint can be momentarily wrong if a diart label drifts between
  physical speakers between verified turns; it re-verifies on the next
  >= 2.0s pure turn and never enters the transcript data.
- Splice discontinuities at concatenation boundaries measured harmless
  (0.909 vs 0.903 clean, spec §3.2); no windowing added.

## Review: PASS (iteration 1)

- **Reviewer:** independent reviewer agent (Reporter -> Fixer -> Reviewer loop)
- **Date:** 2026-08-05
- **Reviewed range:** c3737c6 (spec) .. 6230e46 (HEAD) — commits 66fd212,
  a3ce8a0, 7c76ffc, 6230e46; working tree clean; only
  session.py/protocol.py/live.html, 4 test files and this FIX report changed.

Diff re-derived against spec §4: `_apply_overlap` 4-tuple with the pure
concatenation built from the same lock-guarded interval snapshot BEFORE
in-place zeroing (same `round((t-ft.start)*sr)` convention, clamped, empties
dropped; `np.concatenate` allocates so zeroing cannot alias it); majority ->
`identity_pcm=None`, identification skipped, no `_last_match` write; gate on
pure seconds with `min_identify_seconds=2.0` unchanged
(`match_threshold=0.82`, `overlap_max_ratio=0.5` also unchanged);
below-threshold results still write `_last_match`; exception path fails
closed with the transcript emitted. `ratio==0` (and `duration<=0`) return
the intact ring-slice copy as `identity_pcm` — non-overlap scoring chain
bit-identical. live.html change is display-only (`lastVerified` Map fed by
the prediction echo, `≈name?` hint rendered only on nameless lines at render
time); `TranscriptMessage.name` untouched; turn_id partial->final replacement
and the late-partial clobber guard intact; no wire-schema change
(protocol.py docstring only). 2026-07-06 per-turn identity invariants
preserved (strengthened: pure samples only).

Commands run by the reviewer (all on this machine, post-fix HEAD 6230e46):

1. `python -m pytest tests/diart_server -q --ignore=tests/diart_server/test_e2e_websocket.py --ignore=tests/diart_server/test_end_control.py --ignore=tests/diart_server/test_failure_modes.py -p no:cacheprovider`
   -> **162 passed, 1 skipped in 73.84s** (baseline 156 passed, 1 skipped;
   +6 new tests, zero failures).
2. `python -m pytest tests/diart_server/test_identity_overlap_purity.py tests/diart_server/test_live_route.py tests/diart_server/test_overlap_excision.py tests/diart_server/test_session_transcript.py -v -p no:cacheprovider`
   -> **21 passed** — all five §5.1(a-e) tests and the §5.2 static-assert
   test collected and green; adapted files' assertions strengthened, not
   weakened (verified in the diff).
3. `DIART_MODELS_DIR=./models DIART_DEVICE=cuda DIART_VOICEPRINTS_DIR=./voiceprints python scripts/verify_speaker_cross_matrix.py`
   -> **FALSE_ACCEPTS=0 (of 15), MISSED_GENUINE=0 (of 3), SESSION_ERRORS=0,
   RESULT: PASS, exit 0**; spot values match the pre-fix baseline (newsA
   genuine 0.992 accepted, newsB impostor 0.662 rejected, cross-channel
   0.793 exempt-unknown).

Failure-class hunt (no findings): server state is per-session
(`_last_match`, `_overlap_intervals` pruned to the ring window); interval
boundary edges are safe (`duration<=0` -> intact-slice identity; empty kept
list -> `np.empty(0)` -> pure_seconds=0 -> gate skips; shared rounding
partitions kept/zeroed spans exactly; `ring.slice` returns a fresh copy so
the shared-array `ratio==0` path is mutation-safe). One noted residual,
within spec §6's accepted class: the frontend `lastVerified` Map persists
across stop/start cycles on the same page, so a hint from a previous run can
show on a fresh session's same-numbered label until re-verified — display-
only, explicitly uncertain, never enters transcript data (spec §4.3 mandates
exactly this implementation; flagged for a future spec if undesired).
