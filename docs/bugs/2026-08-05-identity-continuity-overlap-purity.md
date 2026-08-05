# Overlapped audio pollutes voiceprint identity; short turns + a prediction-blind frontend leave most lines anonymous

- **Date:** 2026-08-05
- **Reporter:** bug-reporter agent (Reporter -> Fixer -> Reviewer loop), B45
- **HEAD at investigation:** e97677c
- **Status:** REPRODUCED offline (code path deterministically with fakes on CPU;
  score impact measured with the real wespeaker model + committed cohort on
  cuda). Repro scripts under the session scratchpad (§3); all numbers below
  were measured at this HEAD.

## 1. Symptom (user-visible, /demo/live)

Two related failures in the HTTPS :7443 live demo:

1. **Cross-talk corrupts identity.** When two people overlap, transcript
   lines lose or *swap* identity:
   - An enrolled speaker's own turn that partially overlaps another voice
     shows the gray `speakerN` label (their name disappears) even though the
     turn is long and mostly theirs.
   - Worse, a turn that is *mostly* overlapped can display an enrolled
     name that does **not** belong to the turn's speaker (the enrolled person
     merely talked over them), and that wrong identity then also appears in
     the prediction-stream hint for that diart label.
2. **No identity continuity.** In natural conversation most turns are short
   (backchannels, quick replies; `silence_gap=0.7` finalizes them fast).
   Turns under `min_identify_seconds=2.0` correctly emit `name=None`
   (sub-2s utterance embeddings are unreliable — calibrated in
   docs/bugs/2026-07-06-voiceprint-false-accept.md §3.3, do NOT lower it),
   so the timeline reads as anonymous "speaker0/speaker1" almost everywhere,
   *even seconds after the same diart label was confidently verified on a
   long turn*. The server already streams that last turn-verified identity
   on every prediction message — the demo page simply never reads it.

## 2. Root cause (current code, HEAD e97677c)

### 2.1 Identity embedding is fed the ASR-shaped audio, not pure-speech audio

`_transcribe_and_emit` (src/diart_server/session.py:610-678) slices the turn
(:631), calls `_apply_overlap` (:635), then feeds **that same array** to the
identity embedding (:643-646, `self._identify_slice, pcm`):

- **Minority overlap** (`ratio <= settings.overlap_max_ratio`, 0.5 —
  config.py:89-91): `_apply_overlap` **zeros** the overlapped samples in
  place (session.py:560-568). Correct for ASR (timestamps stay aligned), but
  the identity embedding then sees hard-silence gaps that dilute the
  utterance embedding and depress genuine scores (§3.2: 0.903 -> 0.768 with
  2s of 5s zeroed — below the 0.82 threshold, genuine identity lost).
- **Majority overlap** (`ratio > 0.5`): the audio is left **intact** and only
  flagged (session.py:557-559), so a MIXED two-speaker waveform is embedded
  and matched. A mixed embedding can clear the threshold against a speaker
  who was merely talking over the turn (§3.2: 0.700 clean reject -> 0.823
  false accept).
- The match result is then written into `self._last_match[ft.speaker]`
  (session.py:655-659) whenever identification ran — **including the
  majority-overlap mixed case** — and `_on_prediction` echoes `_last_match`
  onto every subsequent `PredictionMessage.name/score` for that label
  (session.py:458-468). The mixed-audio misidentification contaminates the
  display-hint channel.
- The identification gate uses the **total** turn duration
  (session.py:637-641: `dur = ft.end - ft.start`,
  `dur >= settings.min_identify_seconds`) — it is purity-blind: a 3s turn
  with only 1.5s of non-overlapped speech still runs identification on 50%
  silence (§3.1 case 3).

### 2.2 Frontend discards the only identity-continuity channel

`src/diart_server/static/live.html:127-135`: `ws.onmessage` handles exactly
three message types — `ready` (:129), `transcript` (:133), `error` (:134).
`grep -c prediction live.html` = **0**: the `PredictionMessage.name/score`
echo of the last turn-verified identity (written by `_transcribe_and_emit`,
per the 2026-07-06 design §4.1d) is never consumed. Combined with the
(correct) 2.0s gate (config.py:63-65) and `silence_gap=0.7` (config.py:66),
every sub-2s turn renders as the bare gray label forever
(live.html:89-94 `lineHTML`: `m.name` null -> `<span class="unk">` only).

## 3. Evidence (measured at HEAD e97677c)

### 3.1 Code path (deterministic, CPU, fakes — scratchpad `repro_identity_pollution.py`)

Drives the real `DiarizationSession._transcribe_and_emit` with dummy models,
a stub store/ASR, injected `_overlap_intervals`, a ramp signal in the ring,
and an `_identify_slice` spy capturing the exact pcm it receives:

```
CASE 1 minority: turn [0,5]s, overlap [2,3]s (ratio 0.20)
  _identify_slice CALLED with 80000 samples, 16000 hard-zero samples (20.0% silence)
  zero span: [2.00s, 3.00s)                       <- identity input diluted
CASE 2 majority: turn [0,5]s, overlap [0.5,4.5]s (ratio 0.80)
  _identify_slice CALLED, identity input == intact ring slice (mixed audio): True
  _last_match after turn: {'speaker0': ('wz', 0.9)}   <- mixed result cached
CASE 3 gate blindness: turn [0,3]s, overlap [0,1.5]s (ratio 0.50 -> minority)
  _identify_slice CALLED with 48000 samples, 24000 zeros (50.0% silence)
  (only 1.5s pure speech < min_identify_seconds=2.0, yet identification ran)
```

### 3.2 Score impact (real wespeaker + committed cohort, cuda, threshold 0.82 — scratchpad `measure_overlap_scores.py`)

Same-gender AISHELL-3 hard case (the failure mode per 2026-07-06 §2).
A = SSB0341 (enrolled on clips 1-2 via `embedding_extractor.extract`),
B = SSB0590. 5.0s genuine turn of A from held-out clips; scores are
`vp_match` with the committed cohort mean (the exact live scoring chain):

```
variant                                          vs A    named?  vs B   cos->clean
clean   (5.0s pure A)                            0.903   MATCH   0.662   1.000
zero20  (today: overlap [2,3)s zeroed)           0.882   MATCH   0.680   0.830
zero40  (today: overlap [1,3)s zeroed)           0.768   none    0.741   0.166
concat_pure (designed: [0,2)+[3,5)s, 4.0s)       0.909   MATCH   0.677   0.962
mixed80 (today: majority overlap left intact)    0.854   MATCH   0.775   0.634
```

- Zeroing 40% of a genuine turn drops it 0.903 -> **0.768 < 0.82**: genuine
  identity lost; the diluted embedding is nearly orthogonal to the clean one
  (cos 0.166) and *drifts toward the impostor* (0.662 -> 0.741).
- Sample-exact concatenation of the pure sub-slices (the designed identity
  input) scores **0.909** — identical-or-better vs clean. The fix costs
  nothing on genuine turns.
- **Wrong-name false accept** (reverse mix): B's own 5.0s turn with A NOT
  speaking scores 0.700 vs enrolled A — correct reject. The same turn with
  A mixed in over [0.5,4.5)s (majority overlap, left intact today) scores
  **0.823 >= 0.82 -> name=A**: the turn is stamped with the wrong person,
  and (`§3.1` case 2) that pair is written into `_last_match`, poisoning the
  prediction-stream hint for the label.

### 3.3 Continuity + regression baselines

- `grep -c prediction src/diart_server/static/live.html` -> 0 (§2.2).
- Unit suite at this HEAD (the ONLY acceptance command):
  `python -m pytest tests/diart_server -q --ignore=tests/diart_server/test_e2e_websocket.py --ignore=tests/diart_server/test_end_control.py --ignore=tests/diart_server/test_failure_modes.py -p no:cacheprovider`
  -> **156 passed, 1 skipped in 73.82s**.
- Cross-matrix gate at this HEAD (cuda):
  `DIART_MODELS_DIR=./models DIART_DEVICE=cuda DIART_VOICEPRINTS_DIR=./voiceprints python scripts/verify_speaker_cross_matrix.py`
  -> **FALSE_ACCEPTS=0 (of 15), MISSED_GENUINE=0 (of 3), SESSION_ERRORS=0,
  RESULT: PASS**. Its streams contain no overlapped speech, so it pins the
  non-overlap identity math this fix must not change.

## 4. Design (decisions are final; a fixer needs no further choices)

**Principle: the identity embedding sees only the turn's own PURE
(non-overlapped) speech, as a sample-exact concatenation — never zeros,
never a two-speaker mix. ASR keeps today's zeroing/flagging (its timestamps
must stay aligned). The per-turn identity invariants of 2026-07-06 are
untouched: identity only from the turn's own re-embedded audio, no
server-side fallback stamping, fail closed to name=None.**

### 4.1 `_apply_overlap` returns a separate identity input (session.py)

Change the signature to return a 4-tuple
`(asr_pcm, overlapped, overlap_ratio, identity_pcm)`:

- `ratio == 0`: `(pcm, False, 0.0, pcm)` — identity input IS the intact
  slice (same array; nothing downstream mutates it). Guarantees the
  non-overlap scoring chain is bit-identical to today (gate §5.4).
- `0 < ratio <= settings.overlap_max_ratio` (minority): **first** build
  `identity_pcm` from the still-unmutated `pcm`: the complement of the
  intersected overlap sub-intervals `subs` within `[ft.start, ft.end]`
  (kept spans `[ft.start, s0), [e0, s1), ..., [e_last, ft.end)`), each
  mapped with the SAME sample convention already used for zeroing
  (`i = int(round((t - ft.start) * sr))`, clamped to `[0, pcm.size]`,
  dropped if empty), then `np.concatenate` the kept slices (empty list ->
  `np.empty(0, dtype=np.float32)`). **Then** zero `subs` in `pcm` in place
  exactly as today and return `(pcm, False, ratio, identity_pcm)`.
  `subs` come from the one existing lock-guarded snapshot at the top of
  `_apply_overlap` — both outputs derive from the same intervals.
- `ratio > settings.overlap_max_ratio` (majority): audio left intact and
  flagged as today; `identity_pcm = None` -> identification is skipped
  entirely. Return `(pcm, True, ratio, None)`.

No new settings. Update the `_apply_overlap` docstring accordingly.
`tests/diart_server/test_overlap_excision.py` unpacks the old 3-tuple at
lines 64/82/96 — adapt those three unpacks mechanically (add the 4th value;
existing assertions unchanged) and extend with the §5.1a assertions there or
in a new file.

### 4.2 Gate identification on PURE duration (session.py `_transcribe_and_emit`)

Replace the current block (session.py:635-659) with:

```python
pcm, overlapped, overlap_ratio, identity_pcm = self._apply_overlap(pcm, ft)
name, score = None, None
pure_seconds = (
    identity_pcm.size / self.ring.sample_rate
    if identity_pcm is not None else 0.0
)
if (
    self.store is not None
    and self.cohort_mean is not None
    and identity_pcm is not None
    and pure_seconds >= settings.min_identify_seconds
):
    ...  # unchanged try/except: run_in_executor(EMBED_EXECUTOR,
         #   self._identify_slice, identity_pcm)
```

- Majority overlap (`identity_pcm is None`) -> identification never runs,
  `name=None`, **no `_last_match` write** (the write stays in the existing
  `else:` branch of the try, which is only reached when identification ran).
- `pure_seconds < settings.min_identify_seconds` (e.g. 1.5s pure in a 3s
  turn) -> skip, `name=None`, no `_last_match` write. `min_identify_seconds`
  stays 2.0 — it is calibrated (2026-07-06 §3.3); it now measures *pure
  speech*, which is the quantity it was calibrated on.
- **Preserved:** when identification RUNS and returns `name=None` (below
  threshold), `_last_match[ft.speaker]` IS still updated — 2026-07-06 §4.1d
  behaviour, unchanged code path.
- Exception path unchanged (fail closed to `(None, None)`, transcript still
  emitted). Update the `_transcribe_and_emit` docstring and the
  `TranscriptMessage` docstring note in protocol.py (gate is now on pure
  non-overlapped seconds).

### 4.3 Frontend continuity hint (live.html — display-only; server stamps nothing)

- CSS (style block, next to `.name`/`.unk` at live.html:12): add
  `.hint{color:#8b929c;font-style:italic;opacity:.85;font-weight:400}` —
  same muted gray as `.unk`/`.t` so it reads as metadata, italic + the
  `≈`/`?` markers signal uncertainty; visually distinct from the confident
  green bold `.name`.
- State: `const lastVerified = new Map(); // diart label -> {name, score}`.
- `ws.onmessage` (live.html:127-135): add, between the `transcript` and
  `error` branches:
  `else if(m.type==='prediction'){ if(m.name!=null) lastVerified.set(m.speaker,{name:m.name,score:m.score}); }`
- `lineHTML` (live.html:89-94): when `m.name` is null and
  `lastVerified.has(m.speaker)`, render
  `<span class="unk">${m.speaker}</span> <span class="hint">≈${lastVerified.get(m.speaker).name}?</span>`;
  lines with a real `m.name` are unchanged; nameless labels with no verified
  history are unchanged (bare gray label). The hint is computed at render
  time only (no retroactive rewriting of old lines).
- The server must NOT stamp names it did not verify — `TranscriptMessage.name`
  stays exactly as computed in §4.2; this is purely a frontend rendering of
  the already-streamed `PredictionMessage` echo (invariant preserved).

## 5. Verification (Reviewer runs all; numeric criteria)

1. **Identity-input tests** (drive `_apply_overlap`/`_transcribe_and_emit`
   directly with the fixtures of `tests/diart_server/test_overlap_excision.py`;
   suggested new file `tests/diart_server/test_identity_overlap_purity.py`):
   - a. Ramp pcm, turn `[0,5]`s, tracked overlap `[2,3]`s ->
     `identity_pcm.size == 64000` (4.0s at 16k) **and**
     `np.array_equal(identity_pcm, np.concatenate([ramp[:32000], ramp[48000:80000]]))`
     (sample-exact, no zeros inserted); the ASR pcm from the same call has
     zeros exactly in `[32000, 48000)` and equals the ramp elsewhere.
   - b. Majority overlap (turn `[0,5]`s, overlap `[0.5,4.5]`s, ratio 0.8):
     `_identify_slice` spy **NOT called**; pre-seeded
     `_last_match = {"speaker0": ("seed", 0.5)}` unchanged; transcript
     emitted with `name=None`, `overlapped=True`.
   - c. Pure duration 1.5s in a 3s turn (overlap `[0,1.5]`s, ratio 0.5 ->
     minority): spy **NOT called**; `_last_match` unchanged.
   - d. Pure duration >= 2.0 (turn `[0,5]`s, overlap `[2,3]`s -> pure 4.0s):
     spy called exactly once, and `np.array_equal` to the 64000-sample
     concatenation of (a).
   - e. Preserved behaviour: no overlap, dur >= 2.0, spy returns
     `(None, 0.5)` (below threshold) -> `_last_match["speaker0"] == (None, 0.5)`.
2. **Frontend static asserts** (extend `tests/diart_server/test_live_route.py`):
   `/demo/live` response contains `"type==='prediction'"`, `".hint{"`,
   `"≈"` (hint markup path), and `"lastVerified"`.
3. **Full unit suite:**
   `python -m pytest tests/diart_server -q --ignore=tests/diart_server/test_e2e_websocket.py --ignore=tests/diart_server/test_end_control.py --ignore=tests/diart_server/test_failure_modes.py -p no:cacheprovider`
   -> 0 failures; baseline at HEAD e97677c is **156 passed, 1 skipped
   (~74s)** — every previously-passing test stays green (the three
   `test_overlap_excision.py` unpacks may be mechanically adapted per §4.1;
   assertions must not weaken) plus the new tests of §5.1/§5.2 collected and
   passing. The ignored WS-path files are pre-existing flaky integration
   tests and must not be used for verification.
4. **Cross-matrix regression gate** (identity math for non-overlapped turns
   is unchanged — the `ratio == 0` path returns the intact slice as
   `identity_pcm`):
   `DIART_MODELS_DIR=./models DIART_DEVICE=cuda DIART_VOICEPRINTS_DIR=./voiceprints python scripts/verify_speaker_cross_matrix.py`
   -> `FALSE_ACCEPTS=0`, `MISSED_GENUINE=0`, `SESSION_ERRORS=0`,
   `RESULT: PASS` (measured PASS pre-fix at this HEAD, §3.3; exit 3 SKIPPED
   counts as FAIL — models and the AISHELL cache exist on this machine).

## 6. Out of scope / residual risk

- Overlap intervals come from diart's `annotation.get_overlap()`
  (session.py:486-493) at prediction granularity; missed/late overlap
  detection still pollutes identity. Not addressable at this layer.
- The `≈name?` hint is a heuristic display aid: if a diart cluster label
  drifts to a different physical speaker between verified turns, the hint
  can be momentarily wrong until the next >= 2.0s pure turn re-verifies. It
  is rendered as explicitly uncertain and never enters the transcript data.
- A minority-overlap identity input made of concatenated sub-slices contains
  splice discontinuities at sub-slice boundaries; measured impact is
  positive (0.909 vs 0.903 clean, §3.2), so no windowing/cross-fade is added.
