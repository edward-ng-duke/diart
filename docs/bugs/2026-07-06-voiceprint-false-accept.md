# Voiceprint false accept: every turn inherits the enrolled name at 0.8-0.9

- **Date:** 2026-07-06
- **Reporter:** bug-reporter agent (Reporter -> Fixer -> Reviewer loop)
- **HEAD at investigation:** c341aca
- **Status:** REPRODUCED offline (deterministic, real models, cuda). Repro
  gate committed as `scripts/verify_speaker_cross_matrix.py` (exit 1 at HEAD).

## 1. Symptom (live observation)

Cross-machine HTTPS demo (:7443), `embedding=pyannote/wespeaker-voxceleb-resnet34-LM`.
User enrolled "wz" (own voice, same mic/session), spoke: scores 0.80-0.91,
correct. A **completely different person** then spoke on the same mic: their
turns still showed `wz` at 0.80-0.91 (24.5-27.0s wz 0.89; 27.0-29.0s wz 0.91;
30.8-43.5s wz 0.82). Every line in the session was labeled wz; short segments
scored *higher* (0.89-0.91) than long ones (0.80-0.82).

## 2. Offline reproduction (the demanded cross-validation, 用不同的声音交叉检验)

Committed: **`scripts/verify_speaker_cross_matrix.py`**. Real voices only
(no pitch shifting): 5 AISHELL-3 speakers from the local HF cache
(`~/.cache/huggingface/hub/datasets--AISHELL--AISHELL-3/...` — studio
Mandarin, same-gender hard negatives, incl. the near-twin pair
SSB0261/SSB0590) plus the two `data/2news.mp3` broadcast speakers
(newscaster 0.6-6.6s & 19-24s, field reporter 7.8-19s). Each scenario
enrolls ONE speaker through the real enrollment path
(`embedding_extractor.extract`), then replays a composite multi-speaker
stream through a **real `DiarizationSession`** (real segmentation +
wespeaker models, real online clustering, cuda; ASR stubbed — identity does
not depend on ASR text; same drive pattern as `scripts/replay_2news.py`).

### Pre-fix run at HEAD c341aca (recorded 2026-07-06, RTX 4090, cuda)

```
RESULT: FAIL   (exit 1)
FALSE_ACCEPTS=13 (of 14 impostor turns)
MISSED_GENUINE=0 (of 3 asserted genuine turns)
```

Representative turns (`name/score` = what the pipeline emitted;
`truth` = the turn's own slice re-embedded with `embedding_extractor` +
`vp_match`, i.e. what per-turn identity would say):

```
S1 enroll SSB0011:
   0.61-  5.01 speaker0 gt=SSB0011  name=SSB0011 0.935 | truth 0.942            (genuine, correct)
   7.01- 12.01 speaker1 gt=SSB0341  name=SSB0011 0.784 | truth 0.796  FALSE-ACCEPT
  17.01- 20.01 speaker0 gt=SSB0590  name=SSB0011 0.877 | truth 0.757  FALSE-ACCEPT (merged cluster)
  21.79- 25.14 speaker0 gt=SSB0590  name=SSB0011 0.847 | truth 0.737  FALSE-ACCEPT (merged cluster)
S2 enroll SSB0341:
   7.59-  8.01 speaker0 gt=SSB0011  name=SSB0341 0.930 | truth 0.750  FALSE-ACCEPT (STALE _last_match)
   8.01- 12.01 speaker1 gt=SSB0011  name=SSB0341 0.786 | truth 0.771  FALSE-ACCEPT
S3 enroll SSB0261 (near-twin impostor SSB0590):
   8.66- 12.01 speaker1 gt=SSB0590  name=SSB0261 0.795 | truth 0.788  FALSE-ACCEPT
S4 enroll 2news newscaster (the live-shaped scenario, cross-gender impostor):
   7.74- 19.04 speaker1 gt=newsB    name=None    0.436 | truth 0.421            (correctly rejected)
```

This reproduces the live symptom exactly: impostor turns named as the
enrolled user at 0.78-0.93, sub-second impostor turns at *higher* scores than
long ones (S2: 0.42s turn at 0.930 — the score of the genuine speaker's own
previous turn). S4 also explains why the earlier 2news-based validation
(same 0.753 / diff 0.462, commit 058b47f) looked healthy: **cross-gender,
cross-channel impostors are easy; same-gender voices are the failure mode.**

## 3. Root cause (confirmed, with one correction and two additional contributors)

### 3.1 Primary: turn identity is the *cluster label's* identity, never the turn's audio — CONFIRMED

- `src/diart_server/session.py:404-414` (`_on_prediction`, inference thread):
  per prediction window, identity = `vp_match` of the diart **cluster
  centroid** (`_extract_embedding(speaker)` = L2-normed
  `pipeline.clustering.centers[idx]`, an accumulating sum of every embedding
  ever assigned to that cluster) against enrollment refs; cached in
  `self._last_match[diart_label]`.
- `src/diart_server/session.py:577` (`_transcribe_and_emit`, asyncio loop):
  the transcript's `name/score` is just `self._last_match.get(ft.speaker)`.
  **Nothing about the turn's own audio is consulted.**

Measured consequences (S1/S2 above):

1. **Cluster merging (under-segmentation)** — confirmed but *partial*: diart
   assigned SSB0590's audio to the enrolled speaker's cluster `speaker0`
   (raw-cosine distance between wespeaker utterance embeddings of different
   speakers is only 0.6-0.9, mostly < `delta_new=1.057`). The blended
   centroid still matched the enrollment at 0.847-0.877.
2. **Correction to the working theory: merging is NOT required.** SSB0341
   got its *own* cluster `speaker1`, and its pure centroid *still* matched
   the enrolled SSB0011 at 0.784 >= 0.6. See 3.3: the 0.6 threshold itself
   does not reject same-gender impostors.
3. **Stale `_last_match` reuse across speakers** — additional contributor,
   audited in the 964f74e/0373744 changes: `_last_match[label]` is only
   overwritten when `_extract_embedding` returns a vector; when a turn is
   emitted for a label whose centroid was updated later (or never), the turn
   inherits whatever identity that label had minutes ago. S2's 0.42s
   SSB0011-audio turn emitted `SSB0341 @ 0.930` — the *previous* speaker's
   genuine score, verbatim. This also explains "short segments scored
   higher": their score is a snapshot of the label's stale/blended history,
   not of their audio. The partial-ASR path (0373744) and the sub-0.25s
   absorption path (964f74e, `_handle_turn` -> merged re-emit) read
   `_last_match` the same way, so partials and absorbed fragments inherit
   identically. No *other* leak was found in those commits; they widened the
   blast radius of the pre-existing cluster-identity design rather than
   adding a new mechanism.

### 3.2 Adjacent latent bug found while reproducing (must be fixed by this change, the fix path reuses it)

`voiceprint_routes.py:79` builds `make_embedder(embedding_model, registry)`
with **no device**. `diart.blocks.embedding.SpeakerEmbedding.__init__`
(src/diart/blocks/embedding.py:15-18) then calls `self.model.to(cpu)` on the
**shared, registry-cached** `EmbeddingModel` (one instance per model name,
`models_registry.py`). Enrolling while a cuda session is live moves the live
pipeline's embedding weights to CPU; the next inference-thread forward raises
`RuntimeError: Input type (torch.FloatTensor) and weight type
(torch.cuda.FloatTensor)...` -> `INFERENCE_CRASH`. Reproduced offline
(embedder built pre-session worked, identical call after session
construction crashed with exactly that traceback). Any per-turn identity
implementation that calls `make_embedder` inherits this landmine.

### 3.3 Calibration: threshold 0.6 with the 2news-only cohort cannot reject same-gender impostors — NEW FINDING, measured

The truth-matrix column above shows per-turn re-embedding **alone does not
fix the bug**: impostor turns re-embedded score 0.725-0.796, all >= 0.6.
The cohort mean (`assets/cohort/*.npy`) is built from 21 clips of ONE
recording (`scripts/compute_cohort_mean.py`, 2news only), so it does not
capture the shared direction of same-gender/studio/Mandarin voices, and the
0.6 threshold was calibrated on a single cross-gender 2news pair.

Measured on the 5-speaker AISHELL pool + 2news pair (wespeaker,
utterance-level and fixed-length turn slices; "v1" = current 2news-only
cohort, "v2" = extended cohort: 14 x 4s 2news clips from 28-95s + all 20
clips of 5 *held-out* AISHELL speakers SSB0267/0407/0534/0609/0710):

```
slice   v1 same(min)  v1 diff(max)   v2 same(min)  v2 diff(max)
0.5s        0.719         0.828          0.662         0.797     (inseparable)
1.0s        0.801         0.836          0.773         0.800     (inseparable)
2.0s        0.863         0.823          0.837         0.789     (v2 separates)
4.0s        0.908         0.830          0.883         0.788     (v2 separates)
full        0.911         0.814          0.902         0.768
```

Conclusions that drive the design:
- **>= 2s of audio is required** for a reliable utterance identity (1s
  slices overlap between genuine and impostor even under v2).
- **v2 cohort + threshold ~0.82** separates the entire pool at turn level
  (impostor max 0.789/0.797-short vs asserted-genuine min 0.837), including
  the near-twin pair. v1 + 0.6 (status quo) has negative margin.
- Cross-channel genuine (2news newscaster re-appearing at 19-24s scores
  0.796 under v2) falls below 0.82 -> shown as *unknown*, never as a wrong
  name. That is the correct failure direction for a transcript UI.
- `pyannote/embedding` cannot separate same-gender 2s slices under any
  cohort tested (v2: same-min 0.738 < diff-max 0.783). The demo's wespeaker
  model is the validated one; pyannote/embedding sessions will simply show
  more "unknown" (fail-closed). Do not attempt to tune it in this fix.
- Rejected alternatives (measured): adaptive per-ref cohort-max thresholds
  (FA=1 on the near-twin pair + cohort-contamination footgun when the
  enrolled speaker appears in cohort audio) and symmetric S-norm over the
  cohort set (margin *negative*: near-twin z=3.25 vs cross-channel genuine
  z=1.79).

## 4. Design (decisions are final unless measurement during implementation contradicts them)

**Principle: a transcript turn's identity comes from that turn's own
(overlap-excised) audio, embedded with the same extractor used for
enrollment, matched with the same matcher — never from the cluster label.**

### 4.1 Per-turn identity in the session (`src/diart_server/session.py`)

- In `_transcribe_and_emit` (session.py:564-596), replace
  `name, score = self._last_match.get(ft.speaker, (None, None))` with:
  - `dur = ft.end - ft.start`
  - if `self.store is None` or `dur < settings.min_identify_seconds`:
    `name, score = None, None`
  - else: `name, score = await loop.run_in_executor(EMBED_EXECUTOR,
    self._identify_slice, pcm)` where `pcm` is the **same overlap-excised
    slice** already produced for ASR by `_apply_overlap` (session.py:576).
    Any exception inside -> `(None, None)` (fail closed, log warning).
  - After a successful identification, set
    `self._last_match[ft.speaker] = (name, score)` (both partial and final)
    so the prediction stream can surface it (4.4).
- New method `_identify_slice(self, pcm) -> tuple[str|None, float|None]`
  (runs on the embed executor thread): lazily create and cache
  `self._identity_embedder = make_embedder(self.params.embedding_model,
  self.registry, torch.device(settings.device))`, then
  `vec = embedding_extractor.extract(pcm, self._identity_embedder)` and
  `return vp_match(vec, self.store.vectors(self.params.embedding_model),
  settings.match_threshold, mean_embedding=self.cohort_mean)`.
- In `_on_prediction` (session.py:404-414): **delete** the centroid
  `vp_match` and the `_last_match` write. `PredictionMessage.name/score`
  become `self._last_match.get(speaker, (None, None))` (read-only; see 4.4).
  Keep `embedding=` (the centroid) on the wire for API compatibility.

Settled considerations:

- **(a) blocking/executor/device:** identity embedding is offloaded to a
  module-level `EMBED_EXECUTOR = ThreadPoolExecutor(max_workers=1,
  thread_name_prefix="vp-embed")` living in `embedding_extractor.py`.
  A single worker serializes every voiceprint forward pass process-wide
  (all sessions + enrollment), which (i) keeps the asyncio loop free,
  (ii) prevents concurrent `model.to()` device races on the shared registry
  model, (iii) bounds GPU contention (one wespeaker forward is ~10-50 ms on
  cuda — negligible next to ASR). The per-session `SpeakerEmbedding` wrapper
  is cached after first use; constructing it with
  `torch.device(settings.device)` makes its internal `.to()` a no-op against
  the pipeline's device. Concurrent *forwards* from the inference thread
  (diart's own embedding step) and the executor are safe: eval mode,
  `no_grad`, no mutable module state.
  **Also fix `voiceprint_routes.py:79`:** pass
  `torch.device(settings.device)` to `make_embedder` AND run enrollment on
  `EMBED_EXECUTOR` instead of the default executor (fixes 3.2). Change
  `make_embedder`'s `device` default from `None` to
  "resolve from `settings.device`" so no caller can re-trigger the CPU move.
- **(b) minimum duration:** new setting `min_identify_seconds: float = 2.0`
  (`DIART_MIN_IDENTIFY_SECONDS`, config.py). Turns/partials shorter than
  this get `name=None, score=None` — **no centroid fallback, no previous-turn
  fallback** (both are the bug). The frontend already renders nameless turns
  as the bare `speakerN` label; no live.html change is required.
- **(c) partials:** `_transcribe_and_emit(partial=True)` goes through the
  same code path, so a partial slice >= 2s gets its own identity that firms
  up as the turn grows; the final (same `turn_id`) replaces it in the UI.
- **(d) prediction stream:** centroid-based naming is removed entirely.
  `prediction.name/score` now echo the last *turn-verified* identity for
  that cluster label (`_last_match`, now written only by 4.1), i.e. a
  display hint that is always consistent with the transcript; `None` before
  the first identified turn. Document this in `protocol.py` docstrings.
- **(e) threshold + cohort (they change together, they are one calibration):**
  - `scripts/compute_cohort_mean.py`: cohort = 4s 2news segments at starts
    `range(28, 96, 5)` (drop 0-27s: those contain the two news speakers used
    as verification targets) **plus all clips of AISHELL-3 speakers
    SSB0267, SSB0407, SSB0534, SSB0609, SSB0710** from the local HF cache
    (glob `~/.cache/huggingface/hub/datasets--AISHELL--AISHELL-3/snapshots/
    */train/wav/<spk>/*.wav`; skip with a warning if absent). These five
    speakers are *held out* of the verification pool. Regenerate and commit
    both `assets/cohort/*.npy`.
  - `config.py`: `match_threshold` default `0.6 -> 0.82`. Selection rule if
    the regenerated cohort shifts scores: from the truth matrix printed by
    `verify_speaker_cross_matrix.py`, pick `max(impostor)+0.02 <= thr <=
    min(asserted genuine)-0.02`; measured expectation impostor max ~0.79,
    genuine min ~0.84.
  - `tests/diart_server/test_config_new_settings.py:10` pins
    `match_threshold == 0.6` — update that one assertion (config test, not a
    matcher test).
  - `scripts/verify_match_separation.py` asserts SAME(newsA cross-segment)
    `> settings.match_threshold`; under the new calibration that pair scores
    ~0.796 < 0.82 by design (cross-channel genuine -> unknown). Update its
    criterion to: `diff_name is None` AND `same_score >= diff_score + 0.10`
    AND `same_score >= 0.75`. `scripts/verify_voiceprint_separation.py`
    (gap > 0.25, remapped same > 0.40) passes unchanged under v2 (measured
    gap ~0.34).
  - `src/diart_server/voiceprint_matcher.py` and
    `tests/diart_server/test_voiceprint_matcher.py` are **untouched**.
- **(f) absorption / re-emit:** already funnels through
  `_transcribe_and_emit(merged, partial=False)` (session.py:551), which now
  re-embeds the merged slice; the >= 2s rule applies to the *merged*
  duration. No extra code, but add a unit test.

### 4.2 New tests (deterministic, dummy models OK — not the flaky WS path)

Extend `tests/diart_server/test_session_transcript.py` style with a fake
store + monkeypatched `_identify_slice`/embedder:
1. turn >= `min_identify_seconds` -> emitted `name/score` come from the
   turn's own slice (assert the identify call received the excised pcm), not
   from a pre-seeded `_last_match`;
2. turn < `min_identify_seconds` -> `name is None` even with a pre-seeded
   `_last_match`;
3. identify raising -> `(None, None)` and transcript still emitted;
4. absorbed sub-0.25s fragment -> identify called once with the merged span;
5. enrollment route/`make_embedder` default leaves the registry model's
   device unchanged (guards 3.2).

## 5. Verification (Reviewer runs all; numeric criteria)

Pre-fix numbers are recorded in §2 so the fix is provably causal
(`verify_speaker_cross_matrix.py` at HEAD c341aca: exit 1,
FALSE_ACCEPTS=13/14).

1. **Cross-validation gate (the bug):**
   `DIART_MODELS_DIR=./models DIART_DEVICE=cuda python scripts/verify_speaker_cross_matrix.py`
   -> exit 0, `FALSE_ACCEPTS=0` (of ~14 impostor turns), `MISSED_GENUINE=0`
   (of >= 3 asserted genuine turns), `RESULT: PASS`. (Exit 3 = SKIPPED is a
   FAIL for review purposes — models and the AISHELL cache exist on this
   machine.)
2. **2news end-to-end regression (real ASR):**
   `DIART_MODELS_DIR=./models DIART_DEVICE=cuda python scripts/replay_2news.py`
   -> exit 0, prints `REPLAY OK` (all 3 pinned baseline substrings present).
3. **Scoring-chain sanity (regenerated cohort):**
   `DIART_MODELS_DIR=./models DIART_DEVICE=cuda python scripts/verify_voiceprint_separation.py`
   -> `PASS` (centered gap > 0.25 AND remapped same > 0.40);
   `DIART_MODELS_DIR=./models DIART_DEVICE=cuda python scripts/verify_match_separation.py`
   -> `PASS` under the updated criterion of §4.1(e)
   (`diff_name is None`, `same_score - diff_score >= 0.10`, `same_score >= 0.75`).
4. **Test suite (non-flaky set; baseline at HEAD: 131 passed, 1 skipped):**
   `DIART_MODELS_DIR=./models python -m pytest tests/diart_server -q -p no:cacheprovider --timeout=180 --ignore=tests/diart_server/test_e2e_websocket.py --ignore=tests/diart_server/test_end_control.py --ignore=tests/diart_server/test_failure_modes.py --ignore=tests/diart_server/test_load_4_concurrent.py --deselect tests/diart_server/test_metrics.py::test_e2e_metrics_after_websocket_session`
   -> 0 failures, and >= 5 new tests from §4.2 collected and passing.
   `tests/diart_server/test_voiceprint_matcher.py` must be byte-identical to
   HEAD (`git diff c341aca -- tests/diart_server/test_voiceprint_matcher.py`
   is empty).
5. **Stability guard (identity must not stall or crash the pipeline):**
   `verify_speaker_cross_matrix.py` prints `SESSION_ERRORS=<n>` (counts
   `error`-type messages incl. `INFERENCE_CRASH`); it must be 0 (it is part
   of the PASS criterion), and total wall time must stay within 2x of the
   pre-fix run (~3 min incl. model load).

## 6. Out of scope / residual risk (documented, not fixed here)

- Near-twin voices closer than the measured pair (impostor ~0.79 vs
  threshold 0.82 leaves ~0.03 margin) can still false-accept; a larger,
  multi-condition cohort and/or score fusion of both embedding models would
  widen the margin. Track separately.
- `pyannote/embedding` same-gender discrimination is inherently weak (§3.3);
  sessions on that model trade false accepts for "unknown" labels.
- diart's own cluster labels (`speakerN`) still under-segment same-gender
  voices (`delta_new` default); after this fix that only affects the
  anonymous label, never the enrolled name.
