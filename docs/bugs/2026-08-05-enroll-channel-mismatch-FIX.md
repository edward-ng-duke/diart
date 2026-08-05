# FIX report: enroll-channel mismatch (demo enrollment vs live identification)

- **Date:** 2026-08-05
- **Fixer:** fixer agent (Reporter -> Fixer -> Reviewer loop)
- **Base:** aaaa171 — **Fix commits:** 0ff3d2e (live.html channel unification),
  e9463b7 (route-level trim), cdd55df (tests)
- **Spec:** docs/bugs/2026-08-05-enroll-channel-mismatch.md (design §4
  implemented as written; no design decisions were left to the fixer)

## 1. Changes

### src/diart_server/static/live.html (0ff3d2e)

- **One capture channel for everything.** New
  `const MIC = {channelCount:1, echoCancellation:false, noiseSuppression:false, autoGainControl:false};`
  and `openMic()` — the single `getUserMedia({audio:MIC})` call site in the
  file — wiring `MediaStreamSource -> ScriptProcessor(8192,1,1)` on the
  current `audioCtx` and delivering every frame to one router `onMicFrame`.
  Browser DSP (AGC/NS/AEC, previously defaulting to true on both paths) is
  now off on both paths.
- **MediaRecorder path deleted.** The
  `MediaRecorder`/`decodeAudioData`/`OfflineAudioContext` enrollment block is
  gone; the string `MediaRecorder` no longer appears in the file. Enrollment
  buffers raw Float32 frames for `const ENROLL_MS = 6000;` (visible per-second
  countdown `录音中… 剩 Ns`), concatenates them, and uploads
  `encodeWav(f32, audioCtx.sampleRate)` — 16-bit WAV at the NATIVE rate, so
  the server's torchaudio `_resample_if_needed` performs the 16 kHz
  conversion, the same resampler family every live identification frame goes
  through. FormData fields unchanged.
- **Pause-during-enroll guard.**
  `function onMicFrame(f){ if(enrolling){ enrollBuf.push(f); return; } downsampleSend(f); }`
  plus a defensive `if(enrolling) return;` first line in `downsampleSend`:
  enrollment speech never reaches the live WebSocket / diarization. When a
  live session is running, enrollment reuses the live capture (no second
  stream); otherwise it opens its own (`ownCapture`) and tears it down after.
- **WebSocket before mic.** The start handler creates `audioCtx` first (its
  `sampleRate` feeds the `sample_rate` query param), opens the WebSocket
  immediately, and only calls `openMic()` inside the `'ready'` message
  handler — the mic starts once the server is accepting audio, so the old
  first-words drop (`ws.readyState!==1` discard) cannot happen. `onerror` /
  `onclose` before ready tear down and re-enable controls.
- **Selects locked** (`#emb`/`#seg` disabled) while a session runs OR an
  enrollment records; stop/enroll-completion re-enable them only when neither
  is active. Enroll response now surfaces multi-clip feedback:
  `已注册 <name>（第<n_clips>条）— 再录1-2条更稳`; errors show
  `注册失败 <status>：<detail>`. Button label updated to 录 6 秒并注册.

### src/diart_server/voiceprint_routes.py (e9463b7)

- New module constants (values from spec §3.4): `TRIM_FRAME = 320` (20 ms @
  16 kHz), `TRIM_REL_RMS = 0.05`, `TRIM_ABS_FLOOR = 1e-3`,
  `TRIM_PAD_FRAMES = 2` (40 ms), `MIN_SPEECH_SECONDS = 1.5`.
- New `_trim_silence(pcm, sr=16000) -> np.ndarray | None`: non-overlapping
  20 ms frames, float64 per-frame RMS; `peak < TRIM_ABS_FLOOR` -> None
  (all-silence); keeps `[first_voiced - 2 frames, last_voiced + 1 + 2 frames]`
  clamped to the clip; result shorter than 1.5 s of speech -> None.
- `enroll()`: after `_resample_if_needed` and the existing `< 0.5 s` check
  (both unchanged), `_trim_silence(pcm)`; `None` -> HTTP 400 with detail
  `说话时间不足：修剪静音后有效语音不足 1.5 秒，请靠近麦克风重新录制`; otherwise
  the **trimmed** pcm is what `extract` embeds. Everything else in the route
  (EMBED_EXECUTOR offload, explicit `torch.device(settings.device)`) is
  untouched.
- **Not modified:** `embedding_extractor.py`, `voiceprint_matcher.py`,
  `config.py` — `git diff aaaa171 --` on those three files is empty, so the
  per-turn identity chain and `scripts/verify_speaker_cross_matrix.py`
  (which enrolls via `extract` directly) see byte-identical code.

### Tests (cdd55df)

- `tests/diart_server/test_voiceprint_routes.py`: `_wav_bytes` now generates
  a 440 Hz tone (`0.3*sin`, default 2.0 s @ 16 k) via a new `_pcm_to_wav`
  helper — zeros would be rejected as all-silence by the new trim; the three
  pre-existing tests pass **unchanged** (this helper change is the only
  pre-existing-test edit in the whole fix, and it is the one the spec §4.3
  prescribes). New tests:
  - `test_enroll_trims_leading_trailing_silence`: 1 s sil + 2 s tone + 1 s
    sil @16 k -> 200, and the patched `extract` receives
    `1.9*16000 <= size <= 2.2*16000` samples (actual: 33280 = 2.08 s).
  - `test_enroll_all_silence_rejected_400`: 4 s all-zero WAV -> 400,
    `说话时间不足` in detail, `extract` never called.
- `tests/diart_server/test_live_route.py`: new
  `test_live_page_enrollment_channel_static_asserts` — `MediaRecorder`
  absent; `getUserMedia(` count == 1 and `getUserMedia({audio:MIC})`
  present; `echoCancellation:false` / `noiseSuppression:false` /
  `autoGainControl:false` present; `const ENROLL_MS = 6000;` present;
  `if(enrolling) return;` present; `n_clips` present.

## 2. Calibration / constants chosen

None invented by the fixer: `match_threshold` stays 0.82 (config.py
untouched — spec §3.3: the threshold is right, the channel was wrong), and
all five trim constants are the spec §3.4 values, re-validated on this
machine post-implementation: 1 s sil + 2 s 440 Hz tone + 1 s sil trims to
**2.08 s** (spec's own measured value, inside [1.9, 2.2]); 4 s all-zero ->
rejected; 1 s-speech clip -> rejected (< 1.5 s minimum).

## 3. Verification results (all run post-fix on this machine)

### 3.1 Unit suite (the acceptance command) — PASS

```
python -m pytest tests/diart_server -q --ignore=tests/diart_server/test_e2e_websocket.py \
  --ignore=tests/diart_server/test_end_control.py --ignore=tests/diart_server/test_failure_modes.py \
  -p no:cacheprovider
-> 149 passed, 1 skipped in 74.08s
```

Baseline note: the workflow brief says "139 passed, 1 skipped", but that was
measured before the ring-clock fix landed; re-measured at base aaaa171 via
`git stash` on this machine: **146 passed, 1 skipped in 73.72s**. Post-fix
149 = 146 + the 3 new tests of §1; zero previously-passing tests broken, and
149 >= the spec's ">= 142" criterion. The two touched files alone:
`pytest tests/diart_server/test_voiceprint_routes.py tests/diart_server/test_live_route.py`
-> 7 passed.

### 3.2 Trim numeric criteria — PASS

Inside the new tests (and a standalone check of `_trim_silence`):
sil+tone+sil clip -> 33280 samples = **2.08 s**, within [30400, 35200];
4 s all-silence -> **HTTP 400** via TestClient with `说话时间不足` in the
detail and `extract` never invoked.

### 3.3 Frontend static asserts — PASS

All nine asserts of `test_live_page_enrollment_channel_static_asserts` green
(part of the 149 above).

### 3.4 Cross-matrix gate (shared math untouched) — PASS

```
DIART_MODELS_DIR=./models DIART_DEVICE=cuda DIART_VOICEPRINTS_DIR=./voiceprints \
  python scripts/verify_speaker_cross_matrix.py
FALSE_ACCEPTS=0 (of 15 impostor turns)
MISSED_GENUINE=0 (of 3 asserted genuine turns)
SESSION_ERRORS=0
RESULT: PASS
```

(newsA genuine 0.992 accepted; newsB impostor None @ 0.662; the
by-design-exempt cross-channel genuine still None @ 0.793.) Additionally
`git diff aaaa171 -- src/diart_server/embedding_extractor.py
src/diart_server/voiceprint_matcher.py src/diart_server/config.py` is empty.

### 3.5 Per-turn identity invariants — unchanged

This fix touches only live.html capture/enrollment UX and the enrollment
route's preprocessing. Identity still comes solely from a turn's own
re-embedded audio (session.py untouched), no server-side fallback was added,
and identification errors still fail closed to `name=None`.

### 3.6 Manual demo check — not run

Requires a browser + microphone (spec §5.6 marks it optional/not automatable
here). Left to a human: enroll once (6 s countdown), confirm own turns show
the enrolled name >= 0.82 and `n_clips` increments on repeat clips.

## 4. Residual risk (unchanged from spec §6)

- Voiceprints enrolled before this fix went through the old Opus channel and
  will still score low against the raw-PCM live channel; users must delete
  and re-enroll (`DELETE /v1/voiceprints/<name>`). No migration attempted.
- AGC/NS/AEC now off may lower ASR quality in noisy/echoey rooms; the demo
  targets quiet near-field use.
- The 0.826 same-channel genuine floor leaves thin margin; the surfaced
  multi-clip hint is the mitigation, wider calibration stays tracked in
  2026-07-06 §6.
- `ScriptProcessorNode` remains (deprecated); channel identity, not
  modernization, was this fix's principle.

## Review

pending
