# Enroll-channel mismatch: demo enrollment (MediaRecorder/Opus) vs live identification (raw PCM) pushes genuine users under the 0.82 threshold

- **Date:** 2026-08-05
- **Reporter:** bug-reporter agent (Reporter -> Fixer -> Reviewer loop)
- **HEAD at investigation:** a6128f1
- **Status:** Root cause confirmed in current code. Channel + enrollment-quality
  penalty measured offline with the real wespeaker model on cuda (§3, numbers
  recorded inline; measurement scripts were session-scratchpad, the tables
  below are the record). Design is settled (§4); constants validated (§3.4).

## 1. Symptom (user-visible, /demo/live)

Cross-machine HTTPS demo (:7443), `embedding=pyannote/wespeaker-voxceleb-resnet34-LM`.
A user enrolls their own voice ("录 4 秒并注册"), then starts a live session
and speaks. Their own turns come back as the grey anonymous `speakerN` label —
scores hover just *below* the 0.82 match threshold (typically 0.75–0.81)
instead of the >= 0.84 a genuine same-channel speaker scores. Since the
2026-07-06 per-turn identity fix deliberately fails closed (below threshold ->
`name=None`, never a wrong name), the practical effect is that **enrollment
appears to do nothing**: everyone is an unnamed `speakerN`. This is the top
usability killer of the demo.

Secondary user-visible symptoms with the same root file:

- The first words of the first utterance never appear in the transcript
  (mic starts before the WebSocket is open; early frames are dropped).
- Enrolling while a live session runs injects the enrollment speech into the
  live diarization/clustering (the live mic keeps streaming during the
  enrollment recording).
- The model `<select>`s stay active during a session/enrollment; changing
  `embedding` mid-session silently enrolls into a different model namespace
  than the live session matches against.

## 2. Root cause (file:line, verified at HEAD a6128f1)

### 2.1 Enrollment and identification always use different audio channels

Live identification channel (what every per-turn identity vector sees):

- `src/diart_server/static/live.html:58-67` — `capture()`:
  `getUserMedia({audio:{channelCount:1}})` -> `AudioContext` +
  `ScriptProcessor(8192,1,1)` raw **float32 at the context's native rate**
  (typically 48 kHz).
- `live.html:53-56` — `downsampleSend()` ships the raw float32 buffer;
  `src/diart_server/ws_routes.py:193` decodes it via
  `audio_decoder.to_pcm_16k_mono`, which resamples with
  **`torchaudio.functional.resample`** (`src/diart_server/audio_decoder.py:50-56`).
- `src/diart_server/session.py:652-656, 690-714` — per-turn identity embeds
  that 16 kHz live-channel audio with `embedding_extractor.extract`.

Enrollment channel (what the reference vector sees) — entirely different:

- `live.html:124` — `new MediaRecorder(s)`: WebM/**Opus lossy codec**.
- `live.html:129-133` — `decodeAudioData` then
  `OfflineAudioContext(1, n, 16000)`: the **browser's** resampler
  (Chrome renders `AudioBufferSourceNode` with linear interpolation), not
  torchaudio.
- `live.html:134-137` — `encodeWav(rendered, 16000)` uploaded; the server's
  `_resample_if_needed` (`src/diart_server/voiceprint_routes.py:47-55`,
  torchaudio) is then a no-op because the clip is already 16 kHz.

So the enrollment reference and every live query are **always cross-channel**:
lossy Opus + a different resampler on one side, raw PCM + torchaudio on the
other.

### 2.2 The 0.82 threshold was calibrated same-channel with ~0.02 margin

`src/diart_server/config.py:46-57`: `match_threshold=0.82`, calibrated on
turn-level impostor max ~0.79 vs asserted-genuine min ~0.84 — a same-channel
calibration whose comment **explicitly admits the failure mode**
(config.py:53-54): "Cross-channel genuine may fall below 0.82 -> shown as
unknown". That was intended as the rare fail-closed direction; the demo's
enrollment path makes *every* enrollment cross-channel, so the rare direction
became the default experience. Measured same-channel genuine minimum on studio
audio is 0.826 (§3.1) — the entire genuine margin above 0.82 is < 0.02, so any
systematic channel cost lands genuine users under the threshold.

### 2.3 Aggravators (all in current code)

- **4 s clip including reaction time, no trim:** `live.html:141`
  (`setTimeout(()=>rec.stop(), 4000)`) starts timing at button press, so the
  clip contains button-press reaction silence and trailing silence.
  `voiceprint_routes.py:65-72` only rejects `< 0.5 s` total — an all-silence
  4 s clip enrolls successfully today (HTTP 200) and there is **no silence
  trim** before `extract`. Silence dilutes the utterance embedding
  (`embedding_extractor.extract` embeds the whole clip, `embedding_extractor.py:49-56`).
- **No multi-clip guidance:** the store averages clips per name
  (`voiceprint_store.py:50-52`) and the route returns `n_clips`
  (`voiceprint_routes.py:98-99`, `protocol.py:146-151`), but the UI discards it
  (`live.html:139` shows only "已注册 <name>").
- **Browser DSP left on:** both `getUserMedia` call sites (`live.html:59`,
  `live.html:123`) pass only `{channelCount:1}`; `echoCancellation`,
  `noiseSuppression`, `autoGainControl` default to **true**. AGC's
  time-varying gain perturbs embeddings differently per recording.
- **First words dropped:** `live.html:97-102` awaits `capture()` *before*
  constructing the WebSocket; `downsampleSend` (`live.html:55`) silently drops
  frames while `ws.readyState!==1`, and the server only sends `ready` after
  `session.start()` (`ws_routes.py:165-166`).
- **Enrollment leaks into the live session:** the enroll handler
  (`live.html:120-142`) opens a second `getUserMedia` stream but never pauses
  `downsampleSend`, so the live mic keeps streaming the enrollment speech into
  diarization/clustering.
- **Selects never locked:** nothing disables `#emb`/`#seg` (`live.html:18-21`);
  enrollment reads `$('#emb').value` at click time (`live.html:136`) while the
  session was built with the value at start time (`live.html:100`).

## 3. Evidence (measured, wespeaker + committed cohort + threshold 0.82, cuda)

Offline simulation of the two channels on identical audio. Source: AISHELL-3
studio clips (44.1 kHz, local HF cache), 3 speakers (SSB0011/SSB0261/SSB0341);
enroll on clips 1+2 concatenated, genuine queries = the remaining clips
(5 genuine queries pooled). Queries ALWAYS go through the live channel
(float32 -> `torchaudio.functional.resample` -> 16 k -> `extract`), exactly as
the server does per turn. Enrollment-channel surrogate for the current demo:
16-bit WAV -> ffmpeg **libopus** (MediaRecorder surrogate) -> decode @48 k ->
**linear-interpolation** resample 48 k -> 16 k (Chrome
`OfflineAudioContext`/`AudioBufferSourceNode` surrogate) -> 16-bit quantize
(`encodeWav`). Scores are the real chain
`embedding_extractor.extract -> voiceprint_matcher.match(..., mean_embedding=cohort)`.

### 3.1 Channel cost, clean full-length enrollment audio (pooled)

```
enrollment variant              mean    min   genuine below 0.82
same-channel (torchaudio)      0.888  0.826   0/5
opus64k + linear resample      0.887  0.838   0/5
opus32k + linear resample      0.878  0.830   0/5
linear resample only (no opus) 0.888  0.839   0/5
```

On *clean, silence-free, ~8 s* enrollment audio the codec+resampler cost is
small (<= 0.010). But the genuine floor is 0.826 — only 0.006 above the
threshold — before any real-world loss (AGC, mic noise, reaction silence).

### 3.2 The demo's actual clip shape is what breaks it (pooled, decomposed)

The demo clip is 4 s from button press: ~0.7 s reaction silence + ~3.3 s
speech, untrimmed, through the Opus channel:

```
enrollment variant                        mean    min   genuine below 0.82
full-length, same-channel                0.888  0.826   0/5
3.3 s speech only, same-channel          0.877  0.824   0/5   (short: -0.011)
0.7s sil + 3.3s speech, same-channel     0.854  0.803   1/5   (silence: -0.023)
0.7s sil + 3.3s speech, opus64+linear    0.813  0.771   4/5   (channel: -0.041)
same clip, TRIMMED (§4 trim), same-ch    0.875  0.821   0/5   (recovered)
```

**The current demo enrollment (bottom-but-one row) drops pooled genuine mean
0.888 -> 0.813 and puts 4 of 5 genuine queries below 0.82** (0.771–0.817).
The channel cost *interacts* with silence: Opus over the silence-bearing clip
costs -0.041 where it cost <= 0.010 on clean speech. The proposed fix
(raw-PCM channel + route-level trim) recovers to 0.875 / 0 of 5 below.

Not simulatable offline: AGC/noise-suppression/echo-cancellation (enabled by
default on both paths today, §2.3) add further recording-dependent loss on
real mics. Real-condition confirmation from the repo's own measurements:
`docs/bugs/2026-07-06-voiceprint-false-accept.md` §3.3 measured a real
cross-channel genuine pair (2news newscaster, different broadcast segment) at
**0.796 < 0.82**, and `config.py:53-54` codifies that cross-channel genuine
falls below threshold. The demo made that the default path for every user.

### 3.3 Margin arithmetic

Same-channel calibration: impostor max ~0.79, genuine min ~0.84 (config.py
comment), measured studio genuine min 0.826. Threshold 0.82 leaves genuine
margin 0.006–0.02 — smaller than the measured demo-channel cost (0.04–0.08,
§3.2). Chronic sub-threshold genuine scores are the arithmetic consequence.
The threshold itself is NOT wrong (impostor max ~0.79 still needs 0.82); the
enrollment channel must move to the live channel, not the threshold down.

### 3.4 Trim-constant validation (constants fixed for §4.2)

With frames = 320 samples (20 ms @ 16 k), voiced = frame RMS >= 0.05 x peak
frame RMS, all-silence floor = peak frame RMS < 1e-3, pad = 2 frames (40 ms)
each side, minimum kept speech 1.5 s:

```
1 s silence + 2 s 440 Hz tone (amp 0.3) + 1 s silence @16 k -> trimmed 2.08 s  (in [1.9, 2.2])
4 s all-zero clip                                            -> rejected (all-silence)
0.7 s silence + real AISHELL clip (4.16 s total)             -> trimmed 2.50 s
```

## 4. Design (decisions final; a fixer needs no further choices)

Principle: **enroll through the exact channel that identifies** — same capture
node, same wire format, same server-side resampler — and stop feeding silence
to the extractor. `embedding_extractor.py` and `voiceprint_matcher.py` are
untouched; `config.py` is untouched.

### 4.1 Frontend `src/diart_server/static/live.html`

1. **Shared mic constraints + single `getUserMedia` call site.** Add
   `const MIC = {channelCount:1, echoCancellation:false, noiseSuppression:false, autoGainControl:false};`
   and a helper `openMic()` that does
   `stream = await navigator.mediaDevices.getUserMedia({audio:MIC})` and wires
   `MediaStreamSource -> ScriptProcessor(8192,1,1)` on the current `audioCtx`,
   with `proc.onaudioprocess` delivering `Float32Array` frames to a single
   router `onMicFrame`. Both the live path and enrollment use this one call
   site (this is how "constraints on BOTH paths" is implemented; the static
   test in §4.3 asserts exactly one `getUserMedia(` occurrence using `MIC`).
2. **Remove MediaRecorder entirely.** Delete the
   `MediaRecorder`/`decodeAudioData`/`OfflineAudioContext` block
   (live.html:123-141). The string `MediaRecorder` must not appear in the file.
3. **Raw-PCM enrollment at native rate.** New state
   `let enrolling=false, enrollBuf=[];` and `const ENROLL_MS = 6000;`
   (exact literal `const ENROLL_MS = 6000;`). Frame router:
   `function onMicFrame(f){ if(enrolling){ enrollBuf.push(f); return; } downsampleSend(f); }`
   plus a defensive first line in `downsampleSend`: `if(enrolling) return;`
   (this is the pause-during-enroll guard; enrollment speech never reaches the
   live WebSocket). Enroll click handler:
   - empty-name check as today; `if(enrolling) return;`.
   - if not `running`: `audioCtx = new AudioContext(); await openMic();`
     and remember `ownCapture=true`. If `running`: reuse the live capture
     (no second stream).
   - `enrollBuf=[]; enrolling=true;` disable `#enrollBtn`, `#startBtn`,
     `#stopBtn`, `#emb`, `#seg`.
   - visible countdown in `#enrollStatus`, updated every 1 s:
     `录音中… 剩 6s` down to `1s` (driven off `ENROLL_MS`).
   - after `ENROLL_MS`: `enrolling=false` (live frames resume); concatenate
     `enrollBuf` into one `Float32Array`; **`encodeWav(f32, audioCtx.sampleRate)`**
     — 16-bit WAV at the NATIVE rate (the existing `encodeWav` already takes
     `sr`); the server resamples with torchaudio
     (`voiceprint_routes._resample_if_needed`), the same resampler family as
     the live path. Upload with the same FormData fields as today.
     If `ownCapture`: disconnect the node, close the context, stop the tracks.
   - response handling: on OK parse JSON and show
     `已注册 <name>（第<n_clips>条）— 再录1-2条更稳`; on error show
     `注册失败 <status>` plus the JSON `detail` when present.
   - re-enable: `#enrollBtn` always; `#emb`/`#seg`/`#startBtn` only if
     `!running`; `#stopBtn` only if `running`.
4. **WebSocket before mic (start flow).** In the start handler: disable the
   selects; create `audioCtx = new AudioContext()` first and use
   `audioCtx.sampleRate` for the `sample_rate` query param; open the WebSocket
   immediately; call `await openMic()` **inside the `'ready'` message
   handler** — the mic only starts once the server is accepting audio, so no
   frame can be dropped by the `ws.readyState!==1` guard. On `error`/`close`
   before ready, tear down and re-enable controls.
5. **Lock the model selects** whenever a live session is running OR an
   enrollment is recording (set/cleared at the points listed in 3 and 4;
   stop re-enables them only when not enrolling).

### 4.2 Server `src/diart_server/voiceprint_routes.py` — route-level trim

Add module-level constants (values validated in §3.4):

```python
TRIM_FRAME = 320          # 20 ms @ 16 kHz, non-overlapping
TRIM_REL_RMS = 0.05       # voiced: frame RMS >= 5% of peak frame RMS (~ -26 dB)
TRIM_ABS_FLOOR = 1e-3     # peak frame RMS below this => all-silence clip
TRIM_PAD_FRAMES = 2       # keep 40 ms context on each side
MIN_SPEECH_SECONDS = 1.5  # required speech after trimming
```

New helper `_trim_silence(pcm: np.ndarray, sr: int = 16000) -> np.ndarray | None`:
split into non-overlapping `TRIM_FRAME` frames; per-frame RMS (float64);
`peak = rms.max()`; if `peak < TRIM_ABS_FLOOR` return `None`; voiced indices =
`rms >= TRIM_REL_RMS * peak`; keep
`[max(0, first_voiced - TRIM_PAD_FRAMES) * TRIM_FRAME : min(n_frames, last_voiced + 1 + TRIM_PAD_FRAMES) * TRIM_FRAME]`;
if the result is shorter than `MIN_SPEECH_SECONDS * sr` return `None`.

In `enroll()` (voiceprint_routes.py:58-99): after `_resample_if_needed` and the
existing `< 0.5 s` check (both unchanged), do

```python
trimmed = _trim_silence(pcm)
if trimmed is None:
    raise HTTPException(status_code=400,
        detail=f"说话时间不足：修剪静音后有效语音不足 {MIN_SPEECH_SECONDS:g} 秒，请靠近麦克风重新录制")
```

and pass `trimmed` to `extract`. **Do NOT modify
`src/diart_server/embedding_extractor.py`** — the per-turn identity path and
`scripts/verify_speaker_cross_matrix.py` (which enrolls via `extract`
directly, bypassing this route) must remain byte-identical.

### 4.3 Tests (update + add)

- `tests/diart_server/test_voiceprint_routes.py`:
  - change `_wav_bytes` to generate a **tone**, not zeros (zeros would now be
    rejected as all-silence): default `seconds=2.0`,
    `0.3 * sin(2*pi*440*t)` at 16 k, 16-bit — the three existing tests then
    pass unchanged.
  - add `test_enroll_trims_leading_trailing_silence`: monkeypatch
    `make_embedder -> object()` and `extract` to capture the pcm it receives;
    POST a WAV of 1 s silence + 2 s tone + 1 s silence @16 k; assert
    status 200 and `1.9*16000 <= captured.size <= 2.2*16000`.
  - add `test_enroll_all_silence_rejected_400`: POST a 4 s all-zero WAV;
    assert status **400**, `"说话时间不足" in r.json()["detail"]`, and the
    patched `extract` was never called.
- `tests/diart_server/test_live_route.py` — add a static-assert test on
  `GET /demo/live` (`r.text`):
  - `"MediaRecorder" not in r.text`
  - `r.text.count("getUserMedia(") == 1` and `"getUserMedia({audio:MIC})" in r.text`
  - `"echoCancellation:false"`, `"noiseSuppression:false"`,
    `"autoGainControl:false"` all in `r.text`
  - `"const ENROLL_MS = 6000;" in r.text`
  - `"if(enrolling) return;" in r.text` (pause-during-enroll guard)
  - `"n_clips" in r.text` (multi-clip feedback surfaced)

## 5. Verification (Reviewer runs all; numeric criteria)

1. **Unit suite (the only acceptance test command); baseline before this
   workflow: 139 passed, 1 skipped:**

   ```
   python -m pytest tests/diart_server -q --ignore=tests/diart_server/test_e2e_websocket.py --ignore=tests/diart_server/test_end_control.py --ignore=tests/diart_server/test_failure_modes.py -p no:cacheprovider
   ```

   -> 0 failures; all previously passing tests stay green; the 3 new tests of
   §4.3 collected and passing (expected >= 142 passed, 1 skipped).

2. **Trim numeric criteria** (inside the new tests): synthetic
   1 s silence + 2 s tone + 1 s silence @16 k -> trimmed length within
   **[1.9, 2.2] s** ([30400, 35200] samples); 4 s all-silence clip ->
   **HTTP 400** from the enroll route via TestClient
   (`assert r.status_code == 400`).

3. **Frontend static asserts** of §4.3 all pass (no `MediaRecorder`; single
   `getUserMedia` with the three constraints `false`; `ENROLL_MS = 6000`;
   `if(enrolling) return;` guard present).

4. **Cross-matrix gate (no collateral damage to matching/embedding math):**

   ```
   DIART_MODELS_DIR=./models DIART_DEVICE=cuda DIART_VOICEPRINTS_DIR=./voiceprints python scripts/verify_speaker_cross_matrix.py
   ```

   -> **FALSE ACCEPTS = 0 and MISSED GENUINE = 0**. The script enrolls via
   `embedding_extractor.extract` directly (bypasses the HTTP route), so a PASS
   proves the route-level trim added nothing to the shared math. If this
   machine lacks models/cuda at review time, the reviewer records exactly why
   it was skipped. Additionally
   `git diff <this-spec's-parent> -- src/diart_server/embedding_extractor.py src/diart_server/voiceprint_matcher.py src/diart_server/config.py`
   must be empty.

5. **Per-turn identity invariants unchanged**
   (docs/bugs/2026-07-06-voiceprint-false-accept.md): this change touches only
   live.html enrollment/capture UX and the enrollment route's preprocessing;
   identity still comes only from a turn's own re-embedded audio, no
   server-side fallback is introduced, and identification errors still fail
   closed to `name=None`.

6. **Manual demo check (optional, needs a browser + mic; not automatable
   here):** enroll once (6 s countdown visible), speak — own turns should now
   display the enrolled name with score >= 0.82; record 2 more clips per the
   hint and confirm `n_clips` increments.

## 6. Out of scope / residual risk

- AGC/NS/AEC are disabled going forward, but voiceprints enrolled *before*
  this fix were recorded through the old Opus channel and will still score
  low against the new raw-PCM live channel. Users should delete and
  re-enroll (`DELETE /v1/voiceprints/<name>`); no migration is attempted.
- Disabling `noiseSuppression`/`echoCancellation` can lower ASR quality in
  noisy/echoey rooms. The demo targets a quiet near-field mic; revisit only
  with measurements.
- The 0.826 same-channel genuine floor (§3.1) leaves little margin even
  same-channel; multi-clip enrollment (now surfaced via `n_clips` + hint)
  is the mitigation, a wider-margin calibration remains tracked in
  2026-07-06 §6.
- `ScriptProcessorNode` is deprecated (AudioWorklet is the successor); it is
  retained here because the live path already depends on it and this fix's
  principle is channel *identity*, not modernization.
