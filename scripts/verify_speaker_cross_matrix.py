#!/usr/bin/env python
"""Turn-level speaker-identity cross-validation with REAL voices.

Spec: docs/bugs/2026-07-06-voiceprint-false-accept.md

Reproduces (pre-fix) and gates (post-fix) the voiceprint false-accept bug:
an enrolled user's name was stamped on every transcript turn, including turns
spoken by completely different people, at scores 0.77-0.94, because turn
identity was inherited from the diart CLUSTER CENTROID instead of the turn's
own audio (src/diart_server/session.py _on_prediction/_handle_turn).

What it does, per scenario:
  1. enrolls ONE speaker via the real enrollment path
     (embedding_extractor.extract on a clip never reused as a probe),
  2. replays a composite multi-speaker stream through a REAL
     DiarizationSession (real segmentation + wespeaker embedding, real online
     clustering, DIART_DEVICE honored; ASR is stubbed - identity does not
     depend on ASR text),
  3. records every FINAL transcript turn: diart label, name, score, span,
     ground-truth source speaker (known from stream construction),
  4. also prints the per-turn TRUTH MATRIX: each turn's own (overlap-excised)
     slice re-embedded with embedding_extractor and matched with vp_match -
     what a per-turn identity implementation would produce.

Voice pool (all real, no pitch-shifting):
  - AISHELL-3 speakers from the local HF cache (studio Mandarin, same-gender
    hard negatives, incl. the near-twin pair SSB0261/SSB0590),
  - the two data/2news.mp3 broadcast speakers (newscaster 0.6-6.6s & 19-24s,
    field reporter 7.8-19s) - the live-demo-shaped scenario.

PASS criteria (asserted at TURN level, wespeaker model):
  - zero FALSE ACCEPTS: a turn whose ground truth is NOT the enrolled speaker
    must not carry the enrolled name at score >= settings.match_threshold;
  - zero MISSED GENUINE: a turn whose ground truth IS the enrolled speaker,
    with duration >= GENUINE_MIN_SECONDS and same-session audio (scenarios
    S1-S3), must carry the enrolled name at score >= settings.match_threshold.
    (S4's cross-segment 19-24s newscaster turn is exempt: cross-channel
    genuine may legitimately fall below threshold -> name None is acceptable;
    a WRONG name is not.)

Exit codes: 0 PASS, 1 FAIL, 3 SKIPPED (models/audio unavailable).

Usage (from repo root):
    DIART_MODELS_DIR=./models DIART_DEVICE=cuda python scripts/verify_speaker_cross_matrix.py

Recorded pre-fix baseline (HEAD c341aca, threshold 0.6, 2news-only cohort):
see the spec doc - the run FAILS with false accepts in every scenario.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EMB_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
SR = 16000
GAP_SECONDS = 1.2          # inter-utterance silence, > settings.silence_gap
GENUINE_MIN_SECONDS = 2.25  # only turns this long must be correctly accepted
NEWS = REPO / "data" / "2news.mp3"

# AISHELL-3 speakers in the local HF cache. Per speaker: enroll on the first
# two clips, probe/stream with the rest. Never downloads anything.
AISHELL_GLOB = "datasets--AISHELL--AISHELL-3/snapshots/*/train/wav"
POOL_SPEAKERS = ["SSB0011", "SSB0261", "SSB0288", "SSB0341", "SSB0590"]


def find_aishell_root() -> Path | None:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    hits = sorted(hub.glob(AISHELL_GLOB))
    return hits[0] if hits else None


def load_wav(p: Path) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(str(p), dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    if sr != SR:
        import librosa

        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    return np.asarray(data, dtype=np.float32)


def news_slice(ss: float, dur: float, cache: Path) -> np.ndarray:
    out = cache / f"news_{ss:g}_{dur:g}.wav"
    if not out.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(ss), "-t", str(dur),
             "-i", str(NEWS), "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
             str(out)],
            check=True,
        )
    return load_wav(out)


def build_stream(segments: list[tuple[str, np.ndarray]]):
    """Concatenate (speaker_id, pcm) with silence gaps; return pcm + layout."""
    gap = np.zeros(int(GAP_SECONDS * SR), dtype=np.float32)
    parts = [np.zeros(int(0.5 * SR), dtype=np.float32)]
    layout: list[tuple[float, float, str]] = []
    t = 0.5
    for spk, pcm in segments:
        parts.append(pcm)
        layout.append((t, t + len(pcm) / SR, spk))
        t += len(pcm) / SR
        parts.append(gap)
        t += GAP_SECONDS
    return np.concatenate(parts), layout


def gt_speaker(layout, start, end):
    best, best_ov = None, 0.0
    for s, e, spk in layout:
        ov = min(e, end) - max(s, start)
        if ov > best_ov:
            best, best_ov = spk, ov
    return best


class StubAsr:
    async def transcribe(self, pcm, language=None):  # noqa: ANN001, ARG002
        return ("Chinese", f"<{len(pcm) / SR:.1f}s>")


def run_session(registry, store, stream: np.ndarray, params):
    """Replay a pcm stream through a real DiarizationSession; return messages
    and the (closed) session for post-hoc ring access."""
    from diart_server.session import DiarizationSession

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    messages: list[dict] = []
    lock = threading.Lock()

    async def on_message(msg: dict) -> None:
        with lock:
            messages.append(msg)

    session = DiarizationSession(
        params=params, registry=registry, on_message=on_message,
        loop=loop, store=store, asr_client=StubAsr(),
    )
    try:
        session.start()
        step = int(0.5 * SR)
        for i in range(0, len(stream), step):
            session.push_pcm(stream[i:i + step])
            # Keep the bounded source queue shallow so no frame is dropped
            # (faster-than-realtime replay stays lossless).
            while session.source._queue.qsize() > 10:
                time.sleep(0.05)
            time.sleep(0.02)
        # Trailing silence so the silence clock finalizes open turns.
        for _ in range(6):
            session.push_pcm(np.zeros(step, dtype=np.float32))
            time.sleep(0.1)
        fut = asyncio.run_coroutine_threadsafe(session.close(), loop)
        fut.result(timeout=120)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=5.0)
        loop.close()

    with lock:
        return list(messages), session


def main() -> int:
    if not NEWS.exists():
        print(f"SKIPPED: {NEWS} missing")
        return 3
    aishell = find_aishell_root()
    if aishell is None:
        print("SKIPPED: AISHELL-3 HF cache not found")
        return 3
    try:
        import torch

        from diart_server.app import MANIFEST_PATH
        from diart_server.config import settings
        from diart_server.embedding_extractor import extract, make_embedder
        from diart_server.models_registry import ModelsRegistry
        from diart_server.protocol import QueryParams
        from diart_server.session import _load_cohort_mean
        from diart_server.voiceprint_matcher import match as vp_match
        from diart_server.voiceprint_store import VoiceprintStore
    except Exception as exc:  # noqa: BLE001
        print(f"SKIPPED: imports unavailable ({exc})")
        return 3

    try:
        registry = ModelsRegistry(settings.models_dir, MANIFEST_PATH)
        registry.get_segmentation(QueryParams().segmentation_model)
        registry.get_embedding(EMB_MODEL)
    except Exception as exc:  # noqa: BLE001
        print(f"SKIPPED: models unavailable ({exc})")
        return 3

    device = torch.device(settings.device)
    # NOTE: device must be passed explicitly - SpeakerEmbedding(model, None)
    # moves the SHARED registry model to CPU (see spec doc, adjacent bug).
    embedder = make_embedder(EMB_MODEL, registry, device)
    cohort = _load_cohort_mean(EMB_MODEL)
    cache = Path(tempfile.gettempdir()) / "vp_cross_matrix_clips"
    cache.mkdir(parents=True, exist_ok=True)

    def clips(spk: str) -> list[Path]:
        return sorted((aishell / spk).glob("*.wav"))

    for spk in POOL_SPEAKERS:
        if len(clips(spk)) < 3:
            print(f"SKIPPED: AISHELL speaker {spk} has <3 clips in cache")
            return 3

    def enroll_pcm(spk: str) -> np.ndarray:
        return np.concatenate([load_wav(w) for w in clips(spk)[:2]])

    def probes(spk: str) -> list[np.ndarray]:
        return [load_wav(w) for w in clips(spk)[2:]]

    # ---- scenarios: (name, enrolled_id, enrollment_pcm, [(gt_id, pcm), ...],
    #                  assert_genuine)
    scenarios = []
    # S1: enroll SSB0011; impostors SSB0341 + SSB0590 (merged-cluster case).
    scenarios.append((
        "S1-aishell-enroll-SSB0011", "SSB0011", enroll_pcm("SSB0011"),
        [("SSB0011", probes("SSB0011")[0])]
        + [("SSB0341", p) for p in probes("SSB0341")[:2]]
        + [("SSB0590", p) for p in probes("SSB0590")[:2]],
        True,
    ))
    # S2: enroll SSB0341; impostors SSB0011 + SSB0288.
    scenarios.append((
        "S2-aishell-enroll-SSB0341", "SSB0341", enroll_pcm("SSB0341"),
        [("SSB0341", probes("SSB0341")[0]),
         ("SSB0011", probes("SSB0011")[0])]
        + [("SSB0288", p) for p in probes("SSB0288")[:2]],
        True,
    ))
    # S3: near-twin pair - enroll SSB0261, impostor SSB0590 (hardest negative).
    scenarios.append((
        "S3-neartwin-enroll-SSB0261", "SSB0261", enroll_pcm("SSB0261"),
        [("SSB0261", p) for p in probes("SSB0261")[:2]]
        + [("SSB0590", p) for p in probes("SSB0590")[:2]],
        True,
    ))
    # S4: live-demo-shaped - enroll the 2news newscaster on their opening
    # statement, then replay the broadcast (newscaster / field reporter /
    # newscaster). Cross-segment genuine is exempt from the accept assertion.
    scenarios.append((
        "S4-2news-enroll-newscaster", "newsA", news_slice(0.6, 6.0, cache),
        [("newsA", news_slice(0.6, 6.0, cache)),
         ("newsB", news_slice(7.8, 11.2, cache)),
         ("newsA", news_slice(19.0, 4.8, cache))],
        False,
    ))

    thr = settings.match_threshold
    params = QueryParams(embedding_model=EMB_MODEL)
    total_fa = total_miss = total_errors = 0
    total_impostor_turns = total_genuine_turns = 0

    for scen_name, enrolled, epcm, segs, assert_genuine in scenarios:
        store = VoiceprintStore(Path(tempfile.mkdtemp()) / "vp")
        store.enroll(EMB_MODEL, enrolled, extract(epcm, embedder))
        stream, layout = build_stream(segs)

        print(f"\n===== {scen_name} (enrolled={enrolled}, thr={thr}) =====")
        for s, e, spk in layout:
            mark = "  <-- enrolled voice" if spk == enrolled else ""
            print(f"  layout {s:6.2f}-{e:6.2f} {spk}{mark}")

        messages, session = run_session(registry, store, stream, params)
        errors = [m for m in messages if m.get("type") == "error"]
        for e in errors:
            print(f"  ERROR message during session: {e}")
        total_errors += len(errors)
        finals = [m for m in messages
                  if m.get("type") == "transcript" and not m.get("partial")]
        refs = store.vectors(EMB_MODEL)

        print(f"  {'span':>15} {'diart':>9} {'gt':>8} {'dur':>5} "
              f"{'name':>8} {'score':>6} | {'truth-name':>10} {'truth-score':>11}")
        for m in finals:
            gt = gt_speaker(layout, m["start"], m["end"])
            dur = m["end"] - m["start"]
            name, score = m.get("name"), m.get("score")
            # Truth matrix: the turn's own slice re-embedded + matched.
            pcm = session.ring.slice(m["start"], m["end"])
            tname, tscore = (None, None)
            if pcm is not None and pcm.size >= SR // 4:
                tvec = extract(pcm, embedder)
                tname, tscore = vp_match(tvec, refs, thr, mean_embedding=cohort)
            verdict = ""
            if gt != enrolled:
                total_impostor_turns += 1
                if name == enrolled and (score or 0.0) >= thr:
                    total_fa += 1
                    verdict = "FALSE-ACCEPT"
            elif assert_genuine and dur >= GENUINE_MIN_SECONDS:
                total_genuine_turns += 1
                if not (name == enrolled and (score or 0.0) >= thr):
                    total_miss += 1
                    verdict = "MISSED-GENUINE"
            print(f"  {m['start']:7.2f}-{m['end']:7.2f} {m['speaker']:>9} "
                  f"{gt:>8} {dur:5.2f} {str(name):>8} "
                  f"{(score if score is not None else float('nan')):6.3f} | "
                  f"{str(tname):>10} "
                  f"{(tscore if tscore is not None else float('nan')):11.3f} "
                  f"{verdict}")

    print(f"\nFALSE_ACCEPTS={total_fa} (of {total_impostor_turns} impostor turns)")
    print(f"MISSED_GENUINE={total_miss} (of {total_genuine_turns} asserted genuine turns)")
    print(f"SESSION_ERRORS={total_errors}")
    ok = total_fa == 0 and total_miss == 0 and total_errors == 0
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
