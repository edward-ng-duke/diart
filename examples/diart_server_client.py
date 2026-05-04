"""Standalone WebSocket client for diart_server.

Reads a wav file, streams it in 0.5s chunks at real-time pace to
``ws://<host>:<port>/v1/diarize/stream``, prints each ``prediction`` (speaker
label + first 5 embedding values), and dumps the final RTTM at end-of-stream.

Usage::

    python examples/diart_server_client.py [--wav PATH] [--host H] [--port P]

Requires ``websockets`` and ``numpy`` (both installed by ``pip install -e
'.[server]'``).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import wave
from pathlib import Path

import numpy as np
import websockets


async def _send_frames(ws, pcm: np.ndarray, chunk: int, sample_rate: int) -> None:
    """Stream ``pcm`` to ``ws`` as ~0.5s float32 frames at real-time pace."""
    for i in range(0, len(pcm), chunk):
        frame = pcm[i : i + chunk]
        if len(frame) == 0:
            break
        await ws.send(frame.tobytes())
        # real-time pacing: sleep for the duration of the frame we just sent
        await asyncio.sleep(len(frame) / float(sample_rate))
    # signal end-of-stream; server flushes buffer then sends `final`
    await ws.send(json.dumps({"type": "end"}))


async def stream(wav_path: Path, host: str, port: int) -> None:
    with wave.open(str(wav_path), "rb") as w:
        sample_rate = w.getframerate()
        nchannels = w.getnchannels()
        sampwidth = w.getsampwidth()
        raw = w.readframes(w.getnframes())

    if sampwidth != 2:
        raise ValueError(
            f"unsupported sample width {sampwidth} bytes; expected int16 PCM wav"
        )
    arr = np.frombuffer(raw, dtype=np.int16)
    if nchannels == 2:
        arr = arr.reshape(-1, 2).mean(axis=1)
    pcm = (arr.astype(np.float32) / 32768.0).copy()
    chunk = sample_rate // 2  # 0.5s

    url = f"ws://{host}:{port}/v1/diarize/stream?sample_rate={sample_rate}"
    print(f"[connect] {url} ({len(pcm) / sample_rate:.2f}s of audio)")

    async with websockets.connect(url) as ws:
        ready_raw = await ws.recv()
        ready = json.loads(ready_raw)
        print(f"[ready] session_id={ready['session_id']} models={ready['models']}")

        send_task = asyncio.create_task(_send_frames(ws, pcm, chunk, sample_rate))
        try:
            while True:
                raw_msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                msg = json.loads(raw_msg)
                mtype = msg.get("type")
                if mtype == "prediction":
                    emb = msg.get("embedding") or []
                    emb_preview = [round(float(x), 4) for x in emb[:5]]
                    print(
                        f"[pred] speaker={msg['speaker']} "
                        f"start={msg['start']:.2f} dur={msg['duration']:.2f} "
                        f"emb[0:5]={emb_preview} "
                        f"samples_count={msg.get('samples_count')}"
                    )
                elif mtype == "final":
                    print(
                        f"[final] duration={msg['duration']:.2f} "
                        f"num_speakers={msg['num_speakers']} "
                        f"stats={msg.get('stats')}"
                    )
                    print("--- RTTM ---")
                    print(msg["rttm"])
                    break
                elif mtype == "error":
                    print(f"[error] code={msg.get('code')} message={msg.get('message')}")
                    break
                else:
                    print(f"[unknown] {msg}")
        except asyncio.TimeoutError:
            print("[timeout] no message received in 30s")
        finally:
            send_task.cancel()
            try:
                await send_task
            except (asyncio.CancelledError, Exception):
                pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream a wav file to diart_server and print predictions.",
    )
    parser.add_argument(
        "--wav",
        default="tests/data/audio/sample.wav",
        help="path to a 16-bit PCM wav file (default: tests/data/audio/sample.wav)",
    )
    parser.add_argument("--host", default="localhost", help="diart_server host")
    parser.add_argument(
        "--port", default=7007, type=int, help="diart_server port (default: 7007)"
    )
    args = parser.parse_args()

    asyncio.run(stream(Path(args.wav), args.host, args.port))


if __name__ == "__main__":
    main()
