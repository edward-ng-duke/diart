import asyncio
import numpy as np
import pytest
from diart_server.asr_client import parse_asr_output, encode_wav, AsrClient


def test_parse_with_marker():
    assert parse_asr_output("language Chinese<asr_text>你好世界") == ("Chinese", "你好世界")


def test_parse_without_marker():
    assert parse_asr_output("just text") == ("", "just text")


def test_parse_language_none():
    assert parse_asr_output("language None<asr_text>hi") == ("", "hi")


def test_encode_wav_is_riff_pcm16():
    wav = encode_wav(np.zeros(160, dtype=np.float32), 16000)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"


def test_transcribe_parses_response(monkeypatch):
    client = AsrClient("http://x/v1", "m", "EMPTY", timeout=5)

    async def fake_post(self, wav_bytes, language):
        return "language Chinese<asr_text>山东"

    monkeypatch.setattr(AsrClient, "_post_wav", fake_post)
    lang, text = asyncio.run(client.transcribe(np.zeros(160, dtype=np.float32)))
    assert lang == "Chinese" and text == "山东"
