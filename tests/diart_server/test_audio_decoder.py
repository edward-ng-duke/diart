"""Tests for diart_server.audio_decoder."""

from __future__ import annotations

import numpy as np
import pytest

from diart_server.audio_decoder import to_pcm_16k_mono
from diart_server.errors import BadAudioError, ResampleFailedError


def test_passthrough_16k_returns_same_length() -> None:
    """A 500 ms 16 kHz silence buffer returns 8000 float32 samples."""
    silence = np.zeros(8000, dtype=np.float32)
    out = to_pcm_16k_mono(silence.tobytes(), 16000)
    assert out.dtype == np.float32
    assert out.shape == (8000,)


def test_resample_48k_to_16k_500ms() -> None:
    """A 500 ms 48 kHz buffer downsamples to roughly 8000 samples."""
    silence = np.zeros(24000, dtype=np.float32)
    out = to_pcm_16k_mono(silence.tobytes(), 48000)
    assert out.dtype == np.float32
    assert abs(out.shape[0] - 8000) <= 20


def test_empty_frame_raises_bad_audio() -> None:
    """Empty input bytes raise BadAudioError."""
    with pytest.raises(BadAudioError):
        to_pcm_16k_mono(b"", 16000)


def test_resample_22050_to_16k_1s() -> None:
    """A 1 s 22050 Hz buffer resamples to roughly 16000 samples."""
    silence = np.zeros(22050, dtype=np.float32)
    out = to_pcm_16k_mono(silence.tobytes(), 22050)
    assert out.dtype == np.float32
    assert abs(out.shape[0] - 16000) <= 20


def test_output_is_one_dimensional() -> None:
    """Both passthrough and resample paths return 1-D arrays."""
    pass_out = to_pcm_16k_mono(np.zeros(1600, dtype=np.float32).tobytes(), 16000)
    resampled_out = to_pcm_16k_mono(np.zeros(4800, dtype=np.float32).tobytes(), 48000)
    assert pass_out.ndim == 1
    assert resampled_out.ndim == 1


def test_invalid_src_sr_raises_resample_failed() -> None:
    """Negative or zero src_sr is wrapped as ResampleFailedError."""
    silence = np.zeros(1600, dtype=np.float32)
    with pytest.raises(ResampleFailedError):
        to_pcm_16k_mono(silence.tobytes(), -1)


def test_resample_preserves_signal_energy() -> None:
    """A 1 kHz sine resampled 48k→16k retains comparable RMS (non-trivial signal check)."""
    sr = 48000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    sine = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    out = to_pcm_16k_mono(sine.tobytes(), sr)
    rms_in = float(np.sqrt(np.mean(sine.astype(np.float64) ** 2)))
    rms_out = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    # Resampling should preserve RMS to within ~5% for a band-limited sine
    assert abs(rms_in - rms_out) / rms_in < 0.05
