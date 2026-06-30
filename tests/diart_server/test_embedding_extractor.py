import numpy as np
import torch
from diart_server.embedding_extractor import extract


class FakeEmbedder:
    """Mimics diart SpeakerEmbedding: takes (samples, channels) -> (dim,)."""
    def __call__(self, waveform):
        return torch.tensor([3.0, 4.0])  # norm 5 -> expect [0.6, 0.8]


def test_extract_normalizes():
    v = extract(np.zeros(16000, dtype=np.float32), FakeEmbedder())
    np.testing.assert_allclose(v, np.array([0.6, 0.8], dtype=np.float32), atol=1e-6)
    assert v.dtype == np.float32 and v.ndim == 1
