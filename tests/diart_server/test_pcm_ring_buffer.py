import numpy as np
from diart_server.pcm_ring_buffer import PcmRingBuffer


def test_slice_returns_exact_window():
    rb = PcmRingBuffer(max_seconds=10, sample_rate=100)  # 100 Hz for easy math
    rb.append(np.arange(0, 100, dtype=np.float32))   # t=0..1s
    rb.append(np.arange(100, 300, dtype=np.float32))  # t=1..3s
    assert abs(rb.total_seconds - 3.0) < 1e-9
    seg = rb.slice(1.0, 2.0)
    assert seg is not None
    np.testing.assert_array_equal(seg, np.arange(100, 200, dtype=np.float32))


def test_slice_out_of_window_returns_none():
    rb = PcmRingBuffer(max_seconds=1, sample_rate=100)
    rb.append(np.zeros(100, dtype=np.float32))   # t=0..1
    rb.append(np.ones(100, dtype=np.float32))    # t=1..2, evicts t=0..1
    assert rb.slice(0.0, 0.5) is None             # evicted
    np.testing.assert_array_equal(rb.slice(1.0, 2.0), np.ones(100, dtype=np.float32))


def test_empty_slice_returns_none():
    rb = PcmRingBuffer(max_seconds=5, sample_rate=100)
    rb.append(np.zeros(100, dtype=np.float32))
    assert rb.slice(0.5, 0.5) is None
