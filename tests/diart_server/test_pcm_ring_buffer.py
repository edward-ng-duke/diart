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


def test_slice_spans_ring_wrap():
    rb = PcmRingBuffer(max_seconds=10, sample_rate=1)  # capacity 10
    for i in range(0, 14):                              # small appends -> ring wraps
        rb.append(np.array([float(i)], dtype=np.float32))
    np.testing.assert_array_equal(
        rb.slice(11.0, 14.0), np.array([11.0, 12.0, 13.0], dtype=np.float32)
    )


def test_single_append_larger_than_capacity():
    rb = PcmRingBuffer(max_seconds=5, sample_rate=1)    # capacity 5
    rb.append(np.arange(0, 12, dtype=np.float32))       # n=12 > capacity
    assert abs(rb.total_seconds - 12.0) < 1e-9
    np.testing.assert_array_equal(rb.slice(7.0, 12.0), np.arange(7, 12, dtype=np.float32))
    np.testing.assert_array_equal(rb.slice(9.0, 11.0), np.array([9.0, 10.0], dtype=np.float32))


def test_random_equivalence_against_reference():
    import random
    rng = random.Random(12345)
    for _ in range(1500):
        cap = rng.randint(3, 30)
        rb = PcmRingBuffer(max_seconds=cap, sample_rate=1)
        chunks, total = [], 0
        for _ in range(rng.randint(1, 8)):
            n = rng.randint(1, cap + 5)                 # includes big chunks
            c = np.arange(total, total + n, dtype=np.float32)
            rb.append(c); chunks.append(c); total += n
        full = np.concatenate(chunks)
        oldest = max(0, total - cap)
        a = rng.randint(0, total + 2); b = a + rng.randint(0, cap + 2)
        s0, s1 = max(a, oldest), min(b, total)
        exp = None if s1 <= s0 else full[s0:s1]
        got = rb.slice(float(a), float(b))
        if exp is None:
            assert got is None
        else:
            assert got is not None and np.array_equal(got, exp)
