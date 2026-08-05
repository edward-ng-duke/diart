import numpy as np
from diart_server.voiceprint_matcher import match


def test_matches_closest_above_threshold():
    refs = {"张三": np.array([1.0, 0.0]), "李四": np.array([0.0, 1.0])}
    name, score = match(np.array([0.9, 0.1]), refs, threshold=0.5)
    assert name == "张三" and score > 0.5


def test_below_threshold_is_unknown():
    refs = {"张三": np.array([1.0, 0.0])}
    name, score = match(np.array([0.0, 1.0]), refs, threshold=0.5)
    assert name is None and score < 0.5


def test_empty_refs_or_vec_is_unknown():
    assert match(np.array([1.0, 0.0]), {}, 0.5) == (None, 0.0)
    assert match(np.array([0.0, 0.0]), {"a": np.array([1.0, 0.0])}, 0.5)[0] is None


def test_mean_none_preserves_raw_behaviour():
    # Passing mean_embedding=None must be identical to the historical raw path.
    refs = {"张三": np.array([1.0, 0.0]), "李四": np.array([0.0, 1.0])}
    q = np.array([0.9, 0.1])
    assert match(q, refs, threshold=0.5) == match(q, refs, threshold=0.5, mean_embedding=None)


def test_centering_widens_same_vs_different_separation():
    """With a shared common component, mean-centering must separate better.

    Construct three unit-ish vectors that all share a large common direction
    plus a small per-speaker delta. Speaker A has two utterances (a1, a2) that
    share an extra delta; speaker B points a different small delta. Raw cosine
    is dominated by the common component (everything looks similar), while
    mean-centering against the cohort mean removes it and recovers the
    discriminative deltas.
    """
    dim = 16
    common = np.zeros(dim, dtype=np.float32)
    common[0] = 1.0  # large shared, non-discriminative component

    dA = np.zeros(dim, dtype=np.float32); dA[1] = 1.0
    dB = np.zeros(dim, dtype=np.float32); dB[2] = 1.0

    big = 4.0      # common dominates -> compressed raw cosines
    small = 1.0

    # Two same-speaker utterances (slightly different deltas within A) and one
    # different speaker B, all riding the same common component.
    a1 = big * common + small * dA
    a2 = big * common + small * (0.9 * dA + 0.1 * dB)
    b = big * common + small * dB

    # Impostor cohort mean ~ the shared common direction (no per-speaker info).
    cohort = common.copy()

    def _cos(u, v):  # raw cosine helper
        u = u / np.linalg.norm(u); v = v / np.linalg.norm(v)
        return float(np.dot(u, v))

    # --- RAW (mean_embedding=None) separation ---
    # Use a very low threshold so match() always returns the name+score.
    _, raw_same = match(a2, {"A": a1}, threshold=-1.0)
    _, raw_diff = match(b, {"A": a1}, threshold=-1.0)
    raw_gap = raw_same - raw_diff

    # --- CENTERED separation (remapped to [0,1]) ---
    _, c_same = match(a2, {"A": a1}, threshold=-1.0, mean_embedding=cohort)
    _, c_diff = match(b, {"A": a1}, threshold=-1.0, mean_embedding=cohort)
    centered_gap = c_same - c_diff

    # Centering must produce a strictly larger same-vs-different gap.
    assert centered_gap > raw_gap
    # And it must remain usable: same clearly above different on the [0,1] scale.
    assert c_same > c_diff
    assert 0.0 <= c_same <= 1.0 and 0.0 <= c_diff <= 1.0


def test_centering_threshold_separates_same_from_different():
    """A single threshold on the remapped score accepts same, rejects different."""
    dim = 16
    common = np.zeros(dim, dtype=np.float32); common[0] = 1.0
    dA = np.zeros(dim, dtype=np.float32); dA[1] = 1.0
    dB = np.zeros(dim, dtype=np.float32); dB[2] = 1.0

    a1 = 4.0 * common + dA
    a2 = 4.0 * common + (0.9 * dA + 0.1 * dB)
    b = 4.0 * common + dB
    cohort = common.copy()

    _, c_same = match(a2, {"A": a1}, threshold=-1.0, mean_embedding=cohort)
    _, c_diff = match(b, {"A": a1}, threshold=-1.0, mean_embedding=cohort)
    thr = (c_same + c_diff) / 2.0

    name_same, _ = match(a2, {"A": a1}, threshold=thr, mean_embedding=cohort)
    name_diff, _ = match(b, {"A": a1}, threshold=thr, mean_embedding=cohort)
    assert name_same == "A"
    assert name_diff is None


# ---------------------------------------------------------------------------
# Cohort shape validation (2026-08-05 cohort-missing-threshold fix §4.5): a
# mean whose shape does not match the query is discarded UP FRONT, so the call
# runs the exact historical raw path — no centering, no (c+1)/2 remap. The
# remap of a raw cosine inflated impostors onto the centered scale (§2.2).
# ---------------------------------------------------------------------------


def test_wrong_shape_mean_equals_raw_mode_exactly():
    """Wrong-shape mean == mean_embedding=None, float-for-float."""
    rng = np.random.default_rng(20260805)
    dim = 32
    vec = rng.standard_normal(dim).astype(np.float32)
    refs = {
        "甲": rng.standard_normal(dim).astype(np.float32),
        "乙": rng.standard_normal(dim).astype(np.float32),
        "丙": rng.standard_normal(dim).astype(np.float32),
    }
    wrong = np.zeros(dim + 7, dtype=np.float32)
    for thr in (0.0, 0.82):
        assert match(vec, refs, thr, mean_embedding=wrong) == match(
            vec, refs, thr, mean_embedding=None
        )


def test_correct_shape_mean_matches_manual_centered_remap():
    """With a correct-shape mean every candidate score is the manual
    (cos(unit(unit(v)-m), unit(unit(r)-m)) + 1) / 2 in [0, 1], and the
    returned score equals the best manual score within 1e-6."""
    rng = np.random.default_rng(7)
    dim = 24
    vec = rng.standard_normal(dim).astype(np.float32)
    refs = {f"spk{i}": rng.standard_normal(dim).astype(np.float32) for i in range(4)}
    mean = rng.standard_normal(dim).astype(np.float32)
    mean /= np.linalg.norm(mean)

    def _u(v):
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        return v / float(np.linalg.norm(v))

    def _manual(v, r):
        c = float(np.dot(_u(_u(v) - mean), _u(_u(r) - mean)))
        return (c + 1.0) / 2.0

    manual = {name: _manual(vec, r) for name, r in refs.items()}
    assert all(0.0 <= s <= 1.0 for s in manual.values()), manual
    name, score = match(vec, refs, threshold=-1.0, mean_embedding=mean)
    best = max(manual, key=manual.get)
    assert name == best
    assert abs(score - manual[best]) <= 1e-6
    assert 0.0 <= score <= 1.0


def test_wrong_shape_mean_impostor_at_raw_066_is_rejected():
    """Regression pin (spec §3 fact 3): pre-fix, a wrong-shape mean remapped
    the RAW cosine — an impostor at raw 0.66 scored (0.66+1)/2 = 0.83 >= 0.82
    (false accept). Post-fix the call is pure raw mode: 0.66 < 0.82 -> None."""
    q = np.array([1.0, 0.0], dtype=np.float32)
    r = np.array([0.66, float(np.sqrt(1.0 - 0.66 ** 2))], dtype=np.float32)
    wrong = np.zeros(9, dtype=np.float32)
    name, score = match(q, {"impostor": r}, threshold=0.82, mean_embedding=wrong)
    assert name is None
    assert abs(score - 0.66) <= 1e-6
