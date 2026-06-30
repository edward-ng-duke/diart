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
