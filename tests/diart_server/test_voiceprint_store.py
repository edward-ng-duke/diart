import numpy as np
from diart_server.voiceprint_store import VoiceprintStore


def test_enroll_mean_and_namespace(tmp_path):
    s = VoiceprintStore(tmp_path)
    s.enroll("ecapa", "张三", np.array([2.0, 0.0]))
    s.enroll("ecapa", "张三", np.array([0.0, 0.0]))  # mean -> [1,0] before norm
    s.enroll("wespeaker", "张三", np.array([0.0, 1.0]))
    v = s.vectors("ecapa")["张三"]
    np.testing.assert_allclose(v, np.array([1.0, 0.0]), atol=1e-6)  # normalized
    assert "张三" not in s.vectors("wespeaker") or \
        not np.allclose(s.vectors("wespeaker")["张三"], v)  # different namespace
    assert s.list("ecapa") == [{"name": "张三", "n_clips": 2}]


def test_persistence_roundtrip(tmp_path):
    s = VoiceprintStore(tmp_path)
    s.enroll("ecapa", "李四", np.array([1.0, 2.0, 3.0]))
    s.save()
    s2 = VoiceprintStore(tmp_path)
    s2.load()
    assert s2.list("ecapa") == [{"name": "李四", "n_clips": 1}]
    assert s2.vectors("ecapa")["李四"].shape == (3,)


def test_remove(tmp_path):
    s = VoiceprintStore(tmp_path)
    s.enroll("ecapa", "王五", np.array([1.0, 0.0]))
    s.remove("ecapa", "王五")
    assert s.list("ecapa") == []
