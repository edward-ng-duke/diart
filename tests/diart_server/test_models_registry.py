"""Tests for diart_server.models_registry.

These tests mock ``SegmentationModel.from_pyannote``/``from_onnx`` (and
the embedding equivalents) so they never hit the network or the
filesystem. The real manifest at ``docker/models.json`` is used as
source of truth for spec lookups.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from diart_server.errors import ModelNotFoundError
from diart_server.models_registry import ModelSpec, ModelsRegistry


MANIFEST_PATH = Path(__file__).resolve().parents[2] / "docker" / "models.json"
DUMMY_DIR = Path("/tmp/diart_models_registry_test")


def _make_registry() -> ModelsRegistry:
    return ModelsRegistry(DUMMY_DIR, MANIFEST_PATH)


def test_manifest_loads_expected_entries():
    reg = _make_registry()
    # Sanity check: manifest contains the names referenced by other
    # tests below. Guards against silent manifest drift.
    assert "pyannote/segmentation-3.0" in reg.specs
    assert "pyannote/embedding" in reg.specs
    assert "hbredin/wespeaker-voxceleb-resnet34-LM" in reg.specs
    spec = reg.specs["hbredin/wespeaker-voxceleb-resnet34-LM"]
    assert isinstance(spec, ModelSpec)
    assert spec.backend == "onnx"


def test_get_segmentation_unknown_raises_not_found():
    reg = _make_registry()
    with pytest.raises(ModelNotFoundError):
        reg.get_segmentation("does/not-exist")


def test_get_embedding_unknown_raises_not_found():
    reg = _make_registry()
    with pytest.raises(ModelNotFoundError):
        reg.get_embedding("does/not-exist")


def test_segmentation_caching_returns_same_instance():
    reg = _make_registry()
    fake_model = MagicMock(name="seg-model")
    with patch(
        "diart_server.models_registry.SegmentationModel.from_pyannote",
        return_value=fake_model,
    ) as loader:
        first = reg.get_segmentation("pyannote/segmentation-3.0")
        second = reg.get_segmentation("pyannote/segmentation-3.0")
    assert first is second is fake_model
    assert loader.call_count == 1


def test_embedding_caching_returns_same_instance():
    reg = _make_registry()
    fake_model = MagicMock(name="emb-model")
    with patch(
        "diart_server.models_registry.EmbeddingModel.from_pyannote",
        return_value=fake_model,
    ) as loader:
        first = reg.get_embedding("pyannote/embedding")
        second = reg.get_embedding("pyannote/embedding")
    assert first is second is fake_model
    assert loader.call_count == 1


def test_pyannote_segmentation_dispatch_passes_local_path():
    reg = _make_registry()
    fake_model = MagicMock(name="seg-model")
    with patch(
        "diart_server.models_registry.SegmentationModel.from_pyannote",
        return_value=fake_model,
    ) as loader:
        out = reg.get_segmentation("pyannote/segmentation")
    assert out is fake_model
    loader.assert_called_once()
    args, kwargs = loader.call_args
    expected_path = str(DUMMY_DIR / "pyannote/segmentation")
    assert args[0] == expected_path
    assert kwargs.get("use_hf_token") is False


def test_onnx_embedding_dispatch_uses_from_onnx():
    reg = _make_registry()
    fake_model = MagicMock(name="onnx-emb")
    # Ensure from_pyannote is NOT used for an onnx-backend spec.
    with patch(
        "diart_server.models_registry.EmbeddingModel.from_onnx",
        return_value=fake_model,
    ) as onnx_loader, patch(
        "diart_server.models_registry.EmbeddingModel.from_pyannote"
    ) as pyannote_loader:
        out = reg.get_embedding("hbredin/wespeaker-voxceleb-resnet34-LM")
    assert out is fake_model
    onnx_loader.assert_called_once()
    pyannote_loader.assert_not_called()
    expected_path = str(DUMMY_DIR / "onnx/speaker-embedding.onnx")
    assert onnx_loader.call_args.args[0] == expected_path


def test_speechbrain_backend_falls_through_to_pyannote_loader():
    """Non-onnx backends (speechbrain/nemo) dispatch via from_pyannote.

    The pyannote loader internally falls back to PretrainedSpeakerEmbedding
    for non-pyannote checkpoints (see diart.models.PyannoteLoader).
    """
    reg = _make_registry()
    fake_model = MagicMock(name="sb-emb")
    with patch(
        "diart_server.models_registry.EmbeddingModel.from_pyannote",
        return_value=fake_model,
    ) as loader, patch(
        "diart_server.models_registry.EmbeddingModel.from_onnx"
    ) as onnx_loader:
        out = reg.get_embedding("speechbrain/spkrec-ecapa-voxceleb")
    assert out is fake_model
    loader.assert_called_once()
    onnx_loader.assert_not_called()


def test_kind_mismatch_raises_not_found():
    """A segmentation spec is not retrievable via get_embedding (and vice versa)."""
    reg = _make_registry()
    with pytest.raises(ModelNotFoundError):
        reg.get_embedding("pyannote/segmentation-3.0")
    with pytest.raises(ModelNotFoundError):
        reg.get_segmentation("pyannote/embedding")


def test_warm_up_loads_defaults_on_device():
    reg = _make_registry()
    fake_seg = MagicMock(name="seg")
    fake_seg.to.return_value = fake_seg
    fake_seg.eval.return_value = fake_seg
    fake_emb = MagicMock(name="emb")
    fake_emb.to.return_value = fake_emb
    fake_emb.eval.return_value = fake_emb
    device = __import__("torch").device("cpu")
    with patch(
        "diart_server.models_registry.SegmentationModel.from_pyannote",
        return_value=fake_seg,
    ) as seg_loader, patch(
        "diart_server.models_registry.EmbeddingModel.from_pyannote",
        return_value=fake_emb,
    ) as emb_loader:
        reg.warm_up(device, "pyannote/segmentation-3.0", "pyannote/embedding")
    seg_loader.assert_called_once()
    emb_loader.assert_called_once()
    fake_seg.to.assert_called_once_with(device)
    fake_seg.eval.assert_called_once()
    fake_emb.to.assert_called_once_with(device)
    fake_emb.eval.assert_called_once()


def test_wespeaker_onnx_entry_carries_match_threshold():
    """The cnceleb entry declares a per-model threshold (its centered-score
    scale sits higher than the voxceleb models'); entries without the field
    default to None -> settings.match_threshold applies."""
    reg = _make_registry()
    spec = reg.specs["wespeaker/cnceleb-resnet34-LM"]
    assert spec.backend == "wespeaker_onnx"
    assert spec.match_threshold == 0.85
    assert reg.specs["pyannote/embedding"].match_threshold is None


def test_wespeaker_onnx_dispatch_builds_lazy_wrapper():
    """wespeaker_onnx wraps the pyannote fbank-ONNX wrapper in a LAZY
    EmbeddingModel: construction must not instantiate the wrapper (no ORT
    session at registry time); first load() must pass the resolved path."""
    reg = _make_registry()
    fake_cls = MagicMock(name="wespeaker-wrapper-cls")
    with patch(
        "pyannote.audio.pipelines.speaker_verification"
        ".ONNXWeSpeakerPretrainedSpeakerEmbedding",
        fake_cls,
    ), patch(
        "diart_server.models_registry.EmbeddingModel.from_onnx"
    ) as onnx_loader, patch(
        "diart_server.models_registry.EmbeddingModel.from_pyannote"
    ) as pyannote_loader:
        model = reg.get_embedding("wespeaker/cnceleb-resnet34-LM")
        fake_cls.assert_not_called()  # lazy: nothing loaded yet
        model.load()
    fake_cls.assert_called_once_with(
        str(DUMMY_DIR / "wespeaker/cnceleb-resnet34-LM/speaker-embedding.onnx")
    )
    onnx_loader.assert_not_called()
    pyannote_loader.assert_not_called()
