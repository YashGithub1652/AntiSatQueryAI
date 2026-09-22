import os
from pathlib import Path


def test_model_policy_defaults_are_safe():
    from satquery.backend.models import model_policy

    assert model_policy.ALLOW_UNTRAINED_CHANGE_FALLBACK is False
    assert model_policy.ALLOW_HEURISTIC_GROUNDING is False
    assert model_policy.ALLOW_HEURISTIC_VQA is False


def test_adapter_requires_weight_file(tmp_path, monkeypatch):
    from satquery.backend.models import model_policy

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    assert model_policy.adapter_is_complete(adapter) is False

    (adapter / "adapter_model.safetensors").write_bytes(b"placeholder")
    assert model_policy.adapter_is_complete(adapter) is True


def test_grounding_does_not_fabricate_empty_regions():
    from satquery.backend.models.grounding_engine import GroundingEngine

    # Exercise the pure fallback helper with a blank image. The helper must
    # return no fabricated quadrants when no candidate survives filtering.
    import numpy as np
    from PIL import Image

    engine = object.__new__(GroundingEngine)
    image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    boxes, _ = engine._dynamic_spectral_spatial_grounding(
        image, "water bodies", 0.5
    )
    assert boxes == []
