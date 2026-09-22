"""
SatQuery AI — Model Readiness & Safety Policy
==============================================
Central policy for deciding whether an inference path is scientifically
valid for SIH/ISRO-facing results.

The application may have demo fallbacks, but a fallback must never be
silently presented as a trained remote-sensing specialist model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any


REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Safety defaults: untrained/random/demo fallbacks are OFF.
ALLOW_UNTRAINED_CHANGE_FALLBACK = env_flag(
    "SATQUERY_ALLOW_UNTRAINED_CHANGE_FALLBACK", False
)
ALLOW_HEURISTIC_GROUNDING = env_flag(
    "SATQUERY_ALLOW_HEURISTIC_GROUNDING", False
)
ALLOW_BASE_CLIP_FALLBACK = env_flag(
    "SATQUERY_ALLOW_BASE_CLIP_FALLBACK", False
)


def checkpoint_exists(*relative_parts: str) -> bool:
    return (MODELS_DIR.joinpath(*relative_parts)).is_file()


def adapter_is_complete(adapter_dir: Path) -> bool:
    """PEFT adapters require both configuration and actual adapter weights."""
    if not adapter_dir.is_dir():
        return False
    has_config = (adapter_dir / "adapter_config.json").is_file()
    has_weights = any(
        (adapter_dir / name).is_file()
        for name in (
            "adapter_model.safetensors",
            "adapter_model.bin",
            "pytorch_model.bin",
        )
    )
    return has_config and has_weights


def readiness_snapshot() -> Dict[str, Any]:
    geochat_dir = MODELS_DIR / "geochat_lora_bigearthnet"
    return {
        "geochat_lora": {
            "ready": adapter_is_complete(geochat_dir),
            "path": str(geochat_dir),
        },
        "remote_clip": {
            "ready": checkpoint_exists("RemoteCLIP-ViT-B-32.pt"),
            "path": str(MODELS_DIR / "RemoteCLIP-ViT-B-32.pt"),
        },
        "changeformer": {
            "ready": checkpoint_exists("ChangeFormer_LEVIR.pth")
            or checkpoint_exists("checkpoints", "ChangeFormer_LEVIR", "best_ckpt.pt"),
            "path": str(CHECKPOINT_DIR),
        },
        "rsvg": {
            "ready": checkpoint_exists("rsvg_best.pth"),
            "path": str(MODELS_DIR / "rsvg_best.pth"),
        },
        "sam": {
            "ready": checkpoint_exists("sam_vit_b_01ec64.pth"),
            "path": str(MODELS_DIR / "sam_vit_b_01ec64.pth"),
        },
        "sar_optical_fusion": {
            "ready": checkpoint_exists(
                "checkpoints", "sar_optical_cross_attention_best.pth"
            ),
            "path": str(
                CHECKPOINT_DIR / "sar_optical_cross_attention_best.pth"
            ),
        },
    }
