"""
SatQuery AI — Centralized Model Loader
=========================================
Loads ALL ML models ONCE at server startup and caches them in memory.
Prevents redundant loading latency during inference.

Models loaded:
  1. GeoChat-7B (4-bit NF4 quantized) — Primary VLM
  2. RemoteCLIP ViT-B/32 — Visual encoder + confidence scoring
  3. ChangeFormer — Bi-temporal change detection
  4. RSVG — Visual grounding / referring expression comprehension
  5. SAM ViT-Base — Pixel segmentation masks

Design principle: All models are loaded lazily on first use if
startup loading fails, to support CPU-only / low-memory environments.
"""

import os
import logging
import time
from typing import Optional, Dict, Any, Tuple

import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_4BIT = torch.cuda.is_available()  # 4-bit quantization only on GPU
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    DEVICE = "cpu"
    USE_4BIT = False

logger = logging.getLogger(__name__)
logger.info(f"ModelLoader: device={DEVICE}, 4-bit_quant={USE_4BIT}, torch_available={TORCH_AVAILABLE}")


class ModelLoader:
    """
    Singleton model registry. All engines share this instance.
    Usage: loader = get_model_loader()
           geochat_model, geochat_processor = loader.get_geochat()
    """

    def __init__(self):
        self._geochat_model = None
        self._geochat_processor = None
        self._remote_clip_model = None
        self._remote_clip_preprocess = None
        self._changeformer_model = None
        self._rsvg_model = None
        self._rsvg_tokenizer = None
        self._sam_predictor = None
        self._load_status: Dict[str, str] = {}

    # ──────────────────────────────────────────────────────────
    # GEOCHAT — Primary VLM
    # ──────────────────────────────────────────────────────────

    def get_geochat(self) -> Tuple[Any, Any]:
        """
        Returns (model, processor) for GeoChat-7B with multi-fallback (GeoChat -> BLIP-2).
        """
        if self._geochat_model is not None:
            return self._geochat_model, self._geochat_processor

        t0 = time.time()

        # 1. Try GeoChat (GPU only recommended due to 7B size & 4-bit config)
        if DEVICE == "cuda":
            model_id = "MBZUAI/GeoChat"
            lora_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "models", "geochat_lora_bigearthnet"
            )

            try:
                from transformers import (
                    AutoModelForCausalLM,
                    AutoProcessor,
                    BitsAndBytesConfig,
                )
                from peft import PeftModel

                logger.info(f"Loading GeoChat from {model_id} ...")

                quant_config = None
                if USE_4BIT:
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )

                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quant_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )

                if os.path.isdir(lora_path) and os.path.exists(
                    os.path.join(lora_path, "adapter_config.json")
                ):
                    logger.info(f"Loading BigEarthNet LoRA adapter from {lora_path}")
                    model = PeftModel.from_pretrained(model, lora_path)
                    self._load_status["geochat"] = "GeoChat-7B + BigEarthNet-LoRA (4-bit NF4)"
                else:
                    logger.warning(
                        "LoRA adapter not found. Using base GeoChat-7B."
                    )
                    self._load_status["geochat"] = "GeoChat-7B base (no LoRA)"

                processor = AutoProcessor.from_pretrained(
                    model_id, trust_remote_code=True
                )

                model.eval()
                self._geochat_model = model
                self._geochat_processor = processor
                logger.info(f"GeoChat loaded in {time.time() - t0:.1f}s on {DEVICE}")
                return self._geochat_model, self._geochat_processor

            except Exception as e:
                logger.warning(f"GeoChat loading failed on GPU: {e}. Trying fallback VLM.")

        # 2. Fallback to BLIP-2 (CPU-compatible real VLM, ~5.4GB)
        try:
            blip2_id = "Salesforce/blip2-opt-2.7b"
            logger.info(f"Loading fallback VLM (BLIP-2) from {blip2_id} ...")
            from transformers import Blip2Processor, Blip2ForConditionalGeneration

            processor = Blip2Processor.from_pretrained(blip2_id)
            model = Blip2ForConditionalGeneration.from_pretrained(
                blip2_id,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto" if DEVICE == "cuda" else None,
            )
            if DEVICE == "cpu":
                model = model.to("cpu")
            model.eval()
            self._geochat_model = model
            self._geochat_processor = processor
            self._load_status["geochat"] = f"BLIP-2 OPT-2.7B real VLM (CPU fallback)"
            logger.info(f"BLIP-2 loaded in {time.time() - t0:.1f}s")
            return self._geochat_model, self._geochat_processor

        except Exception as e2:
            logger.warning(f"BLIP-2 load failed: {e2}. Trying BLIP-1 VQA-Base ...")

        # 3. Smallest real VLM fallback: BLIP-1 VQA-Base (~200MB)
        try:
            from transformers import BlipProcessor, BlipForQuestionAnswering
            blip1_id = "Salesforce/blip-vqa-base"
            logger.info(f"Loading BLIP-1 VQA-Base ({blip1_id}) ...")
            processor = BlipProcessor.from_pretrained(blip1_id)
            model = BlipForQuestionAnswering.from_pretrained(blip1_id)
            model = model.to(DEVICE)
            model.eval()
            self._geochat_model = model
            self._geochat_processor = processor
            self._load_status["geochat"] = f"BLIP-1 VQA-Base — lightweight real VLM"
            logger.info(f"BLIP-1 VQA-Base loaded in {time.time() - t0:.1f}s")
            return self._geochat_model, self._geochat_processor

        except Exception as e3:
            self._load_status["geochat"] = f"FAILED: {e3}"
            raise RuntimeError(
                f"No VLM could be loaded: {e3}\n"
                "Install: pip install transformers accelerate"
            )


    # ──────────────────────────────────────────────────────────
    # REMOTECLIP — Visual Encoder
    # ──────────────────────────────────────────────────────────

    def get_remote_clip(self) -> Tuple[Any, Any]:
        """Returns (model, preprocess) for RemoteCLIP ViT-B/32."""
        if self._remote_clip_model is not None:
            return self._remote_clip_model, self._remote_clip_preprocess

        try:
            import open_clip

            t0 = time.time()
            logger.info("Loading RemoteCLIP ViT-B-32 ...")

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="openai",  # base weights, then load RS checkpoint
            )

            # Load RemoteCLIP RS-adapted checkpoint
            checkpoint_paths = [
                os.path.join(os.path.dirname(__file__), "..", "..", "models", "RemoteCLIP-ViT-B-32.pt"),
                os.path.expanduser("~/.cache/satquery/RemoteCLIP-ViT-B-32.pt"),
            ]
            checkpoint = None
            for cp in checkpoint_paths:
                if os.path.exists(cp):
                    checkpoint = cp
                    break

            if checkpoint:
                state_dict = torch.load(checkpoint, map_location=DEVICE)
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                model.load_state_dict(state_dict, strict=False)
                self._load_status["remote_clip"] = f"RemoteCLIP-ViT-B-32 (RS-adapted, loaded from {checkpoint})"
                logger.info(f"RemoteCLIP RS checkpoint loaded from {checkpoint}")
            else:
                logger.warning(
                    "RemoteCLIP checkpoint not found. Using base OpenAI CLIP. "
                    "Download from: https://huggingface.co/flywire/RemoteCLIP"
                )
                self._load_status["remote_clip"] = "CLIP-ViT-B-32 base (RemoteCLIP checkpoint missing)"

            model = model.to(DEVICE)
            model.eval()
            tokenizer = open_clip.get_tokenizer("ViT-B-32")

            self._remote_clip_model = (model, tokenizer)
            self._remote_clip_preprocess = preprocess
            logger.info(f"RemoteCLIP loaded in {time.time() - t0:.1f}s")

        except Exception as e:
            logger.error(f"Failed to load RemoteCLIP: {e}")
            self._load_status["remote_clip"] = f"FAILED: {e}"
            raise RuntimeError(
                f"RemoteCLIP could not be loaded: {e}\n"
                "Install: pip install open_clip_torch"
            )

        return self._remote_clip_model, self._remote_clip_preprocess

    # ──────────────────────────────────────────────────────────
    # CHANGEFORMER — Bi-temporal Change Detection
    # ──────────────────────────────────────────────────────────

    def get_changeformer(self) -> Any:
        """Returns ChangeFormer model."""
        if self._changeformer_model is not None:
            return self._changeformer_model

        try:
            t0 = time.time()
            logger.info("Loading ChangeFormer ...")

            # Try to import from local clone or installed package
            try:
                from models.ChangeFormer import ChangeFormer
            except ImportError as e:
                from .model_policy import ALLOW_UNTRAINED_CHANGE_FALLBACK
                message = (
                    "Official ChangeFormer implementation is not installed. "
                    "Install the official ChangeFormer implementation and provide "
                    "a trained checkpoint before enabling scientific change inference."
                )
                if not ALLOW_UNTRAINED_CHANGE_FALLBACK:
                    self._load_status["changeformer"] = "UNAVAILABLE: official implementation missing"
                    raise RuntimeError(message) from e
                logger.warning(
                    "%s SATQUERY_ALLOW_UNTRAINED_CHANGE_FALLBACK is enabled; "
                    "using the untrained demo fallback.", message
                )
                from .changeformer_fallback import SiameseChangeDetector
                model = SiameseChangeDetector().to(DEVICE)
                model.eval()
                self._changeformer_model = model
                self._load_status["changeformer"] = "UNTRAINED_DEMO: SiameseChangeDetector-ResNet18"
                logger.info(f"Siamese change detector loaded in {time.time() - t0:.1f}s")
                return self._changeformer_model

            # Official ChangeFormer
            model = ChangeFormer()
            checkpoint_paths = [
                os.path.join(os.path.dirname(__file__), "..", "..", "models", "ChangeFormer_LEVIR.pth"),
                os.path.expanduser("~/.cache/satquery/ChangeFormer_LEVIR.pth"),
            ]
            for cp in checkpoint_paths:
                if os.path.exists(cp):
                    state_dict = torch.load(cp, map_location=DEVICE)
                    model.load_state_dict(state_dict, strict=False)
                    self._load_status["changeformer"] = f"ChangeFormer-V2 LEVIR-CD checkpoint"
                    break
            else:
                from .model_policy import ALLOW_UNTRAINED_CHANGE_FALLBACK
                if not ALLOW_UNTRAINED_CHANGE_FALLBACK:
                    self._load_status["changeformer"] = "UNAVAILABLE: trained checkpoint missing"
                    raise FileNotFoundError(
                        "ChangeFormer implementation is available, but no trained checkpoint "
                        "was found. Add models/ChangeFormer_LEVIR.pth or enable "
                        "SATQUERY_ALLOW_UNTRAINED_CHANGE_FALLBACK only for development demos."
                    )
                logger.warning(
                    "No trained ChangeFormer checkpoint found; using untrained weights "
                    "because SATQUERY_ALLOW_UNTRAINED_CHANGE_FALLBACK is enabled."
                )
                self._load_status["changeformer"] = "UNTRAINED_DEMO: ChangeFormer random weights"

            model = model.to(DEVICE)
            model.eval()
            self._changeformer_model = model
            logger.info(f"ChangeFormer loaded in {time.time() - t0:.1f}s")

        except Exception as e:
            logger.error(f"ChangeFormer load error: {e}")
            self._load_status["changeformer"] = f"FAILED: {e}"
            raise RuntimeError(f"ChangeFormer could not be loaded: {e}")

        return self._changeformer_model

    # ──────────────────────────────────────────────────────────
    # RSVG — Visual Grounding
    # ──────────────────────────────────────────────────────────

    def get_rsvg(self) -> Tuple[Any, Any]:
        """Returns (rsvg_model, tokenizer) for visual grounding."""
        if self._rsvg_model is not None:
            return self._rsvg_model, self._rsvg_tokenizer

        try:
            t0 = time.time()
            logger.info("Loading RSVG grounding model ...")

            # Try official RSVG
            try:
                from models.RSVG import build_model as build_rsvg
                from transformers import BertTokenizer

                model = build_rsvg()
                checkpoint_paths = [
                    os.path.join(os.path.dirname(__file__), "..", "..", "models", "rsvg_best.pth"),
                    os.path.expanduser("~/.cache/satquery/rsvg_best.pth"),
                ]
                for cp in checkpoint_paths:
                    if os.path.exists(cp):
                        state_dict = torch.load(cp, map_location=DEVICE)
                        model.load_state_dict(state_dict.get("model", state_dict), strict=False)
                        self._load_status["rsvg"] = "RSVG-Swin-Transformer (VRSBench checkpoint)"
                        break
                else:
                    self._load_status["rsvg"] = "UNAVAILABLE: RSVG checkpoint missing"
                    raise FileNotFoundError(
                        "RSVG implementation is available but rsvg_best.pth is missing."
                    )

                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

            except ImportError:
                # Fallback: Grounding DINO (open vocabulary detection)
                logger.warning(
                    "RSVG not found. Using GroundingDINO fallback. "
                    "Clone: https://github.com/ZhanYang-nwpu/RSVG-pytorch"
                )
                from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

                model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    "IDEA-Research/grounding-dino-base"
                ).to(DEVICE)
                tokenizer = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
                self._load_status["rsvg"] = "GroundingDINO-Base (open-vocab fallback)"

            model.to(DEVICE).eval()
            self._rsvg_model = model
            self._rsvg_tokenizer = tokenizer
            logger.info(f"RSVG/Grounding loaded in {time.time() - t0:.1f}s")

        except Exception as e:
            logger.error(f"RSVG load error: {e}")
            self._load_status["rsvg"] = f"FAILED: {e}"
            raise RuntimeError(f"RSVG could not be loaded: {e}")

        return self._rsvg_model, self._rsvg_tokenizer

    # ──────────────────────────────────────────────────────────
    # SAM — Segment Anything Model
    # ──────────────────────────────────────────────────────────

    def get_sam(self) -> Any:
        """Returns SAM predictor for pixel-level mask generation."""
        if self._sam_predictor is not None:
            return self._sam_predictor

        try:
            from segment_anything import sam_model_registry, SamPredictor

            t0 = time.time()
            checkpoint_paths = [
                os.path.join(os.path.dirname(__file__), "..", "..", "models", "sam_vit_b_01ec64.pth"),
                os.path.expanduser("~/.cache/satquery/sam_vit_b_01ec64.pth"),
            ]
            checkpoint = None
            for cp in checkpoint_paths:
                if os.path.exists(cp):
                    checkpoint = cp
                    break

            if checkpoint is None:
                raise FileNotFoundError(
                    "SAM checkpoint not found. Download from: "
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
                )

            sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
            sam = sam.to(DEVICE)
            predictor = SamPredictor(sam)
            self._sam_predictor = predictor
            self._load_status["sam"] = "SAM-ViT-Base"
            logger.info(f"SAM loaded in {time.time() - t0:.1f}s")

        except Exception as e:
            logger.error(f"SAM load error: {e}")
            self._load_status["sam"] = f"FAILED: {e}"
            # SAM is optional — grounding can work without pixel masks
            self._sam_predictor = None

        return self._sam_predictor

    # ──────────────────────────────────────────────────────────
    # STATUS / DIAGNOSTICS
    # ──────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Return loading status for all models — used in /api/v1/models endpoint."""
        return {
            "device": DEVICE,
            "cuda_available": (TORCH_AVAILABLE and torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if (TORCH_AVAILABLE and torch.cuda.is_available()) else "N/A",
            "vram_gb": (
                round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                if (TORCH_AVAILABLE and torch.cuda.is_available())
                else 0
            ),
            "models_loaded": {
                "geochat": self._geochat_model is not None,
                "remote_clip": self._remote_clip_model is not None,
                "changeformer": self._changeformer_model is not None,
                "rsvg": self._rsvg_model is not None,
                "sam": self._sam_predictor is not None,
            },
            "load_status": self._load_status,
        }


# ──────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ──────────────────────────────────────────────────────────────
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get the global model loader instance. Thread-safe via Python GIL."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader
