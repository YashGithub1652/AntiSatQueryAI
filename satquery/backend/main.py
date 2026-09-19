"""
SatQuery AI — FastAPI Gateway
==============================
Complete API with all real endpoints replacing the demo simulation.
Handles: image upload, query submission, WebSocket streaming, model registry, export.

Endpoints:
  POST /api/v1/upload/image            → Upload single GeoTIFF → session
  POST /api/v1/upload/pair             → Upload image pair → session
  GET  /api/v1/upload/{session}/meta   → Real rasterio metadata
  POST /api/v1/query/submit            → Submit query → run agent → result
  GET  /api/v1/query/{task_id}/status  → Task status
  WS   /ws/{session_id}                → Live agent trace stream
  GET  /api/v1/models                  → model_registry.yaml contents
  POST /api/v1/evaluate/iou            → IoU vs reference mask
  POST /api/v1/export/report           → Generate PDF
  GET  /api/v1/health                  → Health check
"""

import os
import uuid
import asyncio
import logging
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor

import yaml
import numpy as np
from fastapi import (
    FastAPI, File, UploadFile, HTTPException, WebSocket,
    WebSocketDisconnect, BackgroundTasks, Form, Request, Response
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_unified_model_registry() -> dict:
    """
    Load configs/model_registry.yaml as the single source of truth.
    Provides backward-compatible normalized aliases for endpoints and tests.
    """
    yaml_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "configs", "model_registry.yaml"
    )
    raw_data = {}
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load model_registry.yaml: {e}")

    models = dict(raw_data.get("models", {}))

    # Normalized aliases ensuring compatibility with test_api.py and legacy callers
    rs_vlm = dict(models.get("rs_vlm", {}))
    models["vqa"] = {
        "name": rs_vlm.get("display_name", "GeoChat-7B (4-bit QLoRA)"),
        "backbone": rs_vlm.get("base_model", "LLaVA-1.5 architecture / Vicuna-7B"),
        "training_dataset": "BigEarthNet.txt (590,000 RS patches) + RSVQA-HR",
        "resolution": "512x512 multi-spectral",
        "task": "Single Image VQA & Complex Remote Sensing QA",
        "status": "active",
        **rs_vlm
    }

    sar_fus = dict(models.get("sar_optical_fusion", {}))
    models["sar_fusion"] = {
        "name": sar_fus.get("display_name", "CrossModal-S1S2-ResNet50"),
        "backbone": "2-channel SAR ResNet-50 + RemoteCLIP + 8-head Cross-Attention",
        "training_dataset": "SEN12MS (Sentinel-1 SAR + Sentinel-2 MSI)",
        "task": "Cloud-penetrating SAR-optical multimodal fusion",
        "status": "active",
        **sar_fus
    }

    grounding = dict(models.get("visual_grounding", {}))
    models["grounding"] = {
        "name": grounding.get("display_name", "RSVG-DINO + Segment Anything (SAM)"),
        "backbone": "GroundingDINO + ViT-H SAM",
        "training_dataset": "RSVG Benchmark + DIOR-RSVG",
        "task": "Text-prompted spatial bounding box grounding & polygon segmentation",
        "status": "active",
        **grounding
    }

    cd = dict(models.get("change_detection", {}))
    models["change_detection"] = {
        "name": cd.get("display_name", "ChangeFormerV2"),
        "backbone": "Hierarchical Transformer (Swin-T)",
        "training_dataset": "LEVIR-CD + WHU-CD + OSCD",
        "task": "Bi-temporal change detection & damage assessment",
        "status": "active",
        **cd
    }

    return {
        "models": models,
        "task_routing": raw_data.get("task_routing", {}),
        "evaluation_benchmarks": raw_data.get("evaluation_benchmarks", {}),
    }

_UNIFIED_REGISTRY = load_unified_model_registry()
MODEL_REGISTRY_DATA = _UNIFIED_REGISTRY["models"]

# ──────────────────────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="SatQuery AI",
    description="ISRO Multimodal Agentic Remote Sensing Platform (SIH26167)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for running blocking ML inference without blocking the event loop
executor = ThreadPoolExecutor(max_workers=8)

# In-memory task store for async query status polling
_tasks: dict = {}   # task_id → {"status": str, "result": dict}

# ──────────────────────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: Optional[str] = ""
    session_id: Optional[str] = None
    scenario_id: Optional[str] = None
    task_id: Optional[str] = None
    mode: Optional[str] = None
    language: Optional[str] = "en"

class EvalRequest(BaseModel):
    session_id: str
    reference_mask_b64: str   # base64-encoded binary PNG reference mask

# ──────────────────────────────────────────────────────────────
# STARTUP: Warm up models
# ──────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Pre-load RemoteCLIP at startup (smallest model, ~500MB)."""
    logger.info("SatQuery AI starting up...")
    # Load RemoteCLIP first (needed for confidence scoring on every request)
    try:
        from .models.model_loader import get_model_loader
        loader = get_model_loader()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, loader.get_remote_clip)
        logger.info("RemoteCLIP warm-up complete.")
    except Exception as e:
        logger.warning(f"RemoteCLIP warm-up failed: {e}. Will load on first request.")


# ──────────────────────────────────────────────────────────────
# IMAGE UPLOAD
# ──────────────────────────────────────────────────────────────

@app.post("/api/v1/upload/image")
async def upload_single_image(file: UploadFile = File(...)):
    """
    Upload a single satellite image (GeoTIFF, PNG, JPEG).
    Returns session_id + extracted metadata.
    """
    from .geospatial.geotiff_loader import get_loader
    from .core.session_store import get_session_store

    allowed_exts = {".tif", ".tiff", ".geotiff", ".png", ".jpg", ".jpeg"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            400, f"Unsupported format: {ext}. Use: {', '.join(allowed_exts)}"
        )

    file_bytes = await file.read()
    if len(file_bytes) > 500 * 1024 * 1024:  # 500MB limit
        raise HTTPException(413, "File too large (max 500MB)")

    try:
        loader = get_loader(target_size=256)
        loop = asyncio.get_event_loop()
        image_data = await loop.run_in_executor(
            executor, lambda: loader.load_from_bytes(file_bytes, file.filename)
        )
    except Exception as e:
        raise HTTPException(422, f"Failed to load image: {e}")

    store = get_session_store()
    session = store.create_session()
    idx = session.add_image(image_data)

    return {
        "session_id": session.session_id,
        "image_index": idx,
        "filename": file.filename,
        "metadata": image_data["metadata"],
        "modality": image_data["modality"],
        "sensor": image_data["sensor"],
        "rgb_preview_b64": image_data.get("rgb_b64"),
    }


@app.post("/api/v1/upload/pair")
async def upload_image_pair(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
):
    """
    Upload two images (T1+T2 or Optical+SAR) for bi-temporal or fusion analysis.
    Returns session_id + co-registration check.
    """
    from .geospatial.geotiff_loader import get_loader
    from .geospatial.coregistration import get_checker
    from .core.session_store import get_session_store

    loader = get_loader(target_size=256)
    loop = asyncio.get_event_loop()

    bytes1 = await file1.read()
    bytes2 = await file2.read()

    try:
        img1 = await loop.run_in_executor(
            executor, lambda: loader.load_from_bytes(bytes1, file1.filename)
        )
        img2 = await loop.run_in_executor(
            executor, lambda: loader.load_from_bytes(bytes2, file2.filename)
        )
    except Exception as e:
        raise HTTPException(422, f"Failed to load images: {e}")

    # Auto-detect pair type
    mod1 = img1["modality"]
    mod2 = img2["modality"]
    pair_type = "unknown"
    if mod1 == "sar" or mod2 == "sar":
        pair_type = "sar_optical"
    else:
        pair_type = "bitemporal"

    # Co-registration check
    checker = get_checker()
    coreg = checker.check(img1["metadata"], img2["metadata"], task=pair_type)

    store = get_session_store()
    session = store.create_session()
    session.add_image(img1)
    session.add_image(img2)

    return {
        "session_id": session.session_id,
        "pair_type": pair_type,
        "image_1": {
            "filename": file1.filename,
            "sensor": img1["sensor"],
            "modality": img1["modality"],
            "metadata": img1["metadata"],
            "rgb_preview_b64": img1.get("rgb_b64"),
        },
        "image_2": {
            "filename": file2.filename,
            "sensor": img2["sensor"],
            "modality": img2["modality"],
            "metadata": img2["metadata"],
            "rgb_preview_b64": img2.get("rgb_b64"),
        },
        "coregistration": coreg,
    }


@app.get("/api/v1/upload/{session_id}/meta")
async def get_session_metadata(session_id: str):
    """Get metadata for all images in a session."""
    from .core.session_store import get_session_store
    store = get_session_store()
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found or expired")
    return session.to_summary_dict()


@app.post("/api/v1/upload/validate")
async def validate_upload(request: Request):
    """Validates single image or pair metadata for format, CRS, band compatibility."""
    from .core.validator import ImageValidator
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        data = await request.json()
    else:
        form = await request.form()
        data = dict(form)

    if "bands" in data and data["bands"] is not None:
        try:
            data["bands"] = int(data["bands"])
        except (ValueError, TypeError):
            pass

    validator = ImageValidator()
    if "image_1" in data and "image_2" in data:
        return validator.validate_pair(data["image_1"], data["image_2"], task=data.get("task", "bi_temporal"))
    return validator.validate_single(data)


# ──────────────────────────────────────────────────────────────
# SCENARIOS CATALOG & PRESETS
# ──────────────────────────────────────────────────────────────

@app.get("/api/v1/scenarios")
async def list_scenarios():
    """Return catalog of Indian benchmark remote sensing scenarios."""
    from ..data.scenarios import INDIAN_SCENARIOS
    return {"scenarios": list(INDIAN_SCENARIOS.values())}


@app.get("/api/v1/scenarios/{scenario_id}/images")
async def get_scenario_imagery(scenario_id: str):
    """Return high-fidelity synthetic/calibrated satellite imagery for preset scenarios."""
    from ..data.scenarios import INDIAN_SCENARIOS, generate_scenario_images
    if scenario_id not in INDIAN_SCENARIOS:
        raise HTTPException(404, f"Scenario '{scenario_id}' not found")
    images = generate_scenario_images(scenario_id)
    return {
        "scenario_id": scenario_id,
        "scenario": INDIAN_SCENARIOS[scenario_id],
        "images": images,
    }


# ──────────────────────────────────────────────────────────────
# QUERY SUBMISSION
# ──────────────────────────────────────────────────────────────

@app.post("/api/v1/query/submit")
async def submit_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Submit a natural language query for agent processing.
    Supports both real uploaded sessions and benchmark scenario evaluation.
    """
    # 1. Benchmark Scenario flow (immediate / synchronous result)
    if request.scenario_id:
        from ..data.scenarios import INDIAN_SCENARIOS
        from .core.agent import AgenticController
        if request.scenario_id not in INDIAN_SCENARIOS:
            raise HTTPException(404, f"Scenario '{request.scenario_id}' not found")
        scenario = INDIAN_SCENARIOS[request.scenario_id]
        agent = AgenticController()
        result = agent.process_query(
            query=request.query or "Analyze this remote sensing scene.",
            scenario_data=scenario,
            requested_mode=request.mode
        )
        task_id = str(uuid.uuid4())
        _tasks[task_id] = {"status": "complete", "result": result}
        result["task_id"] = task_id
        result["status"] = "complete"
        return result

    # 2. Session Upload flow (async + WebSocket streaming)
    from .core.session_store import get_session_store
    store = get_session_store()
    session = store.get_session(request.session_id) if request.session_id else None

    if not session:
        raise HTTPException(404, f"Session {request.session_id} not found")
    if not session.images:
        raise HTTPException(400, "No images in session. Upload images first.")

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {"status": "queued", "result": None}

    # Add to conversation history
    session.add_turn("user", request.query or "")

    background_tasks.add_task(
        _run_agent_task, task_id, request.query or "", session, request.mode
    )

    return {
        "task_id": task_id,
        "session_id": request.session_id,
        "status": "queued",
        "message": "Query accepted. Connect to WebSocket for live trace.",
        "websocket_url": f"/ws/{request.session_id}",
        "poll_url": f"/api/v1/query/{task_id}/status",
    }


@app.get("/api/v1/query/{task_id}/status")
async def get_task_status(task_id: str):
    """Poll task status. Returns result when complete."""
    if task_id not in _tasks:
        raise HTTPException(404, f"Task {task_id} not found")
    task = _tasks[task_id]
    return {
        "task_id": task_id,
        "status": task["status"],
        "result": task.get("result") if task["status"] == "complete" else None,
        "error": task.get("error"),
    }


@app.websocket("/ws/{session_id}")
async def websocket_trace_stream(websocket: WebSocket, session_id: str):
    """
    Live WebSocket streaming of agentic LangGraph execution trace.
    Allows frontend to render each decision step in real-time.
    """
    await websocket.accept()
    from .ws.stream_handler import get_stream_manager
    manager = get_stream_manager()
    manager.register(session_id, websocket)
    logger.info(f"WebSocket client registered for session: {session_id}")
    try:
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        manager.unregister(session_id)
        logger.info(f"WebSocket client disconnected for session: {session_id}")
    except Exception as e:
        logger.debug(f"WebSocket session {session_id} error: {e}")
        manager.unregister(session_id)


async def _run_agent_task(task_id: str, query: str, session, mode: Optional[str] = None):
    """Background task that runs the full agent pipeline."""
    from .core.agent import run_agent
    from .ws.stream_handler import get_stream_manager

    _tasks[task_id]["status"] = "processing"
    stream_mgr = get_stream_manager()
    callback = stream_mgr.get_callback(session.session_id)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: run_agent(
                query=query,
                images=session.images,
                session_id=session.session_id,
                stream_callback=callback,
                forced_mode=mode,
            )
        )
        session.add_turn("assistant", result.get("answer", result.get("fused_findings", "")))
        session.cache_result(result, result.get("task_type", ""), result.get("_change_map_raw"))
        _tasks[task_id]["status"] = "complete"
        _tasks[task_id]["result"] = result

        # Notify WebSocket that we're done
        await stream_mgr.send(session.session_id, {
            "type": "complete",
            "task_id": task_id,
            "result": result,
        })

    except Exception as e:
        logger.error(f"Agent task {task_id} failed: {e}")
        _tasks[task_id]["status"] = "error"
        _tasks[task_id]["error"] = str(e)
        await stream_mgr.send(session.session_id, {
            "type": "error",
            "task_id": task_id,
            "error": str(e),
        })


# ──────────────────────────────────────────────────────────────
# WEBSOCKET — Live agent trace stream
# ──────────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for live agent trace streaming."""
    from .ws.stream_handler import get_stream_manager
    from .core.session_store import get_session_store

    await websocket.accept()
    stream_mgr = get_stream_manager()
    stream_mgr.register(session_id, websocket)

    # Send connection confirmation
    store = get_session_store()
    session = store.get_session(session_id)
    await websocket.send_text(
        f'{{"type":"connected","session_id":"{session_id}",'
        f'"image_count":{session.image_count if session else 0}}}'
    )

    try:
        while True:
            # Keep connection alive — agent sends messages via stream_mgr.send()
            data = await websocket.receive_text()
            # Handle ping-pong for connection keep-alive
            if data == "ping":
                await websocket.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    finally:
        stream_mgr.unregister(session_id)


# ──────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ──────────────────────────────────────────────────────────────

@app.get("/api/v1/agent/models")
@app.get("/api/v1/models")
async def get_model_registry():
    """Return the model registry for both frontend display and SIH evaluation audit."""
    return {
        "status": "HEALTHY",
        "registry": MODEL_REGISTRY_DATA,
        "models": MODEL_REGISTRY_DATA,
    }


@app.get("/api/v1/models/status")
async def get_model_status():
    """Return which models are currently loaded and surface fallback advisories."""
    try:
        from .models.model_loader import get_model_loader
        loader = get_model_loader()
        status = loader.get_status()

        # Check if running on non-adapted weights or CPU fallbacks
        fallbacks = []
        load_stat = status.get("load_status", {})
        for mod, msg in load_stat.items():
            if any(term in str(msg).lower() for term in ["fallback", "no lora", "not found", "spectral differencing", "random"]):
                fallbacks.append(f"{mod}: {msg}")

        status["is_fallback_mode"] = len(fallbacks) > 0
        status["fallback_advisories"] = fallbacks
        return status
    except Exception as e:
        return {"error": str(e), "is_fallback_mode": False, "fallback_advisories": []}


# ──────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────

@app.post("/api/v1/evaluate/iou")
async def evaluate_iou(request: EvalRequest):
    """
    Evaluate cached change map against a user-provided reference mask.
    Computes IoU, F1, Precision, Recall.
    """
    import base64
    import io
    from PIL import Image
    from .core.session_store import get_session_store

    store = get_session_store()
    session = store.get_session(request.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    cached_result = session.last_result
    if not cached_result or session.last_task_type != "BI_TEMPORAL_CHANGE":
        raise HTTPException(400, "No change detection result cached for this session")

    # Decode reference mask
    try:
        mask_bytes = base64.b64decode(request.reference_mask_b64.split(",")[-1])
        ref_pil = Image.open(io.BytesIO(mask_bytes)).convert("L")
        ref_mask = (np.array(ref_pil) > 127).astype(np.uint8)
    except Exception as e:
        raise HTTPException(400, f"Invalid reference mask: {e}")

    # Re-run evaluation using change engine
    from .models.change_engine import get_change_engine
    engine = get_change_engine()

    evaluation = {"note": "For full IoU evaluation, re-run with reference mask in query"}

    if session.last_change_map is not None:
        evaluation = engine._evaluate_against_reference(session.last_change_map, ref_mask)
    else:
        evaluation = {"note": "Raw change map not cached. Re-submit query with reference mask."}

    return {
        "session_id": request.session_id,
        "evaluation": evaluation,
    }


# ──────────────────────────────────────────────────────────────
# EXPORT
# ──────────────────────────────────────────────────────────────

@app.post("/api/v1/export/report")
async def export_pdf_report(request: Request):
    """Generate an official ISRO-format PDF report."""
    from .reports.pdf_generator import generate_pdf_report
    body = await request.json()

    # Case 1: Direct analysis_result passed (e.g. from benchmark test or direct scenario evaluation)
    if "analysis_result" in body:
        analysis_result = body["analysis_result"]
        scenario_id = body.get("scenario_id")
        scenario_meta = {}
        if scenario_id:
            from ..data.scenarios import INDIAN_SCENARIOS
            scenario_meta = INDIAN_SCENARIOS.get(scenario_id, {})
        if not scenario_meta:
            scenario_meta = {
                "location": analysis_result.get("location", "India AOI"),
                "crs": analysis_result.get("crs", "EPSG:32643"),
                "coordinates": analysis_result.get("coordinates", "N/A"),
            }
        loop = asyncio.get_event_loop()
        pdf_bytes = await loop.run_in_executor(
            executor,
            lambda: generate_pdf_report(analysis_result, scenario_meta)
        )
        return Response(content=pdf_bytes, media_type="application/pdf")

    # Case 2: Session-based export (from app.js UI)
    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(400, "Either 'analysis_result' or 'session_id' must be provided")

    from .core.session_store import get_session_store
    store = get_session_store()
    session = store.get_session(session_id)
    if not session or not session.last_result:
        raise HTTPException(404, "No analysis result found for this session")

    summary = session.to_summary_dict()
    loop = asyncio.get_event_loop()
    pdf_bytes = await loop.run_in_executor(
        executor,
        lambda: generate_pdf_report(session.last_result, summary)
    )

    accept_header = request.headers.get("accept", "")
    if "application/pdf" in accept_header:
        return Response(content=pdf_bytes, media_type="application/pdf")

    import base64
    pdf_b64 = base64.b64encode(pdf_bytes).decode()
    return {
        "session_id": session_id,
        "pdf_b64": f"data:application/pdf;base64,{pdf_b64}",
        "filename": f"satquery_report_{session_id[:8]}.pdf",
    }


# ──────────────────────────────────────────────────────────────
# HEALTH & ADMIN
# ──────────────────────────────────────────────────────────────

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
    except (ImportError, Exception):
        cuda_avail = False

    from .core.session_store import get_session_store
    store = get_session_store()
    return {
        "status": "HEALTHY",
        "version": "2.0.0",
        "active_sessions": store.active_count,
        "active_tasks": len([t for t in _tasks.values() if t["status"] == "processing"]),
        "cuda_available": cuda_avail,
        "device": "cuda" if cuda_avail else "cpu",
    }


@app.get("/api/v1/sessions")
async def list_sessions():
    """List active sessions (debug endpoint)."""
    from .core.session_store import get_session_store
    store = get_session_store()
    return {"sessions": store.list_active_sessions()}


# ──────────────────────────────────────────────────────────────
# STATIC FILES (Frontend)
# ──────────────────────────────────────────────────────────────

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if not os.path.isdir(FRONTEND_DIR):
    FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))

@app.get("/", include_in_schema=False)
async def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "SatQuery AI (SIH26167) Backend Running"}

if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    logger.info(f"Mounted static frontend from {FRONTEND_DIR}")
else:
    logger.warning(f"Frontend directory not found at {FRONTEND_DIR}")


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,  # Single worker required for in-memory state
    )
