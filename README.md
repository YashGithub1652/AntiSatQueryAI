# SatQuery AI 🛰️
### Multimodal Agentic Remote Sensing Analysis with SAR-Optical Fusion
**Smart India Hackathon (SIH 2026) — Problem Statement ID: SIH26167**  
**Theme:** Space Technology | **Category:** Software | **Team:** Team PlotVortex

---

## 🌟 Executive Overview
Extracting actionable intelligence from Earth Observation (EO) satellite imagery currently demands deep GIS domain knowledge, manual multi-sensor band arithmetic, and disconnected software tools. 

**SatQuery AI** is an indigenous, agent-orchestrated vision-language platform that empowers anyone—from NDRF disaster commanders to agricultural planners—to upload satellite imagery (GeoTIFF / multispectral / SAR) and ask questions in plain English or Hindi. The system delivers:
1. **Evidence-Grounded Natural Language Answers** (Single Image VQA & Captioning).
2. **Text-Guided Region Grounding** (RSVG bounding box localization).
3. **Bi-Temporal Change Analysis** (ChangeFormer pixel difference mapping & quantitative damage metrics).
4. **All-Weather SAR-Optical Cross-Modal Fusion** (Penetrating clouds with Sentinel-1 C-band radar backscatter and combining with Sentinel-2 optical reflectance).
5. **Observable Agentic Execution Trace** (LangGraph-style 6-node decision graph showing query parsing, band validation, model routing, inference, and audit logging).
6. **Instant ISRO SAC-Standard PDF Audit Reports**.

---

## 🏗️ Master System Architecture & Code Flow

```
                      [ User Query (Web / Voice) + GeoTIFF Images ]
                                            │
                                            ▼
                    ┌──────────────────────────────────────────────┐
                    │      FastAPI Gateway & Static Frontend       │
                    │   (Dark Glassmorphic Aerospace Interface)   │
                    └───────────────────────┬──────────────────────┘
                                            │
                                            ▼
                    ┌──────────────────────────────────────────────┐
                    │       LangGraph Agentic Controller           │
                    │      (satquery/backend/core/agent.py)        │
                    └───────────────────────┬──────────────────────┘
                                            │
         ┌──────────────────────────────────┼──────────────────────────────────┐
         ▼                                  ▼                                  ▼
[Node 1: Query Parser]           [Node 2: Image Validator]         [Node 3: Model Router]
- Intent Classification          - GeoTIFF format check            - Dynamic registry lookup
- Spatial / Temporal semantics   - Band count (MS vs SAR)          - Graph execution planner
- Hindi / English routing        - Sub-pixel co-registration       - 4-bit edge quantization
         │                                  │                                  │
         └──────────────────────────────────┼──────────────────────────────────┘
                                            │
                                            ▼
                    ┌──────────────────────────────────────────────┐
                    │        [Node 4: Inference Executor]          │
                    └───────────────────────┬──────────────────────┘
                                            │
         ┌──────────────────┬───────────────┴──────────────┬──────────────────┐
         ▼                  ▼                              ▼                  ▼
  [GeoChat-LoRA]     [ChangeFormer]                 [SAR-Optical]           [RSVG]
Single Image VQA    Bi-temporal Change             Cross-Attention     Region Grounding
BigEarthNet.txt     Pixel Difference Mask          Cloud Penetration   Bounding Boxes
         │                  │                              │                  │
         └──────────────────┼──────────────────────────────┼──────────────────┘
                                            │
                                            ▼
                    ┌──────────────────────────────────────────────┐
                    │          [Node 5: Output Aggregator]         │
                    │  - Synthesizes text, heatmaps, bboxes        │
                    │  - Multi-factor confidence calibration       │
                    └───────────────────────┬──────────────────────┘
                                            │
                                            ▼
                    ┌──────────────────────────────────────────────┐
                    │           [Node 6: Audit Logger]             │
                    │  - Records immutable execution telemetry     │
                    │  - Generates ISRO SAC-Format PDF Report      │
                    └──────────────────────────────────────────────┘
```

---

## ⚡ Quickstart Guide (Run Locally in 10 Seconds)

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.11)
- All required packages installed via `pip`:
  ```bash
  pip install fastapi uvicorn pydantic python-multipart reportlab pillow numpy scipy httpx
  ```

### 2. Launch the Application Server
```bash
python satquery/backend/main.py
```
*Or via Uvicorn:*
```bash
uvicorn satquery.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Access the Web Dashboard
Open your browser and navigate to:
```
http://127.0.0.1:8000
```
- Interactive Command Center: `http://127.0.0.1:8000`
- Interactive OpenAPI Swagger Docs: `http://127.0.0.1:8000/docs`

---

## 🛰️ Pre-Loaded Indian Benchmark Scenarios
The prototype includes 6 verified Indian satellite benchmark scenarios:
1. **Kerala Floods (2018):** Alappuzha & Ernakulam — Bi-temporal inundation analysis (`142.8 sq km` submerged, `28.4%` impact).
2. **Mumbai Urban Growth (2014–2024):** Navi Mumbai & Thane Creek — Infrastructure growth (`+31.2%`) and coastal vegetation loss.
3. **Delhi Industrial Corridor:** Cloud-piercing SAR microwave backscatter detecting 312 structures hidden under heavy smog.
4. **Uttarakhand Forest Fire (2021):** Nainital & Almora — Burn severity mapping and active perimeter tracking.
5. **Chilika Lake Odisha:** Wetland lagoon dynamics, macrophyte weed infestation, and aquaculture pen localization.
6. **Punjab Agricultural Belt:** Stubble burn parcel detection, crop phenology, and NDVI water stress index.

---

## 📡 REST API Documentation

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Service health, version, and active model count |
| `GET` | `/api/v1/scenarios` | Catalog of Indian benchmark test scenarios |
| `GET` | `/api/v1/scenarios/{id}/images` | Photorealistic synthetic multi-band & SAR imagery |
| `GET` | `/api/v1/agent/models` | Active model registry (BigEarthNet.txt weights, versions) |
| `POST` | `/api/v1/query/submit` | Main agent pipeline execution endpoint |
| `POST` | `/api/v1/upload/validate` | GeoTIFF / SAR / Optical compatibility inspector |
| `POST` | `/api/v1/export/report` | Generates official ISRO SAC-format PDF report |

---

## 🧪 Automated Test Suite
To verify all agent nodes, image validators, specialist engines, and API routes:
```bash
# Agent & engine unit tests
python -m unittest satquery/tests/test_agent.py

# API integration tests
python -m unittest satquery/tests/test_api.py
```

---

## 🎥 75-Second Demo Video Script
1. **0:00 - 0:15:** Introduce Problem Statement SIH26167 and show the SatQuery AI dashboard.
2. **0:15 - 0:35:** Select *Kerala Floods 2018*, submit query, drag the Before/After split slider, and point to the *Observable Agentic Trace* showing LangGraph routing.
3. **0:35 - 0:55:** Select *Delhi SAR Fusion*, switch to Fused Layer, demonstrating radar piercing cloud cover.
4. **0:55 - 1:15:** Click "Download ISRO Report (PDF)" and display the auto-generated publication report.
5. **1:15 - 1:25:** Show the open-source GitHub repository and live deployment link.
