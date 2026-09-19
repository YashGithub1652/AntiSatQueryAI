"""
API integration tests for SatQuery AI FastAPI server.
"""

import unittest
import sys
import os
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from satquery.backend.main import app

class TestSatQueryAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        res = self.client.get("/api/v1/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "HEALTHY")

    def test_scenarios_catalog(self):
        res = self.client.get("/api/v1/scenarios")
        self.assertEqual(res.status_code, 200)
        scenarios = res.json()["scenarios"]
        self.assertGreaterEqual(len(scenarios), 6)
        ids = [s["id"] for s in scenarios]
        self.assertIn("kerala_floods_2018", ids)
        self.assertIn("delhi_sar_fusion", ids)

    def test_scenario_imagery(self):
        res = self.client.get("/api/v1/scenarios/kerala_floods_2018/images")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("images", data)
        self.assertTrue(data["images"]["image_t1"].startswith("data:image/png;base64,"))
        self.assertTrue(data["images"]["image_sar"].startswith("data:image/png;base64,"))

    def test_model_registry(self):
        res = self.client.get("/api/v1/agent/models")
        self.assertEqual(res.status_code, 200)
        reg = res.json()["registry"]
        self.assertIn("vqa", reg)
        self.assertIn("BigEarthNet.txt", reg["vqa"]["training_dataset"])

    def test_query_submission(self):
        payload = {
            "query": "What is the water inundation area?",
            "scenario_id": "kerala_floods_2018",
            "mode": "bi_temporal",
            "language": "en"
        }
        res = self.client.post("/api/v1/query/submit", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("answer", data)
        self.assertIn("trace", data)
        self.assertGreater(data["confidence"], 0.8)

    def test_upload_validate(self):
        res = self.client.post(
            "/api/v1/upload/validate",
            data={"format": "GeoTIFF", "modality": "Optical/MS", "bands": 13, "crs": "EPSG:32643"}
        )
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["valid"])

    def test_export_pdf_report(self):
        # First get a result
        q_res = self.client.post("/api/v1/query/submit", json={
            "query": "Identify flood extents",
            "scenario_id": "kerala_floods_2018"
        })
        analysis_result = q_res.json()

        rep_res = self.client.post("/api/v1/export/report", json={
            "analysis_result": analysis_result,
            "scenario_id": "kerala_floods_2018"
        })
        self.assertEqual(rep_res.status_code, 200)
        self.assertEqual(rep_res.headers["content-type"], "application/pdf")
        self.assertTrue(len(rep_res.content) > 1000)

    def test_evaluate_iou(self):
        import base64
        import io
        from PIL import Image
        import numpy as np
        from satquery.backend.core.session_store import get_session_store
        from satquery.backend.core.agent import run_agent

        store = get_session_store()
        session = store.create_session()
        dummy_img1 = {"sensor": "Sentinel-2", "modality": "optical", "metadata": {"resolution_m": 10}, "array": np.zeros((3, 64, 64), dtype=np.float32)}
        dummy_img2 = {"sensor": "Sentinel-2", "modality": "optical", "metadata": {"resolution_m": 10}, "array": np.ones((3, 64, 64), dtype=np.float32)}
        session.add_image(dummy_img1)
        session.add_image(dummy_img2)

        run_agent("What changed between these two images?", [dummy_img1, dummy_img2], session_id=session.session_id)
        self.assertIsNotNone(session.last_change_map)

        mask_pil = Image.fromarray((np.ones((64, 64)) * 255).astype(np.uint8))
        buf = io.BytesIO()
        mask_pil.save(buf, format="PNG")
        mask_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        eval_res = self.client.post("/api/v1/evaluate/iou", json={
            "session_id": session.session_id,
            "reference_mask_b64": mask_b64,
        })
        self.assertEqual(eval_res.status_code, 200)
        data = eval_res.json()
        self.assertIn("evaluation", data)
        self.assertIn("iou", data["evaluation"])
        self.assertIn("f1", data["evaluation"])

    def test_frontend_serving(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("SatQuery AI", res.text)
        self.assertIn("SIH26167", res.text)

if __name__ == "__main__":
    unittest.main()
