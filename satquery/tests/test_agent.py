"""
Unit tests for SatQuery AI Agentic Controller, Validator, and Specialist Engines.
"""

import unittest
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from satquery.backend.core.agent import AgenticController
from satquery.backend.core.validator import ImageValidator
from satquery.data.scenarios import INDIAN_SCENARIOS, generate_scenario_images
from satquery.backend.reports.pdf_generator import generate_pdf_report

class TestSatQueryAgent(unittest.TestCase):
    def setUp(self):
        self.agent = AgenticController()
        self.validator = ImageValidator()

    def test_single_image_validation(self):
        meta = {"format": "GeoTIFF", "bands": 13, "modality": "Optical/MS"}
        res = self.validator.validate_single(meta)
        self.assertTrue(res["valid"])
        self.assertEqual(res["metadata"]["format"], "GeoTIFF")

    def test_pair_image_validation(self):
        m1 = {"format": "GeoTIFF", "bands": 13, "modality": "Optical/MS", "dimensions": "512x512"}
        m2 = {"format": "GeoTIFF", "bands": 2, "modality": "SAR", "dimensions": "512x512"}
        res = self.validator.validate_pair(m1, m2, task="sar_fusion")
        self.assertTrue(res["valid"])
        self.assertEqual(res["co_registration"], "VERIFIED_SUB_PIXEL")

    def test_vqa_query_execution(self):
        scenario = INDIAN_SCENARIOS["kerala_floods_2018"]
        out = self.agent.process_query(
            query="What is the extent of water inundation?",
            scenario_data=scenario,
            requested_mode="bi_temporal"
        )
        self.assertEqual(out["task_type"], "BI_TEMPORAL_CHANGE")
        self.assertIn("inundation", out["answer"].lower())
        self.assertGreaterEqual(len(out["trace"]), 6)
        self.assertGreater(out["confidence"], 0.8)

    def test_sar_fusion_execution(self):
        scenario = INDIAN_SCENARIOS["delhi_sar_fusion"]
        out = self.agent.process_query(
            query="Detect industrial structures through heavy cloud cover using SAR.",
            scenario_data=scenario,
            requested_mode="sar_fusion"
        )
        self.assertIsNotNone(out["sar_fusion_analysis"])
        self.assertIn("cross-attention", out["sar_fusion_analysis"]["summary"].lower())

    def test_grounding_execution(self):
        scenario = INDIAN_SCENARIOS["punjab_crop_monitoring"]
        out = self.agent.process_query(
            query="Locate crop residue burning and stubble parcels.",
            scenario_data=scenario,
            requested_mode="grounding"
        )
        self.assertIsNotNone(out["grounding_analysis"])
        self.assertGreater(len(out["grounding_analysis"]["grounded_regions"]), 0)

    def test_pdf_report_generation(self):
        scenario = INDIAN_SCENARIOS["kerala_floods_2018"]
        out = self.agent.process_query(
            query="Calculate water inundation metrics.",
            scenario_data=scenario
        )
        pdf_bytes = generate_pdf_report(out, scenario)
        self.assertTrue(len(pdf_bytes) > 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

if __name__ == "__main__":
    unittest.main()
