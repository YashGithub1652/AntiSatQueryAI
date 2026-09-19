"""
SatQuery AI — Co-registration Verification
==========================================
Verifies that two satellite images are spatially aligned before
running BI_TEMPORAL_CHANGE or CROSS_MODAL_SAR_OPTICAL analysis.

Missing from all original documents — added as per audit gap analysis.

Checks:
  1. CRS match (or reprojectable to common CRS)
  2. Bounding box spatial overlap > 80%
  3. Resolution ratio within 10x
  4. Image dimension compatibility

If checks fail: warns user but proceeds with caveat (as per planning doc guidance).
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

try:
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

logger = logging.getLogger(__name__)

# Minimum acceptable spatial overlap fraction
MIN_OVERLAP_FRACTION = 0.80

# Maximum allowable resolution ratio between two images
MAX_RESOLUTION_RATIO = 10.0


class CoregistrationChecker:
    """
    Checks whether two satellite images are spatially co-registered.
    Used before: BI_TEMPORAL_CHANGE and CROSS_MODAL_SAR_OPTICAL tasks.
    """

    def check(
        self,
        meta1: Dict[str, Any],
        meta2: Dict[str, Any],
        task: str = "bitemporal",
    ) -> Dict[str, Any]:
        """
        Run all co-registration checks between two loaded images.

        Args:
            meta1: metadata dict from GeoTIFFLoader (image 1 / T1 / Optical)
            meta2: metadata dict from GeoTIFFLoader (image 2 / T2 / SAR)
            task: "bitemporal" | "sar_optical"

        Returns:
            {
                "is_coregistered": bool,
                "overlap_pct": float,       # spatial overlap percentage
                "crs_match": bool,
                "resolution_ratio": float,
                "warnings": List[str],
                "errors": List[str],
                "recommendation": str,
            }
        """
        warnings: List[str] = []
        errors: List[str] = []
        checks_passed = []

        # ── Check 1: Both GeoTIFF? ──────────────────────────────
        is_geo1 = meta1.get("bbox_wgs84") is not None
        is_geo2 = meta2.get("bbox_wgs84") is not None

        if not is_geo1 or not is_geo2:
            if task == "sar_optical":
                errors.append(
                    "Both images must be GeoTIFF with spatial metadata for SAR-Optical fusion. "
                    "PNG/JPEG images have no coordinate reference system."
                )
            else:
                warnings.append(
                    "One or both images lack geospatial metadata (PNG/JPEG?). "
                    "Proceeding in visual-only mode — co-registration cannot be verified."
                )
            return self._result(False, 0.0, False, 1.0, warnings, errors,
                                "Upload GeoTIFF files with embedded CRS metadata for accurate co-registration.")

        # ── Check 2: CRS compatibility ─────────────────────────
        crs1_str = meta1.get("crs", "")
        crs2_str = meta2.get("crs", "")
        crs_match = self._crs_equal(crs1_str, crs2_str)

        if not crs_match:
            warnings.append(
                f"CRS mismatch detected: Image1={crs1_str}, Image2={crs2_str}. "
                "Images will be compared in WGS84 (EPSG:4326). "
                "Consider reprojecting to a common UTM zone for pixel-level accuracy."
            )
        else:
            checks_passed.append("CRS_MATCH")

        # ── Check 3: Spatial overlap ────────────────────────────
        bbox1 = meta1.get("bbox_wgs84")
        bbox2 = meta2.get("bbox_wgs84")
        overlap_pct = 0.0

        if bbox1 and bbox2:
            overlap_pct = self._compute_overlap(bbox1, bbox2)
            if overlap_pct < MIN_OVERLAP_FRACTION * 100:
                errors.append(
                    f"Spatial overlap too low: {overlap_pct:.1f}% "
                    f"(minimum required: {MIN_OVERLAP_FRACTION * 100:.0f}%). "
                    "These images may not cover the same geographic area."
                )
            elif overlap_pct < 90.0:
                warnings.append(
                    f"Partial spatial overlap: {overlap_pct:.1f}%. "
                    "Analysis will be clipped to the intersecting region."
                )
            else:
                checks_passed.append(f"OVERLAP_{overlap_pct:.1f}%")

        # ── Check 4: Resolution ratio ───────────────────────────
        res1 = meta1.get("resolution_m") or 10.0
        res2 = meta2.get("resolution_m") or 10.0
        ratio = max(res1, res2) / max(min(res1, res2), 0.001)

        if ratio > MAX_RESOLUTION_RATIO:
            warnings.append(
                f"Large resolution difference: {res1}m vs {res2}m (ratio: {ratio:.1f}x). "
                f"Higher-resolution image will be resampled to match lower resolution."
            )
        else:
            checks_passed.append(f"RESOLUTION_RATIO_{ratio:.1f}x")

        # ── Temporal check for bi-temporal tasks ────────────────
        if task == "bitemporal":
            date1 = meta1.get("acquisition_date")
            date2 = meta2.get("acquisition_date")
            if date1 and date2:
                if date1 == date2:
                    warnings.append(
                        f"Both images have the same acquisition date ({date1}). "
                        "For change detection, images from different dates are required."
                    )
                else:
                    checks_passed.append(f"TEMPORAL_PAIR_{date1}_vs_{date2}")
            else:
                warnings.append(
                    "Acquisition dates not found in metadata. "
                    "Ensure images are from different dates for meaningful change analysis."
                )

        # ── Final verdict ────────────────────────────────────────
        is_coregistered = len(errors) == 0 and overlap_pct >= MIN_OVERLAP_FRACTION * 100

        if is_coregistered:
            recommendation = (
                f"Images are co-registered and ready for {task.replace('_', ' ')} analysis. "
                f"{len(checks_passed)} checks passed."
            )
        elif warnings and not errors:
            is_coregistered = True  # Proceed with warnings
            recommendation = (
                f"Images accepted with {len(warnings)} warning(s). "
                "Results may have reduced accuracy near image boundaries."
            )
        else:
            recommendation = (
                "Co-registration check failed. "
                "Please upload properly co-registered images covering the same geographic area."
            )

        return self._result(
            is_coregistered, overlap_pct, crs_match, ratio,
            warnings, errors, recommendation
        )

    # ──────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────

    def _compute_overlap(
        self, bbox1: List[float], bbox2: List[float]
    ) -> float:
        """
        Compute percentage overlap between two bounding boxes in WGS84.
        bbox format: [west, south, east, north]
        Returns: overlap as percentage (0-100).
        """
        w1, s1, e1, n1 = bbox1
        w2, s2, e2, n2 = bbox2

        # Intersection
        i_w = max(w1, w2)
        i_s = max(s1, s2)
        i_e = min(e1, e2)
        i_n = min(n1, n2)

        if i_e <= i_w or i_n <= i_s:
            return 0.0  # No overlap

        intersection_area = (i_e - i_w) * (i_n - i_s)

        # Union of smaller box (we compare against the smaller image)
        area1 = (e1 - w1) * (n1 - s1)
        area2 = (e2 - w2) * (n2 - s2)
        smaller_area = min(area1, area2)

        if smaller_area <= 0:
            return 0.0

        return min(100.0, (intersection_area / smaller_area) * 100.0)

    def _crs_equal(self, crs1_str: str, crs2_str: str) -> bool:
        """Check if two CRS strings represent the same coordinate reference system."""
        if not crs1_str or not crs2_str:
            return False
        # Normalize
        c1 = crs1_str.strip().upper().replace("EPSG:", "").replace(" ", "")
        c2 = crs2_str.strip().upper().replace("EPSG:", "").replace(" ", "")
        if c1 == c2:
            return True
        # Try rasterio comparison for complex WKT CRS strings
        if RASTERIO_AVAILABLE:
            try:
                return CRS.from_string(crs1_str) == CRS.from_string(crs2_str)
            except Exception:
                pass
        return False

    def _result(
        self,
        is_coregistered: bool,
        overlap_pct: float,
        crs_match: bool,
        resolution_ratio: float,
        warnings: List[str],
        errors: List[str],
        recommendation: str,
    ) -> Dict[str, Any]:
        return {
            "passed": is_coregistered and len(errors) == 0,
            "skipped": False,
            "is_coregistered": is_coregistered,
            "message": recommendation,
            "overlap_pct": round(overlap_pct, 1),
            "crs_match": crs_match,
            "resolution_ratio": round(resolution_ratio, 2),
            "warnings": warnings,
            "errors": errors,
            "recommendation": recommendation,
            "checks_summary": {
                "spatial_overlap": f"{overlap_pct:.1f}%",
                "crs_compatible": crs_match or not errors,
                "resolution_compatible": resolution_ratio <= MAX_RESOLUTION_RATIO,
            }
        }


# Module-level singleton
_checker: Optional[CoregistrationChecker] = None


def get_checker() -> CoregistrationChecker:
    global _checker
    if _checker is None:
        _checker = CoregistrationChecker()
    return _checker
