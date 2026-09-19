"""
SatQuery AI — Band Extractor
==============================
Extracts the correct band combination from a multi-band satellite array
depending on the target sensor and task.

Key principle: DO NOT hardcode Sentinel-2 band indices.
All band selection is driven by the detected sensor profile.
This ensures Cartosat-2S, RISAT-1, Landsat-8 all flow through
the same pipeline without modification.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Band index mappings per sensor (0-indexed into the loaded array)
BAND_MAPS = {
    "sentinel_2": {
        # Sentinel-2 L2A: B1(Coastal), B2(Blue), B3(Green), B4(Red),
        #                  B5(RE1), B6(RE2), B7(RE3), B8(NIR), B8A(RE4),
        #                  B9(WV), B10(SWIR-C), B11(SWIR1), B12(SWIR2)
        "rgb":      [2, 1, 0],   # B3, B2, B1 → Red, Green, Blue (visible)
        "rgb_true": [3, 2, 1],   # B4, B3, B2 → True colour composite
        "nir":      [7],         # B8 — vegetation
        "swir":     [10, 11],    # B11, B12 — moisture, burn
        "ndvi":     (7, 3),      # (NIR, Red) → (B8, B4)
        "ndwi":     (2, 7),      # (Green, NIR) → (B3, B8)
        "nbr":      (7, 11),     # (NIR, SWIR2) → (B8, B12)
        "model_rgb": [3, 2, 1],  # bands to feed as RGB into VLM
    },
    "landsat_8": {
        # Landsat-8 OLI: B1(CA), B2(Blue), B3(Green), B4(Red), B5(NIR), B6(SWIR1), B7(SWIR2)
        "rgb":      [1, 2, 3],   # B2, B3, B4
        "rgb_true": [3, 2, 1],
        "nir":      [4],
        "swir":     [5, 6],
        "ndvi":     (4, 3),
        "model_rgb": [3, 2, 1],
    },
    "cartosat_2s": {
        # Panchromatic only — single band
        "rgb":      [0, 0, 0],   # Replicate band 3 times for RGB
        "nir":      [0],
        "model_rgb": [0, 0, 0],
    },
    "risat_1": {
        # SAR: HH, HV
        "sar_channels": [0, 1],  # HH, HV
        "model_rgb": [0, 1, 0],  # HH, HV, HH for false-colour
    },
    "sentinel_1": {
        # SAR: VV, VH
        "sar_channels": [0, 1],  # VV, VH
        "model_rgb": [0, 1, 0],  # VV, VH, VV for false-colour
    },
    "generic_rgb": {
        "rgb":      [0, 1, 2],
        "model_rgb": [0, 1, 2],
    },
}


class BandExtractor:
    """
    Extracts task-appropriate band combinations from loaded satellite arrays.

    All operations return float32 arrays in [0, 1] normalized range.
    """

    def extract_for_vlm(
        self, array: np.ndarray, sensor: str
    ) -> np.ndarray:
        """
        Extract a 3-channel float32 [3, H, W] input suitable for Vision-Language Models.
        Most VLMs (GeoChat, RemoteCLIP) accept RGB-like 3-channel inputs.
        """
        band_map = BAND_MAPS.get(sensor, BAND_MAPS["generic_rgb"])
        idx = band_map.get("model_rgb", [0, 1, 2])
        return self._extract_channels(array, idx)

    def extract_rgb(
        self, array: np.ndarray, sensor: str, true_colour: bool = True
    ) -> np.ndarray:
        """Extract RGB composite [3, H, W] for display purposes."""
        band_map = BAND_MAPS.get(sensor, BAND_MAPS["generic_rgb"])
        key = "rgb_true" if true_colour else "rgb"
        idx = band_map.get(key, band_map.get("rgb", [0, 1, 2]))
        return self._extract_channels(array, idx)

    def extract_sar_pair(
        self, array: np.ndarray, sensor: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract VV and VH (or HH and HV) channels separately.
        Returns: (vv_array [1, H, W], vh_array [1, H, W])
        """
        band_map = BAND_MAPS.get(sensor, BAND_MAPS["sentinel_1"])
        ch = band_map.get("sar_channels", [0, 1])
        vv_idx, vh_idx = ch[0], ch[min(1, len(ch) - 1)]
        n = array.shape[0]
        vv = array[min(vv_idx, n - 1)][np.newaxis, ...]
        vh = array[min(vh_idx, n - 1)][np.newaxis, ...]
        return vv, vh

    def compute_ndvi(
        self, array: np.ndarray, sensor: str
    ) -> Optional[np.ndarray]:
        """
        Compute NDVI = (NIR - Red) / (NIR + Red + eps).
        Returns [H, W] float32 in [-1, 1] or None if bands unavailable.
        """
        band_map = BAND_MAPS.get(sensor, {})
        if "ndvi" not in band_map:
            return None
        nir_idx, red_idx = band_map["ndvi"]
        n = array.shape[0]
        if nir_idx >= n or red_idx >= n:
            return None
        nir = array[nir_idx].astype(np.float32)
        red = array[red_idx].astype(np.float32)
        ndvi = (nir - red) / (nir + red + 1e-8)
        return np.clip(ndvi, -1.0, 1.0)

    def compute_ndwi(
        self, array: np.ndarray, sensor: str
    ) -> Optional[np.ndarray]:
        """
        Compute NDWI = (Green - NIR) / (Green + NIR + eps).
        Highlights water bodies.
        """
        band_map = BAND_MAPS.get(sensor, {})
        if "ndwi" not in band_map:
            return None
        green_idx, nir_idx = band_map["ndwi"]
        n = array.shape[0]
        if green_idx >= n or nir_idx >= n:
            return None
        green = array[green_idx].astype(np.float32)
        nir = array[nir_idx].astype(np.float32)
        ndwi = (green - nir) / (green + nir + 1e-8)
        return np.clip(ndwi, -1.0, 1.0)

    def compute_nbr(
        self, array: np.ndarray, sensor: str
    ) -> Optional[np.ndarray]:
        """
        Compute NBR = (NIR - SWIR2) / (NIR + SWIR2 + eps).
        Normalized Burn Ratio — highlights fire scars.
        """
        band_map = BAND_MAPS.get(sensor, {})
        if "nbr" not in band_map:
            return None
        nir_idx, swir2_idx = band_map["nbr"]
        n = array.shape[0]
        if nir_idx >= n or swir2_idx >= n:
            return None
        nir = array[nir_idx].astype(np.float32)
        swir2 = array[swir2_idx].astype(np.float32)
        nbr = (nir - swir2) / (nir + swir2 + 1e-8)
        return np.clip(nbr, -1.0, 1.0)

    def get_spectral_stats(
        self, array: np.ndarray, sensor: str
    ) -> Dict[str, Any]:
        """
        Compute per-band statistics for display in the execution trace.
        """
        stats = {}
        n_bands = array.shape[0]
        for i in range(n_bands):
            band = array[i]
            stats[f"band_{i}"] = {
                "min": float(band.min()),
                "max": float(band.max()),
                "mean": float(band.mean()),
                "std": float(band.std()),
            }

        # Vegetation health indicator
        ndvi = self.compute_ndvi(array, sensor)
        if ndvi is not None:
            stats["ndvi"] = {
                "min": float(ndvi.min()),
                "max": float(ndvi.max()),
                "mean": float(ndvi.mean()),
                "healthy_fraction": float((ndvi > 0.3).mean()),
            }

        # Water body indicator
        ndwi = self.compute_ndwi(array, sensor)
        if ndwi is not None:
            stats["ndwi"] = {
                "mean": float(ndwi.mean()),
                "water_fraction": float((ndwi > 0.0).mean()),
            }

        return stats

    # ──────────────────────────────────────────────────────────
    def _extract_channels(
        self, array: np.ndarray, indices: List[int]
    ) -> np.ndarray:
        """Extract channels by index list, returning [len(indices), H, W]."""
        n = array.shape[0]
        channels = []
        for idx in indices:
            safe_idx = min(idx, n - 1)
            ch = array[safe_idx].astype(np.float32)
            # Ensure [0, 1] range
            if ch.max() > 1.0:
                ch = ch / ch.max()
            channels.append(ch)
        return np.stack(channels, axis=0)


# Module-level singleton
_extractor: Optional[BandExtractor] = None


def get_extractor() -> BandExtractor:
    global _extractor
    if _extractor is None:
        _extractor = BandExtractor()
    return _extractor
