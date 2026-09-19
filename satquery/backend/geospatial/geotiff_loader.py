"""
SatQuery AI — Real GeoTIFF Loader
==================================
Reads actual satellite GeoTIFF files using rasterio.
Extracts CRS, bounding box, bands, resolution, acquisition date from metadata.
Returns normalized numpy arrays ready for model inference.

Supports: Sentinel-1 SAR, Sentinel-2 MSI, Landsat-8, Cartosat-2S, RISAT-1, Generic RGB
"""

import os
import io
import base64
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
from PIL import Image

# Lazy imports — rasterio may not be installed on every machine
try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    logging.warning("rasterio not installed. Using fallback PNG/JPEG loader.")

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Sensor profile defaults (matches model_registry.yaml)
# ──────────────────────────────────────────────────────────────
SENSOR_PROFILES = {
    "sentinel_2": {
        "expected_bands": 13,
        "rgb_bands": [3, 2, 1],    # B4, B3, B2 (0-indexed)
        "nir_band": 7,             # B8
        "swir_bands": [10, 11],    # B11, B12
        "resolution_m": 10,
    },
    "sentinel_1": {
        "expected_bands": 2,
        "polarizations": ["VV", "VH"],
        "resolution_m": 10,
        "valid_range_db": (-25.0, 0.0),
    },
    "landsat_8": {
        "expected_bands": 7,
        "rgb_bands": [3, 2, 1],
        "resolution_m": 30,
    },
    "cartosat_2s": {
        "expected_bands": 1,
        "resolution_m": 0.65,
    },
    "generic_rgb": {
        "expected_bands": 3,
        "resolution_m": None,
    },
}


class GeoTIFFLoader:
    """
    Loads and normalizes satellite imagery for model inference.
    Handles GeoTIFF (rasterio) and PNG/JPEG (PIL fallback).
    """

    def __init__(self, target_size: int = 256):
        """
        Args:
            target_size: Output spatial dimension (HxW) for model input.
                         Default 256 works for ChangeFormer; use 224 for CLIP/ViT.
        """
        self.target_size = target_size

    # ──────────────────────────────────────────────────────────
    # PRIMARY ENTRY POINT
    # ──────────────────────────────────────────────────────────

    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load a satellite image from disk and return everything needed by agents.

        Returns:
            {
                "array": np.ndarray,        # normalized [C, H, W] float32 in [0, 1]
                "rgb_preview": np.ndarray,  # uint8 [H, W, 3] for display
                "rgb_b64": str,             # base64-encoded JPEG preview
                "metadata": dict,           # CRS, bbox, bands, resolution, date, sensor
                "modality": str,            # "optical" | "sar" | "rgb"
                "sensor": str,              # detected sensor type
                "is_geotiff": bool,
            }
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext in (".tif", ".tiff", ".geotiff") and RASTERIO_AVAILABLE:
            return self._load_geotiff(file_path)
        elif ext in (".png", ".jpg", ".jpeg"):
            return self._load_rgb(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use GeoTIFF, TIFF, PNG, or JPEG.")

    def load_from_bytes(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Load from in-memory bytes (used with FastAPI UploadFile)."""
        ext = os.path.splitext(filename)[1].lower()

        if ext in (".tif", ".tiff", ".geotiff") and RASTERIO_AVAILABLE:
            return self._load_geotiff_bytes(file_bytes)
        elif ext in (".png", ".jpg", ".jpeg"):
            return self._load_rgb_bytes(file_bytes)
        else:
            raise ValueError(f"Unsupported format: {ext}")

    # ──────────────────────────────────────────────────────────
    # GEOTIFF LOADING (rasterio)
    # ──────────────────────────────────────────────────────────

    def _load_geotiff(self, file_path: str) -> Dict[str, Any]:
        with rasterio.open(file_path) as src:
            return self._process_rasterio_dataset(src)

    def _load_geotiff_bytes(self, file_bytes: bytes) -> Dict[str, Any]:
        with rasterio.open(io.BytesIO(file_bytes)) as src:
            return self._process_rasterio_dataset(src)

    def _process_rasterio_dataset(self, src: "rasterio.DatasetReader") -> Dict[str, Any]:
        """Core GeoTIFF processing using an open rasterio dataset."""
        band_count = src.count
        crs_str = str(src.crs) if src.crs else "Unknown"
        resolution = abs(src.transform.a)  # pixel size in CRS units (metres for UTM)
        width, height = src.width, src.height

        # Bounding box in native CRS
        bbox_native = src.bounds
        # Convert to WGS84 for universal comparison
        try:
            bbox_wgs84 = transform_bounds(src.crs, CRS.from_epsg(4326), *bbox_native)
        except Exception:
            bbox_wgs84 = bbox_native

        # Acquisition date from metadata tags
        acq_date = self._extract_date(src.tags())

        # Detect sensor / modality
        sensor, modality = self._detect_sensor(band_count, src.tags(), src.descriptions)

        # Read all bands as float32
        raw_data = src.read().astype(np.float32)   # shape: [C, H, W]

        # Build normalized model-ready array
        model_array = self._normalize_bands(raw_data, modality, sensor)

        # Build RGB preview (uint8, [H, W, 3])
        rgb_preview = self._make_rgb_preview(raw_data, sensor, modality)

        # Resize to target_size
        model_array = self._resize_array(model_array, self.target_size)
        rgb_preview = self._resize_rgb(rgb_preview, self.target_size)

        # Encode preview as base64 JPEG
        rgb_b64 = self._array_to_b64(rgb_preview)

        metadata = {
            "format": "GeoTIFF",
            "sensor": sensor,
            "modality": modality,
            "band_count": band_count,
            "crs": crs_str,
            "resolution_m": round(resolution, 2),
            "dimensions": f"{width}x{height} px",
            "bbox_native": list(bbox_native),
            "bbox_wgs84": list(bbox_wgs84),
            "acquisition_date": acq_date,
            "nodata_value": src.nodata,
            "dtype": str(src.dtypes[0]),
        }

        return {
            "array": model_array,
            "rgb_preview": rgb_preview,
            "rgb_b64": rgb_b64,
            "metadata": metadata,
            "modality": modality,
            "sensor": sensor,
            "is_geotiff": True,
        }

    # ──────────────────────────────────────────────────────────
    # PNG / JPEG FALLBACK (benchmark images — VRSBench, RSVQA)
    # ──────────────────────────────────────────────────────────

    def _load_rgb(self, file_path: str) -> Dict[str, Any]:
        img = Image.open(file_path).convert("RGB")
        return self._process_pil(img)

    def _load_rgb_bytes(self, file_bytes: bytes) -> Dict[str, Any]:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return self._process_pil(img)

    def _process_pil(self, img: Image.Image) -> Dict[str, Any]:
        arr = np.array(img, dtype=np.float32) / 255.0   # [H, W, 3] normalized
        arr_chw = arr.transpose(2, 0, 1)                 # [3, H, W]

        rgb_preview = (arr * 255).astype(np.uint8)
        rgb_resized = self._resize_rgb(rgb_preview, self.target_size)
        arr_resized = self._resize_array(arr_chw, self.target_size)
        rgb_b64 = self._array_to_b64(rgb_resized)

        return {
            "array": arr_resized,
            "rgb_preview": rgb_resized,
            "rgb_b64": rgb_b64,
            "metadata": {
                "format": "PNG/JPEG",
                "sensor": "generic_rgb",
                "modality": "rgb",
                "band_count": 3,
                "crs": "None",
                "resolution_m": None,
                "dimensions": f"{img.width}x{img.height} px",
                "bbox_native": None,
                "bbox_wgs84": None,
                "acquisition_date": None,
            },
            "modality": "rgb",
            "sensor": "generic_rgb",
            "is_geotiff": False,
        }

    # ──────────────────────────────────────────────────────────
    # SENSOR DETECTION
    # ──────────────────────────────────────────────────────────

    def _detect_sensor(
        self, band_count: int, tags: dict, descriptions: Optional[List]
    ) -> Tuple[str, str]:
        """
        Returns (sensor_name, modality) based on band count + metadata tags.
        modality: "optical" | "sar" | "rgb"
        """
        tags_str = str(tags).lower() + str(descriptions).lower()

        # SAR detection: VV/VH polarization in metadata, or 1-2 bands + SAR tags
        sar_keywords = ["vv", "vh", "hh", "hv", "sar", "sentinel-1", "risat", "backscatter", "sigma0"]
        if any(k in tags_str for k in sar_keywords) or (band_count <= 2 and "sigma" in tags_str):
            if "risat" in tags_str:
                return "risat_1", "sar"
            return "sentinel_1", "sar"

        # Multispectral Sentinel-2
        if band_count >= 10:
            return "sentinel_2", "optical"

        # Cartosat-2S panchromatic
        if band_count == 1 and ("cartosat" in tags_str or "pan" in tags_str):
            return "cartosat_2s", "optical"

        # Landsat-8
        if 6 <= band_count <= 8:
            return "landsat_8", "optical"

        # Generic SAR (2-band, no metadata)
        if band_count == 2:
            return "sentinel_1", "sar"

        # Default RGB
        return "generic_rgb", "rgb"

    # ──────────────────────────────────────────────────────────
    # NORMALIZATION
    # ──────────────────────────────────────────────────────────

    def _normalize_bands(
        self, raw: np.ndarray, modality: str, sensor: str
    ) -> np.ndarray:
        """
        Normalize raw band data to [0, 1] float32.
        SAR: clip to [-25, 0] dB then normalize.
        Optical: percentile stretch (2nd - 98th).
        """
        if modality == "sar":
            # Convert to dB if raw linear values
            # BigEarthNet S1 is already in dB. Raw SNAP output may be linear.
            data = raw.copy()
            # If values look like linear (very small floats), convert to dB
            if data.max() < 1.0 and data.min() >= 0.0:
                data = np.where(data > 0, 10 * np.log10(data), -30.0)
            # Clip to valid SAR range
            data = np.clip(data, -25.0, 0.0)
            # Normalize to [0, 1]
            data = (data - (-25.0)) / (0.0 - (-25.0))
            return data.astype(np.float32)

        else:
            # Optical: per-band percentile stretch
            out = np.zeros_like(raw, dtype=np.float32)
            for i in range(raw.shape[0]):
                band = raw[i]
                p2, p98 = np.percentile(band[band > 0], [2, 98]) if band.max() > 0 else (0, 1)
                p98 = max(p98, p2 + 1e-6)
                out[i] = np.clip((band - p2) / (p98 - p2), 0.0, 1.0)
            return out

    # ──────────────────────────────────────────────────────────
    # RGB PREVIEW GENERATION
    # ──────────────────────────────────────────────────────────

    def _make_rgb_preview(
        self, raw: np.ndarray, sensor: str, modality: str
    ) -> np.ndarray:
        """
        Build a uint8 [H, W, 3] RGB preview image from raw band data.
        SAR: false color (VV=R, VH=G, VV-VH=B) or grayscale.
        Optical: true color RGB.
        """
        profile = SENSOR_PROFILES.get(sensor, SENSOR_PROFILES["generic_rgb"])

        if modality == "sar":
            # SAR false-color composite
            n_bands = raw.shape[0]
            if n_bands >= 2:
                vv = raw[0]; vh = raw[1]
                # Convert to dB if needed
                if vv.max() < 1.0:
                    vv = np.where(vv > 0, 10 * np.log10(vv), -30.0)
                    vh = np.where(vh > 0, 10 * np.log10(vh), -30.0)
                vv = np.clip(vv, -25, 0); vv = (vv + 25) / 25
                vh = np.clip(vh, -25, 0); vh = (vh + 25) / 25
                ratio = np.clip(vv - vh, 0, 1)
                rgb = np.stack([vv, vh, ratio], axis=-1)
            else:
                gray = raw[0]; gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
                rgb = np.stack([gray, gray, gray], axis=-1)
        elif modality in ("optical", "rgb"):
            rgb_idx = profile.get("rgb_bands", [0, 1, 2])
            n_bands = raw.shape[0]
            r_idx = min(rgb_idx[0], n_bands - 1)
            g_idx = min(rgb_idx[1], n_bands - 1)
            b_idx = min(rgb_idx[2], n_bands - 1)
            r = raw[r_idx]; g = raw[g_idx]; b = raw[b_idx]

            def _stretch(x):
                p2, p98 = np.percentile(x[x > 0], [2, 98]) if x.max() > 0 else (0, 1)
                p98 = max(p98, p2 + 1e-6)
                return np.clip((x - p2) / (p98 - p2), 0, 1)

            rgb = np.stack([_stretch(r), _stretch(g), _stretch(b)], axis=-1)
        else:
            # Fallback: first 3 bands
            bands = raw[:3] if raw.shape[0] >= 3 else np.repeat(raw[:1], 3, axis=0)
            rgb = bands.transpose(1, 2, 0)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

        return (rgb * 255).clip(0, 255).astype(np.uint8)

    # ──────────────────────────────────────────────────────────
    # UTILITIES
    # ──────────────────────────────────────────────────────────

    def _resize_array(self, arr: np.ndarray, size: int) -> np.ndarray:
        """Resize [C, H, W] array to [C, size, size] using PIL."""
        c = arr.shape[0]
        result = np.zeros((c, size, size), dtype=np.float32)
        for i in range(c):
            band_img = Image.fromarray(arr[i].astype(np.float32), mode="F")
            band_resized = band_img.resize((size, size), Image.BILINEAR)
            result[i] = np.array(band_resized)
        return result

    def _resize_rgb(self, rgb: np.ndarray, size: int) -> np.ndarray:
        """Resize [H, W, 3] uint8 to [size, size, 3]."""
        img = Image.fromarray(rgb)
        img = img.resize((size, size), Image.LANCZOS)
        return np.array(img)

    def _array_to_b64(self, rgb: np.ndarray, quality: int = 85) -> str:
        """Encode uint8 RGB array to base64 JPEG string."""
        img = Image.fromarray(rgb.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    def _extract_date(self, tags: dict) -> Optional[str]:
        """Extract acquisition date from GeoTIFF metadata tags."""
        date_keys = [
            "ACQUISITION_DATE", "DATE_ACQUIRED", "TIFFTAG_DATETIME",
            "SENSING_TIME", "StartTime", "acquisitionDate",
            "system:time_start"
        ]
        for key in date_keys:
            val = tags.get(key)
            if val:
                # Try to parse and standardize
                try:
                    # Handle various date formats
                    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y:%m:%d %H:%M:%S", "%Y-%m-%d"):
                        try:
                            return datetime.strptime(val[:19], fmt).strftime("%Y-%m-%d")
                        except ValueError:
                            continue
                    return str(val)[:10]
                except Exception:
                    return str(val)[:10]
        return None


# ──────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ──────────────────────────────────────────────────────────────
_loader_256 = None
_loader_224 = None


def get_loader(target_size: int = 256) -> GeoTIFFLoader:
    """Get or create a cached GeoTIFFLoader instance."""
    global _loader_256, _loader_224
    if target_size == 256:
        if _loader_256 is None:
            _loader_256 = GeoTIFFLoader(target_size=256)
        return _loader_256
    elif target_size == 224:
        if _loader_224 is None:
            _loader_224 = GeoTIFFLoader(target_size=224)
        return _loader_224
    return GeoTIFFLoader(target_size=target_size)
