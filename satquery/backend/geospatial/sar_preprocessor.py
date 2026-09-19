"""
SatQuery AI — SAR Preprocessing Pipeline
==========================================
Applies standard SAR-specific preprocessing before model inference.
Missing from all original documents — added as per audit gap analysis.

Pipeline:
  Step 1: Calibration to sigma0 (dB conversion if raw linear)
  Step 2: Speckle filtering (Lee filter via scipy)
  Step 3: Valid range clipping (-25 dB to 0 dB)
  Step 4: Normalization to [0, 1]
  Step 5: VV/VH 2-channel tensor stacking → shape [2, H, W]
"""

import logging
from typing import Tuple, Dict, Any, Optional

import numpy as np

# Optional scipy for Lee filter
try:
    from scipy.ndimage import uniform_filter, variance as ndimage_variance
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not installed. Speckle filtering disabled (Lee filter unavailable).")

logger = logging.getLogger(__name__)

# SAR valid intensity range in dB
SAR_DB_MIN = -25.0
SAR_DB_MAX = 0.0

# Lee filter window size (3×3 recommended for Sentinel-1)
LEE_WINDOW_SIZE = 3


class SARPreprocessor:
    """
    Prepares SAR imagery for model inference.

    Input:  numpy array [C, H, W] — raw SAR data (linear or dB scale)
    Output: numpy array [2, H, W] — normalized [0, 1] VV+VH stack
    """

    def __init__(
        self,
        apply_lee_filter: bool = True,
        lee_window: int = LEE_WINDOW_SIZE,
        db_min: float = SAR_DB_MIN,
        db_max: float = SAR_DB_MAX,
    ):
        self.apply_lee = apply_lee_filter and SCIPY_AVAILABLE
        self.lee_window = lee_window
        self.db_min = db_min
        self.db_max = db_max

    def preprocess(
        self,
        raw_array: np.ndarray,
        polarization_order: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Full SAR preprocessing pipeline.

        Args:
            raw_array: [C, H, W] float32. Bands may be in any order.
            polarization_order: (vv_idx, vh_idx) if known. Defaults to (0, 1).

        Returns:
            {
                "tensor": np.ndarray [2, H, W]  — model-ready normalized VV/VH,
                "vv_db": np.ndarray [H, W]      — VV in dB (for display),
                "vh_db": np.ndarray [H, W]      — VH in dB (for display),
                "stats": dict                   — min/max/mean of each channel,
                "applied_steps": List[str]      — which preprocessing steps ran,
            }
        """
        applied_steps = []
        vv_idx, vh_idx = polarization_order if polarization_order else (0, 1)

        # Step 0: Extract VV and VH
        n_bands = raw_array.shape[0]
        vv = raw_array[min(vv_idx, n_bands - 1)].astype(np.float64)
        vh = raw_array[min(vh_idx, n_bands - 1)].astype(np.float64) if n_bands > 1 else vv.copy()
        applied_steps.append("band_extraction")

        # Step 1: Convert to dB if data appears to be linear power
        vv = self._to_db(vv, applied_steps, "VV")
        vh = self._to_db(vh, applied_steps, "VH")

        # Step 2: Lee speckle filter
        if self.apply_lee:
            vv = self._lee_filter(vv)
            vh = self._lee_filter(vh)
            applied_steps.append(f"lee_filter_{self.lee_window}x{self.lee_window}")
        else:
            applied_steps.append("lee_filter_skipped")

        # Step 3: Clip to valid SAR range
        vv_clipped = np.clip(vv, self.db_min, self.db_max)
        vh_clipped = np.clip(vh, self.db_min, self.db_max)
        applied_steps.append(f"range_clip_{self.db_min}dB_to_{self.db_max}dB")

        # Step 4: Normalize to [0, 1]
        vv_norm = (vv_clipped - self.db_min) / (self.db_max - self.db_min)
        vh_norm = (vh_clipped - self.db_min) / (self.db_max - self.db_min)
        applied_steps.append("normalize_0_1")

        # Step 5: Stack as [2, H, W] tensor
        tensor = np.stack([vv_norm, vh_norm], axis=0).astype(np.float32)
        applied_steps.append("stack_vv_vh_tensor")

        stats = {
            "vv_min_db": float(vv_clipped.min()),
            "vv_max_db": float(vv_clipped.max()),
            "vv_mean_db": float(vv_clipped.mean()),
            "vh_min_db": float(vh_clipped.min()),
            "vh_max_db": float(vh_clipped.max()),
            "vh_mean_db": float(vh_clipped.mean()),
        }

        return {
            "tensor": tensor,
            "vv_db": vv_clipped.astype(np.float32),
            "vh_db": vh_clipped.astype(np.float32),
            "stats": stats,
            "applied_steps": applied_steps,
        }

    # ──────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ──────────────────────────────────────────────────────────

    def _to_db(
        self, band: np.ndarray, applied_steps: list, name: str
    ) -> np.ndarray:
        """
        Convert to dB if the band appears to be in linear power scale.
        BigEarthNet S1 is already in dB. Raw SNAP output may be linear.

        Detection heuristic:
          - If values are in [-30, 10]: assume already dB → skip
          - If values are in [0, 1] (very small floats): linear → convert
          - If values are large integers: linear DN → convert
        """
        band_min, band_max = float(band.min()), float(band.max())

        if band_max <= 1.0 and band_min >= 0.0:
            # Linear power values [0, 1] — likely normalized by satellite provider
            # Convert: sigma0_dB = 10 * log10(sigma0)
            band_db = np.where(band > 1e-10, 10.0 * np.log10(band + 1e-10), -30.0)
            applied_steps.append(f"{name}_linear_to_db_conversion")
            logger.info(f"SAR {name}: Converted from linear [0,1] to dB.")
            return band_db

        elif band_max > 10.0:
            # Large linear DN values (e.g., amplitude from SNAP)
            # square first: power = amplitude² → then dB
            power = band ** 2
            band_db = np.where(power > 0, 10.0 * np.log10(power + 1e-10), -30.0)
            applied_steps.append(f"{name}_amplitude_to_db_conversion")
            logger.info(f"SAR {name}: Converted from amplitude DN to dB.")
            return band_db

        else:
            # Already in dB range (roughly -30 to +10)
            applied_steps.append(f"{name}_already_in_db")
            return band

    def _lee_filter(self, band_db: np.ndarray) -> np.ndarray:
        """
        Lee speckle filter implementation using scipy uniform_filter.
        The Lee filter estimates the filtered value using local mean and variance.

        Formula:
            output = mean + k * (input - mean)
            where k = var_signal / (var_signal + var_noise)
            and   var_signal = max(0, local_var - var_noise)
        """
        # Local mean (equivalent to box filter)
        mean = uniform_filter(band_db, size=self.lee_window)

        # Local variance: E[X²] - E[X]²
        mean_sq = uniform_filter(band_db ** 2, size=self.lee_window)
        local_var = mean_sq - mean ** 2
        local_var = np.maximum(local_var, 0.0)  # numerical stability

        # Estimate noise variance from global statistics
        noise_var = np.mean(local_var)
        noise_var = max(noise_var, 1e-8)

        # Weighting factor k
        k = np.where(
            local_var + noise_var > 0,
            local_var / (local_var + noise_var),
            0.0
        )

        # Apply Lee filter
        filtered = mean + k * (band_db - mean)
        return filtered.astype(band_db.dtype)

    def make_false_color_preview(
        self, vv_db: np.ndarray, vh_db: np.ndarray
    ) -> np.ndarray:
        """
        Generate a SAR false-color composite for display.
        Channel mapping:
          R = VV (normalized)
          G = VH (normalized)
          B = VV - VH ratio (normalized) — highlights contrast
        Returns: uint8 [H, W, 3]
        """
        def norm(x):
            mn, mx = self.db_min, self.db_max
            return np.clip((x - mn) / (mx - mn), 0, 1)

        vv_n = norm(vv_db)
        vh_n = norm(vh_db)
        ratio = norm(np.clip(vv_db - vh_db, self.db_min, self.db_max))

        rgb = np.stack([vv_n, vh_n, ratio], axis=-1)
        return (rgb * 255).clip(0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ──────────────────────────────────────────────────────────────
_preprocessor: Optional[SARPreprocessor] = None


def get_preprocessor(apply_lee: bool = True) -> SARPreprocessor:
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = SARPreprocessor(apply_lee_filter=apply_lee)
    return _preprocessor
