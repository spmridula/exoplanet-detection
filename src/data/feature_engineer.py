# src/data/feature_engineer.py
"""
feature_engineer.py
────────────────────
Day 4: Extracts handcrafted features from light curves for classical ML models.

These features feed into Random Forest / XGBoost in Week 2. Each feature
captures something a domain expert would look at: dip depth, periodicity,
shape statistics, etc.

Commit: feature: handcrafted feature extraction pipeline
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.signal import find_peaks

from src.data.preprocessor import phase_fold

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts a fixed-length feature vector from a preprocessed light curve.

    Two feature groups:
      1. Global statistics  — describe the whole light curve
      2. Folded-curve shape  — describe the transit dip (requires period + t0)

    Example
    -------
        fe = FeatureExtractor()
        features = fe.extract(lc_clean, period=289.86, t0=170.5)
        # features is a dict of ~20 named scalar values
    """

    def extract(
        self,
        lc: pd.DataFrame,
        period: Optional[float] = None,
        t0: Optional[float] = None,
        n_bins: int = 200,
    ) -> dict:
        """
        Extract all features from a single light curve.

        Parameters
        ----------
        lc : pd.DataFrame
            Preprocessed light curve with 'time' and 'flux' columns.
        period, t0 : float, optional
            Orbital period and reference transit time. If provided,
            folded-curve features are also computed.
        n_bins : int
            Bins for phase folding (smaller than the 2000 used for CNN —
            classical models need fewer, more aggregated features).

        Returns
        -------
        dict of feature_name -> float
        """
        flux = lc["flux"].values
        time = lc["time"].values

        features = {}
        features.update(self._global_stats(flux))
        features.update(self._shape_stats(flux))
        features.update(self._peak_stats(flux, time))

        if period is not None and t0 is not None:
            features.update(self._folded_stats(time, flux, period, t0, n_bins))
        else:
            # Fill with NaN so the feature vector stays fixed-length
            for k in self._folded_feature_names():
                features[k] = np.nan

        return features

    # ─── Feature groups ─────────────────────────────────────────────────────

    def _global_stats(self, flux: np.ndarray) -> dict:
        """Basic distributional statistics of the flux time series."""
        return {
            "flux_mean":   float(np.mean(flux)),
            "flux_std":    float(np.std(flux)),
            "flux_median": float(np.median(flux)),
            "flux_min":    float(np.min(flux)),
            "flux_max":    float(np.max(flux)),
            "flux_range":  float(np.max(flux) - np.min(flux)),
            "flux_mad":    float(sp_stats.median_abs_deviation(flux)),
        }

    def _shape_stats(self, flux: np.ndarray) -> dict:
        """
        Distribution shape — transits create negative skew
        (long tail toward lower flux values from dips).
        """
        return {
            "flux_skew": float(sp_stats.skew(flux)),
            "flux_kurtosis": float(sp_stats.kurtosis(flux)),
            # Fraction of points below 1 - 3*std — proxy for "how often does it dip"
            "frac_below_3sigma": float(
                np.mean(flux < (np.mean(flux) - 3 * np.std(flux)))
            ),
        }

    def _peak_stats(self, flux: np.ndarray, time: np.ndarray) -> dict:
        """
        Detects negative peaks (dips) in the light curve using scipy.
        A real transiting planet produces multiple similar-depth dips.
        """
        # Invert flux so dips become peaks
        inverted = -flux
        # Minimum prominence: dip must be at least 0.0005 (500 ppm) deep
        peaks, props = find_peaks(inverted, prominence=0.0005, distance=10)

        n_dips = len(peaks)
        if n_dips > 0:
            depths = props["prominences"]
            mean_depth = float(np.mean(depths))
            std_depth = float(np.std(depths)) if n_dips > 1 else 0.0
            max_depth = float(np.max(depths))

            # Consistency of dip depths — real planets have similar-depth transits
            # Eclipsing binaries often show two different depths (primary/secondary)
            depth_cv = std_depth / mean_depth if mean_depth > 0 else 0.0
        else:
            mean_depth = std_depth = max_depth = depth_cv = 0.0

        return {
            "n_dips_detected": float(n_dips),
            "dip_mean_depth": mean_depth,
            "dip_depth_std": std_depth,
            "dip_max_depth": max_depth,
            "dip_depth_consistency": depth_cv,  # lower = more consistent = more planet-like
        }

    def _folded_stats(
        self, time: np.ndarray, flux: np.ndarray,
        period: float, t0: float, n_bins: int,
    ) -> dict:
        """
        Features from the phase-folded light curve.
        This is where the orbital period information gets used.
        """
        phases, binned_flux = phase_fold(time, flux, period, t0, n_bins)

        baseline = np.median(binned_flux)
        min_flux = np.min(binned_flux)
        transit_depth = baseline - min_flux

        # Transit width: how many bins are "in transit" (below baseline - 0.5*depth)
        in_transit = binned_flux < (baseline - 0.5 * transit_depth)
        transit_width_frac = float(np.mean(in_transit))

        # Symmetry: compare flux just before vs just after the dip center
        dip_center_idx = np.argmin(binned_flux)
        window = max(1, n_bins // 20)
        left = binned_flux[max(0, dip_center_idx - window):dip_center_idx]
        right = binned_flux[dip_center_idx + 1:dip_center_idx + 1 + window]
        if len(left) > 0 and len(right) > 0:
            symmetry = float(np.abs(np.mean(left) - np.mean(right)))
        else:
            symmetry = 0.0

        return {
            "folded_transit_depth": float(transit_depth),
            "folded_transit_depth_ppm": float(transit_depth * 1e6),
            "folded_transit_width_frac": transit_width_frac,
            "folded_symmetry": symmetry,
            "period_days": float(period),
            "log_period": float(np.log10(period)) if period > 0 else 0.0,
        }

    def _folded_feature_names(self) -> list:
        return [
            "folded_transit_depth", "folded_transit_depth_ppm",
            "folded_transit_width_frac", "folded_symmetry",
            "period_days", "log_period",
        ]

    def feature_names(self) -> list:
        """Returns the full ordered list of feature names this extractor produces."""
        return (
            list(self._global_stats(np.zeros(10)).keys())
            + list(self._shape_stats(np.zeros(10)).keys())
            + list(self._peak_stats(np.zeros(10), np.zeros(10)).keys())
            + self._folded_feature_names()
        )


def build_feature_matrix(
    light_curves: list[pd.DataFrame],
    labels: list[int],
    periods: Optional[list[float]] = None,
    t0s: Optional[list[float]] = None,
) -> pd.DataFrame:
    """
    Build a feature matrix (DataFrame) from a list of light curves.

    Parameters
    ----------
    light_curves : list of preprocessed DataFrames
    labels : list of 0/1 labels
    periods, t0s : list of orbital periods / transit times, optional
                   (None entries are allowed — folded features become NaN)

    Returns
    -------
    pd.DataFrame, one row per star, columns = features + 'label'

    Example
    -------
        X = build_feature_matrix(light_curves, labels, periods, t0s)
        X.to_csv("data/processed/features.csv", index=False)
    """
    fe = FeatureExtractor()
    rows = []

    periods = periods or [None] * len(light_curves)
    t0s = t0s or [None] * len(light_curves)

    for lc, label, period, t0 in zip(light_curves, labels, periods, t0s):
        feats = fe.extract(lc, period=period, t0=t0)
        feats["label"] = label
        rows.append(feats)

    df = pd.DataFrame(rows)
    logger.info(f"Feature matrix built: {df.shape[0]} rows x {df.shape[1]-1} features")
    return df