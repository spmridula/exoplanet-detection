import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')

from src.data.feature_engineer import FeatureExtractor, build_feature_matrix


def make_flat_lc(n=500, seed=0):
    """Pure noise, no transit — represents a 'boring' star."""
    np.random.seed(seed)
    time = np.linspace(0, 100, n)
    flux = 1.0 + np.random.normal(0, 0.001, n)
    return pd.DataFrame({"time": time, "flux": flux})


def make_transit_lc(n=2000, period=10.0, depth=0.01, seed=0):
    """Synthetic light curve with periodic transits — represents a planet host."""
    np.random.seed(seed)
    time = np.linspace(0, 100, n)
    flux = 1.0 + np.random.normal(0, 0.0005, n)

    # Add a transit dip every `period` days
    t = 0.0
    while t < time[-1]:
        dip_mask = np.abs(time - t) < 0.3
        flux[dip_mask] -= depth
        t += period

    return pd.DataFrame({"time": time, "flux": flux})


class TestFeatureExtractor:
    def setup_method(self):
        self.fe = FeatureExtractor()

    def test_global_stats_present(self):
        lc = make_flat_lc()
        feats = self.fe.extract(lc)
        for key in ["flux_mean", "flux_std", "flux_skew", "flux_kurtosis"]:
            assert key in feats
            assert np.isfinite(feats[key])

    def test_flat_curve_has_no_dips(self):
        """Flat noisy curve — adaptive threshold should suppress noise peaks."""
        lc = make_flat_lc(n=500, seed=1)
        feats = self.fe.extract(lc)
        # At 4.5x std threshold, Gaussian noise rarely produces significant peaks
        assert feats["n_dips_detected"] <= 3

    def test_transit_curve_detects_dips(self):
        """Deep, dense transits with low noise should be clearly detected."""
        lc = make_transit_lc(n=2000, period=10.0, depth=0.02, seed=0)
        feats = self.fe.extract(lc)
        assert feats["n_dips_detected"] >= 5
        # depth=0.02 → 20,000 ppm; prominence may be less due to noise baseline
        assert feats["dip_mean_depth"] > 0.003

    def test_folded_features_with_period(self):
        lc = make_transit_lc(period=10.0, depth=0.015)
        feats = self.fe.extract(lc, period=10.0, t0=0.0, n_bins=100)
        assert feats["folded_transit_depth"] > 0
        assert feats["folded_transit_depth_ppm"] > 5000  # ~15000 ppm expected
        assert feats["period_days"] == 10.0

    def test_folded_features_nan_without_period(self):
        lc = make_transit_lc()
        feats = self.fe.extract(lc, period=None, t0=None)
        assert np.isnan(feats["folded_transit_depth"])

    def test_transit_vs_flat_dip_depth_separates_classes(self):
        """Planet curve (deep dips) must have larger mean dip than flat curve."""
        flat    = self.fe.extract(make_flat_lc(n=2000, seed=42))
        transit = self.fe.extract(make_transit_lc(n=2000, depth=0.03, seed=42))
        assert transit["dip_mean_depth"] > flat["dip_mean_depth"]


class TestFeatureMatrix:
    def test_build_matrix_shape(self):
        lcs = [make_flat_lc(seed=i) for i in range(3)] + \
              [make_transit_lc(seed=i) for i in range(3)]
        labels = [0, 0, 0, 1, 1, 1]
        df = build_feature_matrix(lcs, labels)

        assert len(df) == 6
        assert "label" in df.columns
        assert df["label"].sum() == 3

    def test_matrix_has_no_unexpected_nans_without_period(self):
        """Global/shape/peak features should always be present;
        only folded_* features are NaN without period info."""
        lcs = [make_flat_lc(seed=0)]
        df = build_feature_matrix(lcs, [0])

        non_folded_cols = [c for c in df.columns if not c.startswith("folded_")
                            and c not in ("period_days", "log_period", "label")]
        assert df[non_folded_cols].isna().sum().sum() == 0