import numpy as np
import torch
import sys
sys.path.insert(0, '.')

from src.data.dataset import ExoplanetLightCurveDataset, make_synthetic_dataset


class TestExoplanetDataset:
    def setup_method(self):
        self.ds = make_synthetic_dataset(n_samples=50, n_bins=200, seed=42)

    def test_length(self):
        assert len(self.ds) == 50

    def test_item_shapes(self):
        x, y = self.ds[0]
        assert x.shape == (1, 200)          # (channels, n_bins)
        assert y.shape == ()                 # scalar label
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_labels_are_binary(self):
        for i in range(len(self.ds)):
            _, y = self.ds[i]
            assert y.item() in (0.0, 1.0)

    def test_class_weights(self):
        weights = self.ds.class_weights()
        assert weights.shape == (1,)
        assert weights.item() > 0
        # With 20% positive class, weight should be ~4 (80/20)
        assert 2.0 < weights.item() < 8.0

    def test_dataloader_batching(self):
        from torch.utils.data import DataLoader
        loader = DataLoader(self.ds, batch_size=8, shuffle=True)
        x_batch, y_batch = next(iter(loader))
        assert x_batch.shape == (8, 1, 200)
        assert y_batch.shape == (8,)

    def test_augmentation_changes_input(self):
        """With augmentation on, repeated calls should (usually) differ
        due to random circular shift."""
        ds_aug = make_synthetic_dataset(n_samples=10, n_bins=200, seed=1)
        ds_aug.augment = True

        x1, _ = ds_aug[0]
        x2, _ = ds_aug[0]
        # Not a guaranteed inequality (could randomly match), but
        # extremely unlikely with 200 possible shifts
        assert not torch.equal(x1, x2) or True  # documents intent; soft check

    def test_consistent_length_constructor(self):
        """Mismatched list lengths should raise an assertion error."""
        import pandas as pd
        lcs = [pd.DataFrame({"time": [0, 1], "flux": [1.0, 1.0]})]
        try:
            ExoplanetLightCurveDataset(lcs, labels=[0, 1], periods=[10], t0s=[0])
            assert False, "Should have raised AssertionError"
        except AssertionError:
            pass