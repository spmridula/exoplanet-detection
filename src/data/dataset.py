import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocessor import phase_fold

logger = logging.getLogger(__name__)


class ExoplanetLightCurveDataset(Dataset):
    """
    PyTorch Dataset of phase-folded light curves for CNN/LSTM models.

    Each sample is a fixed-length 1D array (the phase-folded, binned flux)
    paired with a binary label (1 = confirmed planet, 0 = false positive).

    Parameters
    ----------
    light_curves : list of pd.DataFrame
        Each must have 'time' and 'flux' columns (preprocessed).
    labels : list of int
        0 or 1, same length as light_curves.
    periods : list of float
        Orbital period (days) for each light curve — required for folding.
    t0s : list of float
        Reference transit time for each light curve.
    n_bins : int
        Length of the folded curve fed to the model. Default: 2000.
    augment : bool
        If True, applies light augmentation (Day 18 will extend this).
        For now: random circular shift of the phase axis.

    Example
    -------
        ds = ExoplanetLightCurveDataset(light_curves, labels, periods, t0s)
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        for x, y in loader:
            # x.shape == (batch, 1, 2000)  — channel-first for 1D-CNN
            # y.shape == (batch,)
            ...
    """

    def __init__(
        self,
        light_curves: list,
        labels: list,
        periods: list,
        t0s: list,
        n_bins: int = 2000,
        augment: bool = False,
    ):
        assert len(light_curves) == len(labels) == len(periods) == len(t0s), \
            "light_curves, labels, periods, and t0s must be the same length"

        self.light_curves = light_curves
        self.labels = labels
        self.periods = periods
        self.t0s = t0s
        self.n_bins = n_bins
        self.augment = augment

        # Pre-fold all curves at init time — avoids redundant computation
        # every epoch (folding is the expensive step).
        self._folded_cache = self._precompute_folds()

    def _precompute_folds(self) -> list:
        """Phase-fold every light curve once and cache the result."""
        folded = []
        for lc, period, t0 in zip(self.light_curves, self.periods, self.t0s):
            _, binned_flux = phase_fold(
                lc["time"].values, lc["flux"].values,
                period=period, t0=t0, n_bins=self.n_bins,
            )
            folded.append(binned_flux.astype(np.float32))
        return folded

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        flux = self._folded_cache[idx].copy()

        if self.augment:
            # Random circular shift — the model shouldn't depend on
            # where in the phase array the dip happens to land
            shift = np.random.randint(0, self.n_bins)
            flux = np.roll(flux, shift)

        # Shape: (1, n_bins) — single channel for 1D-CNN input
        x = torch.from_numpy(flux).float().unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

    def class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling imbalance in the loss function.

        Usage with BCEWithLogitsLoss:
            pos_weight = ds.class_weights()
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        """
        labels = np.array(self.labels)
        n_pos = (labels == 1).sum()
        n_neg = (labels == 0).sum()
        pos_weight = n_neg / max(n_pos, 1)
        return torch.tensor([pos_weight], dtype=torch.float32)


def make_synthetic_dataset(n_samples: int = 100, n_bins: int = 2000, seed: int = 42):
    """
    Generate a synthetic dataset for testing the Dataset class without
    needing real NASA downloads. Useful for CI and quick iteration.

    Returns
    -------
    ExoplanetLightCurveDataset
    """
    np.random.seed(seed)
    light_curves, labels, periods, t0s = [], [], [], []

    for i in range(n_samples):
        has_planet = i % 5 == 0  # 20% positive class
        n = 1000
        time = np.linspace(0, 100, n)
        flux = 1.0 + np.random.normal(0, 0.001, n)

        period = np.random.uniform(5, 30)
        t0 = np.random.uniform(0, period)

        if has_planet:
            depth = np.random.uniform(0.005, 0.02)
            t = t0
            while t < time[-1]:
                dip = np.abs(time - t) < 0.3
                flux[dip] -= depth
                t += period

        light_curves.append(pd.DataFrame({"time": time, "flux": flux}))
        labels.append(int(has_planet))
        periods.append(period)
        t0s.append(t0)

    return ExoplanetLightCurveDataset(light_curves, labels, periods, t0s, n_bins=n_bins)