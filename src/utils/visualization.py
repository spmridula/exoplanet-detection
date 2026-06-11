# src/utils/visualization.py  (replace the entire file)
"""
visualization.py
────────────────
Day 3: EDA visualization utilities.
All functions return matplotlib Figure objects — caller decides save/show.

Commit: eda: visualize stellar light curves
"""

from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


COLORS = {
    "planet":   "#EF9F27",   # amber  — confirmed planet
    "negative": "#185FA5",   # blue   — false positive / no planet
    "flux":     "#378ADD",   # mid blue — raw flux line
    "trend":    "#E24B4A",   # red    — trend line
    "transit":  "#EF9F27",   # amber  — transit markers
    "grid":     "#e8e8e8",
}


# ─── 1. Single light curve ─────────────────────────────────────────────────────

def plot_light_curve(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    title: str = "Light Curve",
    transit_times: Optional[List[float]] = None,
    zoom_window: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (14, 4),
) -> plt.Figure:
    """
    Plot a single star's light curve (raw or processed).

    Parameters
    ----------
    zoom_window : (t_start, t_end), optional
        If given, adds a zoomed inset showing one transit in detail.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if flux_err is not None:
        ax.errorbar(time, flux, yerr=flux_err,
                    fmt=",", color=COLORS["flux"], alpha=0.35,
                    linewidth=0.4, elinewidth=0.2, rasterized=True)
    else:
        ax.scatter(time, flux, s=0.4, color=COLORS["flux"],
                   alpha=0.5, rasterized=True)

    # Mark known transit times
    if transit_times:
        for t in transit_times:
            ax.axvline(t, color=COLORS["transit"], alpha=0.6,
                       linewidth=0.8, linestyle="--")
        ax.axvline(transit_times[0], color=COLORS["transit"],
                   linewidth=0.8, linestyle="--", label="Transit")
        ax.legend(fontsize=9, loc="upper right")

    # Zoomed inset
    if zoom_window is not None:
        t0, t1 = zoom_window
        mask = (time >= t0) & (time <= t1)
        if mask.sum() > 5:
            ax_inset = ax.inset_axes([0.72, 0.1, 0.25, 0.55])
            ax_inset.scatter(time[mask], flux[mask], s=3,
                             color=COLORS["transit"], alpha=0.8)
            ax_inset.set_title("Transit zoom", fontsize=8)
            ax_inset.tick_params(labelsize=7)
            ax_inset.grid(True, alpha=0.3)
            ax.indicate_inset_zoom(ax_inset, edgecolor=COLORS["transit"])

    ax.set_xlabel("Time (BKJD days)", fontsize=11)
    ax.set_ylabel("Normalized Flux", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="normal")
    ax.grid(True, alpha=0.25, linewidth=0.5, color=COLORS["grid"])
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ─── 2. Phase-folded light curve ──────────────────────────────────────────────

def plot_phase_fold(
    phase: np.ndarray,
    flux: np.ndarray,
    period: float,
    star_name: str = "",
    figsize: Tuple[int, int] = (10, 4),
) -> plt.Figure:
    """
    Plot a phase-folded light curve.
    The transit dip should appear as a sharp V or U shape near phase 0.5.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(phase, flux, s=0.6, color=COLORS["flux"],
               alpha=0.4, rasterized=True, label="Data")

    # Smooth overlay
    smooth = uniform_filter1d(flux, size=max(1, len(flux) // 80))
    ax.plot(phase, smooth, color=COLORS["planet"],
            linewidth=1.8, alpha=0.9, label="Smoothed")

    # Annotate the dip depth
    dip_idx = np.argmin(flux)
    dip_depth_ppm = (1.0 - flux[dip_idx]) * 1e6
    ax.annotate(
        f"Dip depth\n{dip_depth_ppm:,.0f} ppm",
        xy=(phase[dip_idx], flux[dip_idx]),
        xytext=(phase[dip_idx] + 0.08, flux[dip_idx] + 0.002),
        fontsize=8,
        color=COLORS["planet"],
        arrowprops=dict(arrowstyle="->", color=COLORS["planet"], lw=0.8),
    )

    ax.set_xlabel("Orbital Phase", fontsize=11)
    ax.set_ylabel("Normalized Flux", fontsize=11)
    name_str = f" — {star_name}" if star_name else ""
    ax.set_title(
        f"Phase-folded Light Curve{name_str}  |  Period = {period:.4f} days",
        fontsize=11, fontweight="normal"
    )
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, markerscale=5)
    plt.tight_layout()
    return fig


# ─── 3. Planet vs False Positive side-by-side ─────────────────────────────────

def plot_comparison(
    lc_planet: pd.DataFrame,
    lc_fp: pd.DataFrame,
    planet_name: str = "Confirmed Planet",
    fp_name: str = "False Positive",
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Side-by-side comparison of a confirmed planet host vs a false positive.
    This is the most important EDA plot — it shows the human eye what
    the model has to learn to distinguish.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=False)

    for ax, lc, name, color in zip(
        axes,
        [lc_planet, lc_fp],
        [planet_name, fp_name],
        [COLORS["planet"], COLORS["negative"]],
    ):
        ax.scatter(
            lc["time"], lc["flux"],
            s=0.3, color=color, alpha=0.5, rasterized=True
        )
        # Rolling median to show trend
        window = max(1, len(lc) // 200)
        rolling = lc["flux"].rolling(window, center=True).median()
        ax.plot(lc["time"], rolling, color=color,
                linewidth=0.8, alpha=0.7)

        ax.set_ylabel("Normalized Flux", fontsize=10)
        ax.set_title(name, fontsize=11, fontweight="normal", color=color)
        ax.grid(True, alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Time (BKJD days)", fontsize=11)
    fig.suptitle(
        "Planet Host vs False Positive — Can you see the difference?",
        fontsize=12, fontweight="normal", y=1.01
    )
    plt.tight_layout()
    return fig


# ─── 4. Class distribution ────────────────────────────────────────────────────

def plot_class_distribution(
    labels: np.ndarray,
    figsize: Tuple[int, int] = (7, 4),
) -> plt.Figure:
    """Bar chart showing class imbalance — the core ML challenge of this dataset."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    total = len(labels)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        ["False Positive\n(no planet)", "Confirmed\nExoplanet"],
        [n_neg, n_pos],
        color=[COLORS["negative"], COLORS["planet"]],
        width=0.45,
        edgecolor="white",
        linewidth=1.5,
    )
    for bar, count in zip(bars, [n_neg, n_pos]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.008,
            f"{count:,}\n({count/total*100:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )

    ratio = n_neg / max(n_pos, 1)
    ax.set_title(
        f"Class Imbalance  |  Ratio {ratio:.0f}:1  (negative:positive)",
        fontsize=12, fontweight="normal"
    )
    ax.set_ylabel("Number of KOIs", fontsize=11)
    ax.set_ylim(0, max(n_neg, n_pos) * 1.2)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ─── 5. Transit shape zoom ────────────────────────────────────────────────────

def plot_transit_zoom(
    time: np.ndarray,
    flux: np.ndarray,
    t_center: float,
    window_hours: float = 12.0,
    title: str = "Single Transit Event",
    figsize: Tuple[int, int] = (8, 4),
) -> plt.Figure:
    """
    Zoom into a single transit event to show the characteristic U/V dip shape.
    window_hours: how many hours either side of the transit center to show.
    """
    window_days = window_hours / 24.0
    mask = np.abs(time - t_center) <= window_days
    t_zoom = time[mask]
    f_zoom = flux[mask]

    if len(t_zoom) < 3:
        raise ValueError(f"Not enough points near t={t_center:.2f} — check transit time")

    # Convert to hours from center for readability
    t_hours = (t_zoom - t_center) * 24.0

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(t_hours, f_zoom, s=15, color=COLORS["flux"],
               alpha=0.8, zorder=3, label="Data points")

    # Smooth fit for the dip shape
    if len(f_zoom) > 10:
        smooth = uniform_filter1d(f_zoom, size=max(1, len(f_zoom) // 10))
        ax.plot(t_hours, smooth, color=COLORS["planet"],
                linewidth=2, alpha=0.9, label="Transit shape", zorder=4)

    # Baseline reference
    baseline = np.median(f_zoom[:5].tolist() + f_zoom[-5:].tolist())
    ax.axhline(baseline, color="gray", linewidth=0.8,
               linestyle="--", alpha=0.6, label="Baseline flux")

    # Depth annotation
    dip_depth = baseline - f_zoom.min()
    dip_ppm = dip_depth * 1e6
    ax.annotate(
        f"{dip_ppm:,.0f} ppm\n({dip_depth*100:.3f}%)",
        xy=(0, f_zoom.min()),
        xytext=(window_hours * 0.4, f_zoom.min() + dip_depth * 0.3),
        fontsize=9, color=COLORS["planet"],
        arrowprops=dict(arrowstyle="->", color=COLORS["planet"], lw=1),
    )

    ax.set_xlabel("Hours from transit center", fontsize=11)
    ax.set_ylabel("Normalized Flux", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="normal")
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


# ─── 6. Flux statistics summary ───────────────────────────────────────────────

def plot_flux_stats(
    light_curves: List[pd.DataFrame],
    labels: List[int],
    figsize: Tuple[int, int] = (12, 4),
) -> plt.Figure:
    """
    Distribution plots of flux statistics split by class.
    Shows whether simple statistics (std, skew) are discriminative.
    This motivates why we need ML rather than just thresholds.
    """
    from scipy import stats as sp_stats

    records = []
    for lc, label in zip(light_curves, labels):
        flux = lc["flux"].values
        records.append({
            "label": label,
            "std":   flux.std(),
            "skew":  float(sp_stats.skew(flux)),
            "kurt":  float(sp_stats.kurtosis(flux)),
        })
    df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    metrics = [("std", "Flux Std Dev"), ("skew", "Flux Skewness"), ("kurt", "Flux Kurtosis")]

    for ax, (col, label) in zip(axes, metrics):
        for cls, color, name in [
            (1, COLORS["planet"], "Planet"),
            (0, COLORS["negative"], "No planet"),
        ]:
            vals = df[df["label"] == cls][col].dropna()
            if len(vals) > 1:
                ax.hist(vals, bins=15, color=color, alpha=0.55,
                        label=name, edgecolor="white", linewidth=0.5)

        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Flux Statistics by Class — Do simple features separate the classes?",
        fontsize=11, fontweight="normal"
    )
    plt.tight_layout()
    return fig