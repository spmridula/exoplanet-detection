# src/data/downloader.py
"""
downloader.py
─────────────
Day 2: Full Kepler DR25 dataset download + catalog fetch from NASA IPAC.

Two things this module does:
  1. download_catalog()  — fetches the full Kepler DR25 label table from NASA
                           (~9,564 KOIs with CONFIRMED / FALSE POSITIVE labels)
  2. LightCurveDownloader — downloads individual star light curves via lightkurve

Commit: data: fetch Kepler light curve data
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ─── NASA IPAC Exoplanet Archive — DR25 KOI table ─────────────────────────────
# This is the official labeled dataset. Each row is one Kepler Object of Interest.
# koi_disposition: CONFIRMED / FALSE POSITIVE / CANDIDATE
IPAC_API_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+kepid,kepoi_name,koi_disposition,koi_period,"
    "koi_time0bk,koi_duration,koi_depth,koi_prad,koi_teq,koi_steff,"
    "koi_slogg,koi_srad+from+cumulative"
    "&format=csv"
)

# Small curated sample for quick testing (no internet needed for unit tests)
SAMPLE_KOIS = {
    "confirmed": [
        "Kepler-22",     # First Kepler habitable-zone planet — Kepler-22b
        "Kepler-442",    # Super-Earth, rocky, habitable zone
        "Kepler-452",    # Earth's closest twin found by Kepler
        "Kepler-186",    # First Earth-sized planet in habitable zone
        "Kepler-62",     # System with two habitable-zone planets
    ],
    "false_positive": [
        "KIC 6922244",   # Eclipsing binary — classic transit mimic
        "KIC 5088536",
        "KIC 4851217",
        "KIC 3542116",
        "KIC 7670943",
    ],
}


def download_catalog(
    cache_path: str = "data/raw/kepler_dr25_catalog.csv",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download the full Kepler DR25 KOI catalog from NASA IPAC.

    This is the ground-truth label table for the entire project.
    Contains ~9,564 Kepler Objects of Interest with disposition labels.

    Parameters
    ----------
    cache_path : str
        Where to save/load the catalog CSV.
    use_cache : bool
        If True, load from disk if file exists. Default: True.

    Returns
    -------
    pd.DataFrame with columns:
        kepid           — Kepler Input Catalog ID (star ID)
        kepoi_name      — KOI designation (e.g. K00001.01)
        koi_disposition — CONFIRMED / FALSE POSITIVE / CANDIDATE
        koi_period      — orbital period (days)
        koi_time0bk     — first transit time (BKJD)
        koi_duration    — transit duration (hours)
        koi_depth       — transit depth (ppm — parts per million)
        koi_prad        — planet radius (Earth radii)
        ... + stellar parameters

    Usage
    -----
        df = download_catalog()
        confirmed = df[df["koi_disposition"] == "CONFIRMED"]
        print(f"{len(confirmed)} confirmed exoplanets")
    """
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and cache_file.exists():
        logger.info(f"Loading catalog from cache: {cache_path}")
        df = pd.read_csv(cache_path, comment="#")
        logger.info(f"Catalog loaded: {len(df):,} KOIs")
        return df

    logger.info("Downloading Kepler DR25 catalog from NASA IPAC...")
    logger.info("This is ~2 MB and takes a few seconds.")

    try:
        resp = requests.get(IPAC_API_URL, timeout=60)
        resp.raise_for_status()

        # NASA IPAC adds comment lines starting with '#' — skip them
        lines = [l for l in resp.text.splitlines() if not l.startswith("#")]
        from io import StringIO
        df = pd.read_csv(StringIO("\n".join(lines)))

        # Clean up: keep only CONFIRMED and FALSE POSITIVE (drop CANDIDATE)
        df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])]
        df = df.reset_index(drop=True)

        df.to_csv(cache_path, index=False)
        logger.info(f"Catalog saved: {len(df):,} KOIs → {cache_path}")
        logger.info(
            f"  Confirmed planets : {(df['koi_disposition']=='CONFIRMED').sum():,}"
        )
        logger.info(
            f"  False positives   : {(df['koi_disposition']=='FALSE POSITIVE').sum():,}"
        )
        return df

    except requests.RequestException as e:
        logger.error(f"Failed to download catalog: {e}")
        logger.error("Check your internet connection or try again later.")
        raise


class LightCurveDownloader:
    """
    Downloads and caches individual Kepler light curves from NASA MAST.

    Uses the `lightkurve` library which handles the MAST API calls,
    quarter stitching, and file caching automatically.

    Parameters
    ----------
    cache_dir : str
        Directory to cache downloaded FITS files.
    mission : str
        "Kepler" or "TESS". Default: "Kepler".

    Example
    -------
        dl = LightCurveDownloader()
        lc_df = dl.download_by_kepid(10593626)   # Kepler-22
        print(lc_df.head())
    """

    def __init__(self, cache_dir: str = "data/cache", mission: str = "Kepler"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mission = mission

    def download_by_name(self, star_name: str) -> Optional[pd.DataFrame]:
        """
        Download light curve by star name (e.g. "Kepler-22", "KIC 10593626").
        Returns DataFrame with columns: time, flux, flux_err
        """
        return self._download(star_name, star_name.replace(" ", "_"))

    def download_by_kepid(self, kepid: int) -> Optional[pd.DataFrame]:
        """
        Download light curve by Kepler Input Catalog ID (integer).
        This is the primary key used in the DR25 catalog.
        """
        return self._download(f"KIC {kepid}", f"KIC_{kepid}")

    def _download(self, query: str, cache_key: str) -> Optional[pd.DataFrame]:
        """Internal download + cache logic."""
        try:
            import lightkurve as lk
        except ImportError:
            raise ImportError("Install with: pip install lightkurve")

        cache_file = self.cache_dir / f"{cache_key}.csv"
        if cache_file.exists():
            logger.debug(f"Cache hit: {cache_key}")
            return pd.read_csv(cache_file)

        logger.info(f"Downloading: {query}")
        try:
            search = lk.search_lightcurve(query, mission=self.mission, cadence="long")
            if len(search) == 0:
                logger.warning(f"No light curves found for: {query}")
                return None

            lc = search.download_all().stitch()
            df = pd.DataFrame({
                "time":     lc.time.value,
                "flux":     lc.flux.value,
                "flux_err": lc.flux_err.value
                              if lc.flux_err is not None
                              else np.zeros(len(lc.flux)),
            })
            df.to_csv(cache_file, index=False)
            logger.info(f"  {len(df):,} points | {df['time'].min():.1f}–{df['time'].max():.1f} days")
            return df

        except Exception as e:
            logger.error(f"Failed {query}: {e}")
            return None

    def download_batch(
        self,
        catalog_df: pd.DataFrame,
        n: int = 50,
        delay_sec: float = 0.5,
    ) -> list[dict]:
        """
        Download light curves for the first `n` stars in a catalog DataFrame.

        Parameters
        ----------
        catalog_df : pd.DataFrame
            Must have 'kepid' and 'koi_disposition' columns (from download_catalog).
        n : int
            How many stars to download. Start small (50), scale up later.
        delay_sec : float
            Pause between requests to be polite to NASA's servers.

        Returns
        -------
        List of dicts: [{"kepid": int, "label": int, "lc": DataFrame}, ...]
        """
        results = []
        rows = catalog_df.head(n)

        for i, row in rows.iterrows():
            kepid = int(row["kepid"])
            label = 1 if row["koi_disposition"] == "CONFIRMED" else 0

            lc = self.download_by_kepid(kepid)
            if lc is not None:
                results.append({"kepid": kepid, "label": label, "lc": lc})

            if i % 10 == 0:
                logger.info(f"  Progress: {len(results)}/{n} downloaded")

            time.sleep(delay_sec)  # respect NASA's rate limits

        confirmed_count = sum(r["label"] for r in results)
        logger.info(
            f"Batch complete: {len(results)} stars "
            f"({confirmed_count} planets, {len(results)-confirmed_count} negatives)"
        )
        return results


def download_sample() -> None:
    """Quick smoke test — downloads one star and prints summary. No NASA catalog needed."""
    logging.basicConfig(level=logging.INFO)
    dl = LightCurveDownloader()
    lc = dl.download_by_name("Kepler-22")
    if lc is not None:
        print(f"\n✅ Kepler-22 downloaded successfully")
        print(f"   Time steps : {len(lc):,}")
        print(f"   Time range : {lc['time'].min():.1f} – {lc['time'].max():.1f} days")
        print(f"   Flux (mean): {lc['flux'].mean():.4f}")
        print(f"   Flux (std) : {lc['flux'].std():.6f}")
    else:
        print("❌ Download failed — check your internet connection")


if __name__ == "__main__":
    # Run as: python -m src.data.downloader
    # Step 1: download catalog
    catalog = download_catalog()
    print(f"\nCatalog shape: {catalog.shape}")
    print(catalog["koi_disposition"].value_counts())
    print("\nSample rows:")
    print(catalog[["kepid", "kepoi_name", "koi_disposition", "koi_period", "koi_depth"]].head(10).to_string())