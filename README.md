<<<<<<< HEAD
# 🔭 Exoplanet Detection using Transit Photometry + ML

> Detecting exoplanets from NASA Kepler light curves using classical ML and deep learning.  

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-green)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What This Project Does

When a planet orbits a star, it periodically passes between the star and Earth — briefly blocking a tiny fraction of the star's light. This dimming event, called a **transit**, creates a measurable dip in the star's brightness over time (its "light curve").

This project builds a complete ML pipeline that:
1. Downloads real NASA Kepler telescope data (light curves of ~9,500 stars)
2. Preprocesses and engineers features from raw flux time-series
3. Trains classical ML models (Random Forest, XGBoost) and deep learning models (1D-CNN, LSTM)
4. Tracks all experiments with MLflow
5. Serves predictions via a FastAPI REST endpoint
6. Provides an interactive Streamlit UI for exploration

**Dataset:** [NASA Kepler DR25](https://exoplanetarchive.ipac.caltech.edu/) — 9,564 labeled stellar light curves  
**Problem type:** Binary time-series classification (planet / no planet)  
**Key challenge:** Severe class imbalance (~5% positive class)

---

## Project Architecture

```
exoplanet-detector/
├── src/
│   ├── data/
│   │   ├── downloader.py       # Fetch Kepler light curves via lightkurve
│   │   ├── preprocessor.py     # Normalization, sigma clipping, interpolation
│   │   ├── feature_engineer.py # Phase folding, BLS periodogram features
│   │   └── dataset.py          # PyTorch Dataset class
│   ├── models/
│   │   ├── baseline.py         # Random Forest + XGBoost
│   │   ├── cnn1d.py            # 1D Convolutional Neural Network
│   │   ├── lstm.py             # LSTM sequence classifier
│   │   └── ensemble.py         # Model stacking/ensembling
│   ├── utils/
│   │   ├── metrics.py          # Precision, recall, F1, AUC-ROC
│   │   ├── visualization.py    # Light curve and result plots
│   │   └── mlflow_utils.py     # Experiment tracking helpers
│   ├── api/
│   │   └── main.py             # FastAPI inference endpoint
│   └── ui/
│       └── app.py              # Streamlit dashboard
├── tests/                      # pytest test suite
├── notebooks/                  # EDA and experiment notebooks
├── configs/
│   └── config.yaml             # All hyperparameters and paths
├── docs/                       # Architecture diagrams and notes
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/spmridula/exoplanet-detector.git
cd exoplanet-detector
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download a sample light curve and verify setup
python -c "from src.data.downloader import download_sample; download_sample()"

# 4. Run the Streamlit UI
streamlit run src/ui/app.py

# 5. Start the API
uvicorn src.api.main:app --reload
```

---

## Key Results (updated as project progresses)

| Model | Precision | Recall | F1 | AUC-ROC |
|-------|-----------|--------|----|---------|
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |
| 1D-CNN | — | — | — | — |
| LSTM | — | — | — | — |
| Ensemble | — | — | — | — |

---

## Background: How Transit Photometry Works

A transit event looks like this in a light curve:

```
Flux
1.000 ─────────────────╮     ╭──────────────────
                        │     │
0.995                   │     │    ← ~0.5% dip for Earth-sized planet
                        ╰─────╯
         ─────────────────────────────────► Time
```

Key parameters extracted from each transit:
- **Depth** → ratio of planet radius to star radius
- **Duration** → planet's orbital velocity
- **Period** → orbital year length (days or months)
- **Shape** → limb darkening coefficient, orbital inclination

---

## Why This Problem Is Hard (and Interesting)

- **Class imbalance:** Only ~5% of stars have detectable exoplanets
- **Noise:** Stellar variability, instrument artifacts, and cosmic ray hits contaminate signals
- **Scale:** Each light curve has ~65,000 time steps spanning 4 years
- **False positives:** Eclipsing binary stars mimic transit signals perfectly

This makes it a realistic, non-trivial ML problem — not just pattern matching on a clean dataset.

---

## Data Source

All data is from NASA's publicly available archives:
- **Kepler DR25 Stellar Catalog** — [IPAC Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- **lightkurve library** — [docs.lightkurve.org](https://docs.lightkurve.org/)

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Data | lightkurve, pandas, numpy, scipy |
| ML | scikit-learn, xgboost |
| Deep Learning | PyTorch |
| Experiment Tracking | MLflow |
| API | FastAPI, uvicorn |
| UI | Streamlit |
| Deployment | Docker, Hugging Face Spaces |
| CI/CD | GitHub Actions |
| Testing | pytest |

---

## License

MIT — see [LICENSE](LICENSE)
=======
# exoplanet-detection
Pipeline for detecting exoplanets from NASA Kepler light curves.
>>>>>>> b9af75cd86cbdf00b87a84844b75449b150c8758
