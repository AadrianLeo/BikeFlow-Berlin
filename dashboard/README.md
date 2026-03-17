# BikeFlowBerlin V2 — Dashboard

A production-style Streamlit dashboard for exploring Berlin cycling demand,
XGBoost model results, and infrastructure coverage analysis.

---

## Quick Start — Run Locally

### 1. Prerequisites

Make sure the trained model exists at `dashboard/model.pkl`
(see [Regenerate model.pkl](#regenerate-modelpkl) below).

### 2. Install dependencies

```bash
# From the project root
pip install -r dashboard/requirements.txt
```

Or, if using the project virtual environment:

```bash
venv/Scripts/activate         # Windows
# source venv/bin/activate    # macOS / Linux
pip install -r dashboard/requirements.txt
```

### 3. Launch the dashboard

```bash
# From the project root
streamlit run dashboard/streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## Deploy to Streamlit Cloud

1. Push the full repository to GitHub (including `data/` and `dashboard/model.pkl`).
   - `data/final_merged/model_training_data.csv` is ~200 MB — use Git LFS if needed.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Set:
   - **Repository**: your GitHub repo
   - **Branch**: `main`
   - **Main file path**: `dashboard/streamlit_app.py`
4. Under **Advanced settings → Python packages**, point to `dashboard/requirements.txt`.
5. Click **Deploy**.

> **Note**: Streamlit Cloud has a 1 GB memory limit on the free tier.
> The training CSV is 2.2 M rows (~250 MB in memory). If you hit the limit,
> add `@st.cache_data` with `ttl` or pre-aggregate the data before deploying.

---

## Required Data Files

| File | Size | Source script |
|------|------|---------------|
| `data/final_merged/model_training_data.csv` | ~250 MB | `src/07_merge_final.py` |
| `data/final_merged/spatial_coverage_data.csv` | ~1 MB | `src/07_merge_final.py` |
| `data/osm/cleaned/osm_cycling_edges.csv` | ~80 MB | `src/06_collect_osm.py` |
| `dashboard/model.pkl` | ~5 MB | see below |

---

## Regenerate model.pkl

Run the modeling notebook end-to-end and export the trained model:

```bash
# Option A — Jupyter
jupyter notebook Notebooks/02_Modeling.ipynb
# Run all cells; the last cell saves model.pkl to dashboard/

# Option B — nbconvert
python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 \
  --ExecutePreprocessor.kernel_name=python3 \
  Notebooks/02_Modeling.ipynb
```

Or run the training code directly:

```python
import pickle, pandas as pd, numpy as np
from xgboost import XGBRegressor
from pathlib import Path

FEATURE_COLS = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "weekday", "is_weekend", "is_holiday",
    "temperature", "precipitation", "wind_speed",
    "infrastructure_quality_score", "nearest_osm_distance_m",
]

df = pd.read_csv("data/final_merged/model_training_data.csv")
train = df[df["year"] <= 2022].dropna(subset=FEATURE_COLS + ["bike_count"])

model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                     subsample=0.8, colsample_bytree=0.8, random_state=42)
model.fit(train[FEATURE_COLS], train["bike_count"])

with open("dashboard/model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved dashboard/model.pkl")
```

---

## Dashboard Sections

| Section | Description |
|---------|-------------|
| City Overview | Key metrics, Folium sensor coverage map, dataset summary |
| Temporal Patterns | Annual volume, monthly/hourly profiles, weekday vs weekend |
| Weather Impact | Temperature bands, rain effect, correlation heatmap |
| Model Performance | Feature importances, predicted vs actual, per-station R² |
| Live Prediction | Interactive sliders → real-time XGBoost prediction + gauge |
| Infrastructure Audit | Mismatch map, underserved/overbuilt sensors, top-10 table |

---

## Project Structure

```
BikeFlowBerlin_v2/
├── .streamlit/
│   └── config.toml          # Dark theme
├── dashboard/
│   ├── streamlit_app.py     # Main entrypoint
│   ├── model.pkl            # Trained XGBoost model
│   ├── requirements.txt
│   ├── README.md
│   └── components/
│       ├── __init__.py
│       ├── loaders.py       # Cached data / model loading
│       ├── charts.py        # Plotly chart functions
│       └── maps.py          # Folium map builders
├── data/
│   ├── final_merged/
│   │   ├── model_training_data.csv
│   │   └── spatial_coverage_data.csv
│   └── osm/cleaned/
│       └── osm_cycling_edges.csv
├── Notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Modeling.ipynb
└── src/
    └── *.py                 # Data collection & merge scripts
```
