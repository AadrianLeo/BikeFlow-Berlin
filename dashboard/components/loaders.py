"""
dashboard/components/loaders.py
Data loading helpers with Streamlit caching.
All paths are resolved relative to this file so the app works both
locally and on Streamlit Cloud without hardcoded absolute paths.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
# This file lives at:  <repo>/dashboard/components/loaders.py
# Data lives at:       <repo>/data/
# Model lives at:      <repo>/dashboard/model.pkl

_COMPONENTS_DIR = Path(__file__).parent          # dashboard/components/
_DASHBOARD_DIR  = _COMPONENTS_DIR.parent         # dashboard/
_REPO_ROOT      = _DASHBOARD_DIR.parent          # repo root
_DATA_DIR       = _REPO_ROOT / "data"

TRAINING_DATA_PATH = _DATA_DIR / "final_merged" / "model_training_data.csv"
SPATIAL_DATA_PATH  = _DATA_DIR / "final_merged" / "spatial_coverage_data.csv"
OSM_EDGES_PATH     = _DATA_DIR / "osm" / "cleaned" / "osm_cycling_edges.csv"
MODELED_MISMATCH_PATH = _DATA_DIR / "final_merged" / "osm_modeled_mismatch.csv"
MODEL_PATH         = _DASHBOARD_DIR / "model.pkl"

# Paths for sensor name lookups (matches notebook logic)
OFFICIAL_COUNTERS_PATH = _DATA_DIR / "official_counters" / "cleaned" / "official_counters_cleaned.csv"
TELRAAM_GEOJSON_PATH   = _DATA_DIR / "telraam" / "raw" / "bzm_telraam_segments.geojson"

# Feature columns — must match training order exactly
FEATURE_COLS: list[str] = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "weekday",
    "is_weekend",
    "is_holiday",
    "temperature",
    "precipitation",
    "wind_speed",
    "infrastructure_quality_score",
    "nearest_osm_distance_m",
]


# ---------------------------------------------------------------------------
# Sensor Name Lookup (matches notebook logic exactly)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_official_counter_names() -> dict[str, str]:
    """
    Load station_id → station_name mapping from official counters.
    Returns empty dict if file not found.
    """
    if not OFFICIAL_COUNTERS_PATH.exists():
        return {}
    try:
        df = pd.read_csv(
            OFFICIAL_COUNTERS_PATH,
            usecols=["station_id", "station_name"]
        ).drop_duplicates("station_id")
        return df.set_index("station_id")["station_name"].to_dict()
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _load_telraam_street_names() -> dict[int, str]:
    """
    Load segment_id → street_name mapping from Telraam geojson.
    Street name comes from osm.name or osm.address.road.
    Returns empty dict if file not found.
    """
    if not TELRAAM_GEOJSON_PATH.exists():
        return {}
    try:
        with open(TELRAAM_GEOJSON_PATH, encoding="utf-8") as f:
            gj = json.load(f)
        tel_name: dict[int, str] = {}
        for feat in gj.get("features", []):
            props = feat.get("properties", {})
            sid = props.get("segment_id")
            osm = props.get("osm", {})
            name = osm.get("name") or osm.get("address", {}).get("road", "")
            if sid and name:
                tel_name[int(sid)] = name
        return tel_name
    except Exception:
        return {}


def get_sensor_display_name(location_id: str, source: str) -> str:
    """
    Get human-readable name for a sensor location.
    - Official Berlin counters: return station_name (e.g., "Paul-und-Paula-Uferweg")
    - Telraam: return street name from geojson (e.g., "Handjerystraße")
    - Nextbike/other: return empty string (no name data available)

    This matches the notebook logic exactly.
    """
    loc = str(location_id)
    source_lower = source.lower().strip() if source else ""

    if source_lower == "official_berlin":
        name_dict = _load_official_counter_names()
        return name_dict.get(loc, "")
    elif source_lower == "telraam":
        name_dict = _load_telraam_street_names()
        try:
            return name_dict.get(int(loc), "")
        except (ValueError, TypeError):
            return ""
    else:
        return ""


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading training data…")
def load_training_data() -> pd.DataFrame | None:
    """Load model_training_data.csv (2.2 M rows)."""
    if not TRAINING_DATA_PATH.exists():
        st.warning(
            f"Training data not found at `{TRAINING_DATA_PATH}`. "
            "Some sections will be unavailable."
        )
        return None
    df = pd.read_csv(TRAINING_DATA_PATH, parse_dates=["timestamp"])
    # Ensure numeric types for key columns
    for col in ["bike_count", "temperature", "precipitation",
                "wind_speed", "humidity", "year", "month", "hour",
                "weekday", "is_weekend", "is_holiday"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner="Loading spatial coverage data…")
def load_spatial_data() -> pd.DataFrame | None:
    """Load spatial_coverage_data.csv (3 989 rows)."""
    if not SPATIAL_DATA_PATH.exists():
        st.warning(
            f"Spatial data not found at `{SPATIAL_DATA_PATH}`. "
            "Map sections will be unavailable."
        )
        return None
    df = pd.read_csv(SPATIAL_DATA_PATH)
    for col in ["lat", "lon", "avg_daily_demand",
                "infrastructure_quality_score", "nearest_osm_distance_m"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner="Loading OSM cycling edges…")
def load_osm_edges() -> pd.DataFrame | None:
    """Load osm_cycling_edges.csv (436 k rows). Optional — used for audit section."""
    if not OSM_EDGES_PATH.exists():
        st.warning(
            f"OSM edges not found at `{OSM_EDGES_PATH}`. "
            "Infrastructure audit will fall back to sensor data only."
        )
        return None
    df = pd.read_csv(
        OSM_EDGES_PATH,
        usecols=["osmid", "name", "highway", "infra_type",
                 "infra_quality_score", "length_m", "lat", "lon"],
        low_memory=False,
    )
    return df


@st.cache_data(show_spinner=False)
def _load_osm_name_lookup() -> pd.DataFrame:
    """
    Load OSM edge names for merging into modeled mismatch data.
    Returns DataFrame with lat, lon, name, osmid, infra_type columns.
    Uses rounded lat/lon for faster merge.
    """
    if not OSM_EDGES_PATH.exists():
        return pd.DataFrame(columns=["lat_r", "lon_r", "name", "osmid", "infra_type"])

    df = pd.read_csv(
        OSM_EDGES_PATH,
        usecols=["osmid", "name", "infra_type", "lat", "lon"],
        low_memory=False,
    )
    # Round to 6 decimal places for merge key (sub-meter precision)
    df["lat_r"] = df["lat"].round(6)
    df["lon_r"] = df["lon"].round(6)
    # Keep only necessary columns, drop duplicates by location
    return df[["lat_r", "lon_r", "name", "osmid", "infra_type"]].drop_duplicates(
        subset=["lat_r", "lon_r"], keep="first"
    )


@st.cache_data(show_spinner="Loading modeled mismatch data…")
def load_modeled_mismatch() -> pd.DataFrame | None:
    """
    Load osm_modeled_mismatch.csv (436 k rows).
    This is the ML-predicted demand across all OSM edges with mismatch analysis.
    Automatically merges in name, osmid, infra_type from OSM edges file.
    """
    # Debug: print actual resolved path
    import sys
    full_path = MODELED_MISMATCH_PATH.resolve()
    exists = full_path.exists()

    # Log to stderr for Streamlit visibility (stderr shows in terminal, not in app)
    print(f"[loaders.py] Modeled mismatch path: {full_path}", file=sys.stderr)
    print(f"[loaders.py] Path exists: {exists}", file=sys.stderr)

    if not exists:
        st.warning(
            f"Modeled mismatch data not found at `{full_path}`. "
            "Citywide mismatch analysis will be unavailable. "
            "Please export from `Notebooks/02_Modeling.ipynb` Cell 6."
        )
        return None

    try:
        df = pd.read_csv(full_path)
        # Ensure numeric types for key columns
        for col in ["lat", "lon", "predicted_demand", "infrastructure_quality_score", "mismatch"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Merge in OSM names if not already present
        if "name" not in df.columns or df["name"].isna().all():
            osm_names = _load_osm_name_lookup()
            if not osm_names.empty:
                # Create merge keys
                df["lat_r"] = df["lat"].round(6)
                df["lon_r"] = df["lon"].round(6)

                # Merge name, osmid, infra_type
                df = df.merge(
                    osm_names,
                    on=["lat_r", "lon_r"],
                    how="left",
                    suffixes=("", "_osm")
                )
                # Clean up merge columns
                df = df.drop(columns=["lat_r", "lon_r"], errors="ignore")

                print(f"[loaders.py] Merged OSM names: {df['name'].notna().sum():,} / {len(df):,} edges have names", file=sys.stderr)

        print(f"[loaders.py] Modeled mismatch loaded: {len(df)} rows", file=sys.stderr)
        return df
    except Exception as e:
        st.error(f"Error loading modeled mismatch data: {e}")
        return None


@st.cache_resource(show_spinner="Loading XGBoost model…")
def load_model():
    """Load the trained XGBoost model from dashboard/model.pkl."""
    if not MODEL_PATH.exists():
        st.warning(
            f"Model not found at `{MODEL_PATH}`. "
            "Prediction and performance sections will be unavailable."
        )
        return None
    with open(MODEL_PATH, "rb") as fh:
        model = pickle.load(fh)
    return model
