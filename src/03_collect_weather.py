"""
BikeFlowBerlin V2 — Step 3: Weather Data Collection
Downloads hourly weather for Berlin (Tempelhof) 2012–2024 via meteostat.
"""

import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ("pandas", "meteostat"):
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

import os
from datetime import datetime
import pandas as pd
from meteostat import stations, Point, hourly, Provider, config

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)
RAW_DIR    = os.path.join(ROOT, "data", "weather", "raw")
CLEAN_DIR  = os.path.join(ROOT, "data", "weather", "cleaned")
os.makedirs(RAW_DIR,   exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

RAW_FILE   = os.path.join(RAW_DIR,   "weather_raw.csv")
CLEAN_FILE = os.path.join(CLEAN_DIR, "weather_cleaned.csv")

# ─── Parameters ───────────────────────────────────────────────────────────────
START = datetime(2012, 1, 1)
END   = datetime(2024, 12, 31, 23, 59)

# Berlin-Tempelhof: 52.4675 N, 13.4021 E, 48 m
BERLIN = Point(52.4675, 13.4021, 48)

# ─── Download ─────────────────────────────────────────────────────────────────
print("\n🌤  Step 3 — Collecting Weather Data (meteostat)")
print("=" * 60)
print(f"   Station: Berlin-Tempelhof (nearest available)")
print(f"   Period : {START.date()} → {END.date()}")
print("   Downloading hourly data ... (this may take a moment)")

# Meteostat 2.x blocks requests > 3 years by default
config.block_large_requests = False

try:
    # Resolve nearest DWD station to Berlin-Tempelhof coordinates
    nearby = stations.nearby(BERLIN, radius=50000, limit=5)
    if nearby.empty:
        raise RuntimeError("No stations found near Berlin-Tempelhof")
    station_id = nearby.index[0]
    station_name = nearby.iloc[0]["name"]
    print(f"   Resolved station: {station_id} — {station_name}")

    ts = hourly(station_id, START, END, providers=[Provider.DWD_HOURLY])
    if ts.empty:
        raise RuntimeError(f"No hourly data available for station {station_id}")
    df_raw = ts.fetch()
except Exception as e:
    print(f"❌ Download failed: {e}")
    print("   Possible fix: check your internet connection or try again later.")
    sys.exit(1)

if df_raw is None or df_raw.empty:
    print("❌ No data returned — check meteostat availability and retry.")
    sys.exit(1)

rows_before = len(df_raw)
print(f"   ✅ Downloaded {rows_before:,} hourly rows")

# ─── Save raw ─────────────────────────────────────────────────────────────────
df_raw_save = df_raw.reset_index()
df_raw_save.to_csv(RAW_FILE, index=False)
print(f"   Raw saved → {RAW_FILE}")

# ─── Clean ────────────────────────────────────────────────────────────────────
print("\n🔧 Cleaning weather data ...")
df = df_raw.reset_index().copy()

# Rename meteostat columns to human-friendly names
RENAME = {
    "time":  "timestamp",
    "temp":  "temperature",
    "prcp":  "precipitation",
    "wspd":  "wind_speed",
    "rhum":  "humidity",
    "coco":  "condition",
}
df.rename(columns=RENAME, inplace=True)

# Keep only needed columns (create missing ones as NaN)
needed = ["timestamp", "temperature", "precipitation", "wind_speed", "humidity", "condition"]
for col in needed:
    if col not in df.columns:
        df[col] = float("nan")
df = df[needed]

# Standardise timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

# Forward fill — max 2 consecutive missing values
numeric_cols = ["temperature", "precipitation", "wind_speed", "humidity"]
before_fill = df[numeric_cols].isna().sum().sum()
df[numeric_cols] = df[numeric_cols].ffill(limit=2)
after_fill_missing = df[numeric_cols].isna().sum().sum()
print(f"   Forward-filled {before_fill - after_fill_missing:,} missing values (max 2 consecutive)")

# Drop rows still missing after fill
before_drop = len(df)
df.dropna(subset=["temperature"], inplace=True)   # temperature is the core signal
dropped = before_drop - len(df)
print(f"   Dropped {dropped:,} rows still missing after forward fill")

# Source tag
df["source"] = "dwd_meteostat"

df.to_csv(CLEAN_FILE, index=False)

# ─── Report ───────────────────────────────────────────────────────────────────
ts_series = pd.to_datetime(df["timestamp"], errors="coerce")
sample = df.head(5).to_string(index=False)

print(f"\n✅ Step 3 — Weather Data COMPLETE")
print(f"📁 Raw saved to:     data/weather/raw/weather_raw.csv")
print(f"📁 Cleaned saved to: data/weather/cleaned/weather_cleaned.csv")
print(f"📊 Rows before cleaning: {rows_before:,}")
print(f"📊 Rows after cleaning:  {len(df):,}")
print(f"📅 Date range: {ts_series.min().date()} to {ts_series.max().date()}")
print(f"📋 Columns: {list(df.columns)}")
print(f"🔍 Sample:\n{sample}")
print("\n⏭️  Ready for Step 4 — Telraam / Nextbike Data? Type YES to continue.")
