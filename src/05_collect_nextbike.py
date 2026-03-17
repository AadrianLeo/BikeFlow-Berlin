# -*- coding: utf-8 -*-
"""
Step 5 — Collect Nextbike Berlin data
Sources:
  A) Live API snapshot:  https://api.nextbike.net/maps/nextbike-live.json?city=362
  B) Zenodo trip data:   https://zenodo.org/records/10046531
     - city-lab-bike-sharing-data.zip  (2019 Apr-Dec, Nextbike + Call-a-Bike)
     - nextbike-webscraped-data.zip    (2022 Jun-Dec, Nextbike)
Outputs:
  data/nextbike/raw/   — raw JSON + zip files
  data/nextbike/cleaned/nextbike_stations.csv   — infrastructure snapshot
  data/nextbike/cleaned/nextbike_trips.csv      — historical trips (if parseable)
"""

import io
import json
import zipfile
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "nextbike" / "raw"
CLEAN_DIR    = PROJECT_ROOT / "data" / "nextbike" / "cleaned"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# PART A — Live station infrastructure snapshot
# ════════════════════════════════════════════════════════════════════════════
print("=" * 55)
print("Part A — Nextbike Berlin live station snapshot")
print("=" * 55)

url = "https://api.nextbike.net/maps/nextbike-live.json?city=362&list_cities=0"
r = requests.get(url, timeout=60)
r.raise_for_status()
data = r.json()

collected_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
print(f"  Collected at: {collected_at} UTC")

raw_path = RAW_DIR / "nextbike_stations_raw.json"
raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  Raw saved: {raw_path.name}")

stations = []
for country in data.get("countries", []):
    for city in country.get("cities", []):
        for place in city.get("places", []):
            stations.append({
                "station_id":      str(place.get("uid", "")),
                "station_name":    place.get("name", ""),
                "address":         place.get("address", ""),
                "lat":             place.get("lat"),
                "lon":             place.get("lng"),
                "bike_racks":      place.get("bike_racks", 0),
                "bikes_available": place.get("bikes_available_to_rent", 0),
                "bikes_total":     place.get("bikes", 0),
                "free_racks":      place.get("free_racks", 0),
                "terminal_type":   place.get("terminal_type", ""),
                "place_type":      place.get("place_type", 0),
                "active":          place.get("active_place", True),
                "collected_at":    collected_at,
                "source":          "nextbike_berlin",
            })

df_stations = pd.DataFrame(stations)
df_stations = df_stations[df_stations["active"] == True].copy()
df_stations = df_stations.dropna(subset=["lat", "lon"])
df_stations = df_stations[(df_stations["lat"] != 0) & (df_stations["lon"] != 0)]
for col in ["bike_racks", "bikes_available", "bikes_total", "free_racks"]:
    df_stations[col] = pd.to_numeric(df_stations[col], errors="coerce").fillna(0).astype(int)
df_stations.drop_duplicates(subset=["station_id"], inplace=True)
df_stations.reset_index(drop=True, inplace=True)

stations_path = CLEAN_DIR / "nextbike_stations.csv"
df_stations.to_csv(stations_path, index=False, encoding="utf-8")

print(f"  Stations:       {len(df_stations):,}")
print(f"  Total bikes:    {df_stations['bikes_total'].sum():,}")
print(f"  Total capacity: {df_stations['bike_racks'].sum():,}")
print(f"  Saved: nextbike_stations.csv")

# ════════════════════════════════════════════════════════════════════════════
# PART B — Zenodo historical trip data
# ════════════════════════════════════════════════════════════════════════════
print()
print("=" * 55)
print("Part B — Zenodo historical trip/snapshot data")
print("=" * 55)

ZENODO_FILES = {
    "city-lab-bike-sharing-data.zip":
        "https://zenodo.org/records/10046531/files/city-lab-bike-sharing-data.zip?download=1",
    "nextbike-webscraped-data.zip":
        "https://zenodo.org/records/10046531/files/nextbike-webscraped-data.zip?download=1",
}

for fname, url in ZENODO_FILES.items():
    dest = RAW_DIR / fname
    if not dest.exists():
        print(f"  Downloading {fname} ...")
        try:
            r = requests.get(url, timeout=300, stream=True)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"  Saved: {fname}  ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  FAILED: {fname} — {e}")
    else:
        print(f"  Already exists: {fname}")

# ── Inspect and parse zip contents ───────────────────────────────────────────
trip_frames = []

for fname in ZENODO_FILES:
    zpath = RAW_DIR / fname
    if not zpath.exists():
        print(f"  [skip] {fname} — not downloaded")
        continue

    print(f"\n  Inspecting {fname} ...")
    try:
        with zipfile.ZipFile(zpath, "r") as zf:
            names = zf.namelist()
            csv_names = [n for n in names if n.lower().endswith(".csv")]
            print(f"    Files inside: {len(names)} total, {len(csv_names)} CSV")

            for csv_name in csv_names:
                print(f"    Parsing: {csv_name}")
                try:
                    with zf.open(csv_name) as f:
                        df = pd.read_csv(f, low_memory=False)
                    print(f"      Rows: {len(df):,}  Columns: {list(df.columns)}")

                    # Detect column patterns and normalise
                    cols_lower = {c.lower(): c for c in df.columns}

                    # Find timestamp column
                    ts_col = next((cols_lower[k] for k in
                                   ["timestamp", "start_time", "starttime",
                                    "rental_start", "time", "date", "datetime",
                                    "start_at", "started_at"] if k in cols_lower), None)
                    # Find lat/lon
                    lat_col = next((cols_lower[k] for k in
                                    ["start_lat", "lat_start", "latitude", "lat"] if k in cols_lower), None)
                    lon_col = next((cols_lower[k] for k in
                                    ["start_lon", "start_lng", "lon_start", "longitude", "lon", "lng"] if k in cols_lower), None)
                    # Find bike count / trip identifier
                    bike_col = next((cols_lower[k] for k in
                                     ["bike_number", "bike_id", "bikeid", "rental_id",
                                      "trip_id", "id", "number"] if k in cols_lower), None)

                    if ts_col is None:
                        print(f"      WARNING: no timestamp column found — skipping")
                        continue

                    out = pd.DataFrame()
                    out["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce") \
                                         .dt.strftime("%Y-%m-%d %H:%M:%S")
                    out["lat"] = pd.to_numeric(df[lat_col], errors="coerce") if lat_col else None
                    out["lon"] = pd.to_numeric(df[lon_col], errors="coerce") if lon_col else None
                    out["bike_id"] = df[bike_col].astype(str) if bike_col else None
                    out["source"] = "nextbike_zenodo"

                    # Copy any useful extra columns
                    for extra_key, extra_label in [
                        ("station_id", "station_id"),
                        ("start_station", "station_id"),
                        ("stationid", "station_id"),
                        ("duration", "duration_s"),
                        ("trip_duration", "duration_s"),
                    ]:
                        if extra_key in cols_lower:
                            out[extra_label] = df[cols_lower[extra_key]]

                    out = out.dropna(subset=["timestamp"])
                    trip_frames.append(out)
                    print(f"      Normalised rows: {len(out):,}")

                except Exception as e:
                    print(f"      ERROR parsing {csv_name}: {e}")

    except Exception as e:
        print(f"  ERROR opening {fname}: {e}")

# ── Merge and save trips ──────────────────────────────────────────────────────
if trip_frames:
    print("\nMerging trip frames ...")
    trips = pd.concat(trip_frames, ignore_index=True)
    trips.sort_values("timestamp", inplace=True)
    trips.reset_index(drop=True, inplace=True)

    trips_path = CLEAN_DIR / "nextbike_trips.csv"
    trips.to_csv(trips_path, index=False, encoding="utf-8")

    date_min = trips["timestamp"].min()
    date_max = trips["timestamp"].max()
    n_rows   = len(trips)

    print()
    print("=" * 55)
    print("Step 5 -- Nextbike COMPLETE")
    print(f"Stations saved:   data/nextbike/cleaned/nextbike_stations.csv")
    print(f"                  {len(df_stations):,} stations, collected {collected_at[:10]}")
    print(f"Trips saved:      data/nextbike/cleaned/nextbike_trips.csv")
    print(f"                  {n_rows:,} rows")
    print(f"                  date range: {date_min}  to  {date_max}")
    print("=" * 55)
else:
    print()
    print("=" * 55)
    print("Step 5 -- Nextbike COMPLETE (stations only)")
    print(f"Stations saved:   data/nextbike/cleaned/nextbike_stations.csv")
    print(f"                  {len(df_stations):,} stations, collected {collected_at[:10]}")
    print("NOTE: No parseable trip/snapshot CSVs found in Zenodo zips.")
    print("      Sufficient data available from official counters + Telraam.")
    print("=" * 55)

print("Ready for Step 6 -- OSM infrastructure data? Type YES to continue.")
