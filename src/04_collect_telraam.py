# -*- coding: utf-8 -*-
"""
Step 4 — Collect Telraam / berlin-zaehlt.de citizen sensor data
Source: https://berlin-zaehlt.de/csv/
Coverage: 2022-01 to 2024-12 (bzm_telraam_YYYY_MM.csv.gz)
Output:
  data/telraam/raw/  — one .csv.gz per month
  data/telraam/cleaned/telraam_cleaned.csv
"""

import os
import gzip
import json
import requests
import pandas as pd
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "telraam" / "raw"
CLEAN_DIR    = PROJECT_ROOT / "data" / "telraam" / "cleaned"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://berlin-zaehlt.de/csv/"

# ── 1. Download segment geometry (lat/lon) ───────────────────────────────────
print("Fetching segment geometry ...")
geo_url  = BASE_URL + "bzm_telraam_segments.geojson"
geo_path = RAW_DIR / "bzm_telraam_segments.geojson"

if not geo_path.exists():
    r = requests.get(geo_url, timeout=60)
    r.raise_for_status()
    geo_path.write_bytes(r.content)
    print(f"  Saved: {geo_path.name}")
else:
    print(f"  Already exists: {geo_path.name}")

# Parse centroid lat/lon per segment
with open(geo_path, encoding="utf-8") as f:
    geojson = json.load(f)

seg_coords = {}
for feat in geojson.get("features", []):
    seg_id = feat.get("properties", {}).get("segment_id")
    geom   = feat.get("geometry", {})
    if seg_id is None or geom is None:
        continue
    coords = geom.get("coordinates", [])
    gtype  = geom.get("type", "")
    all_pts = []
    if gtype == "Point":
        all_pts = [coords]
    elif gtype == "LineString":
        all_pts = coords
    elif gtype == "MultiLineString":
        for line in coords:
            all_pts.extend(line)
    if all_pts:
        lons = [c[0] for c in all_pts]
        lats = [c[1] for c in all_pts]
        seg_coords[str(seg_id)] = (sum(lats) / len(lats), sum(lons) / len(lons))

print(f"  Segments with geometry: {len(seg_coords)}")

# ── 2. Download monthly CSV.GZ files ────────────────────────────────────────
years  = range(2022, 2025)   # 2022, 2023, 2024
months = range(1, 13)

to_download = []
for yr in years:
    for mo in months:
        fname = f"bzm_telraam_{yr}_{mo:02d}.csv.gz"
        to_download.append((yr, mo, fname))

print(f"\nDownloading {len(to_download)} monthly files ...")

failed = []
for yr, mo, fname in to_download:
    dest = RAW_DIR / fname
    if dest.exists():
        print(f"  [skip] {fname}")
        continue
    url = BASE_URL + fname
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"  [ok]   {fname}  ({len(r.content) / 1024:.0f} KB)")
    except Exception as e:
        print(f"  [FAIL] {fname}  — {e}")
        failed.append(fname)

if failed:
    print(f"\nWARNING: {len(failed)} file(s) failed to download: {failed}")
else:
    print("All files downloaded successfully.")

# ── 3. Parse and clean ───────────────────────────────────────────────────────
print("\nParsing monthly files ...")

frames = []
for yr, mo, fname in to_download:
    src = RAW_DIR / fname
    if not src.exists():
        print(f"  [missing] {fname} — skipped")
        continue
    try:
        try:
            with gzip.open(src, "rb") as gz:
                df = pd.read_csv(gz, low_memory=False)
        except Exception:
            # Server serves plain CSV despite .gz extension — disable auto-decompress
            df = pd.read_csv(src, compression=None, low_memory=False)

        # Keep only necessary columns
        keep = ["segment_id", "date_local", "bike_total", "uptime"]
        missing_cols = [c for c in keep if c not in df.columns]
        if missing_cols:
            print(f"  [warn] {fname} missing columns: {missing_cols} — skipping")
            continue

        df = df[keep].copy()

        # Filter: sensor must have been running (uptime > 0)
        df = df[df["uptime"] > 0].copy()

        # Standardize timestamp: "YYYY-MM-DD HH:MM" → "YYYY-MM-DD HH:MM:SS"
        df["timestamp"] = pd.to_datetime(df["date_local"], errors="coerce") \
                            .dt.strftime("%Y-%m-%d %H:%M:%S")
        df = df.dropna(subset=["timestamp"])

        # Rename
        df.rename(columns={"segment_id": "location_id",
                            "bike_total": "bike_count"}, inplace=True)

        # Cast
        df["location_id"] = df["location_id"].astype(str)
        df["bike_count"]   = pd.to_numeric(df["bike_count"], errors="coerce")

        # Drop negatives and nulls in bike_count
        df = df[df["bike_count"] >= 0].dropna(subset=["bike_count"])
        df["bike_count"] = df["bike_count"].astype(int)

        # Add lat/lon from geometry lookup
        def get_lat(loc_id):
            coords = seg_coords.get(str(loc_id))
            return coords[0] if coords else None

        def get_lon(loc_id):
            coords = seg_coords.get(str(loc_id))
            return coords[1] if coords else None

        df["lat"] = df["location_id"].map(get_lat)
        df["lon"] = df["location_id"].map(get_lon)

        # Source tag
        df["source"] = "telraam_berlin"

        # Final column order
        df = df[["timestamp", "location_id", "bike_count", "lat", "lon", "source"]]

        frames.append(df)
        print(f"  [parsed] {fname}  rows={len(df):,}")

    except Exception as e:
        print(f"  [ERROR] {fname} — {e}")

# ── 4. Merge, deduplicate and save ───────────────────────────────────────────
print("\nMerging all months ...")
combined = pd.concat(frames, ignore_index=True)

# Remove exact duplicates
before = len(combined)
combined.drop_duplicates(subset=["timestamp", "location_id"], inplace=True)
after = len(combined)
if before != after:
    print(f"  Removed {before - after:,} duplicate rows.")

# Sort
combined.sort_values(["location_id", "timestamp"], inplace=True)
combined.reset_index(drop=True, inplace=True)

out_path = CLEAN_DIR / "telraam_cleaned.csv"
combined.to_csv(out_path, index=False, encoding="utf-8")

# ── 5. Quality report ────────────────────────────────────────────────────────
n_rows     = len(combined)
n_locs     = combined["location_id"].nunique()
date_min   = combined["timestamp"].min()
date_max   = combined["timestamp"].max()
n_with_geo = combined["lat"].notna().sum()
pct_geo    = 100 * n_with_geo / n_rows if n_rows else 0

print()
print("=" * 55)
print("Step 4 -- Telraam COMPLETE")
print(f"Raw saved to:     data/telraam/raw/")
print(f"Cleaned saved to: data/telraam/cleaned/telraam_cleaned.csv")
print(f"Rows:             {n_rows:,}")
print(f"Unique locations: {n_locs:,}")
print(f"Date range:       {date_min}  to  {date_max}")
print(f"Rows with lat/lon:{n_with_geo:,}  ({pct_geo:.1f}%)")
print("=" * 55)
print("Ready for Step 5 -- Nextbike trips? Type YES to continue.")
