# -*- coding: utf-8 -*-
"""
Step 7 — Final Merge
Outputs:
  data/final_merged/model_training_data.csv   — RQ1/RQ2/RQ3/RQ4 (demand model)
  data/final_merged/spatial_coverage_data.csv — RQ4/RQ5 (demand map + infra audit)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import KDTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MERGED_DIR   = PROJECT_ROOT / "data" / "final_merged"
MERGED_DIR.mkdir(parents=True, exist_ok=True)

# ── Paths ────────────────────────────────────────────────────────────────────
COUNTERS_PATH  = PROJECT_ROOT / "data" / "official_counters" / "cleaned" / "official_counters_cleaned.csv"
WEATHER_PATH   = PROJECT_ROOT / "data" / "weather"           / "cleaned" / "weather_cleaned.csv"
CALENDAR_PATH  = PROJECT_ROOT / "data" / "calendar"          / "raw"     / "calendar_data_2012_2025.csv"
TELRAAM_PATH   = PROJECT_ROOT / "data" / "telraam"           / "cleaned" / "telraam_cleaned.csv"
NEXTBIKE_PATH  = PROJECT_ROOT / "data" / "nextbike"          / "cleaned" / "nextbike_stations.csv"
OSM_EDGES_PATH = PROJECT_ROOT / "data" / "osm"               / "cleaned" / "osm_cycling_edges.csv"

# ════════════════════════════════════════════════════════════════════════════
# 1. Load all sources
# ════════════════════════════════════════════════════════════════════════════
print("Loading data sources ...")

print("  Official counters ...")
counters = pd.read_csv(COUNTERS_PATH, low_memory=False)
print(f"    {len(counters):,} rows, {counters['station_id'].nunique()} stations")

print("  Weather ...")
weather = pd.read_csv(WEATHER_PATH, low_memory=False)
print(f"    {len(weather):,} rows")

print("  Calendar ...")
calendar = pd.read_csv(CALENDAR_PATH, low_memory=False)
print(f"    {len(calendar):,} rows")

print("  Telraam ...")
telraam = pd.read_csv(TELRAAM_PATH, low_memory=False)
print(f"    {len(telraam):,} rows, {telraam['location_id'].nunique()} locations")

print("  Nextbike stations ...")
nextbike = pd.read_csv(NEXTBIKE_PATH, low_memory=False)
print(f"    {len(nextbike):,} stations")

print("  OSM edges ...")
osm_edges = pd.read_csv(OSM_EDGES_PATH, low_memory=False)
print(f"    {len(osm_edges):,} edges")

# ════════════════════════════════════════════════════════════════════════════
# 2. Build OSM spatial index (KDTree) for nearest-edge lookup
# ════════════════════════════════════════════════════════════════════════════
print("\nBuilding OSM spatial index ...")

osm_valid = osm_edges.dropna(subset=["lat", "lon"]).copy()
osm_valid.reset_index(drop=True, inplace=True)

# Scale lon by cos(centre_lat) so Euclidean ≈ great-circle distance
LAT_CENTRE_RAD = np.radians(osm_valid["lat"].mean())   # ≈ 52.5° for Berlin
LON_SCALE      = np.cos(LAT_CENTRE_RAD)                 # ≈ 0.607

osm_coords = osm_valid[["lat", "lon"]].copy()
osm_coords["lon_scaled"] = osm_coords["lon"] * LON_SCALE
osm_tree = KDTree(osm_coords[["lat", "lon_scaled"]].values)
print(f"  KDTree built on {len(osm_valid):,} OSM edge midpoints")


def find_nearest_osm(lats, lons):
    """
    For arrays of lat/lon query points, return
    (infra_type[], infra_quality_score[], distance_m[]).
    """
    query = np.column_stack([lats, np.array(lons) * LON_SCALE])
    dists_deg, idxs = osm_tree.query(query)
    dists_m = np.round(dists_deg * 111_320, 1)   # 1 degree lat ≈ 111,320 m
    infra_types    = osm_valid["infra_type"].iloc[idxs].values
    quality_scores = osm_valid["infra_quality_score"].iloc[idxs].values
    return infra_types, quality_scores, dists_m


# ════════════════════════════════════════════════════════════════════════════
# DATASET 1 — Model Training Data
# ════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Building Dataset 1 — Model Training Data")
print("=" * 60)

# ── 2a. Parse timestamps, extract date + hour ─────────────────────────────
print("  Parsing timestamps ...")
counters["timestamp"] = pd.to_datetime(counters["timestamp"], errors="coerce")
counters.dropna(subset=["timestamp"], inplace=True)
counters["date"]    = counters["timestamp"].dt.strftime("%Y-%m-%d")
counters["hour"]    = counters["timestamp"].dt.hour
counters["ts_hour"] = counters["timestamp"].dt.floor("h")

# ── 2b. Join weather by hour ──────────────────────────────────────────────
print("  Joining weather by hour ...")
weather["ts_hour"] = pd.to_datetime(weather["timestamp"], errors="coerce").dt.floor("h")
weather_slim = (weather[["ts_hour", "temperature", "precipitation",
                           "wind_speed", "humidity"]]
                .drop_duplicates(subset=["ts_hour"]))

merged = counters.merge(weather_slim, on="ts_hour", how="left")
n_w = merged["temperature"].notna().sum()
print(f"    {n_w:,} / {len(merged):,} rows matched ({100*n_w/len(merged):.1f}%)")

# ── 2c. Join calendar by date ─────────────────────────────────────────────
print("  Joining calendar by date ...")
CALENDAR_COLS = ["date", "year", "month", "day", "weekday", "weekday_name",
                 "is_weekend", "is_holiday", "holiday_name",
                 "is_school_holiday", "season", "week_of_year"]
calendar["date"] = calendar["date"].astype(str)
cal_slim = calendar[CALENDAR_COLS].drop_duplicates(subset=["date"])

merged = merged.merge(cal_slim, on="date", how="left")
n_c = merged["year"].notna().sum()
print(f"    {n_c:,} / {len(merged):,} rows matched ({100*n_c/len(merged):.1f}%)")

# ── 2d. Join nearest OSM edge (per station — 38 lookups, not 2.2M) ────────
print("  Finding nearest OSM edge per station ...")
station_locs = (counters[["station_id", "lat", "lon"]]
                .drop_duplicates(subset=["station_id"])
                .dropna(subset=["lat", "lon"]))

it, qs, dm = find_nearest_osm(station_locs["lat"].values,
                               station_locs["lon"].values)
station_osm = station_locs[["station_id"]].copy()
station_osm["infrastructure_type"]          = it
station_osm["infrastructure_quality_score"] = qs
station_osm["nearest_osm_distance_m"]       = dm

merged = merged.merge(station_osm, on="station_id", how="left")
print(f"    Matched {station_osm['station_id'].nunique()} stations")
for _, r in station_osm.iterrows():
    print(f"      {r['station_id']}: {r['infrastructure_type']} "
          f"(score={r['infrastructure_quality_score']}, "
          f"dist={r['nearest_osm_distance_m']:.0f} m)")

# ── 2e. Engineered time features ─────────────────────────────────────────
print("  Computing cyclical time features ...")
merged["hour_sin"]  = np.sin(2 * np.pi * merged["hour"]  / 24)
merged["hour_cos"]  = np.cos(2 * np.pi * merged["hour"]  / 24)
merged["month_sin"] = np.sin(2 * np.pi * merged["month"] / 12)
merged["month_cos"] = np.cos(2 * np.pi * merged["month"] / 12)

# ── 2f. Select final columns ──────────────────────────────────────────────
KEEP_1 = [
    "timestamp", "station_id", "bike_count", "lat", "lon", "source",
    "temperature", "precipitation", "wind_speed", "humidity",
    "year", "month", "day", "hour", "weekday", "weekday_name",
    "is_weekend", "is_holiday", "holiday_name", "is_school_holiday",
    "season", "week_of_year",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "infrastructure_type", "infrastructure_quality_score",
    "nearest_osm_distance_m",
]
cols_1 = [c for c in KEEP_1 if c in merged.columns]
model_data = merged[cols_1].copy()
model_data["timestamp"] = model_data["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

out1 = MERGED_DIR / "model_training_data.csv"
model_data.to_csv(out1, index=False, encoding="utf-8")

print(f"\n  Saved: {out1.name}")
print(f"  Rows:    {len(model_data):,}")
print(f"  Columns ({len(cols_1)}): {cols_1}")

# ════════════════════════════════════════════════════════════════════════════
# DATASET 2 — Spatial Coverage Data
# ════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Building Dataset 2 — Spatial Coverage Data")
print("=" * 60)

spatial_parts = []

# ── Source 1: official counters ───────────────────────────────────────────
print("  Processing official counter stations ...")
ctr2 = pd.read_csv(COUNTERS_PATH, low_memory=False)
ctr2["date"] = pd.to_datetime(ctr2["timestamp"], errors="coerce").dt.date
daily_ctr    = (ctr2.groupby(["station_id", "date"])["bike_count"]
                .sum().reset_index())
avg_ctr      = (daily_ctr.groupby("station_id")["bike_count"]
                .mean().round(1).reset_index()
                .rename(columns={"bike_count": "avg_daily_demand"}))
meta_ctr     = (ctr2[["station_id", "lat", "lon"]]
                .drop_duplicates(subset=["station_id"]))
off_sp = meta_ctr.merge(avg_ctr, on="station_id", how="left")
off_sp.rename(columns={"station_id": "location_id"}, inplace=True)
off_sp["location_id"] = off_sp["location_id"].astype(str)
off_sp["source"] = "official_berlin"
spatial_parts.append(off_sp)
print(f"    {len(off_sp)} locations, avg demand: {avg_ctr['avg_daily_demand'].mean():.0f} bikes/day")

# ── Source 2: Telraam ─────────────────────────────────────────────────────
print("  Processing Telraam locations ...")
tel2 = telraam.copy()
tel2["date"]  = pd.to_datetime(tel2["timestamp"], errors="coerce").dt.date
daily_tel     = (tel2.groupby(["location_id", "date"])["bike_count"]
                 .sum().reset_index())
avg_tel       = (daily_tel.groupby("location_id")["bike_count"]
                 .mean().round(1).reset_index()
                 .rename(columns={"bike_count": "avg_daily_demand"}))
meta_tel      = (tel2[["location_id", "lat", "lon"]]
                 .drop_duplicates(subset=["location_id"]))
tel_sp = meta_tel.merge(avg_tel, on="location_id", how="left")
tel_sp["location_id"] = tel_sp["location_id"].astype(str)
tel_sp["source"] = "telraam"
spatial_parts.append(tel_sp)
print(f"    {len(tel_sp)} locations, avg demand: {avg_tel['avg_daily_demand'].mean():.0f} bikes/day")

# ── Source 3: Nextbike stations ───────────────────────────────────────────
print("  Processing Nextbike stations ...")
nb_sp = nextbike[["station_id", "lat", "lon"]].copy()
nb_sp = nb_sp.dropna(subset=["lat", "lon"])
nb_sp = nb_sp[(nb_sp["lat"] != 0) & (nb_sp["lon"] != 0)]
nb_sp.rename(columns={"station_id": "location_id"}, inplace=True)
nb_sp["location_id"]     = nb_sp["location_id"].astype(str)
nb_sp["avg_daily_demand"] = np.nan
nb_sp["source"]           = "nextbike"
spatial_parts.append(nb_sp)
print(f"    {len(nb_sp)} locations (no count data — spatial indicator only)")

# ── Combine ───────────────────────────────────────────────────────────────
spatial = pd.concat(spatial_parts, ignore_index=True)
spatial = spatial.dropna(subset=["lat", "lon"])
print(f"\n  Combined: {len(spatial):,} total locations")

# ── Join nearest OSM edge ─────────────────────────────────────────────────
print("  Finding nearest OSM edge for all locations ...")
it, qs, dm = find_nearest_osm(spatial["lat"].values, spatial["lon"].values)
spatial["infrastructure_type"]          = it
spatial["infrastructure_quality_score"] = qs
spatial["nearest_osm_distance_m"]       = dm

# ── District lookup via osmnx ─────────────────────────────────────────────
print("  Looking up Berlin districts via osmnx ...")
try:
    import osmnx as ox
    import geopandas as gpd
    from shapely.geometry import Point

    # Download Berlin Bezirke (admin_level=9)
    dist_gdf = ox.features_from_place(
        "Berlin, Germany",
        tags={"boundary": "administrative", "admin_level": "9"}
    )
    # Keep only polygon/multipolygon features that have a name
    dist_gdf = dist_gdf[dist_gdf.geometry.notna()].copy()
    poly_mask = dist_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    dist_gdf  = dist_gdf[poly_mask & dist_gdf.index.get_level_values("element").isin(["relation", "way"])].copy()
    dist_gdf  = dist_gdf[["name", "geometry"]].reset_index(drop=True)
    dist_gdf  = dist_gdf[dist_gdf["name"].notna()].copy()
    dist_gdf  = dist_gdf.set_crs("EPSG:4326")
    print(f"    {len(dist_gdf)} district polygons downloaded: {sorted(dist_gdf['name'].tolist())}")

    # Point-in-polygon
    spatial_gdf = gpd.GeoDataFrame(
        spatial,
        geometry=[Point(r.lon, r.lat) for r in spatial.itertuples()],
        crs="EPSG:4326"
    )
    joined = gpd.sjoin(spatial_gdf, dist_gdf[["name", "geometry"]],
                       how="left", predicate="within")
    # sjoin may duplicate rows if a point is in multiple polygons — take first
    joined = joined[~joined.index.duplicated(keep="first")]
    spatial["district"] = joined["name"].values
    n_dist = spatial["district"].notna().sum()
    print(f"    {n_dist:,} / {len(spatial):,} locations assigned to a district")

except Exception as e:
    print(f"    WARNING: District lookup failed — {e}")
    spatial["district"] = None

# ── Final column selection ────────────────────────────────────────────────
KEEP_2 = ["location_id", "lat", "lon", "source", "avg_daily_demand",
          "infrastructure_type", "infrastructure_quality_score",
          "nearest_osm_distance_m", "district"]
cols_2 = [c for c in KEEP_2 if c in spatial.columns]
spatial_out = spatial[cols_2].copy()

out2 = MERGED_DIR / "spatial_coverage_data.csv"
spatial_out.to_csv(out2, index=False, encoding="utf-8")

print(f"\n  Saved: {out2.name}")
print(f"  Rows:    {len(spatial_out):,}")
print(f"  Columns ({len(cols_2)}): {cols_2}")

# ── Quality reports ───────────────────────────────────────────────────────
n_null = spatial_out["avg_daily_demand"].isna().sum()
print(f"\n  Null avg_daily_demand: {n_null:,}")
print("    → Nextbike stations are spatial indicators only — no count data collected")

print("\n  Infrastructure type distribution (by source):")
for src in ["official_berlin", "telraam", "nextbike"]:
    sub = spatial_out[spatial_out["source"] == src]
    dist = sub["infrastructure_type"].value_counts()
    print(f"    [{src}] ({len(sub):,} locations)")
    for infra, cnt in dist.items():
        pct = 100 * cnt / len(sub)
        print(f"      {infra:<20}  {cnt:>5,}  ({pct:.1f}%)")

# ════════════════════════════════════════════════════════════════════════════
# Final summary
# ════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Step 7 -- Final Merge COMPLETE")
print(f"Dataset 1:  data/final_merged/model_training_data.csv")
print(f"            {len(model_data):,} rows  {len(cols_1)} columns")
print(f"Dataset 2:  data/final_merged/spatial_coverage_data.csv")
print(f"            {len(spatial_out):,} rows  {len(cols_2)} columns")
print("=" * 60)
