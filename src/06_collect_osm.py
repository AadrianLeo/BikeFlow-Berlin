# -*- coding: utf-8 -*-
"""
Step 6 — Collect OSM cycling infrastructure for Berlin (full tags)
Source: OpenStreetMap via osmnx / Overpass API
Outputs:
  data/osm/raw/berlin_bike_graph.graphml     — full bike network graph
  data/osm/cleaned/osm_cycling_edges.csv     — edge-level cycling infrastructure
  data/osm/cleaned/osm_cycling_nodes.csv     — node-level intersections
"""

import osmnx as ox
import pandas as pd
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "osm" / "raw"
CLEAN_DIR    = PROJECT_ROOT / "data" / "osm" / "cleaned"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# ── osmnx settings (must be set BEFORE downloading) ─────────────────────────
ox.settings.use_cache    = True
ox.settings.log_console  = False
ox.settings.cache_folder = str(RAW_DIR / "osmnx_cache")

# Extend default tag list to capture all cycling infrastructure attributes
EXTRA_TAGS = [
    "cycleway", "cycleway:left", "cycleway:right", "cycleway:both",
    "bicycle", "surface",
]
ox.settings.useful_tags_way = list(set(
    ox.settings.useful_tags_way + EXTRA_TAGS
))

PLACE   = "Berlin, Germany"
GRAPHML = RAW_DIR / "berlin_bike_graph.graphml"

# ════════════════════════════════════════════════════════════════════════════
# 1. Download bike network with full tags
# ════════════════════════════════════════════════════════════════════════════
# Delete old graph — it was downloaded without cycling tags
if GRAPHML.exists():
    GRAPHML.unlink()
    print("Deleted old graph (missing cycling tags) — re-downloading ...")

print(f"Downloading bike network for '{PLACE}' with full cycling tags ...")
print("(This may take several minutes — please wait.)")
G = ox.graph_from_place(PLACE, network_type="bike", retain_all=False)
ox.save_graphml(G, GRAPHML)
print(f"  Graph saved: {GRAPHML.name}")
print(f"  Nodes: {len(G.nodes):,}   Edges: {len(G.edges):,}")

# ════════════════════════════════════════════════════════════════════════════
# 2. Convert to GeoDataFrames
# ════════════════════════════════════════════════════════════════════════════
print("Converting graph to GeoDataFrames ...")
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
print(f"  Available edge columns: {sorted(edges_gdf.columns.tolist())}")

# ════════════════════════════════════════════════════════════════════════════
# 3. Process edges
# ════════════════════════════════════════════════════════════════════════════
print("Processing edges ...")
edges = edges_gdf.copy()
edges.reset_index(inplace=True)   # u, v, key become regular columns

# Normalise: osmnx can return lists when multiple values exist for a tag
def first_or_val(x):
    if isinstance(x, list):
        return x[0] if x else None
    return x

NORMALISE_COLS = [
    "name", "highway", "cycleway", "cycleway:left", "cycleway:right",
    "cycleway:both", "bicycle", "surface", "maxspeed", "lanes", "oneway",
]
for col in NORMALISE_COLS:
    if col in edges.columns:
        edges[col] = edges[col].apply(first_or_val)

# Helper: safe lower-case string getter
def sget(row, key):
    v = row.get(key)
    if v is None or (isinstance(v, float)):
        return ""
    return str(v).lower().strip()

# ── Mid-point lat/lon ─────────────────────────────────────────────────────
def midpoint(geom):
    if geom is None:
        return None, None
    try:
        if hasattr(geom, "geoms"):
            pts = [c for g in geom.geoms for c in g.coords]
        else:
            pts = list(geom.coords)
        if not pts:
            return None, None
        mid = pts[len(pts) // 2]
        return round(mid[1], 6), round(mid[0], 6)
    except Exception:
        return None, None

edges[["mid_lat", "mid_lon"]] = edges["geometry"].apply(
    lambda g: pd.Series(midpoint(g))
)

# ── Infrastructure classification ─────────────────────────────────────────
#
# Priority (highest → lowest):
#   dedicated_path  → highway=cycleway
#   protected_lane  → cycleway=track  OR  cycleway:left/right/both=track
#   painted_lane    → cycleway=lane   OR  cycleway:left/right/both=lane
#   shared_lane     → cycleway=shared_lane OR cycleway:left/right/both=shared_lane
#   mixed_traffic   → everything else
#
QUALITY_SCORE = {
    "dedicated_path": 4,
    "protected_lane": 3,
    "painted_lane":   2,
    "shared_lane":    1,
    "mixed_traffic":  0,
}

def classify_infra(row):
    hw   = sget(row, "highway")
    cy   = sget(row, "cycleway")
    cy_l = sget(row, "cycleway:left")
    cy_r = sget(row, "cycleway:right")
    cy_b = sget(row, "cycleway:both")

    if hw == "cycleway":
        return "dedicated_path"

    all_cy = {cy, cy_l, cy_r, cy_b} - {""}
    if "track" in all_cy:
        return "protected_lane"
    if "lane" in all_cy:
        return "painted_lane"
    if "shared_lane" in all_cy:
        return "shared_lane"

    return "mixed_traffic"

print("  Classifying infrastructure ...")
edges["infra_type"] = edges.apply(classify_infra, axis=1)
edges["infra_quality_score"] = edges["infra_type"].map(QUALITY_SCORE)

# ── Select & rename output columns ────────────────────────────────────────
KEEP = {
    "osmid":          "osmid",
    "u":              "from_node",
    "v":              "to_node",
    "name":           "name",
    "highway":        "highway",
    "cycleway":       "cycleway",
    "cycleway:left":  "cycleway_left",
    "cycleway:right": "cycleway_right",
    "cycleway:both":  "cycleway_both",
    "bicycle":        "bicycle",
    "surface":        "surface",
    "maxspeed":       "maxspeed",
    "lanes":          "lanes",
    "oneway":         "oneway",
    "length":         "length_m",
    "mid_lat":        "lat",
    "mid_lon":        "lon",
    "infra_type":     "infra_type",
    "infra_quality_score": "infra_quality_score",
}

out_cols = {k: v for k, v in KEEP.items() if k in edges.columns}
edges_out = edges[list(out_cols.keys())].rename(columns=out_cols).copy()
edges_out["source"] = "osm"

# ════════════════════════════════════════════════════════════════════════════
# 4. Process nodes
# ════════════════════════════════════════════════════════════════════════════
print("Processing nodes ...")
nodes = nodes_gdf.copy()
nodes.reset_index(inplace=True)
nodes_out = nodes[["osmid", "y", "x"]].copy()
nodes_out.rename(columns={"osmid": "node_id", "y": "lat", "x": "lon"}, inplace=True)
nodes_out["source"] = "osm"

# ════════════════════════════════════════════════════════════════════════════
# 5. Save
# ════════════════════════════════════════════════════════════════════════════
edges_path = CLEAN_DIR / "osm_cycling_edges.csv"
nodes_path = CLEAN_DIR / "osm_cycling_nodes.csv"
edges_out.to_csv(edges_path, index=False, encoding="utf-8")
nodes_out.to_csv(nodes_path, index=False, encoding="utf-8")

# ════════════════════════════════════════════════════════════════════════════
# 6. Quality report
# ════════════════════════════════════════════════════════════════════════════
n_edges  = len(edges_out)
n_nodes  = len(nodes_out)
total_km = edges_out["length_m"].sum() / 1000 if "length_m" in edges_out else 0
infra_counts = edges_out["infra_type"].value_counts()

print()
print("=" * 60)
print("Step 6 -- OSM Infrastructure COMPLETE (full cycling tags)")
print(f"Raw graph:        data/osm/raw/berlin_bike_graph.graphml")
print(f"Edges saved:      data/osm/cleaned/osm_cycling_edges.csv")
print(f"Nodes saved:      data/osm/cleaned/osm_cycling_nodes.csv")
print(f"Edges:            {n_edges:,}")
print(f"Nodes:            {n_nodes:,}")
print(f"Total network km: {total_km:,.0f} km")
print(f"Infrastructure breakdown (score):")
order = ["dedicated_path", "protected_lane", "painted_lane",
         "shared_lane", "mixed_traffic"]
for infra in order:
    count = infra_counts.get(infra, 0)
    pct   = 100 * count / n_edges if n_edges else 0
    score = QUALITY_SCORE[infra]
    print(f"  [{score}] {infra:<18}  {count:>7,}  ({pct:.1f}%)")
print("=" * 60)
