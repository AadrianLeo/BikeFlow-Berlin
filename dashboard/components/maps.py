"""
dashboard/components/maps.py
Folium map builders for the BikeFlowBerlin V2 dashboard.
Every function returns a folium.Map ready for st_folium().
"""
from __future__ import annotations

import folium
import pandas as pd
from folium.plugins import MarkerCluster

# Import sensor name lookup for human-readable names
from .loaders import get_sensor_display_name

# Berlin centre
_LAT = 52.52
_LON = 13.405
_ZOOM = 11

# Colour palette (keys match actual source values in spatial_coverage_data.csv)
_SOURCE_COLOURS = {
    "official_berlin": "#1f77b4",   # blue
    "telraam":         "#2ca02c",   # green
    "nextbike":        "#d62728",   # red
}

_MISMATCH_COLOURS = {
    "underserved": "#d62728",   # red
    "matched":     "#2ca02c",   # green
    "overbuilt":   "#1f77b4",   # blue
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_demand(val) -> str:
    """Format demand value nicely."""
    if pd.isna(val):
        return "N/A"
    return f"{val:,.0f} bikes/day"


def _fmt_rank(val) -> str:
    """Format rank value (0-1 percentile) with 2 decimals."""
    if pd.isna(val):
        return "N/A"
    return f"{val:.2f}"


def _fmt_score(val) -> str:
    """Format infra quality score with 2 decimals."""
    if pd.isna(val):
        return "N/A"
    try:
        return f"{float(val):.2f}"
    except (ValueError, TypeError):
        return str(val)


def _fmt_mismatch(val) -> str:
    """Format mismatch with explicit sign and 2 decimals."""
    if pd.isna(val):
        return "N/A"
    v = float(val)
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}"


def _base_map(zoom: int = _ZOOM) -> folium.Map:
    return folium.Map(
        location=[_LAT, _LON],
        zoom_start=zoom,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )


# ---------------------------------------------------------------------------
# Popup HTML builders (notebook-style formatting)
# ---------------------------------------------------------------------------

_POPUP_STYLE = """
<div style="
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 13px;
    line-height: 1.6;
    color: #222;
">
{content}
</div>
"""


def _build_coverage_popup(row: pd.Series, source_label: str) -> str:
    """Build rich HTML popup for coverage map locations with human-readable names."""
    loc_id = row.get("location_id", "Unknown")
    source = row.get("source", "")
    district = row.get("district") or "N/A"
    infra = row.get("infrastructure_type") or "N/A"
    demand = row.get("avg_daily_demand")

    # Get human-readable name (matches notebook logic)
    display_name = get_sensor_display_name(str(loc_id), source)
    title = display_name if display_name else str(loc_id)

    demand_line = (
        f"<b>{demand:,.0f}</b> bikes/day" if pd.notna(demand) else "<i>N/A</i>"
    )

    # Show location_id below title only if we have a human-readable name
    id_line = (
        f'<div style="color:#666; font-size:11px; margin-bottom:6px;">{loc_id}</div>'
        if display_name else ""
    )
    source_line = f'<div style="color:#888; font-size:10px; margin-bottom:4px;">{source_label}</div>'

    content = f"""
<div style="font-size:15px; font-weight:600; margin-bottom:2px;">{title}</div>
{id_line}
{source_line}
<hr style="margin:6px 0; border:none; border-top:1px solid #ddd;">
<table style="font-size:12px; border-collapse:collapse;">
  <tr><td style="color:#666; padding-right:8px;">District</td><td>{district}</td></tr>
  <tr><td style="color:#666; padding-right:8px;">Avg daily demand</td><td>{demand_line}</td></tr>
  <tr><td style="color:#666; padding-right:8px;">Infrastructure</td><td>{infra}</td></tr>
</table>
"""
    return _POPUP_STYLE.format(content=content)


def _build_nextbike_popup(row: pd.Series) -> str:
    """Build HTML popup for Nextbike stations (no demand data)."""
    loc_id = row.get("location_id", "Unknown")
    district = row.get("district") or "N/A"

    content = f"""
<div style="font-size:15px; font-weight:600; margin-bottom:2px;">{loc_id}</div>
<div style="color:#666; font-size:11px; margin-bottom:6px;">nextbike_berlin</div>
<hr style="margin:6px 0; border:none; border-top:1px solid #ddd;">
<table style="font-size:12px; border-collapse:collapse;">
  <tr><td style="color:#666; padding-right:8px;">District</td><td>{district}</td></tr>
  <tr><td style="color:#666; padding-right:8px;">Demand data</td><td><i>not available</i></td></tr>
</table>
<div style="font-size:11px; color:#888; margin-top:6px;">Location reference only</div>
"""
    return _POPUP_STYLE.format(content=content)


# ---------------------------------------------------------------------------
# Section 1 — Coverage Map
# ---------------------------------------------------------------------------

def coverage_map(spatial: pd.DataFrame) -> folium.Map:
    """
    Folium map with three toggleable layers:
      - Official Berlin sensors (blue circles)
      - Telraam sensors (green, clustered)
      - Nextbike stations (red, small, hidden by default)
    """
    m = _base_map()

    # Normalise source values so filter strings always match
    spatial = spatial.copy()
    spatial["source"] = spatial["source"].str.strip().str.lower()

    # --- Official Berlin (blue circles, radius=10) ---
    official = spatial[spatial["source"] == "official_berlin"].dropna(
        subset=["lat", "lon"]
    )
    fg_official = folium.FeatureGroup(
        name=f"Official Berlin Sensors ({len(official)})", show=True
    )
    for _, row in official.iterrows():
        district = row.get("district") or "N/A"
        loc_id = row.get("location_id", "")
        # Get human-readable name for tooltip
        display_name = get_sensor_display_name(str(loc_id), "official_berlin")
        tooltip_title = display_name if display_name else str(loc_id)

        popup_html = _build_coverage_popup(row, "official_berlin")
        tooltip = f"{tooltip_title} | {district}"
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=10,
            color=_SOURCE_COLOURS["official_berlin"],
            fill=True,
            fill_color=_SOURCE_COLOURS["official_berlin"],
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=tooltip,
        ).add_to(fg_official)
    fg_official.add_to(m)

    # --- Telraam (green, clustered, radius=6) ---
    telraam = spatial[spatial["source"] == "telraam"].dropna(
        subset=["lat", "lon"]
    )
    fg_telraam = folium.FeatureGroup(
        name=f"Telraam Sensors ({len(telraam)})", show=True
    )
    cluster = MarkerCluster(
        options={"maxClusterRadius": 30, "disableClusteringAtZoom": 14}
    )
    for _, row in telraam.iterrows():
        district = row.get("district") or "N/A"
        loc_id = row.get("location_id", "")
        # Get human-readable name for tooltip
        display_name = get_sensor_display_name(str(loc_id), "telraam")
        tooltip_title = display_name if display_name else str(loc_id)

        popup_html = _build_coverage_popup(row, "telraam_berlin")
        tooltip = f"{tooltip_title} | {district}"
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color=_SOURCE_COLOURS["telraam"],
            fill=True,
            fill_color=_SOURCE_COLOURS["telraam"],
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=tooltip,
        ).add_to(cluster)
    cluster.add_to(fg_telraam)
    fg_telraam.add_to(m)

    # --- Nextbike (red circles, radius=3, hidden by default) ---
    nextbike = spatial[spatial["source"] == "nextbike"].dropna(
        subset=["lat", "lon"]
    )
    fg_nextbike = folium.FeatureGroup(
        name=f"Nextbike Stations ({len(nextbike)}, snapshot)", show=False
    )
    for _, row in nextbike.iterrows():
        district = row.get("district") or "N/A"
        popup_html = _build_nextbike_popup(row)
        tooltip = f"{row['location_id']} | {district}"
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            color=_SOURCE_COLOURS["nextbike"],
            fill=True,
            fill_color=_SOURCE_COLOURS["nextbike"],
            fill_opacity=0.5,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=tooltip,
        ).add_to(fg_nextbike)
    fg_nextbike.add_to(m)

    # Coverage legend (notebook-style)
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px; left: 30px; z-index: 1000;
        background-color: rgba(14,17,23,0.92);
        border: 1px solid #444;
        padding: 14px 18px;
        border-radius: 8px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 13px;
        color: #fafafa;
        line-height: 2.0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    ">
      <div style="font-weight:600; margin-bottom:6px; font-size:14px;">Data Sources</div>
      <span style="color:#1f77b4;">&#9679;</span> Official counters <span style="color:#888;">&mdash; demand data</span><br>
      <span style="color:#2ca02c;">&#9679;</span> Telraam sensors <span style="color:#888;">&mdash; demand data</span><br>
      <span style="color:#d62728;">&#9679;</span> Nextbike stations <span style="color:#888;">&mdash; location only</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ---------------------------------------------------------------------------
# Section 6 — Infrastructure Mismatch Map (Observed)
# ---------------------------------------------------------------------------

def _build_mismatch_popup(row: pd.Series, status: str, colour: str) -> str:
    """Build rich HTML popup for mismatch map locations with human-readable names."""
    loc_id = row.get("location_id", "Unknown")
    source = row.get("source", "")
    district = row.get("district") or "N/A"
    infra_type = row.get("infrastructure_type") or "N/A"

    # Get human-readable name (matches notebook logic)
    display_name = get_sensor_display_name(str(loc_id), source)
    title = display_name if display_name else str(loc_id)

    # Demand formatting
    demand = row.get("avg_daily_demand")
    demand_str = f"<b>{demand:,.0f}</b>" if pd.notna(demand) else "<i>N/A</i>"

    # Infra quality score
    infra_score = row.get("infrastructure_quality_score")
    score_str = _fmt_score(infra_score)

    # Mismatch value with sign
    mismatch_val = row.get("mismatch", 0.0)
    mismatch_str = _fmt_mismatch(mismatch_val)

    # Optional rank columns (graceful skip if missing)
    rank_rows = ""
    if "demand_rank" in row.index and pd.notna(row.get("demand_rank")):
        rank_rows += f'<tr><td style="color:#666; padding-right:10px;">Demand rank</td><td>{_fmt_rank(row["demand_rank"])}</td></tr>'
    if "infra_rank" in row.index and pd.notna(row.get("infra_rank")):
        rank_rows += f'<tr><td style="color:#666; padding-right:10px;">Infra rank</td><td>{_fmt_rank(row["infra_rank"])}</td></tr>'

    # Status label with colour
    status_label = status.replace("_", " ").title()

    # Show location_id below title only if we have a human-readable name
    id_line = (
        f'<div style="color:#666; font-size:11px; margin-bottom:2px;">{loc_id}</div>'
        if display_name else ""
    )
    source_line = f'<div style="color:#888; font-size:10px; margin-bottom:4px;">{source}</div>'

    content = f"""
<div style="font-size:15px; font-weight:600; margin-bottom:2px;">{title}</div>
{id_line}
{source_line}
<hr style="margin:6px 0; border:none; border-top:1px solid #ddd;">
<table style="font-size:12px; border-collapse:collapse;">
  <tr><td style="color:#666; padding-right:10px;">District</td><td>{district}</td></tr>
  <tr><td style="color:#666; padding-right:10px;">Avg daily demand</td><td>{demand_str} bikes/day</td></tr>
  <tr><td style="color:#666; padding-right:10px;">Infrastructure</td><td>{infra_type}</td></tr>
  <tr><td style="color:#666; padding-right:10px;">Quality score</td><td>{score_str}</td></tr>
  {rank_rows}
</table>
<hr style="margin:8px 0; border:none; border-top:1px solid #ddd;">
<div style="font-size:13px;">
  <b>Mismatch: {mismatch_str}</b>
  <span style="color:{colour}; margin-left:6px;">&#9632;</span>
  <span style="font-weight:500;"> {status_label}</span>
</div>
"""
    return _POPUP_STYLE.format(content=content)


def mismatch_map(
    spatial_with_mismatch: pd.DataFrame,
    district_filter: str = "All Berlin",
) -> folium.Map:
    """
    Folium map coloured by infrastructure mismatch status:
      Red  = underserved (mismatch > 0.3)
      Green = matched
      Blue  = overbuilt (mismatch < -0.3)

    Expects columns: lat, lon, location_id, district, avg_daily_demand,
                     infrastructure_type, mismatch, status
    Optional columns: demand_rank, infra_rank
    """
    m = _base_map()

    display = spatial_with_mismatch.dropna(subset=["lat", "lon", "mismatch"])
    if district_filter != "All Berlin":
        display = display[display["district"] == district_filter]

    for _, row in display.iterrows():
        status = row.get("status", "matched")
        colour = _MISMATCH_COLOURS.get(status, "#888888")

        popup_html = _build_mismatch_popup(row, status, colour)

        # Get human-readable name for tooltip
        loc_id = row.get("location_id", "")
        source = row.get("source", "")
        display_name = get_sensor_display_name(str(loc_id), source)
        tooltip_title = display_name if display_name else str(loc_id)

        mismatch_val = row.get("mismatch", 0.0)
        mismatch_str = _fmt_mismatch(mismatch_val)
        district = row.get("district") or "N/A"
        tooltip = f"{tooltip_title} | {district} | mismatch={mismatch_str}"

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=8,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=tooltip,
        ).add_to(m)

    # Colour legend (notebook-style)
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px; left: 30px; z-index: 1000;
        background-color: rgba(14,17,23,0.92);
        border: 1px solid #444;
        padding: 14px 18px;
        border-radius: 8px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 13px;
        color: #fafafa;
        line-height: 2.0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    ">
      <div style="font-weight:600; margin-bottom:6px; font-size:14px;">Mismatch Status</div>
      <span style="color:#d62728;">&#9632;</span> Underserved <span style="color:#888;">&mdash; demand &gt; infrastructure</span><br>
      <span style="color:#2ca02c;">&#9632;</span> Well matched<br>
      <span style="color:#1f77b4;">&#9632;</span> Overbuilt <span style="color:#888;">&mdash; infrastructure &gt; demand</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ---------------------------------------------------------------------------
# Section 6b — Modeled Citywide Mismatch Map
# ---------------------------------------------------------------------------

def _build_modeled_popup(row: pd.Series, status: str, colour: str) -> str:
    """Build rich HTML popup for modeled mismatch map (OSM edges) with street names."""
    # Location label: use name if available, else "Unnamed road" (matches notebook)
    street_name = row.get("name")
    if pd.isna(street_name) or not street_name:
        title = "Unnamed road"
    else:
        title = str(street_name)

    # Show osmid beneath the name (matches notebook: "OSM {osmid}")
    osmid = row.get("osmid")
    osmid_str = ""
    if pd.notna(osmid):
        osmid_display = str(osmid)
        if len(osmid_display) > 22:
            osmid_display = osmid_display[:22] + "\u2026"
        osmid_str = f'<div style="color:#666; font-size:11px; margin-bottom:4px;">OSM {osmid_display}</div>'

    district = row.get("district") or "N/A"

    # Predicted demand (from local_demand or predicted_demand)
    demand = row.get("predicted_demand") or row.get("local_demand")
    demand_str = f"<b>{demand:,.1f}</b>" if pd.notna(demand) else "<i>N/A</i>"

    # Infra quality score
    infra_score = row.get("infra_quality_score") or row.get("infrastructure_quality_score")
    score_str = _fmt_score(infra_score)

    # Infra type if available
    infra_type = row.get("infra_type") or row.get("infrastructure_type") or ""

    # Mismatch value with sign
    mismatch_val = row.get("mismatch", 0.0)
    mismatch_str = _fmt_mismatch(mismatch_val)

    # Optional rank columns (graceful skip if missing)
    rank_rows = ""
    if "demand_rank" in row.index and pd.notna(row.get("demand_rank")):
        rank_rows += f'<tr><td style="color:#666; padding-right:10px;">Demand rank</td><td>{_fmt_rank(row["demand_rank"])}</td></tr>'
    if "infra_rank" in row.index and pd.notna(row.get("infra_rank")):
        rank_rows += f'<tr><td style="color:#666; padding-right:10px;">Infra rank</td><td>{_fmt_rank(row["infra_rank"])}</td></tr>'

    # Status label with colour
    status_label = status.replace("_", " ").title()

    # Infra type row if available
    infra_row = ""
    if infra_type:
        infra_row = f'<tr><td style="color:#666; padding-right:10px;">Type</td><td>{infra_type}</td></tr>'

    content = f"""
<div style="font-size:14px; font-weight:600; margin-bottom:2px;">{title}</div>
{osmid_str}
<hr style="margin:6px 0; border:none; border-top:1px solid #ddd;">
<table style="font-size:12px; border-collapse:collapse;">
  <tr><td style="color:#666; padding-right:10px;">District</td><td>{district}</td></tr>
  <tr><td style="color:#666; padding-right:10px;">Predicted demand</td><td>{demand_str} bikes/day</td></tr>
  {infra_row}
  <tr><td style="color:#666; padding-right:10px;">Quality score</td><td>{score_str}</td></tr>
  {rank_rows}
</table>
<hr style="margin:8px 0; border:none; border-top:1px solid #ddd;">
<div style="font-size:13px;">
  <b>Mismatch: {mismatch_str}</b>
  <span style="color:{colour}; margin-left:6px;">&#9679;</span>
  <span style="font-weight:500;"> {status_label}</span>
</div>
"""
    return _POPUP_STYLE.format(content=content)


def modeled_mismatch_map(df: pd.DataFrame) -> folium.Map:
    """
    Folium map coloured by modeled infrastructure mismatch status.
    Matches the visual logic from modelling notebook Cell 7.

    Expects columns: lat, lon, predicted_demand, infrastructure_quality_score,
                     mismatch, status
    Optional columns: district, infra_type, local_demand, demand_rank, infra_rank, name, osmid
    """
    m = _base_map()

    display = df.dropna(subset=["lat", "lon", "mismatch"])

    for _, row in display.iterrows():
        status = row.get("status", "matched")
        colour = _MISMATCH_COLOURS.get(status, "#888888")

        popup_html = _build_modeled_popup(row, status, colour)

        # Tooltip: use street name if available
        street_name = row.get("name")
        if pd.isna(street_name) or not street_name:
            tooltip_title = "Unnamed"
        else:
            tooltip_title = str(street_name)
            if len(tooltip_title) > 25:
                tooltip_title = tooltip_title[:25] + "\u2026"

        mismatch_val = row.get("mismatch", 0.0)
        mismatch_str = _fmt_mismatch(mismatch_val)
        district = row.get("district") or ""
        tooltip_parts = [tooltip_title, f"mismatch={mismatch_str}", status]
        if district:
            tooltip_parts.insert(1, district)
        tooltip = " | ".join(tooltip_parts)

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=tooltip,
        ).add_to(m)

    # Legend (notebook-style)
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px; left: 30px; z-index: 1000;
        background-color: rgba(14,17,23,0.92);
        border: 1px solid #444;
        padding: 14px 18px;
        border-radius: 8px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 13px;
        color: #fafafa;
        line-height: 2.0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    ">
      <div style="font-weight:600; margin-bottom:6px; font-size:14px;">Infrastructure Mismatch (OSM Edges)</div>
      <span style="color:#d62728;">&#9679;</span> Underserved <span style="color:#888;">&mdash; high demand, poor infra</span><br>
      <span style="color:#2ca02c;">&#9679;</span> Well matched<br>
      <span style="color:#1f77b4;">&#9679;</span> Overbuilt <span style="color:#888;">&mdash; good infra, lower demand</span><br>
      <div style="font-size:11px; color:#888; margin-top:6px; border-top:1px solid #444; padding-top:6px;">
        demand = nearest sensor avg_daily_demand
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m
