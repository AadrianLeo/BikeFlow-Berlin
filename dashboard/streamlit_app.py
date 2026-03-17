"""
dashboard/streamlit_app.py
BikeFlowBerlin V2 — Main Streamlit dashboard entrypoint.

Run locally:
    streamlit run dashboard/streamlit_app.py

Deploy to Streamlit Cloud:
    Set main file path to  dashboard/streamlit_app.py
    requirements:          dashboard/requirements.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the dashboard/ directory is on sys.path so `components.*` imports work
# both when running locally (from repo root) and on Streamlit Cloud.
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from components.charts import (
    annual_volume_chart,
    correlation_heatmap_chart,
    feature_importance_chart,
    gauge_chart,
    hourly_profile_chart,
    per_station_r2_chart,
    predicted_vs_actual_chart,
    weather_rain_chart,
    weather_temp_bands_chart,
)
from components.loaders import (
    FEATURE_COLS,
    load_model,
    load_modeled_mismatch,
    load_spatial_data,
    load_training_data,
)
from components.maps import coverage_map, mismatch_map, modeled_mismatch_map

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BikeFlowBerlin V2",
    page_icon="\U0001f6b2",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
INFRA_OPTIONS = {
    "0 — Mixed traffic":   0,
    "1 — Shared lane":     1,
    "2 — Painted lane":    2,
    "3 — Protected lane":  3,
    "4 — Dedicated path":  4,
}
BERLIN_AVG_BIKES_HOUR = 91.87


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("\U0001f6b2 BikeFlowBerlin V2")
st.sidebar.caption("Berlin cycling demand  |  2012–2024")

section = st.sidebar.radio(
    "Navigate",
    options=[
        "\U0001f3d9\ufe0f City Overview",
        "\U0001f4c8 Temporal Patterns",
        "\U0001f324\ufe0f Weather Impact",
        "\U0001f916 Model Performance",
        "\U0001f52e Live Prediction",
        "\U0001f5fa\ufe0f Infrastructure Audit",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption(
    "Data: Official Berlin counters, DWD weather,\n"
    "Telraam, Nextbike, OpenStreetMap\n\n"
    "Model: XGBoost (train 2012–2022, test 2023–2024)"
)

# ---------------------------------------------------------------------------
# Load data once — shared across all sections
# ---------------------------------------------------------------------------
with st.spinner("Initialising dashboard…"):
    df             = load_training_data()
    spatial        = load_spatial_data()
    model          = load_model()
    df_modeled_mm  = load_modeled_mismatch()


# ---------------------------------------------------------------------------
# Helper: unavailable banner
# ---------------------------------------------------------------------------
def _unavailable(name: str) -> None:
    st.warning(f"{name} is not available. Check the data file paths in `components/loaders.py`.")


# ===========================================================================
# SECTION 1 — City Overview
# ===========================================================================
def section_city_overview() -> None:
    st.title("\U0001f3d9\ufe0f City Overview")
    st.caption(
        "Berlin cycling demand aggregated across 38 official counter stations, "
        "205 Telraam sensors, and 3,751 Nextbike dock locations."
    )

    # --- Key metric cards ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("\U0001f6b2 Records", "2.2 M", help="Official counter rows 2012–2024")
    c2.metric("\U0001f4cd Stations", "38", help="Official Berlin counter stations")
    c3.metric("\U0001f916 R\u00b2", "0.69", help="XGBoost test-set R² (2023–2024)")
    c4.metric("\U0001f5fa\ufe0f OSM Edges", "436 k", help="Cycling network edges citywide")

    st.divider()

    # --- Coverage map ---
    st.subheader("Sensor Coverage Map")
    st.caption(
        "\U0001f535 Official Berlin (38)  \u2003"
        "\U0001f7e2 Telraam (205, clustered)  \u2003"
        "\U0001f534 Nextbike stations (3 751, hidden by default)"
    )
    if spatial is not None:
        m = coverage_map(spatial)
        st_folium(m, width=900, height=540, returned_objects=[])
    else:
        _unavailable("Spatial data")

    st.divider()

    # --- Dataset summary table ---
    st.subheader("Dataset Summary")
    table = pd.DataFrame(
        {
            "Dataset": [
                "Official counters",
                "Weather (DWD)",
                "Telraam sensors",
                "Nextbike stations",
                "OSM network",
            ],
            "Size": ["2.2 M rows", "113 k rows", "1.1 M rows", "3,751", "436 k edges"],
            "Coverage": ["2012–2024", "2012–2024", "2022–2025", "2026 snapshot", "Citywide"],
            "Purpose": [
                "Ground truth demand",
                "External weather drivers",
                "Spatial demand extension",
                "Infrastructure proxy",
                "Prediction & audit grid",
            ],
        }
    )
    st.table(table)

    st.info(
        "\U0001f4a1 **Coverage gap**: Official sensors cover < 0.01 % of the cycling network. "
        "The XGBoost model extrapolates demand to all 436 k OSM edges."
    )


# ===========================================================================
# SECTION 2 — Temporal Patterns
# ===========================================================================
def section_temporal_patterns() -> None:
    st.title("\U0001f4c8 Temporal Patterns")

    if df is None:
        _unavailable("Training data")
        return

    # --- Annual volume ---
    st.subheader("Annual Cycling Volume")
    year_min = int(df["year"].min())
    year_max = int(df["year"].max())
    year_range = st.slider(
        "Year range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
        step=1,
    )
    fig_annual = annual_volume_chart(df, year_range)
    st.plotly_chart(fig_annual, use_container_width=True)

    st.divider()

    # --- Hourly profile ---
    st.subheader("Average Hourly Demand Profile")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        month_name = st.selectbox(
            "Select month", MONTHS, index=5, key="month_select"
        )
    with col_b:
        show_both = st.toggle("Weekday vs Weekend", value=True)

    month_num = MONTHS.index(month_name) + 1
    fig_hourly = hourly_profile_chart(df, month_num, show_both)
    st.plotly_chart(fig_hourly, use_container_width=True)

    st.info(
        "\U0001f4a1 **Key finding**: Twin commuter peaks at **08:00** and **18:00** confirm "
        "cycling as a primary daily transport mode. Weekend profiles shift to a broader "
        "midday peak, indicating leisure riding."
    )

    st.divider()

    # --- Quick stats ---
    st.subheader("Summary Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Peak Month", "June", help="Highest average monthly demand")
    col2.metric("Lowest Month", "January", help="Lowest average monthly demand")
    col3.metric("Seasonal Swing", "~3\u00d7", help="Summer vs winter demand ratio")


# ===========================================================================
# SECTION 3 — Weather Impact
# ===========================================================================
def section_weather_impact() -> None:
    st.title("\U0001f324\ufe0f Weather Impact")

    if df is None:
        _unavailable("Training data")
        return

    # --- Metric cards ---
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "\U0001f321\ufe0f Temp correlation",
        "r = +0.363",
        help="Pearson r between temperature and bike_count",
    )
    c2.metric(
        "\U0001f4a7 Humidity correlation",
        "r = \u22120.398",
        help="Pearson r between humidity and bike_count",
    )
    c3.metric(
        "\U0001f327\ufe0f Rain impact",
        "\u221234.7 %",
        help="Average demand reduction on rainy hours vs dry",
    )

    st.divider()

    # --- Temperature bands + rain side by side ---
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Demand by Temperature Band")
        st.plotly_chart(
            weather_temp_bands_chart(df), use_container_width=True
        )
    with col_right:
        st.subheader("Demand: Dry vs Rainy")
        st.plotly_chart(
            weather_rain_chart(df), use_container_width=True
        )

    st.divider()

    # --- Correlation heatmap ---
    st.subheader("Correlation Matrix — Weather \u00d7 Demand")
    st.plotly_chart(correlation_heatmap_chart(df), use_container_width=True)

    st.info(
        "\U0001f4a1 **Weather elasticity**: Demand on warm days (>20\u00b0C) is "
        "100–200 % higher than on cold days (<8\u00b0C). "
        "Rain suppresses cycling by ~35 %, wind has a smaller but consistent effect."
    )


# ===========================================================================
# SECTION 4 — Model Performance
# ===========================================================================
def section_model_performance() -> None:
    st.title("\U0001f916 Model Performance")
    st.caption("XGBoost trained on 2012–2022 data, evaluated on 2023–2024 hold-out set.")

    # --- Metric cards ---
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", "63.64 bikes/hour", help="Root Mean Squared Error on test set")
    c2.metric("MAE", "35.51 bikes/hour", help="Mean Absolute Error on test set")
    c3.metric("R\u00b2", "0.6925", help="Coefficient of determination on test set")

    if model is None:
        _unavailable("Model (model.pkl)")
        return
    if df is None:
        _unavailable("Training data")
        return

    st.divider()

    # --- Feature importances ---
    st.subheader("Feature Importances")
    st.plotly_chart(
        feature_importance_chart(model, FEATURE_COLS), use_container_width=True
    )

    st.divider()

    # --- Prepare test set (lazy, cached inline) ---
    @st.cache_data(show_spinner="Building test-set predictions…")
    def _get_test_df(_df_hash: int) -> pd.DataFrame:
        test_cols = FEATURE_COLS + ["bike_count", "station_id"]
        return (
            df[df["year"] >= 2023]
            .dropna(subset=test_cols)
            [test_cols]
            .reset_index(drop=True)
        )

    df_test = _get_test_df(len(df))

    if len(df_test) == 0:
        st.warning("No test-set data found (year >= 2023). Check training CSV.")
        return

    # --- Predicted vs Actual ---
    st.subheader("Predicted vs Actual (5 000-row sample from test set)")
    st.plotly_chart(
        predicted_vs_actual_chart(df_test, model, FEATURE_COLS),
        use_container_width=True,
    )

    st.divider()

    # --- Per-station R² ---
    st.subheader("Per-Station R\u00b2 — Test Set")
    st.plotly_chart(
        per_station_r2_chart(df_test, model, FEATURE_COLS),
        use_container_width=True,
    )

    st.info(
        "\U0001f4a1 **Key insight**: Best station **21-NK-MAY** achieves R\u00b2 = 0.908. "
        "The model explains **69 %** of total hourly demand variance. "
        "Residual variance reflects unmeasured events (protests, festivals, road closures)."
    )


# ===========================================================================
# SECTION 5 — Live Prediction
# ===========================================================================
def section_live_prediction() -> None:
    st.title("\U0001f52e Live Prediction")
    st.caption(
        "Adjust the sliders to simulate conditions and see the XGBoost model's "
        "real-time demand forecast for a typical Berlin station."
    )

    if model is None:
        _unavailable("Model (model.pkl)")
        return

    col_inputs, col_output = st.columns([1, 1], gap="large")

    # --- LEFT: Inputs ---
    with col_inputs:
        st.subheader("Conditions")

        hour = st.slider("Hour of day", 0, 23, 17, step=1)

        month_name = st.select_slider(
            "Month",
            options=MONTHS,
            value="Jun",
        )
        month_num = MONTHS.index(month_name) + 1

        temperature = st.slider(
            "Temperature (\u00b0C)", -10.0, 40.0, 22.0, step=0.5
        )
        precipitation = st.slider(
            "Precipitation (mm)", 0.0, 20.0, 0.0, step=0.1
        )
        wind_speed = st.slider(
            "Wind speed (m/s)", 0.0, 15.0, 3.0, step=0.1
        )

        is_weekend = st.toggle("Weekend", value=False)
        is_holiday = st.toggle("Public Holiday", value=False)

        infra_label = st.selectbox(
            "Infrastructure quality",
            options=list(INFRA_OPTIONS.keys()),
            index=0,
        )
        infra_score = INFRA_OPTIONS[infra_label]

        osm_distance = st.slider(
            "Distance to nearest OSM edge (m)", 0, 500, 50, step=5
        )

    # --- RIGHT: Output ---
    with col_output:
        st.subheader("Prediction")

        # Build feature vector in exact training order
        hour_sin  = float(np.sin(2 * np.pi * hour / 24))
        hour_cos  = float(np.cos(2 * np.pi * hour / 24))
        month_sin = float(np.sin(2 * np.pi * month_num / 12))
        month_cos = float(np.cos(2 * np.pi * month_num / 12))
        weekday   = 6 if is_weekend else 2   # Sat=6, Wed=2

        feature_vector = [
            hour_sin,
            hour_cos,
            month_sin,
            month_cos,
            weekday,
            int(is_weekend),
            int(is_holiday),
            temperature,
            precipitation,
            wind_speed,
            float(infra_score),
            float(osm_distance),
        ]

        raw_pred = model.predict([feature_vector])[0]
        prediction = max(0.0, float(raw_pred))

        st.metric(
            "\U0001f6b2 Predicted demand",
            f"{prediction:,.0f} bikes/hour",
            delta=f"{prediction - BERLIN_AVG_BIKES_HOUR:+.1f} vs city avg",
        )

        st.plotly_chart(
            gauge_chart(prediction, max_val=500),
            use_container_width=True,
        )

        st.markdown(
            f"Under these conditions, expect approximately "
            f"**{prediction:,.0f} bikes/hour** at a typical Berlin station."
        )

        cmp_pct = ((prediction / BERLIN_AVG_BIKES_HOUR) - 1) * 100
        direction = "above" if cmp_pct >= 0 else "below"
        st.caption(
            f"Berlin city average: **{BERLIN_AVG_BIKES_HOUR} bikes/hour** "
            f"— your scenario is **{abs(cmp_pct):.1f} % {direction}** the average."
        )

        with st.expander("Feature vector (debug)"):
            labels = [
                "hour_sin", "hour_cos", "month_sin", "month_cos",
                "weekday", "is_weekend", "is_holiday",
                "temperature", "precipitation", "wind_speed",
                "infrastructure_quality_score", "nearest_osm_distance_m",
            ]
            st.dataframe(
                pd.DataFrame({"feature": labels, "value": feature_vector}),
                use_container_width=True,
                hide_index=True,
            )


# ===========================================================================
# SECTION 6 — Infrastructure Audit
# ===========================================================================
def section_infrastructure_audit() -> None:
    st.title("\U0001f5fa\ufe0f Infrastructure Audit")
    st.caption(
        "Mismatch analysis: locations where cycling demand is out of step "
        "with the quality of local cycling infrastructure."
    )

    st.markdown(
        """
**Infrastructure Audit Overview**

- **Observed mismatch** → based on real sensor data (high accuracy, limited coverage)
- **Modeled mismatch** → citywide estimation using ML (full coverage, predictive)

Together, they provide both local evidence and system-wide insight.
        """
    )

    tabs = st.tabs(["📍 Observed Mismatch", "🧠 Citywide Modeled Mismatch"])

    # -----------------------------------------------------------------------
    # TAB 1 — Observed Mismatch (existing EDA-based logic)
    # -----------------------------------------------------------------------
    with tabs[0]:
        st.info(
            "Observed mismatch is based on measured demand from official Berlin counters "
            "and Telraam sensors. This reflects real-world usage but is limited to "
            "monitored locations."
        )

        if spatial is None:
            _unavailable("Spatial data")
        else:
            s = spatial[spatial["avg_daily_demand"].notna()].copy()

            if len(s) == 0:
                st.warning("No rows with avg_daily_demand in spatial_coverage_data.csv.")
            else:
                s["demand_rank"] = s["avg_daily_demand"].rank(pct=True)
                s["infra_rank"]  = s["infrastructure_quality_score"].rank(pct=True)
                s["mismatch"]    = s["demand_rank"] - s["infra_rank"]
                s["status"] = s["mismatch"].apply(
                    lambda x: "underserved" if x > 0.3 else ("overbuilt" if x < -0.3 else "matched")
                )

                # District filter
                districts = ["All Berlin"] + sorted(s["district"].dropna().unique().tolist())
                col_filter, _ = st.columns([2, 4])
                with col_filter:
                    selected_district = st.selectbox(
                        "Filter by District", districts, index=0, key="obs_district"
                    )

                filtered = s if selected_district == "All Berlin" else s[s["district"] == selected_district]

                # Metric cards
                n_under   = int((filtered["status"] == "underserved").sum())
                n_matched = int((filtered["status"] == "matched").sum())
                n_over    = int((filtered["status"] == "overbuilt").sum())

                c1, c2, c3 = st.columns(3)
                c1.metric("\U0001f534 Underserved", n_under,
                          help="High demand, low infrastructure quality (mismatch > 0.3)")
                c2.metric("\U0001f7e2 Matched",     n_matched,
                          help="Demand and infrastructure quality are aligned (|mismatch| \u2264 0.3)")
                c3.metric("\U0001f535 Overbuilt",   n_over,
                          help="High infra quality, low demand (mismatch < \u22120.3)")

                st.divider()

                # Mismatch map
                st.subheader("Infrastructure Mismatch Map")
                st.caption(
                    "\U0001f534 Underserved  \u2003\U0001f7e2 Matched  \u2003\U0001f535 Overbuilt"
                )
                m = mismatch_map(s, district_filter=selected_district)
                st_folium(m, width=900, height=520, returned_objects=[])

                st.divider()

                # Top-10 underserved table
                st.subheader("Top 10 Underserved Locations")
                underserved = (
                    filtered[filtered["status"] == "underserved"]
                    .nlargest(10, "mismatch")
                    [["location_id", "district", "avg_daily_demand",
                      "infrastructure_type", "infrastructure_quality_score", "mismatch"]]
                    .copy()
                )

                if len(underserved) == 0:
                    st.info("No underserved locations in the selected district.")
                else:
                    underserved = underserved.rename(
                        columns={
                            "location_id": "Location",
                            "district": "District",
                            "avg_daily_demand": "Demand (bikes/day)",
                            "infrastructure_type": "Infrastructure",
                            "infrastructure_quality_score": "Quality Score",
                            "mismatch": "Mismatch Score",
                        }
                    )
                    underserved["Demand (bikes/day)"] = underserved[
                        "Demand (bikes/day)"
                    ].map("{:,.0f}".format)
                    underserved["Mismatch Score"] = underserved["Mismatch Score"].map(
                        "{:+.3f}".format
                    )
                    st.dataframe(underserved, use_container_width=True, hide_index=True)

                st.divider()

                st.info(
                    "\U0001f4a1 **Methodology**: Mismatch score = demand percentile rank \u2212 "
                    "infrastructure quality percentile rank. "
                    "A score > 0.3 means the location sees more cyclists than the infrastructure quality "
                    "would suggest; < \u22120.3 means over-provision. "
                    "This is a planning screening tool — field verification is recommended."
                )

                with st.expander("Demand by Infrastructure Type (all sensors with demand data)"):
                    infra_summary = (
                        s.groupby("infrastructure_type")["avg_daily_demand"]
                        .agg(["mean", "count"])
                        .rename(columns={"mean": "Avg Daily Demand", "count": "Sensors"})
                        .sort_values("Avg Daily Demand", ascending=False)
                        .reset_index()
                        .rename(columns={"infrastructure_type": "Infrastructure Type"})
                    )
                    infra_summary["Avg Daily Demand"] = infra_summary["Avg Daily Demand"].map(
                        "{:,.0f}".format
                    )
                    st.dataframe(infra_summary, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # TAB 2 — Citywide Modeled Mismatch (new, model-based)
    # -----------------------------------------------------------------------
    with tabs[1]:
        st.info(
            "Citywide modeled mismatch extends demand estimation to the full cycling network "
            "using the trained XGBoost model. This identifies potential infrastructure gaps "
            "even where no sensors exist."
        )

        # Debug info expander
        with st.expander("Debug Information"):
            from components.loaders import MODELED_MISMATCH_PATH
            full_path = MODELED_MISMATCH_PATH.resolve()
            st.write(f"**File path:** `{full_path}`")
            st.write(f"**Exists:** {full_path.exists()}")
            if df_modeled_mm is not None:
                st.write(f"**Rows loaded:** {len(df_modeled_mm):,}")
                st.write(f"**Columns:** {list(df_modeled_mm.columns)}")
            else:
                st.write("**Data:** None (not loaded)")

        if df_modeled_mm is None:
            st.warning(
                "Modeled mismatch data not found. "
                "Please ensure `data/final_merged/osm_modeled_mismatch.csv` exists "
                "and was exported from `Notebooks/02_Modeling.ipynb`."
            )
            return

        # Trust notebook-exported mismatch/status; only compute as fallback
        if "mismatch" not in df_modeled_mm.columns or "status" not in df_modeled_mm.columns:
            df_modeled_mm["demand_rank"] = df_modeled_mm["predicted_demand"].rank(pct=True)
            df_modeled_mm["infra_rank"]  = df_modeled_mm["infrastructure_quality_score"].rank(pct=True)
            df_modeled_mm["mismatch"]    = df_modeled_mm["demand_rank"] - df_modeled_mm["infra_rank"]
            df_modeled_mm["status"] = "matched"
            df_modeled_mm.loc[df_modeled_mm["mismatch"] >  0.3, "status"] = "underserved"
            df_modeled_mm.loc[df_modeled_mm["mismatch"] < -0.3, "status"] = "overbuilt"

        total_edges = len(df_modeled_mm)

        # Optional district filter
        has_district = "district" in df_modeled_mm.columns
        if has_district:
            mod_districts = ["All Berlin"] + sorted(
                df_modeled_mm["district"].dropna().unique().tolist()
            )
            col_filter2, _ = st.columns([2, 4])
            with col_filter2:
                selected_mod_district = st.selectbox(
                    "Filter by District", mod_districts, index=0, key="mod_district"
                )
            df_filtered = (
                df_modeled_mm if selected_mod_district == "All Berlin"
                else df_modeled_mm[df_modeled_mm["district"] == selected_mod_district]
            )
        else:
            df_filtered = df_modeled_mm

        # Metric cards — counts from the FULL filtered dataset (not sampled)
        n_under   = int((df_filtered["status"] == "underserved").sum())
        n_matched = int((df_filtered["status"] == "matched").sum())
        n_over    = int((df_filtered["status"] == "overbuilt").sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("\U0001f5fa\ufe0f Total Edges", f"{total_edges:,}",
                  help="Total OSM cycling network edges in the modeled dataset")
        c2.metric("\U0001f534 Underserved", f"{n_under:,}",
                  help="High demand, low infrastructure quality (mismatch > 0.3)")
        c3.metric("\U0001f7e2 Matched", f"{n_matched:,}",
                  help="Demand and infrastructure quality are aligned (|mismatch| \u2264 0.3)")
        c4.metric("\U0001f535 Overbuilt", f"{n_over:,}",
                  help="High infra quality, low predicted demand (mismatch < \u22120.3)")

        st.divider()

        # --- Slider for max edges to display ---
        max_edges = st.slider(
            "Max edges to display",
            min_value=5000,
            max_value=100000,
            value=20000,
            step=5000,
            key="mod_max_edges",
        )

        # Sample: sort by absolute mismatch descending, take top N
        df_map = df_filtered.dropna(subset=["lat", "lon", "mismatch"])
        total_valid = len(df_map)
        if total_valid > max_edges:
            df_map = (
                df_map
                .reindex(df_map["mismatch"].abs().sort_values(ascending=False).index)
                .head(max_edges)
            )

        st.caption(
            f"Map showing top {len(df_map):,} edges by absolute mismatch score "
            f"(of {total_valid:,} with valid coordinates)."
        )

        # Mismatch map
        st.subheader("Citywide Infrastructure Mismatch Map")
        st.caption(
            "\U0001f534 Underserved  \u2003\U0001f7e2 Matched  \u2003\U0001f535 Overbuilt"
        )
        m2 = modeled_mismatch_map(df_map)
        st_folium(m2, width=900, height=520, returned_objects=[])

        st.divider()

        # Top-10 underserved table
        st.subheader("Top 10 Underserved Edges (Modeled)")
        top_cols = [c for c in ["district", "predicted_demand",
                                "infrastructure_quality_score",
                                "mismatch"] if c in df_filtered.columns]
        top10 = (
            df_filtered[df_filtered["status"] == "underserved"]
            .nlargest(10, "mismatch")
            [top_cols]
            .copy()
        )

        if len(top10) == 0:
            st.info("No underserved edges in the selected filter.")
        else:
            top10 = top10.rename(columns={
                "district": "District",
                "predicted_demand": "Predicted Demand",
                "infrastructure_quality_score": "Quality Score",
                "mismatch": "Mismatch",
            })
            if "Predicted Demand" in top10.columns:
                top10["Predicted Demand"] = top10["Predicted Demand"].map("{:,.0f}".format)
            if "Mismatch" in top10.columns:
                top10["Mismatch"] = top10["Mismatch"].map("{:+.3f}".format)
            st.dataframe(top10, use_container_width=True, hide_index=True)


# ===========================================================================
# Router
# ===========================================================================
_ROUTES = {
    "\U0001f3d9\ufe0f City Overview":        section_city_overview,
    "\U0001f4c8 Temporal Patterns":          section_temporal_patterns,
    "\U0001f324\ufe0f Weather Impact":        section_weather_impact,
    "\U0001f916 Model Performance":          section_model_performance,
    "\U0001f52e Live Prediction":            section_live_prediction,
    "\U0001f5fa\ufe0f Infrastructure Audit": section_infrastructure_audit,
}

if section in _ROUTES:
    _ROUTES[section]()
else:
    st.error(f"Unknown section: {section}")
