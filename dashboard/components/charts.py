"""
dashboard/components/charts.py
All Plotly chart builders for the BikeFlowBerlin V2 dashboard.
Every function accepts processed pandas DataFrames and returns a Plotly Figure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score

_DARK = "plotly_dark"
_BLUE = "#1f77b4"
_GREEN = "#2ca02c"
_RED = "#d62728"
_ORANGE = "#ff7f0e"


# ---------------------------------------------------------------------------
# Section 2 — Temporal Patterns
# ---------------------------------------------------------------------------

def annual_volume_chart(df: pd.DataFrame, year_range: tuple[int, int]) -> go.Figure:
    """Plotly bar: total annual bike count filtered by year_range."""
    filtered = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
    annual = (
        filtered.groupby("year")["bike_count"]
        .sum()
        .reset_index()
        .rename(columns={"bike_count": "total_count"})
    )
    fig = px.bar(
        annual,
        x="year",
        y="total_count",
        template=_DARK,
        color_discrete_sequence=[_BLUE],
        title="Annual Cycling Volume",
        labels={"year": "Year", "total_count": "Total Bike Counts"},
    )
    fig.update_layout(
        xaxis=dict(tickmode="linear", dtick=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
    )
    fig.update_traces(marker_line_width=0)
    return fig


def hourly_profile_chart(
    df: pd.DataFrame,
    month_num: int,
    show_both: bool,
) -> go.Figure:
    """
    Plotly line: average hourly bike count for a given month.
    If show_both=True, overlays weekday and weekend profiles.
    Otherwise shows the combined average.
    """
    subset = df[df["month"] == month_num]
    fig = go.Figure()

    if show_both:
        for label, flag in [("Weekday", 0), ("Weekend", 1)]:
            profile = (
                subset[subset["is_weekend"] == flag]
                .groupby("hour")["bike_count"]
                .mean()
                .reset_index()
            )
            fig.add_trace(
                go.Scatter(
                    x=profile["hour"],
                    y=profile["bike_count"],
                    mode="lines+markers",
                    name=label,
                    line=dict(width=2),
                )
            )
    else:
        profile = subset.groupby("hour")["bike_count"].mean().reset_index()
        fig.add_trace(
            go.Scatter(
                x=profile["hour"],
                y=profile["bike_count"],
                mode="lines+markers",
                name="All days",
                line=dict(color=_BLUE, width=2),
                marker=dict(size=6),
            )
        )

    fig.update_layout(
        template=_DARK,
        title="Average Hourly Demand Profile",
        xaxis_title="Hour of Day",
        yaxis_title="Avg Bikes / Hour",
        xaxis=dict(tickmode="linear", dtick=2),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=340,
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Section 3 — Weather Impact
# ---------------------------------------------------------------------------

def weather_temp_bands_chart(df: pd.DataFrame) -> go.Figure:
    """Plotly bar: avg bike_count by temperature band (Cold / Mild / Warm)."""
    tmp = df.copy()
    tmp["temp_band"] = pd.cut(
        tmp["temperature"],
        bins=[-np.inf, 8, 20, np.inf],
        labels=["Cold (<8\u00b0C)", "Mild (8\u201320\u00b0C)", "Warm (>20\u00b0C)"],
    )
    band_avg = (
        tmp.groupby("temp_band", observed=True)["bike_count"]
        .mean()
        .reset_index()
        .rename(columns={"bike_count": "avg_count"})
    )
    fig = px.bar(
        band_avg,
        x="temp_band",
        y="avg_count",
        template=_DARK,
        color="temp_band",
        color_discrete_sequence=["#636EFA", _ORANGE, _RED],
        title="Average Demand by Temperature Band",
        labels={"temp_band": "Temperature Band", "avg_count": "Avg Bikes / Hour"},
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
    )
    return fig


def weather_rain_chart(df: pd.DataFrame) -> go.Figure:
    """Plotly bar: avg bike_count for dry vs rainy hours."""
    tmp = df.copy()
    tmp["rain_label"] = tmp["precipitation"].apply(
        lambda x: "Rainy (>0.1 mm)" if (pd.notna(x) and x > 0.1) else "Dry"
    )
    rain_avg = (
        tmp.groupby("rain_label")["bike_count"]
        .mean()
        .reset_index()
        .rename(columns={"bike_count": "avg_count"})
    )
    fig = px.bar(
        rain_avg,
        x="rain_label",
        y="avg_count",
        template=_DARK,
        color="rain_label",
        color_discrete_map={"Dry": _BLUE, "Rainy (>0.1 mm)": "#636EFA"},
        title="Average Demand: Dry vs Rainy",
        labels={"rain_label": "Precipitation", "avg_count": "Avg Bikes / Hour"},
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
    )
    return fig


def correlation_heatmap_chart(df: pd.DataFrame) -> go.Figure:
    """Plotly heatmap: correlation matrix of weather variables + bike_count."""
    cols = [c for c in
            ["temperature", "precipitation", "wind_speed", "humidity", "bike_count"]
            if c in df.columns]
    corr = df[cols].corr().round(3)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        template=_DARK,
        title="Weather \u00d7 Demand Correlation Matrix",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
        coloraxis_colorbar=dict(title="r"),
    )
    return fig


# ---------------------------------------------------------------------------
# Section 4 — Model Performance
# ---------------------------------------------------------------------------

def feature_importance_chart(model, feature_names: list[str]) -> go.Figure:
    """Horizontal bar: XGBoost feature importances sorted descending."""
    try:
        importances = model.feature_importances_
    except AttributeError:
        # fallback for pipeline-wrapped models
        importances = np.ones(len(feature_names))

    df_fi = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=True)
    )

    # Friendly labels
    label_map = {
        "hour_sin": "Hour (sin)",
        "hour_cos": "Hour (cos)",
        "month_sin": "Month (sin)",
        "month_cos": "Month (cos)",
        "weekday": "Day of Week",
        "is_weekend": "Is Weekend",
        "is_holiday": "Is Holiday",
        "temperature": "Temperature",
        "precipitation": "Precipitation",
        "wind_speed": "Wind Speed",
        "infrastructure_quality_score": "Infra Quality",
        "nearest_osm_distance_m": "Dist. to OSM Edge",
    }
    df_fi["label"] = df_fi["feature"].map(label_map).fillna(df_fi["feature"])

    fig = go.Figure(
        go.Bar(
            x=df_fi["importance"],
            y=df_fi["label"],
            orientation="h",
            marker_color=_BLUE,
        )
    )
    fig.update_layout(
        template=_DARK,
        title="Feature Importances (XGBoost)",
        xaxis_title="Importance Score",
        yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=360,
    )
    return fig


def predicted_vs_actual_chart(
    df_test: pd.DataFrame,
    model,
    feature_cols: list[str],
    n_sample: int = 5000,
) -> go.Figure:
    """Plotly scatter: predicted vs actual with perfect-prediction diagonal."""
    sample = df_test.sample(min(n_sample, len(df_test)), random_state=42)
    X = sample[feature_cols].values
    y_true = sample["bike_count"].values
    y_pred = model.predict(X)

    r2 = r2_score(y_true, y_pred)
    max_val = max(y_true.max(), y_pred.max()) * 1.05

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker=dict(color=_BLUE, opacity=0.4, size=4),
            name="Predictions",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="#fafafa", width=1.5, dash="dash"),
            name="Perfect prediction",
        )
    )
    fig.update_layout(
        template=_DARK,
        title=f"Predicted vs Actual Bike Count  (R\u00b2 = {r2:.4f})",
        xaxis_title="Actual Bikes / Hour",
        yaxis_title="Predicted Bikes / Hour",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=420,
        hovermode="closest",
    )
    return fig


def per_station_r2_chart(
    df_test: pd.DataFrame,
    model,
    feature_cols: list[str],
    top_n: int = 10,
    bottom_n: int = 5,
) -> go.Figure:
    """
    Horizontal bar: per-station R² on the test set.
    Shows top_n best + bottom_n worst stations, colored green/red.
    """
    X = df_test[feature_cols].values
    y_pred_all = model.predict(X)
    df_test = df_test.copy()
    df_test["_pred"] = y_pred_all

    rows = []
    for station, grp in df_test.groupby("station_id"):
        if len(grp) < 20:
            continue
        r2 = r2_score(grp["bike_count"], grp["_pred"])
        rows.append({"station_id": station, "r2": r2})

    if not rows:
        return go.Figure()

    results = pd.DataFrame(rows).sort_values("r2", ascending=False)
    top = results.head(top_n)
    bottom = results.tail(bottom_n)
    display = pd.concat([top, bottom]).drop_duplicates("station_id")
    display = display.sort_values("r2", ascending=True)
    display["color"] = display["r2"].apply(
        lambda x: _GREEN if x >= 0.5 else _RED
    )

    fig = go.Figure(
        go.Bar(
            x=display["r2"],
            y=display["station_id"].astype(str),
            orientation="h",
            marker_color=display["color"].tolist(),
        )
    )
    fig.add_vline(x=0, line_width=1, line_dash="solid", line_color="white")
    fig.update_layout(
        template=_DARK,
        title=f"Per-Station R\u00b2 — Top {top_n} Best & Bottom {bottom_n} Worst",
        xaxis_title="R\u00b2",
        yaxis_title="Station ID",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=max(300, len(display) * 24),
        xaxis=dict(range=[-0.1, 1.0]),
    )
    return fig


# ---------------------------------------------------------------------------
# Section 5 — Live Prediction
# ---------------------------------------------------------------------------

def gauge_chart(value: float, max_val: float = 500) -> go.Figure:
    """Plotly indicator gauge showing predicted bikes/hour."""
    # Colour thresholds
    steps = [
        {"range": [0, max_val * 0.25], "color": "#2a2a3e"},
        {"range": [max_val * 0.25, max_val * 0.6], "color": "#1a3a5c"},
        {"range": [max_val * 0.6, max_val], "color": "#0a5276"},
    ]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": " bikes/hr", "font": {"size": 26}},
            gauge={
                "axis": {"range": [0, max_val], "tickcolor": "#fafafa"},
                "bar": {"color": _BLUE, "thickness": 0.25},
                "bgcolor": "#262730",
                "borderwidth": 1,
                "bordercolor": "#555",
                "steps": steps,
                "threshold": {
                    "line": {"color": _RED, "width": 3},
                    "thickness": 0.75,
                    "value": value,
                },
            },
            title={"text": "Predicted Demand", "font": {"size": 16}},
        )
    )
    fig.update_layout(
        template=_DARK,
        paper_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
