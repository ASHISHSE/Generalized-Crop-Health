# =====================================================================
# REPLACE THESE FUNCTIONS IN YOUR MAIN CODE
# Changes:
#   - Both years plotted on same MM-DD axis (year stripped)
#   - Same MM-DD points averaged across multiple readings
#   - Lines smoothed with rolling average (window=3)
#   - 2023 → faint blue (#a8c8e8), 2024 → dark blue (#1a4a7a)
#   - Deviation chart uses the same smoothed values
# =====================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, date, timedelta
import requests
from io import BytesIO
import plotly.express as px



# ── helper ────────────────────────────────────────────────────────────────────

def _prepare_parallel(df_year: pd.DataFrame, value_col: str,
                      smooth_window: int = 3) -> pd.DataFrame:
    """
    Strip the year from Date_dt → 'MM-DD' key.
    Average multiple readings on the same MM-DD, then smooth.
    Returns DataFrame with columns: [mm_dd, value_col]
    """
    tmp = df_year.copy()
    tmp["mm_dd"] = tmp["Date_dt"].dt.strftime("%m-%d")          # strip year
    avg = (tmp.groupby("mm_dd")[value_col]
               .mean()
               .reset_index()
               .sort_values("mm_dd"))
    # smooth
    avg[value_col] = (avg[value_col]
                      .rolling(window=smooth_window, center=True, min_periods=1)
                      .mean()
                      .round(4))
    return avg


def _filter_location(df: pd.DataFrame,
                     district: str, taluka: str, circle: str,
                     start_date, end_date) -> pd.DataFrame:
    """Common location + date-range filter (keeps both 2023 & 2024)."""
    out = df.copy()
    if district:
        out = out[out["District"] == district]
    if taluka:
        out = out[out["Taluka"] == taluka]
    if circle:
        out = out[out["Circle"] == circle]
    out = out[
        (out["Date_dt"] >= pd.to_datetime(start_date)) &
        (out["Date_dt"] <= pd.to_datetime(end_date)) &
        (out["Date_dt"].dt.year.isin([2023, 2024]))
    ]
    return out


def _level_title(district, taluka, circle):
    level_name = circle if circle else (taluka if taluka else district)
    level      = "Circle" if circle else ("Taluka" if taluka else "District")
    return level, level_name


# ── NDVI comparison ──────────────────────────────────────────────────────────

def create_ndvi_comparison_chart(ndvi_ndwi_df, district, taluka, circle,
                                 start_date, end_date):
    """
    NDVI line chart — both years on the same Month-Day axis.
    2023 : faint blue  (#a8c8e8)
    2024 : dark  blue  (#1a4a7a)
    Lines are smoothed rolling averages.
    """
    filtered = _filter_location(ndvi_ndwi_df, district, taluka, circle,
                                start_date, end_date)
    if filtered.empty:
        return None

    d23 = _prepare_parallel(filtered[filtered["Date_dt"].dt.year == 2023], "NDVI")
    d24 = _prepare_parallel(filtered[filtered["Date_dt"].dt.year == 2024], "NDVI")

    if d23.empty and d24.empty:
        return None

    level, level_name = _level_title(district, taluka, circle)

    fig = go.Figure()

    if not d23.empty:
        fig.add_trace(go.Scatter(
            x=d23["mm_dd"], y=d23["NDVI"],
            mode="lines+markers",
            name="2023",
            line=dict(color="#a8c8e8", width=2.5),   # faint blue
            marker=dict(size=5, color="#a8c8e8"),
            hovertemplate="<b>2023</b><br>Date: %{x}<br>NDVI: %{y:.4f}<extra></extra>"
        ))

    if not d24.empty:
        fig.add_trace(go.Scatter(
            x=d24["mm_dd"], y=d24["NDVI"],
            mode="lines+markers",
            name="2024",
            line=dict(color="#1a4a7a", width=2.5),   # dark blue
            marker=dict(size=5, color="#1a4a7a"),
            hovertemplate="<b>2024</b><br>Date: %{x}<br>NDVI: %{y:.4f}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(
            text=f"NDVI Comparison: 2023 vs 2024 — {level}: {level_name}",
            x=0.5, xanchor="center"
        ),
        xaxis=dict(
            title="Month-Day (MM-DD)",
            tickangle=45,
            tickfont=dict(size=11)
        ),
        yaxis_title="NDVI",
        template="plotly_white",
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1)
    )
    return fig


# ── NDWI comparison ──────────────────────────────────────────────────────────

def create_ndwi_comparison_chart(ndvi_ndwi_df, district, taluka, circle,
                                 start_date, end_date):
    """
    NDWI line chart — same design as NDVI chart above.
    """
    filtered = _filter_location(ndvi_ndwi_df, district, taluka, circle,
                                start_date, end_date)
    if filtered.empty:
        return None

    d23 = _prepare_parallel(filtered[filtered["Date_dt"].dt.year == 2023], "NDWI")
    d24 = _prepare_parallel(filtered[filtered["Date_dt"].dt.year == 2024], "NDWI")

    if d23.empty and d24.empty:
        return None

    level, level_name = _level_title(district, taluka, circle)

    fig = go.Figure()

    if not d23.empty:
        fig.add_trace(go.Scatter(
            x=d23["mm_dd"], y=d23["NDWI"],
            mode="lines+markers",
            name="2023",
            line=dict(color="#a8c8e8", width=2.5),
            marker=dict(size=5, color="#a8c8e8"),
            hovertemplate="<b>2023</b><br>Date: %{x}<br>NDWI: %{y:.4f}<extra></extra>"
        ))

    if not d24.empty:
        fig.add_trace(go.Scatter(
            x=d24["mm_dd"], y=d24["NDWI"],
            mode="lines+markers",
            name="2024",
            line=dict(color="#1a4a7a", width=2.5),
            marker=dict(size=5, color="#1a4a7a"),
            hovertemplate="<b>2024</b><br>Date: %{x}<br>NDWI: %{y:.4f}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(
            text=f"NDWI Comparison: 2023 vs 2024 — {level}: {level_name}",
            x=0.5, xanchor="center"
        ),
        xaxis=dict(
            title="Month-Day (MM-DD)",
            tickangle=45,
            tickfont=dict(size=11)
        ),
        yaxis_title="NDWI",
        template="plotly_white",
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1)
    )
    return fig


# ── Deviation chart ───────────────────────────────────────────────────────────

def create_ndvi_ndwi_deviation_chart(ndvi_ndwi_df, district, taluka, circle,
                                     start_date, end_date):
    """
    Deviation chart built from the SAME smoothed parallel lines used in the
    comparison charts above.

    Deviation = (2024_value - 2023_value) / |2023_value| × 100
    Plotted on shared MM-DD x-axis.
    NDVI deviation : green  (#27ae60)
    NDWI deviation : teal   (#0e7490)
    Zero reference dashed black line.
    """
    filtered = _filter_location(ndvi_ndwi_df, district, taluka, circle,
                                start_date, end_date)
    if filtered.empty:
        return None

    # ── NDVI deviation ──────────────────────────────────────────────
    ndvi_23 = _prepare_parallel(filtered[filtered["Date_dt"].dt.year == 2023], "NDVI")
    ndvi_24 = _prepare_parallel(filtered[filtered["Date_dt"].dt.year == 2024], "NDVI")

    # ── NDWI deviation ──────────────────────────────────────────────
    ndwi_23 = _prepare_parallel(filtered[filtered["Date_dt"].dt.year == 2023], "NDWI")
    ndwi_24 = _prepare_parallel(filtered[filtered["Date_dt"].dt.year == 2024], "NDWI")

    def _deviation_series(df23, df24, col):
        """Inner-join on mm_dd, compute % deviation, return (x, y) lists."""
        merged = pd.merge(df23, df24, on="mm_dd", suffixes=("_23", "_24"))
        merged = merged.sort_values("mm_dd")
        merged["dev"] = np.where(
            merged[f"{col}_23"] != 0,
            (merged[f"{col}_24"] - merged[f"{col}_23"]) / merged[f"{col}_23"].abs() * 100,
            np.nan
        )
        merged["dev"] = merged["dev"].round(2)
        return merged["mm_dd"].tolist(), merged["dev"].tolist()

    ndvi_x, ndvi_dev = _deviation_series(ndvi_23, ndvi_24, "NDVI")
    ndwi_x, ndwi_dev = _deviation_series(ndwi_23, ndwi_24, "NDWI")

    if not ndvi_x and not ndwi_x:
        return None

    level, level_name = _level_title(district, taluka, circle)

    fig = go.Figure()

    if ndvi_x:
        fig.add_trace(go.Scatter(
            x=ndvi_x, y=ndvi_dev,
            mode="lines+markers",
            name="NDVI Deviation (%)",
            line=dict(color="#27ae60", width=2.5),
            marker=dict(size=5),
            hovertemplate="<b>NDVI</b><br>Date: %{x}<br>Dev: %{y:.2f}%<extra></extra>"
        ))

    if ndwi_x:
        fig.add_trace(go.Scatter(
            x=ndwi_x, y=ndwi_dev,
            mode="lines+markers",
            name="NDWI Deviation (%)",
            line=dict(color="#0e7490", width=2.5),
            marker=dict(size=5),
            hovertemplate="<b>NDWI</b><br>Date: %{x}<br>Dev: %{y:.2f}%<extra></extra>"
        ))

    # zero reference
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4,
                  annotation_text="No change", annotation_position="bottom right")

    fig.update_layout(
        title=dict(
            text=f"NDVI & NDWI Deviation (2024 vs 2023) — {level}: {level_name}",
            x=0.5, xanchor="center"
        ),
        xaxis=dict(
            title="Month-Day (MM-DD)",
            tickangle=45,
            tickfont=dict(size=11)
        ),
        yaxis_title="Deviation (%)",
        template="plotly_white",
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1)
    )
    return fig



# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    """
    <div style='text-align: center; font-size: 16px; margin-top: 20px;'>
        💻 <b>Developed by:</b> Ashish Selokar <br>
        📧 For suggestions or queries, please email at:
        <a href="mailto:ashish111.selokar@gmail.com">ashish111.selokar@gmail.com</a> <br><br>
        <span style="font-size:15px; color:green;">
            🌾 Empowering Farmers with Data-Driven Insights 🌾
        </span><br>
        <span style="font-size:13px; color:gray;">
            Version 2.0 | Powered by Agricose | Last Updated: Oct 2024
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
