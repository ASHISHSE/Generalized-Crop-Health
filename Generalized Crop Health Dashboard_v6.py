import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, date, timedelta
import requests
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Config ---
st.set_page_config(
    page_title="👨‍🌾 Generalized Crop & Weather Dashboard",
    page_icon="👨‍🌾",
    layout="wide"
)

# --- Styling ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #f0f2f6;
            color: #262730;
        }
        .main-header {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            margin-top: 40px;
            margin-bottom: 25px;
            width: 100%;
        }
        .logo-icon {
            width: 120px; height: 120px;
            object-fit: cover; border-radius: 50%;
            box-shadow: 0 0 18px rgba(116,198,157,0.45);
            border: 2px solid rgba(116,198,157,0.3);
            transition: transform 0.4s ease, box-shadow 0.4s ease;
        }
        .logo-icon:hover { transform: scale(1.08); box-shadow: 0 0 25px rgba(116,198,157,0.6); }
        .main-title {
            font-size: clamp(1.8rem, 3vw, 2.8rem);
            font-weight: 700; color: #2d6a4f;
            letter-spacing: 0.6px;
            text-shadow: 0 0 10px rgba(116,198,157,0.25);
            margin-top: 15px;
        }
        .subtitle {
            font-size: clamp(1rem, 1.5vw, 1.2rem);
            color: #52b788; font-weight: 500;
            margin-top: 8px; letter-spacing: 0.4px;
        }
        div.stButton > button:first-child {
            background: linear-gradient(90deg, #2d6a4f, #52b788);
            color: white; border-radius: 8px; font-weight: 600;
            padding: 0.6rem 1.4rem; border: none;
            box-shadow: 0 0 10px rgba(45,106,79,0.3);
            transition: all 0.25s ease;
        }
        div.stButton > button:hover {
            background: linear-gradient(90deg, #52b788, #2d6a4f);
            transform: scale(1.02); box-shadow: 0 0 15px rgba(82,183,136,0.4);
        }
        @media (max-width: 768px) {
            .main-header { margin-top: 25px; margin-bottom: 20px; }
            .logo-icon { width: 90px; height: 90px; }
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
    <div class="main-header">
        <img src="https://raw.githubusercontent.com/ASHISHSE/App_test/main/icon.png" class="logo-icon" alt="Farmer Icon">
        <div class="main-title">Generalized Crop & Weather Dashboard</div>
        <div class="subtitle">Empowering Farmers with Data-Driven Insights</div>
    </div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR — Weather file upload
# ─────────────────────────────────────────────
st.sidebar.info("🌤️ Weather Data Options")
st.sidebar.info("📊 Please upload weather data file for analysis")
uploaded_weather_file = st.sidebar.file_uploader(
    "Upload Weather Data (.xlsx)",
    type=["xlsx"],
    help="File should contain sheets: 'Weather_data_23' and 'Weather_data_24'"
)

# ─────────────────────────────────────────────
# SAMPLE DATA FALLBACK
# ─────────────────────────────────────────────
def create_sample_weather_data():
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
    districts = ["Ahilyanagar", "Pune", "Nagpur"]
    talukas   = ["Akole", "Ahilyanagar", "Bhingar"]
    circles   = ["Akole", "Nalegaon", "Bhingar"]
    rows = []
    for i, dv in enumerate(dates):
        rows.append({
            "District": districts[i % 3], "Taluka": talukas[i % 3],
            "Circle": circles[i % 3],
            "Date(DD-MM-YYYY)": dv.strftime("%d-%m-%Y"),
            "Rainfall": np.random.uniform(0, 50),
            "Tmax": np.random.uniform(25, 40), "Tmin": np.random.uniform(15, 25),
            "max_Rh": np.random.uniform(60, 95), "min_Rh": np.random.uniform(30, 70),
            "Date_dt": dv
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data(_uploaded_file=None):
    ndvi_url    = "https://github.com/ASHISHSE/Generalized-Crop-Health/raw/main/1Maharashtra_NDVI_NDWI_old_circle_2023_2024_upload.xlsx"
    mai_url     = "https://github.com/ASHISHSE/Generalized-Crop-Health/raw/main/1Circlewise_Data_MAI_2023_24_upload.xlsx"
    weather_url = "https://github.com/ASHISHSE/Generalized-Crop-Health/raw/main/Final_test_weather_data_2023_24_upload.xlsx"
    try:
        ndvi_ndwi_df = pd.read_excel(BytesIO(requests.get(ndvi_url, timeout=60).content))
        mai_df       = pd.read_excel(BytesIO(requests.get(mai_url,  timeout=60).content))

        if _uploaded_file is not None:
            try:
                w23 = pd.read_excel(_uploaded_file, sheet_name="Weather_data_23")
                w24 = pd.read_excel(_uploaded_file, sheet_name="Weather_data_24")
                weather_df = pd.concat([w23, w24], ignore_index=True)
            except Exception as e:
                st.sidebar.error(f"Error reading weather file: {e}")
                weather_df = create_sample_weather_data()
        else:
            try:
                wb = BytesIO(requests.get(weather_url, timeout=60).content)
                w23 = pd.read_excel(wb, sheet_name="Weather_data_23")
                wb  = BytesIO(requests.get(weather_url, timeout=60).content)
                w24 = pd.read_excel(wb, sheet_name="Weather_data_24")
                weather_df = pd.concat([w23, w24], ignore_index=True)
            except:
                weather_df = create_sample_weather_data()

        # Parse dates
        ndvi_ndwi_df["Date_dt"] = pd.to_datetime(ndvi_ndwi_df["Date(DD-MM-YYYY)"], format="%d-%m-%Y", errors="coerce")
        ndvi_ndwi_df = ndvi_ndwi_df.dropna(subset=["Date_dt"]).copy()

        mai_df["Year"]    = pd.to_numeric(mai_df["Year"],    errors="coerce")
        mai_df["MAI (%)"] = pd.to_numeric(mai_df["MAI (%)"], errors="coerce")

        weather_df["Date_dt"] = pd.to_datetime(weather_df["Date(DD-MM-YYYY)"], format="%d-%m-%Y", errors="coerce")
        weather_df = weather_df.dropna(subset=["Date_dt"]).copy()
        for col in ["Rainfall", "Tmax", "Tmin", "max_Rh", "min_Rh"]:
            if col in weather_df.columns:
                weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")

        districts = sorted(weather_df["District"].dropna().unique().tolist())
        talukas   = sorted(weather_df["Taluka"].dropna().unique().tolist())
        circles   = sorted(weather_df["Circle"].dropna().unique().tolist())
        return weather_df, ndvi_ndwi_df, mai_df, districts, talukas, circles

    except Exception as e:
        st.error(f"Error loading data: {e}")
        sample = create_sample_weather_data()
        return sample, pd.DataFrame(), pd.DataFrame(), [], [], []

weather_df, ndvi_ndwi_df, mai_df, districts, talukas, circles = load_data(uploaded_weather_file)

if uploaded_weather_file is not None:
    st.sidebar.success("✅ Weather data loaded successfully!")
else:
    st.sidebar.warning("⚠️ Please upload weather data file for analysis")

# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────
def get_fortnight(date_obj):
    return f"1FN {date_obj.strftime('%b')}" if date_obj.day <= 15 else f"2FN {date_obj.strftime('%b')}"

def calculate_fortnightly_metrics(data_df, current_year, last_year, metric_col, agg_func="sum"):
    metrics = {}
    for year in [current_year, last_year]:
        yd = data_df[data_df["Date_dt"].dt.year == year].copy()
        yd["Fortnight"] = yd["Date_dt"].apply(get_fortnight)
        if agg_func == "sum":
            fm = yd.groupby("Fortnight")[metric_col].sum()
        elif agg_func == "mean":
            nz = yd[yd[metric_col] != 0]
            fm = nz.groupby("Fortnight")[metric_col].mean() if not nz.empty else pd.Series(dtype=float)
        elif agg_func == "count":
            fm = (yd[metric_col] > 0).groupby(yd["Fortnight"]).sum()
        metrics[year] = fm.round(2)
    return metrics

def calculate_monthly_metrics(data_df, current_year, last_year, metric_col, agg_func="sum"):
    metrics = {}
    for year in [current_year, last_year]:
        yd = data_df[data_df["Date_dt"].dt.year == year].copy()
        yd["Month"] = yd["Date_dt"].dt.strftime("%B")
        if agg_func == "sum":
            mm = yd.groupby("Month")[metric_col].sum()
        elif agg_func == "mean":
            nz = yd[yd[metric_col] != 0]
            mm = nz.groupby("Month")[metric_col].mean() if not nz.empty else pd.Series(dtype=float)
        elif agg_func == "count":
            mm = (yd[metric_col] > 0).groupby(yd["Month"]).sum()
        metrics[year] = mm.round(2)
    return metrics

# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
def create_fortnightly_comparison_chart(current_year_data, last_year_data, title, yaxis_title):
    all_periods = sorted(set(current_year_data.index) | set(last_year_data.index))
    curr_vals = [current_year_data.get(p, 0) for p in all_periods]
    last_vals = [last_year_data.get(p, 0)    for p in all_periods]
    fig = go.Figure()
    fig.add_trace(go.Scatter(name="2023", x=all_periods, y=last_vals, mode="lines+markers",
                             line=dict(color="#1f77b4", width=3, dash="dash"), marker=dict(size=8, symbol="square")))
    fig.add_trace(go.Scatter(name="2024", x=all_periods, y=curr_vals, mode="lines+markers",
                             line=dict(color="#3498db", width=3), marker=dict(size=8, symbol="circle")))
    fig.update_layout(title=dict(text=title, x=0.5, xanchor="center"),
                      xaxis_title="Fortnight", yaxis_title=yaxis_title,
                      template="plotly_white", height=400, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def create_monthly_clustered_chart(current_year_data, last_year_data, title, yaxis_title):
    month_order = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    all_months = sorted(set(current_year_data.index) | set(last_year_data.index),
                        key=lambda x: month_order.index(x) if x in month_order else 99)
    curr_vals = [current_year_data.get(m, 0) for m in all_months]
    last_vals = [last_year_data.get(m, 0)    for m in all_months]
    deviations, deviation_labels = [], []
    for c, l in zip(curr_vals, last_vals):
        if l != 0:
            d = ((c - l) / l) * 100
            deviations.append(round(d, 2)); deviation_labels.append(f"{d:+.1f}%")
        else:
            deviations.append(0); deviation_labels.append("N/A")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(name="2023", x=all_months, y=last_vals, marker_color="#1f77b4",
                         text=[f"{x:.2f}" for x in last_vals], textposition="auto",
                         textfont=dict(weight="bold")), secondary_y=False)
    fig.add_trace(go.Bar(name="2024", x=all_months, y=curr_vals, marker_color="#3498db",
                         text=[f"{x:.2f}" for x in curr_vals], textposition="auto",
                         textfont=dict(weight="bold")), secondary_y=False)
    fig.add_trace(go.Scatter(name="Deviation (%)", x=all_months, y=deviations,
                             mode="lines+markers+text", line=dict(color="#ff6b6b", width=3),
                             marker=dict(size=8, symbol="diamond"),
                             text=deviation_labels, textposition="top center",
                             textfont=dict(color="#d63031", size=10, weight="bold")), secondary_y=True)
    fig.update_layout(title=dict(text=title, x=0.5, xanchor="center"),
                      xaxis_title="Month", yaxis_title=yaxis_title,
                      barmode="group", template="plotly_white", height=450,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="Deviation (%)", secondary_y=True)
    return fig

def create_fortnightly_deviation_chart(current_year_data, last_year_data, title, yaxis_title):
    all_periods = sorted(set(current_year_data.index) | set(last_year_data.index))
    deviations, deviation_labels = [], []
    for p in all_periods:
        c = current_year_data.get(p, 0); l = last_year_data.get(p, 0)
        if l != 0:
            d = ((c - l) / l) * 100
            deviations.append(round(d, 2)); deviation_labels.append(f"{d:+.1f}%")
        else:
            deviations.append(0); deviation_labels.append("N/A")
    colors = ["#2ecc71" if x >= 0 else "#e74c3c" for x in deviations]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=all_periods, y=deviations, marker_color=colors,
                         text=deviation_labels, textposition="auto",
                         textfont=dict(weight="bold"), name="Deviation"))
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    fig.update_layout(title=dict(text=title, x=0.5, xanchor="center"),
                      xaxis_title="Fortnight", yaxis_title=yaxis_title,
                      template="plotly_white", height=400, showlegend=False)
    return fig

# ─────────────────────────────────────────────
# NDVI / NDWI PARALLEL CHART HELPERS
# ─────────────────────────────────────────────
def _prepare_parallel(df_year: pd.DataFrame, value_col: str, smooth_window: int = 3) -> pd.DataFrame:
    tmp = df_year.copy()
    tmp["mm_dd"] = tmp["Date_dt"].dt.strftime("%m-%d")
    avg = tmp.groupby("mm_dd")[value_col].mean().reset_index().sort_values("mm_dd")
    avg[value_col] = avg[value_col].rolling(window=smooth_window, center=True, min_periods=1).mean().round(4)
    return avg

def _filter_location(df, district, taluka, circle, start_date, end_date):
    out = df.copy()
    if district: out = out[out["District"] == district]
    if taluka:   out = out[out["Taluka"]   == taluka]
    if circle:   out = out[out["Circle"]   == circle]
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

def create_ndvi_comparison_chart(ndvi_ndwi_df, district, taluka, circle, start_date, end_date):
    filtered = _filter_location(ndvi_ndwi_df, district, taluka, circle, start_date, end_date)
    if filtered.empty: return None
    d23 = _prepare_parallel(filtered[filtered["Date_dt"].dt.year == 2023], "NDVI")
    d24 = _prepare_parallel(filtered[filtered["Date_dt"].dt.year == 2024], "NDVI")
    if d23.empty and d24.empty: return None
    level, level_name = _level_title(district, taluka, circle)
    fig = go.Figure()
    if not d23.empty:
        fig.add_trace(go.Scatter(x=d23["mm_dd"], y=d23["NDVI"], mode="lines+markers", name="2023",
                                 line=dict(color="#a8c8e8", width=2.5), marker=dict(size=5, color="#a8c8e8"),
                                 hovertemplate="<b>2023</b><br>Date: %{x}<br>NDVI: %{y:.4f}<extra></extra>"))
    if not d24.empty:
        fig.add_trace(go.Scatter(x=d24["mm_dd"], y=d24["NDVI"], mode="lines+markers", name="2024",
                                 line=dict(color="#1a4a7a", width=2.5), marker=dict(size=5, color="#1a4a7a"),
                                 hovertemplate="<b>2024</b><br>Date: %{x}<br>NDVI: %{y:.4f}<extra></extra>"))
    fig.update_layout(title=dict(text=f"NDVI Comparison: 2023 vs 2024 — {level}: {level_name}", x=0.5, xanchor="center"),
                      xaxis=dict(title="Month-Day (MM-DD)", tickangle=45, tickfont=dict(size=11)),
                      yaxis_title="NDVI", template="plotly_white", height=420, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def create_ndwi_comparison_chart(ndvi_ndwi_df, district, taluka, circle, start_date, end_date):
    filtered = _filter_location(ndvi_ndwi_df, district, taluka, circle, start_date, end_date)
    if filtered.empty: return None
    d23 = _prepare_parallel(filtered[filtered["Date_dt"].dt.year == 2023], "NDWI")
    d24 = _prepare_parallel(filtered[filtered["Date_dt"].dt.year == 2024], "NDWI")
    if d23.empty and d24.empty: return None
    level, level_name = _level_title(district, taluka, circle)
    fig = go.Figure()
    if not d23.empty:
        fig.add_trace(go.Scatter(x=d23["mm_dd"], y=d23["NDWI"], mode="lines+markers", name="2023",
                                 line=dict(color="#a8c8e8", width=2.5), marker=dict(size=5, color="#a8c8e8"),
                                 hovertemplate="<b>2023</b><br>Date: %{x}<br>NDWI: %{y:.4f}<extra></extra>"))
    if not d24.empty:
        fig.add_trace(go.Scatter(x=d24["mm_dd"], y=d24["NDWI"], mode="lines+markers", name="2024",
                                 line=dict(color="#1a4a7a", width=2.5), marker=dict(size=5, color="#1a4a7a"),
                                 hovertemplate="<b>2024</b><br>Date: %{x}<br>NDWI: %{y:.4f}<extra></extra>"))
    fig.update_layout(title=dict(text=f"NDWI Comparison: 2023 vs 2024 — {level}: {level_name}", x=0.5, xanchor="center"),
                      xaxis=dict(title="Month-Day (MM-DD)", tickangle=45, tickfont=dict(size=11)),
                      yaxis_title="NDWI", template="plotly_white", height=420, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def create_ndvi_ndwi_deviation_chart(ndvi_ndwi_df, district, taluka, circle, start_date, end_date):
    filtered = _filter_location(ndvi_ndwi_df, district, taluka, circle, start_date, end_date)
    if filtered.empty: return None
    ndvi_23 = _prepare_parallel(filtered[filtered["Date_dt"].dt.year == 2023], "NDVI")
    ndvi_24 = _prepare_parallel(filtered[filtered["Date_dt"].dt.year == 2024], "NDVI")
    ndwi_23 = _prepare_parallel(filtered[filtered["Date_dt"].dt.year == 2023], "NDWI")
    ndwi_24 = _prepare_parallel(filtered[filtered["Date_dt"].dt.year == 2024], "NDWI")

    def _dev_series(df23, df24, col):
        merged = pd.merge(df23, df24, on="mm_dd", suffixes=("_23","_24")).sort_values("mm_dd")
        merged["dev"] = np.where(merged[f"{col}_23"] != 0,
                                 (merged[f"{col}_24"] - merged[f"{col}_23"]) / merged[f"{col}_23"].abs() * 100,
                                 np.nan)
        return merged["mm_dd"].tolist(), merged["dev"].round(2).tolist()

    ndvi_x, ndvi_dev = _dev_series(ndvi_23, ndvi_24, "NDVI")
    ndwi_x, ndwi_dev = _dev_series(ndwi_23, ndwi_24, "NDWI")
    if not ndvi_x and not ndwi_x: return None
    level, level_name = _level_title(district, taluka, circle)
    fig = go.Figure()
    if ndvi_x:
        fig.add_trace(go.Scatter(x=ndvi_x, y=ndvi_dev, mode="lines+markers", name="NDVI Deviation (%)",
                                 line=dict(color="#27ae60", width=2.5), marker=dict(size=5),
                                 hovertemplate="<b>NDVI</b><br>Date: %{x}<br>Dev: %{y:.2f}%<extra></extra>"))
    if ndwi_x:
        fig.add_trace(go.Scatter(x=ndwi_x, y=ndwi_dev, mode="lines+markers", name="NDWI Deviation (%)",
                                 line=dict(color="#0e7490", width=2.5), marker=dict(size=5),
                                 hovertemplate="<b>NDWI</b><br>Date: %{x}<br>Dev: %{y:.2f}%<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4,
                  annotation_text="No change", annotation_position="bottom right")
    fig.update_layout(title=dict(text=f"NDVI & NDWI Deviation (2024 vs 2023) — {level}: {level_name}", x=0.5, xanchor="center"),
                      xaxis=dict(title="Month-Day (MM-DD)", tickangle=45, tickfont=dict(size=11)),
                      yaxis_title="Deviation (%)", template="plotly_white", height=420, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def create_mai_monthly_comparison_chart(mai_df, district, taluka, circle):
    filtered = mai_df.copy()
    if district: filtered = filtered[filtered["District"] == district]
    if taluka:   filtered = filtered[filtered["Taluka"]   == taluka]
    if circle:   filtered = filtered[filtered["Circle"]   == circle]
    filtered = filtered[filtered["Year"].isin([2023, 2024])]
    if filtered.empty: return None
    monthly = filtered.groupby(["Year","Month"])["MAI (%)"].mean().reset_index()
    monthly["MAI (%)"] = monthly["MAI (%)"].round(2)
    pivot = monthly.pivot(index="Month", columns="Year", values="MAI (%)").reset_index()
    month_order = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    pivot["Month"] = pd.Categorical(pivot["Month"], categories=month_order, ordered=True)
    pivot = pivot.sort_values("Month")
    months = pivot["Month"].tolist()
    v23 = [pivot[2023].iloc[i] if 2023 in pivot.columns else 0 for i in range(len(months))]
    v24 = [pivot[2024].iloc[i] if 2024 in pivot.columns else 0 for i in range(len(months))]
    devs, dev_labels = [], []
    for c, l in zip(v24, v23):
        if l != 0:
            d = ((c - l) / l) * 100; devs.append(round(d,2)); dev_labels.append(f"{d:+.1f}%")
        else:
            devs.append(0); dev_labels.append("N/A")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(name="2023", x=months, y=v23, marker_color="#1f77b4",
                         text=[f"{x:.2f}%" for x in v23], textposition="auto",
                         textfont=dict(weight="bold")), secondary_y=False)
    fig.add_trace(go.Bar(name="2024", x=months, y=v24, marker_color="#3498db",
                         text=[f"{x:.2f}%" for x in v24], textposition="auto",
                         textfont=dict(weight="bold")), secondary_y=False)
    fig.add_trace(go.Scatter(name="Deviation (%)", x=months, y=devs, mode="lines+markers+text",
                             line=dict(color="#ff6b6b", width=3), marker=dict(size=8, symbol="diamond"),
                             text=dev_labels, textposition="top center",
                             textfont=dict(color="#d63031", size=10, weight="bold")), secondary_y=True)
    level_name = circle if circle else (taluka if taluka else district)
    level      = "Circle" if circle else ("Taluka" if taluka else "District")
    fig.update_layout(title=dict(text=f"MAI Monthly Comparison: 2023 vs 2024 — {level}: {level_name}", x=0.5, xanchor="center"),
                      xaxis_title="Month", yaxis_title="MAI (%)", barmode="group",
                      template="plotly_white", height=450,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="Deviation (%)", secondary_y=True)
    return fig

def download_data_as_csv(data_df, filename):
    csv = data_df.to_csv(index=False)
    st.download_button(label=f"📥 Download {filename} as CSV", data=csv,
                       file_name=f"{filename}.csv", mime="text/csv",
                       key=f"dl_{filename}_{np.random.randint(10000)}")

# ─────────────────────────────────────────────
# MAIN UI — Location + Date selectors
# ─────────────────────────────────────────────
st.markdown("### 📅 Select Location & Date Range")

col1, col2 = st.columns(2)
with col1:
    district = st.selectbox("District *", [""] + districts)
    taluka_options = ([""] + sorted(weather_df[weather_df["District"] == district]["Taluka"].dropna().unique().tolist())
                      if district else [""] + talukas)
    taluka = st.selectbox("Taluka", taluka_options)

with col2:
    circle_options = ([""] + sorted(weather_df[weather_df["Taluka"] == taluka]["Circle"].dropna().unique().tolist())
                      if taluka and taluka != "" else [""] + circles)
    circle = st.selectbox("Circle", circle_options)

col3, col4 = st.columns(2)
with col3:
    sowing_date  = st.date_input("Start Date (Sowing Date) *",  value=date.today() - timedelta(days=30), format="DD/MM/YYYY")
with col4:
    current_date = st.date_input("End Date (Current Date) *",   value=date.today(), format="DD/MM/YYYY")

generate = st.button("🌱 Generate Analysis", use_container_width=True)

# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────
if generate:
    if not district or not sowing_date or not current_date:
        st.error("Please select all required fields (District, Start Date, End Date).")
    else:
        # Determine level
        if circle and circle != "":
            level, level_name = "Circle", circle
        elif taluka and taluka != "":
            level, level_name = "Taluka", taluka
        else:
            level, level_name = "District", district

        st.info(f"📊 Calculating metrics for **{level}**: {level_name}")

        filtered_weather = weather_df.copy()
        if district: filtered_weather = filtered_weather[filtered_weather["District"] == district]
        if taluka:   filtered_weather = filtered_weather[filtered_weather["Taluka"]   == taluka]
        if circle:   filtered_weather = filtered_weather[filtered_weather["Circle"]   == circle]

        current_year, last_year = 2024, 2023

        # ── TABS ──────────────────────────────────────────────────────
        tab1, tab2, tab3 = st.tabs(["🌤️ Weather Metrics", "📡 Remote Sensing Indices", "💾 Download Data"])

        # ══════════════════════════════════════════════════════════════
        # TAB 1 — WEATHER METRICS
        # ══════════════════════════════════════════════════════════════
        with tab1:
            st.header(f"🌤️ Weather Metrics — {level}: {level_name}")

            if not filtered_weather.empty:

                # I. Rainfall
                st.subheader("I. Rainfall Analysis")
                st.markdown("##### Fortnightly Analysis")
                c1, c2 = st.columns(2)
                fn_rain = calculate_fortnightly_metrics(filtered_weather, current_year, last_year, "Rainfall", "sum")
                with c1:
                    st.plotly_chart(create_fortnightly_comparison_chart(
                        fn_rain[current_year], fn_rain[last_year],
                        "Rainfall — Fortnightly Comparison (2023 vs 2024)", "Rainfall (mm)"),
                        use_container_width=True)
                with c2:
                    st.plotly_chart(create_fortnightly_deviation_chart(
                        fn_rain[current_year], fn_rain[last_year],
                        "Rainfall Deviation — Fortnightly (%)", "Deviation (%)"),
                        use_container_width=True)
                st.markdown("##### Monthly Analysis")
                mo_rain = calculate_monthly_metrics(filtered_weather, current_year, last_year, "Rainfall", "sum")
                st.plotly_chart(create_monthly_clustered_chart(
                    mo_rain[current_year], mo_rain[last_year],
                    "Rainfall — Monthly Comparison with Deviation (2023 vs 2024)", "Rainfall (mm)"),
                    use_container_width=True)

                # II. Rainy Days
                st.subheader("II. Rainy Days Analysis")
                st.markdown("##### Fortnightly Analysis")
                c1, c2 = st.columns(2)
                fn_rd = calculate_fortnightly_metrics(filtered_weather, current_year, last_year, "Rainfall", "count")
                with c1:
                    st.plotly_chart(create_fortnightly_comparison_chart(
                        fn_rd[current_year], fn_rd[last_year],
                        "Rainy Days — Fortnightly Comparison (2023 vs 2024)", "Number of Rainy Days"),
                        use_container_width=True)
                with c2:
                    st.plotly_chart(create_fortnightly_deviation_chart(
                        fn_rd[current_year], fn_rd[last_year],
                        "Rainy Days Deviation — Fortnightly (%)", "Deviation (%)"),
                        use_container_width=True)
                st.markdown("##### Monthly Analysis")
                mo_rd = calculate_monthly_metrics(filtered_weather, current_year, last_year, "Rainfall", "count")
                st.plotly_chart(create_monthly_clustered_chart(
                    mo_rd[current_year], mo_rd[last_year],
                    "Rainy Days — Monthly Comparison with Deviation (2023 vs 2024)", "Number of Rainy Days"),
                    use_container_width=True)

                # III. Temperature
                st.subheader("III. Temperature Analysis")
                st.markdown("##### Maximum Temperature")
                c1, c2 = st.columns(2)
                fn_tmax = calculate_fortnightly_metrics(filtered_weather, current_year, last_year, "Tmax", "mean")
                with c1:
                    st.plotly_chart(create_fortnightly_comparison_chart(
                        fn_tmax[current_year], fn_tmax[last_year],
                        "Max Temperature — Fortnightly Average (2023 vs 2024)", "Temperature (°C)"),
                        use_container_width=True)
                with c2:
                    st.plotly_chart(create_fortnightly_deviation_chart(
                        fn_tmax[current_year], fn_tmax[last_year],
                        "Max Temperature Deviation — Fortnightly", "Deviation (°C)"),
                        use_container_width=True)
                mo_tmax = calculate_monthly_metrics(filtered_weather, current_year, last_year, "Tmax", "mean")
                st.plotly_chart(create_monthly_clustered_chart(
                    mo_tmax[current_year], mo_tmax[last_year],
                    "Max Temperature — Monthly Average with Deviation (2023 vs 2024)", "Temperature (°C)"),
                    use_container_width=True)

                st.markdown("##### Minimum Temperature")
                c1, c2 = st.columns(2)
                fn_tmin = calculate_fortnightly_metrics(filtered_weather, current_year, last_year, "Tmin", "mean")
                with c1:
                    st.plotly_chart(create_fortnightly_comparison_chart(
                        fn_tmin[current_year], fn_tmin[last_year],
                        "Min Temperature — Fortnightly Average (2023 vs 2024)", "Temperature (°C)"),
                        use_container_width=True)
                with c2:
                    st.plotly_chart(create_fortnightly_deviation_chart(
                        fn_tmin[current_year], fn_tmin[last_year],
                        "Min Temperature Deviation — Fortnightly", "Deviation (°C)"),
                        use_container_width=True)
                mo_tmin = calculate_monthly_metrics(filtered_weather, current_year, last_year, "Tmin", "mean")
                st.plotly_chart(create_monthly_clustered_chart(
                    mo_tmin[current_year], mo_tmin[last_year],
                    "Min Temperature — Monthly Average with Deviation (2023 vs 2024)", "Temperature (°C)"),
                    use_container_width=True)

                # IV. Relative Humidity
                st.subheader("IV. Relative Humidity Analysis")
                st.markdown("##### Maximum Relative Humidity")
                c1, c2 = st.columns(2)
                fn_rhmax = calculate_fortnightly_metrics(filtered_weather, current_year, last_year, "max_Rh", "mean")
                with c1:
                    st.plotly_chart(create_fortnightly_comparison_chart(
                        fn_rhmax[current_year], fn_rhmax[last_year],
                        "Max RH — Fortnightly Average (2023 vs 2024)", "Relative Humidity (%)"),
                        use_container_width=True)
                with c2:
                    st.plotly_chart(create_fortnightly_deviation_chart(
                        fn_rhmax[current_year], fn_rhmax[last_year],
                        "Max RH Deviation — Fortnightly", "Deviation (%)"),
                        use_container_width=True)
                mo_rhmax = calculate_monthly_metrics(filtered_weather, current_year, last_year, "max_Rh", "mean")
                st.plotly_chart(create_monthly_clustered_chart(
                    mo_rhmax[current_year], mo_rhmax[last_year],
                    "Max RH — Monthly Average with Deviation (2023 vs 2024)", "Relative Humidity (%)"),
                    use_container_width=True)

                st.markdown("##### Minimum Relative Humidity")
                c1, c2 = st.columns(2)
                fn_rhmin = calculate_fortnightly_metrics(filtered_weather, current_year, last_year, "min_Rh", "mean")
                with c1:
                    st.plotly_chart(create_fortnightly_comparison_chart(
                        fn_rhmin[current_year], fn_rhmin[last_year],
                        "Min RH — Fortnightly Average (2023 vs 2024)", "Relative Humidity (%)"),
                        use_container_width=True)
                with c2:
                    st.plotly_chart(create_fortnightly_deviation_chart(
                        fn_rhmin[current_year], fn_rhmin[last_year],
                        "Min RH Deviation — Fortnightly", "Deviation (%)"),
                        use_container_width=True)
                mo_rhmin = calculate_monthly_metrics(filtered_weather, current_year, last_year, "min_Rh", "mean")
                st.plotly_chart(create_monthly_clustered_chart(
                    mo_rhmin[current_year], mo_rhmin[last_year],
                    "Min RH — Monthly Average with Deviation (2023 vs 2024)", "Relative Humidity (%)"),
                    use_container_width=True)
            else:
                st.info("No weather data available for the selected location.")

        # ══════════════════════════════════════════════════════════════
        # TAB 2 — REMOTE SENSING INDICES
        # ══════════════════════════════════════════════════════════════
        with tab2:
            st.header(f"📡 Remote Sensing Indices — {level}: {level_name}")

            st.subheader("I. NDVI Analysis")
            fig_ndvi = create_ndvi_comparison_chart(ndvi_ndwi_df, district, taluka, circle, sowing_date, current_date)
            st.plotly_chart(fig_ndvi, use_container_width=True) if fig_ndvi else st.info("No NDVI data available.")

            st.subheader("II. NDWI Analysis")
            fig_ndwi = create_ndwi_comparison_chart(ndvi_ndwi_df, district, taluka, circle, sowing_date, current_date)
            st.plotly_chart(fig_ndwi, use_container_width=True) if fig_ndwi else st.info("No NDWI data available.")

            st.subheader("III. NDVI & NDWI Deviation Analysis")
            fig_dev = create_ndvi_ndwi_deviation_chart(ndvi_ndwi_df, district, taluka, circle, sowing_date, current_date)
            st.plotly_chart(fig_dev, use_container_width=True) if fig_dev else st.info("No deviation data available.")

            st.subheader("IV. MAI Analysis")
            fig_mai = create_mai_monthly_comparison_chart(mai_df, district, taluka, circle)
            st.plotly_chart(fig_mai, use_container_width=True) if fig_mai else st.info("No MAI data available.")

        # ══════════════════════════════════════════════════════════════
        # TAB 3 — DOWNLOAD DATA
        # ══════════════════════════════════════════════════════════════
        with tab3:
            st.header(f"💾 Download Data — {level}: {level_name}")
            st.subheader("Download Data Tables")
            c1, c2, c3 = st.columns(3)

            with c1:
                if not filtered_weather.empty:
                    download_data_as_csv(filtered_weather, f"Weather_Data_{level}_{level_name}")
                else:
                    st.write("No weather data available")

            with c2:
                fn = ndvi_ndwi_df.copy()
                if district: fn = fn[fn["District"] == district]
                if taluka:   fn = fn[fn["Taluka"]   == taluka]
                if circle:   fn = fn[fn["Circle"]   == circle]
                if not fn.empty:
                    download_data_as_csv(fn, f"NDVI_NDWI_Data_{level}_{level_name}")
                else:
                    st.write("No NDVI/NDWI data available")

            with c3:
                fm = mai_df.copy()
                if district: fm = fm[fm["District"] == district]
                if taluka:   fm = fm[fm["Taluka"]   == taluka]
                if circle:   fm = fm[fm["Circle"]   == circle]
                if not fm.empty:
                    download_data_as_csv(fm, f"MAI_Data_{level}_{level_name}")
                else:
                    st.write("No MAI data available")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
    <div style='text-align: center; font-size: 16px; margin-top: 20px;'>
        💻 <b>Developed by:</b> Ashish Selokar <br>
        📧 For suggestions or queries, please email at:
        <a href="mailto:ashish111.selokar@gmail.com">ashish111.selokar@gmail.com</a> <br><br>
        <span style="font-size:15px; color:green;">🌾 Empowering Farmers with Data-Driven Insights 🌾</span><br>
        <span style="font-size:13px; color:gray;">Version 2.0 | Powered by Agricose | Last Updated: Oct 2024</span>
    </div>
""", unsafe_allow_html=True)
