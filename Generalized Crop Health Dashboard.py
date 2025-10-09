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
from PIL import Image


# --- Page Config --- 
st.set_page_config(
    page_title="👨‍🌾 Smart Crop Health Dashboard",
    page_icon="👨‍🌾",
    layout="wide"
)

# --- Modern Tech Dashboard Styling ---
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
            width: 120px;
            height: 120px;
            object-fit: cover;
            border-radius: 50%;
            box-shadow: 0 0 18px rgba(116,198,157,0.45);
            border: 2px solid rgba(116,198,157,0.3);
            transition: transform 0.4s ease, box-shadow 0.4s ease;
        }

        .logo-icon:hover {
            transform: scale(1.08);
            box-shadow: 0 0 25px rgba(116,198,157,0.6);
        }

        .main-title {
            font-size: clamp(1.8rem, 3vw, 2.8rem);
            font-weight: 700;
            color: #2d6a4f;
            letter-spacing: 0.6px;            
            text-shadow: 0 0 10px rgba(116,198,157,0.25);
            margin-top: 15px;
        }

        .subtitle {
            font-size: clamp(1rem, 1.5vw, 1.2rem);
            color: #52b788;
            font-weight: 500;
            margin-top: 8px;
            letter-spacing: 0.4px;
        }

        div.stButton > button:first-child {
            background: linear-gradient(90deg, #2d6a4f, #52b788);
            color: white;
            border-radius: 8px;
            font-weight: 600;
            padding: 0.6rem 1.4rem;
            border: none;
            box-shadow: 0 0 10px rgba(45,106,79,0.3);
            transition: all 0.25s ease;
        }

        div.stButton > button:hover {
            background: linear-gradient(90deg, #52b788, #2d6a4f);
            transform: scale(1.02);
            box-shadow: 0 0 15px rgba(82,183,136,0.4);
        }

        .footer {
            text-align: center;
            font-size: 0.85rem;
            color: #adb5bd;
            margin-top: 40px;
        }

        @media (max-width: 768px) {
            .main-header {
                margin-top: 25px;
                margin-bottom: 20px;
            }
            .logo-icon {
                width: 90px;
                height: 90px;
            }
        }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
    <div class="main-header">
        <img src="https://raw.githubusercontent.com/ASHISHSE/App_test/main/icon.png" class="logo-icon" alt="Farmer Icon">
        <div class="main-title">Smart Crop Health Dashboard</div>
        <div class="subtitle">Comparing 2023 & 2024 Crop Health Data</div>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA - UPDATED URLs as per requirements
# -----------------------------
@st.cache_data
def load_data():
    # Updated URLs as per requirements
    ndvi_ndwi_url = "https://github.com/ASHISHSE/Generalized-Crop-Health/raw/main/1Maharashtra_NDVI_NDWI_old_circle_2023_2024_upload.xlsx"
    weather_url = "https://docs.google.com/spreadsheets/d/1IsximMN9KrKpsREnWiNu0pbAtQ3idtjl/export?format=xlsx"
    mai_url = "https://github.com/ASHISHSE/Generalized-Crop-Health/raw/main/1Circlewise_Data_MAI_2023_24_upload.xlsx"

    try:
        # Load NDVI & NDWI data
        ndvi_ndwi_res = requests.get(ndvi_ndwi_url, timeout=30)
        ndvi_ndwi_df = pd.read_excel(BytesIO(ndvi_ndwi_res.content))
        
        # Load Weather data (both 2023 and 2024 sheets)
        weather_res = requests.get(weather_url, timeout=30)
        weather_23_df = pd.read_excel(BytesIO(weather_res.content), sheet_name='Weather_data_23')
        weather_24_df = pd.read_excel(BytesIO(weather_res.content), sheet_name='Weather_data_24')
        weather_df = pd.concat([weather_23_df, weather_24_df], ignore_index=True)
        
        # Load MAI data
        mai_res = requests.get(mai_url, timeout=30)
        mai_df = pd.read_excel(BytesIO(mai_res.content))
        
        # Process NDVI & NDWI data
        ndvi_ndwi_df["Date_dt"] = pd.to_datetime(ndvi_ndwi_df["Date(DD-MM-YYYY)"], format="%Y_%m_%d", errors="coerce")
        
        # Process Weather data
        weather_df["Date_dt"] = pd.to_datetime(weather_df["Date(DD-MM-YYYY)"], format="%d-%m-%Y", errors="coerce")
        
        # Convert numeric columns for weather data
        for col in ["Rainfall", "Tmax", "Tmin", "max_Rh", "min_Rh"]:
            if col in weather_df.columns:
                weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")
        
        # Get unique locations for filters
        districts = sorted(ndvi_ndwi_df["District"].dropna().unique().tolist())
        talukas = sorted(ndvi_ndwi_df["Taluka"].dropna().unique().tolist())
        circles = sorted(ndvi_ndwi_df["Circle"].dropna().unique().tolist())
        
        return ndvi_ndwi_df, weather_df, mai_df, districts, talukas, circles
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], [], []

# Load data
ndvi_ndwi_df, weather_df, mai_df, districts, talukas, circles = load_data()

# -----------------------------
# FORTNIGHT DEFINITION FUNCTION
# -----------------------------
def get_fortnight(date_obj):
    """Define fortnight: 1st-15th as 1st Fortnight, 16th-end as 2nd Fortnight"""
    if date_obj.day <= 15:
        return f"1FN {date_obj.strftime('%B')}"
    else:
        return f"2FN {date_obj.strftime('%B')}"

def get_fortnight_from_string(date_str, year):
    """Get fortnight from date string"""
    try:
        date_obj = datetime.strptime(date_str, "%d-%m-%Y")
        return get_fortnight(date_obj)
    except:
        try:
            date_obj = datetime.strptime(date_str, "%Y_%m_%d")
            return get_fortnight(date_obj)
        except:
            return None

# -----------------------------
# WEATHER METRICS CALCULATIONS
# -----------------------------
def calculate_weather_metrics_comparison(weather_data, district, taluka, circle, start_date, end_date):
    """Calculate weather metrics for comparison between years"""
    
    # Filter data based on selection
    filtered_data = weather_data.copy()
    if district:
        filtered_data = filtered_data[filtered_data["District"] == district]
    if taluka:
        filtered_data = filtered_data[filtered_data["Taluka"] == taluka]
    if circle:
        filtered_data = filtered_data[filtered_data["Circle"] == circle]
    
    # Extract year from date
    filtered_data["Year"] = filtered_data["Date_dt"].dt.year
    
    # Filter by date range (across years)
    date_range_data = filtered_data[
        (filtered_data["Date_dt"].dt.month >= start_date.month) & 
        (filtered_data["Date_dt"].dt.month <= end_date.month)
    ].copy()
    
    # Add fortnight column
    date_range_data["Fortnight"] = date_range_data["Date_dt"].apply(get_fortnight)
    
    results = {}
    
    # Fortnightly calculations
    fortnightly_data = date_range_data.groupby(["Year", "Fortnight"]).agg({
        "Rainfall": "sum",
        "Tmax": lambda x: x[x != 0].mean() if any(x != 0) else None,
        "Tmin": lambda x: x[x != 0].mean() if any(x != 0) else None,
        "max_Rh": lambda x: x[x != 0].mean() if any(x != 0) else None,
        "min_Rh": lambda x: x[x != 0].mean() if any(x != 0) else None
    }).reset_index()
    
    # Rainy days calculation
    rainy_days_fortnightly = date_range_data[date_range_data["Rainfall"] > 0].groupby(["Year", "Fortnight"]).size().reset_index(name="Rainy_Days")
    fortnightly_data = fortnightly_data.merge(rainy_days_fortnightly, on=["Year", "Fortnight"], how="left")
    fortnightly_data["Rainy_Days"] = fortnightly_data["Rainy_Days"].fillna(0)
    
    # Monthly calculations
    date_range_data["Month"] = date_range_data["Date_dt"].dt.strftime("%B")
    monthly_data = date_range_data.groupby(["Year", "Month"]).agg({
        "Rainfall": "sum",
        "Tmax": lambda x: x[x != 0].mean() if any(x != 0) else None,
        "Tmin": lambda x: x[x != 0].mean() if any(x != 0) else None,
        "max_Rh": lambda x: x[x != 0].mean() if any(x != 0) else None,
        "min_Rh": lambda x: x[x != 0].mean() if any(x != 0) else None
    }).reset_index()
    
    # Rainy days monthly
    rainy_days_monthly = date_range_data[date_range_data["Rainfall"] > 0].groupby(["Year", "Month"]).size().reset_index(name="Rainy_Days")
    monthly_data = monthly_data.merge(rainy_days_monthly, on=["Year", "Month"], how="left")
    monthly_data["Rainy_Days"] = monthly_data["Rainy_Days"].fillna(0)
    
    results["fortnightly"] = fortnightly_data
    results["monthly"] = monthly_data
    
    return results

# -----------------------------
# REMOTE SENSING INDICES CALCULATIONS
# -----------------------------
def calculate_remote_sensing_comparison(ndvi_ndwi_data, mai_data, district, taluka, circle, start_date, end_date):
    """Calculate remote sensing indices for comparison between years"""
    
    # Filter NDVI/NDWI data
    filtered_ndvi_ndwi = ndvi_ndwi_data.copy()
    if district:
        filtered_ndvi_ndwi = filtered_ndvi_ndwi[filtered_ndvi_ndwi["District"] == district]
    if taluka:
        filtered_ndvi_ndwi = filtered_ndvi_ndwi[filtered_ndvi_ndwi["Taluka"] == taluka]
    if circle:
        filtered_ndvi_ndwi = filtered_ndvi_ndwi[filtered_ndvi_ndwi["Circle"] == circle]
    
    # Filter by date range
    filtered_ndvi_ndwi = filtered_ndvi_ndwi[
        (filtered_ndvi_ndwi["Date_dt"] >= pd.Timestamp(start_date)) & 
        (filtered_ndvi_ndwi["Date_dt"] <= pd.Timestamp(end_date))
    ]
    
    # Add year column
    filtered_ndvi_ndwi["Year"] = filtered_ndvi_ndwi["Date_dt"].dt.year
    
    # Filter MAI data
    filtered_mai = mai_data.copy()
    if district:
        filtered_mai = filtered_mai[filtered_mai["District"] == district]
    if taluka:
        filtered_mai = filtered_mai[filtered_mai["Taluka"] == taluka]
    if circle:
        filtered_mai = filtered_mai[filtered_mai["Circle"] == circle]
    
    # Filter MAI by months in date range
    start_month = start_date.strftime("%B")
    end_month = end_date.strftime("%B")
    months_in_range = []
    current = start_date.replace(day=1)
    end = end_date.replace(day=1)
    
    while current <= end:
        months_in_range.append(current.strftime("%B"))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    months_in_range = list(dict.fromkeys(months_in_range))
    filtered_mai = filtered_mai[filtered_mai["Month"].isin(months_in_range)]
    
    # Calculate monthly averages for MAI (excluding 0 values)
    mai_monthly = filtered_mai.groupby(["Year", "Month"]).agg({
        "MAI (%)": lambda x: x[x != 0].mean() if any(x != 0) else None
    }).reset_index()
    
    results = {
        "ndvi_ndwi_timeseries": filtered_ndvi_ndwi,
        "mai_monthly": mai_monthly
    }
    
    return results

# -----------------------------
# CHART FUNCTIONS
# -----------------------------

def create_rainfall_chart(weather_results, period_type="fortnightly"):
    """Create clustered column chart for Rainfall"""
    data = weather_results[period_type]
    
    # Pivot data for chart
    pivot_data = data.pivot_table(
        index=period_type.title()[:-2],  # Remove 'ly' from fortnightly/monthly
        columns="Year", 
        values="Rainfall", 
        aggfunc="sum"
    ).reset_index()
    
    fig = go.Figure()
    
    for year in pivot_data.columns[1:]:
        fig.add_trace(go.Bar(
            name=str(year),
            x=pivot_data[period_type.title()[:-2]],
            y=pivot_data[year],
            text=pivot_data[year].round(1),
            textposition='auto',
        ))
    
    title_period = "Fortnightly" if period_type == "fortnightly" else "Monthly"
    fig.update_layout(
        title=f"{title_period} Rainfall Comparison (mm)",
        xaxis_title=title_period,
        yaxis_title="Rainfall (mm)",
        barmode='group',
        template="plotly_white"
    )
    
    return fig

def create_rainy_days_chart(weather_results, period_type="fortnightly"):
    """Create clustered column chart for Rainy Days"""
    data = weather_results[period_type]
    
    pivot_data = data.pivot_table(
        index=period_type.title()[:-2],
        columns="Year", 
        values="Rainy_Days", 
        aggfunc="sum"
    ).reset_index()
    
    fig = go.Figure()
    
    for year in pivot_data.columns[1:]:
        fig.add_trace(go.Bar(
            name=str(year),
            x=pivot_data[period_type.title()[:-2]],
            y=pivot_data[year],
            text=pivot_data[year].round(0),
            textposition='auto',
        ))
    
    title_period = "Fortnightly" if period_type == "fortnightly" else "Monthly"
    fig.update_layout(
        title=f"{title_period} Rainy Days Comparison",
        xaxis_title=title_period,
        yaxis_title="Number of Rainy Days",
        barmode='group',
        template="plotly_white"
    )
    
    return fig

def create_temperature_chart(weather_results, temp_type="Tmax", period_type="fortnightly"):
    """Create clustered column chart for Temperature"""
    data = weather_results[period_type]
    
    pivot_data = data.pivot_table(
        index=period_type.title()[:-2],
        columns="Year", 
        values=temp_type, 
        aggfunc="mean"
    ).reset_index()
    
    fig = go.Figure()
    
    for year in pivot_data.columns[1:]:
        fig.add_trace(go.Bar(
            name=str(year),
            x=pivot_data[period_type.title()[:-2]],
            y=pivot_data[year],
            text=pivot_data[year].round(1),
            textposition='auto',
        ))
    
    title_period = "Fortnightly" if period_type == "fortnightly" else "Monthly"
    temp_name = "Maximum Temperature" if temp_type == "Tmax" else "Minimum Temperature"
    fig.update_layout(
        title=f"{title_period} {temp_name} Comparison (°C)",
        xaxis_title=title_period,
        yaxis_title=f"{temp_name} (°C)",
        barmode='group',
        template="plotly_white"
    )
    
    return fig

def create_humidity_chart(weather_results, humidity_type="max_Rh", period_type="fortnightly"):
    """Create clustered column chart for Humidity"""
    data = weather_results[period_type]
    
    pivot_data = data.pivot_table(
        index=period_type.title()[:-2],
        columns="Year", 
        values=humidity_type, 
        aggfunc="mean"
    ).reset_index()
    
    fig = go.Figure()
    
    for year in pivot_data.columns[1:]:
        fig.add_trace(go.Bar(
            name=str(year),
            x=pivot_data[period_type.title()[:-2]],
            y=pivot_data[year],
            text=pivot_data[year].round(1),
            textposition='auto',
        ))
    
    title_period = "Fortnightly" if period_type == "fortnightly" else "Monthly"
    humidity_name = "Maximum Relative Humidity" if humidity_type == "max_Rh" else "Minimum Relative Humidity"
    fig.update_layout(
        title=f"{title_period} {humidity_name} Comparison (%)",
        xaxis_title=title_period,
        yaxis_title=f"{humidity_name} (%)",
        barmode='group',
        template="plotly_white"
    )
    
    return fig

def create_ndvi_ndwi_line_chart(remote_sensing_results, index_type="NDVI"):
    """Create line chart for NDVI/NDWI comparison"""
    data = remote_sensing_results["ndvi_ndwi_timeseries"]
    
    fig = go.Figure()
    
    for year in data["Year"].unique():
        year_data = data[data["Year"] == year].sort_values("Date_dt")
        fig.add_trace(go.Scatter(
            x=year_data["Date_dt"],
            y=year_data[index_type],
            mode='lines+markers',
            name=str(year),
            line=dict(width=2),
            marker=dict(size=4)
        ))
    
    index_name = "NDVI" if index_type == "NDVI" else "NDWI"
    fig.update_layout(
        title=f"{index_name} Comparison (2023 vs 2024)",
        xaxis_title="Date",
        yaxis_title=index_name,
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig

def create_mai_chart(remote_sensing_results):
    """Create clustered column chart for MAI"""
    data = remote_sensing_results["mai_monthly"]
    
    pivot_data = data.pivot_table(
        index="Month",
        columns="Year", 
        values="MAI (%)", 
        aggfunc="mean"
    ).reset_index()
    
    # Define month order for proper sorting
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    pivot_data['Month'] = pd.Categorical(pivot_data['Month'], categories=month_order, ordered=True)
    pivot_data = pivot_data.sort_values('Month')
    
    fig = go.Figure()
    
    for year in pivot_data.columns[1:]:
        fig.add_trace(go.Bar(
            name=str(year),
            x=pivot_data["Month"],
            y=pivot_data[year],
            text=pivot_data[year].round(1),
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Monthly MAI Comparison (%)",
        xaxis_title="Month",
        yaxis_title="MAI (%)",
        barmode='group',
        template="plotly_white"
    )
    
    return fig

# -----------------------------
# MAIN UI
# -----------------------------

st.markdown("""
<div style='
    background-color: rgba(255, 193, 7, 0.1);
    border-left: 4px solid #f4a261;
    padding: 10px 16px;
    border-radius: 6px;
    margin-bottom: 20px;
    font-size: 0.95rem;
'>
    <span style='color: red; font-weight: 700;'>📊 Crop Health Comparison Dashboard:</span>
    <span style='color: blue;'>
        Compare crop health parameters between <b>2023 and 2024</b>.
    </span>
    <br><br>
    <span style='color: black;'>
        🔹 <b>Level of Selection:</b> You can select data from 
        <b>Circle → Taluka → District</b> level.
    </span>
</div>
""", unsafe_allow_html=True)

# Selection UI
col1, col2, col3 = st.columns(3)

with col1:
    district = st.selectbox("District", [""] + districts)
    if district:
        taluka_options = [""] + sorted(ndvi_ndwi_df[ndvi_ndwi_df["District"] == district]["Taluka"].dropna().unique().tolist())
    else:
        taluka_options = [""] + talukas
    taluka = st.selectbox("Taluka", taluka_options)

with col2:
    if taluka and taluka != "":
        circle_options = [""] + sorted(ndvi_ndwi_df[ndvi_ndwi_df["Taluka"] == taluka]["Circle"].dropna().unique().tolist())
    else:
        circle_options = [""] + circles
    circle = st.selectbox("Circle", circle_options)

with col3:
    sowing_date = st.date_input("Start Date (Sowing Date)", value=date(2024, 6, 1), format="DD/MM/YYYY")
    current_date = st.date_input("End Date (Current Date)", value=date(2024, 10, 31), format="DD/MM/YYYY")

generate = st.button("📈 Generate Comparison Analysis")

if generate:
    if not district:
        st.error("Please select at least a District.")
    else:
        # Determine level name for display
        if circle and circle != "":
            level = "Circle"
            level_name = circle
        elif taluka and taluka != "":
            level = "Taluka"
            level_name = taluka
        else:
            level = "District"
            level_name = district

        st.info(f"📊 Generating comparison analysis for **{level}**: {level_name}")

        # Calculate metrics
        weather_results = calculate_weather_metrics_comparison(
            weather_df, district, taluka, circle, sowing_date, current_date
        )
        
        remote_sensing_results = calculate_remote_sensing_comparison(
            ndvi_ndwi_df, mai_df, district, taluka, circle, sowing_date, current_date
        )

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["🌤️ Weather Metrics", "📊 Remote Sensing Indices", "💾 Downloadable Data"])

        with tab1:
            st.header(f"🌤️ Weather Metrics Comparison - {level}: {level_name}")
            
            # Fortnightly Analysis
            st.subheader("Fortnightly Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_rainfall_chart(weather_results, "fortnightly"), use_container_width=True)
                st.plotly_chart(create_temperature_chart(weather_results, "Tmax", "fortnightly"), use_container_width=True)
                st.plotly_chart(create_humidity_chart(weather_results, "max_Rh", "fortnightly"), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_rainy_days_chart(weather_results, "fortnightly"), use_container_width=True)
                st.plotly_chart(create_temperature_chart(weather_results, "Tmin", "fortnightly"), use_container_width=True)
                st.plotly_chart(create_humidity_chart(weather_results, "min_Rh", "fortnightly"), use_container_width=True)
            
            # Monthly Analysis
            st.subheader("Monthly Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_rainfall_chart(weather_results, "monthly"), use_container_width=True)
                st.plotly_chart(create_temperature_chart(weather_results, "Tmax", "monthly"), use_container_width=True)
                st.plotly_chart(create_humidity_chart(weather_results, "max_Rh", "monthly"), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_rainy_days_chart(weather_results, "monthly"), use_container_width=True)
                st.plotly_chart(create_temperature_chart(weather_results, "Tmin", "monthly"), use_container_width=True)
                st.plotly_chart(create_humidity_chart(weather_results, "min_Rh", "monthly"), use_container_width=True)

        with tab2:
            st.header(f"📊 Remote Sensing Indices - {level}: {level_name}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_ndvi_ndwi_line_chart(remote_sensing_results, "NDVI"), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_ndvi_ndwi_line_chart(remote_sensing_results, "NDWI"), use_container_width=True)
            
            st.plotly_chart(create_mai_chart(remote_sensing_results), use_container_width=True)

        with tab3:
            st.header(f"💾 Downloadable Data - {level}: {level_name}")
            
            # Data download sections
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Weather Data")
                if not weather_results["fortnightly"].empty:
                    weather_csv = weather_results["fortnightly"].to_csv(index=False)
                    st.download_button(
                        label="Download Fortnightly Weather Data (CSV)",
                        data=weather_csv,
                        file_name=f"weather_fortnightly_{level}_{level_name}.csv",
                        mime="text/csv"
                    )
                
                if not weather_results["monthly"].empty:
                    weather_monthly_csv = weather_results["monthly"].to_csv(index=False)
                    st.download_button(
                        label="Download Monthly Weather Data (CSV)",
                        data=weather_monthly_csv,
                        file_name=f"weather_monthly_{level}_{level_name}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                st.subheader("Remote Sensing Data")
                if not remote_sensing_results["ndvi_ndwi_timeseries"].empty:
                    ndvi_ndwi_csv = remote_sensing_results["ndvi_ndwi_timeseries"].to_csv(index=False)
                    st.download_button(
                        label="Download NDVI/NDWI Data (CSV)",
                        data=ndvi_ndwi_csv,
                        file_name=f"ndvi_ndwi_{level}_{level_name}.csv",
                        mime="text/csv"
                    )
                
                if not remote_sensing_results["mai_monthly"].empty:
                    mai_csv = remote_sensing_results["mai_monthly"].to_csv(index=False)
                    st.download_button(
                        label="Download MAI Data (CSV)",
                        data=mai_csv,
                        file_name=f"mai_{level}_{level_name}.csv",
                        mime="text/csv"
                    )
            
            # Data previews
            st.subheader("Data Previews")
            
            preview_tabs = st.tabs(["Weather Fortnightly", "Weather Monthly", "NDVI/NDWI", "MAI"])
            
            with preview_tabs[0]:
                if not weather_results["fortnightly"].empty:
                    st.dataframe(weather_results["fortnightly"], use_container_width=True)
                else:
                    st.info("No fortnightly weather data available")
            
            with preview_tabs[1]:
                if not weather_results["monthly"].empty:
                    st.dataframe(weather_results["monthly"], use_container_width=True)
                else:
                    st.info("No monthly weather data available")
            
            with preview_tabs[2]:
                if not remote_sensing_results["ndvi_ndwi_timeseries"].empty:
                    st.dataframe(remote_sensing_results["ndvi_ndwi_timeseries"].head(20), use_container_width=True)
                else:
                    st.info("No NDVI/NDWI data available")
                    
            with preview_tabs[3]:
                if not remote_sensing_results["mai_monthly"].empty:
                    st.dataframe(remote_sensing_results["mai_monthly"], use_container_width=True)
                else:
                    st.info("No MAI data available")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    """
    <div class='footer'>
        💻 <b>Developed by:</b> Ashish Selokar <br>
        📧 For suggestions or queries, please email at:
        <a href="mailto:ashish111.selokar@gmail.com">ashish111.selokar@gmail.com</a> <br><br>
        <span style="color:green;">
            🌾 Crop Health Comparison Dashboard 🌾
        </span><br>
        <span style="font-size:13px; color:gray;">
            Version 2.0 | Last Updated: Oct 2024
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
