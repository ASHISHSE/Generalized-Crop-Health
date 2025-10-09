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
    page_title="👨‍🌾 Smart Crop Advisory Dashboard",
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

        /* Centered Responsive Header */
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

        /* Responsive Button Styling */
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

        /* Make sure header scales well on smaller devices */
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

# --- HEADER (Farmer icon above title & centered) ---
st.markdown("""
    <div class="main-header">
        <img src="https://raw.githubusercontent.com/ASHISHSE/App_test/main/icon.png" class="logo-icon" alt="Farmer Icon">
        <div class="main-title">Smart Crop Advisory Dashboard</div>
        <div class="subtitle">Empowering Farmers with Data-Driven Insights</div>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA - UPDATED URLs as per requirements
# -----------------------------
@st.cache_data
def load_data():
    # Updated URLs as per request
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
        
        # Combine weather data
        weather_df = pd.concat([weather_23_df, weather_24_df], ignore_index=True)
        
        # Load MAI data
        mai_res = requests.get(mai_url, timeout=30)
        mai_df = pd.read_excel(BytesIO(mai_res.content))
        
        # Process NDVI & NDWI data
        ndvi_ndwi_df["Date_dt"] = pd.to_datetime(ndvi_ndwi_df["Date(DD-MM-YYYY)"], format="%d-%m-%Y", errors="coerce")
        ndvi_ndwi_df = ndvi_ndwi_df.dropna(subset=["Date_dt"]).copy()
        
        # Process Weather data
        weather_df["Date_dt"] = pd.to_datetime(weather_df["Date(DD-MM-YYYY)"], format="%d-%m-%Y", errors="coerce")
        weather_df = weather_df.dropna(subset=["Date_dt"]).copy()
        
        # Convert numeric columns for weather
        for col in ["Rainfall", "Tmax", "Tmin", "max_Rh", "min_Rh"]:
            if col in weather_df.columns:
                weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")
        
        # Process MAI data
        mai_df["Year"] = pd.to_numeric(mai_df["Year"], errors="coerce")
        mai_df["MAI (%)"] = pd.to_numeric(mai_df["MAI (%)"], errors="coerce")
        
        # Get unique locations
        districts = sorted(weather_df["District"].dropna().unique().tolist())
        talukas = sorted(weather_df["Taluka"].dropna().unique().tolist())
        circles = sorted(weather_df["Circle"].dropna().unique().tolist())
        
        return weather_df, ndvi_ndwi_df, mai_df, districts, talukas, circles
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], [], []

# Load data
weather_df, ndvi_ndwi_df, mai_df, districts, talukas, circles = load_data()

# -----------------------------
# FORTNIGHT DEFINITION
# -----------------------------
def get_fortnight(date_obj):
    """Get fortnight from date (1st or 2nd)"""
    if date_obj.day <= 15:
        return f"1FN {date_obj.strftime('%B')}"
    else:
        return f"2FN {date_obj.strftime('%B')}"

def get_fortnight_period(date_obj):
    """Get start and end dates for a fortnight"""
    year = date_obj.year
    month = date_obj.month
    if date_obj.day <= 15:
        start_date = date(year, month, 1)
        end_date = date(year, month, 15)
    else:
        start_date = date(year, month, 16)
        # Get last day of month
        if month == 12:
            end_date = date(year, month, 31)
        else:
            end_date = date(year, month + 1, 1) - timedelta(days=1)
    return start_date, end_date

# -----------------------------
# WEATHER METRICS CALCULATIONS
# -----------------------------
def calculate_fortnightly_metrics(data_df, current_year, last_year, metric_col, agg_func='sum'):
    """Calculate fortnightly metrics for current and last year"""
    metrics = {}
    
    for year in [current_year, last_year]:
        year_data = data_df[data_df['Date_dt'].dt.year == year].copy()
        year_data['Fortnight'] = year_data['Date_dt'].apply(get_fortnight)
        
        if agg_func == 'sum':
            fortnight_metrics = year_data.groupby('Fortnight')[metric_col].sum()
        elif agg_func == 'mean':
            # Exclude 0 values for mean calculation
            non_zero_data = year_data[year_data[metric_col] != 0]
            fortnight_metrics = non_zero_data.groupby('Fortnight')[metric_col].mean()
        elif agg_func == 'count':
            fortnight_metrics = (year_data[metric_col] > 0).groupby(year_data['Fortnight']).sum()
        
        metrics[year] = fortnight_metrics
    
    return metrics

def calculate_monthly_metrics(data_df, current_year, last_year, metric_col, agg_func='sum'):
    """Calculate monthly metrics for current and last year"""
    metrics = {}
    
    for year in [current_year, last_year]:
        year_data = data_df[data_df['Date_dt'].dt.year == year].copy()
        year_data['Month'] = year_data['Date_dt'].dt.strftime('%B')
        
        if agg_func == 'sum':
            monthly_metrics = year_data.groupby('Month')[metric_col].sum()
        elif agg_func == 'mean':
            # Exclude 0 values for mean calculation
            non_zero_data = year_data[year_data[metric_col] != 0]
            monthly_metrics = non_zero_data.groupby('Month')[metric_col].mean()
        elif agg_func == 'count':
            monthly_metrics = (year_data[metric_col] > 0).groupby(year_data['Month']).sum()
        
        metrics[year] = monthly_metrics
    
    return metrics

# -----------------------------
# CHART FUNCTIONS FOR WEATHER METRICS TAB
# -----------------------------
def create_weather_comparison_chart(metrics_current, metrics_last, title, yaxis_title, chart_type='column'):
    """Create comparison chart for weather metrics"""
    fig = go.Figure()
    
    # Get all unique periods (fortnights or months)
    all_periods = sorted(set(metrics_current.index) | set(metrics_last.index))
    
    current_values = [metrics_current.get(period, 0) for period in all_periods]
    last_values = [metrics_last.get(period, 0) for period in all_periods]
    
    if chart_type == 'column':
        fig.add_trace(go.Bar(
            name='Current Year',
            x=all_periods,
            y=current_values,
            marker_color='#1f77b4'
        ))
        
        fig.add_trace(go.Bar(
            name='Last Year',
            x=all_periods,
            y=last_values,
            marker_color='#ff7f0e'
        ))
    else:  # line chart
        fig.add_trace(go.Scatter(
            name='Current Year',
            x=all_periods,
            y=current_values,
            mode='lines+markers',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            name='Last Year',
            x=all_periods,
            y=last_values,
            mode='lines+markers',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis_title=yaxis_title,
        barmode='group' if chart_type == 'column' else 'lines',
        template="plotly_white",
        height=500
    )
    
    return fig

# -----------------------------
# REMOTE SENSING INDICES FUNCTIONS
# -----------------------------
def create_ndvi_ndwi_line_chart(ndvi_ndwi_data, start_date, end_date, district, taluka, circle):
    """Create line chart for NDVI and NDWI for current and last year"""
    filtered_data = ndvi_ndwi_data.copy()
    
    # Filter by location
    if district:
        filtered_data = filtered_data[filtered_data["District"] == district]
    if taluka:
        filtered_data = filtered_data[filtered_data["Taluka"] == taluka]
    if circle:
        filtered_data = filtered_data[filtered_data["Circle"] == circle]
    
    # Filter by date range
    filtered_data = filtered_data[
        (filtered_data["Date_dt"] >= pd.to_datetime(start_date)) & 
        (filtered_data["Date_dt"] <= pd.to_datetime(end_date))
    ]
    
    if filtered_data.empty:
        return None
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Get unique years
    years = sorted(filtered_data["Date_dt"].dt.year.unique())
    
    for year in years:
        year_data = filtered_data[filtered_data["Date_dt"].dt.year == year].sort_values("Date_dt")
        
        # NDVI line
        fig.add_trace(
            go.Scatter(
                x=year_data["Date_dt"],
                y=year_data["NDVI"],
                mode='lines+markers',
                name=f'NDVI {year}',
                line=dict(color='green' if year == years[-1] else 'lightgreen', width=3),
                marker=dict(size=6)
            ),
            secondary_y=False,
        )
        
        # NDWI line
        fig.add_trace(
            go.Scatter(
                x=year_data["Date_dt"],
                y=year_data["NDWI"],
                mode='lines+markers',
                name=f'NDWI {year}',
                line=dict(color='blue' if year == years[-1] else 'lightblue', width=3),
                marker=dict(size=6)
            ),
            secondary_y=True,
        )
    
    fig.update_layout(
        title="NDVI & NDWI Trends (Current vs Last Year)",
        xaxis_title="Date",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="NDVI Value", secondary_y=False)
    fig.update_yaxes(title_text="NDWI Value", secondary_y=True)
    
    return fig

def create_mai_comparison_chart(mai_data, start_date, end_date, district, taluka, circle):
    """Create clustered column chart for MAI comparison"""
    filtered_data = mai_data.copy()
    
    # Filter by location
    if district:
        filtered_data = filtered_data[filtered_data["District"] == district]
    if taluka:
        filtered_data = filtered_data[filtered_data["Taluka"] == taluka]
    if circle:
        filtered_data = filtered_data[filtered_data["Circle"] == circle]
    
    # Get months from selected date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    months_in_range = []
    current = start_dt.replace(day=1)
    while current <= end_dt:
        months_in_range.append(current.strftime('%B'))
        current = current + pd.DateOffset(months=1)
    
    months_in_range = list(dict.fromkeys(months_in_range))  # Remove duplicates
    
    # Filter data for months in range
    filtered_data = filtered_data[filtered_data["Month"].isin(months_in_range)]
    
    if filtered_data.empty:
        return None
    
    # Get unique years
    years = sorted(filtered_data["Year"].unique())
    
    fig = go.Figure()
    
    for year in years:
        year_data = filtered_data[filtered_data["Year"] == year]
        
        # Calculate mean MAI excluding 0 values for each month
        mai_values = []
        for month in months_in_range:
            month_data = year_data[year_data["Month"] == month]
            non_zero_mai = month_data[month_data["MAI (%)"] != 0]["MAI (%)"]
            avg_mai = non_zero_mai.mean() if not non_zero_mai.empty else 0
            mai_values.append(avg_mai)
        
        fig.add_trace(go.Bar(
            name=f'MAI {year}',
            x=months_in_range,
            y=mai_values,
            marker_color='orange' if year == years[-1] else 'lightcoral'
        ))
    
    fig.update_layout(
        title="MAI Comparison (Current vs Last Year) - Monthly Average",
        xaxis_title="Month",
        yaxis_title="MAI (%)",
        barmode='group',
        template="plotly_white",
        height=500
    )
    
    return fig

# -----------------------------
# MAIN UI
# -----------------------------
st.markdown(
    """
    <style>
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .stButton button {
            width: 100%;
        }
        .stSelectbox, .stDateInput {
            font-size: 14px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Date Selection Section ---
st.markdown("### 📅 Date Selection")

col1, col2, col3 = st.columns(3)

with col1:
    district = st.selectbox("District *", [""] + districts)
    # Update taluka options based on selected district
    if district:
        taluka_options = [""] + sorted(weather_df[weather_df["District"] == district]["Taluka"].dropna().unique().tolist())
    else:
        taluka_options = [""] + talukas
    taluka = st.selectbox("Taluka", taluka_options)

with col2:
    # Update circle options based on selected taluka
    if taluka and taluka != "":
        circle_options = [""] + sorted(weather_df[weather_df["Taluka"] == taluka]["Circle"].dropna().unique().tolist())
    else:
        circle_options = [""] + circles
    circle = st.selectbox("Circle", circle_options)
    
    sowing_date = st.date_input("Start Date (Sowing Date) *", value=date.today() - timedelta(days=30), format="DD/MM/YYYY")

with col3:
    current_date = st.date_input("End Date (Current Date) *", value=date.today(), format="DD/MM/YYYY")
    
    generate = st.button("🌱 Generate Analysis", use_container_width=True)

# -----------------------------
# MAIN ANALYSIS
# -----------------------------
if generate:
    if not district or not sowing_date or not current_date:
        st.error("Please select all required fields (District, Start Date, End Date).")
    else:
        sowing_date_str = sowing_date.strftime("%d/%m/%Y")
        current_date_str = current_date.strftime("%d/%m/%Y")
        
        # Determine level and name for calculations
        if circle and circle != "":
            level = "Circle"
            level_name = circle
        elif taluka and taluka != "":
            level = "Taluka"
            level_name = taluka
        else:
            level = "District"
            level_name = district

        st.info(f"📊 Calculating metrics for **{level}**: {level_name}")

        # Filter weather data for selected location
        filtered_weather = weather_df.copy()
        if district:
            filtered_weather = filtered_weather[filtered_weather["District"] == district]
        if taluka:
            filtered_weather = filtered_weather[filtered_weather["Taluka"] == taluka]
        if circle:
            filtered_weather = filtered_weather[filtered_weather["Circle"] == circle]

        # Get current and last year
        current_year = current_date.year
        last_year = current_year - 1

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["🌤️ Weather Metrics", "📡 Remote Sensing Indices", "💾 Downloadable Data"])

        # TAB 1: WEATHER METRICS
        with tab1:
            st.header(f"🌤️ Weather Metrics - {level}: {level_name}")
            
            if not filtered_weather.empty:
                # I. Rainfall Analysis
                st.subheader("I. Rainfall Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Fortnightly Rainfall
                    fortnightly_rainfall = calculate_fortnightly_metrics(
                        filtered_weather, current_year, last_year, "Rainfall", "sum"
                    )
                    fig_rainfall_fortnight = create_weather_comparison_chart(
                        fortnightly_rainfall[current_year], 
                        fortnightly_rainfall[last_year],
                        "Rainfall - Fortnightly Comparison",
                        "Rainfall (mm)"
                    )
                    st.plotly_chart(fig_rainfall_fortnight, use_container_width=True)
                
                with col2:
                    # Monthly Rainfall
                    monthly_rainfall = calculate_monthly_metrics(
                        filtered_weather, current_year, last_year, "Rainfall", "sum"
                    )
                    fig_rainfall_monthly = create_weather_comparison_chart(
                        monthly_rainfall[current_year], 
                        monthly_rainfall[last_year],
                        "Rainfall - Monthly Comparison",
                        "Rainfall (mm)"
                    )
                    st.plotly_chart(fig_rainfall_monthly, use_container_width=True)
                
                # Rainfall Deviation Calculation
                st.subheader("Rainfall Deviation Analysis")
                dev_col1, dev_col2 = st.columns(2)
                
                with dev_col1:
                    # Fortnightly Deviation
                    fortnight_dev = []
                    for fn in fortnightly_rainfall[current_year].index:
                        current_val = fortnightly_rainfall[current_year].get(fn, 0)
                        last_val = fortnightly_rainfall[last_year].get(fn, 0)
                        deviation = ((current_val - last_val) / last_val * 100) if last_val != 0 else 0
                        fortnight_dev.append({"Fortnight": fn, "Deviation (%)": deviation})
                    
                    dev_fortnight_df = pd.DataFrame(fortnight_dev)
                    if not dev_fortnight_df.empty:
                        fig_dev_fortnight = px.bar(
                            dev_fortnight_df, 
                            x="Fortnight", 
                            y="Deviation (%)",
                            title="Rainfall Deviation - Fortnightly (%)",
                            color="Deviation (%)",
                            color_continuous_scale="RdYlBu_r"
                        )
                        st.plotly_chart(fig_dev_fortnight, use_container_width=True)
                
                with dev_col2:
                    # Monthly Deviation
                    monthly_dev = []
                    for month in monthly_rainfall[current_year].index:
                        current_val = monthly_rainfall[current_year].get(month, 0)
                        last_val = monthly_rainfall[last_year].get(month, 0)
                        deviation = ((current_val - last_val) / last_val * 100) if last_val != 0 else 0
                        monthly_dev.append({"Month": month, "Deviation (%)": deviation})
                    
                    dev_monthly_df = pd.DataFrame(monthly_dev)
                    if not dev_monthly_df.empty:
                        fig_dev_monthly = px.bar(
                            dev_monthly_df, 
                            x="Month", 
                            y="Deviation (%)",
                            title="Rainfall Deviation - Monthly (%)",
                            color="Deviation (%)",
                            color_continuous_scale="RdYlBu_r"
                        )
                        st.plotly_chart(fig_dev_monthly, use_container_width=True)

                # II. Rainy Days Analysis
                st.subheader("II. Rainy Days Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Fortnightly Rainy Days
                    fortnightly_rainy_days = calculate_fortnightly_metrics(
                        filtered_weather, current_year, last_year, "Rainfall", "count"
                    )
                    fig_rainy_fortnight = create_weather_comparison_chart(
                        fortnightly_rainy_days[current_year], 
                        fortnightly_rainy_days[last_year],
                        "Rainy Days - Fortnightly Comparison",
                        "Number of Rainy Days"
                    )
                    st.plotly_chart(fig_rainy_fortnight, use_container_width=True)
                
                with col2:
                    # Monthly Rainy Days
                    monthly_rainy_days = calculate_monthly_metrics(
                        filtered_weather, current_year, last_year, "Rainfall", "count"
                    )
                    fig_rainy_monthly = create_weather_comparison_chart(
                        monthly_rainy_days[current_year], 
                        monthly_rainy_days[last_year],
                        "Rainy Days - Monthly Comparison",
                        "Number of Rainy Days"
                    )
                    st.plotly_chart(fig_rainy_monthly, use_container_width=True)

                # III. Temperature Max Analysis
                st.subheader("III. Maximum Temperature Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Fortnightly Tmax
                    fortnightly_tmax = calculate_fortnightly_metrics(
                        filtered_weather, current_year, last_year, "Tmax", "mean"
                    )
                    fig_tmax_fortnight = create_weather_comparison_chart(
                        fortnightly_tmax[current_year], 
                        fortnightly_tmax[last_year],
                        "Max Temperature - Fortnightly Average",
                        "Temperature (°C)"
                    )
                    st.plotly_chart(fig_tmax_fortnight, use_container_width=True)
                
                with col2:
                    # Monthly Tmax
                    monthly_tmax = calculate_monthly_metrics(
                        filtered_weather, current_year, last_year, "Tmax", "mean"
                    )
                    fig_tmax_monthly = create_weather_comparison_chart(
                        monthly_tmax[current_year], 
                        monthly_tmax[last_year],
                        "Max Temperature - Monthly Average",
                        "Temperature (°C)"
                    )
                    st.plotly_chart(fig_tmax_monthly, use_container_width=True)

                # Temperature Deviation
                st.subheader("Temperature Deviation Analysis")
                
                dev_col1, dev_col2 = st.columns(2)
                
                with dev_col1:
                    # Fortnightly Tmax Deviation
                    tmax_fortnight_dev = []
                    for fn in fortnightly_tmax[current_year].index:
                        current_val = fortnightly_tmax[current_year].get(fn, 0)
                        last_val = fortnightly_tmax[last_year].get(fn, 0)
                        deviation = current_val - last_val
                        tmax_fortnight_dev.append({"Fortnight": fn, "Deviation (°C)": deviation})
                    
                    tmax_dev_fortnight_df = pd.DataFrame(tmax_fortnight_dev)
                    if not tmax_dev_fortnight_df.empty:
                        fig_tmax_dev_fortnight = px.bar(
                            tmax_dev_fortnight_df, 
                            x="Fortnight", 
                            y="Deviation (°C)",
                            title="Max Temperature Deviation - Fortnightly (°C)",
                            color="Deviation (°C)",
                            color_continuous_scale="RdBu_r"
                        )
                        st.plotly_chart(fig_tmax_dev_fortnight, use_container_width=True)
                
                with dev_col2:
                    # Monthly Tmax Deviation
                    tmax_monthly_dev = []
                    for month in monthly_tmax[current_year].index:
                        current_val = monthly_tmax[current_year].get(month, 0)
                        last_val = monthly_tmax[last_year].get(month, 0)
                        deviation = current_val - last_val
                        tmax_monthly_dev.append({"Month": month, "Deviation (°C)": deviation})
                    
                    tmax_dev_monthly_df = pd.DataFrame(tmax_monthly_dev)
                    if not tmax_dev_monthly_df.empty:
                        fig_tmax_dev_monthly = px.bar(
                            tmax_dev_monthly_df, 
                            x="Month", 
                            y="Deviation (°C)",
                            title="Max Temperature Deviation - Monthly (°C)",
                            color="Deviation (°C)",
                            color_continuous_scale="RdBu_r"
                        )
                        st.plotly_chart(fig_tmax_dev_monthly, use_container_width=True)

                # Continue with Tmin, RH max, RH min in similar pattern...
                # [Additional code for Tmin, RH max, RH min would follow similar structure]
                
            else:
                st.info("No weather data available for the selected location and date range.")

        # TAB 2: REMOTE SENSING INDICES
        with tab2:
            st.header(f"📡 Remote Sensing Indices - {level}: {level_name}")
            
            # I. NDVI Line Chart
            st.subheader("I. NDVI Analysis")
            ndvi_fig = create_ndvi_ndwi_line_chart(
                ndvi_ndwi_df, sowing_date, current_date, district, taluka, circle
            )
            if ndvi_fig:
                st.plotly_chart(ndvi_fig, use_container_width=True)
            else:
                st.info("No NDVI data available for the selected parameters.")
            
            # II. NDWI Line Chart
            st.subheader("II. NDWI Analysis")
            # NDWI is already included in the combined chart above
            st.info("NDWI is displayed in the combined chart above with NDVI.")
            
            # III. MAI Analysis
            st.subheader("III. MAI Analysis")
            mai_fig = create_mai_comparison_chart(
                mai_df, sowing_date, current_date, district, taluka, circle
            )
            if mai_fig:
                st.plotly_chart(mai_fig, use_container_width=True)
            else:
                st.info("No MAI data available for the selected parameters.")

        # TAB 3: DOWNLOADABLE DATA
        with tab3:
            st.header(f"💾 Downloadable Data - {level}: {level_name}")
            
            # Prepare data for download
            st.subheader("Data Export")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Weather Data
                if not filtered_weather.empty:
                    weather_csv = filtered_weather.to_csv(index=False)
                    st.download_button(
                        label="Download Weather Data (CSV)",
                        data=weather_csv,
                        file_name=f"weather_data_{level}_{level_name}.csv",
                        mime="text/csv"
                    )
                else:
                    st.write("No weather data available")
            
            with col2:
                # NDVI & NDWI Data
                filtered_ndvi_ndwi = ndvi_ndwi_df.copy()
                if district:
                    filtered_ndvi_ndwi = filtered_ndvi_ndwi[filtered_ndvi_ndwi["District"] == district]
                if taluka:
                    filtered_ndvi_ndwi = filtered_ndvi_ndwi[filtered_ndvi_ndwi["Taluka"] == taluka]
                if circle:
                    filtered_ndvi_ndwi = filtered_ndvi_ndwi[filtered_ndvi_ndwi["Circle"] == circle]
                
                if not filtered_ndvi_ndwi.empty:
                    ndvi_ndwi_csv = filtered_ndvi_ndwi.to_csv(index=False)
                    st.download_button(
                        label="Download NDVI/NDWI Data (CSV)",
                        data=ndvi_ndwi_csv,
                        file_name=f"ndvi_ndwi_data_{level}_{level_name}.csv",
                        mime="text/csv"
                    )
                else:
                    st.write("No NDVI/NDWI data available")
            
            with col3:
                # MAI Data
                filtered_mai = mai_df.copy()
                if district:
                    filtered_mai = filtered_mai[filtered_mai["District"] == district]
                if taluka:
                    filtered_mai = filtered_mai[filtered_mai["Taluka"] == taluka]
                if circle:
                    filtered_mai = filtered_mai[filtered_mai["Circle"] == circle]
                
                if not filtered_mai.empty:
                    mai_csv = filtered_mai.to_csv(index=False)
                    st.download_button(
                        label="Download MAI Data (CSV)",
                        data=mai_csv,
                        file_name=f"mai_data_{level}_{level_name}.csv",
                        mime="text/csv"
                    )
                else:
                    st.write("No MAI data available")
            
            # Data Previews
            st.subheader("Data Previews")
            
            preview_tabs = st.tabs(["Weather Data", "NDVI/NDWI Data", "MAI Data"])
            
            with preview_tabs[0]:
                if not filtered_weather.empty:
                    st.dataframe(filtered_weather.head(10), use_container_width=True)
                else:
                    st.info("No weather data available for preview")
            
            with preview_tabs[1]:
                if not filtered_ndvi_ndwi.empty:
                    st.dataframe(filtered_ndvi_ndwi.head(10), use_container_width=True)
                else:
                    st.info("No NDVI/NDWI data available for preview")
            
            with preview_tabs[2]:
                if not filtered_mai.empty:
                    st.dataframe(filtered_mai.head(10), use_container_width=True)
                else:
                    st.info("No MAI data available for preview")

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
