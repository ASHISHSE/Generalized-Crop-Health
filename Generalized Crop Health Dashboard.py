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
import drive
drive.mount('/content/drive')

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
        <div class="main-title">Smart Crop Health Dashboard</div>
        <div class="subtitle">Compare 2023 vs 2024 Crop Health Data</div>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA - UPDATED URLS AS PER REQUIREMENTS
# -----------------------------
@st.cache_data
def load_data():
    # Updated URLs as per requirements
    ndvi_ndwi_url = "https://github.com/ASHISHSE/Generalized-Crop-Health/raw/main/1Maharashtra_NDVI_NDWI_old_circle_2023_2024_upload.xlsx"
    weather_url = pd.read_excel("/content/drive/MyDrive/advisory_Input/weather_data_2023_24_upload.xlsx")
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
        
        # Convert numeric columns for weather data
        for col in ["Rainfall", "Tmax", "Tmin", "max_Rh", "min_Rh"]:
            if col in weather_df.columns:
                weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")
        
        # Convert numeric columns for NDVI/NDWI data
        for col in ["NDVI", "NDWI"]:
            if col in ndvi_ndwi_df.columns:
                ndvi_ndwi_df[col] = pd.to_numeric(ndvi_ndwi_df[col], errors="coerce")
        
        # Get unique locations
        districts = sorted(weather_df["District"].dropna().unique().tolist())
        talukas = sorted(weather_df["Taluka"].dropna().unique().tolist())
        circles = sorted(weather_df["Circle"].dropna().unique().tolist())
        
        return ndvi_ndwi_df, weather_df, mai_df, districts, talukas, circles
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], [], []

# Load data
ndvi_ndwi_df, weather_df, mai_df, districts, talukas, circles = load_data()

# -----------------------------
# HELPER FUNCTIONS FOR DATA PROCESSING
# -----------------------------
def get_fortnightly_periods(start_date, end_date):
    """Generate fortnightly periods between start and end dates"""
    periods = []
    current = start_date.replace(day=1)
    
    while current <= end_date:
        # First fortnight (1-15)
        fn1_start = current.replace(day=1)
        fn1_end = current.replace(day=15)
        
        # Second fortnight (16-end of month)
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1)
        else:
            next_month = current.replace(month=current.month + 1)
        fn2_start = current.replace(day=16)
        fn2_end = next_month - timedelta(days=1)
        
        periods.append(("1FN", fn1_start, min(fn1_end, end_date)))
        if fn2_start <= end_date:
            periods.append(("2FN", fn2_start, min(fn2_end, end_date)))
        
        # Move to next month
        current = next_month
    
    return periods

def get_monthly_periods(start_date, end_date):
    """Generate monthly periods between start and end dates"""
    periods = []
    current = start_date.replace(day=1)
    
    while current <= end_date:
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1)
        else:
            next_month = current.replace(month=current.month + 1)
        
        month_end = next_month - timedelta(days=1)
        periods.append((current.strftime("%B"), current, min(month_end, end_date)))
        
        current = next_month
    
    return periods

def filter_data_by_location(df, district, taluka, circle):
    """Filter dataframe based on selected location"""
    filtered_df = df.copy()
    
    if district:
        filtered_df = filtered_df[filtered_df["District"] == district]
    if taluka and taluka != "":
        filtered_df = filtered_df[filtered_df["Taluka"] == taluka]
    if circle and circle != "":
        filtered_df = filtered_df[filtered_df["Circle"] == circle]
    
    return filtered_df

def calculate_rainfall_metrics(weather_data, periods, year):
    """Calculate rainfall metrics for given periods"""
    results = []
    
    for period_name, start_date, end_date in periods:
        # Filter data for current year
        current_year_data = weather_data[
            (weather_data["Date_dt"] >= start_date.replace(year=year)) & 
            (weather_data["Date_dt"] <= end_date.replace(year=year))
        ]
        
        # Filter data for previous year
        prev_year_data = weather_data[
            (weather_data["Date_dt"] >= start_date.replace(year=year-1)) & 
            (weather_data["Date_dt"] <= end_date.replace(year=year-1))
        ]
        
        # Calculate metrics
        current_rainfall = current_year_data["Rainfall"].sum() if not current_year_data.empty else 0
        prev_rainfall = prev_year_data["Rainfall"].sum() if not prev_year_data.empty else 0
        deviation = current_rainfall - prev_rainfall
        
        current_rainy_days = (current_year_data["Rainfall"] > 0).sum() if not current_year_data.empty else 0
        prev_rainy_days = (prev_year_data["Rainfall"] > 0).sum() if not prev_year_data.empty else 0
        rainy_days_deviation = current_rainy_days - prev_rainy_days
        
        results.append({
            'Period': period_name,
            'Year': year,
            'Rainfall': current_rainfall,
            'Rainfall_Prev_Year': prev_rainfall,
            'Rainfall_Deviation': deviation,
            'Rainy_Days': current_rainy_days,
            'Rainy_Days_Prev_Year': prev_rainy_days,
            'Rainy_Days_Deviation': rainy_days_deviation
        })
    
    return pd.DataFrame(results)

def calculate_temperature_metrics(weather_data, periods, year, temp_type):
    """Calculate temperature metrics (Tmax or Tmin)"""
    results = []
    col_name = "Tmax" if temp_type == "max" else "Tmin"
    
    for period_name, start_date, end_date in periods:
        # Filter data for current year (exclude 0 values)
        current_year_data = weather_data[
            (weather_data["Date_dt"] >= start_date.replace(year=year)) & 
            (weather_data["Date_dt"] <= end_date.replace(year=year)) &
            (weather_data[col_name] > 0)
        ]
        
        # Filter data for previous year (exclude 0 values)
        prev_year_data = weather_data[
            (weather_data["Date_dt"] >= start_date.replace(year=year-1)) & 
            (weather_data["Date_dt"] <= end_date.replace(year=year-1)) &
            (weather_data[col_name] > 0)
        ]
        
        # Calculate averages
        current_avg = current_year_data[col_name].mean() if not current_year_data.empty else 0
        prev_avg = prev_year_data[col_name].mean() if not prev_year_data.empty else 0
        deviation = current_avg - prev_avg
        
        results.append({
            'Period': period_name,
            'Year': year,
            f'{temp_type.capitalize()}_Temp_Avg': current_avg,
            f'{temp_type.capitalize()}_Temp_Prev_Year': prev_avg,
            f'{temp_type.capitalize()}_Temp_Deviation': deviation
        })
    
    return pd.DataFrame(results)

def calculate_humidity_metrics(weather_data, periods, year, rh_type):
    """Calculate humidity metrics (max_Rh or min_Rh)"""
    results = []
    col_name = "max_Rh" if rh_type == "max" else "min_Rh"
    
    for period_name, start_date, end_date in periods:
        # Filter data for current year (exclude 0 values)
        current_year_data = weather_data[
            (weather_data["Date_dt"] >= start_date.replace(year=year)) & 
            (weather_data["Date_dt"] <= end_date.replace(year=year)) &
            (weather_data[col_name] > 0)
        ]
        
        # Filter data for previous year (exclude 0 values)
        prev_year_data = weather_data[
            (weather_data["Date_dt"] >= start_date.replace(year=year-1)) & 
            (weather_data["Date_dt"] <= end_date.replace(year=year-1)) &
            (weather_data[col_name] > 0)
        ]
        
        # Calculate averages
        current_avg = current_year_data[col_name].mean() if not current_year_data.empty else 0
        prev_avg = prev_year_data[col_name].mean() if not prev_year_data.empty else 0
        deviation = current_avg - prev_avg
        
        results.append({
            'Period': period_name,
            'Year': year,
            f'{rh_type.capitalize()}_RH_Avg': current_avg,
            f'{rh_type.capitalize()}_RH_Prev_Year': prev_avg,
            f'{rh_type.capitalize()}_RH_Deviation': deviation
        })
    
    return pd.DataFrame(results)

def get_ndvi_ndwi_comparison(ndvi_ndwi_data, start_date, end_date, district, taluka, circle):
    """Get NDVI and NDWI data for comparison between years"""
    filtered_data = filter_data_by_location(ndvi_ndwi_data, district, taluka, circle)
    
    # Filter for 2023 data within date range
    data_2023 = filtered_data[
        (filtered_data["Date_dt"] >= start_date.replace(year=2023)) & 
        (filtered_data["Date_dt"] <= end_date.replace(year=2023))
    ].copy()
    data_2023['Year'] = 2023
    
    # Filter for 2024 data within date range
    data_2024 = filtered_data[
        (filtered_data["Date_dt"] >= start_date.replace(year=2024)) & 
        (filtered_data["Date_dt"] <= end_date.replace(year=2024))
    ].copy()
    data_2024['Year'] = 2024
    
    # Combine data
    combined_data = pd.concat([data_2023, data_2024], ignore_index=True)
    
    return combined_data

def get_mai_comparison(mai_data, start_date, end_date, district, taluka, circle):
    """Get MAI data for comparison between years"""
    filtered_data = filter_data_by_location(mai_data, district, taluka, circle)
    
    # Generate months in range
    months_in_range = []
    current = start_date.replace(day=1)
    while current <= end_date:
        months_in_range.append(current.strftime("%B"))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    # Filter data for relevant months and years
    comparison_data = filtered_data[
        (filtered_data["Month"].isin(months_in_range)) & 
        (filtered_data["Year"].isin([2023, 2024]))
    ].copy()
    
    # Calculate monthly averages excluding 0 values
    monthly_avg = []
    for year in [2023, 2024]:
        for month in months_in_range:
            month_data = comparison_data[
                (comparison_data["Year"] == year) & 
                (comparison_data["Month"] == month)
            ]
            # Exclude 0 values for averaging
            mai_values = month_data["MAI (%)"].replace(0, np.nan).dropna()
            avg_mai = mai_values.mean() if not mai_values.empty else 0
            
            monthly_avg.append({
                'Year': year,
                'Month': month,
                'Avg_MAI': avg_mai
            })
    
    return pd.DataFrame(monthly_avg)

# -----------------------------
# CHART FUNCTIONS
# -----------------------------
def create_rainfall_chart(rainfall_df, period_type):
    """Create clustered column chart for rainfall"""
    fig = go.Figure()
    
    # Current year bars
    fig.add_trace(go.Bar(
        name='2024',
        x=rainfall_df['Period'],
        y=rainfall_df['Rainfall'],
        marker_color='#1f77b4'
    ))
    
    # Previous year bars
    fig.add_trace(go.Bar(
        name='2023',
        x=rainfall_df['Period'],
        y=rainfall_df['Rainfall_Prev_Year'],
        marker_color='#ff7f0e'
    ))
    
    fig.update_layout(
        title=f"Rainfall Comparison - {period_type}",
        xaxis_title="Period",
        yaxis_title="Rainfall (mm)",
        barmode='group',
        template="plotly_white"
    )
    
    return fig

def create_rainy_days_chart(rainfall_df, period_type):
    """Create clustered column chart for rainy days"""
    fig = go.Figure()
    
    # Current year bars
    fig.add_trace(go.Bar(
        name='2024',
        x=rainfall_df['Period'],
        y=rainfall_df['Rainy_Days'],
        marker_color='#1f77b4'
    ))
    
    # Previous year bars
    fig.add_trace(go.Bar(
        name='2023',
        x=rainfall_df['Period'],
        y=rainfall_df['Rainy_Days_Prev_Year'],
        marker_color='#ff7f0e'
    ))
    
    fig.update_layout(
        title=f"Rainy Days Comparison - {period_type}",
        xaxis_title="Period",
        yaxis_title="Number of Rainy Days",
        barmode='group',
        template="plotly_white"
    )
    
    return fig

def create_temperature_chart(temp_df, period_type, temp_type):
    """Create clustered column chart for temperature"""
    fig = go.Figure()
    
    col_avg = f'{temp_type.capitalize()}_Temp_Avg'
    col_prev = f'{temp_type.capitalize()}_Temp_Prev_Year'
    
    # Current year bars
    fig.add_trace(go.Bar(
        name='2024',
        x=temp_df['Period'],
        y=temp_df[col_avg],
        marker_color='#1f77b4'
    ))
    
    # Previous year bars
    fig.add_trace(go.Bar(
        name='2023',
        x=temp_df['Period'],
        y=temp_df[col_prev],
        marker_color='#ff7f0e'
    ))
    
    temp_name = "Maximum" if temp_type == "max" else "Minimum"
    fig.update_layout(
        title=f"{temp_name} Temperature Comparison - {period_type}",
        xaxis_title="Period",
        yaxis_title=f"Temperature (°C)",
        barmode='group',
        template="plotly_white"
    )
    
    return fig

def create_humidity_chart(rh_df, period_type, rh_type):
    """Create clustered column chart for relative humidity"""
    fig = go.Figure()
    
    col_avg = f'{rh_type.capitalize()}_RH_Avg'
    col_prev = f'{rh_type.capitalize()}_RH_Prev_Year'
    
    # Current year bars
    fig.add_trace(go.Bar(
        name='2024',
        x=rh_df['Period'],
        y=rh_df[col_avg],
        marker_color='#1f77b4'
    ))
    
    # Previous year bars
    fig.add_trace(go.Bar(
        name='2023',
        x=rh_df['Period'],
        y=rh_df[col_prev],
        marker_color='#ff7f0e'
    ))
    
    rh_name = "Maximum" if rh_type == "max" else "Minimum"
    fig.update_layout(
        title=f"{rh_name} Relative Humidity Comparison - {period_type}",
        xaxis_title="Period",
        yaxis_title="Relative Humidity (%)",
        barmode='group',
        template="plotly_white"
    )
    
    return fig

def create_ndvi_ndwi_line_chart(comparison_data, index_type):
    """Create line chart for NDVI or NDWI comparison"""
    fig = go.Figure()
    
    for year in [2023, 2024]:
        year_data = comparison_data[comparison_data['Year'] == year].sort_values('Date_dt')
        
        if not year_data.empty:
            fig.add_trace(go.Scatter(
                x=year_data['Date_dt'],
                y=year_data[index_type],
                mode='lines+markers',
                name=str(year),
                line=dict(width=2)
            ))
    
    index_name = "NDVI" if index_type == "NDVI" else "NDWI"
    fig.update_layout(
        title=f"{index_name} Comparison (2023 vs 2024)",
        xaxis_title="Date",
        yaxis_title=f"{index_name} Value",
        template="plotly_white"
    )
    
    return fig

def create_mai_chart(mai_comparison_df):
    """Create clustered column chart for MAI comparison"""
    fig = go.Figure()
    
    # Convert month names to proper order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    mai_comparison_df['Month_Num'] = mai_comparison_df['Month'].apply(
        lambda x: month_order.index(x) + 1 if x in month_order else 13
    )
    mai_comparison_df = mai_comparison_df.sort_values('Month_Num')
    
    for year in [2023, 2024]:
        year_data = mai_comparison_df[mai_comparison_df['Year'] == year]
        
        fig.add_trace(go.Bar(
            name=str(year),
            x=year_data['Month'],
            y=year_data['Avg_MAI'],
            marker_color='#1f77b4' if year == 2024 else '#ff7f0e'
        ))
    
    fig.update_layout(
        title="MAI Comparison - Monthly Averages (2023 vs 2024)",
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

# Location selection
col1, col2, col3 = st.columns(3)

with col1:
    district = st.selectbox("District", [""] + districts)
    
with col2:
    if district:
        taluka_options = [""] + sorted(weather_df[weather_df["District"] == district]["Taluka"].dropna().unique().tolist())
    else:
        taluka_options = [""] + talukas
    taluka = st.selectbox("Taluka", taluka_options)
    
with col3:
    if taluka and taluka != "":
        circle_options = [""] + sorted(weather_df[weather_df["Taluka"] == taluka]["Circle"].dropna().unique().tolist())
    else:
        circle_options = [""] + circles
    circle = st.selectbox("Circle", circle_options)

# Date selection
col1, col2 = st.columns(2)

with col1:
    sowing_date = st.date_input("Sowing Date (dd/mm/yyyy)", value=date(2024, 1, 1), format="DD/MM/YYYY")
    
with col2:
    current_date = st.date_input("Current Date (dd/mm/yyyy)", value=date.today(), format="DD/MM/YYYY")

# Generate button
generate = st.button("📊 Generate Comparison Report")

# -----------------------------
# MAIN LOGIC
# -----------------------------
if generate:
    if not district:
        st.error("Please select at least a District.")
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

        st.info(f"📊 Generating comparison report for **{level}**: {level_name}")

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["🌤️ Weather Metrics", "🛰️ Remote Sensing Indices", "💾 Downloadable Data"])
        
        # Filter data by location
        filtered_weather = filter_data_by_location(weather_df, district, taluka, circle)
        filtered_ndvi_ndwi = filter_data_by_location(ndvi_ndwi_df, district, taluka, circle)
        filtered_mai = filter_data_by_location(mai_df, district, taluka, circle)
        
        # TAB 1: WEATHER METRICS
        with tab1:
            st.header(f"🌤️ Weather Metrics Comparison - {level}: {level_name}")
            
            # Calculate periods
            fortnightly_periods = get_fortnightly_periods(sowing_date, current_date)
            monthly_periods = get_monthly_periods(sowing_date, current_date)
            
            # Calculate metrics for 2024 (comparing with 2023)
            rainfall_fortnightly = calculate_rainfall_metrics(filtered_weather, fortnightly_periods, 2024)
            rainfall_monthly = calculate_rainfall_metrics(filtered_weather, monthly_periods, 2024)
            
            tmax_fortnightly = calculate_temperature_metrics(filtered_weather, fortnightly_periods, 2024, "max")
            tmax_monthly = calculate_temperature_metrics(filtered_weather, monthly_periods, 2024, "max")
            
            tmin_fortnightly = calculate_temperature_metrics(filtered_weather, fortnightly_periods, 2024, "min")
            tmin_monthly = calculate_temperature_metrics(filtered_weather, monthly_periods, 2024, "min")
            
            max_rh_fortnightly = calculate_humidity_metrics(filtered_weather, fortnightly_periods, 2024, "max")
            max_rh_monthly = calculate_humidity_metrics(filtered_weather, monthly_periods, 2024, "max")
            
            min_rh_fortnightly = calculate_humidity_metrics(filtered_weather, fortnightly_periods, 2024, "min")
            min_rh_monthly = calculate_humidity_metrics(filtered_weather, monthly_periods, 2024, "min")
            
            # Display charts in expandable sections
            with st.expander("🌧️ Rainfall Analysis", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    if not rainfall_fortnightly.empty:
                        st.plotly_chart(create_rainfall_chart(rainfall_fortnightly, "Fortnightly"), use_container_width=True)
                    else:
                        st.info("No fortnightly rainfall data available")
                
                with col2:
                    if not rainfall_monthly.empty:
                        st.plotly_chart(create_rainfall_chart(rainfall_monthly, "Monthly"), use_container_width=True)
                    else:
                        st.info("No monthly rainfall data available")
            
            with st.expander("☔ Rainy Days Analysis", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    if not rainfall_fortnightly.empty:
                        st.plotly_chart(create_rainy_days_chart(rainfall_fortnightly, "Fortnightly"), use_container_width=True)
                    else:
                        st.info("No fortnightly rainy days data available")
                
                with col2:
                    if not rainfall_monthly.empty:
                        st.plotly_chart(create_rainy_days_chart(rainfall_monthly, "Monthly"), use_container_width=True)
                    else:
                        st.info("No monthly rainy days data available")
            
            with st.expander("🌡️ Maximum Temperature Analysis", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    if not tmax_fortnightly.empty:
                        st.plotly_chart(create_temperature_chart(tmax_fortnightly, "Fortnightly", "max"), use_container_width=True)
                    else:
                        st.info("No fortnightly maximum temperature data available")
                
                with col2:
                    if not tmax_monthly.empty:
                        st.plotly_chart(create_temperature_chart(tmax_monthly, "Monthly", "max"), use_container_width=True)
                    else:
                        st.info("No monthly maximum temperature data available")
            
            with st.expander("🌡️ Minimum Temperature Analysis", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    if not tmin_fortnightly.empty:
                        st.plotly_chart(create_temperature_chart(tmin_fortnightly, "Fortnightly", "min"), use_container_width=True)
                    else:
                        st.info("No fortnightly minimum temperature data available")
                
                with col2:
                    if not tmin_monthly.empty:
                        st.plotly_chart(create_temperature_chart(tmin_monthly, "Monthly", "min"), use_container_width=True)
                    else:
                        st.info("No monthly minimum temperature data available")
            
            with st.expander("💧 Maximum Relative Humidity Analysis", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    if not max_rh_fortnightly.empty:
                        st.plotly_chart(create_humidity_chart(max_rh_fortnightly, "Fortnightly", "max"), use_container_width=True)
                    else:
                        st.info("No fortnightly maximum humidity data available")
                
                with col2:
                    if not max_rh_monthly.empty:
                        st.plotly_chart(create_humidity_chart(max_rh_monthly, "Monthly", "max"), use_container_width=True)
                    else:
                        st.info("No monthly maximum humidity data available")
            
            with st.expander("💧 Minimum Relative Humidity Analysis", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    if not min_rh_fortnightly.empty:
                        st.plotly_chart(create_humidity_chart(min_rh_fortnightly, "Fortnightly", "min"), use_container_width=True)
                    else:
                        st.info("No fortnightly minimum humidity data available")
                
                with col2:
                    if not min_rh_monthly.empty:
                        st.plotly_chart(create_humidity_chart(min_rh_monthly, "Monthly", "min"), use_container_width=True)
                    else:
                        st.info("No monthly minimum humidity data available")
        
        # TAB 2: REMOTE SENSING INDICES
        with tab2:
            st.header(f"🛰️ Remote Sensing Indices - {level}: {level_name}")
            
            # Get NDVI/NDWI comparison data
            ndvi_ndwi_comparison = get_ndvi_ndwi_comparison(
                filtered_ndvi_ndwi, sowing_date, current_date, district, taluka, circle
            )
            
            # Get MAI comparison data
            mai_comparison = get_mai_comparison(
                filtered_mai, sowing_date, current_date, district, taluka, circle
            )
            
            # Display charts
            col1, col2 = st.columns(2)
            
            with col1:
                if not ndvi_ndwi_comparison.empty:
                    st.plotly_chart(create_ndvi_ndwi_line_chart(ndvi_ndwi_comparison, "NDVI"), use_container_width=True)
                else:
                    st.info("No NDVI comparison data available")
            
            with col2:
                if not ndvi_ndwi_comparison.empty:
                    st.plotly_chart(create_ndvi_ndwi_line_chart(ndvi_ndwi_comparison, "NDWI"), use_container_width=True)
                else:
                    st.info("No NDWI comparison data available")
            
            # MAI Chart
            st.subheader("🌧️ MAI Analysis")
            if not mai_comparison.empty:
                st.plotly_chart(create_mai_chart(mai_comparison), use_container_width=True)
            else:
                st.info("No MAI comparison data available")
        
        # TAB 3: DOWNLOADABLE DATA
        with tab3:
            st.header(f"💾 Downloadable Data - {level}: {level_name}")
            
            # Prepare data for download
            col1, col2 = st.columns(2)
            
            with col1:
                # Weather Data
                st.subheader("🌤️ Weather Data")
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
                
                # NDVI/NDWI Data
                st.subheader("🛰️ NDVI & NDWI Data")
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
            
            with col2:
                # MAI Data
                st.subheader("🌧️ MAI Data")
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
                
                # Comparison Data
                st.subheader("📊 Comparison Data")
                if not rainfall_monthly.empty:
                    comparison_data = pd.concat([
                        rainfall_monthly,
                        tmax_monthly,
                        tmin_monthly,
                        max_rh_monthly,
                        min_rh_monthly
                    ], ignore_index=True)
                    
                    comparison_csv = comparison_data.to_csv(index=False)
                    st.download_button(
                        label="Download Comparison Data (CSV)",
                        data=comparison_csv,
                        file_name=f"comparison_data_{level}_{level_name}.csv",
                        mime="text/csv"
                    )
                else:
                    st.write("No comparison data available")

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
            🌾 Crop Health Comparison Dashboard 🌾
        </span><br>
        <span style="font-size:13px; color:gray;">
            Version 2.0 | Last Updated: Oct 2024
        </span>
    </div>
    """,
    unsafe_allow_html=True
)


