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
    page_title="👨‍🌾 Generalized Crop Health Dashboard",
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
        <div class="main-title">Smart Crop Advisory Dashboard</div>
        <div class="subtitle">Weather Parameters & Satellite-Based Crop Health Indicators</div>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA - UPDATED AS PER REQUIREMENTS
# -----------------------------
@st.cache_data
def load_data():
    # Updated file paths as per requirements
    ndvi_ndwi_url = "https://github.com/ASHISHSE/Generalized-Crop-Health/blob/main/Maharashtra_NDVI_NDWI_old_circle_2023_2024_upload.xlsx"
    weather_url = "https://docs.google.com/spreadsheets/d/1VZ58Kv3_cC_IP_0eSWBzOvnb2sOpPUOU/edit?usp=sharing&ouid=112417876654948113262&rtpof=true&sd=true"
    mai_url = "https://github.com/ASHISHSE/Generalized-Crop-Health/blob/main/Circlewise_Data_MAI_2023_24_upload.xlsx"
    
    try:
        # Load NDVI & NDWI data
        ndvi_ndwi_df = pd.read_excel(ndvi_ndwi_url)
        
        # Load Weather data (both sheets)
        weather_2023_df = pd.read_excel(weather_url, sheet_name='Weather_data_23')
        weather_2024_df = pd.read_excel(weather_url, sheet_name='Weather_data_24')
        
        # Load MAI data
        mai_df = pd.read_excel(mai_url)
        
        return ndvi_ndwi_df, weather_2023_df, weather_2024_df, mai_df
        
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        return None, None, None, None

# Load the data
ndvi_ndwi_df, weather_2023_df, weather_2024_df, mai_df = load_data()

# -----------------------------
# DATA PROCESSING FUNCTIONS
# -----------------------------
def process_ndvi_ndwi_data(df):
    """Process NDVI & NDWI data"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%Y_%m_%d', errors='coerce')
    
    # Extract year and month for comparison
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month_name()
    
    return df

def process_weather_data(df_2023, df_2024):
    """Process weather data from both years"""
    if df_2023 is None or df_2024 is None:
        return pd.DataFrame(), pd.DataFrame()
    
    # Process 2023 data
    df_2023['Date(DD-MM-YYYY)'] = pd.to_datetime(df_2023['Date(DD-MM-YYYY)'], format='%d-%m-%Y', errors='coerce')
    df_2023['Year'] = 2023
    df_2023['Month'] = df_2023['Date(DD-MM-YYYY)'].dt.month_name()
    
    # Process 2024 data  
    df_2024['Date(DD-MM-YYYY)'] = pd.to_datetime(df_2024['Date(DD-MM-YYYY)'], format='%d-%m-%Y', errors='coerce')
    df_2024['Year'] = 2024
    df_2024['Month'] = df_2024['Date(DD-MM-YYYY)'].dt.month_name()
    
    return df_2023, df_2024

def process_mai_data(df):
    """Process MAI data"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Map month numbers to names
    month_map = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
        7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    df['Month'] = df['Month'].map(month_map)
    
    return df

# Process the data
ndvi_ndwi_processed = process_ndvi_ndwi_data(ndvi_ndwi_df)
weather_2023_processed, weather_2024_processed = process_weather_data(weather_2023_df, weather_2024_df)
mai_processed = process_mai_data(mai_df)

# -----------------------------
# FORTNIGHT CALCULATION FUNCTIONS
# -----------------------------
def get_fortnight(date_obj):
    """Get fortnight from date (1FN or 2FN)"""
    if date_obj.day <= 15:
        return f"1FN {date_obj.strftime('%B')}"
    else:
        return f"2FN {date_obj.strftime('%B')}"

def get_fortnight_dates(year, month, fortnight):
    """Get start and end dates for a given fortnight"""
    month_num = datetime.strptime(month, '%B').month
    
    if fortnight == '1FN':
        start_date = datetime(year, month_num, 1)
        end_date = datetime(year, month_num, 15)
    else:
        start_date = datetime(year, month_num, 16)
        # Get last day of month
        if month_num == 12:
            end_date = datetime(year, month_num, 31)
        else:
            end_date = datetime(year, month_num + 1, 1) - timedelta(days=1)
    
    return start_date, end_date

# -----------------------------
# WEATHER METRICS CALCULATIONS
# -----------------------------
def calculate_rainfall_metrics(weather_data, circle_code, start_date, end_date, comparison_year):
    """Calculate rainfall metrics for current and comparison year"""
    current_data = weather_data[weather_data['CIRNCODE'] == circle_code].copy()
    current_data = current_data[(current_data['Date(DD-MM-YYYY)'] >= start_date) & 
                               (current_data['Date(DD-MM-YYYY)'] <= end_date)]
    
    # Get comparison data (same dates but previous year)
    comp_start = start_date - timedelta(days=365)
    comp_end = end_date - timedelta(days=365)
    comp_data = weather_data[weather_data['CIRNCODE'] == circle_code].copy()
    comp_data = comp_data[(comp_data['Date(DD-MM-YYYY)'] >= comp_start) & 
                         (comp_data['Date(DD-MM-YYYY)'] <= comp_end)]
    
    current_rainfall = current_data['Rainfall'].sum() if not current_data.empty else 0
    comp_rainfall = comp_data['Rainfall'].sum() if not comp_data.empty else 0
    deviation = current_rainfall - comp_rainfall
    
    return current_rainfall, comp_rainfall, deviation

def calculate_rainy_days(weather_data, circle_code, start_date, end_date, comparison_year):
    """Calculate rainy days for current and comparison year"""
    current_data = weather_data[weather_data['CIRNCODE'] == circle_code].copy()
    current_data = current_data[(current_data['Date(DD-MM-YYYY)'] >= start_date) & 
                               (current_data['Date(DD-MM-YYYY)'] <= end_date)]
    
    comp_start = start_date - timedelta(days=365)
    comp_end = end_date - timedelta(days=365)
    comp_data = weather_data[weather_data['CIRNCODE'] == circle_code].copy()
    comp_data = comp_data[(comp_data['Date(DD-MM-YYYY)'] >= comp_start) & 
                         (comp_data['Date(DD-MM-YYYY)'] <= comp_end)]
    
    current_rainy_days = (current_data['Rainfall'] > 0).sum() if not current_data.empty else 0
    comp_rainy_days = (comp_data['Rainfall'] > 0).sum() if not comp_data.empty else 0
    
    return current_rainy_days, comp_rainy_days

def calculate_temperature_metrics(weather_data, circle_code, start_date, end_date, comparison_year, temp_type='Tmax'):
    """Calculate temperature metrics (average excluding 0 values)"""
    current_data = weather_data[weather_data['CIRNCODE'] == circle_code].copy()
    current_data = current_data[(current_data['Date(DD-MM-YYYY)'] >= start_date) & 
                               (current_data['Date(DD-MM-YYYY)'] <= end_date)]
    
    comp_start = start_date - timedelta(days=365)
    comp_end = end_date - timedelta(days=365)
    comp_data = weather_data[weather_data['CIRNCODE'] == circle_code].copy()
    comp_data = comp_data[(comp_data['Date(DD-MM-YYYY)'] >= comp_start) & 
                         (comp_data['Date(DD-MM-YYYY)'] <= comp_end)]
    
    # Calculate average excluding 0 values
    current_temp = current_data[temp_type].replace(0, np.nan).mean() if not current_data.empty else 0
    comp_temp = comp_data[temp_type].replace(0, np.nan).mean() if not comp_data.empty else 0
    deviation = current_temp - comp_temp if current_temp and comp_temp else 0
    
    return current_temp, comp_temp, deviation

def calculate_rh_metrics(weather_data, circle_code, start_date, end_date, comparison_year, rh_type='max_Rh'):
    """Calculate relative humidity metrics (average excluding 0 values)"""
    current_data = weather_data[weather_data['CIRNCODE'] == circle_code].copy()
    current_data = current_data[(current_data['Date(DD-MM-YYYY)'] >= start_date) & 
                               (current_data['Date(DD-MM-YYYY)'] <= end_date)]
    
    comp_start = start_date - timedelta(days=365)
    comp_end = end_date - timedelta(days=365)
    comp_data = weather_data[weather_data['CIRNCODE'] == circle_code].copy()
    comp_data = comp_data[(comp_data['Date(DD-MM-YYYY)'] >= comp_start) & 
                         (comp_data['Date(DD-MM-YYYY)'] <= comp_end)]
    
    # Calculate average excluding 0 values
    current_rh = current_data[rh_type].replace(0, np.nan).mean() if not current_data.empty else 0
    comp_rh = comp_data[rh_type].replace(0, np.nan).mean() if not comp_data.empty else 0
    deviation = current_rh - comp_rh if current_rh and comp_rh else 0
    
    return current_rh, comp_rh, deviation

# -----------------------------
# CHART FUNCTIONS FOR WEATHER METRICS TAB
# -----------------------------
def create_rainfall_chart(fortnightly_data, monthly_data):
    """Create clustered column chart for rainfall"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Fortnightly Rainfall', 'Monthly Rainfall'))
    
    # Fortnightly data
    if fortnightly_data:
        periods = [d['period'] for d in fortnightly_data]
        current_values = [d['current'] for d in fortnightly_data]
        comp_values = [d['comparison'] for d in fortnightly_data]
        
        fig.add_trace(go.Bar(name='Current Year', x=periods, y=current_values), 1, 1)
        fig.add_trace(go.Bar(name='Last Year', x=periods, y=comp_values), 1, 1)
    
    # Monthly data
    if monthly_data:
        periods = [d['period'] for d in monthly_data]
        current_values = [d['current'] for d in monthly_data]
        comp_values = [d['comparison'] for d in monthly_data]
        
        fig.add_trace(go.Bar(name='Current Year', x=periods, y=current_values, showlegend=False), 1, 2)
        fig.add_trace(go.Bar(name='Last Year', x=periods, y=comp_values, showlegend=False), 1, 2)
    
    fig.update_layout(barmode='group', title_text="Rainfall Comparison")
    return fig

def create_rainy_days_chart(fortnightly_data, monthly_data):
    """Create clustered column chart for rainy days"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Fortnightly Rainy Days', 'Monthly Rainy Days'))
    
    # Fortnightly data
    if fortnightly_data:
        periods = [d['period'] for d in fortnightly_data]
        current_values = [d['current'] for d in fortnightly_data]
        comp_values = [d['comparison'] for d in fortnightly_data]
        
        fig.add_trace(go.Bar(name='Current Year', x=periods, y=current_values), 1, 1)
        fig.add_trace(go.Bar(name='Last Year', x=periods, y=comp_values), 1, 1)
    
    # Monthly data
    if monthly_data:
        periods = [d['period'] for d in monthly_data]
        current_values = [d['current'] for d in monthly_data]
        comp_values = [d['comparison'] for d in monthly_data]
        
        fig.add_trace(go.Bar(name='Current Year', x=periods, y=current_values, showlegend=False), 1, 2)
        fig.add_trace(go.Bar(name='Last Year', x=periods, y=comp_values, showlegend=False), 1, 2)
    
    fig.update_layout(barmode='group', title_text="Rainy Days Comparison")
    return fig

def create_temperature_chart(fortnightly_data, monthly_data, temp_type='Tmax'):
    """Create clustered column chart for temperature"""
    title = f"{temp_type} Comparison"
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'Fortnightly {temp_type}', f'Monthly {temp_type}'))
    
    # Fortnightly data
    if fortnightly_data:
        periods = [d['period'] for d in fortnightly_data]
        current_values = [d['current'] for d in fortnightly_data]
        comp_values = [d['comparison'] for d in fortnightly_data]
        
        fig.add_trace(go.Bar(name='Current Year', x=periods, y=current_values), 1, 1)
        fig.add_trace(go.Bar(name='Last Year', x=periods, y=comp_values), 1, 1)
    
    # Monthly data
    if monthly_data:
        periods = [d['period'] for d in monthly_data]
        current_values = [d['current'] for d in monthly_data]
        comp_values = [d['comparison'] for d in monthly_data]
        
        fig.add_trace(go.Bar(name='Current Year', x=periods, y=current_values, showlegend=False), 1, 2)
        fig.add_trace(go.Bar(name='Last Year', x=periods, y=comp_values, showlegend=False), 1, 2)
    
    fig.update_layout(barmode='group', title_text=title)
    return fig

def create_rh_chart(fortnightly_data, monthly_data, rh_type='max_Rh'):
    """Create clustered column chart for relative humidity"""
    title = f"{rh_type} Comparison"
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'Fortnightly {rh_type}', f'Monthly {rh_type}'))
    
    # Fortnightly data
    if fortnightly_data:
        periods = [d['period'] for d in fortnightly_data]
        current_values = [d['current'] for d in fortnightly_data]
        comp_values = [d['comparison'] for d in fortnightly_data]
        
        fig.add_trace(go.Bar(name='Current Year', x=periods, y=current_values), 1, 1)
        fig.add_trace(go.Bar(name='Last Year', x=periods, y=comp_values), 1, 1)
    
    # Monthly data
    if monthly_data:
        periods = [d['period'] for d in monthly_data]
        current_values = [d['current'] for d in monthly_data]
        comp_values = [d['comparison'] for d in monthly_data]
        
        fig.add_trace(go.Bar(name='Current Year', x=periods, y=current_values, showlegend=False), 1, 2)
        fig.add_trace(go.Bar(name='Last Year', x=periods, y=comp_values, showlegend=False), 1, 2)
    
    fig.update_layout(barmode='group', title_text=title)
    return fig

# -----------------------------
# CHART FUNCTIONS FOR REMOTE SENSING INDICES TAB
# -----------------------------
def create_ndvi_line_chart(ndvi_data, circle_code, start_date, end_date):
    """Create line chart for NDVI comparison"""
    if ndvi_data is None or ndvi_data.empty:
        return None
    
    circle_data = ndvi_data[ndvi_data['CIRNCODE'] == circle_code].copy()
    circle_data = circle_data[(circle_data['Date'] >= start_date) & (circle_data['Date'] <= end_date)]
    
    if circle_data.empty:
        return None
    
    fig = go.Figure()
    
    # Current year data (2024)
    current_data = circle_data[circle_data['Year'] == 2024]
    if not current_data.empty:
        fig.add_trace(go.Scatter(
            x=current_data['Date'], 
            y=current_data['NDVI'],
            mode='lines+markers',
            name='2024',
            line=dict(color='blue', width=2)
        ))
    
    # Last year data (2023)
    last_year_data = circle_data[circle_data['Year'] == 2023]
    if not last_year_data.empty:
        fig.add_trace(go.Scatter(
            x=last_year_data['Date'], 
            y=last_year_data['NDVI'],
            mode='lines+markers',
            name='2023',
            line=dict(color='red', width=2)
        ))
    
    fig.update_layout(
        title='NDVI Comparison (2023 vs 2024)',
        xaxis_title='Date',
        yaxis_title='NDVI Value'
    )
    
    return fig

def create_ndwi_line_chart(ndvi_data, circle_code, start_date, end_date):
    """Create line chart for NDWI comparison"""
    if ndvi_data is None or ndvi_data.empty:
        return None
    
    circle_data = ndvi_data[ndvi_data['CIRNCODE'] == circle_code].copy()
    circle_data = circle_data[(circle_data['Date'] >= start_date) & (circle_data['Date'] <= end_date)]
    
    if circle_data.empty:
        return None
    
    fig = go.Figure()
    
    # Current year data (2024)
    current_data = circle_data[circle_data['Year'] == 2024]
    if not current_data.empty:
        fig.add_trace(go.Scatter(
            x=current_data['Date'], 
            y=current_data['NDWI'],
            mode='lines+markers',
            name='2024',
            line=dict(color='green', width=2)
        ))
    
    # Last year data (2023)
    last_year_data = circle_data[circle_data['Year'] == 2023]
    if not last_year_data.empty:
        fig.add_trace(go.Scatter(
            x=last_year_data['Date'], 
            y=last_year_data['NDWI'],
            mode='lines+markers',
            name='2023',
            line=dict(color='orange', width=2)
        ))
    
    fig.update_layout(
        title='NDWI Comparison (2023 vs 2024)',
        xaxis_title='Date',
        yaxis_title='NDWI Value'
    )
    
    return fig

def create_mai_chart(mai_data, circle_code, selected_months):
    """Create clustered column chart for MAI"""
    if mai_data is None or mai_data.empty:
        return None
    
    circle_data = mai_data[mai_data['CIRNCODE'] == circle_code].copy()
    circle_data = circle_data[circle_data['Month'].isin(selected_months)]
    
    if circle_data.empty:
        return None
    
    # Separate current year and last year data
    current_data = circle_data[circle_data['Year'] == 2024]
    last_year_data = circle_data[circle_data['Year'] == 2023]
    
    fig = go.Figure()
    
    # Current year
    if not current_data.empty:
        fig.add_trace(go.Bar(
            name='2024',
            x=current_data['Month'],
            y=current_data['MAI (%)'],
            marker_color='blue'
        ))
    
    # Last year
    if not last_year_data.empty:
        fig.add_trace(go.Bar(
            name='2023',
            x=last_year_data['Month'],
            y=last_year_data['MAI (%)'],
            marker_color='red'
        ))
    
    fig.update_layout(
        title='MAI Comparison (2023 vs 2024)',
        xaxis_title='Month',
        yaxis_title='MAI (%)',
        barmode='group'
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
    <span style='color: red; font-weight: 700;'>⚠️ Data Comparison Dashboard:</span>
    <span style='color: blue;'>
        Compare 2023 & 2024 data at selected level and date range
    </span>
</div>
""", unsafe_allow_html=True)

# Selection columns
col1, col2, col3 = st.columns(3)

with col1:
    # Level selection
    level = st.selectbox("Select Level", ["Circle", "Taluka", "District"])
    
    # Dynamic options based on selected level
    if level == "Circle" and ndvi_ndwi_processed is not None:
        circle_options = [""] + sorted(ndvi_ndwi_processed['CIRNAME'].dropna().unique().tolist())
        selected_circle = st.selectbox("Select Circle", circle_options)
        circle_code = ndvi_ndwi_processed[ndvi_ndwi_processed['CIRNAME'] == selected_circle]['CIRNCODE'].iloc[0] if selected_circle else ""
    elif level == "Taluka" and ndvi_ndwi_processed is not None:
        taluka_options = [""] + sorted(ndvi_ndwi_processed['THENAME'].dropna().unique().tolist())
        selected_taluka = st.selectbox("Select Taluka", taluka_options)
        # For demonstration, using first circle code in taluka
        if selected_taluka:
            circle_code = ndvi_ndwi_processed[ndvi_ndwi_processed['THENAME'] == selected_taluka]['CIRNCODE'].iloc[0]
        else:
            circle_code = ""
    elif level == "District" and ndvi_ndwi_processed is not None:
        district_options = [""] + sorted(ndvi_ndwi_processed['DTENAME'].dropna().unique().tolist())
        selected_district = st.selectbox("Select District", district_options)
        # For demonstration, using first circle code in district
        if selected_district:
            circle_code = ndvi_ndwi_processed[ndvi_ndwi_processed['DTENAME'] == selected_district]['CIRNCODE'].iloc[0]
        else:
            circle_code = ""
    else:
        circle_code = ""

with col2:
    # Date selection (replaced Start & End with Sowing Date & Current Date)
    sowing_date = st.date_input("Sowing Date", value=date(2024, 1, 1))
    current_date = st.date_input("Current Date", value=date.today())

with col3:
    # Analysis type
    analysis_type = st.selectbox("Analysis Type", ["Fortnightly", "Monthly"])
    
    # Month selection for MAI analysis
    if mai_processed is not None:
        available_months = sorted(mai_processed['Month'].dropna().unique())
        selected_months = st.multiselect("Select Months for MAI Analysis", available_months, default=available_months[:3])

# Generate button
generate = st.button("📊 Generate Analysis")

# -----------------------------
# MAIN ANALYSIS LOGIC
# -----------------------------
if generate and circle_code:
    sowing_date_str = sowing_date.strftime("%d/%m/%Y")
    current_date_str = current_date.strftime("%d/%m/%Y")
    
    st.info(f"📊 Analyzing data for {level}: {selected_circle if level == 'Circle' else selected_taluka if level == 'Taluka' else selected_district}")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🌤️ Weather Metrics", "📡 Remote Sensing Indices", "💾 Downloadable Data"])
    
    # TAB 1: WEATHER METRICS
    with tab1:
        st.header("🌤️ Weather Metrics Comparison")
        
        # Combine weather data for analysis
        all_weather_data = pd.concat([weather_2023_processed, weather_2024_processed], ignore_index=True)
        
        if analysis_type == "Fortnightly":
            # Calculate fortnightly periods between sowing and current date
            current = sowing_date
            fortnightly_periods = []
            
            while current <= current_date:
                fortnight = get_fortnight(current)
                fortnightly_periods.append(fortnight)
                # Move to next fortnight
                if current.day <= 15:
                    current = current.replace(day=16)
                else:
                    if current.month == 12:
                        current = current.replace(year=current.year + 1, month=1, day=1)
                    else:
                        current = current.replace(month=current.month + 1, day=1)
            
            # Remove duplicates
            fortnightly_periods = list(dict.fromkeys(fortnightly_periods))
            
            # Calculate metrics for each fortnight
            rainfall_fortnightly = []
            rainy_days_fortnightly = []
            tmax_fortnightly = []
            tmin_fortnightly = []
            rh_max_fortnightly = []
            rh_min_fortnightly = []
            
            for period in fortnightly_periods:
                fn, month = period.split(' ')
                start_date, end_date = get_fortnight_dates(2024, month, fn)
                
                # Rainfall
                current_rain, comp_rain, rain_dev = calculate_rainfall_metrics(
                    all_weather_data, circle_code, start_date, end_date, 2023)
                rainfall_fortnightly.append({
                    'period': period, 'current': current_rain, 
                    'comparison': comp_rain, 'deviation': rain_dev
                })
                
                # Rainy days
                current_rd, comp_rd = calculate_rainy_days(
                    all_weather_data, circle_code, start_date, end_date, 2023)
                rainy_days_fortnightly.append({
                    'period': period, 'current': current_rd, 'comparison': comp_rd
                })
                
                # Temperature Max
                current_tmax, comp_tmax, tmax_dev = calculate_temperature_metrics(
                    all_weather_data, circle_code, start_date, end_date, 2023, 'Tmax')
                tmax_fortnightly.append({
                    'period': period, 'current': current_tmax, 
                    'comparison': comp_tmax, 'deviation': tmax_dev
                })
                
                # Temperature Min
                current_tmin, comp_tmin, tmin_dev = calculate_temperature_metrics(
                    all_weather_data, circle_code, start_date, end_date, 2023, 'Tmin')
                tmin_fortnightly.append({
                    'period': period, 'current': current_tmin, 
                    'comparison': comp_tmin, 'deviation': tmin_dev
                })
                
                # RH Max
                current_rhmax, comp_rhmax, rhmax_dev = calculate_rh_metrics(
                    all_weather_data, circle_code, start_date, end_date, 2023, 'max_Rh')
                rh_max_fortnightly.append({
                    'period': period, 'current': current_rhmax, 
                    'comparison': comp_rhmax, 'deviation': rhmax_dev
                })
                
                # RH Min
                current_rhmin, comp_rhmin, rhmin_dev = calculate_rh_metrics(
                    all_weather_data, circle_code, start_date, end_date, 2023, 'min_Rh')
                rh_min_fortnightly.append({
                    'period': period, 'current': current_rhmin, 
                    'comparison': comp_rhmin, 'deviation': rhmin_dev
                })
            
            # Display charts
            st.subheader("I. Rainfall (Cumulative & Deviation)")
            rainfall_chart = create_rainfall_chart(rainfall_fortnightly, [])
            st.plotly_chart(rainfall_chart, use_container_width=True)
            
            st.subheader("II. Rainy Days")
            rainy_days_chart = create_rainy_days_chart(rainy_days_fortnightly, [])
            st.plotly_chart(rainy_days_chart, use_container_width=True)
            
            st.subheader("III. Temperature Max (Average & Deviation)")
            tmax_chart = create_temperature_chart(tmax_fortnightly, [], 'Tmax')
            st.plotly_chart(tmax_chart, use_container_width=True)
            
            st.subheader("IV. Temperature Min (Average & Deviation)")
            tmin_chart = create_temperature_chart(tmin_fortnightly, [], 'Tmin')
            st.plotly_chart(tmin_chart, use_container_width=True)
            
            st.subheader("V. Relative Humidity Max (Average & Deviation)")
            rhmax_chart = create_rh_chart(rh_max_fortnightly, [], 'max_Rh')
            st.plotly_chart(rhmax_chart, use_container_width=True)
            
            st.subheader("VI. Relative Humidity Min (Average & Deviation)")
            rhmin_chart = create_rh_chart(rh_min_fortnightly, [], 'min_Rh')
            st.plotly_chart(rhmin_chart, use_container_width=True)
        
        else:  # Monthly analysis
            # Similar implementation for monthly analysis
            st.info("Monthly analysis implementation would follow similar pattern")
    
    # TAB 2: REMOTE SENSING INDICES
    with tab2:
        st.header("📡 Remote Sensing Indices Comparison")
        
        # NDVI Line Chart
        st.subheader("I. NDVI Comparison")
        ndvi_chart = create_ndvi_line_chart(ndvi_ndwi_processed, circle_code, sowing_date, current_date)
        if ndvi_chart:
            st.plotly_chart(ndvi_chart, use_container_width=True)
        else:
            st.info("No NDVI data available for the selected parameters")
        
        # NDWI Line Chart
        st.subheader("II. NDWI Comparison")
        ndwi_chart = create_ndwi_line_chart(ndvi_ndwi_processed, circle_code, sowing_date, current_date)
        if ndwi_chart:
            st.plotly_chart(ndwi_chart, use_container_width=True)
        else:
            st.info("No NDWI data available for the selected parameters")
        
        # MAI Clustered Column Chart
        st.subheader("III. MAI Comparison")
        mai_chart = create_mai_chart(mai_processed, circle_code, selected_months)
        if mai_chart:
            st.plotly_chart(mai_chart, use_container_width=True)
        else:
            st.info("No MAI data available for the selected parameters")
    
    # TAB 3: DOWNLOADABLE DATA
    with tab3:
        st.header("💾 Downloadable Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Weather Data")
            if all_weather_data is not None:
                weather_csv = all_weather_data.to_csv(index=False)
                st.download_button(
                    label="Download Weather Data (CSV)",
                    data=weather_csv,
                    file_name="weather_data_2023_2024.csv",
                    mime="text/csv"
                )
            
            st.subheader("NDVI/NDWI Data")
            if ndvi_ndwi_processed is not None:
                ndvi_csv = ndvi_ndwi_processed.to_csv(index=False)
                st.download_button(
                    label="Download NDVI/NDWI Data (CSV)",
                    data=ndvi_csv,
                    file_name="ndvi_ndwi_data.csv",
                    mime="text/csv"
                )
        
        with col2:
            st.subheader("MAI Data")
            if mai_processed is not None:
                mai_csv = mai_processed.to_csv(index=False)
                st.download_button(
                    label="Download MAI Data (CSV)",
                    data=mai_csv,
                    file_name="mai_data.csv",
                    mime="text/csv"
                )
        
        # Display data previews
        st.subheader("Data Previews")
        preview_tab1, preview_tab2, preview_tab3 = st.tabs(["Weather", "NDVI/NDWI", "MAI"])
        
        with preview_tab1:
            if all_weather_data is not None:
                st.dataframe(all_weather_data.head(10), use_container_width=True)
        
        with preview_tab2:
            if ndvi_ndwi_processed is not None:
                st.dataframe(ndvi_ndwi_processed.head(10), use_container_width=True)
        
        with preview_tab3:
            if mai_processed is not None:
                st.dataframe(mai_processed.head(10), use_container_width=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    """
    <div class=footer; style='text-align: center; font-size: 16px; margin-top: 20px;'>
        💻 <b>Developed by:</b> Ashish Selokar <br>
        📧 For suggestions or queries, please email at:
        <a href="mailto:ashish111.selokar@gmail.com">ashish111.selokar@gmail.com</a> <br><br>
        <span style="font-size:15px; color:green;">
            🌾 Empowering Farmers with Data-Driven Insights 🌾
        </span><br>
        <span style="font-size:13px; color:gray;">
            Version 2.0 | 2023-2024 Data Comparison | Last Updated: Jan 2024
        </span>
    </div>
    """,
    unsafe_allow_html=True

)



