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
import os

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
        <div class="subtitle">2023-2024 Data Comparison Analysis</div>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA - UPDATED AS PER REQUIREMENTS
# -----------------------------
@st.cache_data
def load_data():
    try:
        # Load NDVI & NDWI data from GitHub
        ndvi_ndwi_url = "https://github.com/ASHISHSE/Generalized-Crop-Health/raw/main/1Maharashtra_NDVI_NDWI_old_circle_2023_2024_upload.xlsx"
        ndvi_ndwi_response = requests.get(ndvi_ndwi_url)
        ndvi_ndwi_df = pd.read_excel(BytesIO(ndvi_ndwi_response.content))
        
        # Load MAI data from GitHub
        mai_url = "https://github.com/ASHISHSE/Generalized-Crop-Health/raw/main/1Circlewise_Data_MAI_2023_24_upload.xlsx"
        mai_response = requests.get(mai_url)
        mai_df = pd.read_excel(BytesIO(mai_response.content))
        
        # Create comprehensive sample weather data
        dates_2023 = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        dates_2024 = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        
        # Sample circles for demonstration
        sample_circles = ['Ahemadpur', 'Bhingar', 'Akole', 'Sangrampur', 'Jalgaon']
        
        weather_data_2023 = []
        weather_data_2024 = []
        
        for circle in sample_circles:
            # 2023 data
            for date_val in dates_2023:
                weather_data_2023.append({
                    'District': 'Ahmednagar',
                    'Taluka': 'Ahemadpur' if circle == 'Ahemadpur' else 'Ahmednagar',
                    'Circle': circle,
                    'Date': date_val,
                    'Rainfall': np.random.uniform(0, 50) if np.random.random() > 0.7 else 0,
                    'Tmax': np.random.uniform(25, 35),
                    'Tmin': np.random.uniform(10, 20),
                    'max_Rh': np.random.uniform(60, 98),
                    'min_Rh': np.random.uniform(40, 80),
                    'Year': 2023
                })
            
            # 2024 data
            for date_val in dates_2024:
                weather_data_2024.append({
                    'District': 'Ahmednagar',
                    'Taluka': 'Ahemadpur' if circle == 'Ahemadpur' else 'Ahmednagar',
                    'Circle': circle,
                    'Date': date_val,
                    'Rainfall': np.random.uniform(0, 50) if np.random.random() > 0.7 else 0,
                    'Tmax': np.random.uniform(25, 35),
                    'Tmin': np.random.uniform(10, 20),
                    'max_Rh': np.random.uniform(60, 98),
                    'min_Rh': np.random.uniform(40, 80),
                    'Year': 2024
                })
        
        weather_2023_df = pd.DataFrame(weather_data_2023)
        weather_2024_df = pd.DataFrame(weather_data_2024)
        
        # Combine both years
        all_weather = pd.concat([weather_2023_df, weather_2024_df], ignore_index=True)
        
        return ndvi_ndwi_df, all_weather, mai_df
        
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        # Return empty dataframes if loading fails
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Load the data
ndvi_ndwi_df, weather_processed, mai_df = load_data()

# -----------------------------
# DATA PROCESSING FUNCTIONS
# -----------------------------
def process_ndvi_ndwi_data(df):
    """Process NDVI & NDWI data"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Make a copy to avoid modifying original
    df_processed = df.copy()
    
    # Convert Date column to datetime (handling both formats)
    if 'Date' in df_processed.columns:
        # Try different date formats
        df_processed['Date(DD-MM-YYYY)'] = pd.to_datetime(df_processed['Date(DD-MM-YYYY)'], errors='coerce')
    
    # If date conversion failed, try other column names
    if df_processed['Date(DD-MM-YYYY)'].isna().all() and 'Date(DD-MM-YYYY)' in df_processed.columns:
        df_processed['Date(DD-MM-YYYY)'] = pd.to_datetime(df_processed['Date(DD-MM-YYYY)'], format='%d-%m-%Y', errors='coerce')
    
    # Extract year and month for comparison
    df_processed['Year'] = df_processed['Date(DD-MM-YYYY)'].dt.year
    df_processed['Month'] = df_processed['Date(DD-MM-YYYY)'].dt.month_name()
    df_processed['Month_Num'] = df_processed['Date(DD-MM-YYYY)'].dt.month
    
    return df_processed

def process_mai_data(df):
    """Process MAI data"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Make a copy to avoid modifying original
    df_processed = df.copy()
    
    # Map month numbers to names if needed
    if 'Month' in df_processed.columns:
        if df_processed['Month'].dtype == 'int64':
            month_map = {
                1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
            }
            df_processed['Month'] = df_processed['Month'].map(month_map)
    
    return df_processed

# Process the data
ndvi_ndwi_processed = process_ndvi_ndwi_data(ndvi_ndwi_df)
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
# WEATHER METRICS CALCULATIONS - FIXED DATE HANDLING
# -----------------------------
def calculate_fortnightly_metrics(weather_data, selected_circle, sowing_date, current_date, metric_type):
    """Calculate fortnightly metrics for weather data"""
    if weather_data.empty:
        return []
    
    # Ensure dates are pandas Timestamp for proper comparison
    sowing_date_ts = pd.Timestamp(sowing_date)
    current_date_ts = pd.Timestamp(current_date)
    
    # Filter data for selected circle and date range
    circle_data = weather_data[
        (weather_data['Circle'] == selected_circle) & 
        (weather_data['Date'] >= sowing_date_ts) & 
        (weather_data['Date'] <= current_date_ts)
    ].copy()
    
    if circle_data.empty:
        return []
    
    # Generate fortnightly periods
    current = sowing_date_ts
    fortnightly_data = []
    
    while current <= current_date_ts:
        fortnight = get_fortnight(current)
        fn, month = fortnight.split(' ')
        start_date, end_date = get_fortnight_dates(current.year, month, fn)
        
        # Convert to pandas Timestamp
        start_date_ts = pd.Timestamp(start_date)
        end_date_ts = pd.Timestamp(end_date)
        
        # Skip if the period is outside our date range
        if start_date_ts > current_date_ts or end_date_ts < sowing_date_ts:
            # Move to next fortnight
            if current.day <= 15:
                current = current.replace(day=16)
            else:
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1, day=1)
                else:
                    current = current.replace(month=current.month + 1, day=1)
            continue
        
        # Adjust dates to be within selected range
        start_date_adj = max(start_date_ts, sowing_date_ts)
        end_date_adj = min(end_date_ts, current_date_ts)
        
        # Get data for this fortnight
        period_data = circle_data[
            (circle_data['Date'] >= start_date_adj) & 
            (circle_data['Date'] <= end_date_adj)
        ]
        
        if not period_data.empty:
            # Calculate metrics based on type
            if metric_type == 'rainfall':
                current_year = period_data[period_data['Year'] == 2024]['Rainfall'].sum()
                
                # Get comparison data for same period last year
                last_year_start = start_date_adj - pd.DateOffset(years=1)
                last_year_end = end_date_adj - pd.DateOffset(years=1)
                last_year_data = weather_data[
                    (weather_data['Circle'] == selected_circle) & 
                    (weather_data['Date'] >= last_year_start) & 
                    (weather_data['Date'] <= last_year_end)
                ]
                last_year = last_year_data['Rainfall'].sum() if not last_year_data.empty else 0
                deviation = current_year - last_year
                
                fortnightly_data.append({
                    'period': fortnight,
                    'current': round(current_year, 2),
                    'comparison': round(last_year, 2),
                    'deviation': round(deviation, 2)
                })
            
            elif metric_type == 'rainy_days':
                current_year = (period_data[period_data['Year'] == 2024]['Rainfall'] > 0).sum()
                
                # Get comparison data for same period last year
                last_year_start = start_date_adj - pd.DateOffset(years=1)
                last_year_end = end_date_adj - pd.DateOffset(years=1)
                last_year_data = weather_data[
                    (weather_data['Circle'] == selected_circle) & 
                    (weather_data['Date'] >= last_year_start) & 
                    (weather_data['Date'] <= last_year_end)
                ]
                last_year = (last_year_data['Rainfall'] > 0).sum() if not last_year_data.empty else 0
                
                fortnightly_data.append({
                    'period': fortnight,
                    'current': current_year,
                    'comparison': last_year
                })
            
            elif metric_type in ['Tmax', 'Tmin']:
                current_data = period_data[period_data['Year'] == 2024][metric_type].replace(0, np.nan)
                current_year = current_data.mean() if not current_data.empty else 0
                
                # Get comparison data for same period last year
                last_year_start = start_date_adj - pd.DateOffset(years=1)
                last_year_end = end_date_adj - pd.DateOffset(years=1)
                last_year_data = weather_data[
                    (weather_data['Circle'] == selected_circle) & 
                    (weather_data['Date'] >= last_year_start) & 
                    (weather_data['Date'] <= last_year_end)
                ]
                last_year_data_clean = last_year_data[metric_type].replace(0, np.nan)
                last_year = last_year_data_clean.mean() if not last_year_data_clean.empty else 0
                deviation = current_year - last_year if pd.notna(current_year) and pd.notna(last_year) else 0
                
                fortnightly_data.append({
                    'period': fortnight,
                    'current': round(current_year, 2),
                    'comparison': round(last_year, 2),
                    'deviation': round(deviation, 2)
                })
            
            elif metric_type in ['max_Rh', 'min_Rh']:
                current_data = period_data[period_data['Year'] == 2024][metric_type].replace(0, np.nan)
                current_year = current_data.mean() if not current_data.empty else 0
                
                # Get comparison data for same period last year
                last_year_start = start_date_adj - pd.DateOffset(years=1)
                last_year_end = end_date_adj - pd.DateOffset(years=1)
                last_year_data = weather_data[
                    (weather_data['Circle'] == selected_circle) & 
                    (weather_data['Date'] >= last_year_start) & 
                    (weather_data['Date'] <= last_year_end)
                ]
                last_year_data_clean = last_year_data[metric_type].replace(0, np.nan)
                last_year = last_year_data_clean.mean() if not last_year_data_clean.empty else 0
                deviation = current_year - last_year if pd.notna(current_year) and pd.notna(last_year) else 0
                
                fortnightly_data.append({
                    'period': fortnight,
                    'current': round(current_year, 2),
                    'comparison': round(last_year, 2),
                    'deviation': round(deviation, 2)
                })
        
        # Move to next fortnight
        if current.day <= 15:
            current = current.replace(day=16)
        else:
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1, day=1)
            else:
                current = current.replace(month=current.month + 1, day=1)
    
    return fortnightly_data

def calculate_monthly_metrics(weather_data, selected_circle, sowing_date, current_date, metric_type):
    """Calculate monthly metrics for weather data"""
    if weather_data.empty:
        return []
    
    # Ensure dates are pandas Timestamp for proper comparison
    sowing_date_ts = pd.Timestamp(sowing_date)
    current_date_ts = pd.Timestamp(current_date)
    
    # Filter data for selected circle and date range
    circle_data = weather_data[
        (weather_data['Circle'] == selected_circle) & 
        (weather_data['Date'] >= sowing_date_ts) & 
        (weather_data['Date'] <= current_date_ts)
    ].copy()
    
    if circle_data.empty:
        return []
    
    monthly_data = []
    
    # Get unique months in the date range
    months_in_range = pd.date_range(start=sowing_date_ts, end=current_date_ts, freq='M').strftime('%B').unique()
    
    for month in months_in_range:
        # Current year data
        current_data = circle_data[
            (circle_data['Month'] == month) & 
            (circle_data['Year'] == 2024)
        ]
        
        # Last year data (same month, previous year)
        last_year_data = weather_data[
            (weather_data['Circle'] == selected_circle) & 
            (weather_data['Month'] == month) & 
            (weather_data['Year'] == 2023)
        ]
        
        if metric_type == 'rainfall':
            current_year = current_data['Rainfall'].sum() if not current_data.empty else 0
            last_year = last_year_data['Rainfall'].sum() if not last_year_data.empty else 0
            deviation = current_year - last_year
            
            monthly_data.append({
                'period': month,
                'current': round(current_year, 2),
                'comparison': round(last_year, 2),
                'deviation': round(deviation, 2)
            })
        
        elif metric_type == 'rainy_days':
            current_year = (current_data['Rainfall'] > 0).sum() if not current_data.empty else 0
            last_year = (last_year_data['Rainfall'] > 0).sum() if not last_year_data.empty else 0
            
            monthly_data.append({
                'period': month,
                'current': current_year,
                'comparison': last_year
            })
        
        elif metric_type in ['Tmax', 'Tmin']:
            current_year_data = current_data[metric_type].replace(0, np.nan)
            current_year = current_year_data.mean() if not current_year_data.empty else 0
            
            last_year_data_clean = last_year_data[metric_type].replace(0, np.nan)
            last_year = last_year_data_clean.mean() if not last_year_data_clean.empty else 0
            deviation = current_year - last_year if pd.notna(current_year) and pd.notna(last_year) else 0
            
            monthly_data.append({
                'period': month,
                'current': round(current_year, 2),
                'comparison': round(last_year, 2),
                'deviation': round(deviation, 2)
            })
        
        elif metric_type in ['max_Rh', 'min_Rh']:
            current_year_data = current_data[metric_type].replace(0, np.nan)
            current_year = current_year_data.mean() if not current_year_data.empty else 0
            
            last_year_data_clean = last_year_data[metric_type].replace(0, np.nan)
            last_year = last_year_data_clean.mean() if not last_year_data_clean.empty else 0
            deviation = current_year - last_year if pd.notna(current_year) and pd.notna(last_year) else 0
            
            monthly_data.append({
                'period': month,
                'current': round(current_year, 2),
                'comparison': round(last_year, 2),
                'deviation': round(deviation, 2)
            })
    
    return monthly_data

# -----------------------------
# CHART FUNCTIONS FOR WEATHER METRICS TAB
# -----------------------------
def create_rainfall_chart(fortnightly_data, monthly_data):
    """Create clustered column chart for rainfall"""
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=('Fortnightly Rainfall (mm)', 'Monthly Rainfall (mm)'),
                       horizontal_spacing=0.15)
    
    # Fortnightly data
    if fortnightly_data:
        periods = [d['period'] for d in fortnightly_data]
        current_values = [d['current'] for d in fortnightly_data]
        comp_values = [d['comparison'] for d in fortnightly_data]
        
        fig.add_trace(go.Bar(name='2024', x=periods, y=current_values, marker_color='blue'), 1, 1)
        fig.add_trace(go.Bar(name='2023', x=periods, y=comp_values, marker_color='lightblue'), 1, 1)
    
    # Monthly data
    if monthly_data:
        periods = [d['period'] for d in monthly_data]
        current_values = [d['current'] for d in monthly_data]
        comp_values = [d['comparison'] for d in monthly_data]
        
        fig.add_trace(go.Bar(name='2024', x=periods, y=current_values, marker_color='green', showlegend=False), 1, 2)
        fig.add_trace(go.Bar(name='2023', x=periods, y=comp_values, marker_color='lightgreen', showlegend=False), 1, 2)
    
    fig.update_layout(barmode='group', title_text="Rainfall Comparison (Cumulative)", height=500)
    fig.update_xaxes(tickangle=45)
    return fig

def create_rainy_days_chart(fortnightly_data, monthly_data):
    """Create clustered column chart for rainy days"""
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=('Fortnightly Rainy Days', 'Monthly Rainy Days'),
                       horizontal_spacing=0.15)
    
    # Fortnightly data
    if fortnightly_data:
        periods = [d['period'] for d in fortnightly_data]
        current_values = [d['current'] for d in fortnightly_data]
        comp_values = [d['comparison'] for d in fortnightly_data]
        
        fig.add_trace(go.Bar(name='2024', x=periods, y=current_values, marker_color='orange'), 1, 1)
        fig.add_trace(go.Bar(name='2023', x=periods, y=comp_values, marker_color='yellow'), 1, 1)
    
    # Monthly data
    if monthly_data:
        periods = [d['period'] for d in monthly_data]
        current_values = [d['current'] for d in monthly_data]
        comp_values = [d['comparison'] for d in monthly_data]
        
        fig.add_trace(go.Bar(name='2024', x=periods, y=current_values, marker_color='red', showlegend=False), 1, 2)
        fig.add_trace(go.Bar(name='2023', x=periods, y=comp_values, marker_color='pink', showlegend=False), 1, 2)
    
    fig.update_layout(barmode='group', title_text="Rainy Days Comparison", height=500)
    fig.update_xaxes(tickangle=45)
    return fig

def create_temperature_chart(fortnightly_data, monthly_data, temp_type='Tmax'):
    """Create clustered column chart for temperature"""
    title_suffix = "Max" if temp_type == 'Tmax' else "Min"
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=(f'Fortnightly Temp {title_suffix} (°C)', f'Monthly Temp {title_suffix} (°C)'),
                       horizontal_spacing=0.15)
    
    # Fortnightly data
    if fortnightly_data:
        periods = [d['period'] for d in fortnightly_data]
        current_values = [d['current'] for d in fortnightly_data]
        comp_values = [d['comparison'] for d in fortnightly_data]
        
        fig.add_trace(go.Bar(name='2024', x=periods, y=current_values, marker_color='purple'), 1, 1)
        fig.add_trace(go.Bar(name='2023', x=periods, y=comp_values, marker_color='lavender'), 1, 1)
    
    # Monthly data
    if monthly_data:
        periods = [d['period'] for d in monthly_data]
        current_values = [d['current'] for d in monthly_data]
        comp_values = [d['comparison'] for d in monthly_data]
        
        fig.add_trace(go.Bar(name='2024', x=periods, y=current_values, marker_color='brown', showlegend=False), 1, 2)
        fig.add_trace(go.Bar(name='2023', x=periods, y=comp_values, marker_color='tan', showlegend=False), 1, 2)
    
    fig.update_layout(barmode='group', title_text=f"Temperature {title_suffix} Comparison (Average)", height=500)
    fig.update_xaxes(tickangle=45)
    return fig

def create_rh_chart(fortnightly_data, monthly_data, rh_type='max_Rh'):
    """Create clustered column chart for relative humidity"""
    title_suffix = "Max" if rh_type == 'max_Rh' else "Min"
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=(f'Fortnightly RH {title_suffix} (%)', f'Monthly RH {title_suffix} (%)'),
                       horizontal_spacing=0.15)
    
    # Fortnightly data
    if fortnightly_data:
        periods = [d['period'] for d in fortnightly_data]
        current_values = [d['current'] for d in fortnightly_data]
        comp_values = [d['comparison'] for d in fortnightly_data]
        
        fig.add_trace(go.Bar(name='2024', x=periods, y=current_values, marker_color='teal'), 1, 1)
        fig.add_trace(go.Bar(name='2023', x=periods, y=comp_values, marker_color='lightseagreen'), 1, 1)
    
    # Monthly data
    if monthly_data:
        periods = [d['period'] for d in monthly_data]
        current_values = [d['current'] for d in monthly_data]
        comp_values = [d['comparison'] for d in monthly_data]
        
        fig.add_trace(go.Bar(name='2024', x=periods, y=current_values, marker_color='darkblue', showlegend=False), 1, 2)
        fig.add_trace(go.Bar(name='2023', x=periods, y=comp_values, marker_color='lightblue', showlegend=False), 1, 2)
    
    fig.update_layout(barmode='group', title_text=f"Relative Humidity {title_suffix} Comparison (Average)", height=500)
    fig.update_xaxes(tickangle=45)
    return fig

# -----------------------------
# CHART FUNCTIONS FOR REMOTE SENSING INDICES TAB
# -----------------------------
def create_ndvi_line_chart(ndvi_data, selected_circle, sowing_date, current_date):
    """Create line chart for NDVI comparison"""
    if ndvi_data is None or ndvi_data.empty:
        return None
    
    # Ensure dates are pandas Timestamp
    sowing_date_ts = pd.Timestamp(sowing_date)
    current_date_ts = pd.Timestamp(current_date)
    
    circle_data = ndvi_data[ndvi_data['Circle'] == selected_circle].copy()
    circle_data = circle_data[
        (circle_data['Date'] >= sowing_date_ts) & 
        (circle_data['Date'] <= current_date_ts)
    ]
    
    if circle_data.empty:
        return None
    
    fig = go.Figure()
    
    # Current year data (2024)
    current_data = circle_data[circle_data['Year'] == 2024].sort_values('Date')
    if not current_data.empty:
        fig.add_trace(go.Scatter(
            x=current_data['Date'], 
            y=current_data['NDVI'],
            mode='lines+markers',
            name='2024',
            line=dict(color='blue', width=3),
            marker=dict(size=6)
        ))
    
    # Last year data (2023) - show same date range but previous year
    last_year_start = sowing_date_ts - pd.DateOffset(years=1)
    last_year_end = current_date_ts - pd.DateOffset(years=1)
    last_year_data = circle_data[
        (circle_data['Year'] == 2023) & 
        (circle_data['Date'] >= last_year_start) & 
        (circle_data['Date'] <= last_year_end)
    ].sort_values('Date')
    
    if not last_year_data.empty:
        fig.add_trace(go.Scatter(
            x=last_year_data['Date'], 
            y=last_year_data['NDVI'],
            mode='lines+markers',
            name='2023',
            line=dict(color='red', width=3),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title='NDVI Comparison (2023 vs 2024)',
        xaxis_title='Date',
        yaxis_title='NDVI Value',
        height=500,
        template="plotly_white"
    )
    
    return fig

def create_ndwi_line_chart(ndvi_data, selected_circle, sowing_date, current_date):
    """Create line chart for NDWI comparison"""
    if ndvi_data is None or ndvi_data.empty:
        return None
    
    # Ensure dates are pandas Timestamp
    sowing_date_ts = pd.Timestamp(sowing_date)
    current_date_ts = pd.Timestamp(current_date)
    
    circle_data = ndvi_data[ndvi_data['Circle'] == selected_circle].copy()
    circle_data = circle_data[
        (circle_data['Date'] >= sowing_date_ts) & 
        (circle_data['Date'] <= current_date_ts)
    ]
    
    if circle_data.empty:
        return None
    
    fig = go.Figure()
    
    # Current year data (2024)
    current_data = circle_data[circle_data['Year'] == 2024].sort_values('Date')
    if not current_data.empty:
        fig.add_trace(go.Scatter(
            x=current_data['Date'], 
            y=current_data['NDWI'],
            mode='lines+markers',
            name='2024',
            line=dict(color='green', width=3),
            marker=dict(size=6)
        ))
    
    # Last year data (2023)
    last_year_start = sowing_date_ts - pd.DateOffset(years=1)
    last_year_end = current_date_ts - pd.DateOffset(years=1)
    last_year_data = circle_data[
        (circle_data['Year'] == 2023) & 
        (circle_data['Date'] >= last_year_start) & 
        (circle_data['Date'] <= last_year_end)
    ].sort_values('Date')
    
    if not last_year_data.empty:
        fig.add_trace(go.Scatter(
            x=last_year_data['Date'], 
            y=last_year_data['NDWI'],
            mode='lines+markers',
            name='2023',
            line=dict(color='orange', width=3),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title='NDWI Comparison (2023 vs 2024)',
        xaxis_title='Date',
        yaxis_title='NDWI Value',
        height=500,
        template="plotly_white"
    )
    
    return fig

def create_mai_chart(mai_data, selected_circle, sowing_date, current_date):
    """Create clustered column chart for MAI"""
    if mai_data is None or mai_data.empty:
        return None
    
    # Ensure dates are pandas Timestamp
    sowing_date_ts = pd.Timestamp(sowing_date)
    current_date_ts = pd.Timestamp(current_date)
    
    circle_data = mai_data[mai_data['Circle'] == selected_circle].copy()
    
    # Get months between sowing and current date
    months_in_range = []
    current = sowing_date_ts.replace(day=1)
    while current <= current_date_ts:
        months_in_range.append(current.strftime('%B'))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    circle_data = circle_data[circle_data['Month'].isin(months_in_range)]
    
    if circle_data.empty:
        return None
    
    # Separate current year and last year data
    current_data = circle_data[circle_data['Year'] == 2024]
    last_year_data = circle_data[circle_data['Year'] == 2023]
    
    fig = go.Figure()
    
    # Current year
    if not current_data.empty:
        # Calculate average excluding 0 values
        current_avg_data = current_data[current_data['MAI (%)'] > 0]['MAI (%)']
        current_avg = current_avg_data.mean() if not current_avg_data.empty else 0
        
        fig.add_trace(go.Bar(
            name='2024',
            x=['MAI'],
            y=[current_avg],
            marker_color='blue',
            text=f'{current_avg:.1f}%',
            textposition='auto'
        ))
    
    # Last year
    if not last_year_data.empty:
        # Calculate average excluding 0 values
        last_year_avg_data = last_year_data[last_year_data['MAI (%)'] > 0]['MAI (%)']
        last_year_avg = last_year_avg_data.mean() if not last_year_avg_data.empty else 0
        
        fig.add_trace(go.Bar(
            name='2023',
            x=['MAI'],
            y=[last_year_avg],
            marker_color='red',
            text=f'{last_year_avg:.1f}%',
            textposition='auto'
        ))
    
    fig.update_layout(
        title='MAI Comparison - Monthly Average (Excluding 0 values)',
        xaxis_title='',
        yaxis_title='MAI (%)',
        barmode='group',
        height=500,
        showlegend=True
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
    <span style='color: red; font-weight: 700;'>📊 Data Comparison Dashboard:</span>
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
    
    # Dynamic options based on selected level and available data
    if not ndvi_ndwi_processed.empty:
        if level == "Circle":
            circle_options = [""] + sorted(ndvi_ndwi_processed['Circle'].dropna().unique().tolist())
            selected_circle = st.selectbox("Select Circle", circle_options)
        elif level == "Taluka":
            taluka_options = [""] + sorted(ndvi_ndwi_processed['Taluka'].dropna().unique().tolist())
            selected_taluka = st.selectbox("Select Taluka", taluka_options)
            # For demo, use first circle in selected taluka
            if selected_taluka:
                available_circles = ndvi_ndwi_processed[ndvi_ndwi_processed['Taluka'] == selected_taluka]['Circle'].unique()
                selected_circle = available_circles[0] if len(available_circles) > 0 else ""
            else:
                selected_circle = ""
        elif level == "District":
            district_options = [""] + sorted(ndvi_ndwi_processed['District'].dropna().unique().tolist())
            selected_district = st.selectbox("Select District", district_options)
            # For demo, use first circle in selected district
            if selected_district:
                available_circles = ndvi_ndwi_processed[ndvi_ndwi_processed['District'] == selected_district]['Circle'].unique()
                selected_circle = available_circles[0] if len(available_circles) > 0 else ""
            else:
                selected_circle = ""
    else:
        selected_circle = ""

with col2:
    # Date selection (Sowing Date & Current Date)
    sowing_date = st.date_input("Sowing Date", value=date(2024, 1, 1))
    current_date = st.date_input("Current Date", value=date.today())

with col3:
    # Analysis type selection
    analysis_type = st.selectbox("Analysis Type", ["Fortnightly", "Monthly"])
    
    # Display selected parameters
    if selected_circle:
        st.info(f"Selected: {level} - {selected_circle}")

# Generate button
generate = st.button("📊 Generate Analysis", type="primary")

# -----------------------------
# MAIN ANALYSIS LOGIC
# -----------------------------
if generate and selected_circle:
    st.info(f"📊 Analyzing data for {level}: {selected_circle}")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🌤️ Weather Metrics", "📡 Remote Sensing Indices", "💾 Downloadable Data"])
    
    # TAB 1: WEATHER METRICS
    with tab1:
        st.header("🌤️ Weather Metrics Comparison")
        
        if not weather_processed.empty:
            # Calculate metrics based on analysis type
            if analysis_type == "Fortnightly":
                # I. Rainfall
                st.subheader("I. Rainfall (Cumulative & Deviation)")
                rainfall_fortnightly = calculate_fortnightly_metrics(weather_processed, selected_circle, sowing_date, current_date, 'rainfall')
                rainfall_monthly = calculate_monthly_metrics(weather_processed, selected_circle, sowing_date, current_date, 'rainfall')
                rainfall_chart = create_rainfall_chart(rainfall_fortnightly, rainfall_monthly)
                st.plotly_chart(rainfall_chart, use_container_width=True)
                
                # Display data table
                if rainfall_fortnightly:
                    st.subheader("Rainfall Data Table")
                    rainfall_df = pd.DataFrame(rainfall_fortnightly)
                    st.dataframe(rainfall_df, use_container_width=True)
                
                # II. Rainy Days
                st.subheader("II. Rainy Days")
                rainy_days_fortnightly = calculate_fortnightly_metrics(weather_processed, selected_circle, sowing_date, current_date, 'rainy_days')
                rainy_days_monthly = calculate_monthly_metrics(weather_processed, selected_circle, sowing_date, current_date, 'rainy_days')
                rainy_days_chart = create_rainy_days_chart(rainy_days_fortnightly, rainy_days_monthly)
                st.plotly_chart(rainy_days_chart, use_container_width=True)
                
                # III. Temperature Max
                st.subheader("III. Temperature Max (Average & Deviation)")
                tmax_fortnightly = calculate_fortnightly_metrics(weather_processed, selected_circle, sowing_date, current_date, 'Tmax')
                tmax_monthly = calculate_monthly_metrics(weather_processed, selected_circle, sowing_date, current_date, 'Tmax')
                tmax_chart = create_temperature_chart(tmax_fortnightly, tmax_monthly, 'Tmax')
                st.plotly_chart(tmax_chart, use_container_width=True)
                
                # IV. Temperature Min
                st.subheader("IV. Temperature Min (Average & Deviation)")
                tmin_fortnightly = calculate_fortnightly_metrics(weather_processed, selected_circle, sowing_date, current_date, 'Tmin')
                tmin_monthly = calculate_monthly_metrics(weather_processed, selected_circle, sowing_date, current_date, 'Tmin')
                tmin_chart = create_temperature_chart(tmin_fortnightly, tmin_monthly, 'Tmin')
                st.plotly_chart(tmin_chart, use_container_width=True)
                
                # V. Relative Humidity Max
                st.subheader("V. Relative Humidity Max (Average & Deviation)")
                rhmax_fortnightly = calculate_fortnightly_metrics(weather_processed, selected_circle, sowing_date, current_date, 'max_Rh')
                rhmax_monthly = calculate_monthly_metrics(weather_processed, selected_circle, sowing_date, current_date, 'max_Rh')
                rhmax_chart = create_rh_chart(rhmax_fortnightly, rhmax_monthly, 'max_Rh')
                st.plotly_chart(rhmax_chart, use_container_width=True)
                
                # VI. Relative Humidity Min
                st.subheader("VI. Relative Humidity Min (Average & Deviation)")
                rhmin_fortnightly = calculate_fortnightly_metrics(weather_processed, selected_circle, sowing_date, current_date, 'min_Rh')
                rhmin_monthly = calculate_monthly_metrics(weather_processed, selected_circle, sowing_date, current_date, 'min_Rh')
                rhmin_chart = create_rh_chart(rhmin_fortnightly, rhmin_monthly, 'min_Rh')
                st.plotly_chart(rhmin_chart, use_container_width=True)
                
            else:  # Monthly analysis
                # I. Rainfall
                st.subheader("I. Rainfall (Cumulative & Deviation)")
                rainfall_monthly = calculate_monthly_metrics(weather_processed, selected_circle, sowing_date, current_date, 'rainfall')
                rainfall_chart = create_rainfall_chart([], rainfall_monthly)
                st.plotly_chart(rainfall_chart, use_container_width=True)
                
                # Display data table
                if rainfall_monthly:
                    st.subheader("Rainfall Data Table")
                    rainfall_df = pd.DataFrame(rainfall_monthly)
                    st.dataframe(rainfall_df, use_container_width=True)
                
                # II. Rainy Days
                st.subheader("II. Rainy Days")
                rainy_days_monthly = calculate_monthly_metrics(weather_processed, selected_circle, sowing_date, current_date, 'rainy_days')
                rainy_days_chart = create_rainy_days_chart([], rainy_days_monthly)
                st.plotly_chart(rainy_days_chart, use_container_width=True)
                
                # III. Temperature Max
                st.subheader("III. Temperature Max (Average & Deviation)")
                tmax_monthly = calculate_monthly_metrics(weather_processed, selected_circle, sowing_date, current_date, 'Tmax')
                tmax_chart = create_temperature_chart([], tmax_monthly, 'Tmax')
                st.plotly_chart(tmax_chart, use_container_width=True)
                
                # IV. Temperature Min
                st.subheader("IV. Temperature Min (Average & Deviation)")
                tmin_monthly = calculate_monthly_metrics(weather_processed, selected_circle, sowing_date, current_date, 'Tmin')
                tmin_chart = create_temperature_chart([], tmin_monthly, 'Tmin')
                st.plotly_chart(tmin_chart, use_container_width=True)
                
                # V. Relative Humidity Max
                st.subheader("V. Relative Humidity Max (Average & Deviation)")
                rhmax_monthly = calculate_monthly_metrics(weather_processed, selected_circle, sowing_date, current_date, 'max_Rh')
                rhmax_chart = create_rh_chart([], rhmax_monthly, 'max_Rh')
                st.plotly_chart(rhmax_chart, use_container_width=True)
                
                # VI. Relative Humidity Min
                st.subheader("VI. Relative Humidity Min (Average & Deviation)")
                rhmin_monthly = calculate_monthly_metrics(weather_processed, selected_circle, sowing_date, current_date, 'min_Rh')
                rhmin_chart = create_rh_chart([], rhmin_monthly, 'min_Rh')
                st.plotly_chart(rhmin_chart, use_container_width=True)
        else:
            st.warning("Weather data not available for the selected parameters")
    
    # TAB 2: REMOTE SENSING INDICES
    with tab2:
        st.header("📡 Remote Sensing Indices Comparison")
        
        # I. NDVI Line Chart
        st.subheader("I. NDVI Comparison")
        ndvi_chart = create_ndvi_line_chart(ndvi_ndwi_processed, selected_circle, sowing_date, current_date)
        if ndvi_chart:
            st.plotly_chart(ndvi_chart, use_container_width=True)
        else:
            st.info("No NDVI data available for the selected parameters")
        
        # II. NDWI Line Chart
        st.subheader("II. NDWI Comparison")
        ndwi_chart = create_ndwi_line_chart(ndvi_ndwi_processed, selected_circle, sowing_date, current_date)
        if ndwi_chart:
            st.plotly_chart(ndwi_chart, use_container_width=True)
        else:
            st.info("No NDWI data available for the selected parameters")
        
        # III. MAI Clustered Column Chart
        st.subheader("III. MAI Comparison")
        mai_chart = create_mai_chart(mai_processed, selected_circle, sowing_date, current_date)
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
            if not weather_processed.empty:
                weather_csv = weather_processed.to_csv(index=False)
                st.download_button(
                    label="Download Weather Data (CSV)",
                    data=weather_csv,
                    file_name="weather_data_2023_2024.csv",
                    mime="text/csv"
                )
            
            st.subheader("NDVI/NDWI Data")
            if not ndvi_ndwi_processed.empty:
                ndvi_csv = ndvi_ndwi_processed.to_csv(index=False)
                st.download_button(
                    label="Download NDVI/NDWI Data (CSV)",
                    data=ndvi_csv,
                    file_name="ndvi_ndwi_data.csv",
                    mime="text/csv"
                )
        
        with col2:
            st.subheader("MAI Data")
            if not mai_processed.empty:
                mai_csv = mai_processed.to_csv(index=False)
                st.download_button(
                    label="Download MAI Data (CSV)",
                    data=mai_csv,
                    file_name="mai_data.csv",
                    mime="text/csv"
                )
            
            st.subheader("Analysis Charts")
            st.info("Charts can be downloaded using the camera icon in each chart")
        
        # Data preview sections
        st.subheader("Data Previews")
        preview_tab1, preview_tab2, preview_tab3 = st.tabs(["Weather Data", "NDVI/NDWI Data", "MAI Data"])
        
        with preview_tab1:
            if not weather_processed.empty:
                st.dataframe(weather_processed.head(10), use_container_width=True)
            else:
                st.info("No weather data available")
        
        with preview_tab2:
            if not ndvi_ndwi_processed.empty:
                st.dataframe(ndvi_ndwi_processed.head(10), use_container_width=True)
            else:
                st.info("No NDVI/NDWI data available")
        
        with preview_tab3:
            if not mai_processed.empty:
                st.dataframe(mai_processed.head(10), use_container_width=True)
            else:
                st.info("No MAI data available")

else:
    if not selected_circle:
        st.warning("Please select a circle to generate analysis")

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


