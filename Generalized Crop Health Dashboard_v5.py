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

        /* Percentage text in graphs - Bold */
        .percentage-text {
            font-weight: bold !important;
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
        <div class="main-title">Generalized Crop & Weather Dashboard</div>
        <div class="subtitle">Empowering Farmers with Data-Driven Insights</div>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# WEATHER DATA OPTIONS (UPDATED - Only Uploaded File)
# -----------------------------
st.sidebar.info("🌤️ Weather Data Options")
st.sidebar.info("📊 Please upload weather data file for analysis")

uploaded_weather_file = st.sidebar.file_uploader(
    "Upload Weather Data (.xlsx)", 
    type=['xlsx'],
    help="File should contain sheets: 'Weather_data_23' and 'Weather_data_24'"
)

# -----------------------------
# LOAD DATA - UPDATED with .xlsx weather file
# -----------------------------
@st.cache_data
def load_data(_uploaded_file=None):
    # Updated URLs as per request
    ndvi_ndwi_url = "https://github.com/ASHISHSE/Generalized-Crop-Health/raw/main/1Maharashtra_NDVI_NDWI_old_circle_2023_2024_upload.xlsx"
    mai_url = "https://github.com/ASHISHSE/Generalized-Crop-Health/raw/main/1Circlewise_Data_MAI_2023_24_upload.xlsx"
    weather_url = "https://github.com/ASHISHSE/Generalized-Crop-Health/raw/main/Final_test_weather_data_2023_24_upload.xlsx"

    try:
        # Load NDVI & NDWI data
        ndvi_ndwi_res = requests.get(ndvi_ndwi_url, timeout=60)
        ndvi_ndwi_df = pd.read_excel(BytesIO(ndvi_ndwi_res.content))
        
        # Load MAI data
        mai_res = requests.get(mai_url, timeout=60)
        mai_df = pd.read_excel(BytesIO(mai_res.content))
        
        # Handle weather data based on uploaded file
        if _uploaded_file is not None:
            try:
                # Read uploaded .xlsx file
                weather_23_df = pd.read_excel(_uploaded_file, sheet_name='Weather_data_23')
                weather_24_df = pd.read_excel(_uploaded_file, sheet_name='Weather_data_24')
                
                # Combine weather data
                weather_df = pd.concat([weather_23_df, weather_24_df], ignore_index=True)
                
            except Exception as e:
                st.sidebar.error(f"Error reading weather file: {e}")
                st.sidebar.info("Using uploaded file data instead.")
                weather_df = create_sample_weather_data()
        else:
            # Use uploaded weather data only
            try:
                weather_res = requests.get(weather_url, timeout=60)
                weather_23_df = pd.read_excel(BytesIO(weather_res.content), sheet_name='Weather_data_23')
                weather_24_df = pd.read_excel(BytesIO(weather_res.content), sheet_name='Weather_data_24')
                weather_df = pd.concat([weather_23_df, weather_24_df], ignore_index=True)
            except:
                st.sidebar.info("Using uploaded weather data.")
                weather_df = create_sample_weather_data()
        
        # Process NDVI & NDWI data
        ndvi_ndwi_df["Date_dt"] = pd.to_datetime(ndvi_ndwi_df["Date(DD-MM-YYYY)"], format="%d-%m-%Y", errors="coerce")
        ndvi_ndwi_df = ndvi_ndwi_df.dropna(subset=["Date_dt"]).copy()
        
        # Process MAI data
        mai_df["Year"] = pd.to_numeric(mai_df["Year"], errors="coerce")
        mai_df["MAI (%)"] = pd.to_numeric(mai_df["MAI (%)"], errors="coerce")
        
        # Process Weather data
        weather_df["Date_dt"] = pd.to_datetime(weather_df["Date(DD-MM-YYYY)"], format="%d-%m-%Y", errors="coerce")
        weather_df = weather_df.dropna(subset=["Date_dt"]).copy()
        
        # Convert numeric columns for weather
        for col in ["Rainfall", "Tmax", "Tmin", "max_Rh", "min_Rh"]:
            if col in weather_df.columns:
                weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")
        
        # ADDED: Calculate Rainy_days column
        weather_df["Rainy_days"] = (weather_df["Rainfall"] > 0).astype(int)
        
        # Get unique locations from all datasets
        districts = sorted(weather_df["District"].dropna().unique().tolist())
        talukas = sorted(weather_df["Taluka"].dropna().unique().tolist())
        circles = sorted(weather_df["Circle"].dropna().unique().tolist())
        
        return weather_df, ndvi_ndwi_df, mai_df, districts, talukas, circles
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return sample data if main loading fails
        sample_weather = create_sample_weather_data()
        return sample_weather, pd.DataFrame(), pd.DataFrame(), [], [], []

def create_sample_weather_data():
    """Create sample weather data for demonstration"""
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    sample_data = []
    
    districts = ['Ahilyanagar', 'Pune', 'Nagpur']
    talukas = ['Akole', 'Ahilyanagar', 'Bhingar']
    circles = ['Akole', 'Nalegaon', 'Bhingar']
    
    for i, date_val in enumerate(dates):
        district = districts[i % len(districts)]
        taluka = talukas[i % len(talukas)]
        circle = circles[i % len(circles)]
        
        sample_data.append({
            'District': district,
            'Taluka': taluka,
            'Circle': circle,
            'Date(DD-MM-YYYY)': date_val.strftime('%d-%m-%Y'),
            'Rainfall': np.random.uniform(0, 50),
            'Tmax': np.random.uniform(25, 40),
            'Tmin': np.random.uniform(15, 25),
            'max_Rh': np.random.uniform(60, 95),
            'min_Rh': np.random.uniform(30, 70),
            'Date_dt': date_val
        })
    
    sample_df = pd.DataFrame(sample_data)
    # ADDED: Calculate Rainy_days column for sample data
    sample_df["Rainy_days"] = (sample_df["Rainfall"] > 0).astype(int)
    return sample_df

# Load data with the uploaded file
weather_df, ndvi_ndwi_df, mai_df, districts, talukas, circles = load_data(uploaded_weather_file)

# Show info message based on weather data source
if uploaded_weather_file is not None:
    st.sidebar.success("✅ Weather data loaded successfully!")
else:
    st.sidebar.warning("⚠️ Please upload weather data file for analysis")

# -----------------------------
# UPDATED DATA AGGREGATION FUNCTIONS
# -----------------------------
def aggregate_weather_data_by_level(data_df, district=None, taluka=None, circle=None):
    """
    Aggregate weather data based on selected level
    - District level: Show all talukas with calculations
    - Taluka level: Show all circles with calculations
    - Circle level: Show individual circle data
    """
    if circle and circle != "":
        # Circle level - return individual circle data
        filtered_data = data_df[data_df["Circle"] == circle].copy()
        return filtered_data, "Circle"
    
    elif taluka and taluka != "":
        # Taluka level - aggregate all circles under the taluka
        filtered_data = data_df[data_df["Taluka"] == taluka].copy()
        
        # Group by date and calculate metrics
        aggregated = filtered_data.groupby(['Date_dt', 'District', 'Taluka']).agg({
            'Rainfall': 'sum',  # Sum for rainfall
            'Rainy_days': 'sum',  # Sum for rainy days
            'Tmax': 'mean',  # Mean for Tmax
            'Tmin': 'mean',  # Mean for Tmin
            'max_Rh': 'mean',  # Mean for max_Rh
            'min_Rh': 'mean'   # Mean for min_Rh
        }).reset_index()
        
        # Create Circle column as "All Circles in [Taluka]"
        aggregated['Circle'] = f"All Circles in {taluka}"
        return aggregated, "Taluka"
    
    elif district and district != "":
        # District level - aggregate all talukas under the district
        filtered_data = data_df[data_df["District"] == district].copy()
        
        # Group by date and taluka, calculate metrics for each taluka
        aggregated = filtered_data.groupby(['Date_dt', 'District', 'Taluka']).agg({
            'Rainfall': 'sum',  # Sum for rainfall
            'Rainy_days': 'sum',  # Sum for rainy days
            'Tmax': 'mean',  # Mean for Tmax
            'Tmin': 'mean',  # Mean for Tmin
            'max_Rh': 'mean',  # Mean for max_Rh
            'min_Rh': 'mean'   # Mean for min_Rh
        }).reset_index()
        
        # Create Circle column as "All Circles in [Taluka]"
        aggregated['Circle'] = aggregated['Taluka'].apply(lambda x: f"All Circles in {x}")
        return aggregated, "District"
    
    else:
        # No selection - return original data
        return data_df, "All"

def aggregate_remote_sensing_data_by_level(data_df, metric_col, district=None, taluka=None, circle=None):
    """
    Aggregate remote sensing data (NDVI, NDWI, MAI) based on selected level
    - District level: Show all talukas with mean (excluding 0 values)
    - Taluka level: Show all circles with mean (excluding 0 values)
    - Circle level: Show individual circle data
    """
    if circle and circle != "":
        # Circle level - return individual circle data
        filtered_data = data_df[data_df["Circle"] == circle].copy()
        return filtered_data, "Circle"
    
    elif taluka and taluka != "":
        # Taluka level - aggregate all circles under the taluka
        filtered_data = data_df[data_df["Taluka"] == taluka].copy()
        
        # Exclude 0 values for mean calculation
        non_zero_data = filtered_data[filtered_data[metric_col] != 0]
        
        if not non_zero_data.empty:
            # Group by date and calculate mean (excluding 0 values)
            aggregated = non_zero_data.groupby(['Date_dt', 'District', 'Taluka']).agg({
                metric_col: 'mean'
            }).reset_index()
            
            # Create Circle column as "All Circles in [Taluka]"
            aggregated['Circle'] = f"All Circles in {taluka}"
            return aggregated, "Taluka"
        else:
            return filtered_data, "Taluka"
    
    elif district and district != "":
        # District level - aggregate all talukas under the district
        filtered_data = data_df[data_df["District"] == district].copy()
        
        # Exclude 0 values for mean calculation
        non_zero_data = filtered_data[filtered_data[metric_col] != 0]
        
        if not non_zero_data.empty:
            # Group by date and taluka, calculate mean for each taluka
            aggregated = non_zero_data.groupby(['Date_dt', 'District', 'Taluka']).agg({
                metric_col: 'mean'
            }).reset_index()
            
            # Create Circle column as "All Circles in [Taluka]"
            aggregated['Circle'] = aggregated['Taluka'].apply(lambda x: f"All Circles in {x}")
            return aggregated, "District"
        else:
            return filtered_data, "District"
    
    else:
        # No selection - return original data
        return data_df, "All"

# -----------------------------
# FORTNIGHT DEFINITION
# -----------------------------
def get_fortnight(date_obj):
    """Get fortnight from date (1st or 2nd)"""
    if date_obj.day <= 15:
        return f"1FN {date_obj.strftime('%b')}"
    else:
        return f"2FN {date_obj.strftime('%b')}"

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
# WEATHER METRICS CALCULATIONS - UPDATED
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
            if not non_zero_data.empty:
                fortnight_metrics = non_zero_data.groupby('Fortnight')[metric_col].mean()
            else:
                fortnight_metrics = pd.Series(dtype=float)
        elif agg_func == 'count':
            fortnight_metrics = year_data.groupby('Fortnight')[metric_col].sum()  # Use sum for Rainy_days
        
        # Round to 2 decimal places
        fortnight_metrics = fortnight_metrics.round(2)
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
            if not non_zero_data.empty:
                monthly_metrics = non_zero_data.groupby('Month')[metric_col].mean()
            else:
                monthly_metrics = pd.Series(dtype=float)
        elif agg_func == 'count':
            monthly_metrics = year_data.groupby('Month')[metric_col].sum()  # Use sum for Rainy_days
        
        # Round to 2 decimal places
        monthly_metrics = monthly_metrics.round(2)
        metrics[year] = monthly_metrics
    
    return metrics

# -----------------------------
# CHART CREATION FUNCTIONS - UPDATED
# -----------------------------
def create_fortnightly_comparison_chart(current_year_data, last_year_data, title, yaxis_title):
    """Create fortnightly comparison chart with better visualization"""
    # Get all possible periods
    all_periods = sorted(set(current_year_data.index) | set(last_year_data.index))
    
    current_values = [current_year_data.get(period, 0) for period in all_periods]
    last_values = [last_year_data.get(period, 0) for period in all_periods]
    
    fig = go.Figure()
    
    # UPDATED: Changed legend order and colors - Blue shades for better differentiation
    fig.add_trace(go.Scatter(
        name='2023',
        x=all_periods,
        y=last_values,
        mode='lines+markers',
        line=dict(color='#1f77b4', width=3, dash='dash'),  # Blue color for 2023
        marker=dict(size=8, symbol='square')
    ))
    
    fig.add_trace(go.Scatter(
        name='2024',
        x=all_periods,
        y=current_values,
        mode='lines+markers',
        line=dict(color='#3498db', width=3),  # Lighter blue for 2024
        marker=dict(size=8, symbol='circle')
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Fortnight",
        yaxis_title=yaxis_title,
        template='plotly_white',
        height=400,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_monthly_clustered_chart(current_year_data, last_year_data, title, yaxis_title):
    """Create monthly clustered column chart with side-by-side bars and deviation"""
    # Month order for proper sorting
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    
    all_months = sorted(set(current_year_data.index) | set(last_year_data.index), 
                       key=lambda x: month_order.index(x) if x in month_order else len(month_order))
    
    current_values = [current_year_data.get(month, 0) for month in all_months]
    last_values = [last_year_data.get(month, 0) for month in all_months]
    
    # Calculate deviations
    deviations = []
    deviation_labels = []
    for curr, last in zip(current_values, last_values):
        if last != 0:
            deviation = ((curr - last) / last) * 100
            deviations.append(round(deviation, 2))
            deviation_labels.append(f"{deviation:+.1f}%")
        else:
            deviations.append(0)
            deviation_labels.append("N/A")
    
    # Create subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # UPDATED: Changed legend order and colors - Blue shades
    fig.add_trace(go.Bar(
        name='2023',
        x=all_months,
        y=last_values,
        marker_color='#1f77b4',  # Darker blue for 2023
        text=[f"{x:.2f}" for x in last_values],
        textposition='auto',
        textfont=dict(weight='bold')
    ), secondary_y=False)
    
    fig.add_trace(go.Bar(
        name='2024',
        x=all_months,
        y=current_values,
        marker_color='#3498db',  # Lighter blue for 2024
        text=[f"{x:.2f}" for x in current_values],
        textposition='auto',
        textfont=dict(weight='bold')
    ), secondary_y=False)
    
    # Add deviation line (keeping deviation color as it is - red)
    fig.add_trace(go.Scatter(
        name='Deviation (%)',
        x=all_months,
        y=deviations,
        mode='lines+markers+text',
        line=dict(color='#ff6b6b', width=3),
        marker=dict(size=8, symbol='diamond'),
        text=deviation_labels,
        textposition="top center",
        textfont=dict(color='#d63031', size=10, weight='bold')
    ), secondary_y=True)
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Month",
        yaxis_title=yaxis_title,
        barmode='group',
        template='plotly_white',
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Deviation (%)", secondary_y=True)
    
    return fig

def create_fortnightly_deviation_chart(current_year_data, last_year_data, title, yaxis_title):
    """Create fortnightly deviation chart"""
    all_periods = sorted(set(current_year_data.index) | set(last_year_data.index))
    
    deviations = []
    deviation_labels = []
    for period in all_periods:
        current_val = current_year_data.get(period, 0)
        last_val = last_year_data.get(period, 0)
        
        if last_val != 0:
            deviation = ((current_val - last_val) / last_val) * 100
            deviations.append(round(deviation, 2))
            deviation_labels.append(f"{deviation:+.1f}%")
        else:
            deviations.append(0)
            deviation_labels.append("N/A")
    
    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in deviations]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=all_periods,
        y=deviations,
        marker_color=colors,
        text=deviation_labels,
        textposition='auto',
        textfont=dict(weight='bold'),
        name='Deviation'
    ))
    
    # Add zero reference line
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Fortnight",
        yaxis_title=yaxis_title,
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig

# -----------------------------
# UPDATED NDVI & NDWI CHART FUNCTIONS - FIXED with Date-Month format
# -----------------------------
def create_ndvi_comparison_chart(ndvi_df, district, taluka, circle, start_date, end_date):
    """Create NDVI comparison chart between 2023 and 2024 with Date-Month x-axis"""
    # Use the new aggregation function for remote sensing data
    filtered_df, level = aggregate_remote_sensing_data_by_level(ndvi_df, "NDVI", district, taluka, circle)
    
    # Filter for selected date range across both years
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    
    # Prepare data for both years
    data_2023 = pd.DataFrame()
    data_2024 = pd.DataFrame()
    
    for year in [2023, 2024]:
        year_start = start_date_dt.replace(year=year)
        year_end = end_date_dt.replace(year=year)
        
        # Filter data for this year and date range
        year_data = filtered_df[
            (filtered_df["Date_dt"] >= year_start) & 
            (filtered_df["Date_dt"] <= year_end) &
            (filtered_df["Date_dt"].dt.year == year)
        ].copy()
        
        if not year_data.empty:
            # Create Date-Month format (DD-MM) for x-axis
            year_data['Date_Month'] = year_data['Date_dt'].dt.strftime('%d-%m')
            year_data['DayOfYear'] = year_data['Date_dt'].dt.dayofyear
            
            # Sort by date
            year_data = year_data.sort_values('DayOfYear')
            
            if year == 2023:
                data_2023 = year_data
            else:
                data_2024 = year_data
    
    # Check if we have data for at least one year
    if data_2023.empty and data_2024.empty:
        return None
    
    # Create line chart
    fig = go.Figure()
    
    # Add 2023 trace
    if not data_2023.empty:
        fig.add_trace(go.Scatter(
            x=data_2023['Date_Month'],
            y=data_2023['NDVI'],
            mode='lines+markers',
            name='2023',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6),
            hovertemplate='%{x}<br>NDVI: %{y:.3f}<extra></extra>'
        ))
    
    # Add 2024 trace
    if not data_2024.empty:
        fig.add_trace(go.Scatter(
            x=data_2024['Date_Month'],
            y=data_2024['NDVI'],
            mode='lines+markers',
            name='2024',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=6),
            hovertemplate='%{x}<br>NDVI: %{y:.3f}<extra></extra>'
        ))
    
    # Determine level name for title
    if level == "Circle":
        level_name = circle
    elif level == "Taluka":
        level_name = taluka
    else:
        level_name = district
    
    fig.update_layout(
        title=dict(text=f"NDVI Trend Comparison: 2023 vs 2024 - {level}: {level_name}", x=0.5, xanchor='center'),
        xaxis_title="Date (DD-MM)",
        yaxis_title="NDVI",
        template='plotly_white',
        height=400,
        hovermode='x unified',
        xaxis=dict(
            tickangle=45,
            type='category'  # Ensure all dates are shown
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_ndwi_comparison_chart(ndwi_df, district, taluka, circle, start_date, end_date):
    """Create NDWI comparison chart between 2023 and 2024 with Date-Month x-axis"""
    # Use the new aggregation function for remote sensing data
    filtered_df, level = aggregate_remote_sensing_data_by_level(ndwi_df, "NDWI", district, taluka, circle)
    
    # Filter for selected date range across both years
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    
    # Prepare data for both years
    data_2023 = pd.DataFrame()
    data_2024 = pd.DataFrame()
    
    for year in [2023, 2024]:
        year_start = start_date_dt.replace(year=year)
        year_end = end_date_dt.replace(year=year)
        
        # Filter data for this year and date range
        year_data = filtered_df[
            (filtered_df["Date_dt"] >= year_start) & 
            (filtered_df["Date_dt"] <= year_end) &
            (filtered_df["Date_dt"].dt.year == year)
        ].copy()
        
        if not year_data.empty:
            # Create Date-Month format (DD-MM) for x-axis
            year_data['Date_Month'] = year_data['Date_dt'].dt.strftime('%d-%m')
            year_data['DayOfYear'] = year_data['Date_dt'].dt.dayofyear
            
            # Sort by date
            year_data = year_data.sort_values('DayOfYear')
            
            if year == 2023:
                data_2023 = year_data
            else:
                data_2024 = year_data
    
    # Check if we have data for at least one year
    if data_2023.empty and data_2024.empty:
        return None
    
    # Create line chart
    fig = go.Figure()
    
    # Add 2023 trace
    if not data_2023.empty:
        fig.add_trace(go.Scatter(
            x=data_2023['Date_Month'],
            y=data_2023['NDWI'],
            mode='lines+markers',
            name='2023',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6),
            hovertemplate='%{x}<br>NDWI: %{y:.3f}<extra></extra>'
        ))
    
    # Add 2024 trace
    if not data_2024.empty:
        fig.add_trace(go.Scatter(
            x=data_2024['Date_Month'],
            y=data_2024['NDWI'],
            mode='lines+markers',
            name='2024',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=6),
            hovertemplate='%{x}<br>NDWI: %{y:.3f}<extra></extra>'
        ))
    
    # Determine level name for title
    if level == "Circle":
        level_name = circle
    elif level == "Taluka":
        level_name = taluka
    else:
        level_name = district
    
    fig.update_layout(
        title=dict(text=f"NDWI Trend Comparison: 2023 vs 2024 - {level}: {level_name}", x=0.5, xanchor='center'),
        xaxis_title="Date (DD-MM)",
        yaxis_title="NDWI",
        template='plotly_white',
        height=400,
        hovermode='x unified',
        xaxis=dict(
            tickangle=45,
            type='category'  # Ensure all dates are shown
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_ndvi_ndwi_deviation_chart(ndvi_ndwi_df, district, taluka, circle, start_date, end_date):
    """Create deviation column chart for NDVI and NDWI"""
    # Use the new aggregation function for remote sensing data
    filtered_df, level = aggregate_remote_sensing_data_by_level(ndvi_ndwi_df, "NDVI", district, taluka, circle)
    
    # Filter for selected date range
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    
    # Calculate average values for both years
    avg_2023 = pd.DataFrame()
    avg_2024 = pd.DataFrame()
    
    for year in [2023, 2024]:
        year_start = start_date_dt.replace(year=year)
        year_end = end_date_dt.replace(year=year)
        
        year_data = filtered_df[
            (filtered_df["Date_dt"] >= year_start) & 
            (filtered_df["Date_dt"] <= year_end) &
            (filtered_df["Date_dt"].dt.year == year)
        ].copy()
        
        if not year_data.empty:
            # Calculate average NDVI and NDWI for the period
            ndvi_avg = year_data['NDVI'].mean()
            ndwi_avg = year_data['NDWI'].mean()
            
            if year == 2023:
                avg_2023 = pd.DataFrame({
                    'Metric': ['NDVI', 'NDWI'],
                    'Value': [ndvi_avg, ndwi_avg],
                    'Year': [2023, 2023]
                })
            else:
                avg_2024 = pd.DataFrame({
                    'Metric': ['NDVI', 'NDWI'],
                    'Value': [ndvi_avg, ndwi_avg],
                    'Year': [2024, 2024]
                })
    
    # Check if we have data for both years
    if avg_2023.empty or avg_2024.empty:
        return None
    
    # Calculate deviations
    deviations = []
    for metric in ['NDVI', 'NDWI']:
        val_2023 = avg_2023[avg_2023['Metric'] == metric]['Value'].iloc[0]
        val_2024 = avg_2024[avg_2024['Metric'] == metric]['Value'].iloc[0]
        
        if val_2023 != 0:
            deviation = ((val_2024 - val_2023) / val_2023) * 100
        else:
            deviation = 0
        deviations.append(round(deviation, 2))
    
    # Create deviation data
    deviation_data = pd.DataFrame({
        'Metric': ['NDVI', 'NDWI'],
        'Deviation (%)': deviations
    })
    
    # Create column chart
    fig = go.Figure()
    
    # Add bars with color based on positive/negative deviation
    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in deviations]
    
    fig.add_trace(go.Bar(
        x=deviation_data['Metric'],
        y=deviation_data['Deviation (%)'],
        marker_color=colors,
        text=[f"{x:+.1f}%" for x in deviations],
        textposition='auto',
        textfont=dict(size=14, weight='bold'),
        name='Deviation'
    ))
    
    # Add zero reference line
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    # Determine level name for title
    if level == "Circle":
        level_name = circle
    elif level == "Taluka":
        level_name = taluka
    else:
        level_name = district
    
    fig.update_layout(
        title=dict(text=f"NDVI & NDWI Deviation (2024 vs 2023) - {level}: {level_name}", x=0.5, xanchor='center'),
        xaxis_title="Metric",
        yaxis_title="Deviation (%)",
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig

def create_mai_monthly_comparison_chart(mai_df, district, taluka, circle):
    """Create monthly MAI comparison chart for 2023 vs 2024 with deviation"""
    # Use the new aggregation function for remote sensing data
    filtered_df, level = aggregate_remote_sensing_data_by_level(mai_df, "MAI (%)", district, taluka, circle)
    
    # Filter for 2023 and 2024
    filtered_df = filtered_df[filtered_df["Year"].isin([2023, 2024])]
    
    if filtered_df.empty:
        return None
    
    # Group by Year and Month
    monthly_mai = filtered_df.groupby(['Year', 'Month'])['MAI (%)'].mean().reset_index()
    monthly_mai['MAI (%)'] = monthly_mai['MAI (%)'].round(2)
    
    # Pivot to get 2023 and 2024 columns
    pivot_df = monthly_mai.pivot(index='Month', columns='Year', values='MAI (%)').reset_index()
    
    # Month order for proper sorting
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    
    # Sort months
    pivot_df['Month'] = pd.Categorical(pivot_df['Month'], categories=month_order, ordered=True)
    pivot_df = pivot_df.sort_values('Month')
    
    # Get values
    months = pivot_df['Month'].tolist()
    values_2023 = [pivot_df[2023].iloc[i] if 2023 in pivot_df.columns else 0 for i in range(len(months))]
    values_2024 = [pivot_df[2024].iloc[i] if 2024 in pivot_df.columns else 0 for i in range(len(months))]
    
    # Calculate deviations
    deviations = []
    deviation_labels = []
    for curr, last in zip(values_2024, values_2023):
        if last != 0:
            deviation = ((curr - last) / last) * 100
            deviations.append(round(deviation, 2))
            deviation_labels.append(f"{deviation:+.1f}%")
        else:
            deviations.append(0)
            deviation_labels.append("N/A")
    
    # Create subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # UPDATED: Changed legend order and colors - Blue shades
    fig.add_trace(go.Bar(
        name='2023',
        x=months,
        y=values_2023,
        marker_color='#1f77b4',  # Darker blue for 2023
        text=[f"{x:.2f}%" for x in values_2023],
        textposition='auto',
        textfont=dict(weight='bold')
    ), secondary_y=False)
    
    fig.add_trace(go.Bar(
        name='2024',
        x=months,
        y=values_2024,
        marker_color='#3498db',  # Lighter blue for 2024
        text=[f"{x:.2f}%" for x in values_2024],
        textposition='auto',
        textfont=dict(weight='bold')
    ), secondary_y=False)
    
    # Add deviation line (keeping deviation color as it is - red)
    fig.add_trace(go.Scatter(
        name='Deviation (%)',
        x=months,
        y=deviations,
        mode='lines+markers+text',
        line=dict(color='#ff6b6b', width=3),
        marker=dict(size=8, symbol='diamond'),
        text=deviation_labels,
        textposition="top center",
        textfont=dict(color='#d63031', size=10, weight='bold')
    ), secondary_y=True)
    
    # Determine level name for title
    if level == "Circle":
        level_name = circle
    elif level == "Taluka":
        level_name = taluka
    else:
        level_name = district
    
    fig.update_layout(
        title=dict(text=f"MAI Monthly Comparison: 2023 vs 2024 with Deviation - {level}: {level_name}", x=0.5, xanchor='center'),
        xaxis_title="Month",
        yaxis_title="MAI (%)",
        barmode='group',
        template='plotly_white',
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Deviation (%)", secondary_y=True)
    
    return fig

# -----------------------------
# DOWNLOAD FUNCTIONS - FIXED
# -----------------------------
def download_chart_as_image(fig, filename):
    """Download Plotly chart as PNG image - FIXED version"""
    try:
        # Check if kaleido is available, if not provide installation instructions
        try:
            import kaleido
        except ImportError:
            st.warning(f"Kaleido package not found. Please install it using: pip install kaleido")
            return
            
        img_bytes = fig.to_image(format="png", width=1200, height=600)
        st.download_button(
            label=f"📥 Download {filename} as PNG",
            data=img_bytes,
            file_name=f"{filename}.png",
            mime="image/png",
            key=f"download_{filename}_{np.random.randint(1000)}"
        )
    except Exception as e:
        st.warning(f"Could not download {filename}: {str(e)}")
        st.info("Please make sure Kaleido is installed: pip install kaleido")

def download_data_as_csv(data_df, filename):
    """Download DataFrame as CSV"""
    csv = data_df.to_csv(index=False)
    st.download_button(
        label=f"📥 Download {filename} as CSV",
        data=csv,
        file_name=f"{filename}.csv",
        mime="text/csv",
        key=f"download_{filename}_{np.random.randint(1000)}"
    )

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

# MODIFIED LAYOUT: Dynamic location selection based on requirements
col1, col2 = st.columns(2)

with col1:
    district = st.selectbox("District *", [""] + districts)
    
    # Taluka selection - show all talukas if no district selected
    if district:
        taluka_options = [""] + sorted(weather_df[weather_df["District"] == district]["Taluka"].dropna().unique().tolist())
    else:
        # If no district selected, show all talukas
        taluka_options = [""] + talukas
    taluka = st.selectbox("Taluka", taluka_options)

with col2:
    # Circle selection logic
    if taluka and taluka != "":
        # If taluka is selected, show circles from that taluka
        circle_options = [""] + sorted(weather_df[weather_df["Taluka"] == taluka]["Circle"].dropna().unique().tolist())
    elif district and district != "" and (not taluka or taluka == ""):
        # If only district is selected, show all circles from that district
        circle_options = [""] + sorted(weather_df[weather_df["District"] == district]["Circle"].dropna().unique().tolist())
    else:
        # If nothing selected, show all circles
        circle_options = [""] + circles
    circle = st.selectbox("Circle", circle_options)

# Start Date and End Date in separate rows
col3, col4 = st.columns(2)
with col3:
    sowing_date = st.date_input("Start Date (Sowing Date) *", value=date.today() - timedelta(days=30), format="DD/MM/YYYY")

with col4:
    current_date = st.date_input("End Date (Current Date) *", value=date.today(), format="DD/MM/YYYY")

# Generate button
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
        
        # Use the new aggregation functions
        aggregated_weather, weather_level = aggregate_weather_data_by_level(weather_df, district, taluka, circle)
        
        st.info(f"📊 Calculating metrics for **{weather_level}**: {district}")

        # Set years to 2023 and 2024 as requested
        current_year = 2024
        last_year = 2023

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["🌤️ Weather Metrics", "📡 Remote Sensing Indices", "💾 Download Data"])

        # TAB 1: WEATHER METRICS
        with tab1:
            st.header(f"🌤️ Weather Metrics - {weather_level}: {district}")
            
            if not aggregated_weather.empty:
                # I. Rainfall Analysis
                st.subheader("I. Rainfall Analysis")
                
                # Fortnightly Analysis
                st.markdown("##### Fortnightly Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Fortnightly Rainfall Comparison
                    fortnightly_rainfall = calculate_fortnightly_metrics(
                        aggregated_weather, current_year, last_year, "Rainfall", "sum"
                    )
                    fig_rainfall_fortnight = create_fortnightly_comparison_chart(
                        fortnightly_rainfall[current_year], 
                        fortnightly_rainfall[last_year],
                        "Rainfall - Fortnightly Comparison (2023 vs 2024)",
                        "Rainfall (mm)"
                    )
                    st.plotly_chart(fig_rainfall_fortnight, use_container_width=True)
                
                with col2:
                    # Fortnightly Rainfall Deviation
                    fig_rainfall_dev_fortnight = create_fortnightly_deviation_chart(
                        fortnightly_rainfall[current_year], 
                        fortnightly_rainfall[last_year],
                        "Rainfall Deviation - Fortnightly (%)",
                        "Deviation (%)"
                    )
                    st.plotly_chart(fig_rainfall_dev_fortnight, use_container_width=True)
                
                # Monthly Analysis
                st.markdown("##### Monthly Analysis")
                monthly_rainfall = calculate_monthly_metrics(
                    aggregated_weather, current_year, last_year, "Rainfall", "sum"
                )
                fig_rainfall_monthly = create_monthly_clustered_chart(
                    monthly_rainfall[current_year], 
                    monthly_rainfall[last_year],
                    "Rainfall - Monthly Comparison with Deviation (2023 vs 2024)",
                    "Rainfall (mm)"
                )
                st.plotly_chart(fig_rainfall_monthly, use_container_width=True)

                # II. Rainy Days Analysis
                st.subheader("II. Rainy Days Analysis")
                
                # Fortnightly Analysis
                st.markdown("##### Fortnightly Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Fortnightly Rainy Days
                    fortnightly_rainy_days = calculate_fortnightly_metrics(
                        aggregated_weather, current_year, last_year, "Rainy_days", "count"
                    )
                    fig_rainy_fortnight = create_fortnightly_comparison_chart(
                        fortnightly_rainy_days[current_year], 
                        fortnightly_rainy_days[last_year],
                        "Rainy Days - Fortnightly Comparison (2023 vs 2024)",
                        "Number of Rainy Days"
                    )
                    st.plotly_chart(fig_rainy_fortnight, use_container_width=True)
                
                with col2:
                    # Fortnightly Rainy Days Deviation
                    fig_rainy_dev_fortnight = create_fortnightly_deviation_chart(
                        fortnightly_rainy_days[current_year], 
                        fortnightly_rainy_days[last_year],
                        "Rainy Days Deviation - Fortnightly (%)",
                        "Deviation (%)"
                    )
                    st.plotly_chart(fig_rainy_dev_fortnight, use_container_width=True)
                
                # Monthly Analysis
                st.markdown("##### Monthly Analysis")
                monthly_rainy_days = calculate_monthly_metrics(
                    aggregated_weather, current_year, last_year, "Rainy_days", "count"
                )
                fig_rainy_monthly = create_monthly_clustered_chart(
                    monthly_rainy_days[current_year], 
                    monthly_rainy_days[last_year],
                    "Rainy Days - Monthly Comparison with Deviation (2023 vs 2024)",
                    "Number of Rainy Days"
                )
                st.plotly_chart(fig_rainy_monthly, use_container_width=True)

                # III. Temperature Analysis
                st.subheader("III. Temperature Analysis")
                
                # Maximum Temperature
                st.markdown("##### Maximum Temperature")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Fortnightly Tmax
                    fortnightly_tmax = calculate_fortnightly_metrics(
                        aggregated_weather, current_year, last_year, "Tmax", "mean"
                    )
                    fig_tmax_fortnight = create_fortnightly_comparison_chart(
                        fortnightly_tmax[current_year], 
                        fortnightly_tmax[last_year],
                        "Max Temperature - Fortnightly Average (2023 vs 2024)",
                        "Temperature (°C)"
                    )
                    st.plotly_chart(fig_tmax_fortnight, use_container_width=True)
                
                with col2:
                    # Fortnightly Tmax Deviation
                    fig_tmax_dev_fortnight = create_fortnightly_deviation_chart(
                        fortnightly_tmax[current_year], 
                        fortnightly_tmax[last_year],
                        "Max Temperature Deviation - Fortnightly",
                        "Deviation (°C)"
                    )
                    st.plotly_chart(fig_tmax_dev_fortnight, use_container_width=True)
                
                # Monthly Tmax
                monthly_tmax = calculate_monthly_metrics(
                    aggregated_weather, current_year, last_year, "Tmax", "mean"
                )
                fig_tmax_monthly = create_monthly_clustered_chart(
                    monthly_tmax[current_year], 
                    monthly_tmax[last_year],
                    "Max Temperature - Monthly Average with Deviation (2023 vs 2024)",
                    "Temperature (°C)"
                )
                st.plotly_chart(fig_tmax_monthly, use_container_width=True)

                # Minimum Temperature
                st.markdown("##### Minimum Temperature")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Fortnightly Tmin
                    fortnightly_tmin = calculate_fortnightly_metrics(
                        aggregated_weather, current_year, last_year, "Tmin", "mean"
                    )
                    fig_tmin_fortnight = create_fortnightly_comparison_chart(
                        fortnightly_tmin[current_year], 
                        fortnightly_tmin[last_year],
                        "Min Temperature - Fortnightly Average (2023 vs 2024)",
                        "Temperature (°C)"
                    )
                    st.plotly_chart(fig_tmin_fortnight, use_container_width=True)
                
                with col2:
                    # Fortnightly Tmin Deviation
                    fig_tmin_dev_fortnight = create_fortnightly_deviation_chart(
                        fortnightly_tmin[current_year], 
                        fortnightly_tmin[last_year],
                        "Min Temperature Deviation - Fortnightly",
                        "Deviation (°C)"
                    )
                    st.plotly_chart(fig_tmin_dev_fortnight, use_container_width=True)
                
                # Monthly Tmin
                monthly_tmin = calculate_monthly_metrics(
                    aggregated_weather, current_year, last_year, "Tmin", "mean"
                )
                fig_tmin_monthly = create_monthly_clustered_chart(
                    monthly_tmin[current_year], 
                    monthly_tmin[last_year],
                    "Min Temperature - Monthly Average with Deviation (2023 vs 2024)",
                    "Temperature (°C)"
                )
                st.plotly_chart(fig_tmin_monthly, use_container_width=True)

                # IV. Relative Humidity Analysis
                st.subheader("IV. Relative Humidity Analysis")
                
                # Maximum RH
                st.markdown("##### Maximum Relative Humidity")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Fortnightly RH max
                    fortnightly_rh_max = calculate_fortnightly_metrics(
                        aggregated_weather, current_year, last_year, "max_Rh", "mean"
                    )
                    fig_rh_max_fortnight = create_fortnightly_comparison_chart(
                        fortnightly_rh_max[current_year], 
                        fortnightly_rh_max[last_year],
                        "Max RH - Fortnightly Average (2023 vs 2024)",
                        "Relative Humidity (%)"
                    )
                    st.plotly_chart(fig_rh_max_fortnight, use_container_width=True)
                
                with col2:
                    # Fortnightly RH Max Deviation
                    fig_rh_max_dev_fortnight = create_fortnightly_deviation_chart(
                        fortnightly_rh_max[current_year], 
                        fortnightly_rh_max[last_year],
                        "Max RH Deviation - Fortnightly",
                        "Deviation (%)"
                    )
                    st.plotly_chart(fig_rh_max_dev_fortnight, use_container_width=True)
                
                # Monthly RH max
                monthly_rh_max = calculate_monthly_metrics(
                    aggregated_weather, current_year, last_year, "max_Rh", "mean"
                )
                fig_rh_max_monthly = create_monthly_clustered_chart(
                    monthly_rh_max[current_year], 
                    monthly_rh_max[last_year],
                    "Max RH - Monthly Average with Deviation (2023 vs 2024)",
                    "Relative Humidity (%)"
                )
                st.plotly_chart(fig_rh_max_monthly, use_container_width=True)

                # Minimum RH
                st.markdown("##### Minimum Relative Humidity")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Fortnightly RH min
                    fortnightly_rh_min = calculate_fortnightly_metrics(
                        aggregated_weather, current_year, last_year, "min_Rh", "mean"
                    )
                    fig_rh_min_fortnight = create_fortnightly_comparison_chart(
                        fortnightly_rh_min[current_year], 
                        fortnightly_rh_min[last_year],
                        "Min RH - Fortnightly Average (2023 vs 2024)",
                        "Relative Humidity (%)"
                    )
                    st.plotly_chart(fig_rh_min_fortnight, use_container_width=True)
                
                with col2:
                    # Fortnightly RH Min Deviation
                    fig_rh_min_dev_fortnight = create_fortnightly_deviation_chart(
                        fortnightly_rh_min[current_year], 
                        fortnightly_rh_min[last_year],
                        "Min RH Deviation - Fortnightly",
                        "Deviation (%)"
                    )
                    st.plotly_chart(fig_rh_min_dev_fortnight, use_container_width=True)
                
                # Monthly RH min
                monthly_rh_min = calculate_monthly_metrics(
                    aggregated_weather, current_year, last_year, "min_Rh", "mean"
                )
                fig_rh_min_monthly = create_monthly_clustered_chart(
                    monthly_rh_min[current_year], 
                    monthly_rh_min[last_year],
                    "Min RH - Monthly Average with Deviation (2023 vs 2024)",
                    "Relative Humidity (%)"
                )
                st.plotly_chart(fig_rh_min_monthly, use_container_width=True)
                
            else:
                st.info("No weather data available for the selected location and date range.")

        # TAB 2: REMOTE SENSING INDICES (UPDATED)
        with tab2:
            st.header(f"📡 Remote Sensing Indices - {weather_level}: {district}")
            
            # I. NDVI Trend Analysis
            st.subheader("I. NDVI Trend Analysis")
            ndvi_comparison_fig = create_ndvi_comparison_chart(
                ndvi_ndwi_df, district, taluka, circle, sowing_date, current_date
            )
            if ndvi_comparison_fig:
                st.plotly_chart(ndvi_comparison_fig, use_container_width=True)
            else:
                st.info("No NDVI data available for the selected parameters.")
            
            # II. NDWI Trend Analysis
            st.subheader("II. NDWI Trend Analysis")
            ndwi_comparison_fig = create_ndwi_comparison_chart(
                ndvi_ndwi_df, district, taluka, circle, sowing_date, current_date
            )
            if ndwi_comparison_fig:
                st.plotly_chart(ndwi_comparison_fig, use_container_width=True)
            else:
                st.info("No NDWI data available for the selected parameters.")
            
            # III. Deviation Analysis
            st.subheader("III. Deviation Analysis")
            ndvi_ndwi_dev_fig = create_ndvi_ndwi_deviation_chart(
                ndvi_ndwi_df, district, taluka, circle, sowing_date, current_date
            )
            if ndvi_ndwi_dev_fig:
                st.plotly_chart(ndvi_ndwi_dev_fig, use_container_width=True)
            else:
                st.info("No deviation data available for the selected parameters.")
            
            # IV. MAI Analysis (UPDATED - Monthly Comparison with Deviation)
            st.subheader("IV. MAI Analysis")
            mai_monthly_fig = create_mai_monthly_comparison_chart(
                mai_df, district, taluka, circle
            )
            if mai_monthly_fig:
                st.plotly_chart(mai_monthly_fig, use_container_width=True)
            else:
                st.info("No MAI data available for the selected parameters.")

        # TAB 3: DOWNLOAD DATA (FIXED - Removed download charts section)
        with tab3:
            st.header(f"💾 Download Data - {weather_level}: {district}")
            
            # Download Data Section only
            st.subheader("Download Data Tables")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Weather Data
                if not aggregated_weather.empty:
                    download_data_as_csv(aggregated_weather, f"Weather_Data_{weather_level}_{district}")
                else:
                    st.write("No weather data available")
            
            with col2:
                # NDVI & NDWI Data
                filtered_ndvi_ndwi, ndvi_level = aggregate_remote_sensing_data_by_level(ndvi_ndwi_df, "NDVI", district, taluka, circle)
                
                if not filtered_ndvi_ndwi.empty:
                    download_data_as_csv(filtered_ndvi_ndwi, f"NDVI_NDWI_Data_{ndvi_level}_{district}")
                else:
                    st.write("No NDVI/NDWI data available")
            
            with col3:
                # MAI Data
                filtered_mai, mai_level = aggregate_remote_sensing_data_by_level(mai_df, "MAI (%)", district, taluka, circle)
                
                if not filtered_mai.empty:
                    download_data_as_csv(filtered_mai, f"MAI_Data_{mai_level}_{district}")
                else:
                    st.write("No MAI data available")

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
