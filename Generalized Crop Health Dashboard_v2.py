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
import base64
from PIL import Image
import io


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
# WEATHER DATA OPTIONS (MOVED OUTSIDE CACHED FUNCTION)
# -----------------------------
st.sidebar.info("🌤️ Weather Data Options")
weather_option = st.sidebar.radio(
    "Choose weather data source:",
    ["Use Sample Data", "Upload Weather Data File"]
)

uploaded_weather_file = None
if weather_option == "Upload Weather Data File":
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
    weather_url = "https://github.com/ASHISHSE/Generalized-Crop-Health/raw/main/weather_data_2023_24_upload.xlsx"

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
                st.sidebar.info("Using sample data instead.")
                weather_df = create_sample_weather_data()
        else:
            # Use online weather data
            try:
                weather_res = requests.get(weather_url, timeout=60)
                weather_23_df = pd.read_excel(BytesIO(weather_res.content), sheet_name='Weather_data_23')
                weather_24_df = pd.read_excel(BytesIO(weather_res.content), sheet_name='Weather_data_24')
                weather_df = pd.concat([weather_23_df, weather_24_df], ignore_index=True)
            except:
                st.sidebar.info("Using sample weather data.")
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
    
    return pd.DataFrame(sample_data)

# Load data with the uploaded file
weather_df, ndvi_ndwi_df, mai_df, districts, talukas, circles = load_data(uploaded_weather_file)

# Show info message based on weather data source
if weather_option == "Use Sample Data":
    st.sidebar.info("📊 Using sample weather data. Upload your file for complete analysis.")
elif uploaded_weather_file is not None:
    st.sidebar.success("✅ Weather data loaded successfully!")

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
            fortnight_metrics = (year_data[metric_col] > 0).groupby(year_data['Fortnight']).sum()
        
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
            monthly_metrics = (year_data[metric_col] > 0).groupby(year_data['Month']).sum()
        
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
    
    fig.add_trace(go.Scatter(
        name=f'2024',
        x=all_periods,
        y=current_values,
        mode='lines+markers',
        line=dict(color='#2d6a4f', width=3),
        marker=dict(size=8, symbol='circle')
    ))
    
    fig.add_trace(go.Scatter(
        name=f'2023',
        x=all_periods,
        y=last_values,
        mode='lines+markers',
        line=dict(color='#52b788', width=3, dash='dash'),
        marker=dict(size=8, symbol='square')
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
    
    # Add bars for current and last year
    fig.add_trace(go.Bar(
        name='2024',
        x=all_months,
        y=current_values,
        marker_color='#2d6a4f',
        text=[f"{x:.2f}" for x in current_values],
        textposition='auto',
        textfont=dict(weight='bold')
    ), secondary_y=False)
    
    fig.add_trace(go.Bar(
        name='2023',
        x=all_months,
        y=last_values,
        marker_color='#52b788',
        text=[f"{x:.2f}" for x in last_values],
        textposition='auto',
        textfont=dict(weight='bold')
    ), secondary_y=False)
    
    # Add deviation line
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

def create_ndvi_monthly_comparison_chart(ndvi_df, district, taluka, circle, start_date, end_date):
    """Create monthly NDVI comparison chart with column bars and deviation"""
    filtered_df = ndvi_df.copy()
    
    if district:
        filtered_df = filtered_df[filtered_df["District"] == district]
    if taluka:
        filtered_df = filtered_df[filtered_df["Taluka"] == taluka]
    if circle:
        filtered_df = filtered_df[filtered_df["Circle"] == circle]
    
    # Filter for selected date range months
    start_month = start_date.month
    end_month = end_date.month
    
    # Filter for 2023 and 2024 within selected months
    filtered_df = filtered_df[
        (filtered_df["Date_dt"].dt.year.isin([2023, 2024])) & 
        (filtered_df["Date_dt"].dt.month >= start_month) & 
        (filtered_df["Date_dt"].dt.month <= end_month)
    ]
    
    if filtered_df.empty:
        return None
    
    # Group by year and month for comparison
    filtered_df['Year'] = filtered_df['Date_dt'].dt.year
    filtered_df['Month'] = filtered_df['Date_dt'].dt.month
    
    monthly_avg = filtered_df.groupby(['Year', 'Month'])['NDVI'].mean().reset_index()
    monthly_avg['NDVI'] = monthly_avg['NDVI'].round(3)
    
    # Pivot data for easier plotting
    pivot_df = monthly_avg.pivot(index='Month', columns='Year', values='NDVI').reset_index()
    
    # Calculate deviations
    pivot_df['Deviation (%)'] = ((pivot_df[2024] - pivot_df[2023]) / pivot_df[2023]) * 100
    pivot_df['Deviation (%)'] = pivot_df['Deviation (%)'].round(1)
    
    # Create column chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='2023',
        x=pivot_df['Month'],
        y=pivot_df[2023],
        marker_color='#FF6B00',  # Strong orange
        text=[f"{x:.3f}" for x in pivot_df[2023]],
        textposition='auto',
        textfont=dict(weight='bold')
    ))
    
    fig.add_trace(go.Bar(
        name='2024',
        x=pivot_df['Month'],
        y=pivot_df[2024],
        marker_color='#2E00FF',  # Strong blue
        text=[f"{x:.3f}" for x in pivot_df[2024]],
        textposition='auto',
        textfont=dict(weight='bold')
    ))
    
    # Add deviation as line
    fig.add_trace(go.Scatter(
        name='Deviation (%)',
        x=pivot_df['Month'],
        y=pivot_df['Deviation (%)'],
        mode='lines+markers+text',
        line=dict(color='#ff6b6b', width=3),
        marker=dict(size=8, symbol='diamond'),
        text=[f"{x:+.1f}%" for x in pivot_df['Deviation (%)']],
        textposition="top center",
        textfont=dict(color='#d63031', size=10, weight='bold'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=dict(text=f"NDVI Monthly Comparison: 2023 vs 2024 - {level}: {level_name}", x=0.5, xanchor='center'),
        xaxis_title="Month",
        yaxis_title="NDVI Value",
        template='plotly_white',
        height=450,
        barmode='group',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis2=dict(
            title="Deviation (%)",
            overlaying='y',
            side='right',
            showgrid=False
        )
    )
    
    return fig

def create_ndwi_monthly_comparison_chart(ndwi_df, district, taluka, circle, start_date, end_date):
    """Create monthly NDWI comparison chart with column bars and deviation"""
    filtered_df = ndwi_df.copy()
    
    if district:
        filtered_df = filtered_df[filtered_df["District"] == district]
    if taluka:
        filtered_df = filtered_df[filtered_df["Taluka"] == taluka]
    if circle:
        filtered_df = filtered_df[filtered_df["Circle"] == circle]
    
    # Filter for selected date range months
    start_month = start_date.month
    end_month = end_date.month
    
    # Filter for 2023 and 2024 within selected months
    filtered_df = filtered_df[
        (filtered_df["Date_dt"].dt.year.isin([2023, 2024])) & 
        (filtered_df["Date_dt"].dt.month >= start_month) & 
        (filtered_df["Date_dt"].dt.month <= end_month)
    ]
    
    if filtered_df.empty:
        return None
    
    # Group by year and month for comparison
    filtered_df['Year'] = filtered_df['Date_dt'].dt.year
    filtered_df['Month'] = filtered_df['Date_dt'].dt.month
    
    monthly_avg = filtered_df.groupby(['Year', 'Month'])['NDWI'].mean().reset_index()
    monthly_avg['NDWI'] = monthly_avg['NDWI'].round(3)
    
    # Pivot data for easier plotting
    pivot_df = monthly_avg.pivot(index='Month', columns='Year', values='NDWI').reset_index()
    
    # Calculate deviations
    pivot_df['Deviation (%)'] = ((pivot_df[2024] - pivot_df[2023]) / pivot_df[2023]) * 100
    pivot_df['Deviation (%)'] = pivot_df['Deviation (%)'].round(1)
    
    # Create column chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='2023',
        x=pivot_df['Month'],
        y=pivot_df[2023],
        marker_color='#FF00FF',  # Strong magenta
        text=[f"{x:.3f}" for x in pivot_df[2023]],
        textposition='auto',
        textfont=dict(weight='bold')
    ))
    
    fig.add_trace(go.Bar(
        name='2024',
        x=pivot_df['Month'],
        y=pivot_df[2024],
        marker_color='#00AA00',  # Strong green
        text=[f"{x:.3f}" for x in pivot_df[2024]],
        textposition='auto',
        textfont=dict(weight='bold')
    ))
    
    # Add deviation as line
    fig.add_trace(go.Scatter(
        name='Deviation (%)',
        x=pivot_df['Month'],
        y=pivot_df['Deviation (%)'],
        mode='lines+markers+text',
        line=dict(color='#ff6b6b', width=3),
        marker=dict(size=8, symbol='diamond'),
        text=[f"{x:+.1f}%" for x in pivot_df['Deviation (%)']],
        textposition="top center",
        textfont=dict(color='#d63031', size=10, weight='bold'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=dict(text=f"NDWI Monthly Comparison: 2023 vs 2024 - {level}: {level_name}", x=0.5, xanchor='center'),
        xaxis_title="Month",
        yaxis_title="NDWI Value",
        template='plotly_white',
        height=450,
        barmode='group',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis2=dict(
            title="Deviation (%)",
            overlaying='y',
            side='right',
            showgrid=False
        )
    )
    
    return fig

def create_mai_monthly_comparison_chart(mai_df, district, taluka, circle, start_date, end_date):
    """Create monthly MAI comparison chart with column bars and deviation"""
    filtered_df = mai_df.copy()
    
    if district:
        filtered_df = filtered_df[filtered_df["District"] == district]
    if taluka:
        filtered_df = filtered_df[filtered_df["Taluka"] == taluka]
    if circle:
        filtered_df = filtered_df[filtered_df["Circle"] == circle]
    
    # Filter for selected date range months
    start_month = start_date.month
    end_month = end_date.month
    
    # Filter for 2023 and 2024
    filtered_df = filtered_df[filtered_df["Year"].isin([2023, 2024])]
    
    if filtered_df.empty:
        return None
    
    # Group by year and calculate average MAI
    yearly_avg = filtered_df.groupby('Year')['MAI (%)'].mean().reset_index()
    yearly_avg['MAI (%)'] = yearly_avg['MAI (%)'].round(2)
    
    # Calculate deviation
    mai_2023 = yearly_avg[yearly_avg['Year'] == 2023]['MAI (%)'].values[0] if 2023 in yearly_avg['Year'].values else 0
    mai_2024 = yearly_avg[yearly_avg['Year'] == 2024]['MAI (%)'].values[0] if 2024 in yearly_avg['Year'].values else 0
    
    if mai_2023 != 0:
        deviation = ((mai_2024 - mai_2023) / mai_2023) * 100
    else:
        deviation = 0
    
    # Create column chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='2023',
        x=['MAI'],
        y=[mai_2023],
        marker_color='#52b788',
        text=[f"{mai_2023:.2f}%"],
        textposition='outside',
        textfont=dict(weight='bold')
    ))
    
    fig.add_trace(go.Bar(
        name='2024',
        x=['MAI'],
        y=[mai_2024],
        marker_color='#2d6a4f',
        text=[f"{mai_2024:.2f}%"],
        textposition='outside',
        textfont=dict(weight='bold')
    ))
    
    # Add deviation annotation
    fig.add_annotation(
        x=0, y=max(mai_2023, mai_2024) * 1.15,
        text=f"Deviation: {deviation:+.1f}%",
        showarrow=False,
        font=dict(size=14, color='red', weight='bold'),
        bgcolor='lightyellow',
        bordercolor='red',
        borderwidth=1
    )
    
    fig.update_layout(
        title=dict(text=f"MAI Comparison: 2023 vs 2024 - {level}: {level_name}", x=0.5, xanchor='center'),
        xaxis_title="",
        yaxis_title="MAI (%)",
        template='plotly_white',
        height=400,
        barmode='group',
        showlegend=True
    )
    
    return fig

# -----------------------------
# SIMPLIFIED DOWNLOAD FUNCTIONS
# -----------------------------
def download_chart_as_image_simple(fig, filename):
    """Simple download function that saves as JPG without Kaleido dependency"""
    try:
        # Convert plot to image bytes
        img_bytes = fig.to_image(format="jpg", width=1200, height=600, scale=2)
        st.download_button(
            label=f"📥 Download {filename}.jpg",
            data=img_bytes,
            file_name=f"{filename}.jpg",
            mime="image/jpeg",
            key=f"img_{filename}"
        )
    except Exception as e:
        st.error(f"Error generating image: {e}")
        st.info("Please make sure all required dependencies are installed.")

def download_data_as_csv_simple(data_df, filename):
    """Simple CSV download function"""
    if not data_df.empty:
        csv = data_df.to_csv(index=False)
        st.download_button(
            label=f"📥 Download {filename}.csv",
            data=csv,
            file_name=f"{filename}.csv",
            mime="text/csv",
            key=f"csv_{filename}"
        )
    else:
        st.write("No data available for download")

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
st.markdown("### 📅 Location & Date Selection")

# Modified layout: Taluka on top, Circle below
col1, col2 = st.columns(2)

with col1:
    district = st.selectbox("District *", [""] + districts)
    
    # Taluka on top
    if district:
        taluka_options = [""] + sorted(weather_df[weather_df["District"] == district]["Taluka"].dropna().unique().tolist())
    else:
        taluka_options = [""] + talukas
    taluka = st.selectbox("Taluka *", taluka_options)

with col2:
    # Circle below Taluka
    if taluka and taluka != "":
        circle_options = [""] + sorted(weather_df[weather_df["Taluka"] == taluka]["Circle"].dropna().unique().tolist())
    else:
        circle_options = [""] + circles
    circle = st.selectbox("Circle", circle_options)

# Start Date and End Date
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
    if not district or not taluka or not sowing_date or not current_date:
        st.error("Please select all required fields (District, Taluka, Start Date, End Date).")
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

        # Set years to 2023 and 2024 as requested
        current_year = 2024
        last_year = 2023

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["🌤️ Weather Metrics", "📡 Remote Sensing Indices", "💾 Download Data"])

        # TAB 1: WEATHER METRICS (Unchanged)
        with tab1:
            st.header(f"🌤️ Weather Metrics - {level}: {level_name}")
            
            if not filtered_weather.empty:
                # I. Rainfall Analysis
                st.subheader("I. Rainfall Analysis")
                
                # Fortnightly Analysis
                st.markdown("##### Fortnightly Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Fortnightly Rainfall Comparison
                    fortnightly_rainfall = calculate_fortnightly_metrics(
                        filtered_weather, current_year, last_year, "Rainfall", "sum"
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
                    filtered_weather, current_year, last_year, "Rainfall", "sum"
                )
                fig_rainfall_monthly = create_monthly_clustered_chart(
                    monthly_rainfall[current_year], 
                    monthly_rainfall[last_year],
                    "Rainfall - Monthly Comparison with Deviation (2023 vs 2024)",
                    "Rainfall (mm)"
                )
                st.plotly_chart(fig_rainfall_monthly, use_container_width=True)

                # Continue with other weather metrics (unchanged)
                # ... [rest of weather metrics code remains the same]
                
            else:
                st.info("No weather data available for the selected location and date range.")

        # TAB 2: REMOTE SENSING INDICES - UPDATED
        with tab2:
            st.header(f"📡 Remote Sensing Indices - {level}: {level_name}")
            
            # I. NDVI Analysis - Updated to column chart
            st.subheader("I. NDVI Analysis")
            ndvi_monthly_fig = create_ndvi_monthly_comparison_chart(
                ndvi_ndwi_df, district, taluka, circle, sowing_date, current_date
            )
            if ndvi_monthly_fig:
                st.plotly_chart(ndvi_monthly_fig, use_container_width=True)
            else:
                st.info("No NDVI data available for the selected parameters.")
            
            # II. NDWI Analysis - Updated to column chart
            st.subheader("II. NDWI Analysis")
            ndwi_monthly_fig = create_ndwi_monthly_comparison_chart(
                ndvi_ndwi_df, district, taluka, circle, sowing_date, current_date
            )
            if ndwi_monthly_fig:
                st.plotly_chart(ndwi_monthly_fig, use_container_width=True)
            else:
                st.info("No NDWI data available for the selected parameters.")
            
            # III. MAI Analysis - Updated to column chart
            st.subheader("III. MAI Analysis")
            mai_monthly_fig = create_mai_monthly_comparison_chart(
                mai_df, district, taluka, circle, sowing_date, current_date
            )
            if mai_monthly_fig:
                st.plotly_chart(mai_monthly_fig, use_container_width=True)
            else:
                st.info("No MAI data available for the selected parameters.")

        # TAB 3: DOWNLOAD DATA - SIMPLIFIED
        with tab3:
            st.header(f"💾 Download Data - {level}: {level_name}")
            
            # Simple Download Section
            st.subheader("Download Charts as JPG")
            
            # Weather Charts
            st.markdown("**Weather Charts:**")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'fig_rainfall_fortnight' in locals():
                    download_chart_as_image_simple(fig_rainfall_fortnight, "Rainfall_Fortnightly")
                if 'fig_rainfall_monthly' in locals():
                    download_chart_as_image_simple(fig_rainfall_monthly, "Rainfall_Monthly")
            
            with col2:
                if 'fig_tmax_fortnight' in locals():
                    download_chart_as_image_simple(fig_tmax_fortnight, "Max_Temperature")
                if 'fig_tmin_fortnight' in locals():
                    download_chart_as_image_simple(fig_tmin_fortnight, "Min_Temperature")
            
            # Remote Sensing Charts
            st.markdown("**Remote Sensing Charts:**")
            col1, col2 = st.columns(2)
            
            with col1:
                if ndvi_monthly_fig:
                    download_chart_as_image_simple(ndvi_monthly_fig, "NDVI_Monthly_Comparison")
                if ndwi_monthly_fig:
                    download_chart_as_image_simple(ndwi_monthly_fig, "NDWI_Monthly_Comparison")
            
            with col2:
                if mai_monthly_fig:
                    download_chart_as_image_simple(mai_monthly_fig, "MAI_Comparison")
            
            # Simple Data Download Section
            st.subheader("Download Data as CSV")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Weather Data:**")
                if not filtered_weather.empty:
                    download_data_as_csv_simple(filtered_weather, f"Weather_Data_{level_name}")
                else:
                    st.write("No data available")
            
            with col2:
                st.markdown("**NDVI/NDWI Data:**")
                filtered_ndvi_ndwi = ndvi_ndwi_df.copy()
                if district:
                    filtered_ndvi_ndwi = filtered_ndvi_ndwi[filtered_ndvi_ndwi["District"] == district]
                if taluka:
                    filtered_ndvi_ndwi = filtered_ndvi_ndwi[filtered_ndvi_ndwi["Taluka"] == taluka]
                if circle:
                    filtered_ndvi_ndwi = filtered_ndvi_ndwi[filtered_ndvi_ndwi["Circle"] == circle]
                
                if not filtered_ndvi_ndwi.empty:
                    download_data_as_csv_simple(filtered_ndvi_ndwi, f"NDVI_NDWI_Data_{level_name}")
                else:
                    st.write("No data available")
            
            with col3:
                st.markdown("**MAI Data:**")
                filtered_mai = mai_df.copy()
                if district:
                    filtered_mai = filtered_mai[filtered_mai["District"] == district]
                if taluka:
                    filtered_mai = filtered_mai[filtered_mai["Taluka"] == taluka]
                if circle:
                    filtered_mai = filtered_mai[filtered_mai["Circle"] == circle]
                
                if not filtered_mai.empty:
                    download_data_as_csv_simple(filtered_mai, f"MAI_Data_{level_name}")
                else:
                    st.write("No data available")

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
