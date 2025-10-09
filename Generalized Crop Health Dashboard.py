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
import pyxlsb  # For reading .xlsb files

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
        "Upload Weather Data (.xlsx or .xlsb)", 
        type=['xlsx', 'xlsb'],
        help="File should contain sheets: 'Weather_data_23' and 'Weather_data_24'"
    )

# -----------------------------
# LOAD DATA - UPDATED with widget removed
# -----------------------------
@st.cache_data
def load_data(_uploaded_file=None):
    # Updated URLs as per request
    ndvi_ndwi_url = "https://github.com/ASHISHSE/Generalized-Crop-Health/raw/main/1Maharashtra_NDVI_NDWI_old_circle_2023_2024_upload.xlsx"
    mai_url = "https://github.com/ASHISHSE/Generalized-Crop-Health/raw/main/1Circlewise_Data_MAI_2023_24_upload.xlsx"

    try:
        # Load NDVI & NDWI data
        ndvi_ndwi_res = requests.get(ndvi_ndwi_url, timeout=60)
        ndvi_ndwi_df = pd.read_excel(BytesIO(ndvi_ndwi_res.content))
        
        # Load MAI data
        mai_res = requests.get(mai_url, timeout=60)
        mai_df = pd.read_excel(BytesIO(mai_res.content))
        
        # Process NDVI & NDWI data
        ndvi_ndwi_df["Date_dt"] = pd.to_datetime(ndvi_ndwi_df["Date(DD-MM-YYYY)"], format="%d-%m-%Y", errors="coerce")
        ndvi_ndwi_df = ndvi_ndwi_df.dropna(subset=["Date_dt"]).copy()
        
        # Process MAI data
        mai_df["Year"] = pd.to_numeric(mai_df["Year"], errors="coerce")
        mai_df["MAI (%)"] = pd.to_numeric(mai_df["MAI (%)"], errors="coerce")
        
        # Handle weather data based on uploaded file
        if _uploaded_file is not None:
            try:
                if _uploaded_file.name.endswith('.xlsb'):
                    # Read .xlsb file
                    with pyxlsb.open_workbook(_uploaded_file) as wb:
                        weather_23_df = pd.DataFrame()
                        weather_24_df = pd.DataFrame()
                        
                        # Read Weather_data_23 sheet
                        with wb.get_sheet('Weather_data_23') as sheet:
                            data = []
                            for row in sheet.rows():
                                data.append([item.v for item in row])
                            if data:
                                weather_23_df = pd.DataFrame(data[1:], columns=data[0])
                        
                        # Read Weather_data_24 sheet
                        with wb.get_sheet('Weather_data_24') as sheet:
                            data = []
                            for row in sheet.rows():
                                data.append([item.v for item in row])
                            if data:
                                weather_24_df = pd.DataFrame(data[1:], columns=data[0])
                
                else:
                    # Read .xlsx file
                    weather_23_df = pd.read_excel(_uploaded_file, sheet_name='Weather_data_23')
                    weather_24_df = pd.read_excel(_uploaded_file, sheet_name='Weather_data_24')
                
                # Combine weather data
                weather_df = pd.concat([weather_23_df, weather_24_df], ignore_index=True)
                
            except Exception as e:
                st.sidebar.error(f"Error reading weather file: {e}")
                st.sidebar.info("Using sample data instead.")
                weather_df = create_sample_weather_data()
        else:
            # Use sample data
            weather_df = create_sample_weather_data()
        
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
# CHART CREATION FUNCTIONS
# -----------------------------
def create_weather_comparison_chart(current_year_data, last_year_data, title, yaxis_title):
    """Create comparison chart for weather metrics"""
    # Get all possible periods
    all_periods = sorted(set(current_year_data.index) | set(last_year_data.index))
    
    current_values = [current_year_data.get(period, 0) for period in all_periods]
    last_values = [last_year_data.get(period, 0) for period in all_periods]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=f'Current Year',
        x=all_periods,
        y=current_values,
        marker_color='#2d6a4f'
    ))
    
    fig.add_trace(go.Bar(
        name=f'Previous Year',
        x=all_periods,
        y=last_values,
        marker_color='#52b788'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis_title=yaxis_title,
        barmode='group',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_deviation_chart(current_year_data, last_year_data, title, yaxis_title):
    """Create deviation chart"""
    all_periods = sorted(set(current_year_data.index) | set(last_year_data.index))
    
    deviations = []
    for period in all_periods:
        current_val = current_year_data.get(period, 0)
        last_val = last_year_data.get(period, 0)
        
        if last_val != 0:
            if 'Deviation (%)' in title:
                deviation = ((current_val - last_val) / last_val) * 100
            else:
                deviation = current_val - last_val
        else:
            deviation = 0
        
        deviations.append(deviation)
    
    colors = ['green' if x >= 0 else 'red' for x in deviations]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=all_periods,
        y=deviations,
        marker_color=colors,
        name='Deviation'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis_title=yaxis_title,
        template='plotly_white',
        height=400
    )
    
    return fig

def create_ndvi_line_chart(ndvi_df, sowing_date, current_date, district, taluka, circle):
    """Create NDVI line chart"""
    filtered_df = ndvi_df.copy()
    
    if district:
        filtered_df = filtered_df[filtered_df["District"] == district]
    if taluka:
        filtered_df = filtered_df[filtered_df["Taluka"] == taluka]
    if circle:
        filtered_df = filtered_df[filtered_df["Circle"] == circle]
    
    filtered_df = filtered_df[
        (filtered_df["Date_dt"] >= pd.to_datetime(sowing_date)) & 
        (filtered_df["Date_dt"] <= pd.to_datetime(current_date))
    ]
    
    if filtered_df.empty:
        return None
    
    fig = px.line(
        filtered_df, 
        x="Date_dt", 
        y="NDVI", 
        title="NDVI Trend Over Time",
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="NDVI",
        template='plotly_white',
        height=400
    )
    
    return fig

def create_ndwi_line_chart(ndwi_df, sowing_date, current_date, district, taluka, circle):
    """Create NDWI line chart"""
    filtered_df = ndwi_df.copy()
    
    if district:
        filtered_df = filtered_df[filtered_df["District"] == district]
    if taluka:
        filtered_df = filtered_df[filtered_df["Taluka"] == taluka]
    if circle:
        filtered_df = filtered_df[filtered_df["Circle"] == circle]
    
    filtered_df = filtered_df[
        (filtered_df["Date_dt"] >= pd.to_datetime(sowing_date)) & 
        (filtered_df["Date_dt"] <= pd.to_datetime(current_date))
    ]
    
    if filtered_df.empty:
        return None
    
    fig = px.line(
        filtered_df, 
        x="Date_dt", 
        y="NDWI", 
        title="NDWI Trend Over Time",
        markers=True,
        color_discrete_sequence=['blue']
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="NDWI",
        template='plotly_white',
        height=400
    )
    
    return fig

def create_mai_comparison_chart(mai_df, sowing_date, current_date, district, taluka, circle):
    """Create MAI comparison chart"""
    filtered_df = mai_df.copy()
    
    if district:
        filtered_df = filtered_df[filtered_df["District"] == district]
    if taluka:
        filtered_df = filtered_df[filtered_df["Taluka"] == taluka]
    if circle:
        filtered_df = filtered_df[filtered_df["Circle"] == circle]
    
    # Filter by year range
    current_year = current_date.year
    last_year = current_year - 1
    
    filtered_df = filtered_df[filtered_df["Year"].isin([current_year, last_year])]
    
    if filtered_df.empty:
        return None
    
    fig = px.bar(
        filtered_df,
        x="Year",
        y="MAI (%)",
        color="Year",
        title="MAI Comparison Between Years",
        color_discrete_map={current_year: '#2d6a4f', last_year: '#52b788'}
    )
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="MAI (%)",
        template='plotly_white',
        height=400
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
                
                # Rainfall Deviation
                st.subheader("Rainfall Deviation")
                dev_col1, dev_col2 = st.columns(2)
                
                with dev_col1:
                    # Fortnightly Deviation
                    fig_rainfall_dev_fortnight = create_deviation_chart(
                        fortnightly_rainfall[current_year], 
                        fortnightly_rainfall[last_year],
                        "Rainfall Deviation - Fortnightly (%)",
                        "Deviation (%)"
                    )
                    st.plotly_chart(fig_rainfall_dev_fortnight, use_container_width=True)
                
                with dev_col2:
                    # Monthly Deviation
                    fig_rainfall_dev_monthly = create_deviation_chart(
                        monthly_rainfall[current_year], 
                        monthly_rainfall[last_year],
                        "Rainfall Deviation - Monthly (%)",
                        "Deviation (%)"
                    )
                    st.plotly_chart(fig_rainfall_dev_monthly, use_container_width=True)

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

                # III. Maximum Temperature Analysis
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
                
                # Tmax Deviation
                st.subheader("Max Temperature Deviation")
                dev_col1, dev_col2 = st.columns(2)
                
                with dev_col1:
                    # Fortnightly Tmax Deviation
                    fig_tmax_dev_fortnight = create_deviation_chart(
                        fortnightly_tmax[current_year], 
                        fortnightly_tmax[last_year],
                        "Max Temperature Deviation - Fortnightly",
                        "Deviation (°C)"
                    )
                    st.plotly_chart(fig_tmax_dev_fortnight, use_container_width=True)
                
                with dev_col2:
                    # Monthly Tmax Deviation
                    fig_tmax_dev_monthly = create_deviation_chart(
                        monthly_tmax[current_year], 
                        monthly_tmax[last_year],
                        "Max Temperature Deviation - Monthly",
                        "Deviation (°C)"
                    )
                    st.plotly_chart(fig_tmax_dev_monthly, use_container_width=True)

                # IV. Minimum Temperature Analysis
                st.subheader("IV. Minimum Temperature Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Fortnightly Tmin
                    fortnightly_tmin = calculate_fortnightly_metrics(
                        filtered_weather, current_year, last_year, "Tmin", "mean"
                    )
                    fig_tmin_fortnight = create_weather_comparison_chart(
                        fortnightly_tmin[current_year], 
                        fortnightly_tmin[last_year],
                        "Min Temperature - Fortnightly Average",
                        "Temperature (°C)"
                    )
                    st.plotly_chart(fig_tmin_fortnight, use_container_width=True)
                
                with col2:
                    # Monthly Tmin
                    monthly_tmin = calculate_monthly_metrics(
                        filtered_weather, current_year, last_year, "Tmin", "mean"
                    )
                    fig_tmin_monthly = create_weather_comparison_chart(
                        monthly_tmin[current_year], 
                        monthly_tmin[last_year],
                        "Min Temperature - Monthly Average",
                        "Temperature (°C)"
                    )
                    st.plotly_chart(fig_tmin_monthly, use_container_width=True)
                
                # Tmin Deviation
                st.subheader("Min Temperature Deviation")
                dev_col1, dev_col2 = st.columns(2)
                
                with dev_col1:
                    # Fortnightly Tmin Deviation
                    fig_tmin_dev_fortnight = create_deviation_chart(
                        fortnightly_tmin[current_year], 
                        fortnightly_tmin[last_year],
                        "Min Temperature Deviation - Fortnightly",
                        "Deviation (°C)"
                    )
                    st.plotly_chart(fig_tmin_dev_fortnight, use_container_width=True)
                
                with dev_col2:
                    # Monthly Tmin Deviation
                    fig_tmin_dev_monthly = create_deviation_chart(
                        monthly_tmin[current_year], 
                        monthly_tmin[last_year],
                        "Min Temperature Deviation - Monthly",
                        "Deviation (°C)"
                    )
                    st.plotly_chart(fig_tmin_dev_monthly, use_container_width=True)

                # V. Maximum Relative Humidity Analysis
                st.subheader("V. Maximum Relative Humidity Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Fortnightly RH max
                    fortnightly_rh_max = calculate_fortnightly_metrics(
                        filtered_weather, current_year, last_year, "max_Rh", "mean"
                    )
                    fig_rh_max_fortnight = create_weather_comparison_chart(
                        fortnightly_rh_max[current_year], 
                        fortnightly_rh_max[last_year],
                        "Max RH - Fortnightly Average",
                        "Relative Humidity (%)"
                    )
                    st.plotly_chart(fig_rh_max_fortnight, use_container_width=True)
                
                with col2:
                    # Monthly RH max
                    monthly_rh_max = calculate_monthly_metrics(
                        filtered_weather, current_year, last_year, "max_Rh", "mean"
                    )
                    fig_rh_max_monthly = create_weather_comparison_chart(
                        monthly_rh_max[current_year], 
                        monthly_rh_max[last_year],
                        "Max RH - Monthly Average",
                        "Relative Humidity (%)"
                    )
                    st.plotly_chart(fig_rh_max_monthly, use_container_width=True)
                
                # RH Max Deviation
                st.subheader("Max RH Deviation")
                dev_col1, dev_col2 = st.columns(2)
                
                with dev_col1:
                    # Fortnightly RH Max Deviation
                    fig_rh_max_dev_fortnight = create_deviation_chart(
                        fortnightly_rh_max[current_year], 
                        fortnightly_rh_max[last_year],
                        "Max RH Deviation - Fortnightly",
                        "Deviation (%)"
                    )
                    st.plotly_chart(fig_rh_max_dev_fortnight, use_container_width=True)
                
                with dev_col2:
                    # Monthly RH Max Deviation
                    fig_rh_max_dev_monthly = create_deviation_chart(
                        monthly_rh_max[current_year], 
                        monthly_rh_max[last_year],
                        "Max RH Deviation - Monthly",
                        "Deviation (%)"
                    )
                    st.plotly_chart(fig_rh_max_dev_monthly, use_container_width=True)

                # VI. Minimum Relative Humidity Analysis
                st.subheader("VI. Minimum Relative Humidity Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Fortnightly RH min
                    fortnightly_rh_min = calculate_fortnightly_metrics(
                        filtered_weather, current_year, last_year, "min_Rh", "mean"
                    )
                    fig_rh_min_fortnight = create_weather_comparison_chart(
                        fortnightly_rh_min[current_year], 
                        fortnightly_rh_min[last_year],
                        "Min RH - Fortnightly Average",
                        "Relative Humidity (%)"
                    )
                    st.plotly_chart(fig_rh_min_fortnight, use_container_width=True)
                
                with col2:
                    # Monthly RH min
                    monthly_rh_min = calculate_monthly_metrics(
                        filtered_weather, current_year, last_year, "min_Rh", "mean"
                    )
                    fig_rh_min_monthly = create_weather_comparison_chart(
                        monthly_rh_min[current_year], 
                        monthly_rh_min[last_year],
                        "Min RH - Monthly Average",
                        "Relative Humidity (%)"
                    )
                    st.plotly_chart(fig_rh_min_monthly, use_container_width=True)
                
                # RH Min Deviation
                st.subheader("Min RH Deviation")
                dev_col1, dev_col2 = st.columns(2)
                
                with dev_col1:
                    # Fortnightly RH Min Deviation
                    fig_rh_min_dev_fortnight = create_deviation_chart(
                        fortnightly_rh_min[current_year], 
                        fortnightly_rh_min[last_year],
                        "Min RH Deviation - Fortnightly",
                        "Deviation (%)"
                    )
                    st.plotly_chart(fig_rh_min_dev_fortnight, use_container_width=True)
                
                with dev_col2:
                    # Monthly RH Min Deviation
                    fig_rh_min_dev_monthly = create_deviation_chart(
                        monthly_rh_min[current_year], 
                        monthly_rh_min[last_year],
                        "Min RH Deviation - Monthly",
                        "Deviation (%)"
                    )
                    st.plotly_chart(fig_rh_min_dev_monthly, use_container_width=True)
                
            else:
                st.info("No weather data available for the selected location and date range.")

        # TAB 2: REMOTE SENSING INDICES
        with tab2:
            st.header(f"📡 Remote Sensing Indices - {level}: {level_name}")
            
            # I. NDVI Line Chart
            st.subheader("I. NDVI Analysis")
            ndvi_fig = create_ndvi_line_chart(
                ndvi_ndwi_df, sowing_date, current_date, district, taluka, circle
            )
            if ndvi_fig:
                st.plotly_chart(ndvi_fig, use_container_width=True)
            else:
                st.info("No NDVI data available for the selected parameters.")
            
            # II. NDWI Line Chart
            st.subheader("II. NDWI Analysis")
            ndwi_fig = create_ndwi_line_chart(
                ndvi_ndwi_df, sowing_date, current_date, district, taluka, circle
            )
            if ndwi_fig:
                st.plotly_chart(ndwi_fig, use_container_width=True)
            else:
                st.info("No NDWI data available for the selected parameters.")
            
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
