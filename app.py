# app.py - Complete Farm Analysis Dashboard (2020-2024)
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster, HeatMap
import geopandas as gpd
from shapely.geometry import Point
import json
import os
import re
from datetime import datetime
import base64
from io import BytesIO
import tempfile
import zipfile
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Farm Analysis Dashboard - Aman Chauhan",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #E8F5E9, #C8E6C9);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1B5E20;
        padding: 0.5rem;
        border-bottom: 2px solid #4CAF50;
        margin-top: 1rem;
    }
    .metric-card {
        background: #f0f4f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E7D32;
        margin: 0.5rem 0;
    }
    .highlight-box {
        background: #FFF3E0;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #FFB74D;
    }
    .farm-card {
        background: #E8F5E9;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 5px;
        border-left: 4px solid #2E7D32;
    }
    .critical-farm {
        background: #FFEBEE;
        border-left: 4px solid #F44336;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        color: #666;
        border-top: 1px solid #ddd;
        margin-top: 2rem;
    }
    .stButton > button {
        background-color: #2E7D32;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background-color: #1B5E20;
        color: white;
    }
    .upload-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 2px dashed #4CAF50;
        margin-bottom: 1rem;
    }
    .data-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
    .status-loaded {
        background: #E8F5E9;
        border-left: 4px solid #4CAF50;
    }
    .status-missing {
        background: #FFEBEE;
        border-left: 4px solid #F44336;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'shapefile_data' not in st.session_state:
    st.session_state.shapefile_data = None
if 'merged_data' not in st.session_state:
    st.session_state.merged_data = None
if 'farm_analysis' not in st.session_state:
    st.session_state.farm_analysis = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'selected_farm_key' not in st.session_state:
    st.session_state.selected_farm_key = None

# Constants
PROJECT_FOLDER = "./project_farm"
SHAPEFILE_NAME = "bg_monitoring_32.shp"
YEARS = [2020, 2021, 2022, 2023, 2024]

# Helper Functions
@st.cache_data
def load_shapefile(shp_path):
    """Load shapefile from folder"""
    try:
        gdf = gpd.read_file(shp_path)
        return gdf
    except Exception as e:
        st.error(f"Error loading shapefile: {str(e)}")
        return None

@st.cache_data
def load_csv_data(csv_path):
    """Load CSV data"""
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

def extract_median_columns(df):
    """Extract median height columns for years 2020-2024"""
    median_cols = {}
    for year in YEARS:
        col_name = f'median_height_{year}'
        if col_name in df.columns:
            median_cols[year] = col_name
        else:
            # Try alternative naming
            alt_cols = [col for col in df.columns if str(year) in col and 'median' in col.lower()]
            if alt_cols:
                median_cols[year] = alt_cols[0]
    return median_cols

def merge_shapefile_with_csv(shapefile_gdf, csv_df, key_column='KEY'):
    """Merge shapefile with CSV using KEY column"""
    try:
        # Ensure KEY column exists
        if key_column not in shapefile_gdf.columns:
            st.error(f"Column '{key_column}' not found in shapefile. Available: {list(shapefile_gdf.columns)}")
            return None
        
        if key_column not in csv_df.columns:
            st.error(f"Column '{key_column}' not found in CSV. Available: {list(csv_df.columns)}")
            return None
        
        # Convert KEY to string for consistent matching
        shapefile_gdf[key_column] = shapefile_gdf[key_column].astype(str)
        csv_df[key_column] = csv_df[key_column].astype(str)
        
        # Merge on KEY
        merged = shapefile_gdf.merge(csv_df, on=key_column, how='inner')
        
        if len(merged) == 0:
            st.warning(f"No matching records found between shapefile and CSV on '{key_column}'")
            return None
        
        return merged
    except Exception as e:
        st.error(f"Error merging data: {str(e)}")
        return None

def analyze_farm_data(merged_df, threshold=7.0):
    """Analyze farm data for trends using median heights"""
    results = []
    
    # Extract median columns
    median_cols = extract_median_columns(merged_df)
    
    if len(median_cols) < 2:
        st.warning("Need at least 2 years of median height data for trend analysis")
        return None
    
    for idx, row in merged_df.iterrows():
        # Get geometry centroid
        if hasattr(row.geometry, 'centroid'):
            centroid = row.geometry.centroid
            lat = centroid.y
            lon = centroid.x
        else:
            lat = row.get('LAT', row.get('latitude', np.nan))
            lon = row.get('LON', row.get('longitude', np.nan))
        
        # Extract median values for each year
        values = []
        years = []
        stds = []
        pixel_counts = []
        
        for year, col in sorted(median_cols.items()):
            val = row[col]
            if pd.notna(val) and val is not None:
                values.append(float(val))
                years.append(year)
                
                # Get std if available
                std_col = f'std_{year}'
                if std_col in row and pd.notna(row[std_col]):
                    stds.append(float(row[std_col]))
                else:
                    stds.append(0)
                
                # Get pixel count if available
                count_col = f'pixel_count_{year}'
                if count_col in row and pd.notna(row[count_col]):
                    pixel_counts.append(float(row[count_col]))
                else:
                    pixel_counts.append(0)
        
        if len(values) >= 2:
            # Calculate statistics
            is_increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
            is_decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1))
            is_low = values[-1] < threshold
            
            growth_rate = ((values[-1] - values[0]) / values[0] * 100) if values[0] > 0 else 0
            
            results.append({
                'KEY': row.get('KEY', idx),
                'latitude': lat,
                'longitude': lon,
                'values': values,
                'years': years,
                'stds': stds,
                'pixel_counts': pixel_counts,
                'is_increasing': is_increasing,
                'is_decreasing': is_decreasing,
                'is_low': is_low,
                'growth_rate': growth_rate,
                'last_value': values[-1],
                'first_value': values[0],
                'avg_value': np.mean(values),
                'median_value': np.median(values),
                'std_value': np.std(values) if len(values) > 1 else 0,
                'max_value': max(values),
                'min_value': min(values),
                'total_growth': values[-1] - values[0] if len(values) > 1 else 0,
                'max_growth_year': years[np.argmax(np.diff(values))] if len(values) > 1 else None,
                'farm_data': row.to_dict()
            })
    
    return pd.DataFrame(results)

def create_farm_map(farm_analysis_df, threshold=7.0):
    """Create interactive map with farms"""
    if farm_analysis_df.empty:
        return None
    
    center_lat = farm_analysis_df['latitude'].mean()
    center_lon = farm_analysis_df['longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    # Add satellite basemap
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite'
    ).add_to(m)
    
    # Add farms with color coding
    marker_cluster = MarkerCluster().add_to(m)
    
    for _, farm in farm_analysis_df.iterrows():
        # Determine color based on status
        if farm['is_low'] and farm['is_increasing']:
            color = 'orange'
            status_text = "⚠️ Low & Increasing"
        elif farm['is_low']:
            color = 'red'
            status_text = "🔴 Low & Not Increasing"
        elif farm['is_increasing']:
            color = 'green'
            status_text = "✅ Good & Increasing"
        else:
            color = 'blue'
            status_text = "ℹ️ Good & Stable"
        
        # Build yearly values string
        yearly_str = ''.join([
            f"{year}: {value:.2f}m (±{std:.2f})<br>" 
            for year, value, std in zip(farm['years'], farm['values'], farm['stds'])
        ])
        
        popup_html = f"""
        <div style="font-size: 13px; font-family: Arial, sans-serif; max-width: 300px;">
            <b style="color: #2E7D32;">Farm KEY: {farm['KEY']}</b><br>
            <hr style="margin: 5px 0;">
            <b>Status:</b> {status_text}<br>
            <b>Current Height:</b> {farm['last_value']:.2f} m<br>
            <b>Growth Rate:</b> <span style="color: {'#2E7D32' if farm['growth_rate'] > 0 else '#F44336'};">{farm['growth_rate']:.1f}%</span><br>
            <b>Avg Height:</b> {farm['avg_value']:.2f} m<br>
            <b>Median Height:</b> {farm['median_value']:.2f} m<br>
            <hr style="margin: 5px 0;">
            <b>Yearly Values (Mean ± Std):</b><br>
            {yearly_str}
        </div>
        """
        
        folium.CircleMarker(
            location=[farm['latitude'], farm['longitude']],
            radius=10,
            popup=folium.Popup(popup_html, max_width=350),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=3,
            tooltip=f"KEY: {farm['KEY']} - {farm['last_value']:.2f}m - {status_text}"
        ).add_to(marker_cluster)
    
    # Add a heatmap layer for height distribution
    heat_data = [[row['latitude'], row['longitude'], row['last_value']] 
                 for _, row in farm_analysis_df.iterrows()]
    HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
    
    folium.LayerControl(position='topright').add_to(m)
    
    return m

def create_trend_chart(farm_analysis_df, selected_key=None):
    """Create trend chart for farm(s) with error bars"""
    fig = go.Figure()
    
    if selected_key is not None:
        farm = farm_analysis_df[farm_analysis_df['KEY'] == selected_key]
        if not farm.empty:
            farm = farm.iloc[0]
            fig.add_trace(go.Scatter(
                x=farm['years'],
                y=farm['values'],
                mode='lines+markers',
                name=f"KEY: {selected_key}",
                line=dict(color='#2E7D32', width=4),
                marker=dict(size=12),
                error_y=dict(
                    type='data',
                    array=farm['stds'],
                    visible=True,
                    color='#666'
                ),
                hovertemplate='Year: %{x}<br>Height: %{y:.2f}m<extra></extra>'
            ))
            
            # Add annotations for each point
            for year, value, std in zip(farm['years'], farm['values'], farm['stds']):
                fig.add_annotation(
                    x=year,
                    y=value,
                    text=f"{value:.1f}m",
                    showarrow=True,
                    arrowhead=1,
                    ay=-20
                )
    else:
        # Show top 5 farms with highest growth
        filtered = farm_analysis_df.nlargest(5, 'growth_rate')
        
        colors = px.colors.qualitative.Set3
        for i, (_, farm) in enumerate(filtered.iterrows()):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=farm['years'],
                y=farm['values'],
                mode='lines+markers',
                name=f"KEY: {farm['KEY']}",
                line=dict(color=color, width=2),
                marker=dict(size=8),
                error_y=dict(
                    type='data',
                    array=farm['stds'],
                    visible=True
                ),
                hovertemplate=f"KEY: {farm['KEY']}<br>Year: %{{x}}<br>Height: %{{y:.2f}}m<extra></extra>"
            ))
    
    fig.update_layout(
        title="Farm Height Trends (2020-2024)",
        xaxis_title="Year",
        yaxis_title="Median Height (m)",
        height=400,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_summary_dashboard(farm_analysis_df, threshold=7.0):
    """Create comprehensive summary dashboard"""
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Farm Status Distribution",
            "Height Distribution (2024)",
            "Growth Rate Distribution",
            f"Farms Below {threshold}m (2024)",
            "Top 10 Growth Rates",
            "Yearly Height Trends (Box Plot)"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # 1. Status Distribution
    status_counts = {
        'Low & Increasing': len(farm_analysis_df[(farm_analysis_df['is_low']) & (farm_analysis_df['is_increasing'])]),
        'Low & Not Increasing': len(farm_analysis_df[(farm_analysis_df['is_low']) & (~farm_analysis_df['is_increasing'])]),
        'Good & Increasing': len(farm_analysis_df[(~farm_analysis_df['is_low']) & (farm_analysis_df['is_increasing'])]),
        'Good & Stable': len(farm_analysis_df[(~farm_analysis_df['is_low']) & (~farm_analysis_df['is_increasing'])])
    }
    
    fig.add_trace(
        go.Pie(
            labels=list(status_counts.keys()),
            values=list(status_counts.values()),
            hole=0.4,
            marker=dict(colors=['#FFA500', '#FF4444', '#4CAF50', '#2196F3']),
            textinfo='label+percent'
        ),
        row=1, col=1
    )
    
    # 2. Height Distribution (2024)
    heights = farm_analysis_df['last_value'].dropna()
    fig.add_trace(
        go.Histogram(
            x=heights,
            nbinsx=20,
            marker_color='#4CAF50',
            hovertemplate='Height: %{x:.2f}m<br>Count: %{y}<extra></extra>'
        ),
        row=1, col=2
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_vline(x=farm_analysis_df['median_value'].median(), line_dash="dot", line_color="blue", 
                  annotation_text=f"Median: {farm_analysis_df['median_value'].median():.1f}m", row=1, col=2)
    
    # 3. Growth Rate Distribution
    growth_rates = farm_analysis_df['growth_rate'].dropna()
    fig.add_trace(
        go.Histogram(
            x=growth_rates,
            nbinsx=20,
            marker_color='#2196F3',
            hovertemplate='Growth Rate: %{x:.1f}%<br>Count: %{y}<extra></extra>'
        ),
        row=1, col=3
    )
    fig.add_vline(x=0, line_dash="dash", line_color="green", row=1, col=3)
    
    # 4. Low Farms
    low_farms = farm_analysis_df[farm_analysis_df['is_low']].sort_values('last_value').head(10)
    fig.add_trace(
        go.Bar(
            x=low_farms['KEY'].astype(str),
            y=low_farms['last_value'],
            text=low_farms['last_value'].apply(lambda x: f"{x:.1f}m"),
            textposition='outside',
            marker_color=['#FF6B6B' if v < 5 else '#FFA07A' for v in low_farms['last_value']]
        ),
        row=2, col=1
    )
    
    # 5. Top 10 Growth Rates
    top_growth = farm_analysis_df.nlargest(10, 'growth_rate')
    fig.add_trace(
        go.Bar(
            x=top_growth['KEY'].astype(str),
            y=top_growth['growth_rate'],
            text=top_growth['growth_rate'].apply(lambda x: f"{x:.1f}%"),
            textposition='outside',
            marker_color='#4CAF50'
        ),
        row=2, col=2
    )
    
    # 6. Yearly Height Trends (Box plot)
    years = sorted(set().union(*farm_analysis_df['years'].apply(set)))
    for year in years:
        values = []
        for _, farm in farm_analysis_df.iterrows():
            if year in farm['years']:
                idx = farm['years'].index(year)
                values.append(farm['values'][idx])
        if values:
            fig.add_trace(
                go.Box(
                    y=values,
                    name=str(year),
                    boxmean='sd',
                    marker_color='#4CAF50'
                ),
                row=2, col=3
            )
    
    fig.update_layout(
        height=700,
        template='plotly_white',
        showlegend=False
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Height (m)", row=1, col=2)
    fig.update_xaxes(title_text="Growth Rate (%)", row=1, col=3)
    fig.update_xaxes(title_text="Farm KEY", row=2, col=1)
    fig.update_xaxes(title_text="Farm KEY", row=2, col=2)
    fig.update_xaxes(title_text="Year", row=2, col=3)
    
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=3)
    fig.update_yaxes(title_text="Height (m)", row=2, col=1)
    fig.update_yaxes(title_text="Growth Rate (%)", row=2, col=2)
    fig.update_yaxes(title_text="Height (m)", row=2, col=3)
    
    return fig

def create_correlation_heatmap(farm_analysis_df):
    """Create correlation heatmap for farm metrics"""
    numeric_cols = ['last_value', 'avg_value', 'median_value', 'growth_rate', 'total_growth', 'std_value']
    available_cols = [col for col in numeric_cols if col in farm_analysis_df.columns]
    
    if len(available_cols) < 2:
        return None
    
    corr_df = farm_analysis_df[available_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.index,
        y=corr_df.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_df.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Metric Correlation Matrix",
        height=400,
        template='plotly_white'
    )
    
    return fig

def load_shapefile_from_folder():
    """Load shapefile from project_farm folder"""
    shp_path = os.path.join(PROJECT_FOLDER, SHAPEFILE_NAME)
    if os.path.exists(shp_path):
        gdf = load_shapefile(shp_path)
        if gdf is not None:
            st.session_state.shapefile_data = gdf
            return True
    return False

# Main Title
st.markdown('<div class="main-header">🌾 Farm Analysis Dashboard (2020-2024)</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #555;">Internship Project at Varaha ClimateAg Private Limited | Aman Chauhan (22BCE0476)</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📁 Data Loading")
    
    # CSV Upload Section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📊 Upload CSV File")
    
    csv_file = st.file_uploader(
        "Upload CSV with farm height data (2020-2024)",
        type=['csv'],
        help="CSV file containing median_height_2020, median_height_2021, etc."
    )
    
    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            st.session_state.csv_data = df
            st.success(f"✅ Loaded CSV with {len(df)} rows")
            st.markdown("**CSV Preview:**")
            st.dataframe(df.head(3))
            
            # Detect median columns
            median_cols = extract_median_columns(df)
            if median_cols:
                st.success(f"✅ Found median height columns: {', '.join([f'{year}' for year in median_cols.keys()])}")
            else:
                st.warning("No median_height_YYYY columns found. Ensure column names like 'median_height_2020'")
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Shapefile Loading from Folder
    st.markdown("### 🗺️ Load Shapefile from Folder")
    
    if st.button("📂 Load Shapefile from project_farm"):
        with st.spinner("Loading shapefile..."):
            if load_shapefile_from_folder():
                st.success(f"✅ Loaded shapefile with {len(st.session_state.shapefile_data)} features")
                st.rerun()
            else:
                st.warning(f"Shapefile not found at {PROJECT_FOLDER}/{SHAPEFILE_NAME}")
    
    if st.session_state.shapefile_data is not None:
        st.markdown(f'<div class="data-status status-loaded">✅ Shapefile: {len(st.session_state.shapefile_data)} features</div>', unsafe_allow_html=True)
        st.info(f"Available columns: {list(st.session_state.shapefile_data.columns)}")
    else:
        st.markdown('<div class="data-status status-missing">⏳ Shapefile: Not loaded</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Merge and Analyze
    if st.session_state.csv_data is not None and st.session_state.shapefile_data is not None:
        st.markdown("## 🎯 Analysis Controls")
        
        # Threshold
        height_threshold = st.slider(
            "Height Threshold (meters)",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.5,
            help="Farms below this height in 2024 will be highlighted"
        )
        
        if st.button("🔗 Merge & Analyze", type="primary", use_container_width=True):
            with st.spinner("Merging and analyzing data..."):
                # Merge shapefile with CSV
                merged = merge_shapefile_with_csv(
                    st.session_state.shapefile_data,
                    st.session_state.csv_data,
                    'KEY'
                )
                
                if merged is not None:
                    st.session_state.merged_data = merged
                    
                    # Analyze trends
                    analysis_df = analyze_farm_data(merged, height_threshold)
                    if analysis_df is not None and not analysis_df.empty:
                        st.session_state.farm_analysis = analysis_df
                        st.success(f"✅ Analyzed {len(analysis_df)} farms (2020-2024)")
                        st.rerun()
                    else:
                        st.error("No valid data found for analysis")

# Main Content
if st.session_state.farm_analysis is not None:
    analysis_df = st.session_state.farm_analysis
    
    # Summary statistics
    st.markdown('<div class="section-header">📊 Farm Analysis Summary (2020-2024)</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    total_farms = len(analysis_df)
    low_farms = len(analysis_df[analysis_df['is_low']])
    increasing_farms = len(analysis_df[analysis_df['is_increasing']])
    low_increasing = len(analysis_df[(analysis_df['is_low']) & (analysis_df['is_increasing'])])
    avg_height = analysis_df['last_value'].mean()
    median_height = analysis_df['median_value'].median()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Total Farms</strong><br>
            <span style="font-size: 1.6rem; color: #2E7D32;">{total_farms}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #F44336;">
            <strong>🔴 Low Farms (2024)</strong><br>
            <span style="font-size: 1.6rem; color: #F44336;">{low_farms}</span>
            <br><span style="color: #666;">({low_farms/total_farms*100:.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #2196F3;">
            <strong>📈 Increasing</strong><br>
            <span style="font-size: 1.6rem; color: #2196F3;">{increasing_farms}</span>
            <br><span style="color: #666;">({increasing_farms/total_farms*100:.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #FFA500;">
            <strong>⚠️ Low & Increasing</strong><br>
            <span style="font-size: 1.6rem; color: #FFA500;">{low_increasing}</span>
            <br><span style="color: #666;">({low_increasing/total_farms*100:.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #9C27B0;">
            <strong>📊 Avg Height (2024)</strong><br>
            <span style="font-size: 1.6rem; color: #9C27B0;">{avg_height:.1f}m</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #00BCD4;">
            <strong>📊 Median Height</strong><br>
            <span style="font-size: 1.6rem; color: #00BCD4;">{median_height:.1f}m</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ Farm Map", "📈 Trends", "📊 Dashboard", "📋 Data", "📈 Correlation"])
    
    with tab1:
        st.subheader("Interactive Farm Map (2024)")
        
        m = create_farm_map(analysis_df, height_threshold)
        if m:
            folium_static(m, width=750, height=600)
            
            st.markdown(f"""
            <div style="background: #E3F2FD; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem;">
                <small>💡 <b>Legend:</b> 
                    🟢 Good & Increasing | 🔵 Good & Stable | 🟠 Low & Increasing | 🔴 Low & Not Increasing
                    <br>📊 Heatmap shows height distribution (2024)
                </small>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Farm Trend Analysis (2020-2024)")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_option = st.selectbox(
                "Filter farms:",
                ["All Farms", "Low Farms (<7m)", "Increasing Farms", "Low & Increasing", "High Farms (>7m)", "Top 10 Growth"]
            )
        
        with col2:
            if filter_option == "Low Farms (<7m)":
                filtered_df = analysis_df[analysis_df['is_low']]
            elif filter_option == "Increasing Farms":
                filtered_df = analysis_df[analysis_df['is_increasing']]
            elif filter_option == "Low & Increasing":
                filtered_df = analysis_df[(analysis_df['is_low']) & (analysis_df['is_increasing'])]
            elif filter_option == "High Farms (>7m)":
                filtered_df = analysis_df[~analysis_df['is_low']]
            elif filter_option == "Top 10 Growth":
                filtered_df = analysis_df.nlargest(10, 'growth_rate')
            else:
                filtered_df = analysis_df
            
            if not filtered_df.empty:
                farm_options = filtered_df['KEY'].tolist()
                selected_key = st.selectbox(
                    "Select farm by KEY:",
                    options=farm_options,
                    format_func=lambda x: f"KEY: {x} - Height: {filtered_df[filtered_df['KEY']==x]['last_value'].iloc[0]:.2f}m" if x in filtered_df['KEY'].values else str(x)
                )
            else:
                selected_key = None
                st.warning("No farms match the filter")
        
        with col3:
            if st.button("📊 Show Selected Farm"):
                if selected_key is not None:
                    st.session_state.selected_farm_key = selected_key
                    st.rerun()
        
        if not filtered_df.empty:
            # Trend chart
            if st.session_state.selected_farm_key is not None and st.session_state.selected_farm_key in filtered_df['KEY'].values:
                trend_fig = create_trend_chart(analysis_df, st.session_state.selected_farm_key)
            else:
                trend_fig = create_trend_chart(analysis_df, None)
            
            st.plotly_chart(trend_fig, use_container_width=True)
            
            # Show selected farm details
            if st.session_state.selected_farm_key is not None:
                farm = analysis_df[analysis_df['KEY'] == st.session_state.selected_farm_key]
                if not farm.empty:
                    farm = farm.iloc[0]
                    
                    st.markdown("### 📋 Farm Details (2020-2024)")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="farm-card {'critical-farm' if farm['is_low'] else ''}">
                            <strong>📊 Current Height (2024)</strong><br>
                            <span style="font-size: 1.4rem;">{farm['last_value']:.2f} m</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="farm-card {'critical-farm' if farm['is_low'] else ''}">
                            <strong>📈 Growth Rate</strong><br>
                            <span style="font-size: 1.4rem; color: {'#2E7D32' if farm['growth_rate'] > 0 else '#F44336'};">
                                {farm['growth_rate']:.1f}%
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="farm-card {'critical-farm' if farm['is_low'] else ''}">
                            <strong>📊 Median Height</strong><br>
                            <span style="font-size: 1.4rem; color: #00BCD4;">
                                {farm['median_value']:.2f} m
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        status = "🟢 Good" if not farm['is_low'] else "🔴 Critical"
                        trend = "📈 Increasing" if farm['is_increasing'] else "📉 Not Increasing"
                        st.markdown(f"""
                        <div class="farm-card {'critical-farm' if farm['is_low'] else ''}">
                            <strong>Status</strong><br>
                            {status} - {trend}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show yearly data table
                    st.markdown("**Yearly Data:**")
                    yearly_data = pd.DataFrame({
                        'Year': farm['years'],
                        'Median Height (m)': [f"{v:.2f}" for v in farm['values']],
                        'Std Dev': [f"{s:.2f}" for s in farm['stds']],
                        'Pixel Count': [int(c) for c in farm['pixel_counts']]
                    })
                    st.dataframe(yearly_data, use_container_width=True)
    
    with tab3:
        st.subheader("Summary Dashboard (2020-2024)")
        
        summary_fig = create_summary_dashboard(analysis_df, height_threshold)
        st.plotly_chart(summary_fig, use_container_width=True)
        
        # Additional stats
        st.markdown("### 📊 Key Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <strong>🏆 Best Growth</strong><br>
                <span style="font-size: 1.2rem; color: #2E7D32;">
                    {analysis_df['growth_rate'].max():.1f}%
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #F44336;">
                <strong>📉 Worst Growth</strong><br>
                <span style="font-size: 1.2rem; color: #F44336;">
                    {analysis_df['growth_rate'].min():.1f}%
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #9C27B0;">
                <strong>📊 Avg Growth</strong><br>
                <span style="font-size: 1.2rem; color: #9C27B0;">
                    {analysis_df['growth_rate'].mean():.1f}%
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #FF9800;">
                <strong>📈 Median Growth</strong><br>
                <span style="font-size: 1.2rem; color: #FF9800;">
                    {analysis_df['growth_rate'].median():.1f}%
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.subheader("Farm Data Table (2020-2024)")
        
        # Prepare display dataframe
        display_df = analysis_df.copy()
        display_df['values'] = display_df['values'].apply(
            lambda x: ', '.join([f"{v:.2f}" for v in x]) if x else 'N/A'
        )
        display_df['years'] = display_df['years'].apply(
            lambda x: ', '.join(map(str, x)) if x else 'N/A'
        )
        display_df['stds'] = display_df['stds'].apply(
            lambda x: ', '.join([f"{s:.2f}" for s in x]) if x else 'N/A'
        )
        
        # Select columns to display
        cols_to_show = ['KEY', 'latitude', 'longitude', 'last_value', 'avg_value', 
                       'median_value', 'growth_rate', 'is_low', 'is_increasing', 'years', 'values', 'stds']
        display_cols = [col for col in cols_to_show if col in display_df.columns]
        
        st.dataframe(display_df[display_cols], use_container_width=True)
        
        # Export option
        csv = analysis_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Farm Analysis CSV",
            data=csv,
            file_name=f"farm_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with tab5:
        st.subheader("Correlation Analysis")
        
        corr_fig = create_correlation_heatmap(analysis_df)
        if corr_fig:
            st.plotly_chart(corr_fig, use_container_width=True)
            
            st.markdown("""
            <div style="background: #E3F2FD; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem;">
                <small>💡 <b>Interpretation:</b> 
                    🔴 Strong positive correlation | 🔵 Strong negative correlation | ⚪ Weak/no correlation
                </small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Need at least 2 numeric metrics for correlation analysis")

else:
    st.info("📂 Upload a CSV file and load shapefile to begin analysis.")
    
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: #f5f5f5; border-radius: 10px;">
        <h3>🌾 Farm Analysis Dashboard (2020-2024)</h3>
        <p style="color: #666;">Upload CSV with median height data and load shapefile for visualization</p>
        <div style="margin-top: 1rem;">
            <span style="background: #E8F5E9; padding: 0.5rem 1rem; border-radius: 5px;">📊 CSV-based analysis</span>
            <span style="background: #E3F2FD; padding: 0.5rem 1rem; border-radius: 5px; margin-left: 0.5rem;">🗺️ Shapefile integration</span>
            <span style="background: #FFF3E0; padding: 0.5rem 1rem; border-radius: 5px; margin-left: 0.5rem;">📈 Trend detection</span>
        </div>
        <div style="margin-top: 2rem; text-align: left; background: #fafafa; padding: 1rem; border-radius: 5px;">
            <p><b>Expected CSV format:</b></p>
            <ul>
                <li><b>KEY</b> - Farm identifier (matches shapefile)</li>
                <li><b>median_height_2020</b> - Median height for 2020</li>
                <li><b>std_2020</b> - Standard deviation for 2020</li>
                <li><b>pixel_count_2020</b> - Pixel count for 2020</li>
                <li>... and similarly for 2021, 2022, 2023, 2024</li>
            </ul>
            <p><b>Shapefile:</b> Place in project_farm folder</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>Developed by Aman Chauhan (22BCE0476) | Internship Project at Varaha ClimateAg Private Limited</p>
    <p style="font-size: 0.8rem;">Features: CSV Analysis • Shapefile Integration • Farm Trend Detection • Interactive Maps</p>
</div>
""", unsafe_allow_html=True)
