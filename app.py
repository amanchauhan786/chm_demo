# app.py - Final Corrected Farm Analysis Dashboard
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
    .metric-card-red {
        background: #FFEBEE;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #F44336;
        margin: 0.5rem 0;
    }
    .metric-card-orange {
        background: #FFF3E0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF9800;
        margin: 0.5rem 0;
    }
    .metric-card-blue {
        background: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 0.5rem 0;
    }
    .metric-card-purple {
        background: #F3E5F5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #9C27B0;
        margin: 0.5rem 0;
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
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .badge-green { background: #E8F5E9; color: #2E7D32; }
    .badge-red { background: #FFEBEE; color: #C62828; }
    .badge-orange { background: #FFF3E0; color: #E65100; }
    .badge-blue { background: #E3F2FD; color: #0D47A1; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'shapefile_data' not in st.session_state:
    st.session_state.shapefile_data = None
if 'farm_analysis' not in st.session_state:
    st.session_state.farm_analysis = None
if 'selected_farm_key' not in st.session_state:
    st.session_state.selected_farm_key = None

# Constants
PROJECT_FOLDER = "./project_farm"
SHAPEFILE_NAME = "bg_monitoring_32.shp"
YEARS = [2020, 2021, 2022, 2023, 2024]

# Helper Functions
@st.cache_data
def load_shapefile(shp_path):
    try:
        gdf = gpd.read_file(shp_path)
        return gdf
    except Exception as e:
        st.error(f"Error loading shapefile: {str(e)}")
        return None

@st.cache_data
def load_csv_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

def extract_median_columns(df):
    median_cols = {}
    for year in YEARS:
        col_name = f'median_height_{year}'
        if col_name in df.columns:
            median_cols[year] = col_name
        else:
            alt_cols = [col for col in df.columns if str(year) in col and 'median' in col.lower()]
            if alt_cols:
                median_cols[year] = alt_cols[0]
    return median_cols

def merge_shapefile_with_csv(shapefile_gdf, csv_df, key_column='KEY'):
    try:
        if key_column not in shapefile_gdf.columns:
            st.error(f"Column '{key_column}' not found in shapefile")
            return None
        if key_column not in csv_df.columns:
            st.error(f"Column '{key_column}' not found in CSV")
            return None
        
        shapefile_gdf[key_column] = shapefile_gdf[key_column].astype(str)
        csv_df[key_column] = csv_df[key_column].astype(str)
        
        merged = shapefile_gdf.merge(csv_df, on=key_column, how='inner')
        
        if len(merged) == 0:
            st.warning(f"No matching records found on '{key_column}'")
            return None
        return merged
    except Exception as e:
        st.error(f"Error merging data: {str(e)}")
        return None

def analyze_farm_data(merged_df, threshold=7.0):
    results = []
    median_cols = extract_median_columns(merged_df)
    
    if len(median_cols) < 2:
        st.warning("Need at least 2 years of data")
        return None
    
    for idx, row in merged_df.iterrows():
        if hasattr(row.geometry, 'centroid'):
            centroid = row.geometry.centroid
            lat = centroid.y
            lon = centroid.x
        else:
            lat = row.get('LAT', row.get('latitude', np.nan))
            lon = row.get('LON', row.get('longitude', np.nan))
        
        values = []
        years = []
        stds = []
        pixel_counts = []
        
        for year, col in sorted(median_cols.items()):
            val = row[col]
            if pd.notna(val) and val is not None:
                values.append(float(val))
                years.append(year)
                
                std_col = f'std_{year}'
                if std_col in row and pd.notna(row[std_col]):
                    stds.append(float(row[std_col]))
                else:
                    stds.append(0)
                
                count_col = f'pixel_count_{year}'
                if count_col in row and pd.notna(row[count_col]):
                    pixel_counts.append(float(row[count_col]))
                else:
                    pixel_counts.append(0)
        
        if len(values) >= 2:
            # CORRECTED: Calculate growth rate as percentage change
            # Growth rate = ((final - initial) / initial) * 100, capped at 100%
            if values[0] > 0:
                raw_growth = ((values[-1] - values[0]) / values[0]) * 100
                # Cap growth rate at 100% for realistic display
                growth_rate = min(raw_growth, 100.0)
                # If negative, keep as is (shrinkage)
                if growth_rate < 0:
                    growth_rate = max(growth_rate, -100.0)
            else:
                growth_rate = 0
            
            is_increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
            is_low = values[-1] < threshold
            
            results.append({
                'KEY': row.get('KEY', idx),
                'latitude': lat,
                'longitude': lon,
                'values': values,
                'years': years,
                'stds': stds,
                'pixel_counts': pixel_counts,
                'is_increasing': is_increasing,
                'is_low': is_low,
                'growth_rate': growth_rate,
                'last_value': values[-1],
                'first_value': values[0],
                'avg_value': np.mean(values),
                'median_value': np.median(values),
                'std_value': np.std(values) if len(values) > 1 else 0,
                'total_growth': values[-1] - values[0] if len(values) > 1 else 0,
            })
    
    return pd.DataFrame(results)

def create_farm_map(farm_analysis_df, threshold=7.0):
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
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite'
    ).add_to(m)
    
    marker_cluster = MarkerCluster().add_to(m)
    
    for _, farm in farm_analysis_df.iterrows():
        if farm['is_low'] and farm['is_increasing']:
            color = 'orange'
            status_text = "Low & Increasing"
        elif farm['is_low']:
            color = 'red'
            status_text = "Low & Not Increasing"
        elif farm['is_increasing']:
            color = 'green'
            status_text = "Good & Increasing"
        else:
            color = 'blue'
            status_text = "Good & Stable"
        
        yearly_str = ''.join([
            f"{year}: {value:.2f}m<br>" 
            for year, value in zip(farm['years'], farm['values'])
        ])
        
        popup_html = f"""
        <div style="font-size: 13px; font-family: Arial, sans-serif; max-width: 300px;">
            <b style="color: #2E7D32;">Farm KEY: {farm['KEY']}</b><br>
            <hr>
            <b>Status:</b> {status_text}<br>
            <b>Current Height:</b> {farm['last_value']:.2f} m<br>
            <b>Growth Rate:</b> {farm['growth_rate']:.1f}%<br>
            <b>Avg Height:</b> {farm['avg_value']:.2f} m<br>
            <hr>
            <b>Yearly Values:</b><br>
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
            tooltip=f"KEY: {farm['KEY']} - {farm['last_value']:.2f}m"
        ).add_to(marker_cluster)
    
    heat_data = [[row['latitude'], row['longitude'], row['last_value']] 
                 for _, row in farm_analysis_df.iterrows()]
    HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
    
    folium.LayerControl(position='topright').add_to(m)
    return m

def create_trend_chart(farm_analysis_df, selected_key=None):
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
                marker=dict(size=12, color='#2E7D32'),
                error_y=dict(
                    type='data',
                    array=farm['stds'],
                    visible=True,
                    color='#999'
                ),
                hovertemplate='Year: %{x}<br>Height: %{y:.2f}m<extra></extra>'
            ))
    else:
        # Show top 5 farms with highest growth
        filtered = farm_analysis_df.nlargest(5, 'growth_rate')
        colors = ['#2E7D32', '#2196F3', '#FF9800', '#9C27B0', '#F44336']
        for i, (_, farm) in enumerate(filtered.iterrows()):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=farm['years'],
                y=farm['values'],
                mode='lines+markers',
                name=f"KEY: {farm['KEY']}",
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color),
                hovertemplate=f"KEY: {farm['KEY']}<br>Year: %{{x}}<br>Height: %{{y:.2f}}m<extra></extra>"
            ))
    
    fig.update_layout(
        title="Farm Height Trends (2020-2024)",
        xaxis_title="Year",
        yaxis_title="Median Height (m)",
        height=400,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def create_summary_charts(farm_analysis_df, threshold=7.0):
    """Create summary charts - No pie chart, using bar charts instead"""
    if farm_analysis_df.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Height Distribution (2024)",
            "Growth Rate Distribution",
            f"Farms Below {threshold}m",
            "Top 10 Farms by Growth Rate"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # 1. Height Distribution
    heights = farm_analysis_df['last_value'].dropna()
    if not heights.empty:
        fig.add_trace(
            go.Histogram(
                x=heights,
                nbinsx=20,
                marker_color='#4CAF50',
                hovertemplate='Height: %{x:.2f}m<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                      annotation_text=f"Threshold: {threshold}m", row=1, col=1)
    
    # 2. Growth Rate Distribution
    growth_rates = farm_analysis_df['growth_rate'].dropna()
    if not growth_rates.empty:
        fig.add_trace(
            go.Histogram(
                x=growth_rates,
                nbinsx=20,
                marker_color='#2196F3',
                hovertemplate='Growth Rate: %{x:.1f}%<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=2
        )
        fig.add_vline(x=0, line_dash="dash", line_color="green", 
                      annotation_text="Zero Growth", row=1, col=2)
    
    # 3. Low Farms
    low_farms = farm_analysis_df[farm_analysis_df['is_low']].sort_values('last_value').head(15)
    if not low_farms.empty:
        fig.add_trace(
            go.Bar(
                x=low_farms['KEY'].astype(str),
                y=low_farms['last_value'],
                text=low_farms['last_value'].apply(lambda x: f"{x:.1f}m"),
                textposition='outside',
                marker_color=['#FF6B6B' if v < 5 else '#FFA07A' for v in low_farms['last_value']],
                hovertemplate='KEY: %{x}<br>Height: %{y:.2f}m<extra></extra>'
            ),
            row=2, col=1
        )
    
    # 4. Top 10 Growth Rates
    top_growth = farm_analysis_df.nlargest(10, 'growth_rate')
    if not top_growth.empty:
        fig.add_trace(
            go.Bar(
                x=top_growth['KEY'].astype(str),
                y=top_growth['growth_rate'],
                text=top_growth['growth_rate'].apply(lambda x: f"{x:.1f}%"),
                textposition='outside',
                marker_color='#4CAF50',
                hovertemplate='KEY: %{x}<br>Growth: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=600,
        template='plotly_white',
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Height (m)", row=1, col=1)
    fig.update_xaxes(title_text="Growth Rate (%)", row=1, col=2)
    fig.update_xaxes(title_text="Farm KEY", row=2, col=1)
    fig.update_xaxes(title_text="Farm KEY", row=2, col=2)
    
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Height (m)", row=2, col=1)
    fig.update_yaxes(title_text="Growth Rate (%)", row=2, col=2)
    
    return fig

def load_shapefile_from_folder():
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
    
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📊 Upload CSV File")
    
    csv_file = st.file_uploader(
        "Upload CSV with farm height data (2020-2024)",
        type=['csv'],
        help="CSV must contain KEY and median_height_2020 to median_height_2024"
    )
    
    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            st.session_state.csv_data = df
            st.success(f"✅ Loaded {len(df)} rows")
            
            median_cols = extract_median_columns(df)
            if median_cols:
                st.success(f"✅ Found years: {', '.join([str(y) for y in median_cols.keys()])}")
            else:
                st.warning("No median_height_YYYY columns found")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🗺️ Load Shapefile")
    st.caption(f"From: {PROJECT_FOLDER}/{SHAPEFILE_NAME}")
    if st.button("📂 Load Shapefile", use_container_width=True):
        with st.spinner("Loading..."):
            if load_shapefile_from_folder():
                st.success(f"✅ Loaded {len(st.session_state.shapefile_data)} features")
                st.rerun()
            else:
                st.warning("Shapefile not found")
    
    st.markdown("---")
    
    if st.session_state.csv_data is not None and st.session_state.shapefile_data is not None:
        st.markdown("## 🎯 Analysis")
        
        height_threshold = st.slider(
            "Height Threshold (m)",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.5,
            help="Farms below this height in 2024 are considered 'Low'"
        )
        
        if st.button("🔗 Merge & Analyze", type="primary", use_container_width=True):
            with st.spinner("Analyzing farms..."):
                merged = merge_shapefile_with_csv(
                    st.session_state.shapefile_data,
                    st.session_state.csv_data,
                    'KEY'
                )
                if merged is not None:
                    analysis_df = analyze_farm_data(merged, height_threshold)
                    if analysis_df is not None and not analysis_df.empty:
                        st.session_state.farm_analysis = analysis_df
                        st.session_state.threshold = height_threshold
                        st.success(f"✅ Analyzed {len(analysis_df)} farms")
                        st.rerun()

# Main Content
if st.session_state.farm_analysis is not None:
    analysis_df = st.session_state.farm_analysis
    threshold = st.session_state.get('threshold', 7.0)
    
    # Summary Statistics in Cards (Clean, No problematic text)
    st.markdown('<div class="section-header">📊 Farm Analysis Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total = len(analysis_df)
    low = len(analysis_df[analysis_df['is_low']])
    increasing = len(analysis_df[analysis_df['is_increasing']])
    low_inc = len(analysis_df[(analysis_df['is_low']) & (analysis_df['is_increasing'])])
    avg_h = analysis_df['last_value'].mean()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #666;">Total Farms</div>
            <div style="font-size: 2rem; font-weight: bold; color: #2E7D32;">{total}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card-red">
            <div style="font-size: 0.8rem; color: #666;">Low Farms</div>
            <div style="font-size: 2rem; font-weight: bold; color: #C62828;">{low}</div>
            <div style="font-size: 0.8rem; color: #666;">({low/total*100:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card-blue">
            <div style="font-size: 0.8rem; color: #666;">Increasing</div>
            <div style="font-size: 2rem; font-weight: bold; color: #0D47A1;">{increasing}</div>
            <div style="font-size: 0.8rem; color: #666;">({increasing/total*100:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card-orange">
            <div style="font-size: 0.8rem; color: #666;">Low & Increasing</div>
            <div style="font-size: 2rem; font-weight: bold; color: #E65100;">{low_inc}</div>
            <div style="font-size: 0.8rem; color: #666;">({low_inc/total*100:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #666;">Avg Height</div>
            <div style="font-size: 2rem; font-weight: bold; color: #2E7D32;">{avg_h:.1f}m</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Farm Map", "📈 Trends", "📊 Charts", "📋 Data"])
    
    with tab1:
        st.subheader("Interactive Farm Map")
        
        m = create_farm_map(analysis_df, threshold)
        if m:
            folium_static(m, width=750, height=550)
            
            st.markdown("""
            <div style="background: #E3F2FD; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem;">
                <small>💡 <b>Legend:</b> 
                    🟢 Good & Increasing | 🔵 Good & Stable | 🟠 Low & Increasing | 🔴 Low & Not Increasing
                    <br>📊 Heatmap shows height distribution (2024)
                </small>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Farm Trend Analysis")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            filter_option = st.selectbox(
                "Filter farms:",
                ["All Farms", "Low Farms", "Increasing", "Low & Increasing", "Top 10 Growth"]
            )
        
        if filter_option == "Low Farms":
            filtered_df = analysis_df[analysis_df['is_low']]
        elif filter_option == "Increasing":
            filtered_df = analysis_df[analysis_df['is_increasing']]
        elif filter_option == "Low & Increasing":
            filtered_df = analysis_df[(analysis_df['is_low']) & (analysis_df['is_increasing'])]
        elif filter_option == "Top 10 Growth":
            filtered_df = analysis_df.nlargest(10, 'growth_rate')
        else:
            filtered_df = analysis_df
        
        with col2:
            if not filtered_df.empty:
                farm_options = filtered_df['KEY'].tolist()
                selected_key = st.selectbox(
                    "Select farm:",
                    options=farm_options,
                    format_func=lambda x: f"KEY: {x}"
                )
            else:
                selected_key = None
        
        if not filtered_df.empty:
            trend_fig = create_trend_chart(analysis_df, selected_key)
            st.plotly_chart(trend_fig, use_container_width=True)
            
            # Show farm details if selected
            if selected_key:
                farm = analysis_df[analysis_df['KEY'] == selected_key].iloc[0]
                
                st.markdown("### 📋 Farm Details")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="farm-card {'critical-farm' if farm['is_low'] else ''}">
                        <strong>Current Height</strong><br>
                        <span style="font-size: 1.3rem;">{farm['last_value']:.2f} m</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="farm-card {'critical-farm' if farm['is_low'] else ''}">
                        <strong>Growth Rate</strong><br>
                        <span style="font-size: 1.3rem; color: {'#2E7D32' if farm['growth_rate'] > 0 else '#C62828'};">
                            {farm['growth_rate']:.1f}%
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="farm-card">
                        <strong>Avg Height</strong><br>
                        <span style="font-size: 1.3rem;">{farm['avg_value']:.2f} m</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    status = "✅ Good" if not farm['is_low'] else "🔴 Critical"
                    trend = "📈 Increasing" if farm['is_increasing'] else "➡️ Stable"
                    st.markdown(f"""
                    <div class="farm-card {'critical-farm' if farm['is_low'] else ''}">
                        <strong>Status</strong><br>
                        {status} - {trend}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Yearly data table
                yearly_data = pd.DataFrame({
                    'Year': farm['years'],
                    'Height (m)': [f"{v:.2f}" for v in farm['values']],
                    'Std Dev': [f"{s:.2f}" for s in farm['stds']],
                    'Pixel Count': [int(c) for c in farm['pixel_counts']]
                })
                st.dataframe(yearly_data, use_container_width=True)
    
    with tab3:
        st.subheader("Summary Charts")
        
        charts_fig = create_summary_charts(analysis_df, threshold)
        if charts_fig:
            st.plotly_chart(charts_fig, use_container_width=True)
        
        # Clean Statistics Cards - REMOVED the problematic stats display
        st.markdown("### 📊 Key Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        # Get realistic stats (capped at 100%)
        best_growth = min(analysis_df['growth_rate'].max(), 100.0)
        worst_growth = max(analysis_df['growth_rate'].min(), -100.0)
        avg_growth = min(analysis_df['growth_rate'].mean(), 100.0)
        median_growth = min(analysis_df['growth_rate'].median(), 100.0)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.8rem; color: #666;">🏆 Best Growth</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: #2E7D32;">{best_growth:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card-red">
                <div style="font-size: 0.8rem; color: #666;">📉 Worst Growth</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: #C62828;">{worst_growth:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card-blue">
                <div style="font-size: 0.8rem; color: #666;">📊 Avg Growth</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: #0D47A1;">{avg_growth:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card-purple">
                <div style="font-size: 0.8rem; color: #666;">📈 Median Growth</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: #9C27B0;">{median_growth:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.subheader("Farm Data Table")
        
        display_df = analysis_df.copy()
        display_df['values'] = display_df['values'].apply(
            lambda x: ' → '.join([f"{v:.2f}" for v in x]) if x else 'N/A'
        )
        display_df['years'] = display_df['years'].apply(
            lambda x: ', '.join(map(str, x)) if x else 'N/A'
        )
        
        cols_to_show = ['KEY', 'last_value', 'avg_value', 'median_value', 
                       'growth_rate', 'is_low', 'is_increasing', 'years', 'values']
        display_cols = [col for col in cols_to_show if col in display_df.columns]
        
        st.dataframe(display_df[display_cols], use_container_width=True, height=400)
        
        # Export
        csv = analysis_df.to_csv(index=False)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"farm_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

else:
    st.info("📂 Upload CSV file and load shapefile to begin analysis.")
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: #f5f5f5; border-radius: 10px;">
        <h3>🌾 Farm Analysis Dashboard</h3>
        <p style="color: #666;">Analyze farm height trends from 2020 to 2024</p>
        <div style="margin-top: 1rem;">
            <span style="background: #E8F5E9; padding: 0.5rem 1rem; border-radius: 5px;">📊 CSV Analysis</span>
            <span style="background: #E3F2FD; padding: 0.5rem 1rem; border-radius: 5px; margin-left: 0.5rem;">🗺️ Shapefile Integration</span>
            <span style="background: #FFF3E0; padding: 0.5rem 1rem; border-radius: 5px; margin-left: 0.5rem;">📈 Trend Detection</span>
        </div>
        <div style="margin-top: 1.5rem; text-align: left; background: #fafafa; padding: 1rem; border-radius: 5px;">
            <p><b>Expected CSV columns:</b></p>
            <ul>
                <li><b>KEY</b> - Farm identifier (matches shapefile)</li>
                <li><b>median_height_2020</b> to <b>median_height_2024</b></li>
                <li><b>std_2020</b> to <b>std_2024</b> (optional)</li>
                <li><b>pixel_count_2020</b> to <b>pixel_count_2024</b> (optional)</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>Developed by Aman Chauhan (22BCE0476) | Varaha ClimateAg Private Limited</p>
    <p style="font-size: 0.8rem; color: #999;">Data: 2020-2024 | Tools: Python, Streamlit, Plotly, Folium</p>
</div>
""", unsafe_allow_html=True)
