# app.py - Complete CHM Dashboard with Shapefile-CSV Integration
import streamlit as st
import rasterio
from rasterio.plot import show
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
from folium.plugins import Draw, MarkerCluster, HeatMap
import geopandas as gpd
from shapely.geometry import Point, Polygon
from branca.colormap import LinearColormap
import json
import os
import re
from datetime import datetime
import base64
from io import BytesIO
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="CHM Farm Analysis Dashboard - Aman Chauhan",
    page_icon="🌳",
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
if 'chm_data' not in st.session_state:
    st.session_state.chm_data = {}
if 'shapefile_data' not in st.session_state:
    st.session_state.shapefile_data = None
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
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
CSV_FILENAME = "farm_data.csv"  # Change this to your CSV name
SHAPEFILE_NAME = "bg_monitoring_32.shp"  # Change this to your shapefile name

# Helper Functions
def extract_year_from_filename(filename):
    """Extract year from filename using regex pattern"""
    year_match = re.search(r'(19|20)\d{2}', filename)
    if year_match:
        return int(year_match.group())
    return None

@st.cache_data
def load_chm_tiff(file_path):
    """Load CHM TIFF file and extract data with caching"""
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)
            meta = src.meta
            bounds = src.bounds
            transform = src.transform
            
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
            
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                stats = {
                    'mean': float(np.mean(valid_data)),
                    'median': float(np.median(valid_data)),
                    'std': float(np.std(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'count': len(valid_data)
                }
            else:
                stats = None
            
            return {
                'data': data,
                'meta': meta,
                'bounds': bounds,
                'transform': transform,
                'stats': stats,
                'shape': data.shape
            }
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return None

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

def get_chm_value_at_point(data_obj, lat, lon):
    """Get CHM value at specific lat/lon point"""
    data = data_obj['data']
    transform = data_obj['transform']
    
    try:
        col, row = ~transform * (lon, lat)
        col, row = int(round(col)), int(round(row))
        
        if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
            value = data[row, col]
            if not np.isnan(value):
                return float(value)
    except:
        pass
    return None

def merge_shapefile_with_csv(shapefile_gdf, csv_df, key_column='KEY'):
    """Merge shapefile with CSV using KEY column"""
    try:
        # Ensure KEY column exists in both
        if key_column not in shapefile_gdf.columns:
            st.error(f"Column '{key_column}' not found in shapefile. Available: {list(shapefile_gdf.columns)}")
            return None
        
        if key_column not in csv_df.columns:
            st.error(f"Column '{key_column}' not found in CSV. Available: {list(csv_df.columns)}")
            return None
        
        # Merge on KEY
        merged = shapefile_gdf.merge(csv_df, on=key_column, how='inner')
        
        if len(merged) == 0:
            st.warning(f"No matching records found between shapefile and CSV on '{key_column}'")
            return None
        
        return merged
    except Exception as e:
        st.error(f"Error merging data: {str(e)}")
        return None

def extract_chm_values_for_farms(chm_data_dict, farm_gdf, years):
    """Extract CHM values for each farm from shapefile geometry"""
    results = []
    
    for idx, farm in farm_gdf.iterrows():
        # Get centroid from geometry
        centroid = farm.geometry.centroid
        lat = centroid.y
        lon = centroid.x
        
        farm_data = {
            'KEY': farm.get('KEY', idx),
            'latitude': lat,
            'longitude': lon
        }
        
        # Add all shapefile attributes
        for col in farm_gdf.columns:
            if col not in ['geometry', 'latitude', 'longitude']:
                farm_data[col] = farm[col]
        
        # Get CHM values for each year
        for year in years:
            if year in chm_data_dict:
                value = get_chm_value_at_point(chm_data_dict[year], lat, lon)
                farm_data[f'chm_{year}'] = value if value is not None else np.nan
        
        results.append(farm_data)
    
    return pd.DataFrame(results)

def analyze_farm_trends(farm_df, threshold=7.0):
    """Analyze farm trends"""
    results = []
    
    year_cols = [col for col in farm_df.columns if col.startswith('chm_')]
    years = sorted([int(col.split('_')[1]) for col in year_cols])
    
    for idx, row in farm_df.iterrows():
        values = [row[f'chm_{year}'] for year in years if f'chm_{year}' in row]
        valid_values = [v for v in values if not np.isnan(v)]
        
        if len(valid_values) >= 2:
            is_increasing = all(valid_values[i] <= valid_values[i+1] for i in range(len(valid_values)-1))
            is_decreasing = all(valid_values[i] >= valid_values[i+1] for i in range(len(valid_values)-1))
            is_low = valid_values[-1] < threshold
            
            growth_rate = ((valid_values[-1] - valid_values[0]) / valid_values[0] * 100) if valid_values[0] > 0 else 0
            
            results.append({
                'KEY': row.get('KEY', idx),
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'values': valid_values,
                'years': years[:len(valid_values)],
                'is_increasing': is_increasing,
                'is_decreasing': is_decreasing,
                'is_low': is_low,
                'growth_rate': growth_rate,
                'last_value': valid_values[-1] if valid_values else None,
                'first_value': valid_values[0] if valid_values else None,
                'avg_value': np.mean(valid_values) if valid_values else None,
                'farm_data': row.to_dict()
            })
    
    return pd.DataFrame(results)

def create_raster_image(data_obj, colormap='viridis', threshold=7.0):
    """Create raster image for map overlay"""
    data = data_obj['data']
    bounds = data_obj['bounds']
    
    fig, ax = plt.subplots(figsize=(12, 12), dpi=80)
    data_plot = np.ma.masked_where(np.isnan(data), data)
    
    im = ax.imshow(data_plot, 
                   extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                   cmap=colormap,
                   vmin=0,
                   vmax=25,
                   alpha=0.7,
                   interpolation='bilinear')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, label='Canopy Height (m)')
    cbar.ax.tick_params(labelsize=10)
    ax.axis('off')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=80, facecolor='white')
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_data

def create_farm_map(farm_analysis_df, chm_data_dict, selected_year, threshold=7.0):
    """Create interactive map with farms"""
    if not chm_data_dict or selected_year not in chm_data_dict:
        return None
    
    data_obj = chm_data_dict[selected_year]
    bounds = data_obj['bounds']
    center_lat = (bounds.top + bounds.bottom) / 2
    center_lon = (bounds.left + bounds.right) / 2
    
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
    
    # Add CHM overlay
    img_data = create_raster_image(data_obj, 'viridis', threshold)
    folium.raster_layers.ImageOverlay(
        image=f'data:image/png;base64,{img_data}',
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.6,
        name=f'CHM {selected_year}'
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
        
        popup_html = f"""
        <div style="font-size: 13px; font-family: Arial, sans-serif; max-width: 300px;">
            <b style="color: #2E7D32;">Farm KEY: {farm['KEY']}</b><br>
            <hr style="margin: 5px 0;">
            <b>Status:</b> {status_text}<br>
            <b>Current Height:</b> {farm['last_value']:.2f} m<br>
            <b>Growth Rate:</b> <span style="color: {'#2E7D32' if farm['growth_rate'] > 0 else '#F44336'};">{farm['growth_rate']:.1f}%</span><br>
            <b>Avg Height:</b> {farm['avg_value']:.2f} m<br>
            <hr style="margin: 5px 0;">
            <b>Yearly Values:</b><br>
            {''.join([f"{year}: {value:.2f}m<br>" for year, value in zip(farm['years'], farm['values'])])}
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
    
    folium.LayerControl(position='topright').add_to(m)
    
    return m

def create_trend_chart(farm_analysis_df, selected_key=None):
    """Create trend chart for farm(s)"""
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
                hovertemplate='Year: %{x}<br>Height: %{y:.2f}m<extra></extra>'
            ))
            
            # Add annotations for each point
            for year, value in zip(farm['years'], farm['values']):
                fig.add_annotation(
                    x=year,
                    y=value,
                    text=f"{value:.1f}m",
                    showarrow=True,
                    arrowhead=1,
                    ay=-20
                )
    else:
        # Show low and increasing farms
        filtered = farm_analysis_df[
            (farm_analysis_df['is_low']) & 
            (farm_analysis_df['is_increasing'])
        ].head(10)
        
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
                hovertemplate=f"KEY: {farm['KEY']}<br>Year: %{{x}}<br>Height: %{{y:.2f}}m<extra></extra>"
            ))
    
    fig.update_layout(
        title="Farm CHM Trends",
        xaxis_title="Year",
        yaxis_title="Canopy Height (m)",
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
            "Height Distribution",
            "Growth Rate Distribution",
            f"Low Farms (<{threshold}m)",
            "Increasing Farms Trend",
            "Top 10 Farms by Height"
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
    
    # 2. Height Distribution
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
    
    # 5. Increasing Farms Trend (average)
    increasing_farms = farm_analysis_df[farm_analysis_df['is_increasing']]
    if not increasing_farms.empty:
        years = sorted(farm_analysis_df['years'].iloc[0]) if not farm_analysis_df.empty else []
        avg_values = []
        for i in range(len(years)):
            vals = [farm['values'][i] for farm in increasing_farms.iloc if len(farm['values']) > i]
            avg_values.append(np.mean(vals))
        
        fig.add_trace(
            go.Scatter(
                x=years,
                y=avg_values,
                mode='lines+markers',
                name='Average Increasing',
                line=dict(color='#2E7D32', width=3),
                marker=dict(size=10)
            ),
            row=2, col=2
        )
    
    # 6. Top 10 Farms by Height
    top_farms = farm_analysis_df.nlargest(10, 'last_value')
    fig.add_trace(
        go.Bar(
            x=top_farms['KEY'].astype(str),
            y=top_farms['last_value'],
            text=top_farms['last_value'].apply(lambda x: f"{x:.1f}m"),
            textposition='outside',
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
    fig.update_xaxes(title_text="Year", row=2, col=2)
    fig.update_xaxes(title_text="Farm KEY", row=2, col=3)
    
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=3)
    fig.update_yaxes(title_text="Height (m)", row=2, col=1)
    fig.update_yaxes(title_text="Height (m)", row=2, col=2)
    fig.update_yaxes(title_text="Height (m)", row=2, col=3)
    
    return fig

def load_all_data():
    """Load all data from project_farm folder"""
    st.markdown("### 📂 Loading Data from Project Farm")
    
    # Check if folder exists
    if not os.path.exists(PROJECT_FOLDER):
        st.error(f"❌ Folder '{PROJECT_FOLDER}' not found!")
        return False
    
    # 1. Load CHM TIFF files
    st.markdown("**1. Loading CHM TIFF files...**")
    tiff_files = [f for f in os.listdir(PROJECT_FOLDER) if f.endswith(('.tif', '.tiff'))]
    
    if not tiff_files:
        st.warning(f"No TIFF files found in {PROJECT_FOLDER}")
    else:
        loaded_count = 0
        for filename in tiff_files:
            year = extract_year_from_filename(filename)
            if year:
                file_path = os.path.join(PROJECT_FOLDER, filename)
                data_obj = load_chm_tiff(file_path)
                if data_obj:
                    st.session_state.chm_data[year] = data_obj
                    loaded_count += 1
        st.success(f"✅ Loaded {loaded_count} CHM files")
    
    # 2. Load Shapefile
    st.markdown("**2. Loading Shapefile...**")
    shp_path = os.path.join(PROJECT_FOLDER, SHAPEFILE_NAME)
    if os.path.exists(shp_path):
        gdf = load_shapefile(shp_path)
        if gdf is not None:
            st.session_state.shapefile_data = gdf
            st.success(f"✅ Loaded shapefile with {len(gdf)} features")
            st.info(f"Shapefile columns: {list(gdf.columns)}")
        else:
            st.warning(f"⚠️ Could not load shapefile from {shp_path}")
    else:
        st.warning(f"⚠️ Shapefile not found: {shp_path}")
    
    # 3. Load CSV
    st.markdown("**3. Loading CSV data...**")
    csv_path = os.path.join(PROJECT_FOLDER, CSV_FILENAME)
    if os.path.exists(csv_path):
        df = load_csv_data(csv_path)
        if df is not None:
            st.session_state.csv_data = df
            st.success(f"✅ Loaded CSV with {len(df)} rows")
            st.info(f"CSV columns: {list(df.columns)}")
        else:
            st.warning(f"⚠️ Could not load CSV from {csv_path}")
    else:
        st.warning(f"⚠️ CSV not found: {csv_path}")
    
    st.session_state.data_loaded = True
    return True

# Main Title
st.markdown('<div class="main-header">🌳 CHM Farm Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #555;">Internship Project at Varaha ClimateAg Private Limited | Aman Chauhan (22BCE0476)</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📁 Data Loading")
    
    # Display current folder
    st.markdown(f"**Project Folder:** `{PROJECT_FOLDER}`")
    st.markdown(f"**CSV File:** `{CSV_FILENAME}`")
    st.markdown(f"**Shapefile:** `{SHAPEFILE_NAME}`")
    st.markdown("**Key Column:** `KEY` (used for matching)")
    
    # Load data button
    if st.button("🔄 Load All Data from Project Farm", type="primary", use_container_width=True):
        with st.spinner("Loading all data..."):
            success = load_all_data()
            if success:
                st.success("✅ All data loaded successfully!")
                st.rerun()
    
    st.markdown("---")
    
    # Show loaded status
    st.markdown("### 📊 Loaded Data Status")
    
    if st.session_state.chm_data:
        st.markdown(f'<div class="data-status status-loaded">✅ CHM Data: {len(st.session_state.chm_data)} years</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="data-status status-missing">⏳ CHM Data: Not loaded</div>', unsafe_allow_html=True)
    
    if st.session_state.shapefile_data is not None:
        st.markdown(f'<div class="data-status status-loaded">✅ Shapefile: {len(st.session_state.shapefile_data)} features</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="data-status status-missing">⏳ Shapefile: Not loaded</div>', unsafe_allow_html=True)
    
    if st.session_state.csv_data is not None:
        st.markdown(f'<div class="data-status status-loaded">✅ CSV Data: {len(st.session_state.csv_data)} rows</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="data-status status-missing">⏳ CSV Data: Not loaded</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.session_state.chm_data and st.session_state.shapefile_data is not None and st.session_state.csv_data is not None:
        st.markdown("## 🎯 Analysis Controls")
        
        # Merge and Analyze button
        if st.button("🔗 Merge & Analyze Farms", type="primary", use_container_width=True):
            with st.spinner("Merging and analyzing farms..."):
                # Merge shapefile with CSV
                merged = merge_shapefile_with_csv(
                    st.session_state.shapefile_data,
                    st.session_state.csv_data,
                    'KEY'
                )
                
                if merged is not None:
                    st.session_state.merged_data = merged
                    
                    # Extract CHM values
                    years = sorted(st.session_state.chm_data.keys())
                    farm_df = extract_chm_values_for_farms(
                        st.session_state.chm_data,
                        merged,
                        years
                    )
                    
                    # Analyze trends
                    analysis_df = analyze_farm_trends(farm_df, height_threshold)
                    st.session_state.farm_analysis = analysis_df
                    st.success(f"✅ Analyzed {len(analysis_df)} farms")
                    st.rerun()
        
        st.markdown("---")
        
        # Threshold
        height_threshold = st.slider(
            "Height Threshold (meters)",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.5,
            help="Farms below this height will be highlighted"
        )
        
        # Year selection
        available_years = sorted(st.session_state.chm_data.keys())
        if available_years:
            selected_year = st.selectbox(
                "Select Year to Display",
                options=available_years,
                format_func=lambda x: f"{x}",
                key="year_selector"
            )
        else:
            selected_year = None

# Main Content
if not st.session_state.data_loaded:
    st.info("📂 Click 'Load All Data from Project Farm' in the sidebar to start analysis.")
    
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: #f5f5f5; border-radius: 10px;">
        <h3>🌳 CHM Farm Analysis Dashboard</h3>
        <p style="color: #666;">Analyze canopy height trends for farms using CHM data</p>
        <div style="margin-top: 1rem;">
            <span style="background: #E8F5E9; padding: 0.5rem 1rem; border-radius: 5px;">📊 Farm-level analysis</span>
            <span style="background: #E3F2FD; padding: 0.5rem 1rem; border-radius: 5px; margin-left: 0.5rem;">🗺️ Shapefile integration</span>
            <span style="background: #FFF3E0; padding: 0.5rem 1rem; border-radius: 5px; margin-left: 0.5rem;">📈 Trend detection</span>
        </div>
        <div style="margin-top: 2rem; text-align: left; background: #fafafa; padding: 1rem; border-radius: 5px;">
            <p><b>Expected files in project_farm folder:</b></p>
            <ul>
                <li>Bangladesh_mosaic_2020.tif, 2021.tif, 2022.tif, 2023.tif, 2024.tif</li>
                <li>bg_monitoring_32.shp, .shx, .dbf, .prj, .cpg</li>
                <li>farm_data.csv (with KEY column matching shapefile)</li>
            </ul>
            <p><b>Matching Key:</b> The shapefile and CSV are joined on the 'KEY' column</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.farm_analysis is not None:
    analysis_df = st.session_state.farm_analysis
    
    # Summary statistics
    st.markdown('<div class="section-header">📊 Farm Analysis Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_farms = len(analysis_df)
    low_farms = len(analysis_df[analysis_df['is_low']])
    increasing_farms = len(analysis_df[analysis_df['is_increasing']])
    low_increasing = len(analysis_df[(analysis_df['is_low']) & (analysis_df['is_increasing'])])
    avg_height = analysis_df['last_value'].mean()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Total Farms</strong><br>
            <span style="font-size: 1.8rem; color: #2E7D32;">{total_farms}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #F44336;">
            <strong>🔴 Low Farms</strong><br>
            <span style="font-size: 1.8rem; color: #F44336;">{low_farms}</span>
            <br><span style="color: #666;">({low_farms/total_farms*100:.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #2196F3;">
            <strong>📈 Increasing</strong><br>
            <span style="font-size: 1.8rem; color: #2196F3;">{increasing_farms}</span>
            <br><span style="color: #666;">({increasing_farms/total_farms*100:.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #FFA500;">
            <strong>⚠️ Low & Increasing</strong><br>
            <span style="font-size: 1.8rem; color: #FFA500;">{low_increasing}</span>
            <br><span style="color: #666;">({low_increasing/total_farms*100:.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #9C27B0;">
            <strong>📊 Avg Height</strong><br>
            <span style="font-size: 1.8rem; color: #9C27B0;">{avg_height:.1f}m</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Farm Map", "📈 Trends & Analysis", "📊 Summary Dashboard", "📋 Data Table"])
    
    with tab1:
        st.subheader("Interactive Farm Map")
        
        if selected_year is not None and selected_year in st.session_state.chm_data:
            m = create_farm_map(
                analysis_df,
                st.session_state.chm_data,
                selected_year,
                height_threshold
            )
            
            if m:
                folium_static(m, width=750, height=600)
                
                st.markdown(f"""
                <div style="background: #E3F2FD; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem;">
                    <small>💡 <b>Legend:</b> 
                        🟢 Good & Increasing | 🔵 Good & Stable | 🟠 Low & Increasing | 🔴 Low & Not Increasing
                    </small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Select a year to display the map")
    
    with tab2:
        st.subheader("Farm Trend Analysis")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_option = st.selectbox(
                "Filter farms:",
                ["All Farms", "Low Farms (<7m)", "Increasing Farms", "Low & Increasing", "High Farms (>7m)"]
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
            else:
                filtered_df = analysis_df
            
            farm_options = filtered_df['KEY'].tolist()
            selected_key = st.selectbox(
                "Select farm by KEY:",
                options=farm_options,
                format_func=lambda x: f"KEY: {x} - Height: {filtered_df[filtered_df['KEY']==x]['last_value'].iloc[0]:.2f}m" if not filtered_df.empty and x in filtered_df['KEY'].values else str(x)
            )
        
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
                    
                    st.markdown("### 📋 Farm Details")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="farm-card {'critical-farm' if farm['is_low'] else ''}">
                            <strong>📊 Current Height</strong><br>
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
                        status = "🟢 Good" if not farm['is_low'] else "🔴 Critical"
                        trend = "📈 Increasing" if farm['is_increasing'] else "📉 Not Increasing"
                        st.markdown(f"""
                        <div class="farm-card {'critical-farm' if farm['is_low'] else ''}">
                            <strong>Status</strong><br>
                            {status} - {trend}
                        </div>
                        """, unsafe_allow_html=True)
    
    with tab3:
        st.subheader("Summary Dashboard")
        
        summary_fig = create_summary_dashboard(analysis_df, height_threshold)
        st.plotly_chart(summary_fig, use_container_width=True)
    
    with tab4:
        st.subheader("Farm Data Table")
        
        # Prepare display dataframe
        display_df = analysis_df.copy()
        display_df['values'] = display_df['values'].apply(
            lambda x: ', '.join([f"{v:.2f}" for v in x]) if x else 'N/A'
        )
        display_df['years'] = display_df['years'].apply(
            lambda x: ', '.join(map(str, x)) if x else 'N/A'
        )
        
        # Select columns to display
        cols_to_show = ['KEY', 'latitude', 'longitude', 'last_value', 'avg_value', 
                       'growth_rate', 'is_low', 'is_increasing', 'years', 'values']
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

# Footer
st.markdown("""
<div class="footer">
    <p>Developed by Aman Chauhan (22BCE0476) | Internship Project at Varaha ClimateAg Private Limited</p>
    <p style="font-size: 0.8rem;">Features: Shapefile-CSV Matching • Farm Trend Analysis • CHM Visualization • Interactive Maps</p>
</div>
""", unsafe_allow_html=True)
