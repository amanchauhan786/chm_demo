# app.py - Fixed version with better error handling and file upload
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
from folium.plugins import Draw
import geopandas as gpd
from shapely.geometry import Point
from branca.colormap import LinearColormap
import json
import os
import re
from datetime import datetime
import base64
from io import BytesIO
import tempfile
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="CHM Analysis Dashboard - Aman Chauhan",
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
    .point-card {
        background: #f5f5f5;
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-radius: 5px;
        border-left: 3px solid #2196F3;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        color: #666;
        border-top: 1px solid #ddd;
        margin-top: 2rem;
    }
    .file-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
    .file-loaded {
        background: #E8F5E9;
        border-left: 4px solid #4CAF50;
    }
    .upload-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 2px dashed #4CAF50;
        margin-bottom: 1rem;
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chm_data' not in st.session_state:
    st.session_state.chm_data = {}
if 'selected_points' not in st.session_state:
    st.session_state.selected_points = []
if 'year_data' not in st.session_state:
    st.session_state.year_data = {}
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'uploaded_file_names' not in st.session_state:
    st.session_state.uploaded_file_names = []

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
            
            # Handle NoData values
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
            
            # Calculate statistics
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                stats = {
                    'mean': float(np.mean(valid_data)),
                    'median': float(np.median(valid_data)),
                    'std': float(np.std(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'q1': float(np.percentile(valid_data, 25)),
                    'q3': float(np.percentile(valid_data, 75)),
                    'count': len(valid_data),
                    'total_pixels': data.size,
                    'no_data_pixels': data.size - len(valid_data)
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
        st.error(f"Error loading file: {str(e)}")
        return None

def create_raster_image(data_obj, colormap='viridis', threshold=6.0):
    """Create a raster image from CHM data for map overlay"""
    data = data_obj['data']
    bounds = data_obj['bounds']
    
    # Create figure with matplotlib
    fig, ax = plt.subplots(figsize=(12, 12), dpi=80)
    
    # Handle NaN values for plotting
    data_plot = np.ma.masked_where(np.isnan(data), data)
    
    # Plot with colormap
    im = ax.imshow(data_plot, 
                   extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                   cmap=colormap,
                   vmin=0,
                   vmax=30,
                   alpha=0.8,
                   interpolation='bilinear')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, label='Canopy Height (m)')
    cbar.ax.tick_params(labelsize=10)
    
    ax.axis('off')
    
    # Convert to base64 for folium overlay
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=80, facecolor='white')
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_data

def create_chm_map_with_overlay(data_obj, selected_year, threshold=6.0, color_scheme='viridis'):
    """Create a Folium map with CHM data as an image overlay"""
    # Get bounds
    bounds = data_obj['bounds']
    center_lat = (bounds.top + bounds.bottom) / 2
    center_lon = (bounds.left + bounds.right) / 2
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    # Add satellite basemap option
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite'
    ).add_to(m)
    
    # Create raster overlay
    img_data = create_raster_image(data_obj, color_scheme, threshold)
    
    # Add image overlay
    folium.raster_layers.ImageOverlay(
        image=f'data:image/png;base64,{img_data}',
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.75,
        name=f'CHM {selected_year}',
        interactive=True
    ).add_to(m)
    
    # Add selected points
    for point in st.session_state.selected_points:
        if point.get('year') == selected_year:
            color = 'red' if point.get('below_threshold', False) else 'green'
            folium.CircleMarker(
                location=[point['lat'], point['lon']],
                radius=8,
                popup=f"""
                <b>CHM Value:</b> {point.get('value', 'N/A'):.2f}m<br>
                <b>Year:</b> {selected_year}<br>
                <b>Status:</b> {'Below Threshold' if point.get('below_threshold', False) else 'Above Threshold'}
                """,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                weight=3,
                tooltip=f"CHM: {point.get('value', 'N/A'):.2f}m"
            ).add_to(m)
    
    # Add layer control
    folium.LayerControl(position='topright').add_to(m)
    
    # Add draw plugin for point selection
    draw = Draw(
        draw_options={
            'polyline': False,
            'polygon': False,
            'circle': False,
            'rectangle': False,
            'marker': True,
            'circlemarker': False
        },
        edit_options={'edit': False}
    )
    draw.add_to(m)
    
    return m

def get_point_value(data, transform, lat, lon):
    """Get CHM value at specific lat/lon point"""
    try:
        col, row = ~transform * (lon, lat)
        col, row = int(round(col)), int(round(row))
        
        if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
            value = data[row, col]
            if not np.isnan(value):
                return float(value)
        return None
    except:
        return None

def calculate_low_height_stats(data, threshold=6.0):
    """Calculate statistics for areas below threshold"""
    valid_data = data[~np.isnan(data)]
    if len(valid_data) == 0:
        return None
    
    low_mask = valid_data < threshold
    low_points = valid_data[low_mask]
    
    return {
        'total_points': len(valid_data),
        'low_points_count': len(low_points),
        'low_points_percentage': (len(low_points) / len(valid_data)) * 100,
        'low_mean': float(np.mean(low_points)) if len(low_points) > 0 else 0,
        'low_median': float(np.median(low_points)) if len(low_points) > 0 else 0,
        'low_min': float(np.min(low_points)) if len(low_points) > 0 else 0,
        'low_max': float(np.max(low_points)) if len(low_points) > 0 else 0
    }

def create_yearly_bar_charts(chm_data_dict, threshold=6.0):
    """Create comprehensive yearly bar charts"""
    years = sorted(chm_data_dict.keys())
    
    # Calculate metrics
    metrics = {}
    for year in years:
        data = chm_data_dict[year]['data'].flatten()
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) > 0:
            low_mask = valid_data < threshold
            metrics[year] = {
                'mean': np.mean(valid_data),
                'median': np.median(valid_data),
                'std': np.std(valid_data),
                'min': np.min(valid_data),
                'max': np.max(valid_data),
                'low_percentage': (np.sum(low_mask) / len(valid_data)) * 100,
                'count': len(valid_data)
            }
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Average Height", "Median Height", "Height Range",
            "Std Deviation", "% Below Threshold", "Sample Size"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # 1. Average Height
    fig.add_trace(
        go.Bar(
            x=years,
            y=[metrics[y]['mean'] for y in years],
            text=[f"{m:.2f}m" for m in [metrics[y]['mean'] for y in years]],
            textposition='outside',
            marker_color=['#2E7D32' if metrics[y]['mean'] > threshold else '#FF6B6B' for y in years],
            name='Avg Height'
        ),
        row=1, col=1
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=1, col=1)
    
    # 2. Median Height
    fig.add_trace(
        go.Bar(
            x=years,
            y=[metrics[y]['median'] for y in years],
            text=[f"{m:.2f}m" for m in [metrics[y]['median'] for y in years]],
            textposition='outside',
            marker_color='#2196F3',
            name='Median Height'
        ),
        row=1, col=2
    )
    
    # 3. Height Range
    fig.add_trace(
        go.Bar(
            x=years,
            y=[metrics[y]['max'] - metrics[y]['min'] for y in years],
            text=[f"{metrics[y]['min']:.1f}-{metrics[y]['max']:.1f}m" for y in years],
            textposition='outside',
            marker_color='#FF9800',
            name='Height Range'
        ),
        row=1, col=3
    )
    
    # 4. Std Deviation
    fig.add_trace(
        go.Bar(
            x=years,
            y=[metrics[y]['std'] for y in years],
            text=[f"{m:.2f}m" for m in [metrics[y]['std'] for y in years]],
            textposition='outside',
            marker_color='#9C27B0',
            name='Std Deviation'
        ),
        row=2, col=1
    )
    
    # 5. % Below Threshold
    fig.add_trace(
        go.Bar(
            x=years,
            y=[metrics[y]['low_percentage'] for y in years],
            text=[f"{m:.1f}%" for m in [metrics[y]['low_percentage'] for y in years]],
            textposition='outside',
            marker_color=['#FF6B6B' if metrics[y]['low_percentage'] > 30 else '#FFA07A' if metrics[y]['low_percentage'] > 20 else '#FFC107' for y in years],
            name='% Below Threshold'
        ),
        row=2, col=2
    )
    
    # 6. Sample Size
    fig.add_trace(
        go.Bar(
            x=years,
            y=[metrics[y]['count'] for y in years],
            text=[f"{m:,}" for m in [metrics[y]['count'] for y in years]],
            textposition='outside',
            marker_color='#607D8B',
            name='Sample Size'
        ),
        row=2, col=3
    )
    
    fig.update_layout(
        height=600,
        template='plotly_white',
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=2)
    fig.update_xaxes(title_text="Year", row=2, col=3)
    
    return fig

def create_temporal_trend_chart(chm_data_dict, threshold=6.0):
    """Create temporal trend chart"""
    years = sorted(chm_data_dict.keys())
    
    # Calculate metrics
    metrics = {}
    for year in years:
        data = chm_data_dict[year]['data'].flatten()
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) > 0:
            low_mask = valid_data < threshold
            metrics[year] = {
                'mean': np.mean(valid_data),
                'median': np.median(valid_data),
                'std': np.std(valid_data),
                'low_percentage': (np.sum(low_mask) / len(valid_data)) * 100
            }
    
    fig = go.Figure()
    
    # Add traces for different metrics
    fig.add_trace(go.Scatter(
        x=years,
        y=[metrics[y]['mean'] for y in years],
        mode='lines+markers',
        name='Mean Height',
        line=dict(color='#2E7D32', width=3),
        marker=dict(size=10),
        error_y=dict(
            type='data',
            array=[metrics[y]['std'] for y in years],
            visible=True
        ),
        hovertemplate='Year: %{x}<br>Mean: %{y:.2f}m<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=years,
        y=[metrics[y]['median'] for y in years],
        mode='lines+markers',
        name='Median Height',
        line=dict(color='#2196F3', width=2, dash='dash'),
        marker=dict(size=8),
        hovertemplate='Year: %{x}<br>Median: %{y:.2f}m<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=years,
        y=[metrics[y]['low_percentage'] for y in years],
        mode='lines+markers',
        name='% Below Threshold',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=10),
        yaxis='y2',
        fill='tozeroy',
        fillcolor='rgba(255,107,107,0.2)',
        hovertemplate='Year: %{x}<br>% Below Threshold: %{y:.1f}%<extra></extra>'
    ))
    
    # Add threshold line
    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Threshold: {threshold}m")
    
    fig.update_layout(
        title="Temporal Trends Analysis",
        xaxis_title="Year",
        yaxis_title="Height (m)",
        yaxis2=dict(
            title="Percentage (%)",
            overlaying='y',
            side='right'
        ),
        height=500,
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

# Main Title
st.markdown('<div class="main-header">🌳 Canopy Height Model (CHM) Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #555;">Internship Project at Varaha ClimateAg Private Limited | Aman Chauhan (22BCE0476)</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📁 Upload CHM Files")
    
    # File upload section - using smaller batch processing to avoid disconnects
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    # Option 1: Upload single file at a time (more reliable)
    st.markdown("**Option 1: Upload Single File**")
    single_file = st.file_uploader(
        "Upload a TIFF file",
        type=['tif', 'tiff'],
        key="single_upload",
        help="Upload one file at a time for reliability"
    )
    
    if single_file:
        year = extract_year_from_filename(single_file.name)
        if year:
            if st.button(f"📥 Load {single_file.name}", key="load_single"):
                with st.spinner(f"Loading {single_file.name}..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
                        tmp_file.write(single_file.getbuffer())
                        tmp_path = tmp_file.name
                    
                    data_obj = load_chm_tiff(tmp_path)
                    if data_obj:
                        st.session_state.chm_data[year] = data_obj
                        st.session_state.year_data[year] = {
                            'filename': single_file.name,
                            'size': single_file.size
                        }
                        st.success(f"✅ Loaded {single_file.name}")
                        os.unlink(tmp_path)
                        st.rerun()
        else:
            st.warning(f"Could not extract year from: {single_file.name}")
    
    st.markdown("---")
    st.markdown("**Option 2: Bulk Upload (Load Files from Folder)**")
    
    # Option 2: Use folder path for bulk loading (recommended for large files)
    folder_path = st.text_input(
        "Enter folder path:",
        value="./chm_tiffs" if os.path.exists("./chm_tiffs") else "",
        help="Path to folder containing TIFF files"
    )
    
    if folder_path and os.path.exists(folder_path):
        tiff_files = [f for f in os.listdir(folder_path) if f.endswith(('.tif', '.tiff'))]
        if tiff_files:
            st.markdown(f"**Found {len(tiff_files)} TIFF files:**")
            
            for filename in tiff_files[:5]:  # Show first 5
                year = extract_year_from_filename(filename)
                if year:
                    if year in st.session_state.chm_data:
                        st.markdown(f'<div class="file-status file-loaded">✅ {filename}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'📄 {filename}', unsafe_allow_html=True)
            
            if len(tiff_files) > 5:
                st.markdown(f"... and {len(tiff_files) - 5} more files")
            
            if st.button("📊 Load All Files from Folder", type="primary"):
                with st.spinner(f"Loading {len(tiff_files)} files..."):
                    loaded_count = 0
                    for filename in tiff_files:
                        year = extract_year_from_filename(filename)
                        if year:
                            file_path = os.path.join(folder_path, filename)
                            data_obj = load_chm_tiff(file_path)
                            if data_obj:
                                st.session_state.chm_data[year] = data_obj
                                st.session_state.year_data[year] = {
                                    'filename': filename,
                                    'path': file_path
                                }
                                loaded_count += 1
                    st.success(f"✅ Loaded {loaded_count} files")
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 🎯 Analysis Controls")
    
    # Height threshold
    height_threshold = st.slider(
        "Height Threshold (meters)",
        min_value=0.0,
        max_value=10.0,
        value=6.0,
        step=0.5,
        help="Areas below this height will be highlighted"
    )
    
    # Color scheme
    color_scheme = st.selectbox(
        "Map Color Scheme",
        options=['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'YlGn', 'RdYlGn'],
        index=0
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
        st.info("No CHM data loaded yet")
    
    st.markdown("---")
    st.markdown("## 📍 Point Selection")
    
    # Manual point addition
    if selected_year is not None and selected_year in st.session_state.chm_data:
        bounds = st.session_state.chm_data[selected_year]['bounds']
        center_lat = (bounds.top + bounds.bottom) / 2
        center_lon = (bounds.left + bounds.right) / 2
        
        col_lat, col_lon = st.columns(2)
        with col_lat:
            lat_input = st.number_input(
                "Latitude",
                value=center_lat,
                format="%.6f"
            )
        with col_lon:
            lon_input = st.number_input(
                "Longitude",
                value=center_lon,
                format="%.6f"
            )
        
        if st.button("➕ Add Point"):
            data_obj = st.session_state.chm_data[selected_year]
            value = get_point_value(data_obj['data'], data_obj['transform'], lat_input, lon_input)
            if value is not None:
                st.session_state.selected_points.append({
                    'lat': lat_input,
                    'lon': lon_input,
                    'year': selected_year,
                    'value': value,
                    'below_threshold': value < height_threshold
                })
                st.success(f"✅ Point added! CHM value: {value:.2f}m")
                st.rerun()
            else:
                st.error("❌ Could not get CHM value at this location")

# Main content
if not st.session_state.chm_data:
    st.info("📂 Please upload CHM TIFF files using the sidebar to begin analysis.")
    
    # Show placeholder with instructions
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: #f5f5f5; border-radius: 10px;">
        <h3>🌳 CHM Analysis Dashboard</h3>
        <p style="color: #666;">Upload CHM TIFF files to visualize and analyze canopy height data</p>
        <p style="font-size: 0.9rem; color: #999;">Expected files: Bangladesh_mosaic_2020.tif, Bangladesh_mosaic_2021.tif, ...</p>
        <div style="margin-top: 1rem;">
            <span style="background: #E8F5E9; padding: 0.5rem 1rem; border-radius: 5px;">📊 Multi-year analysis</span>
            <span style="background: #E3F2FD; padding: 0.5rem 1rem; border-radius: 5px; margin-left: 0.5rem;">🗺️ Raster overlay</span>
            <span style="background: #FFF3E0; padding: 0.5rem 1rem; border-radius: 5px; margin-left: 0.5rem;">📈 Interactive charts</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display loaded years
    loaded_years = sorted(st.session_state.chm_data.keys())
    st.success(f"✅ Loaded CHM data for years: {', '.join(map(str, loaded_years))}")
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-header">🗺️ CHM Raster Overlay Map</div>', unsafe_allow_html=True)
        
        if selected_year is not None and selected_year in st.session_state.chm_data:
            data_obj = st.session_state.chm_data[selected_year]
            
            # Create map with CHM overlay
            m = create_chm_map_with_overlay(
                data_obj, 
                selected_year, 
                height_threshold, 
                color_scheme
            )
            
            # Display map
            folium_static(m, width=750, height=550)
            
            # Map instructions
            st.markdown("""
            <div style="background: #E3F2FD; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem;">
                <small>💡 <b>Map Features:</b> Click on map to add points • Use +/- to zoom • Toggle layers in top-right</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Please select a year to display the map")
    
    with col2:
        st.markdown('<div class="section-header">📊 Yearly Statistics</div>', unsafe_allow_html=True)
        
        if selected_year is not None and selected_year in st.session_state.chm_data:
            data_obj = st.session_state.chm_data[selected_year]
            stats = data_obj['stats']
            
            if stats:
                # Statistics cards
                stats_data = {
                    "Mean Height": f"{stats['mean']:.2f} m",
                    "Median Height": f"{stats['median']:.2f} m",
                    "Std Deviation": f"{stats['std']:.2f} m",
                    "Min Height": f"{stats['min']:.2f} m",
                    "Max Height": f"{stats['max']:.2f} m",
                    "Valid Pixels": f"{stats['count']:,}"
                }
                
                for key, value in stats_data.items():
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{key}:</strong> {value}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Low height statistics
                low_stats = calculate_low_height_stats(data_obj['data'], height_threshold)
                if low_stats:
                    st.markdown(f"### 🔴 Below {height_threshold}m")
                    st.markdown(f"""
                    <div class="highlight-box">
                        <strong>📊 Points below threshold:</strong> {low_stats['low_points_count']:,} ({low_stats['low_points_percentage']:.1f}%)<br>
                        <strong>📈 Mean height:</strong> {low_stats['low_mean']:.2f} m<br>
                        <strong>📉 Min height:</strong> {low_stats['low_min']:.2f} m
                    </div>
                    """, unsafe_allow_html=True)
        
        # Selected points
        st.markdown("### 📍 Selected Points")
        
        if st.session_state.selected_points:
            # Filter points for current year
            current_year_points = [p for p in st.session_state.selected_points if p.get('year') == selected_year]
            
            if current_year_points:
                for i, point in enumerate(current_year_points):
                    color = "🔴" if point.get('below_threshold', False) else "🟢"
                    st.markdown(f"""
                    <div class="point-card">
                        {color} <strong>Point {i+1}</strong><br>
                        Lat: {point['lat']:.4f}, Lon: {point['lon']:.4f}<br>
                        CHM: {point['value']:.2f}m
                        {f"<span style='color:red;'> (Below {height_threshold}m)</span>" if point.get('below_threshold', False) else ""}
                    </div>
                    """, unsafe_allow_html=True)
                
                if st.button("🗑️ Clear All Points"):
                    st.session_state.selected_points = []
                    st.rerun()
            else:
                st.info(f"No points selected for {selected_year}")
        else:
            st.info("Click on the map or use manual input to add points")

# Charts Section
if st.session_state.chm_data and len(st.session_state.chm_data) >= 2:
    st.markdown('<div class="section-header">📊 Comprehensive Analysis & Charts</div>', unsafe_allow_html=True)
    
    # Create tabs for different charts
    tab1, tab2 = st.tabs(["📊 Yearly Comparison", "📈 Temporal Trends"])
    
    with tab1:
        st.subheader("Yearly CHM Comparison")
        
        # Create yearly bar charts
        yearly_fig = create_yearly_bar_charts(st.session_state.chm_data, height_threshold)
        st.plotly_chart(yearly_fig, use_container_width=True)
        
        # Summary table
        st.subheader("Yearly Summary Table")
        
        summary_data = []
        for year in sorted(st.session_state.chm_data.keys()):
            data_obj = st.session_state.chm_data[year]
            stats = data_obj['stats']
            if stats:
                low_stats = calculate_low_height_stats(data_obj['data'], height_threshold)
                summary_data.append({
                    'Year': year,
                    'Mean (m)': f"{stats['mean']:.2f}",
                    'Median (m)': f"{stats['median']:.2f}",
                    'Min (m)': f"{stats['min']:.2f}",
                    'Max (m)': f"{stats['max']:.2f}",
                    'Std Dev': f"{stats['std']:.2f}",
                    '% Below Threshold': f"{low_stats['low_points_percentage']:.1f}%" if low_stats else "N/A"
                })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Export option
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Yearly Summary CSV",
            data=csv,
            file_name=f"chm_yearly_summary_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("Temporal Trends Analysis")
        
        # Create temporal trend chart
        trend_fig = create_temporal_trend_chart(st.session_state.chm_data, height_threshold)
        st.plotly_chart(trend_fig, use_container_width=True)
        
        # Calculate growth statistics
        years = sorted(st.session_state.chm_data.keys())
        if len(years) >= 2:
            first_year = years[0]
            last_year = years[-1]
            
            data_first = st.session_state.chm_data[first_year]['data'].flatten()
            data_last = st.session_state.chm_data[last_year]['data'].flatten()
            
            valid_first = data_first[~np.isnan(data_first)]
            valid_last = data_last[~np.isnan(data_last)]
            
            if len(valid_first) > 0 and len(valid_last) > 0:
                growth = ((np.mean(valid_last) - np.mean(valid_first)) / np.mean(valid_first)) * 100
                
                low_first = np.sum(valid_first < height_threshold) / len(valid_first) * 100
                low_last = np.sum(valid_last < height_threshold) / len(valid_last) * 100
                low_change = low_last - low_first
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card" style="background: #E8F5E9;">
                        <strong>📈 Height Growth</strong><br>
                        <span style="font-size: 1.2rem; color: {'#2E7D32' if growth > 0 else '#F44336'};">
                            {growth:+.1f}%
                        </span><br>
                        <span style="color: #666;">({first_year} → {last_year})</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card" style="background: #FFF3E0;">
                        <strong>🔴 Low Area Change</strong><br>
                        <span style="font-size: 1.2rem; color: {'#2E7D32' if low_change < 0 else '#F44336'};">
                            {low_change:+.1f}%
                        </span><br>
                        <span style="color: #666;">Below threshold</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card" style="background: #E3F2FD;">
                        <strong>📊 Data Coverage</strong><br>
                        <span style="font-size: 1.2rem; color: #1565C0;">
                            {len(years)} years
                        </span><br>
                        <span style="color: #666;">Total analyzed</span>
                    </div>
                    """, unsafe_allow_html=True)

# Project Information
st.markdown('---')
st.markdown('<div class="section-header">📚 Project Context</div>', unsafe_allow_html=True)

with st.expander("📖 About This Analysis", expanded=False):
    st.markdown("""
    ### Canopy Height Model (CHM) Development
    
    This dashboard presents the results of the CHM project developed during the internship at 
    **Varaha ClimateAg Private Limited**. The project utilized deep learning methods to predict 
    vegetation canopy height from satellite imagery.
    
    ### Key Components:
    - **Data Source**: Sentinel-2 L2A satellite imagery (2018-2024)
    - **Model Architecture**: U-Net++ with ResNet34/50 backbone
    - **Reference Data**: GEDI L2B v3 spaceborne LiDAR
    - **Resolution**: 10-meter spatial resolution
    
    ### Methodology:
    1. Satellite data acquisition using STAC API pipeline
    2. Cloud filtering and temporal compositing
    3. Deep learning model training (U-Net++ architecture)
    4. Inference with tile artifact handling
    5. Temporal vegetation analysis
    6. Explainability using SHAP and Integrated Gradients
    
    ### Data Files:
    - `Bangladesh_mosaic_2020.tif` - CHM data for 2020
    - `Bangladesh_mosaic_2021.tif` - CHM data for 2021
    - `Bangladesh_mosaic_2022.tif` - CHM data for 2022
    - `Bangladesh_mosaic_2023.tif` - CHM data for 2023
    - `Bangladesh_mosaic_2024.tif` - CHM data for 2024
    """)

# Footer
st.markdown("""
<div class="footer">
    <p>Developed by Aman Chauhan (22BCE0476) | Internship Project at Varaha ClimateAg Private Limited</p>
    <p style="font-size: 0.8rem;">Data sources: Sentinel-2, GEDI, ICESat-2 | Tools: Python, PyTorch, QGIS, GDAL, Streamlit</p>
</div>
""", unsafe_allow_html=True)
