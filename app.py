# app_enhanced.py
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
from folium.plugins import Draw, HeatMap
import geopandas as gpd
from shapely.geometry import Point
from branca.colormap import LinearColormap
import json
import os
import re
from datetime import datetime
import base64
from io import BytesIO
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
    .legend-container {
        background: white;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ddd;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chm_data' not in st.session_state:
    st.session_state.chm_data = {}
if 'selected_points' not in st.session_state:
    st.session_state.selected_points = []
if 'chm_stats' not in st.session_state:
    st.session_state.chm_stats = {}
if 'year_data' not in st.session_state:
    st.session_state.year_data = {}
if 'map_center' not in st.session_state:
    st.session_state.map_center = [23.5, 77.5]  # Default for Bangladesh
if 'map_zoom' not in st.session_state:
    st.session_state.map_zoom = 8

# Helper Functions
def extract_year_from_filename(filename):
    """Extract year from filename using regex pattern"""
    year_match = re.search(r'(19|20)\d{2}', filename)
    if year_match:
        return int(year_match.group())
    return None

def load_chm_tiff(file_path):
    """Load CHM TIFF file and extract data"""
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
        st.error(f"Error loading {file_path}: {str(e)}")
        return None

def create_chm_raster_layer(data_obj, colormap='viridis', threshold=6.0):
    """
    Create a raster layer for Folium map using matplotlib
    This creates an image overlay from the CHM data
    """
    data = data_obj['data']
    bounds = data_obj['bounds']
    
    # Create figure with matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Handle NaN values for plotting
    data_plot = np.ma.masked_where(np.isnan(data), data)
    
    # Plot with colormap
    im = ax.imshow(data_plot, 
                   extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                   cmap=colormap,
                   vmin=0,
                   vmax=30,
                   alpha=0.7)
    
    ax.axis('off')
    
    # Convert to base64 for folium overlay
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_data, (bounds.left, bounds.right, bounds.bottom, bounds.top)

def create_chm_overlay_map(data_obj, selected_year, threshold=6.0, color_scheme='Viridis'):
    """
    Create a Folium map with CHM data as an image overlay
    """
    # Get bounds
    bounds = data_obj['bounds']
    center_lat = (bounds.top + bounds.bottom) / 2
    center_lon = (bounds.left + bounds.right) / 2
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles='OpenStreetMap'
    )
    
    # Create raster overlay
    img_data, extent = create_chm_raster_layer(data_obj, color_scheme, threshold)
    
    # Add image overlay
    folium.raster_layers.ImageOverlay(
        image=f'data:image/png;base64,{img_data}',
        bounds=[[extent[2], extent[0]], [extent[3], extent[1]]],
        opacity=0.7,
        name=f'CHM {selected_year}'
    ).add_to(m)
    
    # Add selected points
    for point in st.session_state.selected_points:
        if point.get('year') == selected_year:
            color = 'red' if point.get('below_threshold', False) else 'green'
            folium.CircleMarker(
                location=[point['lat'], point['lon']],
                radius=8,
                popup=f"CHM: {point.get('value', 'N/A'):.2f}m",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                weight=3
            ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
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

def create_height_distribution_chart(chm_data_dict, years_to_compare, threshold=6.0):
    """Create height distribution bar chart"""
    fig = go.Figure()
    
    for year in years_to_compare:
        if year in chm_data_dict:
            data = chm_data_dict[year]['data'].flatten()
            valid_data = data[~np.isnan(data)]
            
            fig.add_trace(go.Histogram(
                x=valid_data,
                name=f"{year}",
                nbinsx=30,
                opacity=0.7,
                histnorm='probability density'
            ))
    
    fig.update_layout(
        title="Canopy Height Distribution",
        xaxis_title="Height (m)",
        yaxis_title="Density",
        height=400,
        template='plotly_white',
        barmode='overlay',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Add threshold line
    fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Threshold: {threshold}m")
    
    return fig

def create_yearly_bar_chart(chm_data_dict, threshold=6.0):
    """Create comprehensive yearly comparison bar chart"""
    years = sorted(chm_data_dict.keys())
    
    # Calculate metrics for each year
    avg_heights = []
    med_heights = []
    min_heights = []
    max_heights = []
    low_percentages = []
    std_heights = []
    
    for year in years:
        data = chm_data_dict[year]['data'].flatten()
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) > 0:
            avg_heights.append(np.mean(valid_data))
            med_heights.append(np.median(valid_data))
            min_heights.append(np.min(valid_data))
            max_heights.append(np.max(valid_data))
            std_heights.append(np.std(valid_data))
            
            low_mask = valid_data < threshold
            low_percentages.append((np.sum(low_mask) / len(valid_data)) * 100)
        else:
            avg_heights.append(0)
            med_heights.append(0)
            min_heights.append(0)
            max_heights.append(0)
            std_heights.append(0)
            low_percentages.append(0)
    
    # Create subplot with multiple bar charts
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Average Canopy Height by Year",
            "Height Range (Min to Max)",
            "Percentage Below Threshold",
            "Height Distribution (Box Plot)"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # 1. Average Height Bar Chart
    fig.add_trace(
        go.Bar(
            x=years,
            y=avg_heights,
            name='Avg Height',
            text=[f"{h:.2f}m" for h in avg_heights],
            textposition='outside',
            marker_color=['#2E7D32' if h > threshold else '#FF6B6B' for h in avg_heights],
            hovertemplate='Year: %{x}<br>Avg Height: %{y:.2f}m<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add threshold line to avg height chart
    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Threshold: {threshold}m", row=1, col=1)
    
    # 2. Height Range (Min to Max)
    fig.add_trace(
        go.Bar(
            x=years,
            y=[max_h - min_h for max_h, min_h in zip(max_heights, min_heights)],
            name='Height Range',
            text=[f"{max_h:.1f}-{min_h:.1f}m" for max_h, min_h in zip(max_heights, min_heights)],
            textposition='outside',
            marker_color='#2196F3',
            hovertemplate='Year: %{x}<br>Range: %{y:.2f}m<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Percentage Below Threshold
    fig.add_trace(
        go.Bar(
            x=years,
            y=low_percentages,
            name='% Below Threshold',
            text=[f"{p:.1f}%" for p in low_percentages],
            textposition='outside',
            marker_color=['#FF6B6B' if p > 30 else '#FFA07A' if p > 20 else '#FFC107' for p in low_percentages],
            hovertemplate='Year: %{x}<br>Below Threshold: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Box plot of height distribution
    for year in years:
        data = chm_data_dict[year]['data'].flatten()
        valid_data = data[~np.isnan(data)]
        
        fig.add_trace(
            go.Box(
                y=valid_data,
                name=str(year),
                boxmean='sd',
                marker_color='#4CAF50',
                hovertemplate='Year: %{x}<br>Height: %{y:.2f}m<extra></extra>'
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=700,
        template='plotly_white',
        showlegend=False,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(title_text="Height (m)", row=1, col=1)
    fig.update_yaxes(title_text="Height Range (m)", row=1, col=2)
    fig.update_yaxes(title_text="Percentage (%)", row=2, col=1)
    fig.update_yaxes(title_text="Height (m)", row=2, col=2)
    
    return fig

def create_temporal_heatmap(chm_data_dict):
    """Create a heatmap showing CHM changes over time"""
    years = sorted(chm_data_dict.keys())
    
    # Sample points for heatmap
    sample_points = []
    for year in years:
        data = chm_data_dict[year]['data']
        bounds = chm_data_dict[year]['bounds']
        
        # Sample at regular intervals
        lat_step = (bounds.top - bounds.bottom) / 20
        lon_step = (bounds.right - bounds.left) / 20
        
        for i in range(20):
            for j in range(20):
                lat = bounds.bottom + i * lat_step
                lon = bounds.left + j * lon_step
                sample_points.append({
                    'year': year,
                    'latitude': lat,
                    'longitude': lon
                })
    
    return pd.DataFrame(sample_points)

# Main Title
st.markdown('<div class="main-header">🌳 Canopy Height Model (CHM) Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #555;">Internship Project at Varaha ClimateAg Private Limited | Aman Chauhan (22BCE0476)</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📁 CHM Data Management")
    
    # Option to use existing folder
    folder_path = st.text_input(
        "Enter folder path with CHM TIFF files:",
        value="./chm_tiffs" if os.path.exists("./chm_tiffs") else "",
        help="Path to folder containing Bangladesh_mosaic_YYYY.tif files"
    )
    
    if folder_path and os.path.exists(folder_path):
        # List all TIFF files
        tiff_files = [f for f in os.listdir(folder_path) if f.endswith(('.tif', '.tiff'))]
        
        if tiff_files:
            st.markdown(f"**Found {len(tiff_files)} TIFF files:**")
            
            # Display files with year
            year_files = {}
            for filename in tiff_files:
                year = extract_year_from_filename(filename)
                if year:
                    year_files[year] = filename
            
            if year_files:
                if st.button("📊 Load All CHM Files", type="primary"):
                    with st.spinner("Loading CHM TIFF files..."):
                        loaded_count = 0
                        for year, filename in year_files.items():
                            file_path = os.path.join(folder_path, filename)
                            data_obj = load_chm_tiff(file_path)
                            if data_obj:
                                st.session_state.chm_data[year] = data_obj
                                st.session_state.year_data[year] = {
                                    'filename': filename,
                                    'path': file_path
                                }
                                loaded_count += 1
                        st.success(f"✅ Successfully loaded {loaded_count} files")
                        st.rerun()
                
                # Show loaded status
                for year, filename in year_files.items():
                    if year in st.session_state.chm_data:
                        st.markdown(f'<div class="file-status file-loaded">✅ {filename}</div>', unsafe_allow_html=True)
            else:
                st.warning("No valid year found in filenames")
        else:
            st.info("No TIFF files found in the specified folder")
    else:
        if folder_path:
            st.error(f"❌ Folder not found: {folder_path}")
        else:
            st.info("Enter folder path to load CHM files")
    
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
        "Color Scheme",
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

# Main content - only show if data is loaded
if not st.session_state.chm_data:
    st.info("📂 Please load CHM TIFF files to begin analysis. Enter folder path in sidebar and click 'Load All CHM Files'.")
    
    # Show placeholder
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: #f5f5f5; border-radius: 10px;">
        <h3>🌳 CHM Analysis Dashboard</h3>
        <p style="color: #666;">Load CHM TIFF files to visualize and analyze canopy height data</p>
        <p style="font-size: 0.9rem; color: #999;">Expected files: Bangladesh_mosaic_2020.tif, Bangladesh_mosaic_2021.tif, ...</p>
        <p style="font-size: 0.9rem; color: #999;">Files will be displayed as raster layers on the map</p>
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
        
        # Create and display map with CHM raster overlay
        if selected_year is not None and selected_year in st.session_state.chm_data:
            data_obj = st.session_state.chm_data[selected_year]
            
            # Create map with CHM overlay
            m = create_chm_overlay_map(
                data_obj, 
                selected_year, 
                height_threshold, 
                color_scheme
            )
            
            # Display map
            folium_static(m, width=700, height=550)
            
            # Legend and info
            st.markdown(f"""
            <div class="legend-container">
                <small>
                    <span style="display: inline-block; width: 20px; height: 20px; background: red; border-radius: 50%;"></span> 
                    Points below {height_threshold}m &nbsp;&nbsp;
                    <span style="display: inline-block; width: 20px; height: 20px; background: green; border-radius: 50%;"></span> 
                    Points above {height_threshold}m &nbsp;&nbsp;
                    <span style="display: inline-block; width: 20px; height: 20px; background: rgba(0,0,0,0.5);"></span> 
                    CHM Raster Layer (Opacity: 0.7)
                </small>
            </div>
            """, unsafe_allow_html=True)
            
            # Instructions
            st.markdown("""
            <div style="background: #E3F2FD; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem;">
                <small>💡 Click on the map to add points for analysis. The CHM raster layer shows canopy height distribution.</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Manual point addition
            st.markdown("### 📍 Add Point Manually")
            bounds = data_obj['bounds']
            center_lat = (bounds.top + bounds.bottom) / 2
            center_lon = (bounds.left + bounds.right) / 2
            
            col_lat, col_lon = st.columns(2)
            with col_lat:
                lat_input = st.number_input(
                    "Latitude",
                    value=center_lat,
                    format="%.6f",
                    key="lat_input"
                )
            with col_lon:
                lon_input = st.number_input(
                    "Longitude",
                    value=center_lon,
                    format="%.6f",
                    key="lon_input"
                )
            
            if st.button("➕ Add Point", key="add_point_btn"):
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
        else:
            st.warning("Please select a year to display the map")
    
    with col2:
        st.markdown('<div class="section-header">📊 Yearly Statistics</div>', unsafe_allow_html=True)
        
        # Display statistics for selected year
        if selected_year is not None and selected_year in st.session_state.chm_data:
            data_obj = st.session_state.chm_data[selected_year]
            stats = data_obj['stats']
            
            if stats:
                st.markdown(f"### 📊 {selected_year} Statistics")
                
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
                    st.markdown(f"### 🔴 Below {height_threshold}m Analysis")
                    st.markdown(f"""
                    <div class="highlight-box">
                        <strong>📊 Points below {height_threshold}m:</strong> {low_stats['low_points_count']:,} ({low_stats['low_points_percentage']:.1f}%)<br>
                        <strong>📈 Mean height (low points):</strong> {low_stats['low_mean']:.2f} m<br>
                        <strong>📉 Min height (low points):</strong> {low_stats['low_min']:.2f} m
                    </div>
                    """, unsafe_allow_html=True)
        
        # Selected points display
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
if st.session_state.chm_data:
    st.markdown('<div class="section-header">📊 CHM Analysis Charts & Visualizations</div>', unsafe_allow_html=True)
    
    # Create tabs for different charts
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Yearly Bar Charts", "📈 Height Distribution", "📉 Temporal Trends", "🎯 Low Height Analysis"])
    
    with tab1:
        st.subheader("Yearly CHM Comparison - Bar Charts")
        
        # Create comprehensive yearly bar chart
        yearly_fig = create_yearly_bar_chart(st.session_state.chm_data, height_threshold)
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
        st.subheader("Canopy Height Distribution")
        
        # Select years to compare
        years_to_compare = st.multiselect(
            "Select years to compare",
            options=sorted(st.session_state.chm_data.keys()),
            default=[sorted(st.session_state.chm_data.keys())[0], sorted(st.session_state.chm_data.keys())[-1]] if len(st.session_state.chm_data) >= 2 else sorted(st.session_state.chm_data.keys())
        )
        
        if years_to_compare:
            dist_fig = create_height_distribution_chart(st.session_state.chm_data, years_to_compare, height_threshold)
            st.plotly_chart(dist_fig, use_container_width=True)
    
    with tab3:
        st.subheader("Temporal CHM Trends")
        
        # Calculate metrics for trend analysis
        years = sorted(st.session_state.chm_data.keys())
        avg_heights = []
        med_heights = []
        std_heights = []
        low_percentages = []
        
        for year in years:
            data = st.session_state.chm_data[year]['data'].flatten()
            valid_data = data[~np.isnan(data)]
            
            if len(valid_data) > 0:
                avg_heights.append(np.mean(valid_data))
                med_heights.append(np.median(valid_data))
                std_heights.append(np.std(valid_data))
                
                low_mask = valid_data < height_threshold
                low_percentages.append((np.sum(low_mask) / len(valid_data)) * 100)
            else:
                avg_heights.append(0)
                med_heights.append(0)
                std_heights.append(0)
                low_percentages.append(0)
        
        # Create trend figure
        fig_trend = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Average Canopy Height Trend", "Percentage Below Threshold Trend"),
            vertical_spacing=0.15
        )
        
        fig_trend.add_trace(
            go.Scatter(
                x=years,
                y=avg_heights,
                mode='lines+markers',
                name='Avg Height',
                line=dict(color='#2E7D32', width=3),
                marker=dict(size=10),
                error_y=dict(
                    type='data',
                    array=std_heights,
                    visible=True
                )
            ),
            row=1, col=1
        )
        
        fig_trend.add_hline(y=height_threshold, line_dash="dash", line_color="red",
                           annotation_text=f"Threshold: {height_threshold}m", row=1, col=1)
        
        fig_trend.add_trace(
            go.Scatter(
                x=years,
                y=low_percentages,
                mode='lines+markers',
                name='% Below Threshold',
                line=dict(color='#FF6B6B', width=3),
                marker=dict(size=10),
                fill='tozeroy',
                fillcolor='rgba(255,107,107,0.2)'
            ),
            row=2, col=1
        )
        
        fig_trend.update_layout(
            height=500,
            template='plotly_white',
            showlegend=False,
            hovermode='x unified'
        )
        
        fig_trend.update_xaxes(title_text="Year", row=2, col=1)
        fig_trend.update_yaxes(title_text="Height (m)", row=1, col=1)
        fig_trend.update_yaxes(title_text="Percentage (%)", row=2, col=1)
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Calculate growth rate
        if len(avg_heights) >= 2:
            growth_rate = ((avg_heights[-1] - avg_heights[0]) / avg_heights[0]) * 100
            st.markdown(f"""
            <div class="metric-card" style="background: #E8F5E9;">
                <strong>📈 Overall Growth Rate ({years[0]} to {years[-1]}):</strong> 
                <span style="color: {'#2E7D32' if growth_rate > 0 else '#F44336'}; font-size: 1.2rem;">
                    {growth_rate:+.1f}%
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.subheader(f"Low Canopy Height Analysis (< {height_threshold}m)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Overall Summary")
            
            total_points_below = 0
            total_points = 0
            
            for year, data_obj in st.session_state.chm_data.items():
                data = data_obj['data']
                valid_data = data[~np.isnan(data)]
                low_mask = valid_data < height_threshold
                
                total_points += len(valid_data)
                total_points_below += np.sum(low_mask)
            
            if total_points > 0:
                st.markdown(f"""
                <div class="highlight-box">
                    <strong>🔴 Total Points Below {height_threshold}m:</strong> {total_points_below:,}<br>
                    <strong>📊 Percentage of Total Area:</strong> {(total_points_below / total_points * 100):.1f}%<br>
                    <strong>📈 Affected Years:</strong> {len(st.session_state.chm_data)}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Low points by year bar chart
            low_by_year = []
            years_low = sorted(st.session_state.chm_data.keys())
            
            for year in years_low:
                data_obj = st.session_state.chm_data[year]
                data = data_obj['data']
                valid_data = data[~np.isnan(data)]
                low_mask = valid_data < height_threshold
                low_by_year.append((np.sum(low_mask) / len(valid_data) * 100) if len(valid_data) > 0 else 0)
            
            fig_low = go.Figure(data=[
                go.Bar(
                    x=years_low,
                    y=low_by_year,
                    text=[f"{p:.1f}%" for p in low_by_year],
                    textposition='outside',
                    marker_color=['#FF6B6B' if p > 30 else '#FFA07A' if p > 20 else '#FFC107' for p in low_by_year],
                    hovertemplate='Year: %{x}<br>Below Threshold: %{y:.1f}%<extra></extra>'
                )
            ])
            
            fig_low.update_layout(
                title="Percentage Below Threshold by Year",
                xaxis_title="Year",
                yaxis_title="Percentage (%)",
                height=300,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_low, use_container_width=True)
        
        # Detailed low points analysis by year
        st.markdown("### 📊 Yearly Low Points Analysis")
        
        low_analysis_data = []
        for year in sorted(st.session_state.chm_data.keys()):
            data_obj = st.session_state.chm_data[year]
            low_stats = calculate_low_height_stats(data_obj['data'], height_threshold)
            if low_stats:
                low_analysis_data.append({
                    'Year': year,
                    'Total Points': low_stats['total_points'],
                    'Low Points': low_stats['low_points_count'],
                    'Percentage': f"{low_stats['low_points_percentage']:.1f}%",
                    'Mean Low Height': f"{low_stats['low_mean']:.2f}m",
                    'Min Low Height': f"{low_stats['low_min']:.2f}m"
                })
        
        low_df = pd.DataFrame(low_analysis_data)
        st.dataframe(low_df, use_container_width=True)

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
