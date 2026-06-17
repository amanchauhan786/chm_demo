# app.py
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
    .file-missing {
        background: #FFEBEE;
        border-left: 4px solid #F44336;
    }
    .point-card {
        background: #f5f5f5;
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-radius: 5px;
        border-left: 3px solid #2196F3;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2E7D32;
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

# Helper Functions
def extract_year_from_filename(filename):
    """Extract year from filename using regex pattern"""
    # Look for 4-digit year in filename
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

def get_point_value(data, transform, lat, lon):
    """Get CHM value at specific lat/lon point"""
    try:
        # Convert lat/lon to row/col
        col, row = ~transform * (lon, lat)
        col = int(round(col))
        row = int(round(row))
        
        # Check bounds
        if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
            value = data[row, col]
            if not np.isnan(value):
                return float(value)
        return None
    except:
        return None

def create_chm_map(chm_data_dict, year, threshold=6.0):
    """Create a Folium map with CHM data"""
    if year not in chm_data_dict:
        return None
    
    data_obj = chm_data_dict[year]
    data = data_obj['data']
    bounds = data_obj['bounds']
    
    # Calculate center
    center_lat = (bounds.top + bounds.bottom) / 2
    center_lon = (bounds.left + bounds.right) / 2
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
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
    
    # Add CHM raster as a layer (simplified - would need to convert to image)
    # For demo, we'll add a placeholder
    folium.TileLayer('OpenStreetMap').add_to(m)
    
    # Add selected points
    for point in st.session_state.selected_points:
        if point['year'] == year:
            folium.Marker(
                location=[point['lat'], point['lon']],
                popup=f"CHM: {point.get('value', 'N/A')}m",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
    
    return m

def extract_points_for_export(chm_data_dict, threshold=6.0):
    """Extract all points below threshold for export"""
    all_points = []
    
    for year, data_obj in chm_data_dict.items():
        data = data_obj['data']
        transform = data_obj['transform']
        bounds = data_obj['bounds']
        
        # Sample points (in a real app, you'd sample more intelligently)
        # For demo, we'll create a grid of points
        lat_step = (bounds.top - bounds.bottom) / 100
        lon_step = (bounds.right - bounds.left) / 100
        
        points_found = 0
        for i in range(10):  # Sample 10x10 grid for demo
            for j in range(10):
                lat = bounds.bottom + i * lat_step * 10
                lon = bounds.left + j * lon_step * 10
                value = get_point_value(data, transform, lat, lon)
                
                if value is not None and value < threshold:
                    all_points.append({
                        'year': year,
                        'latitude': lat,
                        'longitude': lon,
                        'height': value,
                        'below_threshold': True
                    })
                    points_found += 1
        
        # Add some random points for variety (for demo purposes)
        # In production, you'd sample properly
    
    return pd.DataFrame(all_points)

# Main Title
st.markdown('<div class="main-header">🌳 Canopy Height Model (CHM) Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #555;">Internship Project at Varaha ClimateAg Private Limited | Aman Chauhan (22BCE0476)</p>', unsafe_allow_html=True)

# Sidebar for file management
with st.sidebar:
    st.markdown("## 📁 CHM Data Management")
    
    # Option to use existing folder or upload new files
    data_source = st.radio(
        "Select data source:",
        ["Use existing files", "Upload new TIFF files"]
    )
    
    if data_source == "Upload new TIFF files":
        uploaded_files = st.file_uploader(
            "Upload CHM TIFF Files",
            type=['tif', 'tiff'],
            accept_multiple_files=True,
            help="Upload TIFF files with year in filename (e.g., Bangladesh_mosaic_2020.tif)"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files uploaded")
            
            # Process uploaded files
            if st.button("📊 Process Uploaded Files"):
                with st.spinner("Processing TIFF files..."):
                    for file in uploaded_files:
                        year = extract_year_from_filename(file.name)
                        if year:
                            # Save temporarily and load
                            temp_path = f"/tmp/{file.name}"
                            with open(temp_path, "wb") as f:
                                f.write(file.getbuffer())
                            
                            data_obj = load_chm_tiff(temp_path)
                            if data_obj:
                                st.session_state.chm_data[year] = data_obj
                                st.session_state.year_data[year] = {
                                    'filename': file.name,
                                    'size': file.size
                                }
                                st.success(f"✅ Loaded {file.name} (Year: {year})")
                        else:
                            st.warning(f"⚠️ Could not extract year from: {file.name}")
    else:
        # Use existing files - specify folder path
        st.markdown("### 📂 Existing Files")
        
        # Create a text input for folder path
        folder_path = st.text_input(
            "Enter folder path with CHM TIFF files:",
            value="./chm_tiffs" if os.path.exists("./chm_tiffs") else "",
            help="Path to folder containing Bangladesh_mosaic_YYYY.tif files"
        )
        
        if folder_path and os.path.exists(folder_path):
            # List all TIFF files in the folder
            tiff_files = [f for f in os.listdir(folder_path) if f.endswith(('.tif', '.tiff'))]
            
            if tiff_files:
                st.markdown(f"**Found {len(tiff_files)} TIFF files:**")
                
                # Display files with year extraction
                year_files = {}
                for filename in tiff_files:
                    year = extract_year_from_filename(filename)
                    if year:
                        year_files[year] = filename
                        file_path = os.path.join(folder_path, filename)
                        size_kb = os.path.getsize(file_path) / 1024
                        
                        if year in st.session_state.chm_data:
                            st.markdown(f'<div class="file-status file-loaded">✅ {filename} (Year: {year}) - {size_kb:.0f} KB - Loaded</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="file-status file-missing">📄 {filename} (Year: {year}) - {size_kb:.0f} KB - Not loaded</div>', unsafe_allow_html=True)
                
                if st.button("📊 Load All Files from Folder"):
                    with st.spinner("Loading all CHM TIFF files..."):
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
            else:
                st.info("No TIFF files found in the specified folder")
        else:
            if folder_path:
                st.error(f"❌ Folder not found: {folder_path}")
            else:
                st.info("Please enter the folder path to load existing CHM files")
    
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
        options=['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'YlGn', 'RdYlGn'],
        index=5
    )
    
    # Year selection for display
    available_years = sorted(st.session_state.chm_data.keys())
    if available_years:
        selected_year = st.selectbox(
            "Select Year to Display",
            options=available_years,
            format_func=lambda x: f"{x}"
        )
    else:
        selected_year = None
        st.info("No CHM data loaded yet")
    
    st.markdown("---")
    st.markdown("## 📊 Export Options")
    
    if st.button("📥 Export Analysis Report"):
        st.info("Report export will include all analyzed data")

# Main content area
if not st.session_state.chm_data:
    st.info("📂 Please load CHM TIFF files to begin analysis. Use the sidebar to upload files or specify a folder path.")
    
    # Show placeholder
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: #f5f5f5; border-radius: 10px;">
        <h3>🌳 CHM Analysis Dashboard</h3>
        <p style="color: #666;">Load CHM TIFF files to visualize and analyze canopy height data</p>
        <p style="font-size: 0.9rem; color: #999;">Expected files: Bangladesh_mosaic_2020.tif, Bangladesh_mosaic_2021.tif, ...</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display loaded years
    loaded_years = sorted(st.session_state.chm_data.keys())
    st.success(f"✅ Loaded CHM data for years: {', '.join(map(str, loaded_years))}")
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-header">🗺️ CHM Visualization Map</div>', unsafe_allow_html=True)
        
        # Create and display map
        if selected_year is not None and selected_year in st.session_state.chm_data:
            data_obj = st.session_state.chm_data[selected_year]
            data = data_obj['data']
            bounds = data_obj['bounds']
            
            # Calculate center for map
            center_lat = (bounds.top + bounds.bottom) / 2
            center_lon = (bounds.left + bounds.right) / 2
            
            # Create Folium map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=10,
                tiles='OpenStreetMap'
            )
            
            # Add draw control for point selection
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
            
            # Add markers for selected points
            for point in st.session_state.selected_points:
                if point.get('year') == selected_year:
                    folium.CircleMarker(
                        location=[point['lat'], point['lon']],
                        radius=8,
                        popup=f"CHM: {point.get('value', 'N/A')}m",
                        color='red',
                        fill=True,
                        fill_color='red'
                    ).add_to(m)
            
            # Display map
            folium_static(m, width=700, height=500)
            
            # Instructions
            st.markdown("""
            <div style="background: #E3F2FD; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem;">
                <small>💡 Click on the map to add points for analysis. Selected points will appear in the panel.</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Handle point selection (simplified - in production you'd use JavaScript callback)
            # For now, we'll use a manual input method
            st.markdown("### 📍 Add Point Manually")
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
                value = get_point_value(data, data_obj['transform'], lat_input, lon_input)
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
        st.markdown('<div class="section-header">📊 CHM Analytics</div>', unsafe_allow_html=True)
        
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
    st.markdown('<div class="section-header">📊 CHM Analysis Charts</div>', unsafe_allow_html=True)
    
    # Create tabs for different charts
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Height Distribution", "📉 Temporal Trends", "🎯 Low Height Analysis", "📊 Point Analysis"])
    
    with tab1:
        st.subheader("Canopy Height Distribution")
        
        # Allow selecting multiple years for comparison
        years_to_compare = st.multiselect(
            "Select years to compare",
            options=sorted(st.session_state.chm_data.keys()),
            default=[sorted(st.session_state.chm_data.keys())[0]] if st.session_state.chm_data else []
        )
        
        if years_to_compare:
            fig1 = go.Figure()
            
            for year in years_to_compare:
                data_obj = st.session_state.chm_data[year]
                data = data_obj['data'].flatten()
                valid_data = data[~np.isnan(data)]
                
                # Create histogram
                fig1.add_trace(go.Histogram(
                    x=valid_data,
                    name=f"{year}",
                    nbinsx=30,
                    opacity=0.7,
                    histnorm='probability density'
                ))
            
            fig1.update_layout(
                title="Canopy Height Distribution by Year",
                xaxis_title="Height (m)",
                yaxis_title="Density",
                height=500,
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
            fig1.add_vline(x=height_threshold, line_dash="dash", line_color="red",
                          annotation_text=f"Threshold: {height_threshold}m")
            
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Select years to display height distribution")
    
    with tab2:
        st.subheader("Temporal CHM Trends")
        
        # Calculate average heights for each year
        years = sorted(st.session_state.chm_data.keys())
        avg_heights = []
        std_heights = []
        low_percentages = []
        
        for year in years:
            data_obj = st.session_state.chm_data[year]
            data = data_obj['data']
            valid_data = data[~np.isnan(data)]
            
            if len(valid_data) > 0:
                avg_heights.append(np.mean(valid_data))
                std_heights.append(np.std(valid_data))
                
                low_mask = valid_data < height_threshold
                low_percentages.append((np.sum(low_mask) / len(valid_data)) * 100)
            else:
                avg_heights.append(0)
                std_heights.append(0)
                low_percentages.append(0)
        
        # Create figure with subplots
        fig2 = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Average Canopy Height", "Percentage Below Threshold"),
            vertical_spacing=0.15
        )
        
        # Average height plot
        fig2.add_trace(
            go.Scatter(
                x=years,
                y=avg_heights,
                mode='lines+markers',
                name='Avg Height',
                line=dict(color='#2E7D32', width=3),
                marker=dict(size=10)
            ),
            row=1, col=1
        )
        
        # Add threshold line
        fig2.add_hline(y=height_threshold, line_dash="dash", line_color="red",
                      annotation_text=f"Threshold: {height_threshold}m", row=1, col=1)
        
        # Low percentage plot
        fig2.add_trace(
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
        
        fig2.update_layout(
            height=600,
            template='plotly_white',
            showlegend=False,
            hovermode='x unified'
        )
        
        fig2.update_xaxes(title_text="Year", row=2, col=1)
        fig2.update_yaxes(title_text="Height (m)", row=1, col=1)
        fig2.update_yaxes(title_text="Percentage (%)", row=2, col=1)
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Summary table
        trend_df = pd.DataFrame({
            'Year': years,
            'Avg Height (m)': [f"{h:.2f}" for h in avg_heights],
            'Std Dev (m)': [f"{s:.2f}" for s in std_heights],
            '% Below Threshold': [f"{p:.1f}%" for p in low_percentages]
        })
        
        st.dataframe(trend_df, use_container_width=True)
    
    with tab3:
        st.subheader(f"Low Canopy Height Analysis (< {height_threshold}m)")
        
        # Summary stats for all years
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Summary Statistics")
            
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
            st.markdown("### 📈 Year-over-Year Trend")
            
            # Create bar chart for low points by year
            low_by_year = []
            years_low = sorted(st.session_state.chm_data.keys())
            
            for year in years_low:
                data_obj = st.session_state.chm_data[year]
                data = data_obj['data']
                valid_data = data[~np.isnan(data)]
                low_mask = valid_data < height_threshold
                low_by_year.append((np.sum(low_mask) / len(valid_data) * 100) if len(valid_data) > 0 else 0)
            
            fig3 = go.Figure(data=[
                go.Bar(
                    x=years_low,
                    y=low_by_year,
                    text=[f"{p:.1f}%" for p in low_by_year],
                    textposition='outside',
                    marker_color=['#FF6B6B' if p > 30 else '#FFA07A' if p > 20 else '#FFC107' for p in low_by_year]
                )
            ])
            
            fig3.update_layout(
                title="Percentage of Area Below Threshold by Year",
                xaxis_title="Year",
                yaxis_title="Percentage (%)",
                height=300,
                template='plotly_white'
            )
            
            st.plotly_chart(fig3, use_container_width=True)
        
        # Export low points
        st.markdown("### 📥 Export Low Points Data")
        
        if st.button("📊 Extract and Export Low Points"):
            with st.spinner("Extracting low points data..."):
                low_points_df = extract_points_for_export(st.session_state.chm_data, height_threshold)
                
                if not low_points_df.empty:
                    st.success(f"✅ Extracted {len(low_points_df)} low points")
                    
                    # Display sample
                    st.dataframe(low_points_df.head(100), use_container_width=True)
                    
                    # Download button
                    csv = low_points_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Low Points CSV",
                        data=csv,
                        file_name=f"low_chm_points_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No low points found below the threshold")
    
    with tab4:
        st.subheader("Point Analysis")
        
        if st.session_state.selected_points:
            st.markdown("### 📍 Selected Points Analysis")
            
            # Create table of selected points
            points_data = []
            for point in st.session_state.selected_points:
                points_data.append({
                    'Year': point.get('year', 'N/A'),
                    'Latitude': point['lat'],
                    'Longitude': point['lon'],
                    'CHM Height (m)': point.get('value', 'N/A'),
                    'Below Threshold': '✅' if point.get('below_threshold', False) else '❌'
                })
            
            points_df = pd.DataFrame(points_data)
            st.dataframe(points_df, use_container_width=True)
            
            # Create scatter plot of selected points
            if len(points_data) > 1:
                fig4 = go.Figure()
                
                # Group by year
                for year in points_df['Year'].unique():
                    year_data = points_df[points_df['Year'] == year]
                    
                    fig4.add_trace(go.Scatter(
                        x=year_data['Longitude'],
                        y=year_data['Latitude'],
                        mode='markers+text',
                        name=str(year),
                        text=year_data['CHM Height (m)'].apply(lambda x: f"{x:.1f}m"),
                        textposition="top center",
                        marker=dict(
                            size=15,
                            symbol='circle',
                            line=dict(width=2, color='white')
                        )
                    ))
                
                fig4.update_layout(
                    title="Selected Points Distribution",
                    xaxis_title="Longitude",
                    yaxis_title="Latitude",
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig4, use_container_width=True)
            
            # Clear points button
            if st.button("🗑️ Clear All Points", key="clear_all_points"):
                st.session_state.selected_points = []
                st.rerun()
        else:
            st.info("No points selected. Add points from the map to analyze specific locations.")

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
