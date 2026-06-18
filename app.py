# app.py - Complete Dashboard with Shapefile and CSV Integration
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
from folium.plugins import Draw, MarkerCluster
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
import zipfile
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chm_data' not in st.session_state:
    st.session_state.chm_data = {}
if 'selected_points' not in st.session_state:
    st.session_state.selected_points = []
if 'farm_data' not in st.session_state:
    st.session_state.farm_data = None
if 'shapefile_data' not in st.session_state:
    st.session_state.shapefile_data = None
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'selected_farms' not in st.session_state:
    st.session_state.selected_farms = []
if 'farm_analysis_results' not in st.session_state:
    st.session_state.farm_analysis_results = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

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
            crs = src.crs
            
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
                'crs': crs,
                'stats': stats,
                'shape': data.shape
            }
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

@st.cache_data
def load_shapefile(shp_path):
    """Load shapefile and return GeoDataFrame"""
    try:
        gdf = gpd.read_file(shp_path)
        return gdf
    except Exception as e:
        st.error(f"Error loading shapefile: {str(e)}")
        return None

@st.cache_data
def load_csv_data(csv_path):
    """Load CSV data and return DataFrame"""
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

def extract_chm_values_for_farms(chm_data_dict, farm_gdf, years):
    """Extract CHM values for each farm across all years"""
    results = []
    
    for idx, farm in farm_gdf.iterrows():
        # Get farm centroid
        centroid = farm.geometry.centroid
        lat = centroid.y
        lon = centroid.x
        
        farm_data = {
            'farm_id': idx,
            'latitude': lat,
            'longitude': lon
        }
        
        # Get CHM values for each year
        for year in years:
            if year in chm_data_dict:
                value = get_chm_value_at_point(chm_data_dict[year], lat, lon)
                farm_data[f'chm_{year}'] = value if value is not None else np.nan
        
        results.append(farm_data)
    
    return pd.DataFrame(results)

def analyze_farm_trends(farm_df, threshold=7.0):
    """Analyze farm trends - identify increasing pattern and low height"""
    results = []
    
    # Get year columns
    year_cols = [col for col in farm_df.columns if col.startswith('chm_')]
    years = sorted([int(col.split('_')[1]) for col in year_cols])
    
    for idx, row in farm_df.iterrows():
        values = [row[f'chm_{year}'] for year in years if f'chm_{year}' in row]
        valid_values = [v for v in values if not np.isnan(v)]
        
        if len(valid_values) >= 2:
            # Check if values are increasing
            is_increasing = all(valid_values[i] <= valid_values[i+1] for i in range(len(valid_values)-1))
            
            # Check if last value is below threshold
            is_low = valid_values[-1] < threshold if valid_values else False
            
            # Calculate growth rate
            growth_rate = ((valid_values[-1] - valid_values[0]) / valid_values[0] * 100) if valid_values[0] > 0 else 0
            
            results.append({
                'farm_id': row['farm_id'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'values': valid_values,
                'years': years[:len(valid_values)],
                'is_increasing': is_increasing,
                'is_low': is_low,
                'growth_rate': growth_rate,
                'last_value': valid_values[-1] if valid_values else None,
                'first_value': valid_values[0] if valid_values else None
            })
    
    return pd.DataFrame(results)

def create_farm_map(farm_analysis_df, chm_data_dict, selected_year, threshold=7.0):
    """Create a map with farm markers"""
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
    
    # Create CHM overlay
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
            color = 'orange'  # Low but increasing
        elif farm['is_low']:
            color = 'red'  # Low and not increasing
        elif farm['is_increasing']:
            color = 'green'  # Increasing
        else:
            color = 'blue'  # Stable/Decreasing
        
        # Create popup with detailed info
        popup_html = f"""
        <div style="font-size: 12px;">
            <b>Farm ID:</b> {farm['farm_id']}<br>
            <b>Status:</b> {'🔴 Low' if farm['is_low'] else '🟢 Good'}<br>
            <b>Trend:</b> {'📈 Increasing' if farm['is_increasing'] else '📉 Not Increasing'}<br>
            <b>Current Height:</b> {farm['last_value']:.2f}m<br>
            <b>Growth Rate:</b> {farm['growth_rate']:.1f}%<br>
            <b>Values:</b> {', '.join([f"{v:.2f}m" for v in farm['values']])}
        </div>
        """
        
        folium.CircleMarker(
            location=[farm['latitude'], farm['longitude']],
            radius=8,
            popup=folium.Popup(popup_html, max_width=300),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=2,
            tooltip=f"Farm {farm['farm_id']}: {farm['last_value']:.2f}m"
        ).add_to(marker_cluster)
    
    folium.LayerControl(position='topright').add_to(m)
    
    return m

def create_raster_image(data_obj, colormap='viridis', threshold=6.0):
    """Create a raster image for map overlay"""
    data = data_obj['data']
    bounds = data_obj['bounds']
    
    fig, ax = plt.subplots(figsize=(12, 12), dpi=80)
    data_plot = np.ma.masked_where(np.isnan(data), data)
    
    im = ax.imshow(data_plot, 
                   extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                   cmap=colormap,
                   vmin=0,
                   vmax=25,
                   alpha=0.8,
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

def create_farm_trend_chart(farm_analysis_df, selected_farm_id=None):
    """Create trend chart for farms"""
    fig = go.Figure()
    
    if selected_farm_id is not None:
        # Show single farm
        farm = farm_analysis_df[farm_analysis_df['farm_id'] == selected_farm_id]
        if not farm.empty:
            farm = farm.iloc[0]
            fig.add_trace(go.Scatter(
                x=farm['years'],
                y=farm['values'],
                mode='lines+markers',
                name=f"Farm {selected_farm_id}",
                line=dict(color='#2E7D32', width=3),
                marker=dict(size=10),
                hovertemplate='Year: %{x}<br>Height: %{y:.2f}m<extra></extra>'
            ))
    else:
        # Show top farms (low height and increasing)
        filtered = farm_analysis_df[
            (farm_analysis_df['is_low']) & 
            (farm_analysis_df['is_increasing'])
        ].head(10)
        
        for _, farm in filtered.iterrows():
            fig.add_trace(go.Scatter(
                x=farm['years'],
                y=farm['values'],
                mode='lines+markers',
                name=f"Farm {farm['farm_id']}",
                line=dict(width=2),
                marker=dict(size=6),
                hovertemplate=f"Farm {farm['farm_id']}<br>Year: %{{x}}<br>Height: %{{y:.2f}}m<extra></extra>"
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

def create_summary_charts(farm_analysis_df, threshold=7.0):
    """Create summary charts for farm analysis"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Farm Status Distribution",
            "Height Distribution",
            "Growth Rate Distribution",
            "Low Farms (Below Threshold)"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # 1. Status Distribution
    status_counts = {
        'Low & Increasing': len(farm_analysis_df[(farm_analysis_df['is_low']) & (farm_analysis_df['is_increasing'])]),
        'Low & Not Increasing': len(farm_analysis_df[(farm_analysis_df['is_low']) & (~farm_analysis_df['is_increasing'])]),
        'Good & Increasing': len(farm_analysis_df[(~farm_analysis_df['is_low']) & (farm_analysis_df['is_increasing'])]),
        'Good & Not Increasing': len(farm_analysis_df[(~farm_analysis_df['is_low']) & (~farm_analysis_df['is_increasing'])])
    }
    
    fig.add_trace(
        go.Pie(
            labels=list(status_counts.keys()),
            values=list(status_counts.values()),
            hole=0.4,
            marker=dict(colors=['#FFA500', '#FF4444', '#4CAF50', '#2196F3'])
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
        row=2, col=1
    )
    
    # 4. Low Farms
    low_farms = farm_analysis_df[farm_analysis_df['is_low']].sort_values('last_value').head(20)
    fig.add_trace(
        go.Bar(
            x=low_farms['farm_id'].astype(str),
            y=low_farms['last_value'],
            text=low_farms['last_value'].apply(lambda x: f"{x:.2f}m"),
            textposition='outside',
            marker_color=['#FF6B6B' if v < 5 else '#FFA07A' for v in low_farms['last_value']],
            hovertemplate='Farm: %{x}<br>Height: %{y:.2f}m<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        template='plotly_white',
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Height (m)", row=1, col=2)
    fig.update_xaxes(title_text="Growth Rate (%)", row=2, col=1)
    fig.update_xaxes(title_text="Farm ID", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Height (m)", row=2, col=2)
    
    return fig

# Main Title
st.markdown('<div class="main-header">🌳 CHM Farm Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #555;">Internship Project at Varaha ClimateAg Private Limited | Aman Chauhan (22BCE0476)</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📁 Data Loading")
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    # Option 1: Load Sample Data
    if st.button("🚀 Load Sample Data", type="primary", use_container_width=True):
        st.info("Sample data will be loaded with shapefile and CSV")
        st.session_state.data_loaded = True
        st.rerun()
    
    st.markdown("---")
    
    # Option 2: Load from project_farm folder
    st.markdown("### 📂 Load from Project Farm")
    project_path = st.text_input(
        "Project Farm Path:",
        value="./project_farm" if os.path.exists("./project_farm") else "",
        help="Path to the project_farm folder containing TIFF files"
    )
    
    if project_path and os.path.exists(project_path):
        # Load TIFF files
        tiff_files = [f for f in os.listdir(project_path) if f.endswith(('.tif', '.tiff'))]
        if tiff_files:
            st.markdown(f"**Found {len(tiff_files)} TIFF files:**")
            
            if st.button("📊 Load CHM Data", use_container_width=True):
                with st.spinner("Loading CHM data..."):
                    loaded_count = 0
                    for filename in tiff_files:
                        year = extract_year_from_filename(filename)
                        if year:
                            file_path = os.path.join(project_path, filename)
                            data_obj = load_chm_tiff(file_path)
                            if data_obj:
                                st.session_state.chm_data[year] = data_obj
                                loaded_count += 1
                    st.success(f"✅ Loaded {loaded_count} CHM files")
                    st.rerun()
    
    st.markdown("---")
    
    # Shapefile Upload
    st.markdown("### 🗺️ Upload Shapefile")
    shapefile_zip = st.file_uploader(
        "Upload Shapefile (ZIP)",
        type=['zip'],
        help="Upload a ZIP file containing .shp, .shx, .dbf, .prj files"
    )
    
    if shapefile_zip:
        with st.spinner("Extracting and loading shapefile..."):
            try:
                with zipfile.ZipFile(shapefile_zip, 'r') as z:
                    # Extract to temp directory
                    temp_dir = tempfile.mkdtemp()
                    z.extractall(temp_dir)
                    
                    # Find .shp file
                    shp_files = [f for f in os.listdir(temp_dir) if f.endswith('.shp')]
                    if shp_files:
                        shp_path = os.path.join(temp_dir, shp_files[0])
                        gdf = load_shapefile(shp_path)
                        if gdf is not None:
                            st.session_state.shapefile_data = gdf
                            st.success(f"✅ Loaded shapefile with {len(gdf)} features")
            except Exception as e:
                st.error(f"Error loading shapefile: {str(e)}")
    
    st.markdown("---")
    
    # CSV Upload
    st.markdown("### 📊 Upload CSV Data")
    csv_file = st.file_uploader(
        "Upload CSV with farm data",
        type=['csv'],
        help="CSV file containing farm data"
    )
    
    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            st.session_state.csv_data = df
            st.success(f"✅ Loaded CSV with {len(df)} rows")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 🎯 Analysis Controls")
    
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
        st.info("Load CHM data to enable year selection")

# Main content
if not st.session_state.chm_data:
    st.info("📂 Load CHM data using the sidebar to begin analysis.")
    
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: #f5f5f5; border-radius: 10px;">
        <h3>🌳 CHM Farm Analysis Dashboard</h3>
        <p style="color: #666;">Upload CHM data, shapefile, and CSV to analyze farm trends</p>
        <div style="margin-top: 1rem;">
            <span style="background: #E8F5E9; padding: 0.5rem 1rem; border-radius: 5px;">📊 Farm-level analysis</span>
            <span style="background: #E3F2FD; padding: 0.5rem 1rem; border-radius: 5px; margin-left: 0.5rem;">🗺️ Shapefile integration</span>
            <span style="background: #FFF3E0; padding: 0.5rem 1rem; border-radius: 5px; margin-left: 0.5rem;">📈 Trend detection</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display loaded data
    st.success(f"✅ Loaded CHM data for years: {', '.join(map(str, sorted(st.session_state.chm_data.keys())))}")
    
    # Process shapefile and CSV if available
    if st.session_state.shapefile_data is not None:
        # Extract CHM values for farms
        if st.button("🔄 Analyze Farms", type="primary"):
            with st.spinner("Analyzing farms..."):
                years = sorted(st.session_state.chm_data.keys())
                farm_df = extract_chm_values_for_farms(
                    st.session_state.chm_data,
                    st.session_state.shapefile_data,
                    years
                )
                st.session_state.farm_data = farm_df
                
                # Analyze trends
                analysis_df = analyze_farm_trends(farm_df, height_threshold)
                st.session_state.farm_analysis_results = analysis_df
                
                st.success(f"✅ Analyzed {len(analysis_df)} farms")
                st.rerun()
    
    # Display results
    if st.session_state.farm_analysis_results is not None:
        analysis_df = st.session_state.farm_analysis_results
        
        # Summary statistics
        st.markdown('<div class="section-header">📊 Farm Analysis Summary</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_farms = len(analysis_df)
        low_farms = len(analysis_df[analysis_df['is_low']])
        increasing_farms = len(analysis_df[analysis_df['is_increasing']])
        low_increasing = len(analysis_df[(analysis_df['is_low']) & (analysis_df['is_increasing'])])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <strong>Total Farms</strong><br>
                <span style="font-size: 1.5rem; color: #2E7D32;">{total_farms}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #F44336;">
                <strong>🔴 Low Farms</strong><br>
                <span style="font-size: 1.5rem; color: #F44336;">{low_farms}</span>
                <br><span style="color: #666;">({low_farms/total_farms*100:.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #2196F3;">
                <strong>📈 Increasing</strong><br>
                <span style="font-size: 1.5rem; color: #2196F3;">{increasing_farms}</span>
                <br><span style="color: #666;">({increasing_farms/total_farms*100:.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #FFA500;">
                <strong>⚠️ Low & Increasing</strong><br>
                <span style="font-size: 1.5rem; color: #FFA500;">{low_increasing}</span>
                <br><span style="color: #666;">({low_increasing/total_farms*100:.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Main layout
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Farm Map", "📈 Trends", "📊 Charts", "📋 Data"])
        
        with tab1:
            st.subheader("Farm Map with CHM Overlay")
            
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
                        <small>💡 <b>Legend:</b> 🟢 Good & Increasing | 🔵 Good & Not Increasing | 🟠 Low & Increasing | 🔴 Low & Not Increasing</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Select a year to display the map")
        
        with tab2:
            st.subheader("Farm Trend Analysis")
            
            # Filter farms
            filter_option = st.radio(
                "Show farms:",
                ["All Farms", "Low Farms (<7m)", "Increasing Farms", "Low & Increasing"],
                horizontal=True
            )
            
            if filter_option == "Low Farms (<7m)":
                filtered_df = analysis_df[analysis_df['is_low']]
            elif filter_option == "Increasing Farms":
                filtered_df = analysis_df[analysis_df['is_increasing']]
            elif filter_option == "Low & Increasing":
                filtered_df = analysis_df[(analysis_df['is_low']) & (analysis_df['is_increasing'])]
            else:
                filtered_df = analysis_df
            
            if not filtered_df.empty:
                # Select farm for detailed view
                farm_options = filtered_df['farm_id'].tolist()
                selected_farm = st.selectbox(
                    "Select farm to view trend:",
                    options=farm_options,
                    format_func=lambda x: f"Farm {x} - Height: {filtered_df[filtered_df['farm_id']==x]['last_value'].iloc[0]:.2f}m"
                )
                
                if selected_farm is not None:
                    # Show trend chart
                    trend_fig = create_farm_trend_chart(analysis_df, selected_farm)
                    st.plotly_chart(trend_fig, use_container_width=True)
                    
                    # Show farm details
                    farm = analysis_df[analysis_df['farm_id'] == selected_farm].iloc[0]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div class="farm-card">
                            <strong>📊 Current Height</strong><br>
                            <span style="font-size: 1.2rem;">{farm['last_value']:.2f} m</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="farm-card {'critical-farm' if farm['is_low'] else ''}">
                            <strong>📈 Growth Rate</strong><br>
                            <span style="font-size: 1.2rem; color: {'#2E7D32' if farm['growth_rate'] > 0 else '#F44336'};">
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
            else:
                st.info("No farms match the selected filter")
            
            # Show all trends for selected farms
            if st.checkbox("Show All Farm Trends"):
                all_trend_fig = create_farm_trend_chart(
                    analysis_df[analysis_df['is_low'] & analysis_df['is_increasing']].head(10),
                    None
                )
                st.plotly_chart(all_trend_fig, use_container_width=True)
        
        with tab3:
            st.subheader("Summary Charts")
            
            # Create summary charts
            summary_fig = create_summary_charts(analysis_df, height_threshold)
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
            
            st.dataframe(display_df, use_container_width=True)
            
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
    <p style="font-size: 0.8rem;">Features: Shapefile Integration • CSV Analytics • Farm Trend Detection • CHM Visualization</p>
</div>
""", unsafe_allow_html=True)
