# app.py - Complete Farm Analysis Dashboard (No Stats Text, Fully Fixed)
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
            is_increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
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
        
        yearly_str = ''.join([
            f"{year}: {value:.2f}m<br>" 
            for year, value in zip(farm['years'], farm['values'])
        ])
        
        popup_html = f"""
        <div style="font-size: 13px; font-family: Arial, sans-serif; max-width: 300px;">
            <b>Farm KEY: {farm['KEY']}</b><br>
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
                marker=dict(size=12),
                hovertemplate='Year: %{x}<br>Height: %{y:.2f}m<extra></extra>'
            ))
    else:
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

def create_summary_dashboard(farm_analysis_df, threshold=7.0):
    """Create summary dashboard - FIXED: No empty pie chart"""
    if farm_analysis_df.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Farm Status Distribution",
            "Height Distribution (2024)",
            "Growth Rate Distribution",
            f"Farms Below {threshold}m (2024)",
            "Top 10 Growth Rates",
            "Yearly Height Trends"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # 1. Status Distribution - Only add if there is data
    status_counts = {
        'Low & Increasing': len(farm_analysis_df[(farm_analysis_df['is_low']) & (farm_analysis_df['is_increasing'])]),
        'Low & Not Increasing': len(farm_analysis_df[(farm_analysis_df['is_low']) & (~farm_analysis_df['is_increasing'])]),
        'Good & Increasing': len(farm_analysis_df[(~farm_analysis_df['is_low']) & (farm_analysis_df['is_increasing'])]),
        'Good & Stable': len(farm_analysis_df[(~farm_analysis_df['is_low']) & (~farm_analysis_df['is_increasing'])])
    }
    
    # Filter out zero values
    labels = [k for k, v in status_counts.items() if v > 0]
    values = [v for v in status_counts.values() if v > 0]
    
    if labels and values:
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker=dict(colors=['#FFA500', '#FF4444', '#4CAF50', '#2196F3']),
                textinfo='label+percent'
            ),
            row=1, col=1
        )
    else:
        # Add a placeholder if no data
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode='markers',
                marker=dict(size=1, color='white'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        # Add annotation
        fig.add_annotation(
            text="No Status Data Available",
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            row=1, col=1
        )
    
    # 2. Height Distribution
    heights = farm_analysis_df['last_value'].dropna()
    if not heights.empty:
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
    if not growth_rates.empty:
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
    if not low_farms.empty:
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
    if not top_growth.empty:
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
    
    # 6. Yearly Height Trends
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
        type=['csv']
    )
    
    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            st.session_state.csv_data = df
            st.success(f"✅ Loaded CSV with {len(df)} rows")
            
            median_cols = extract_median_columns(df)
            if median_cols:
                st.success(f"✅ Found columns: {', '.join([f'{year}' for year in median_cols.keys()])}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🗺️ Load Shapefile")
    if st.button("📂 Load Shapefile from project_farm"):
        with st.spinner("Loading..."):
            if load_shapefile_from_folder():
                st.success(f"✅ Loaded {len(st.session_state.shapefile_data)} features")
                st.rerun()
            else:
                st.warning("Shapefile not found")
    
    if st.session_state.shapefile_data is not None:
        st.success("✅ Shapefile loaded")
    
    st.markdown("---")
    
    if st.session_state.csv_data is not None and st.session_state.shapefile_data is not None:
        st.markdown("## 🎯 Analysis")
        
        height_threshold = st.slider(
            "Height Threshold (m)",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.5
        )
        
        if st.button("🔗 Merge & Analyze", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                merged = merge_shapefile_with_csv(
                    st.session_state.shapefile_data,
                    st.session_state.csv_data,
                    'KEY'
                )
                if merged is not None:
                    analysis_df = analyze_farm_data(merged, height_threshold)
                    if analysis_df is not None and not analysis_df.empty:
                        st.session_state.farm_analysis = analysis_df
                        st.success(f"✅ Analyzed {len(analysis_df)} farms")
                        st.rerun()

# Main Content - NO STATS TEXT DISPLAY
if st.session_state.farm_analysis is not None:
    analysis_df = st.session_state.farm_analysis
    
    # Main tabs only - No stats text
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Farm Map", "📈 Trends", "📊 Dashboard", "📋 Data"])
    
    with tab1:
        st.subheader("Interactive Farm Map")
        
        m = create_farm_map(analysis_df, height_threshold if 'height_threshold' in locals() else 7.0)
        if m:
            folium_static(m, width=750, height=600)
            
            st.markdown("""
            <div style="background: #E3F2FD; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem;">
                <small>💡 <b>Legend:</b> 
                    🟢 Good & Increasing | 🔵 Good & Stable | 🟠 Low & Increasing | 🔴 Low & Not Increasing
                </small>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Farm Trend Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            filter_option = st.selectbox(
                "Filter:",
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
        
        if not filtered_df.empty:
            farm_options = filtered_df['KEY'].tolist()
            selected_key = st.selectbox(
                "Select farm:",
                options=farm_options,
                format_func=lambda x: f"KEY: {x} - {filtered_df[filtered_df['KEY']==x]['last_value'].iloc[0]:.2f}m"
            )
            
            if selected_key:
                trend_fig = create_trend_chart(analysis_df, selected_key)
                st.plotly_chart(trend_fig, use_container_width=True)
                
                # Show farm details
                farm = analysis_df[analysis_df['KEY'] == selected_key].iloc[0]
                
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
                
                # Yearly data table
                yearly_data = pd.DataFrame({
                    'Year': farm['years'],
                    'Height (m)': [f"{v:.2f}" for v in farm['values']],
                    'Std Dev': [f"{s:.2f}" for s in farm['stds']]
                })
                st.dataframe(yearly_data, use_container_width=True)
    
    with tab3:
        st.subheader("Summary Dashboard")
        
        summary_fig = create_summary_dashboard(analysis_df, height_threshold if 'height_threshold' in locals() else 7.0)
        if summary_fig:
            st.plotly_chart(summary_fig, use_container_width=True)
    
    with tab4:
        st.subheader("Farm Data Table")
        
        display_df = analysis_df.copy()
        display_df['values'] = display_df['values'].apply(
            lambda x: ', '.join([f"{v:.2f}" for v in x]) if x else 'N/A'
        )
        display_df['years'] = display_df['years'].apply(
            lambda x: ', '.join(map(str, x)) if x else 'N/A'
        )
        
        cols_to_show = ['KEY', 'last_value', 'avg_value', 'median_value', 
                       'growth_rate', 'is_low', 'is_increasing', 'years', 'values']
        display_cols = [col for col in cols_to_show if col in display_df.columns]
        
        st.dataframe(display_df[display_cols], use_container_width=True)
        
        csv = analysis_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"farm_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

else:
    st.info("📂 Upload CSV and load shapefile to begin.")
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: #f5f5f5; border-radius: 10px;">
        <h3>🌾 Farm Analysis Dashboard</h3>
        <p>Upload CSV with median_height_2020 to median_height_2024</p>
        <div style="margin-top: 1rem;">
            <span style="background: #E8F5E9; padding: 0.5rem 1rem; border-radius: 5px;">📊 CSV Analysis</span>
            <span style="background: #E3F2FD; padding: 0.5rem 1rem; border-radius: 5px; margin-left: 0.5rem;">🗺️ Shapefile</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>Developed by Aman Chauhan (22BCE0476) | Varaha ClimateAg Private Limited</p>
</div>
""", unsafe_allow_html=True)
