# data_processor.py
import rasterio
import numpy as np
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import re
from datetime import datetime

def extract_year_from_filename(filename):
    """Extract year from filename"""
    year_match = re.search(r'(19|20)\d{2}', filename)
    return int(year_match.group()) if year_match else None

def load_chm_tiff(file_path):
    """Load CHM TIFF file"""
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)
            meta = src.meta
            bounds = src.bounds
            transform = src.transform
            
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
            
            valid_data = data[~np.isnan(data)]
            stats = None
            if len(valid_data) > 0:
                stats = {
                    'mean': float(np.mean(valid_data)),
                    'median': float(np.median(valid_data)),
                    'std': float(np.std(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'q1': float(np.percentile(valid_data, 25)),
                    'q3': float(np.percentile(valid_data, 75)),
                    'count': len(valid_data)
                }
            
            return {
                'data': data,
                'meta': meta,
                'bounds': bounds,
                'transform': transform,
                'stats': stats,
                'shape': data.shape
            }
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_point_value(data, transform, lat, lon):
    """Get CHM value at specific coordinates"""
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
    """Calculate statistics for points below threshold"""
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

def extract_points_for_export(chm_data_dict, threshold=6.0, num_samples=1000):
    """Extract points for export"""
    all_points = []
    
    for year, data_obj in chm_data_dict.items():
        data = data_obj['data']
        transform = data_obj['transform']
        bounds = data_obj['bounds']
        
        # Create grid of points
        lat_step = (bounds.top - bounds.bottom) / int(np.sqrt(num_samples))
        lon_step = (bounds.right - bounds.left) / int(np.sqrt(num_samples))
        
        points_found = 0
        for i in range(int(np.sqrt(num_samples))):
            for j in range(int(np.sqrt(num_samples))):
                lat = bounds.bottom + i * lat_step
                lon = bounds.left + j * lon_step
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
    
    return pd.DataFrame(all_points)
