"""
GIS Data Utility Module
Handles CSV-based GIS data with efficient search and sort algorithms
Provides location-based data for crop and water management recommendations
"""

import pandas as pd
from typing import Optional, Dict, List, Any
from pathlib import Path
import os

# Import GIS data path from config
from config import GIS_DATA_PATH

# RAINFALL CORRECTION TABLE - Known good data from IMD (Indian Meteorological Dept)
# The CSV has data quality issues for some districts
# Source: IMD Climate Data (annual averages in mm)
RAINFALL_CORRECTIONS = {
    # Format: 'DISTRICT_UPPER': {'monsoon': mm, 'annual': mm, 'post_monsoon': mm, 'summer': mm, 'winter': mm}
    'MUMBAI': {'monsoon': 2200, 'annual': 2400, 'post_monsoon': 100, 'summer': 50, 'winter': 50},
    'MUMBAI SUBURBAN': {'monsoon': 2200, 'annual': 2400, 'post_monsoon': 100, 'summer': 50, 'winter': 50},
    'THANE': {'monsoon': 2500, 'annual': 2700, 'post_monsoon': 120, 'summer': 50, 'winter': 30},
    'NEW DELHI': {'monsoon': 550, 'annual': 700, 'post_monsoon': 35, 'summer': 60, 'winter': 55},
    'DELHI': {'monsoon': 550, 'annual': 700, 'post_monsoon': 35, 'summer': 60, 'winter': 55},
    'SOUTH DELHI': {'monsoon': 550, 'annual': 700, 'post_monsoon': 35, 'summer': 60, 'winter': 55},
    'NORTH DELHI': {'monsoon': 550, 'annual': 700, 'post_monsoon': 35, 'summer': 60, 'winter': 55},
    'EAST DELHI': {'monsoon': 550, 'annual': 700, 'post_monsoon': 35, 'summer': 60, 'winter': 55},
    'WEST DELHI': {'monsoon': 550, 'annual': 700, 'post_monsoon': 35, 'summer': 60, 'winter': 55},
    'CENTRAL DELHI': {'monsoon': 550, 'annual': 700, 'post_monsoon': 35, 'summer': 60, 'winter': 55},
    'NORTH EAST DELHI': {'monsoon': 550, 'annual': 700, 'post_monsoon': 35, 'summer': 60, 'winter': 55},
    'NORTH WEST DELHI': {'monsoon': 550, 'annual': 700, 'post_monsoon': 35, 'summer': 60, 'winter': 55},
    'SOUTH EAST DELHI': {'monsoon': 550, 'annual': 700, 'post_monsoon': 35, 'summer': 60, 'winter': 55},
    'SOUTH WEST DELHI': {'monsoon': 550, 'annual': 700, 'post_monsoon': 35, 'summer': 60, 'winter': 55},
    'SHAHDARA': {'monsoon': 550, 'annual': 700, 'post_monsoon': 35, 'summer': 60, 'winter': 55},
    'BENGALURU URBAN': {'monsoon': 600, 'annual': 970, 'post_monsoon': 200, 'summer': 100, 'winter': 70},
    'BENGALURU RURAL': {'monsoon': 600, 'annual': 970, 'post_monsoon': 200, 'summer': 100, 'winter': 70},
    'BANGALORE': {'monsoon': 600, 'annual': 970, 'post_monsoon': 200, 'summer': 100, 'winter': 70},
    'BANGALORE URBAN': {'monsoon': 600, 'annual': 970, 'post_monsoon': 200, 'summer': 100, 'winter': 70},
    'BANGALORE RURAL': {'monsoon': 600, 'annual': 970, 'post_monsoon': 200, 'summer': 100, 'winter': 70},
    'CHENNAI': {'monsoon': 350, 'annual': 1300, 'post_monsoon': 700, 'summer': 150, 'winter': 100},
    'HYDERABAD': {'monsoon': 600, 'annual': 800, 'post_monsoon': 100, 'summer': 50, 'winter': 50},
    'RANGAREDDY': {'monsoon': 600, 'annual': 800, 'post_monsoon': 100, 'summer': 50, 'winter': 50},
    'PUNE': {'monsoon': 700, 'annual': 900, 'post_monsoon': 100, 'summer': 60, 'winter': 40},
    'KOLKATA': {'monsoon': 1200, 'annual': 1600, 'post_monsoon': 200, 'summer': 150, 'winter': 50},
    'HOWRAH': {'monsoon': 1200, 'annual': 1600, 'post_monsoon': 200, 'summer': 150, 'winter': 50},
    'AHMEDABAD': {'monsoon': 700, 'annual': 800, 'post_monsoon': 30, 'summer': 40, 'winter': 30},
    'JAIPUR': {'monsoon': 500, 'annual': 600, 'post_monsoon': 30, 'summer': 30, 'winter': 40},
    'LUCKNOW': {'monsoon': 750, 'annual': 900, 'post_monsoon': 50, 'summer': 40, 'winter': 60},
    'PATNA': {'monsoon': 900, 'annual': 1100, 'post_monsoon': 100, 'summer': 50, 'winter': 50},
    'GURGAON': {'monsoon': 500, 'annual': 650, 'post_monsoon': 40, 'summer': 50, 'winter': 60},
    'GURUGRAM': {'monsoon': 500, 'annual': 650, 'post_monsoon': 40, 'summer': 50, 'winter': 60},
    'NOIDA': {'monsoon': 550, 'annual': 700, 'post_monsoon': 40, 'summer': 55, 'winter': 55},
    'GHAZIABAD': {'monsoon': 550, 'annual': 700, 'post_monsoon': 40, 'summer': 55, 'winter': 55},
    'FARIDABAD': {'monsoon': 500, 'annual': 650, 'post_monsoon': 40, 'summer': 55, 'winter': 55},
    # High rainfall areas
    'EAST KHASI HILLS': {'monsoon': 10000, 'annual': 11500, 'post_monsoon': 800, 'summer': 500, 'winter': 200},
    'WEST KHASI HILLS': {'monsoon': 5000, 'annual': 6000, 'post_monsoon': 600, 'summer': 300, 'winter': 100},
    'EAST GARO HILLS': {'monsoon': 3500, 'annual': 4000, 'post_monsoon': 300, 'summer': 150, 'winter': 50},
    # Coastal high rainfall
    'SINDHUDURG': {'monsoon': 3200, 'annual': 3500, 'post_monsoon': 200, 'summer': 50, 'winter': 50},
    'RATNAGIRI': {'monsoon': 3500, 'annual': 3800, 'post_monsoon': 200, 'summer': 50, 'winter': 50},
    'UDUPI': {'monsoon': 3500, 'annual': 4000, 'post_monsoon': 300, 'summer': 100, 'winter': 100},
    'DAKSHINA KANNADA': {'monsoon': 3500, 'annual': 4000, 'post_monsoon': 300, 'summer': 100, 'winter': 100},
    # Low rainfall areas
    'JAISALMER': {'monsoon': 150, 'annual': 200, 'post_monsoon': 20, 'summer': 15, 'winter': 15},
    'BIKANER': {'monsoon': 200, 'annual': 280, 'post_monsoon': 30, 'summer': 25, 'winter': 25},
    'KUTCH': {'monsoon': 350, 'annual': 400, 'post_monsoon': 25, 'summer': 15, 'winter': 10},
}


class GISDataManager:
    """Manages GIS data with efficient search and retrieval"""
    
    def __init__(self):
        self.data = None
        self._loaded = False
    
    def load_data(self):
        """Load GIS data from CSV"""
        if self._loaded:
            return True
        
        try:
            csv_path = Path(GIS_DATA_PATH)
            if not csv_path.exists():
                print(f"Warning: GIS data file not found at {GIS_DATA_PATH}")
                return False
            
            self.data = pd.read_csv(GIS_DATA_PATH, low_memory=False)
            self._loaded = True
            print(f"GIS Data loaded: {len(self.data)} records")
            return True
        except Exception as e:
            print(f"Error loading GIS data: {e}")
            return False
    
    def get_location_data(self, pincode: Optional[str] = None, 
                         district: Optional[str] = None,
                         state: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get GIS data for a location using hierarchical search with fallback to nearest.
        Priority: Pincode (exact) > Pincode (prefix match) > District > State
        """
        if not self._loaded:
            if not self.load_data():
                return None
        
        try:
            df = self.data
            
            # Search by pincode (highest priority)
            if pincode:
                pincode_str = str(pincode).strip()
                
                # 1. Try exact pincode match
                result = df[df['pincode'].astype(str) == pincode_str]
                if not result.empty:
                    data = self._aggregate_location_data(result)
                    if data:
                        data['match_type'] = 'exact_pincode'
                        return data
                
                # 2. Try prefix match (first 3 digits = same region)
                if len(pincode_str) >= 3:
                    prefix = pincode_str[:3]
                    result = df[df['pincode'].astype(str).str.startswith(prefix)]
                    if not result.empty:
                        data = self._aggregate_location_data(result)
                        if data:
                            data['match_type'] = 'nearest_pincode'
                            data['note'] = f"Using nearest available data (pincode prefix {prefix}xxx)"
                            print(f"[GIS] Exact pincode {pincode_str} not found, using nearest: {data.get('pincode')} ({data.get('district')})")
                            return data
                
                # 3. Try first 2 digits (broader region)
                if len(pincode_str) >= 2:
                    prefix = pincode_str[:2]
                    result = df[df['pincode'].astype(str).str.startswith(prefix)]
                    if not result.empty:
                        data = self._aggregate_location_data(result)
                        if data:
                            data['match_type'] = 'regional_pincode'
                            data['note'] = f"Using regional data (pincode prefix {prefix}xxxx)"
                            print(f"[GIS] Using regional data for {pincode_str}: {data.get('district')}, {data.get('state')}")
                            return data
            
            # Search by district
            if district:
                result = df[df['District'].str.contains(district, case=False, na=False)]
                if not result.empty:
                    data = self._aggregate_location_data(result)
                    if data:
                        data['match_type'] = 'district'
                        return data
            
            # Search by state (broadest fallback)
            if state:
                result = df[df['State'].str.contains(state, case=False, na=False)]
                if not result.empty:
                    data = self._aggregate_location_data(result)
                    if data:
                        data['match_type'] = 'state'
                        data['note'] = f"Using state-level average data for {state}"
                        return data
            
            return None
            
        except Exception as e:
            print(f"Error retrieving location data: {e}")
            return None
    
    def _aggregate_location_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate data from multiple rows for a location with data quality corrections"""
        try:
            # Take first row for location details
            first_row = df.iloc[0]
            district_name = str(first_row.get('District', '')).upper().strip()
            
            # Check if we have a known correction for this district
            if district_name in RAINFALL_CORRECTIONS:
                correction = RAINFALL_CORRECTIONS[district_name]
                monsoon_val = correction['monsoon']
                post_monsoon_val = correction.get('post_monsoon', 50)
                summer_val = correction.get('summer', 50)
                winter_val = correction.get('winter', 30)
                total_val = correction['annual']
                print(f"[GIS] Using IMD-corrected rainfall for {district_name}: {total_val}mm/year")
            else:
                # Use MEDIAN from CSV data (handles outliers better than mean)
                monsoon_rainfall = df['Monsoon (Jun-Sep)'].median() if 'Monsoon (Jun-Sep)' in df.columns else 0
                post_monsoon = df['Post-Monsoon (Oct-Dec)'].median() if 'Post-Monsoon (Oct-Dec)' in df.columns else 0
                summer_rainfall = df['Summer/Pre-Monsoon (Mar-May)'].median() if 'Summer/Pre-Monsoon (Mar-May)' in df.columns else 0
                winter_rainfall = df['Winter (Jan-Feb)'].median() if 'Winter (Jan-Feb)' in df.columns else 0
                total_rainfall = df['Total Annual Rainfall (mm)'].median() if 'Total Annual Rainfall (mm)' in df.columns else 0
                
                monsoon_val = float(monsoon_rainfall) if pd.notna(monsoon_rainfall) else 0
                post_monsoon_val = float(post_monsoon) if pd.notna(post_monsoon) else 0
                summer_val = float(summer_rainfall) if pd.notna(summer_rainfall) else 0
                winter_val = float(winter_rainfall) if pd.notna(winter_rainfall) else 0
                total_val = float(total_rainfall) if pd.notna(total_rainfall) else 0
                
                # Calculate annual from seasonal if total is invalid
                calculated_annual = monsoon_val + post_monsoon_val + summer_val + winter_val
                if total_val <= 0 or total_val < calculated_annual * 0.5:
                    total_val = calculated_annual
                    print(f"[GIS] Using calculated annual rainfall for {district_name}: {total_val:.0f}mm (from seasonal data)")
            
            # Ground water data (use median for robustness)
            gw_recharge = df['Total Annual Ground Water Recharge (in BCM)'].median() if 'Total Annual Ground Water Recharge (in BCM)' in df.columns else 0
            gw_resource = df['Annual Extractable Ground Water Resource (in BCM)'].median() if 'Annual Extractable Ground Water Resource (in BCM)' in df.columns else 0
            gw_extraction = df['Current Annual Ground Water Extraction (in BCM) - Total'].median() if 'Current Annual Ground Water Extraction (in BCM) - Total' in df.columns else 0
            gw_stage = df['Stage of Ground Water Extraction (%)'].median() if 'Stage of Ground Water Extraction (%)' in df.columns else 0
            
            return {
                'pincode': str(first_row.get('pincode', '')),
                'district': str(first_row.get('District', '')),
                'state': str(first_row.get('State', '')),
                'latitude': float(first_row.get('latitude', 0)) if pd.notna(first_row.get('latitude')) else None,
                'longitude': float(first_row.get('longitude', 0)) if pd.notna(first_row.get('longitude')) else None,
                'rainfall': {
                    'monsoon': monsoon_val,
                    'post_monsoon': post_monsoon_val,
                    'summer': summer_val,
                    'winter': winter_val,
                    'total_annual': total_val
                },
                'groundwater': {
                    'recharge_bcm': float(gw_recharge) if pd.notna(gw_recharge) else 0,
                    'resource_bcm': float(gw_resource) if pd.notna(gw_resource) else 0,
                    'extraction_bcm': float(gw_extraction) if pd.notna(gw_extraction) else 0,
                    'extraction_percentage': float(gw_stage) if pd.notna(gw_stage) else 0
                }
            }
        except Exception as e:
            print(f"Error aggregating location data: {e}")
            return None
    
    def get_nearby_districts(self, district: str, limit: int = 5) -> List[str]:
        """Get nearby districts (simple text similarity for now)"""
        if not self._loaded:
            if not self.load_data():
                return []
        
        try:
            all_districts = self.data['District'].dropna().unique()
            # Simple contains-based search
            nearby = [d for d in all_districts if district.lower() in d.lower() or d.lower() in district.lower()]
            return nearby[:limit]
        except:
            return []
    
    def get_water_stress_level(self, location_data: Dict[str, Any]) -> str:
        """Determine water stress level based on groundwater extraction"""
        try:
            extraction_pct = location_data.get('groundwater', {}).get('extraction_percentage', 0)
            
            if extraction_pct < 0 or extraction_pct > 100:
                return "Data Unavailable"
            elif extraction_pct < 70:
                return "Safe"
            elif extraction_pct < 90:
                return "Semi-Critical"
            elif extraction_pct < 100:
                return "Critical"
            else:
                return "Over-Exploited"
        except:
            return "Unknown"


# Singleton instance
gis_manager = GISDataManager()
