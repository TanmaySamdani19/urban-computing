from flask import Flask, render_template, jsonify
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic
import os
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import json

# Initialize Flask app
app = Flask(__name__)

def get_real_time_data():
    """Fetch real-time data from the Dublin Bikes API"""
    url = 'http://api.citybik.es/v2/networks/dublinbikes'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        data = response.json()
        
        stations_data = []
        for station in data['network']['stations']:
            stations_data.append({
                'station_id': station.get('id'),
                'name': station.get('name'),
                'bikes': station.get('free_bikes', 0),
                'slots': station.get('empty_slots', 0),
                'lat': station.get('latitude'),
                'lng': station.get('longitude'),
                'timestamp': station.get('timestamp')  # Ensure this is in a readable format
            })
        return stations_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Dublin Bikes API: {e}")
        return []


# Initialize Firebase
# cred = credentials.Certificate('./serviceAccountKey.json')
# firebase_admin.initialize_app(cred)
# db = firestore.client()

# def get_real_time_data():
#     """Fetch real-time data from Firebase"""
#     bikes_ref = db.collection('dublin_bikes')
#     docs = bikes_ref.get()
    
#     stations_data = []
#     for doc in docs:
#         data = doc.to_dict()
#         stations_data.append({
#             'station_id': data.get('id'),
#             'name': data.get('name'),
#             'bikes': data.get('free_bikes', 0),
#             'slots': data.get('empty_slots', 0),
#             'lat': data.get('latitude'),
#             'lng': data.get('longitude'),
#             'timestamp': data.get('timestamp').strftime('%Y-%m-%d %H:%M:%S') if data.get('timestamp') else None
#         })
#     return stations_data

def analyze_historical_data():
    """
    Enhanced analysis combining real-time and historical data:
    - Hourly trends with live comparison
    - Daily patterns and deviations
    - Station performance insights
    - Demand predictions
    - Weekly total bikes data for better trend visualization
    """
    try:
        # Load historical data from CSV
        historical_df = pd.read_csv('dublin_bikes_data.csv')

        # Parse 'Last Updated' and add features
        historical_df['Last Updated'] = pd.to_datetime(historical_df['Last Updated'])
        historical_df['hour'] = historical_df['Last Updated'].dt.hour
        historical_df['day_of_week'] = historical_df['Last Updated'].dt.day_name()

        # Fetch live data
        live_data = pd.DataFrame(get_real_time_data())

        # Ensure numeric columns
        live_data[['bikes', 'slots']] = live_data[['bikes', 'slots']].fillna(0).astype(int)
        historical_df[['Free Bikes', 'Empty Slots']] = historical_df[['Free Bikes', 'Empty Slots']].fillna(0).astype(int)

        # Hourly averages from historical data
        hourly_avg = historical_df.groupby('hour')['Free Bikes'].mean().fillna(0).round(2).to_dict()

        # Daily averages from historical data
        daily_avg = historical_df.groupby('day_of_week')['Free Bikes'].mean().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        ).fillna(0).round(2).to_dict()

        # Weekly total bikes (sum of Free Bikes per day)
        weekly_total = historical_df.groupby('day_of_week')['Free Bikes'].sum().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        ).fillna(0).round(2).to_dict()

        # Combine live data with historical data
        current_hour = datetime.now().hour
        current_day = datetime.now().strftime('%A')
        current_hour_avg = float(hourly_avg.get(current_hour, 0))
        current_day_avg = float(daily_avg.get(current_day, 0))

        # Station-level performance
        busiest_stations = live_data.sort_values('bikes', ascending=False).head(3)
        quietest_stations = live_data.sort_values('bikes', ascending=True).head(3)

        # Peak and off-peak analysis
        peak_hours = {int(hour): float(usage) for hour, usage in sorted(hourly_avg.items(), key=lambda x: x[1])[:3]}
        off_peak_hours = {int(hour): float(usage) for hour, usage in sorted(hourly_avg.items(), key=lambda x: x[1], reverse=True)[:3]}

        # Prepare JSON output
        return {
            'current_hour': {
                'hour': f"{current_hour:02d}:00",
                'live_bikes': int(live_data['bikes'].sum()),
                'historical_avg': current_hour_avg
            },
            'daily_usage': [{'day': day, 'avg_usage': float(usage)} for day, usage in daily_avg.items()],
            'weekly_total': [{'day': day, 'total_bikes': int(total)} for day, total in weekly_total.items()],
            'peak_hours': [{'hour': f"{hour:02d}:00", 'avg_bikes_available': avg} for hour, avg in peak_hours.items()],
            'off_peak_hours': [{'hour': f"{hour:02d}:00", 'avg_bikes_available': avg} for hour, avg in off_peak_hours.items()],
            'busiest_stations': [
                {'station_name': row['name'], 'bikes': int(row['bikes']), 'slots': int(row['slots'])}
                for _, row in busiest_stations.iterrows()
            ],
            'quietest_stations': [
                {'station_name': row['name'], 'bikes': int(row['bikes']), 'slots': int(row['slots'])}
                for _, row in quietest_stations.iterrows()
            ],
            'demand_prediction': {
                'live_total_bikes': int(live_data['bikes'].sum()),
                'historical_total_bikes': float(historical_df['Free Bikes'].mean()),
                'weekend_vs_weekday_ratio': round(
                    (daily_avg['Saturday'] + daily_avg['Sunday']) /
                    sum(daily_avg[day] for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']),
                    2
                ) if sum(daily_avg[day] for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']) > 0 else None
            }
        }
    except Exception as e:
        return {"error": str(e)}


def cluster_stations():
    current_data = get_real_time_data()
    df = pd.DataFrame(current_data)

    # Preprocessing: Handle missing values
    df.fillna(0, inplace=True)

    # Prepare features for clustering
    features = df[['lat', 'lng']].copy()
    features['utilization'] = df.apply(
        lambda row: (row['bikes'] / (row['bikes'] + row['slots']))
        if (row['bikes'] + row['slots']) > 0 else 0, axis=1
    )

    # Normalize features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Ensure there are no NaNs
    if np.isnan(features_scaled).any():
        raise ValueError("Features contain NaN after scaling")

    # Perform clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)

    # Prepare cluster data for frontend
    cluster_data = []
    for idx, row in df.iterrows():
        cluster_data.append({
            'station_id': row['station_id'],
            'name': row['name'],
            'lat': row['lat'],
            'lng': row['lng'],
            'cluster': int(clusters[idx]),
            'bikes': row['bikes'],
            'slots': row['slots']
        })

    return cluster_data

def calculate_system_statistics():
    """
    Calculate system-wide statistics for bikes and docking stations.
    """
    try:
        current_data = get_real_time_data()
        if not current_data:
            return {"error": "No data available"}

        total_bikes = sum(station.get('bikes', 0) for station in current_data)
        total_slots = sum(station.get('slots', 0) for station in current_data)
        total_capacity = total_bikes + total_slots

        total_stations = len(current_data)
        empty_stations = sum(1 for station in current_data if station.get('bikes', 0) == 0)
        overloaded_stations = sum(1 for station in current_data if station.get('slots', 0) == 0)

        avg_bikes_per_station = total_bikes / total_stations if total_stations > 0 else 0
        system_utilization = (total_bikes / total_capacity * 100) if total_capacity > 0 else 0

        return {
            "system_overview": {
                "total_bikes": total_bikes,
                "total_slots": total_slots,
                "system_utilization": round(system_utilization, 2),
                "total_stations": total_stations,
                "empty_stations": empty_stations,
                "overloaded_stations": overloaded_stations,
                "avg_bikes_per_station": round(avg_bikes_per_station, 2)
            },
            "station_metrics": current_data
        }
    except Exception as e:
        return {"error": str(e)}

def urban_computing_insights():
    """
    Advanced urban computing analysis for Dublin Bikes system
    """
    try:
        # Load historical data
        historical_df = pd.read_csv('dublin_bikes_data.csv')
        current_data = get_real_time_data()
        
        # Urban Mobility Indicators
        mobility_insights = {
            'temporal_diversity': temporal_mobility_analysis(historical_df),
            'spatial_accessibility': spatial_accessibility_analysis(current_data),
            'environmental_impact': calculate_environmental_benefits(current_data),
            'socio_economic_indicators': socio_economic_mobility_analysis(historical_df)
        }
        
        return mobility_insights
    
    except Exception as e:
        return {"error": str(e)}

def temporal_mobility_analysis(df):
    """
    Analyze temporal patterns of bike usage
    """
    # Parse datetime
    df['timestamp'] = pd.to_datetime(df['Last Updated'])
    
    # Time-based analysis
    hourly_entropy = calculate_entropy(df.groupby(df['timestamp'].dt.hour)['Free Bikes'].count())
    daily_entropy = calculate_entropy(df.groupby(df['timestamp'].dt.day_name())['Free Bikes'].count())
    
    return {
        'hourly_usage_diversity': hourly_entropy,
        'daily_usage_diversity': daily_entropy,
        'peak_transition_times': identify_transition_periods(df)
    }

def calculate_entropy(distribution):
    """
    Calculate Shannon entropy to measure diversity of usage
    """
    probabilities = distribution / distribution.sum()
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def identify_transition_periods(df):
    """
    Identify critical transition periods in bike usage
    """
    df['hour'] = df['timestamp'].dt.hour
    hourly_usage = df.groupby('hour')['Free Bikes'].mean()
    
    # Find inflection points
    derivatives = np.gradient(hourly_usage)
    significant_changes = np.abs(derivatives) > np.std(derivatives)
    
    return {
        'transition_hours': [hour for hour, is_transition in zip(hourly_usage.index, significant_changes) if is_transition],
        'change_magnitude': derivatives[significant_changes].tolist()
    }

def spatial_accessibility_analysis(current_data):
    """
    Analyze spatial distribution and accessibility
    """
    df = pd.DataFrame(current_data)
    
    # Calculate centroid of all stations
    centroid_lat = df['lat'].mean()
    centroid_lng = df['lng'].mean()
    
    # Distance from centroid
    df['distance_from_center'] = df.apply(
        lambda row: geodesic((centroid_lat, centroid_lng), (row['lat'], row['lng'])).kilometers, 
        axis=1
    )
    
    # Spatial clustering metrics
    spatial_spread = df['distance_from_center'].std()
    accessibility_score = calculate_accessibility_index(df)
    
    return {
        'centroid': {'lat': centroid_lat, 'lng': centroid_lng},
        'spatial_spread_km': spatial_spread,
        'accessibility_score': accessibility_score,
        'station_distribution': df[['name', 'distance_from_center']].to_dict(orient='records')
    }

def calculate_accessibility_index(df):
    """
    Calculate an urban accessibility index
    """
    # Normalize bike availability and proximity to center
    scaler = StandardScaler()
    df['normalized_bikes'] = scaler.fit_transform(df[['bikes']])
    df['normalized_distance'] = scaler.fit_transform(df[['distance_from_center']])
    
    # Composite accessibility score
    # Higher score means more accessible
    df['accessibility_score'] = (df['normalized_bikes'] - df['normalized_distance']) / 2
    
    return df['accessibility_score'].mean()

def calculate_environmental_benefits(current_data):
    """
    Estimate environmental benefits of bike-sharing
    """
    df = pd.DataFrame(current_data)
    
    # Assumptions for carbon offset calculation
    # Average car trip replaced: 5 km
    # CO2 per km for average car: 0.192 kg
    avg_trip_length = 5  # km
    co2_per_car_km = 0.192  # kg
    
    total_bikes_available = df['bikes'].sum()
    estimated_trips = total_bikes_available * 0.5  # Assume 50% utilization
    
    return {
        'total_bikes_available': int(total_bikes_available),
        'estimated_daily_trips': int(estimated_trips),
        'co2_avoided_kg': round(estimated_trips * avg_trip_length * co2_per_car_km, 2),
        'equivalent_trees_saved': round(
            (estimated_trips * avg_trip_length * co2_per_car_km) / 22, 2
        )  # Avg tree absorbs ~22 kg CO2 annually
    }

def socio_economic_mobility_analysis(df):
    """
    Analyze socio-economic mobility patterns
    """
    df['timestamp'] = pd.to_datetime(df['Last Updated'])
    df['hour'] = df['timestamp'].dt.hour
    
    # Proxy for work/leisure mobility
    work_hours_usage = df[(df['hour'] >= 7) & (df['hour'] <= 10)]['Free Bikes'].mean()
    leisure_hours_usage = df[(df['hour'] >= 18) & (df['hour'] <= 22)]['Free Bikes'].mean()
    
    return {
        'work_hour_mobility_index': work_hours_usage,
        'leisure_hour_mobility_index': leisure_hours_usage,
        'mobility_balance_ratio': work_hours_usage / leisure_hours_usage if leisure_hours_usage > 0 else None
    }

def urban_computing_insights():
    """
    Advanced urban computing analysis for Dublin Bikes system
    """
    try:
        # Validate CSV file
        csv_path = 'dublin_bikes_data.csv'
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return {"error": "Historical data file not found"}

        # Load historical data with error handling
        try:
            historical_df = pd.read_csv(csv_path)
            
            # Validate required columns
            required_columns = ['Last Updated', 'Free Bikes']
            missing_columns = [col for col in required_columns if col not in historical_df.columns]
            if missing_columns:
                logger.error(f"Missing columns: {missing_columns}")
                return {"error": f"Missing columns: {missing_columns}"}

        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            return {"error": f"Error reading historical data: {e}"}

        current_data = get_real_time_data()
        
        # Urban Mobility Insights
        mobility_insights = {
            'temporal_diversity': temporal_mobility_analysis(historical_df),
            'spatial_accessibility': spatial_accessibility_analysis(current_data),
            'environmental_impact': calculate_environmental_benefits(current_data),
            'socio_economic_indicators': socio_economic_mobility_analysis(historical_df)
        }
        
        # Log insights for debugging
        logger.info("Urban computing insights generated successfully")
        return mobility_insights
    
    except Exception as e:
        logger.error(f"Unexpected error in urban_computing_insights: {traceback.format_exc()}")
        return {
            "error": "Unexpected error generating urban insights", 
            "details": str(e)
        }

# [Rest of your existing functions remain the same]

# Modify the urban insights route to provide more detailed error responses
@app.route('/api/urban-insights')
def urban_insights():
    insights = urban_computing_insights()
    
    # If there's an error, return a 500 status code
    if 'error' in insights:
        return jsonify(insights), 500
    
    return jsonify(insights)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/real-time-data')
def real_time_data_page():
    return render_template('real_time_data.html')

@app.route('/usage-patterns')
def usage_patterns_page():
    return render_template('usage_patterns.html')

@app.route('/system-statistics')
def system_statistics_page():
    return render_template('system_statistics.html')

@app.route('/station-clusters')
def station_clusters_page():
    return render_template('station_clusters.html')

@app.route('/api/real-time-data')
def real_time_data():
    return jsonify(get_real_time_data())

@app.route('/api/usage-patterns')
def usage_patterns():
    """
    Combine real-time and historical data for usage patterns.
    """
    try:
        historical_data = analyze_historical_data()
        return jsonify(historical_data)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/clusters')
def clusters():
    return jsonify(cluster_stations())

@app.route('/api/system-statistics')
def system_statistics():
    return jsonify(calculate_system_statistics())


if __name__ == '__main__':
    app.run(debug=True)
