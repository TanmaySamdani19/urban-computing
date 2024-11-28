import requests
import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import re
import pandas as pd
import time

# Initialize Firebase with service account credentials
cred = credentials.Certificate('./serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'projectId': 'urban-computing-5e47d',
})

db = firestore.client()

url = 'http://api.citybik.es/v2/networks/dublinbikes'

def upload_own_data():
    try:
        data = pd.read_csv('dublin_bikes_data.csv')
        
        for _, row in data.iterrows():
            sanitized_name = re.sub(r'\W+', '_', str(row['Station Name']))
            station_ref = db.collection('dublin_bikes_own_data').document(str(row['Station ID']))

            # Data payload with necessary fields from CSV and current timestamp
            station_data = {
                'station_id': row['Station ID'],
                'station_name': row['Station Name'],
                'latitude': row['Latitude'],
                'longitude': row['Longitude'],
                'free_bikes': row['Free Bikes'],
                'empty_slots': row['Empty Slots'],
                'total_capacity': row['Total Capacity'],
                'operational_status': row['Operational Status'],
                'last_updated': row['Last Updated'],
                'data_fetch_timestamp': row['Data Fetch Timestamp'],
                'timestamp': datetime.datetime.now()
            }
            station_ref.set(station_data)
        print("Own data from CSV uploaded to Firebase Firestore.")
    except Exception as e:
        print(f"Failed to upload own data from CSV: {e}")

def fetch_and_store_bike_data():
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        stations = data['network']['stations']

        for station in stations:
            sanitized_name = re.sub(r'\W+', '_', station['name'])
            station_ref = db.collection('dublin_bikes').document(str(station.get('id', sanitized_name)))
            
            # Data payload with real-time data and current timestamp
            station_data = {
                **station,
                'timestamp': datetime.datetime.now()
            }
            station_ref.set(station_data)
        print("Real-time data from CityBikes API saved to Firebase Firestore.")
    else:
        print("Failed to retrieve data from CityBikes API.")

# Real-time data collection every 5 minutes for both own data and open data
while True:
    upload_own_data()
    fetch_and_store_bike_data()
    time.sleep(300)
