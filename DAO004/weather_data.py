# Student Name: Phuong Quyen Dao
# Student FAN:  dao0047 
# File:        weather_data.py
# Date:        15-5-2026
# Description: Download waather data for NEM regions 
# Usage: weather_data.py

import requests
import pandas as pd

start_date = "2018-01-01"
end_date = "2025-01-01"

#NEM regions
regions = {
    "SA": {"lat": -34.9285, "lon": 138.6007},  #adelaide
    "VIC": {"lat": -37.8136, "lon": 144.9631}, #melbourne
    "NSW": {"lat": -33.8688, "lon": 151.2093}, #sydney
    "QLD": {"lat": -27.4698, "lon": 153.0251}, #brisbane
    "TAS": {"lat": -42.8821, "lon": 147.3272}, #hobart 
}

all_weather_data = []

for region, coords in regions.items():

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={coords['lat']}&longitude={coords['lon']}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        f"&daily=temperature_2m_mean,"
        f"wind_speed_10m_max,"
        f"shortwave_radiation_sum"
        f"&timezone=Australia/Adelaide"
    )

    response = requests.get(url)

    data = response.json()

    region_df = pd.DataFrame({
        "date": data["daily"]["time"],
        "temperature": data["daily"]["temperature_2m_mean"],
        "wind_speed": data["daily"]["wind_speed_10m_max"],
        "solar_radiation": data["daily"]["shortwave_radiation_sum"]
    })

    region_df["region"] = region

    all_weather_data.append(region_df)

#Combine regions
weather_df = pd.concat(all_weather_data)

#Convert date columns
weather_df["date"] = pd.to_datetime(weather_df["date"])

#Filter out 2025-01-01
weather_df = weather_df[weather_df["date"] < end_date]

#Sort data
weather_df = weather_df.sort_values(["region", "date"])

#Preview
print(weather_df.head())
print(weather_df.tail())
print("Start date:", weather_df["date"].min())
print("End date:", weather_df["date"].max())
print(weather_df["region"].value_counts())
print("Missing value:", weather_df.isnull().sum())

weather_df.to_csv("nem_weather_data.csv", index=False)