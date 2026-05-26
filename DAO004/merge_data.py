# Student Name: Phuong Quyen Dao
# Student FAN:  dao0047 
# File:        merge_data.py
# Date:        15-5-2026
# Description: Merge AGL stock log returns with NEM weather data to create final dataset 
# Usage: merge_data.py

import pandas as pd

stock_df = pd.read_csv("agl_stock.csv")
weather_df = pd.read_csv("nem_weather_data.csv")

#Convert date columns
stock_df["Date"] = pd.to_datetime(stock_df["Date"])
weather_df["date"] = pd.to_datetime(weather_df["date"])

#Keep target variable from stock data
stock_returns = stock_df[["Date", "log_return"]].copy()
stock_returns = stock_returns.rename(columns={
    "Close": "agl_close",
    "Volume": "agl_volume"
})

#Convert data from long to wide format
weather_wide = weather_df.pivot(
    index="date",
    columns="region",
    values=["temperature", "wind_speed", "solar_radiation"]
)

#Flatten column names
weather_wide.columns = [
    f"{variable}_{region}"
    for variable, region in weather_wide.columns
]

weather_wide = weather_wide.reset_index()

#Add avg weather across all regions
weather_wide["avg_temperature"] = weather_wide[
    [col for col in weather_wide.columns
     if col.startswith("temperature_")]].mean(axis=1)

weather_wide["avg_wind_speed"] = weather_wide[
    [col for col in weather_wide.columns
     if col.startswith("wind_speed_")]].mean(axis=1)

weather_wide["avg_solar_radiation"] = weather_wide[
    [col for col in weather_wide.columns
     if col.startswith("solar_radiation_")]].mean(axis=1)

#Merge weather features with stock log returns
final_df = pd.merge(
    weather_wide,
    stock_returns,
    left_on="date",
    right_on="Date",
    how="inner"
)

#Remove duplicate date column
final_df = final_df.drop(columns=["Date"])

#Add covid bucket
def assign_period_bucket(date):
    if date < pd.Timestamp("2020-03-01"):
        return "pre_covid"
    elif date < pd.Timestamp("2023-01-01"):
        return "covid"
    else:
        return "post_covid"

final_df["period_bucket"] = final_df["date"].apply(assign_period_bucket)

#Sort dataset
final_df = final_df.sort_values("date")

#Check dataset
print(final_df.head())
print(final_df.tail())
print("Start date:", final_df["date"].min())
print("End date:", final_df["date"].max())
print("Shape:", final_df.shape)
print("Rows per period:", final_df["period_bucket"].value_counts())
print("Missing values:",final_df.isnull().sum())

final_df.to_csv("final_dataset.csv", index=False)