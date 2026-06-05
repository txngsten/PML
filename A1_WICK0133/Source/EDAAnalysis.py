# Student Name: Zechariah Wicks
# Student FAN: wick0133
# File: EDAAnalysis.py
# Date: 14/5/26
# Description: Automated data ingestion and extended Exploratory Data Analysis (EDA) for XEJ stock volatility and BOM cloud cover (from Open-Meteo).
# Usage: (ensure the virtual environment is active, and libraries installed via pip) python3 EDAAnalysis.py
# Licence: GNU General Public License v3.0 

import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

def fetchFinancialData(ticker="^AXEJ", startDate="2023-01-01", endDate="2024-01-01"):
    """Fetches historical index data using Yahoo Finance API."""
    print(f"Fetching financial data for {ticker}...")
    stockData = yf.download(ticker, start=startDate, end=endDate)
    
    if isinstance(stockData.columns, pd.MultiIndex):
        stockData.columns = stockData.columns.get_level_values(0)
        
    stockData = stockData.reset_index()
    stockData['date'] = pd.to_datetime(stockData['Date']).dt.date
    
    stockData['xejVolatility'] = (stockData['High'] - stockData['Low']).squeeze()
    
    return stockData[['date', 'xejVolatility']]

# Coords are centre of Adelaide CBD
def fetchWeatherData(latitude=-34.9285, longitude=138.6007, startDate="2023-01-01", endDate="2024-01-01"):
	"""Fetches historical cloud cover data using Open-Meteo API."""
	print("Fetching weather data from Open-Meteo API...")
	apiUrl = "https://archive-api.open-meteo.com/v1/archive"
	apiParams = {
		"latitude": latitude,
		"longitude": longitude,
		"start_date": startDate,
		"end_date": endDate,
		"daily": "cloudcover_mean", 
		"timezone": "Australia/Adelaide"
	}
	
	apiResponse = requests.get(apiUrl, params=apiParams)
	apiResponse.raise_for_status() 
	weatherJson = apiResponse.json()
	
	weatherData = pd.DataFrame({
		'date': pd.to_datetime(weatherJson['daily']['time']).date,
		'cloudCover': weatherJson['daily']['cloudcover_mean']
	})
	
	return weatherData

def performDeepAnalysis():
	"""Ingests data, performs statistical analysis, and generates visualisations."""
	sns.set_theme(style="whitegrid")
	
	rawStockData = fetchFinancialData()
	rawWeatherData = fetchWeatherData()
	
	print("\nMissing Data Analysis")
	print("This highlights the inherent epistemic uncertainty in the telemetry.")
	print(f"Missing Cloud Cover Records: {rawWeatherData['cloudCover'].isna().sum()}")
	print(f"Missing Stock Records: {rawStockData['xejVolatility'].isna().sum()}")
	
	mergedData = pd.merge(rawStockData, rawWeatherData, on='date')
	cleanData = mergedData.dropna()
	
	print("\nDistribution Statistics")
	volatilityArray = cleanData['xejVolatility'].values
	cloudArray = cleanData['cloudCover'].values
	
	print(f"Volatility Skewness: {skew(volatilityArray):.2f} (Values > 1 indicate heavy right tail)")
	print(f"Volatility Kurtosis: {kurtosis(volatilityArray):.2f} (Values > 3 indicate extreme outliers)")
	print(f"Cloud Cover Mean: {np.mean(cloudArray):.2f}%, Std Dev: {np.std(cloudArray):.2f}%")
	
	print("\nGenerating EDA Visualisations...")
	
	# Figure 1: Distributions (Proving non-normal data)
	fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
	
	sns.histplot(cleanData['xejVolatility'], kde=True, ax=ax1, color='blue')
	ax1.set_title("Distribution of XEJ Volatility")
	ax1.set_xlabel("Volatility (High - Low)")
	
	sns.histplot(cleanData['cloudCover'], kde=True, ax=ax2, color='orange')
	ax2.set_title("Distribution of Mean Cloud Cover")
	ax2.set_xlabel("Cloud Cover (%)")
	
	plt.tight_layout()
	plt.savefig("./wick0133/Misc/Outputs/edaDistributions.png")
	plt.close()
	
	# Figure 2: Time Series & Volatility Clustering
	fig2, ax3 = plt.subplots(figsize=(14, 6))
	
	ax3.plot(cleanData['date'], cleanData['xejVolatility'], label="Daily Volatility", color='blue', alpha=0.6)
	# Adding a 7-day rolling average to show clustering
	cleanData['rollingVolatility'] = cleanData['xejVolatility'].rolling(window=7).mean()
	ax3.plot(cleanData['date'], cleanData['rollingVolatility'], label="7-Day Rolling Avg", color='red', linewidth=2)
	
	ax3.set_title("XEJ Volatility Over Time (Highlighting Volatility Clustering)")
	ax3.set_xlabel("Date")
	ax3.set_ylabel("Volatility Index")
	ax3.legend()
	
	plt.tight_layout()
	plt.savefig("./wick0133/Misc/Outputs/edaTimeSeries.png")
	plt.close()
	
	# Figure 3: Correlation Matrix
	fig3, ax4 = plt.subplots(figsize=(6, 5))
	correlationMatrix = cleanData[['xejVolatility', 'cloudCover']].corr(method='spearman')
	sns.heatmap(correlationMatrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax4)
	ax4.set_title("Spearman Correlation Matrix")
	
	plt.tight_layout()
	plt.savefig("./wick0133/Misc/Outputs/edaCorrelation.png")
	plt.close()
	
	print("Analysis complete. Visualisations saved as PNG files.")

if __name__ == "__main__":
	performDeepAnalysis()