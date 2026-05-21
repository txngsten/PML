# Student Name: Zechariah Wicks
# Student FAN: wick0133
# File: ProbabilisticModel.py
# Date: 14-05-2026
# Description: Probabilistic pipeline comparing Frequentist MLE and Bayesian MCMC for modelling energy stock volatility against solar cloud cover.
# Usage: (ensure the virtual environment is active, and libraries installed via pip) python3 probabilisticModel.py
# Licence: GNU General Public License v3.0 

import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

def preprocessDataFromAPI():
	"""Ingests data from APIs with local CSV caching, merges, and cleans."""
	cacheFile = "./wick0133/Misc/Outputs/cachedPipelineData.csv"
	
	if os.path.exists(cacheFile):
		print("Loading data from local cache...")
		cleanData = pd.read_csv(cacheFile)
		return cleanData
		
	stockData = fetchFinancialData()
	weatherData = fetchWeatherData()
	
	mergedData = pd.merge(stockData, weatherData, on='date')
	cleanData = mergedData.dropna()
	
	cleanData.to_csv(cacheFile, index=False)
	print(f"Data cached locally to {cacheFile}")
	
	return cleanData

def frequentistEstimation(cloudCover, stockVolatility):
	"""Performs Frequentist Maximum Likelihood Estimation (MLE)."""
	def negativeLogLikelihood(parameters):
		interceptParam, slopeParam, stdDevParam = parameters
		
		if stdDevParam <= 0:
			return np.inf
			
		expectedVolatility = interceptParam + slopeParam * cloudCover
		
		logLikelihood = -len(cloudCover) / 2 * np.log(2 * np.pi * stdDevParam**2) - \
						np.sum((stockVolatility - expectedVolatility)**2) / (2 * stdDevParam**2)
						
		return -logLikelihood

	initialGuess = np.array([np.mean(stockVolatility), 0.0, np.std(stockVolatility)])
	optimisationResult = minimize(negativeLogLikelihood, initialGuess, method='L-BFGS-B')
	
	frequentistEstimates = {
		'intercept': optimisationResult.x[0],
		'slope': optimisationResult.x[1],
		'stdDev': optimisationResult.x[2]
	}
	return frequentistEstimates

def bayesianEstimation(cloudCover, stockVolatility):
	"""Performs Bayesian Estimation using Markov Chain Monte Carlo (MCMC)."""
	with pm.Model() as probabilisticModel:
		interceptPrior = pm.Normal('interceptPrior', mu=np.mean(stockVolatility), sigma=100)
		slopePrior = pm.Normal('slopePrior', mu=0, sigma=10)
		stdDevPrior = pm.HalfNormal('stdDevPrior', sigma=50)

		expectedVolatility = interceptPrior + slopePrior * cloudCover
		
		marketObservation = pm.Normal('marketObservation',
									  mu=expectedVolatility,
									  sigma=stdDevPrior,
									  observed=stockVolatility)

		bayesianTrace = pm.sample(draws=2000, tune=1000, chains=4, target_accept=0.95, progressbar=True)
		
	return bayesianTrace

def visualiseResults(cloudCover, stockVolatility, frequentistResults, bayesianTrace):
	"""Generates a comparison plot for the Video Demonstration."""
	plt.figure(figsize=(10, 6))
	plt.scatter(cloudCover, stockVolatility, alpha=0.5, label='Observed Data', color='gray')
	
	# Plot Frequentist Line
	cloudRange = np.linspace(cloudCover.min(), cloudCover.max(), 100)
	frequentistLine = frequentistResults['intercept'] + frequentistResults['slope'] * cloudRange
	plt.plot(cloudRange, frequentistLine, color='red', linewidth=2, label='Frequentist MLE (Point Estimate)')
	
	# Plot Bayesian Credible Intervals
	posteriorIntercept = bayesianTrace.posterior['interceptPrior'].values.flatten()
	posteriorSlope = bayesianTrace.posterior['slopePrior'].values.flatten()
	
	for i in range(100):
		bayesianLine = posteriorIntercept[i] + posteriorSlope[i] * cloudRange
		plt.plot(cloudRange, bayesianLine, color='blue', alpha=0.05)
		
	plt.plot([], [], color='blue', alpha=0.5, label='Bayesian Posterior (Uncertainty Interval)')
	
	plt.title('Energy Stock Volatility vs. Solar Cloud Cover')
	plt.xlabel('Mean Daily Cloud Cover (%)')
	plt.ylabel('XEJ Daily Volatility (High - Low)')
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig("./wick0133/Misc/Outputs/ProbabilisticModels.png")
	plt.show()

def main():
	print("Starting Probabilistic ML Engine Pipeline")
	pipelineData = preprocessDataFromAPI()
	
	cloudData = pipelineData['cloudCover'].values
	volatilityData = pipelineData['xejVolatility'].values
	
	print("\n[1/2] Executing Frequentist Pipeline...")
	frequentistResults = frequentistEstimation(cloudData, volatilityData)
	print("Frequentist Estimates:", frequentistResults)
	
	print("\n[2/2] Executing Bayesian Pipeline...")
	bayesianResults = bayesianEstimation(cloudData, volatilityData)
	print("\nBayesian Summary Statistics:")
	print(az.summary(bayesianResults))
	
	print("\nGenerating Visual Comparison...")
	visualiseResults(cloudData, volatilityData, frequentistResults, bayesianResults)
	print("Pipeline Execution Complete")

if __name__ == "__main__":
	main()