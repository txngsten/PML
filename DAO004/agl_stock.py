# Student Name: Phuong Quyen Dao
# Student FAN:  dao0047 
# File:        afl_stock.py
# Date:        15-5-2026
# Description: Download stock data and calculate log returns 
# Usage: agl_stock.py

import yfinance as yf
import pandas as pd
import numpy as np


start_date = "2018-01-01"
end_date = "2025-01-01"
ticker = "AGL.AX"

#Download stock data
data = yf.download(
    ticker,
    start = start_date,
    end = end_date,
    auto_adjust=True
)

#Flatten multi-index columns
data.columns = data.columns.get_level_values(0)
#Convert index into date column
data.reset_index(inplace=True)

#Calculate log returns
data["log_return"] = np.log(
    data["Close"] / data["Close"].shift(1)
)

#Remove missing rows
data.dropna(inplace=True)

#Preview
print(data.head())
print(data.tail())
print("Start date:", data["Date"].min())
print("End date:", data["Date"].max())

#Save CSV
data.to_csv("agl_stock.csv", index=False)