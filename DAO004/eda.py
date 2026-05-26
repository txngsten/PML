# Student Name: Phuong Quyen Dao
# Student FAN:  dao0047 
# File:         eda.py
# Date:        15-5-2026
# Description: Perform EDA, include missing value checks, distribution plots, correlation analysis, QQ plot, period bucket summarise
# Usage: eda.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


df = pd.read_csv("final_dataset.csv")

#Convert date column
df["date"] = pd.to_datetime(df["date"])

#Sort by date
df = df.sort_values("date")

#Basic dataset check
print("Dataset preview:", df.head())

print("\nDataset shape:", df.shape)

print("\nDate coverage:")
print("Start date:", df["date"].min())
print("End date:", df["date"].max())

print("\nRows per period:")
print(df["period_bucket"].value_counts())

print("\nMissing values:", df.isnull().sum())

print("\nDuplicate dates:", df.duplicated(subset="date").sum())

#Save missing value table
missing_table = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_percent": df.isnull().mean() * 100
})

missing_table.to_csv("./Misc/Outputs/missing_values.csv")


#Summary statistics by bucket
summary_by_bucket = df.groupby("period_bucket")["log_return"].agg(
    count="count",
    mean="mean",
    std="std",
    min="min",
    max="max",
    skew="skew"
)

summary_by_bucket["kurtosis"] = df.groupby("period_bucket")["log_return"].apply(
    pd.Series.kurtosis
)

print("\nSummary statistics by period:", summary_by_bucket)

summary_by_bucket.to_csv("./Misc/Outputs/summary_by_bucket.csv")


#Log return over time
plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["log_return"])
plt.axvspan(pd.Timestamp("2018-01-01"), pd.Timestamp("2020-03-01"), color ="green", alpha = 0.05, label="Pre-Covid")
plt.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2023-01-01"), color ="blue", alpha = 0.05, label="Covid")
plt.axvspan(pd.Timestamp("2023-01-01"), pd.Timestamp("2025-01-01"), color ="orange", alpha = 0.05, label="Post-Covid")
plt.title("AGL Log Returns Over Time")
plt.xlabel("Date")
plt.ylabel("Log Return")
plt.legend()
plt.tight_layout()
plt.savefig("./Misc/Outputs/log_returns_over_time.png")
plt.show()


#Rolling volatility
df["rolling_volatility_21d"] = df["log_return"].rolling(window=21).std()

plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["rolling_volatility_21d"])
plt.axvspan(pd.Timestamp("2018-01-01"), pd.Timestamp("2020-03-01"), color ="green", alpha = 0.05, label="Pre-Covid")
plt.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2023-01-01"), color ="blue", alpha = 0.05, label="Covid")
plt.axvspan(pd.Timestamp("2023-01-01"), pd.Timestamp("2025-01-01"), color ="orange", alpha = 0.05, label="Post-Covid")
plt.title("21-Day Rolling Volatility of AGL Log Returns")
plt.xlabel("Date")
plt.ylabel("Rolling Standard Deviation")
plt.legend()
plt.tight_layout()
plt.savefig("./Misc/Outputs/rolling_volatility_21d.png")
plt.show()


#Histogram with normal and student-t fit
returns = df["log_return"].dropna()

#Fit Normal distribution
normal_mu, normal_sigma = stats.norm.fit(returns)

#Fit Student-t distribution
t_df, t_loc, t_scale = stats.t.fit(returns)

x = np.linspace(returns.min(), returns.max(), 500)

normal_pdf = stats.norm.pdf(x, normal_mu, normal_sigma)
t_pdf = stats.t.pdf(x, t_df, t_loc, t_scale)

plt.figure(figsize=(10, 6))
plt.hist(returns, bins=50, density=True, label="Observed log returns")
plt.plot(x, normal_pdf, label="Normal fit")
plt.plot(x, t_pdf, label="Student-t fit")
plt.title("Distribution of AGL Log Returns with Fitted Distributions")
plt.xlabel("Log Return")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("./Misc/Outputs/log_return_distribution_fit.png")
plt.show()

print("\nNormal fit parameters:")
print("mu:", normal_mu)
print("sigma:", normal_sigma)

print("\nStudent-t fit parameters:")
print("df:", t_df)
print("loc:", t_loc)
print("scale:", t_scale)

#Save distribution parameters
distribution_params = pd.DataFrame({
    "model": ["Normal", "Student-t"],
    "param_1": [normal_mu, t_df],
    "param_2": [normal_sigma, t_loc],
    "param_3": [np.nan, t_scale]
})

distribution_params.to_csv("./Misc/Outputs/distribution_fit_parameters.csv", index=False)


#Q-Q plot
plt.figure(figsize=(8, 6))
qqplot(returns, line="s", fit=True)
plt.title("Q-Q Plot of AGL Log Returns")
plt.tight_layout()
plt.savefig("qq_plot_log_returns.png")
plt.show()

#7. Normality test
returns = df["log_return"].dropna()

shapiro_stat, shapiro_p = stats.shapiro(returns.sample(min(500, len(returns)), random_state=42))
ks_stat, ks_p = stats.kstest(
    returns,
    "norm",
    args=(returns.mean(), returns.std())
)

print("\nNormality tests:")
print("Shapiro-Wilk statistic:", shapiro_stat)
print("Shapiro-Wilk p-value:", shapiro_p)
print("Kolmogorov-Smirnov statistic:", ks_stat)
print("Kolmogorov-Smirnov p-value:", ks_p)

#Correlation matrix
correlation_columns = [
    "avg_temperature",
    "avg_wind_speed",
    "avg_solar_radiation",
    "log_return"
]

correlation = df[correlation_columns].corr()

print("\nCorrelation matrix:", correlation)

correlation.to_csv("./Misc/Outputs/correlation_matrix.csv")

#VIF check for multicollinearity
X_vif = df[[
    "avg_temperature",
    "avg_wind_speed",
    "avg_solar_radiation"
]]

X_vif = sm.add_constant(X_vif)

vif_table = pd.DataFrame()
vif_table["feature"] = X_vif.columns
vif_table["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]

print("\nVIF table:", vif_table)

vif_table.to_csv("./Misc/Outputs/vif_table.csv", index=False)